using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# A plate cell that calls another kernel's endpoint embeds that kernel's source
# operations, so one cell call nests source operations. Julia's inference used
# to widen the inner call while the first kernel holding it compiled: the cells
# and the plate total of that kernel stayed boxed for its lifetime, while an
# identical kernel compiled afterwards inferred concretely (snag
# `first-prepared-s-e848620b`). Every kernel here is new to this module, so the
# first one compiled below is the first to meet the endpoint's operations,
# whatever ran earlier in the session.
module NestedSourceInferenceFixtures
using ReactiveKernels

@kernel shifted(location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    logpdf(x::Float64)::Float64 = begin
        z::Float64 = (x - location) / scale
        -0.5 * z^2 - log_scale
    end
end

# Identical, separately evaluated parents: distinct cell operations sharing the
# endpoint's operations.
for name in (:primal_first, :primal_second, :reverse_first, :reverse_second)
    @eval @kernel $name(xs::Vector{Float64}, m::Vector{Float64}) = begin
        cells = plate(xs, m) do x, mi
            if 1.0 > 0
                (shifted(mi, 0.7)).logpdf(x)
            else
                -Inf
            end
        end
        total::Float64 = sum(cells)
        return total
    end
end

reference(xs, m) = sum(-0.5 * ((x - mi) / 0.7)^2 - log(0.7) for (x, mi) in zip(xs, m))
reference_gradient(xs, m) = (xs .- m) ./ 0.7^2

allocations(kernel, xs, m) = (kernel(xs, m); @allocated kernel(xs, m))
function gradient_allocations(prepared, gradient, xs, m)
    ad_value_and_gradient!(prepared, gradient, xs, m)
    @allocated ad_value_and_gradient!(prepared, gradient, xs, m)
end

# A scan child in a lazy arm of an undeclared plate cell is prepared once and
# applied through `_KernelPreparedChild`. Native reverse of the caller used to
# depend on whether the child had been compiled before the caller: cold, the
# cell's call stayed uninferred and differentiated; warm, it inferred and
# failed with `EnzymeRuntimeActivityError` (Julia 1.10).
@kernel running_total(xs, gain) = begin
    updates = scan(xs; init = 0.0) do carry, x
        next = carry + x * gain
        (next, next)
    end
    total = sum(updates)
    return total
end

@kernel group_totals(groups, gain) = begin
    cells = plate(groups) do xs
        gain > 0.0 ? running_total(xs, gain) : 0.0
    end
    total = sum(cells)
    return total
end

call_allocations(kernel, a, b) = (kernel(a, b); @allocations kernel(a, b))

running_reference(xs, gain) =
    (sum(gain * sum(@view xs[1:i]) for i in eachindex(xs); init = 0.0),
     sum(sum(@view xs[1:i]) for i in eachindex(xs); init = 0.0))
end

@testset "Nested source operations infer in the first compiled kernel" begin
    F = NestedSourceInferenceFixtures
    xs = collect(range(-1.0, 1.0; length=5))
    m = sin.(1.0:5)
    original = (copy(xs), copy(m))

    first, second = prepare(F.primal_first), prepare(F.primal_second)
    first_allocations = Base.invokelatest(F.allocations, first, xs, m)
    second_allocations = Base.invokelatest(F.allocations, second, xs, m)
    @test Base.invokelatest(first, xs, m) ≈ F.reference(xs, m)
    @test Base.invokelatest(second, xs, m) == Base.invokelatest(first, xs, m)
    @test first_allocations == second_allocations == 0

    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    prepared = [prepare_ad(prepare(spec), backend, xs, m; active=:m)
                for spec in (F.reverse_first, F.reverse_second)]
    gradients = [similar(m), similar(m)]
    reverse_allocations = [
        Base.invokelatest(F.gradient_allocations, ad, gradient, xs, m)
        for (ad, gradient) in zip(prepared, gradients)]
    @test gradients[1] ≈ F.reference_gradient(xs, m)
    @test gradients[2] == gradients[1]
    @test reverse_allocations[1] == reverse_allocations[2]
    @test (xs, m) == original
end

@testset "A prepared child compiled first keeps its caller's reverse" begin
    F = NestedSourceInferenceFixtures
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    child = prepare(F.running_total)
    xs = [1.0, 2.0]
    child_ad = prepare_ad(child, backend, xs, 0.5; active=:gain)
    @test Base.invokelatest(child, xs, 0.5) ≈ first(F.running_reference(xs, 0.5))
    @test collect(Base.invokelatest(ad_value_and_gradient, child_ad, xs, 0.5)) ≈
        collect(F.running_reference(xs, 0.5))

    groups = [[1.0, 2.0, 0.5], [2.0], Float64[]]
    saved = deepcopy(groups)
    kernel = prepare(F.group_totals)
    ad = prepare_ad(kernel, backend, groups, 0.5; active=:gain)
    for gain in (0.5, 1.3)
        expected = sum(first(F.running_reference(xs, gain)) for xs in groups)
        derivative = sum(last(F.running_reference(xs, gain)) for xs in groups)
        @test Base.invokelatest(kernel, groups, gain) ≈ expected
        value, gradient = Base.invokelatest(ad_value_and_gradient, ad, groups, gain)
        @test value ≈ expected
        @test gradient ≈ derivative
    end
    @test Base.invokelatest(ad_value_and_gradient, ad, groups, -0.5) == (0.0, 0.0)
    @test groups == saved

    # Calling the child adds nothing per cell: a cell allocates only what the
    # child allocates on its own (boxed forwarded arguments would add more).
    filled = [[1.0, 2.0, 0.5], [2.0], [3.0, 4.0]]
    alone = sum(Base.invokelatest(F.call_allocations, child, xs, 0.5) for xs in filled)
    @test Base.invokelatest(F.call_allocations, kernel, filled, 0.5) == alone
end

@testset "Source-call recursion relation" begin
    RK = ReactiveKernels
    relation = RK._source_call_recursion_well_founded
    a = RK._KernelSourceOp(Val(:nested_a), Val(:fused), sin)
    b = RK._KernelSourceOp(Val(:nested_b), Val(:fused), cos)
    wrapped = RK._KernelSourceOp(Val(:nested_c), Val(:fused),
                                 RK._KernelSource((;), nothing, a))
    A, B, W = typeof(a), typeof(b), typeof(wrapped)
    call = typeof(RK._kernel_source_call)
    # Distinct operations, called directly or forwarded, are well founded.
    @test relation(nothing, nothing, Tuple{A,Float64}, Tuple{B,Float64})
    @test relation(nothing, nothing, Tuple{call,A,Float64,Float64},
                   Tuple{call,B,Float64,Float64})
    # The same operation again, an operation wrapping its caller, and a call
    # that is not a source call keep Julia's default limiting.
    @test !relation(nothing, nothing, Tuple{A,Float64}, Tuple{A,Int})
    @test !relation(nothing, nothing, Tuple{W,Float64}, Tuple{A,Float64})
    @test !relation(nothing, nothing, Tuple{typeof(sin),Float64}, Tuple{A,Float64})
    if hasfield(Method, :recursion_relation)
        for sig in (Tuple{RK._KernelSourceOp,Vararg{Any}},
                    Tuple{RK._KernelSource,Vararg{Any}},
                    Tuple{RK._KernelBranch,Vararg{Any}},
                    Tuple{RK._KernelReduction,Vararg{Any}},
                    Tuple{RK._IgnoredThrowFunction,Vararg{Any}})
            @test which(sig).recursion_relation === relation
        end
        # `@traceable` adds `_ignored_throw_call` methods for the caller's own
        # functions; only the package's methods carry source callables.
        for f in (RK._kernel_source_call, RK._ignored_throw_call)
            own = filter(method -> method.module === RK, collect(methods(f)))
            @test !isempty(own)
            @test all(method -> method.recursion_relation === relation, own)
        end
    end
end

# A prepared child that owns a plate, whose cell calls a child with a lazy arm
# into a scan child, called itself inside a lazy arm: the shape of
# DistributionKernels' one-sided LogNormal `var()`. Each call re-enters the
# prepared-call methods. Inferred cold, as Enzyme infers in a cache of its
# own, the parent inferred `Any` although native inference, warmed by
# preparation, was concrete; an RKPPL sampler whose observation plate read
# that value failed static Reverse (snag `native-lognormal-a666a8e1`).
module PreparedChildInferenceFixtures
using ReactiveKernels
const NODES = (0.1, 0.4, 0.7, 0.95)
const WEIGHTS = (0.3, 0.3, 0.25, 0.15)

@kernel tail_fraction(z::Float64) = begin
    n::Int = ceil(Int, 6 + 4 / (abs(z) + 1))
    path = scan(n:-1:2, Ref(z); init = 1.0) do t, k, z
        next = k / (abs(z) + 1 + t)
        (next, next)
    end
    t::Float64 = last(path)
    return t
end
@kernel cell_value(z::Float64) = begin
    result::Float64 = if z < 0.5
        1 - z^2 / 10
    else
        tail_fraction(z)
    end
    return result
end
@kernel quadrature(c::Float64, s::Float64) = begin
    nodes = NODES
    weights = WEIGHTS
    terms = plate(nodes, weights, Ref(c), Ref(s)) do x, w, c, s
        w * (cell_value(c + s * x) + cell_value(c - s * x))
    end
    return s^2 * sum(terms)
end
@kernel lazy_parent(c::Float64, s::Float64) = begin
    result::Float64 = if s <= 2.0
        quadrature(c, s)
    else
        s^2
    end
    return result
end
@kernel parent_density(q::Vector{Float64}) = begin
    return lazy_parent(q[1], exp(q[2]))
end

function tail_reference(z)
    n = ceil(Int, 6 + 4 / (abs(z) + 1))
    t = 1.0
    for k in n:-1:2
        t = k / (abs(z) + 1 + t)
    end
    t
end
cell_reference(z) = z < 0.5 ? 1 - z^2 / 10 : tail_reference(z)
quadrature_reference(c, s) = s^2 * sum(w * (cell_reference(c + s * x) +
    cell_reference(c - s * x)) for (x, w) in zip(NODES, WEIGHTS))
parent_reference(c, s) = s <= 2.0 ? quadrature_reference(c, s) : s^2
end

@testset "A prepared child in a lazy arm infers concretely in a cold interpreter" begin
    F = PreparedChildInferenceFixtures
    # A fresh Enzyme interpreter cache: cold for every method above, whatever
    # native inference holds (Enzyme's own `test/inference_recursion.jl`).
    cold() = Enzyme.Compiler.Interpreter.EnzymeInterpreter(
        VERSION >= v"1.11.0-DEV.1552" ? gensym(:prepared_child_inference) :
            Enzyme.Compiler.GPUCompiler.CodeCache(),
        nothing, Base.get_world_counter(), false, false, false)
    for (spec, reference) in ((F.quadrature, F.quadrature_reference),
                              (F.lazy_parent, F.parent_reference))
        k = prepare(spec)
        @test only(Base.code_typed(k, (Float64, Float64);
                                   interp = cold(), optimize = false))[2] === Float64
        for (c, s) in ((0.3, 0.7), (-1.2, 1.5), (0.4, 2.5))
            @test Base.invokelatest(k, c, s) ≈ reference(c, s)
        end
    end

    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    k = prepare(F.parent_density)
    for q in ([0.3, log(0.7)], [-1.2, log(1.5)])
        before = copy(q)
        ad = prepare_ad(k, backend, q; active=:q)
        value, gradient = Base.invokelatest(ad_value_and_gradient, ad, q)
        h = 1e-6
        central = map(1:2) do i
            up, down = copy(q), copy(q)
            up[i] += h
            down[i] -= h
            (F.parent_reference(up[1], exp(up[2])) -
             F.parent_reference(down[1], exp(down[2]))) / 2h
        end
        @test value ≈ F.parent_reference(q[1], exp(q[2]))
        @test gradient ≈ central rtol=1e-6
        @test q == before
    end
end

@testset "Prepared-call recursion relation" begin
    RK = ReactiveKernels
    F = PreparedChildInferenceFixtures
    relation = RK._prepared_call_recursion_well_founded
    inner, outer = prepare(F.cell_value), prepare(F.quadrature)
    I, O = typeof(inner), typeof(outer)
    call = typeof(RK._prepared_call)
    # A different prepared program called inside another is well founded.
    @test relation(nothing, nothing, Tuple{call,I,Tuple{Float64},Val{1}},
                   Tuple{call,O,Tuple{Float64,Float64},Val{2}})
    # The same program again, and a call that is not a prepared call, keep
    # Julia's default limiting.
    @test !relation(nothing, nothing, Tuple{call,O,Tuple{Float64,Float64},Val{2}},
                    Tuple{call,O,Tuple{Int,Float64},Val{2}})
    @test !relation(nothing, nothing, Tuple{typeof(sin),Float64},
                    Tuple{call,O,Tuple{Float64,Float64},Val{2}})
    if hasfield(Method, :recursion_relation)
        marked = [methods(RK._prepared_call)..., methods(RK._native_array_body_call)...,
                  which(Tuple{RK._ArrayFunction,Any,Any,Vararg{Any}})]
        @test all(method -> method.recursion_relation === relation, marked)
        # The generated-function methods of ReactiveKernels' own cache; those
        # of other modules keep Julia's default.
        own = filter(method -> method.module === RK,
            collect(methods(RK.RuntimeGeneratedFunctions.generated_callfunc)))
        @test !isempty(own)
        @test all(method -> method.recursion_relation === relation, own)
        @test all(method -> !isdefined(method, :recursion_relation),
            filter(method -> method.module === RK.RuntimeGeneratedFunctions,
                   collect(methods(RK.RuntimeGeneratedFunctions.generated_callfunc))))
    end
end
