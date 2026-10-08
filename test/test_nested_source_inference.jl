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
    updates = scan(xs, Ref(gain); init = 0.0) do carry, x, g
        next = carry + x * g
        (next, next)
    end
    total = sum(updates)
    return total
end

@kernel group_totals(groups, gain) = begin
    cells = plate(groups, Ref(gain)) do xs, g
        g > 0.0 ? running_total(xs, g) : 0.0
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
                                 RK._KernelSourceFunction(a, nothing, nothing))
    A, B, W = typeof(a), typeof(b), typeof(wrapped)
    call = typeof(RK._kernel_source_call)
    # Distinct operations, called directly or forwarded, are well founded.
    @test relation(nothing, nothing, Tuple{A,Float64}, Tuple{B,Float64})
    @test relation(nothing, nothing, Tuple{call,Val{:native},A,Float64,Float64},
                   Tuple{call,Val{:native},B,Float64,Float64})
    # The same operation again, an operation wrapping its caller, and a call
    # that is not a source call keep Julia's default limiting.
    @test !relation(nothing, nothing, Tuple{A,Float64}, Tuple{A,Int})
    @test !relation(nothing, nothing, Tuple{W,Float64}, Tuple{A,Float64})
    @test !relation(nothing, nothing, Tuple{typeof(sin),Float64}, Tuple{A,Float64})
    if hasfield(Method, :recursion_relation)
        for sig in (Tuple{RK._KernelSourceOp,Vararg{Any}},
                    Tuple{RK._KernelSourceFunction,Vararg{Any}},
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
