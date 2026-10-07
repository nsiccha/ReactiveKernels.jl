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
        @test all(method -> method.recursion_relation === relation,
                  methods(RK._kernel_source_call))
        @test all(method -> method.recursion_relation === relation,
                  methods(RK._ignored_throw_call))
    end
end
