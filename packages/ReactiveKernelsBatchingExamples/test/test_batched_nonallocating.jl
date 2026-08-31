# Zero-allocation batched value AND gradient, proven against the pinned
# MutatingFunctions revision. Runs only through
# `test/run_nonallocating_integration.jl`, not the default suite, because it
# needs the unregistered weak dependency.
using ReactiveKernels
using MutatingFunctions
using DifferentiationInterface
using DifferentiationInterface: Cache, Constant
import Enzyme
using BenchmarkTools
using Random
using Test

# One scalar pointwise density, written once. Batching is achieved purely by
# passing a `Vector` for the observation port.
batched_normal_logpdf(x, μ, σ) = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)^2

# The pointwise density is a TYPED port so the elementwise recipe's operation is
# the bare `broadcast`. That is what lets `prepare_nonallocating` route through
# MutatingFunctions' `broadcast!` `apply!!` and reuse a batch buffer; baking the
# function in as a constant makes the operation an anonymous closure that hits
# the allocating generic fallback instead.
batched_spec = @kernel batched_normal(
        pointwise::typeof(batched_normal_logpdf),
        x::Vector{Float64}, μ::Float64, logσ::Float64) = begin
    σ::Float64 = exp(logσ)
    per_obs::Vector{Float64} = broadcast(pointwise, x, μ, σ)
    logdensity::Float64 = sum(per_obs)
end

# The authored spelling has one transparent plate node and one sum consumer.
# Its pointwise cut is the caller-owned-buffer boundary: the weak extension
# fills the borrowed array directly with the same Julia broadcast semantics as
# ordinary execution.
@kernel authored_batched_pointwise() = begin
    logpdf(x::Float64, μ::Float64, σ::Float64)::Float64 =
        batched_normal_logpdf(x, μ, σ)
end

authored_batched_spec = @kernel authored_batched_normal(
        x::Vector{Float64}, μ::Float64, σ::Float64) = begin
    pointwise = plate(x, μ, σ) do xi, mi, si
        authored_batched_pointwise().logpdf(xi, mi, si)
    end
    return sum(pointwise)
end

# The owned DI Cache gives Enzyme explicit activity at the call boundary, so
# ordinary reverse mode suffices; no runtime activity or function annotation.
const BATCHED_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

@testset "Batched log density: zero-allocation value and gradient" begin
    @test Base.get_extension(ReactiveKernels,
                             :ReactiveKernelsMutatingFunctionsExt) !== nothing

    N = 1000
    rng = Random.MersenneTwister(1)
    x = randn(rng, N)
    μ = 0.3
    logσ = log(1.2)
    σ = exp(logσ)

    total_plan  = plan(batched_spec;
                       have = (:pointwise, :x, :μ, :logσ), want = :logdensity)
    perobs_plan = plan(batched_spec;
                       have = (:pointwise, :x, :μ, :logσ), want = :per_obs)

    @testset "value: prepare_nonallocating reuses the per-obs buffer" begin
        total_alloc  = prepare(total_plan)          # allocating baseline
        total_reuse  = prepare_nonallocating(total_plan)
        perobs_reuse = prepare_nonallocating(perobs_plan)

        oracle_total  = sum(broadcast(batched_normal_logpdf, x, μ, σ))
        oracle_perobs = broadcast(batched_normal_logpdf, x, μ, σ)

        @test total_reuse(batched_normal_logpdf, x, μ, logσ) ≈ oracle_total
        @test perobs_reuse(batched_normal_logpdf, x, μ, logσ) ≈ oracle_perobs
        @test length(perobs_reuse(batched_normal_logpdf, x, μ, logσ)) == N

        # The plain `prepare` path allocates the per-observation buffer once.
        alloc_once = @ballocated $total_alloc($batched_normal_logpdf,
                                              $x, $μ, $logσ)
        println("BATCHED_ALLOC\tordinary_value\t", alloc_once)
        @test alloc_once > 0

        # The non-allocating path drives both want boundaries to zero bytes.
        total_bytes = @ballocated $total_reuse($batched_normal_logpdf,
                                              $x, $μ, $logσ)
        perobs_bytes = @ballocated $perobs_reuse($batched_normal_logpdf,
                                               $x, $μ, $logσ)
        println("BATCHED_ALLOC\tnonalloc_total\t", total_bytes)
        println("BATCHED_ALLOC\tnonalloc_perobs\t", perobs_bytes)
        @test total_bytes == 0
        @test perobs_bytes == 0
    end

    @testset "authored plate: pointwise cut reuses its broadcast buffer" begin
        pointwise = prepare_nonallocating(extract(
            authored_batched_spec; want = :pointwise))
        expected = broadcast(batched_normal_logpdf, x, μ, σ)

        first_result = pointwise(x, μ, σ)
        second_result = pointwise(x, μ, σ)
        @test first_result === second_result
        @test second_result ≈ expected

        pointwise_bytes = @ballocated $pointwise($x, $μ, $σ)
        println("BATCHED_ALLOC\tauthored_plate_pointwise\t", pointwise_bytes)
        @test pointwise_bytes == 0
    end

    @testset "gradient: one reverse pass, zero bytes via an owned buffer" begin
        analytic = @. -(x - μ) / σ^2

        # The zero-allocation gradient does NOT differentiate the borrowed-cache
        # `prepare_nonallocating` kernel: its caches are overwritten between the
        # forward and reverse sweeps and silently corrupt the adjoint (the AD
        # instance of the "owned state, not borrowed caches" rule in
        # docs/src/nonallocating.md). Instead the batch buffer is a preallocated
        # DifferentiationInterface `Cache` that Enzyme owns and shadows itself.
        buffer = similar(x)
        primal(v, buf, m, l) = begin
            s = exp(l)
            broadcast!(batched_normal_logpdf, buf, v, m, s)
            sum(buf)
        end
        prep = DifferentiationInterface.prepare_gradient(
            primal, BATCHED_BACKEND, x,
            Cache(buffer), Constant(μ), Constant(logσ))
        grad = similar(x)

        # Correct, and deterministic across repeated calls with fresh data.
        for _ in 1:3
            xi = randn(rng, N)
            DifferentiationInterface.gradient!(
                primal, grad, prep, BATCHED_BACKEND, xi,
                Cache(buffer), Constant(μ), Constant(logσ))
            @test grad ≈ (@. -(xi - μ) / σ^2)
        end
        DifferentiationInterface.gradient!(
            primal, grad, prep, BATCHED_BACKEND, x,
            Cache(buffer), Constant(μ), Constant(logσ))
        @test grad ≈ analytic

        grad_bytes = @ballocated DifferentiationInterface.gradient!(
            $primal, $grad, $prep, $BATCHED_BACKEND, $x,
            Cache($buffer), Constant($μ), Constant($logσ))
        println("BATCHED_ALLOC\tnonalloc_gradient\t", grad_bytes)
        @test grad_bytes == 0
    end
end
