# Generated vectorized lpdf via the `plate` marker: an author writes a SCALAR
# pointwise log density over a batched observation port and wraps it in `plate`,
# and the batched graph (broadcast the scalar op over the batch, then sum-reduce
# to the scalar log density) is generated at authoring time — no hand-written
# `broadcast`/`sum`. Value and gradient are matched against the hand-written
# broadcast+sum oracle, with one fused Enzyme reverse pass over the whole batch.
using ReactiveKernels
using ReactiveKernels: _PointwiseOp
using Test
using DifferentiationInterface
import Enzyme

# One scalar pointwise log density, written once. Nothing here is
# batching-specific: batching comes from passing a `Vector` for `x`.
normal_lpdf(x, μ, σ) = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)^2

@testset "Generated vectorized lpdf (plate marker → broadcast + sum)" begin
    # The author writes the scalar term over the batched `x` port, marked with
    # `plate`. The broadcast + sum are generated: a scalar `logdensity` output ⇒
    # broadcast the pointwise op over the batch, then sum-reduce.
    vlpdf = @kernel vnormal(x::Vector{Float64}, μ::Float64, σ::Float64) = begin
        logdensity::Float64 = plate(normal_lpdf, x, μ, σ)
    end

    x = randn(200)
    μ = 0.3
    σ = 1.2

    k = prepare(vlpdf; have = (:x, :μ, :σ), want = :logdensity)

    @testset "value matches the hand-written broadcast+sum oracle" begin
        got = k(x, μ, σ)
        ref = sum(broadcast(normal_lpdf, x, μ, σ))
        @test got ≈ ref
    end

    @testset "the plate marker generated a pointwise + sum split" begin
        # The marked recipe becomes two: a broadcast lift over the batch and a
        # sum reduction to the scalar. Ops are referenced positionally in the
        # lowered code, so assert on the plan's recipe ops, not on `code_expr`.
        @test length(k.plan.recipes) == 2
        @test count(r -> r.op isa _PointwiseOp, k.plan.recipes) == 1
        @test count(r -> r.op === sum, k.plan.recipes) == 1
    end

    @testset "one fused Enzyme reverse pass, gradient matches the oracle" begin
        # Reverse-mode gradient over the whole N-batch in one pass; the batch
        # mixes the active `x` with constant `μ`, `σ`, so Enzyme needs runtime
        # activity and the closed-over kernel is Const (same config as the
        # hand-authored batched example).
        backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                               function_annotation = Enzyme.Const)
        grad = DifferentiationInterface.gradient(v -> k(v, μ, σ), backend, x)
        analytic = @. -(x - μ) / σ^2
        @test grad ≈ analytic
    end

    @testset "an ARRAY-declared output yields the per-observation vector" begin
        # Same marked term, array-declared output ⇒ the pure broadcast (no sum):
        # the per-observation log density (LOO/WAIC territory).
        vperobs = @kernel vnormal_perobs(x::Vector{Float64}, μ::Float64,
                                         σ::Float64) = begin
            per_obs::Vector{Float64} = plate(normal_lpdf, x, μ, σ)
        end
        kp = prepare(vperobs; have = (:x, :μ, :σ), want = :per_obs)
        @test length(kp.plan.recipes) == 1
        @test kp(x, μ, σ) ≈ broadcast(normal_lpdf, x, μ, σ)
    end

    @testset "a port-valued plate function is rejected, not misbound" begin
        # Passing a declared port as `plate`'s function is the explicit-broadcast
        # pattern, not the marker; the marker rejects it at expansion rather than
        # misbinding the broadcast inputs.
        @test_throws Exception macroexpand(@__MODULE__, :(
            @kernel vbad(pointwise::typeof(normal_lpdf), x::Vector{Float64},
                         μ::Float64, σ::Float64) = begin
                logdensity::Float64 = plate(pointwise, x, μ, σ)
            end))
    end
end
