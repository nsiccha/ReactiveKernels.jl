# `plate`: generate a Stan-parity vectorized log density from a scalar per-element
# `@kernel`. The defining property is NO REPEATED WORK — recipes that depend only
# on the shared (scalar) ports (here `σ = exp(logσ)`) are hoisted and computed
# ONCE above a fused per-element loop; only the batch-dependent recipes run per
# element; the scalar want is summed. No per-observation vector is materialized.
#
# This is a pure planning/codegen concern: ReactiveKernels PLANS and COMPUTES the
# kernel and does no AD, so nothing here differentiates — the claims are value,
# hoisting (verified in the generated AST), and no per-element materialization.
using ReactiveKernels
using ReactiveKernels: code_expr
using Test

# One scalar per-observation Gaussian log density, authored once.
@kernel plate_nlogpdf(x::Float64, μ::Float64, logσ::Float64) = begin
    σ::Float64 = exp(logσ)
    z::Float64 = (x - μ) / σ
    ld::Float64 = -0.5 * log(2π) - logσ - 0.5 * z^2
end

# Function barrier so `@allocated` measures the kernel, not global-ref boxing.
_plate_call(k, xs, μ, logσ) = k(xs, μ, logσ)

@testset "plate: vectorized log density with loop-invariant hoisting" begin
    k = plate(plate_nlogpdf; have = (:x, :μ, :logσ), want = :ld, batched = (:x,))
    xs = randn(200)
    μ = 0.3
    logσ = log(1.2)

    @testset "value equals the summed per-observation density" begin
        ref = sum(-0.5 * log(2π) - logσ - 0.5 * ((xi - μ) / exp(logσ))^2 for xi in xs)
        @test k(xs, μ, logσ) ≈ ref
    end

    @testset "invariant `exp(logσ)` is hoisted ABOVE the loop (Stan-parity)" begin
        # `__ops__[1]` is the `σ = exp(logσ)` recipe. It must be emitted once,
        # above the loop, and NOT recomputed per element.
        body = code_expr(k).args[2].args
        fori = findfirst(s -> s isa Expr && s.head === :for, body)
        @test fori !== nothing
        pre = body[1:(fori - 1)]
        loopbody = body[fori].args[2].args
        @test any(s -> occursin("__ops__[1]", string(s)), pre)
        @test !any(s -> occursin("__ops__[1]", string(s)), loopbody)
        # The batch-dependent recipes DO run in the loop.
        @test any(s -> occursin("__ops__[2]", string(s)), loopbody)
        @test any(s -> occursin("__ops__[3]", string(s)), loopbody)
    end

    @testset "no per-element materialization (allocation is O(1) in the batch)" begin
        ys10 = randn(10)
        ys1000 = randn(1000)
        _plate_call(k, ys10, μ, logσ)
        _plate_call(k, ys1000, μ, logσ)
        a10 = @allocated _plate_call(k, ys10, μ, logσ)
        a1000 = @allocated _plate_call(k, ys1000, μ, logσ)
        # Nothing scales with N — no per-observation vector is built.
        @test a10 == a1000
    end

    @testset "a want with no batched dependency is rejected" begin
        # `σ` depends only on the shared `logσ` — there is nothing to vectorize.
        @test_throws ArgumentError plate(
            plate_nlogpdf; have = (:x, :μ, :logσ), want = :σ, batched = (:x,))
    end
end
