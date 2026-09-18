using ReactiveKernelsReactantODESolvers
using Test

const RKRO = ReactiveKernelsReactantODESolvers

@testset "error norm and estimate" begin
    @test RKRO.rms_norm([3.0, 4.0]) ≈ sqrt(12.5)
    @test RKRO.rms_norm([2.0]) == 2.0
    @test_throws ArgumentError RKRO.rms_norm(Float64[])

    # scale = 1e-6 + max(1, 1)*1e-3 = 0.001001; EEst = 0.002/0.001001.
    EEst = RKRO.error_estimate([0.002], [1.0], [1.0], 1e-6, 1e-3)
    @test EEst ≈ 0.002 / 0.001001
    # Mixed-scale states take the per-component max.
    EEst2 = RKRO.error_estimate([1.0, 1.0], [0.0, 100.0], [0.0, 100.0], 1e-6,
        1e-3)
    @test EEst2 ≈ sqrt(((1 / 1e-6)^2 + (1 / (1e-6 + 0.1))^2) / 2)
end

@testset "PI controller factors" begin
    @test RKRO.pi_factors(0.0, 0.5) == (0.1, 0.0)
    q, q11 = RKRO.pi_factors(1.0, 1e-4)
    @test q11 == 1.0^0.14
    @test q ≈ (1.0 / (1e-4^0.08)) / 0.9
    @test 0.1 <= q <= 5.0
    # Monotone in the estimate at fixed history.
    @test RKRO.pi_factors(0.5, 1.0)[1] < RKRO.pi_factors(1.0, 1.0)[1] <
          RKRO.pi_factors(2.0, 1.0)[1]
    # Clamp branches hit exactly.
    @test RKRO.pi_factors(1e10, 1e-4)[1] == 5.0
    @test RKRO.pi_factors(1e-12, 1.0)[1] == 0.1

    @test RKRO.pi_accept_dt(0.5, 2.0) == 0.25
    @test RKRO.pi_reject_dt(0.5, 1.0) ≈ 0.45
    @test RKRO.pi_reject_dt(0.5, 100.0) == 0.1  # 0.5 / min(5, 100/0.9)
end

@testset "automatic initial step" begin
    u0 = [1.0]
    dt0, f0, spent = RKRO.initial_dt(exponential_decay, u0, [2.0], 0.0, 1.0,
        5.0, 1e-6, 1e-3)
    @test spent == 2
    @test f0 == exponential_decay(u0, [2.0], 0.0)
    @test 0.0 < dt0 <= 5.0

    # Backward spans take a negative step.
    dt0_back, _, _ = RKRO.initial_dt(exponential_decay, u0, [2.0], 5.0, -1.0,
        5.0, 1e-6, 1e-3)
    @test dt0_back < 0.0

    # Constant RHS skips the curvature probe: 100x the Euler guess.
    dt0_const, _, spent_const = RKRO.initial_dt((u, p, t) -> [2.0], [1.0],
        nothing, 0.0, 1.0, 10.0, 1e-6, 1e-3)
    @test spent_const == 2
    @test dt0_const ≈ 0.5

    # Non-finite initial derivative is reported, not stepped from.
    dt0_nan, f0_nan, spent_nan = RKRO.initial_dt((u, p, t) -> [NaN], [1.0],
        nothing, 0.0, 1.0, 10.0, 1e-6, 1e-3)
    @test spent_nan == 1
    @test isnan(f0_nan[1])
    @test 0.0 < dt0_nan <= 10.0
end

@testset "dense output identities" begin
    tab = RKRO.Tsit5Tableau{Float64}()
    dense = RKRO.Tsit5DenseCoefficients{Float64}()
    buffers = RKRO.Tsit5Buffers{Float64}(2)
    uprev = [1.0, 3.0]
    copyto!(buffers.k1, lotka_volterra(uprev, LOTKA_PARAMS, 0.0))
    EEst = RKRO.tsit5_step!(buffers, lotka_volterra, uprev, LOTKA_PARAMS, 0.0,
        0.1, tab, 1e-6, 1e-3)
    @test isfinite(EEst)
    stages = (buffers.k1, buffers.k2, buffers.k3, buffers.k4, buffers.k5,
        buffers.k6, buffers.k7)

    at_start = similar(uprev)
    RKRO.tsit5_dense_eval!(at_start, uprev, stages, 0.1, 0.0, dense)
    @test at_start == uprev

    at_end = similar(uprev)
    RKRO.tsit5_dense_eval!(at_end, uprev, stages, 0.1, 1.0, dense)
    @test at_end ≈ buffers.u rtol = 1e-12

    weights = RKRO.tsit5_dense_weights(0.0, dense)
    @test all(iszero, weights)
end
