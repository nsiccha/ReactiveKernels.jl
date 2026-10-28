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
    dt0 = initial_dt(exponential_decay, u0, (0.0, 5.0); p=[2.0])
    @test 0.0 < dt0 <= 5.0

    # The public form returns a magnitude in both directions.
    dt0_back = initial_dt(exponential_decay, u0, (5.0, 0.0); p=[2.0])
    @test 0.0 < dt0_back <= 5.0

    # The signed core takes a negative step on backward spans.
    dt0_signed, f0, spent = RKRO._initial_dt(exponential_decay, u0, [2.0],
        5.0, -1.0, 5.0, 1e-6, 1e-3)
    @test spent == 2
    @test f0 == exponential_decay(u0, [2.0], 5.0)
    @test dt0_signed < 0.0

    # Constant RHS skips the curvature probe: 100x the Euler guess.
    dt0_const = initial_dt((u, p, t) -> [2.0], [1.0], (0.0, 10.0))
    @test dt0_const ≈ 0.5

    # Non-finite initial derivative is reported, not stepped from.
    dt0_nan, f0_nan, spent_nan = RKRO._initial_dt((u, p, t) -> [NaN], [1.0],
        nothing, 0.0, 1.0, 10.0, 1e-6, 1e-3)
    @test spent_nan == 1
    @test isnan(f0_nan[1])
    @test 0.0 < dt0_nan <= 10.0
end

@testset "dense output identities" begin
    tab = RKRO.Tsit5Tableau{Float64}()
    dense = RKRO.Tsit5DenseCoefficients{Float64}()
    uprev = [1.0, 3.0]
    k1 = lotka_volterra(uprev, LOTKA_PARAMS, 0.0)
    step = RKRO.tsit5_step(lotka_volterra, uprev, k1, LOTKA_PARAMS, 0.0, 0.1,
        tab, 1e-6, 1e-3)
    @test isfinite(step.EEst)
    @test length(step.k) == 7

    @test RKRO.tsit5_dense_eval(uprev, step.k, 0.1, 0.0, dense) == uprev
    @test RKRO.tsit5_dense_eval(uprev, step.k, 0.1, 1.0, dense) ≈ step.u rtol =
        1e-12

    weights = RKRO.tsit5_dense_weights(0.0, dense)
    @test all(iszero, weights)
end

@testset "saveat buffer emission" begin
    tab = RKRO.Tsit5Tableau{Float64}()
    dense = RKRO.Tsit5DenseCoefficients{Float64}()
    uprev = [1.0, 3.0]
    k1 = lotka_volterra(uprev, LOTKA_PARAMS, 0.0)
    step = RKRO.tsit5_step(lotka_volterra, uprev, k1, LOTKA_PARAMS, 0.0, 0.1,
        tab, 1e-6, 1e-3)
    blank = zeros(2, 3)
    saveat = [0.03, 0.07, 0.5]

    out = RKRO._emit_saveat(blank, saveat, uprev, step.k, 0.0, 0.1,
        0.0, 0.1, true, dense)
    @test out[:, 1] ≈ RKRO.tsit5_dense_eval(uprev, step.k, 0.1, 0.3, dense)
    @test out[:, 2] ≈ RKRO.tsit5_dense_eval(uprev, step.k, 0.1, 0.7, dense)
    # Out-of-window points keep their column.
    @test out[:, 3] == zeros(2)

    # Rejected steps emit nothing.
    out_rej = RKRO._emit_saveat(blank, saveat, uprev, step.k, 0.0, 0.1,
        0.0, 0.1, false, dense)
    @test out_rej == zeros(2, 3)
end
