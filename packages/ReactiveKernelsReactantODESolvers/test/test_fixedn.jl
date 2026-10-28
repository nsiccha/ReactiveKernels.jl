# Fixed-budget driver: exact step count, exact landing, viability floor,
# and accuracy-per-budget against OrdinaryDiffEq references.
#
# The honest contract under test: above the viability floor (about the
# adaptive step count) the error is tolerance-limited and extra budget is
# harmless; below the floor the landing fails loudly with `:Unstable`
# instead of returning garbage.

@testset "fixed-N exact count and landing" begin
    sol = solve_fixed_n(exponential_decay, DECAY_U0, DECAY_TSPAN, 100,
        Tsit5(); p=DECAY_RATES)
    @test sol.retcode == :Success
    @test sol.stats.naccepted == 100
    @test sol.stats.nrejected == 0
    @test sol.t[end] == DECAY_TSPAN[2]
    @test length(sol.t) == 101
    @test length(sol.u) == 101
    @test all(isfinite, sol.u[end])

    tiny = solve_fixed_n(exponential_decay, [1.0], (0.0, 1.0), 5, Tsit5();
        p=[0.5])
    @test tiny.retcode == :Success
    @test tiny.stats.naccepted == 5
    @test tiny.t[end] == 1.0
end

@testset "fixed-N budget validation" begin
    @test_throws ArgumentError solve_fixed_n(exponential_decay, DECAY_U0,
        DECAY_TSPAN, 0, Tsit5(); p=DECAY_RATES)
    @test_throws ArgumentError solve_fixed_n(exponential_decay, DECAY_U0,
        DECAY_TSPAN, -3, Tsit5(); p=DECAY_RATES)
end

@testset "fixed-N accuracy per budget" begin
    # Decay: viable at N=100, extra budget harmless (uniform tail).
    ref_decay = ode_reference(exponential_decay, DECAY_U0, DECAY_TSPAN,
        DECAY_RATES)
    d100 = solve_fixed_n(exponential_decay, DECAY_U0, DECAY_TSPAN, 100,
        Tsit5(); p=DECAY_RATES)
    d400 = solve_fixed_n(exponential_decay, DECAY_U0, DECAY_TSPAN, 400,
        Tsit5(); p=DECAY_RATES)
    err100 = max_abs_diff(d100.u[end], ref_decay.u[end])
    err400 = max_abs_diff(d400.u[end], ref_decay.u[end])
    @test err100 < 1e-6
    @test err400 <= err100

    # Lotka: phase-sensitive endpoint; budget viability, not strict
    # improvement (the tolerance-driven prefix sets the phase).
    ref_lotka = ode_reference(lotka_volterra, LOTKA_U0, LOTKA_TSPAN,
        LOTKA_PARAMS)
    l100 = solve_fixed_n(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, 100,
        Tsit5(); p=LOTKA_PARAMS)
    l400 = solve_fixed_n(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, 400,
        Tsit5(); p=LOTKA_PARAMS)
    lerr100 = max_abs_diff(l100.u[end], ref_lotka.u[end])
    lerr400 = max_abs_diff(l400.u[end], ref_lotka.u[end])
    @test l100.retcode == :Success
    @test lerr100 < 1e-2
    @test lerr400 <= lerr100 * 1.01
    @test l100.max_EEst < 10.0

    # Brusselator (40-state): viable at N=100.
    ref_brus = ode_reference(brusselator, BRUS_U0, BRUS_TSPAN, BRUS_PARAMS)
    b100 = solve_fixed_n(brusselator, BRUS_U0, BRUS_TSPAN, 100, Tsit5();
        p=BRUS_PARAMS)
    @test b100.retcode == :Success
    @test max_abs_diff(b100.u[end], ref_brus.u[end]) < 1e-3

    # Van der Pol: needs the guard retries through its spikes; viable
    # at N=400 with the achieved error honestly reported.
    ref_vdp = ode_reference(vanderpol, VDP_U0, VDP_TSPAN, VDP_MU)
    v400 = solve_fixed_n(vanderpol, VDP_U0, VDP_TSPAN, 400, Tsit5();
        p=VDP_MU)
    @test v400.retcode == :Success
    @test v400.stats.nrejected > 0
    @test v400.max_EEst < 10.0
    @test max_abs_diff(v400.u[end], ref_vdp.u[end]) < 0.05
end

@testset "fixed-N viability floor is honest" begin
    # Below the floor the landing step cannot cover the remainder and
    # the solve reports :Unstable instead of returning garbage.
    d10 = solve_fixed_n(exponential_decay, DECAY_U0, DECAY_TSPAN, 10,
        Tsit5(); p=DECAY_RATES)
    @test d10.retcode == :Unstable
    @test d10.stats.naccepted == 9

    l10 = solve_fixed_n(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, 10, Tsit5();
        p=LOTKA_PARAMS)
    @test l10.retcode == :Unstable

    v100 = solve_fixed_n(vanderpol, VDP_U0, VDP_TSPAN, 100, Tsit5();
        p=VDP_MU)
    @test v100.retcode == :Unstable
end

@testset "fixed-N saveat and backward span" begin
    times = [0.5, 1.0, 1.5]
    sol = solve_fixed_n(exponential_decay, [1.0], (0.0, 2.0), 50, Tsit5();
        p=[0.5], saveat=times)
    @test sol.retcode == :Success
    @test sol.t == [0.0, 0.5, 1.0, 1.5, 2.0]
    for (t, u) in zip(sol.t, sol.u)
        @test max_abs_diff(u, [exp(-0.5 * t)]) < 1e-6
    end

    back = solve_fixed_n(exponential_decay, [1.0], (2.0, 0.0), 50,
        Tsit5(); p=[0.5])
    @test back.retcode == :Success
    @test back.stats.naccepted == 50
    @test back.t[end] == 0.0
    # Backward from (t=2, u=1): u(0) = exp(+0.5 * 2).
    @test max_abs_diff(back.u[end], [exp(0.5 * 2.0)]) < 1e-6
end

@testset "fixed-N guards and types" begin
    dmin = solve_fixed_n(exponential_decay, DECAY_U0, DECAY_TSPAN, 50,
        Tsit5(); p=DECAY_RATES, dtmin=1.0)
    @test dmin.retcode == :DtLessThanDtMin

    f32 = solve_fixed_n(exponential_decay, Float32[1, 1], (0f0, 2f0), 50,
        Tsit5(); p=Float32[0.5, 1.0])
    @test f32.retcode == :Success
    @test eltype(f32.u[1]) == Float32
    @test maximum(abs.(f32.u[end] .- exp.(-Float32[0.5, 1.0] * 2))) < 1e-5

    shown = sprint(show, dmin)
    @test occursin("FixedNSolution", shown)
end
