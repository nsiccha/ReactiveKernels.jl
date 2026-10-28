# Guards, retcodes, and input validation. Non-:Success outcomes return the
# partial trajectory with a retcode; only malformed configuration throws.

@testset "maxiters guard" begin
    sol = solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, Tsit5();
        p=LOTKA_PARAMS, maxiters=3)
    @test sol.retcode == :MaxItersExceeded
    @test sol.stats.naccepted + sol.stats.nrejected == 3
    @test length(sol.t) >= 1
    @test sol.t[1] == 0.0

    none = solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, Tsit5();
        p=LOTKA_PARAMS, maxiters=0)
    @test none.retcode == :MaxItersExceeded
    @test none.t == [0.0]
    @test none.u == [LOTKA_U0]
end

@testset "dtmax handling" begin
    # Every accepted step obeys |dt| <= dtmax, so spanning [0, 10] with
    # dtmax = 0.1 takes at least 100 accepted steps.
    sol = solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, Tsit5();
        p=LOTKA_PARAMS, dtmax=0.1)
    @test sol.retcode == :Success
    @test sol.stats.naccepted >= 100
    @test all(d -> d <= 0.1 * (1 + 1e-12), diff(sol.t))
end

@testset "dtmin guard" begin
    # A huge dtmin trips on the first controller proposal: the initial step
    # is floored to dtmin, its error rejects, and the shrunken retry falls
    # below dtmin before anything is accepted.
    sol = solve_ode(vanderpol, VDP_U0, VDP_TSPAN, Tsit5(); p=VDP_MU, dtmin=1.0)
    @test sol.retcode == :DtLessThanDtMin
    @test sol.stats.naccepted == 0
    @test sol.t == [0.0]
end

@testset "NaN handling" begin
    nan_rhs(u, p, t) = [NaN, NaN]
    sol = solve_ode(nan_rhs, [1.0, 1.0], (0.0, 1.0), Tsit5())
    @test sol.retcode == :Unstable
    @test sol.stats.nfevals == 1

    # NaN appearing mid-integration returns the finite prefix.
    function late_nan(u, p, t)
        t < 5.0 ? lotka_volterra(u, p, t) : [NaN, NaN]
    end
    sol2 = solve_ode(late_nan, LOTKA_U0, LOTKA_TSPAN, Tsit5(); p=LOTKA_PARAMS)
    @test sol2.retcode == :Unstable
    @test length(sol2.t) > 1
    @test all(isfinite, sol2.t)
    @test all(u -> all(isfinite, u), sol2.u)
end

@testset "input validation" begin
    @test_throws ArgumentError solve_ode(lotka_volterra, LOTKA_U0, (0.0, 0.0),
        Tsit5(); p=LOTKA_PARAMS)
    @test_throws ArgumentError solve_ode(lotka_volterra, LOTKA_U0,
        (0.0, 1.0, 2.0), Tsit5(); p=LOTKA_PARAMS)
    @test_throws ArgumentError solve_ode(lotka_volterra, LOTKA_U0,
        (0.0, Inf), Tsit5(); p=LOTKA_PARAMS)
    @test_throws ArgumentError solve_ode(lotka_volterra, Float64[], LOTKA_TSPAN,
        Tsit5(); p=LOTKA_PARAMS)
    @test_throws ArgumentError solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN,
        Tsit5(); p=LOTKA_PARAMS, saveat=[-1.0])
    @test_throws ArgumentError solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN,
        Tsit5(); p=LOTKA_PARAMS, saveat=[NaN])
    @test_throws ArgumentError solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN,
        Tsit5(); p=LOTKA_PARAMS, dt=0.0)
    @test_throws ArgumentError solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN,
        Tsit5(); p=LOTKA_PARAMS, dtmax=0.0)
    @test_throws ArgumentError solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN,
        Tsit5(); p=LOTKA_PARAMS, dtmin=-1.0)
    @test_throws ArgumentError solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN,
        Tsit5(); p=LOTKA_PARAMS, abstol=-1.0)
    @test_throws ArgumentError solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN,
        Tsit5(); p=LOTKA_PARAMS, maxiters=-1)
    @test_throws ArgumentError solve_ode((u, p, t) -> [1.0, 2.0, 3.0],
        LOTKA_U0, LOTKA_TSPAN, Tsit5(); p=LOTKA_PARAMS)
    @test_throws ArgumentError solve_ode(lotka_volterra, ComplexF64[1, 1],
        LOTKA_TSPAN, Tsit5(); p=LOTKA_PARAMS)
end

@testset "solution display" begin
    sol = solve_ode(lotka_volterra, LOTKA_U0, (0.0, 1.0), Tsit5();
        p=LOTKA_PARAMS)
    text = repr(sol)
    @test occursin("Success", text)
    @test occursin("accepted", text)
end
