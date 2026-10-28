# Agreement against tight-tolerance OrdinaryDiffEq Tsit5 references.
#
# Bars are tolerance-scaled: error below 100x the test tolerance times the
# solution scale. Reference tolerances (1e-13/1e-11) sit ~1000x below the
# test tolerances so reference error is negligible in the comparison.

function agreement_bar(abstol, reltol, scale)
    100 * max(abstol, reltol * scale)
end

solution_scale(us) = maximum(maximum(abs, u) for u in us)

@testset "Lotka-Volterra agreement" begin
    saveat = collect(0.0:0.5:10.0)
    sol = solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, Tsit5();
        p=LOTKA_PARAMS, abstol=1e-10, reltol=1e-8, saveat=saveat)
    @test sol.retcode == :Success
    @test sol.t == saveat
    ref = ode_reference(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, LOTKA_PARAMS;
        saveat=saveat)
    @test ref.t == saveat
    scale = solution_scale(ref.u)
    bar = agreement_bar(1e-10, 1e-8, scale)
    @test max_abs_diff(sol.u, ref.u) < bar
    @test max_abs_diff([sol.u[end]], [ref.u[end]]) < bar
    @test sol.stats.naccepted > 0
    @test sol.stats.nfevals == 6 * (sol.stats.naccepted + sol.stats.nrejected) + 2
end

@testset "Van der Pol (mu=10) agreement and rejections" begin
    saveat = collect(0.0:1.0:30.0)
    sol = solve_ode(vanderpol, VDP_U0, VDP_TSPAN, Tsit5(); p=VDP_MU,
        abstol=1e-9, reltol=1e-7, saveat=saveat)
    @test sol.retcode == :Success
    ref = ode_reference(vanderpol, VDP_U0, VDP_TSPAN, VDP_MU; saveat=saveat)
    scale = solution_scale(ref.u)
    @test max_abs_diff(sol.u, ref.u) < agreement_bar(1e-9, 1e-7, scale)

    # Relaxation spikes force step rejections at default tolerances.
    loose = solve_ode(vanderpol, VDP_U0, VDP_TSPAN, Tsit5(); p=VDP_MU)
    @test loose.retcode == :Success
    @test loose.stats.nrejected > 0
    @test loose.stats.naccepted > loose.stats.nrejected
end

@testset "Brusselator (40-state) agreement" begin
    saveat = collect(0.0:1.0:10.0)
    sol = solve_ode(brusselator, BRUS_U0, BRUS_TSPAN, Tsit5(); p=BRUS_PARAMS,
        abstol=1e-10, reltol=1e-8, saveat=saveat)
    @test sol.retcode == :Success
    @test length(sol.u[1]) == 2 * BRUS_N
    ref = ode_reference(brusselator, BRUS_U0, BRUS_TSPAN, BRUS_PARAMS;
        saveat=saveat)
    scale = solution_scale(ref.u)
    @test max_abs_diff(sol.u, ref.u) < agreement_bar(1e-10, 1e-8, scale)
end

@testset "tolerance ordering on analytic decay" begin
    exact(t) = DECAY_U0 .* exp.(-DECAY_RATES .* t)
    errs = Float64[]
    for (atol, rtol) in ((1e-9, 1e-6), (1e-12, 1e-9))
        sol = solve_ode(exponential_decay, DECAY_U0, DECAY_TSPAN, Tsit5();
            p=DECAY_RATES, abstol=atol, reltol=rtol)
        @test sol.retcode == :Success
        push!(errs, max_abs_diff(sol.u, [exact(t) for t in sol.t]))
    end
    # Fifth order buys ~1000x per 1000x tolerance; require a modest 50x.
    @test errs[2] < errs[1] / 50
    @test errs[2] < agreement_bar(1e-12, 1e-9, 1.0)
end

@testset "Float32 computes in Float32" begin
    u0 = Float32.(LOTKA_U0)
    tspan = (0.0f0, 10.0f0)
    saveat = collect(0.0f0:0.5f0:10.0f0)
    sol = solve_ode(lotka_volterra, u0, tspan, Tsit5(); p=Float32.(LOTKA_PARAMS),
        abstol=1e-6, reltol=1e-4, saveat=saveat)
    @test sol.retcode == :Success
    @test eltype(sol.t) == Float32
    @test eltype(sol.u[1]) == Float32
    ref = ode_reference(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, LOTKA_PARAMS;
        saveat=Float64.(saveat))
    scale = solution_scale(ref.u)
    @test max_abs_diff(sol.u, ref.u) < agreement_bar(1e-6, 1e-4, scale)
end

@testset "backward integration roundtrip" begin
    forward = solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, Tsit5();
        p=LOTKA_PARAMS, abstol=1e-10, reltol=1e-8)
    @test forward.retcode == :Success
    back = solve_ode(lotka_volterra, forward.u[end], (10.0, 0.0), Tsit5();
        p=LOTKA_PARAMS, abstol=1e-10, reltol=1e-8)
    @test back.retcode == :Success
    @test issorted(back.t, rev=true)
    @test max_abs_diff([back.u[end]], [LOTKA_U0]) <
          agreement_bar(1e-10, 1e-8, 3.0)
end

@testset "saveat contract" begin
    # Unsorted input is emitted sorted with exact endpoints.
    sol = solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, Tsit5();
        p=LOTKA_PARAMS, saveat=[5.0, 2.0])
    @test sol.t == [0.0, 2.0, 5.0, 10.0]
    # Endpoint requests are covered by the endpoint saves, not duplicated.
    sol2 = solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, Tsit5();
        p=LOTKA_PARAMS, saveat=[0.0, 3.0, 10.0])
    @test sol2.t == [0.0, 3.0, 10.0]
    # Empty saveat keeps endpoints only.
    sol3 = solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, Tsit5();
        p=LOTKA_PARAMS, saveat=Float64[])
    @test sol3.t == [0.0, 10.0]
    # Default saves every accepted step.
    sol4 = solve_ode(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, Tsit5();
        p=LOTKA_PARAMS)
    @test sol4.t[1] == 0.0 && sol4.t[end] == 10.0
    @test length(sol4.t) == sol4.stats.naccepted + 1
end
