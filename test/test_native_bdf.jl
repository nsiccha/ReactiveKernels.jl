using ReactiveKernels, OrdinaryDiffEqBDF, SciMLBase, SciMLSensitivity, Enzyme, Test

_bdf_decay(t, u, k) = -k .* u
_bdf_time_rhs(t, u, k) = (-k * t) .* u
_bdf_bad_rhs(t, u) = [u[1], u[1]]
function _bdf_nested_rhs(t, u, gain, cfg)
    dy = similar(u)
    for i in eachindex(u)
        dy[i] = -gain * cfg.rates[cfg.order[i]] * u[i]
    end
    dy
end
_bdf_loss(u, p, ts) = sum(rk_ode_bdf_tol(_bdf_decay, u, p[1], ts,
    1e-9, 1e-9, 10000, p[2]))
_bdf_time_loss(u, p, ts) = sum(rk_ode_bdf_tol(_bdf_time_rhs, u, p[1], ts,
    1e-9, 1e-9, 10000, p[2]))
function _bdf_nested_loss(u, rates, gain)
    cfg = (rates = rates, order = [2, 1])
    out = rk_ode_bdf_tol(_bdf_nested_rhs, u, 0.0, [0.2, 1.4],
        1e-9, 1e-9, 10000, gain, cfg)
    out[1, 1] + 2out[1, 2] + 3out[2, 1] + 4out[2, 2]
end

@testset "native standard Julia BDF bridge" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsBDFExt) !== nothing
    @testset "original call shape and independent values" begin
        for n in (1, 2, 20)
            u, ts, k = [1.4, -0.3], collect(range(0.2, 1.6; length = n + 1))[2:end], 0.7
            before = (copy(u), copy(ts))
            out = rk_ode_bdf_tol(_bdf_decay, u, 0.1, ts, 1e-9, 1e-9, 10000, k)
            @test size(out) == (n, 2)
            @test out ≈ [v * exp(-k * (t - 0.1)) for t in ts, v in u] rtol=2e-7
            @test (u, ts) == before
            @test rk_ode_bdf_tol(_bdf_decay, u, 0.1, ts, 1e-9, 1e-9, 10000, k) == out
            out[1, 1] = 999.0
            @test (u, ts) == before
        end
        rates, order, u, ts = [0.3, 1.2], [2, 1], [1.4, 0.5], [0.2, 1.4]
        before = (copy(rates), copy(order), copy(u), copy(ts))
        out = rk_ode_bdf_tol(_bdf_nested_rhs, u, 0.0, ts, 1e-9, 1e-9, 10000,
            0.8, (rates = rates, order = order))
        @test out ≈ [u[j] * exp(-0.8 * rates[order[j]] * t) for t in ts, j in 1:2] rtol=2e-7
        @test (rates, order, u, ts) == before
        stiff_times = [0.01, 0.1, 0.4]
        stiff = rk_ode_bdf_tol(_bdf_nested_rhs, [1.0, 1.0], 0.0, stiff_times,
            1e-9, 1e-11, 10000, 1.0, (rates = [0.7, 1000.0], order = [1, 2]))
        @test stiff[:, 1] ≈ exp.(-0.7 .* stiff_times) rtol=2e-7
        @test stiff[:, 2] ≈ exp.(-1000.0 .* stiff_times) atol=3e-10
    end
    @testset "ordinary Reverse, including all times" begin
        for (loss, nonautonomous) in ((_bdf_loss, false), (_bdf_time_loss, true))
            u, p, ts = [1.3], [0.15, 0.7], [0.4, 1.6]
            before = (copy(u), copy(p), copy(ts))
            du, dp, dt = zeros(1), zeros(2), zeros(2)
            Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active,
                Enzyme.Duplicated(u, du), Enzyme.Duplicated(p, dp), Enzyme.Duplicated(ts, dt))
            durations = nonautonomous ? (ts .^ 2 .- p[1]^2) ./ 2 : ts .- p[1]
            values = u[1] .* exp.(-p[2] .* durations)
            @test loss(u, p, ts) ≈ sum(values) rtol=2e-7
            @test du[1] ≈ sum(values) / u[1] rtol=2e-6
            @test dp[1] ≈ p[2] * (nonautonomous ? p[1] : 1) * sum(values) rtol=3e-6
            @test dp[2] ≈ sum(-durations .* values) rtol=3e-6
            @test dt ≈ -p[2] .* (nonautonomous ? ts : ones(2)) .* values rtol=3e-6
            @test (u, p, ts) == before
        end
        u, rates, gain = [1.4, 0.5], [0.3, 1.2], 0.8
        du, dr = zeros(2), zeros(2)
        dg = Enzyme.autodiff(Enzyme.Reverse, _bdf_nested_loss, Enzyme.Active,
            Enzyme.Duplicated(u, du), Enzyme.Duplicated(rates, dr), Enzyme.Active(gain))[1][3]
        ts, weights, order = [0.2, 1.4], [1.0 2.0; 3.0 4.0], [2, 1]
        oracle_du, oracle_dr, oracle_dg = zeros(2), zeros(2), 0.0
        for i in eachindex(ts), j in eachindex(u)
            rate = rates[order[j]]
            e = exp(-gain * rate * ts[i])
            v = weights[i, j] * u[j] * e
            oracle_du[j] += weights[i, j] * e
            oracle_dr[order[j]] -= gain * ts[i] * v
            oracle_dg -= rate * ts[i] * v
        end
        @test du ≈ oracle_du rtol=3e-6
        @test dr ≈ oracle_dr rtol=3e-6
        @test dg ≈ oracle_dg rtol=3e-6
        @test u == [1.4, 0.5]
        @test rates == [0.3, 1.2]
    end
    @testset "prepared native kernel Reverse" begin
        graph = Graph()
        q = value!(graph, :q, Vector{Float64})
        loss = value!(graph, :loss, Float64)
        add!(graph, q => loss, p -> _bdf_loss([1.3], p, [0.4, 1.6]))
        kernel = prepare(graph; have = (q,), want = loss)
        point = [0.15, 0.7]
        backend = ReactiveKernels.DifferentiationInterface.AutoEnzyme(mode = Enzyme.Reverse)
        prepared = prepare_ad(kernel, backend, point; active = :q)
        value, gradient = ad_value_and_gradient(prepared, point)
        values = 1.3 .* exp.(-point[2] .* ([0.4, 1.6] .- point[1]))
        @test value ≈ sum(values) rtol=2e-7
        @test gradient ≈ [point[2] * sum(values), sum(-([0.4, 1.6] .- point[1]) .* values)] rtol=3e-6
        @test point == [0.15, 0.7]
    end
    @testset "controls, continuous history, and failure ownership" begin
        u, p, ts = [1.0], [0.7], collect(0.02:0.02:0.4)
        f!(du, y, p, t) = (du[1] = -p[1] * y[1]; nothing)
        normalized_times = ts ./ last(ts)
        prob = ODEProblem{true,SciMLBase.FullSpecialize}(f!, copy(u), (0.0, 1.0), p .* last(ts))
        reference = solve(prob, FBDF(); tstops = normalized_times,
            reltol = 1e-8, abstol = 1e-8, save_start = true, save_everystep = true, dense = true)
        ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsBDFExt)
        wrapped = solve(prob, ext.OutputBudgetBDF(normalized_times, 10000);
            reltol = 1e-8, abstol = 1e-8, save_start = true, save_everystep = true, dense = true)
        @test wrapped.t == reference.t
        @test wrapped.u == reference.u
        @test wrapped.stats.naccept == reference.stats.naccept
        # Each interval is short enough for this budget; the entire solve needs
        # more steps. A global maxiters would reject this valid original call.
        per_output = maximum(count(t -> a < t <= b, reference.t)
            for (a, b) in zip([0.0; normalized_times[1:end-1]], normalized_times))
        @test reference.stats.naccept > per_output
        out = rk_ode_bdf_tol(_bdf_decay, u, 0.0, ts, 1e-8, 1e-8, per_output + 2, p[1])
        @test vec(out) ≈ exp.(-p[1] .* ts) rtol=2e-6
        @test_throws ErrorException rk_ode_bdf_tol(_bdf_decay, u, 0.0, [1.6], 1e-10, 1e-10, 1, p[1])
        @test_throws DimensionMismatch rk_ode_bdf_tol(_bdf_bad_rhs, u, 0.0, ts, 1e-8, 1e-8, 10000)
        @test u == [1.0]
        @test p == [0.7]
        @test ts == collect(0.02:0.02:0.4)
        @test_throws ArgumentError rk_ode_bdf_tol(_bdf_decay, Float64[], 0.0, ts, 1e-8, 1e-8, 10000, p[1])
        @test_throws ArgumentError rk_ode_bdf_tol(_bdf_decay, u, 0.0, Float64[], 1e-8, 1e-8, 10000, p[1])
        for bad in ([0.0], [0.2, 0.2], [0.2, 0.1], [Inf], [NaN])
            @test_throws DomainError rk_ode_bdf_tol(_bdf_decay, u, 0.0, bad, 1e-8, 1e-8, 10000, p[1])
        end
        # These controls are valid for standard FBDF. Compare the adapter's
        # output with the same standard solve, including loose tolerance.
        # Retain the affine RHS arithmetic order: reassociating its products
        # can change adaptive steps, even though the ODE is equivalent.
        function control_rhs!(du, y, p, t)
            du[1] = (p[2] - p[1]) * (-p[3] * y[1])
            nothing
        end
        control_times = [0.4, 1.6]
        control_stops = control_times ./ last(control_times)
        control_prob = ODEProblem{true,SciMLBase.FullSpecialize}(control_rhs!, copy(u),
            (0.0, 1.0), [0.0, last(control_times), p[1]])
        for (rt, at) in ((1.1, 1e-8), (0.0, 1e-8), (1e-8, 0.0))
            standard = solve(control_prob, FBDF(); tstops = control_stops,
                reltol = rt, abstol = at, save_start = true, save_everystep = true, dense = true)
            @test SciMLBase.successful_retcode(standard)
            actual = rk_ode_bdf_tol(_bdf_decay, u, 0.0, control_times, rt, at, 10000, p[1])
            @test vec(actual) ≈ [standard(s)[1] for s in control_stops] rtol=1e-12
        end
        @test control_times == [0.4, 1.6]
        for (rt, at, mx) in ((-1e-8, 1e-8, 100), (1e-8, -1e-8, 100), (0.0, 0.0, 100),
                (NaN, 1e-8, 100), (Inf, 1e-8, 100), (1e-8, NaN, 100),
                (1e-8, Inf, 100), (1e-8, 1e-8, 0))
            @test_throws DomainError rk_ode_bdf_tol(_bdf_decay, u, 0.0, ts, rt, at, mx, p[1])
        end
        @test_throws DomainError rk_ode_bdf_tol(_bdf_decay, [NaN], 0.0, ts, 1e-8, 1e-8, 100, p[1])
        @test_throws DomainError rk_ode_bdf_tol(_bdf_decay, u, Inf, ts, 1e-8, 1e-8, 100, p[1])
    end
end
