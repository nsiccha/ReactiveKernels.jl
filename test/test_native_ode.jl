using ReactiveKernels, OrdinaryDiffEqBDF, OrdinaryDiffEqLowOrderRK, SciMLBase,
    SciMLSensitivity, Enzyme, Test

# Each named entry with its standard Julia algorithm and owning extension. The
# shared adapter must behave identically through every entry.
const _ODE_ENTRIES = (
    (rk_ode_bdf_tol, FBDF(), :ReactiveKernelsBDFExt),
    (rk_ode_rk45_tol, DP5(), :ReactiveKernelsLowOrderRKExt))

_ode_decay(t, u, k) = -k .* u
_ode_time_rhs(t, u, k) = (-k * t) .* u
_ode_bad_rhs(t, u) = [u[1], u[1]]
_ode_oscillator(t, u, w) = [u[2], -w^2 * u[1]]
function _ode_nested_rhs(t, u, gain, cfg)
    dy = similar(u)
    for i in eachindex(u)
        dy[i] = -gain * cfg.rates[cfg.order[i]] * u[i]
    end
    dy
end
_ode_loss(solve, u, p, ts) = sum(solve(_ode_decay, u, p[1], ts,
    1e-9, 1e-9, 10000, p[2]))
_ode_time_loss(solve, u, p, ts) = sum(solve(_ode_time_rhs, u, p[1], ts,
    1e-9, 1e-9, 10000, p[2]))
# Two-compartment amounts with gamma-shaped (shape 3) input pulses at dose times.
function _ode_two_compartment(t, y, lp, dose_times, amounts)
    cl, v1, q, v2, ktr = exp(lp[1]), exp(lp[2]), exp(lp[3]), exp(lp[4]), exp(lp[5])
    input = zero(t * ktr)
    for j in eachindex(dose_times)
        s = t - dose_times[j]
        if s > 0
            input += amounts[j] * ktr^3 * s^2 * exp(-ktr * s) / 2
        end
    end
    k10, k12, k21 = cl / v1, q / v1, q / v2
    [input - (k10 + k12) * y[1] + k21 * y[2], k12 * y[1] - k21 * y[2]]
end
_ode_two_compartment_loss(solve, lp, rt, at) = sum(log.(solve(_ode_two_compartment,
    [0.0, 0.0], 0.0, [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.5, 13.0, 14.0, 16.0, 20.0, 24.0],
    rt, at, 10000, lp, [0.0, 12.0], [100.0, 50.0])[:, 1]))
function _ode_nested_loss(solve, u, rates, gain)
    cfg = (rates = rates, order = [2, 1])
    out = solve(_ode_nested_rhs, u, 0.0, [0.2, 1.4],
        1e-9, 1e-9, 10000, gain, cfg)
    out[1, 1] + 2out[1, 2] + 3out[2, 1] + 4out[2, 2]
end

# Method instances that concretely typed calls to `names` invoke in the typed
# IR of `f(types...)`, and how many of those calls dispatch dynamically instead.
_ode_target(target::Core.MethodInstance) = target
_ode_target(target) = _ode_target(target.def)
function _ode_calls(f, types, names...)
    code = first(only(code_typed(f, types))).code
    invoked = [_ode_target(stmt.args[1]).specTypes for stmt in code
        if Meta.isexpr(stmt, :invoke) && _ode_target(stmt.args[1]).def.name in names]
    dynamic = count(stmt -> Meta.isexpr(stmt, :call) &&
        stmt.args[1] isa GlobalRef && stmt.args[1].name in names, code)
    invoked, dynamic
end

@testset "native standard Julia ODE bridge: $(nameof(solve)) ($(nameof(typeof(alg))))" for
        (solve, alg, extension) in _ODE_ENTRIES
    @test Base.get_extension(ReactiveKernels, extension) !== nothing
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsSciMLODEExt) !== nothing
    @testset "original call shape and independent values" begin
        for n in (1, 2, 20)
            u, ts, k = [1.4, -0.3], collect(range(0.2, 1.6; length = n + 1))[2:end], 0.7
            before = (copy(u), copy(ts))
            out = solve(_ode_decay, u, 0.1, ts, 1e-9, 1e-9, 10000, k)
            @test size(out) == (n, 2)
            @test out ≈ [v * exp(-k * (t - 0.1)) for t in ts, v in u] rtol=2e-7
            @test (u, ts) == before
            @test solve(_ode_decay, u, 0.1, ts, 1e-9, 1e-9, 10000, k) == out
            out[1, 1] = 999.0
            @test (u, ts) == before
        end
        rates, order, u, ts = [0.3, 1.2], [2, 1], [1.4, 0.5], [0.2, 1.4]
        before = (copy(rates), copy(order), copy(u), copy(ts))
        out = solve(_ode_nested_rhs, u, 0.0, ts, 1e-9, 1e-9, 10000,
            0.8, (rates = rates, order = order))
        @test out ≈ [u[j] * exp(-0.8 * rates[order[j]] * t) for t in ts, j in 1:2] rtol=2e-7
        @test (rates, order, u, ts) == before
        # A nonstiff oscillation over several periods.
        w, ts = 3.0, collect(range(0.5, 10.0; length = 12))
        out = solve(_ode_oscillator, [1.0, 0.0], 0.0, ts, 1e-10, 1e-10, 100000, w)
        @test out[:, 1] ≈ cos.(w .* ts) atol=2e-7
        @test out[:, 2] ≈ -w .* sin.(w .* ts) atol=6e-7
        if alg isa FBDF
            stiff_times = [0.01, 0.1, 0.4]
            stiff = solve(_ode_nested_rhs, [1.0, 1.0], 0.0, stiff_times,
                1e-9, 1e-11, 10000, 1.0, (rates = [0.7, 1000.0], order = [1, 2]))
            @test stiff[:, 1] ≈ exp.(-0.7 .* stiff_times) rtol=2e-7
            @test stiff[:, 2] ≈ exp.(-1000.0 .* stiff_times) atol=3e-10
        end
    end
    @testset "ordinary Reverse, including all times" begin
        for (loss, nonautonomous) in ((_ode_loss, false), (_ode_time_loss, true))
            u, p, ts = [1.3], [0.15, 0.7], [0.4, 1.6]
            before = (copy(u), copy(p), copy(ts))
            du, dp, dt = zeros(1), zeros(2), zeros(2)
            Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Const(solve),
                Enzyme.Duplicated(u, du), Enzyme.Duplicated(p, dp), Enzyme.Duplicated(ts, dt))
            durations = nonautonomous ? (ts .^ 2 .- p[1]^2) ./ 2 : ts .- p[1]
            values = u[1] .* exp.(-p[2] .* durations)
            @test loss(solve, u, p, ts) ≈ sum(values) rtol=2e-7
            @test du[1] ≈ sum(values) / u[1] rtol=2e-6
            @test dp[1] ≈ p[2] * (nonautonomous ? p[1] : 1) * sum(values) rtol=3e-6
            @test dp[2] ≈ sum(-durations .* values) rtol=3e-6
            @test dt ≈ -p[2] .* (nonautonomous ? ts : ones(2)) .* values rtol=3e-6
            @test (u, p, ts) == before
        end
        u, rates, gain = [1.4, 0.5], [0.3, 1.2], 0.8
        du, dr = zeros(2), zeros(2)
        dg = Enzyme.autodiff(Enzyme.Reverse, _ode_nested_loss, Enzyme.Active,
            Enzyme.Const(solve), Enzyme.Duplicated(u, du), Enzyme.Duplicated(rates, dr),
            Enzyme.Active(gain))[1][4]
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
    @testset "parameter gradients follow the caller tolerance" begin
        # Independent reference: fourth-order central differences of a tight primal.
        lp = log.([5.0, 20.0, 10.0, 50.0, 1.5])
        h = 1e-4
        reference = map(eachindex(lp)) do j
            e = zeros(length(lp))
            e[j] = h
            f(x) = _ode_two_compartment_loss(solve, x, 1e-12, 1e-12)
            (-f(lp .+ 2e) + 8f(lp .+ e) - 8f(lp .- e) + f(lp .- 2e)) / (12h)
        end
        for tol in (1e-6, 1e-8)
            gradient = zeros(length(lp))
            Enzyme.autodiff(Enzyme.Reverse, _ode_two_compartment_loss, Enzyme.Active,
                Enzyme.Const(solve), Enzyme.Duplicated(copy(lp), gradient),
                Enzyme.Const(tol), Enzyme.Const(tol))
            relative = sqrt(sum(abs2, gradient .- reference) / sum(abs2, reference))
            @test relative < 20tol
        end
    end
    @testset "statically dispatched bridge" begin
        # Enzyme differentiates a boxed `jl_invoke` of a despecialized method,
        # or a dynamic call, through a separate runtime derivative compile at
        # the first gradient evaluation (snag nested-enzyme-co-3b60d53e).
        # The one-line entry may inline, leaving the adapter call in its place.
        invoked, dynamic = _ode_calls(_ode_nested_loss,
            (typeof(solve), Vector{Float64}, Vector{Float64}, Float64),
            nameof(solve), :_native_ode_tol)
        @test dynamic == 0
        @test !isempty(invoked) && all(Base.isdispatchtuple, invoked)
        arguments = (typeof(_ode_nested_rhs), Vector{Float64}, Float64, Vector{Float64},
            Float64, Float64, Int, Float64, @NamedTuple{rates::Vector{Float64}, order::Vector{Int}})
        invoked, dynamic = _ode_calls(solve, arguments, :_native_ode_tol)
        @test dynamic == 0
        @test !isempty(invoked) && all(Base.isdispatchtuple, invoked)
        invoked, dynamic = _ode_calls(ReactiveKernels._native_ode_tol,
            (typeof(alg), arguments...), :_sample)
        @test dynamic == 0
        @test all(Base.isdispatchtuple, invoked)
    end
    @testset "prepared native kernel Reverse" begin
        graph = Graph()
        q = value!(graph, :q, Vector{Float64})
        loss = value!(graph, :loss, Float64)
        add!(graph, q => loss, p -> _ode_loss(solve, [1.3], p, [0.4, 1.6]))
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
        # BDF ends a step at every output; DP5 interpolates its outputs.
        stops(times) = alg isa FBDF ? times : Float64[]
        prob = ODEProblem{true,SciMLBase.FullSpecialize}(f!, copy(u), (0.0, 1.0), p .* last(ts))
        reference = SciMLBase.solve(prob, alg; tstops = stops(normalized_times),
            reltol = 1e-8, abstol = 1e-8, save_start = true, save_everystep = true, dense = true)
        ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsSciMLODEExt)
        wrapped = SciMLBase.solve(prob, ext.OutputBudget(alg, normalized_times, 10000);
            reltol = 1e-8, abstol = 1e-8, save_start = true, save_everystep = true, dense = true)
        @test wrapped.t == reference.t
        @test wrapped.u == reference.u
        @test wrapped.stats.naccept == reference.stats.naccept
        # Each interval is short enough for this budget; the entire solve needs
        # more steps. A global maxiters would reject this valid original call.
        per_output = maximum(count(t -> a < t <= b, reference.t)
            for (a, b) in zip([0.0; normalized_times[1:end-1]], normalized_times))
        @test reference.stats.naccept > per_output
        out = solve(_ode_decay, u, 0.0, ts, 1e-8, 1e-8, per_output + 2, p[1])
        @test vec(out) ≈ exp.(-p[1] .* ts) rtol=2e-6
        @test_throws ErrorException solve(_ode_decay, u, 0.0, [1.6], 1e-10, 1e-10, 1, p[1])
        @test_throws DimensionMismatch solve(_ode_bad_rhs, u, 0.0, ts, 1e-8, 1e-8, 10000)
        @test u == [1.0]
        @test p == [0.7]
        @test ts == collect(0.02:0.02:0.4)
        @test_throws ArgumentError solve(_ode_decay, Float64[], 0.0, ts, 1e-8, 1e-8, 10000, p[1])
        @test_throws ArgumentError solve(_ode_decay, u, 0.0, Float64[], 1e-8, 1e-8, 10000, p[1])
        for bad in ([0.0], [0.2, 0.2], [0.2, 0.1], [Inf], [NaN])
            @test_throws DomainError solve(_ode_decay, u, 0.0, bad, 1e-8, 1e-8, 10000, p[1])
        end
        # These controls are valid for the standard algorithm. Compare the
        # adapter's output with the same standard solve, including loose
        # tolerance. Retain the affine RHS arithmetic order: reassociating its
        # products can change adaptive steps, even though the ODE is equivalent.
        function control_rhs!(du, y, p, t)
            du[1] = (p[2] - p[1]) * (-p[3] * y[1])
            nothing
        end
        control_times = [0.4, 1.6]
        control_stops = control_times ./ last(control_times)
        control_prob = ODEProblem{true,SciMLBase.FullSpecialize}(control_rhs!, copy(u),
            (0.0, 1.0), [0.0, last(control_times), p[1]])
        for (rt, at) in ((1.1, 1e-8), (0.0, 1e-8), (1e-8, 0.0))
            standard = SciMLBase.solve(control_prob, alg; tstops = stops(control_stops),
                reltol = rt, abstol = at, save_start = true, save_everystep = true, dense = true)
            @test SciMLBase.successful_retcode(standard)
            actual = solve(_ode_decay, u, 0.0, control_times, rt, at, 10000, p[1])
            @test vec(actual) ≈ [standard(s)[1] for s in control_stops] rtol=1e-12
        end
        @test control_times == [0.4, 1.6]
        for (rt, at, mx) in ((-1e-8, 1e-8, 100), (1e-8, -1e-8, 100), (0.0, 0.0, 100),
                (NaN, 1e-8, 100), (Inf, 1e-8, 100), (1e-8, NaN, 100),
                (1e-8, Inf, 100), (1e-8, 1e-8, 0))
            @test_throws DomainError solve(_ode_decay, u, 0.0, ts, rt, at, mx, p[1])
        end
        @test_throws DomainError solve(_ode_decay, [NaN], 0.0, ts, 1e-8, 1e-8, 100, p[1])
        @test_throws DomainError solve(_ode_decay, u, Inf, ts, 1e-8, 1e-8, 100, p[1])
    end
end
