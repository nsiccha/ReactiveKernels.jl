module JoinedWindowTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Random
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

bind(model, data; conditioned = (:y,)) =
    bind_data(lower_rkppl(model, data; conditioned, mod = @__MODULE__), data)

# One window of a latent AR(1) series. Its recurrence starts from the boundary
# input `h_in`, the state before the window's first row, and ends at `h_out`,
# the state at its last row.
const AR1 = quote
    phi ~ Normal(0, 0.5)
    sigma ~ Exponential(1)
    tau ~ Exponential(1)
    h_in ~ Normal(0, 2)
    @scan begin
        h[1] = h_in
        for t in 2:T
            h[t] ~ Normal(phi * h[t - 1], sigma)
        end
    end
    h_out = h[T]
    states = h[2:T]
    y .~ Normal.(states, tau)
end

const Y = round.(randn(Xoshiro(3), 9); digits = 3)
ar1_rows(rows) = (; y = Y[rows], T = length(rows) + 1)

# A joined coordinate's position in the whole series' binding: the same name,
# except a later window's scan step `j`, which is the series' step `offset + j`.
function whole_position(whole_names, c, names, offset)
    name = names[c.window][c.position]
    s = string(name)
    m = match(r"^(.*)\.(\d+)$", s)
    c.window > 1 && m !== nothing && c.path != (:h_in,) &&
        return findfirst(==(Symbol(m.captures[1], ".", parse(Int, m.captures[2]) + offset)),
            whole_names)
    findfirst(==(name), whole_names)
end

@testset "two windows joined through their boundary state are the whole series" begin
    whole = bind(AR1, ar1_rows(1:9))
    built = build_kernel(whole)
    plans = [bind(AR1, ar1_rows(1:4)), bind(AR1, ar1_rows(5:9))]
    layouts = [assign_layout(p) for p in plans]
    names = [coordinate_names(l) for l in layouts]
    whole_names = coordinate_names(built.layout)
    u = collect(range(-0.6, 0.7; length = built.layout.total))
    us = [zeros(l.total) for l in layouts]
    joined = prepare_joined_windows(built, plans, us; boundary = :h_out => :h_in,
        backend = BACKEND)
    cs = joined_coordinates(joined)
    # Window 1 holds the globals, the series' start and its steps; window 2 adds
    # only its own steps: its start is window 1's end, its globals window 1's.
    @test length(cs) == built.layout.total
    @test count(c -> c.window == 2, cs) == 5
    @test all(c -> c.path != (:h_in,) || c.window == 1, cs)
    at = [whole_position(whole_names, c, names, 4) for c in cs]
    @test sort(at) == collect(1:built.layout.total)
    x = u[at]
    sampler = prepare_sampler(built, whole, u; backend = BACKEND)
    g_whole = zeros(length(u))
    v_whole, _ = sampler_value_and_gradient!(sampler, g_whole, u)
    g = zeros(length(x))
    value, _ = joined_value_and_gradient!(joined, g, x)
    @test value ≈ v_whole
    @test g ≈ g_whole[at]
    @test joined(x) ≈ v_whole
    # Window 2 starts where window 1 ends.
    points = window_points(joined, x)
    h_out = prepare_query(built, plans[1], (:h_out,))(points[1])
    h_in = findfirst(==(:h_in), names[2])
    @test points[2][h_in] == h_out
    @test joined_point(joined, points) == x
    # Against an independent recursion of the whole series.
    phi, sigma, tau, start = x[1], exp(x[2]), exp(x[3]), x[4]
    steps = vcat(x[5:8], x[9:13])
    h = accumulate((a, z) -> z, steps; init = start)
    lp = logpdf(Normal(0, 0.5), phi) + logpdf(Exponential(1), sigma) + log(sigma) +
        logpdf(Exponential(1), tau) + log(tau) + logpdf(Normal(0, 2), start) +
        sum(logpdf(Normal(phi * a, sigma), b) for (a, b) in zip([start; h[1:(end - 1)]], h)) +
        sum(logpdf(Normal(h[k], tau), Y[k]) for k in eachindex(Y))
    @test value ≈ lp
    # Three windows, the middle one a single row.
    plans3 = [bind(AR1, ar1_rows(r)) for r in (1:4, 5:5, 6:9)]
    us3 = [zeros(assign_layout(p).total) for p in plans3]
    joined3 = prepare_joined_windows(built, plans3, us3; boundary = [:h_out => :h_in],
        backend = BACKEND)
    @test length(joined_coordinates(joined3)) == built.layout.total
    names3 = [coordinate_names(assign_layout(p)) for p in plans3]
    offsets = (0, 4, 5)
    at3 = map(joined_coordinates(joined3)) do c
        whole_position(whole_names, c, names3, offsets[c.window])
    end
    g3 = zeros(length(at3))
    v3, _ = joined_value_and_gradient!(joined3, g3, u[at3])
    @test v3 ≈ v_whole
    @test g3 ≈ g_whole[at3]
end

# A deterministic recurrence with a positive state (a stock washed out by a
# rate and refilled by an input), as in a washout model: the state has no
# coordinates of its own, and the fed start passes through `log`.
const STOCK = quote
    rate ~ Exponential(1)
    tau ~ Exponential(1)
    s_in ~ LogNormal(0, 1)
    @scan begin
        s[1] = s_in
        for t in 2:T
            s[t] = s[t - 1] * exp(-rate) + inflow[t - 1]
        end
    end
    s_out = s[T]
    states = s[2:T]
    y .~ Normal.(log.(states), tau)
end

const INFLOW = round.(abs.(randn(Xoshiro(5), 8)); digits = 3)
const YS = round.(randn(Xoshiro(6), 8) .* 0.3; digits = 3)
stock_rows(rows) = (; y = YS[rows], inflow = INFLOW[rows], T = length(rows) + 1)

@testset "a positive boundary state is fed through its support's transform" begin
    whole = bind(STOCK, stock_rows(1:8))
    built = build_kernel(whole)
    plans = [bind(STOCK, stock_rows(1:3)), bind(STOCK, stock_rows(4:8))]
    us = [zeros(assign_layout(p).total) for p in plans]
    joined = prepare_joined_windows(built, plans, us; boundary = :s_out => :s_in,
        backend = BACKEND)
    whole_names = coordinate_names(built.layout)
    names1 = coordinate_names(assign_layout(plans[1]))
    @test [names1[c.position] for c in joined_coordinates(joined)] == whole_names
    @test all(c -> c.window == 1, joined_coordinates(joined))
    for x in ([0.2, -0.3, 0.4], [-0.5, 0.1, -1.2])
        sampler = prepare_sampler(built, whole, x; backend = BACKEND)
        g_whole = zeros(3)
        v_whole, _ = sampler_value_and_gradient!(sampler, g_whole, x)
        g = zeros(3)
        value, _ = joined_value_and_gradient!(joined, g, x)
        @test value ≈ v_whole
        @test g ≈ g_whole
        s_out = prepare_query(built, plans[1], (:s_out,))(window_points(joined, x)[1])
        @test window_points(joined, x)[2][findfirst(==(:s_in),
            coordinate_names(assign_layout(plans[2])))] ≈ log(s_out)
    end
end

# A deterministic recurrence per unit, its start labelled by the unit: window 2
# continues units a and c, drops b and starts d.
const UNITS = quote
    rate ~ Exponential(1)
    tau ~ Exponential(1)
    s_in[levels(unit)] .~ LogNormal.(0, 1)
    @plate for g in eachindex(y)
        @scan begin
            s[1] = s_in[unit[g]]
            for t in 2:(length(y[g]) + 1)
                s[t] = s[t - 1] * exp(-rate) + inflow[g][t - 1]
            end
        end
        s_out[g] = s[length(y[g]) + 1]
        y[g] .~ Normal.(log.(s[2:(length(y[g]) + 1)]), tau)
    end
end

const FIRST = (; unit = ["a", "b", "c"], y = [[0.1, 0.3], [-0.2], [0.4, 0.0, 0.2]],
    inflow = [[0.5, 0.2], [0.1], [0.3, 0.3, 0.6]])
const SECOND = (; unit = ["a", "c", "d"], y = [[0.5], [-0.1, 0.3], [0.2, 0.2]],
    inflow = [[0.4], [0.2, 0.1], [0.7, 0.2]])

# The joined log density written out: window 1's units start from their own
# stocks; window 2's a and c start from window 1's end stocks, d from its own.
function units_oracle(rate, tau, starts1, start_d)
    lp = logpdf(Exponential(1), rate) + log(rate) + logpdf(Exponential(1), tau) + log(tau)
    ends = Dict{String,Float64}()
    function run!(window, starts)
        for (g, unit) in enumerate(window.unit)
            s = starts[unit]
            for t in eachindex(window.y[g])
                s = s * exp(-rate) + window.inflow[g][t]
                lp += logpdf(Normal(log(s), tau), window.y[g][t])
            end
            ends[unit] = s
        end
    end
    for (unit, s) in starts1
        lp += logpdf(LogNormal(0, 1), s) + log(s)
    end
    run!(FIRST, starts1)
    lp += logpdf(LogNormal(0, 1), start_d) + log(start_d)
    run!(SECOND, Dict("a" => ends["a"], "c" => ends["c"], "d" => start_d))
    lp
end

@testset "per-unit boundary states are matched by their level label" begin
    plans = [bind(UNITS, FIRST), bind(UNITS, SECOND)]
    built = build_kernel(plans[1])
    layouts = [assign_layout(p) for p in plans]
    joined = prepare_joined_windows(built, plans, [zeros(l.total) for l in layouts];
        boundary = :s_out => :s_in, backend = BACKEND)
    cs = joined_coordinates(joined)
    @test [(c.window, c.path, c.index) for c in cs] == [(1, (:rate,), ()), (1, (:tau,), ()),
        (1, (:s_in,), ("a",)), (1, (:s_in,), ("b",)), (1, (:s_in,), ("c",)),
        (2, (:s_in,), ("d",))]
    for x in ([0.1, -0.2, 0.3, -0.4, 0.2, 0.5], [-0.6, 0.4, -0.1, 0.7, -0.3, -0.2])
        g = zeros(length(x))
        value, _ = joined_value_and_gradient!(joined, g, x)
        rate, tau = exp(x[1]), exp(x[2])
        @test value ≈ units_oracle(rate, tau, Dict("a" => exp(x[3]), "b" => exp(x[4]),
            "c" => exp(x[5])), exp(x[6]))
        for i in eachindex(x)
            e = zeros(length(x)); e[i] = 1e-6
            @test g[i] ≈ (joined(x .+ e) - joined(x .- e)) / 2e-6 rtol = 1e-5 atol = 1e-7
        end
        points = window_points(joined, x)
        ends = prepare_query(built, plans[1], (:s_out,))(points[1])
        start2 = points[2][layouts[2].entries[findfirst(e -> e.name === :s_in,
            layouts[2].entries)].offset .+ (0:2)]
        @test exp.(start2[1:2]) ≈ ends[[1, 3]]
        @test start2[3] == x[6]
    end
end

@testset "a rebound binding keeps its own level labels" begin
    plans = [bind(UNITS, FIRST), bind(UNITS, SECOND)]
    built = build_kernel(plans[1])
    sampler = prepare_sampler(built, plans[2], zeros(built.layout.total); backend = BACKEND)
    ids = coordinate_identities(sampler.layout)
    @test ids == coordinate_identities(assign_layout(plans[2]))
    @test [only(id.index) for id in ids if id.path == (:s_in,)] == ["a", "c", "d"]
end

@testset "a scan's sampled steps are one prior event of the catalogue" begin
    plan = bind(AR1, ar1_rows(1:4))
    built = build_kernel(plan)
    u = collect(range(-0.3, 0.4; length = built.layout.total))
    catalogue = density_factors(built, plan)
    scan = [e for e in catalogue if e.kind === :prior && startswith(string(e.site), "_ppl_scan_z_")]
    @test length(scan) == 1
    entry = only(e for e in built.layout.entries if e.kind === :scan)
    @test factor_coordinates(catalogue, only(scan)) == collect(entry.offset:(entry.offset + entry.size - 1))
    @test prepare_query(built, plan, select_factors(catalogue, collect(catalogue)); u0 = u)(u) ≈
        prepare_query(built, plan, :prior)(u) + prepare_query(built, plan, :log_jacobian)(u)
    phi, sigma, steps = u[1], exp(u[2]), u[5:8]
    @test prepare_query(built, plan, select_factors(catalogue, scan); u0 = u)(u) ≈
        sum(logpdf(Normal(phi * a, sigma), b) for (a, b) in zip([u[4]; steps[1:3]], steps))
end

@testset "a boundary input as data: the previous window's end value" begin
    built = build_kernel(bind(AR1, ar1_rows(1:9)))
    plans = [bind(AR1, ar1_rows(1:4)), bind(AR1, ar1_rows(5:9))]
    us = [zeros(assign_layout(p).total) for p in plans]
    joined = prepare_joined_windows(built, plans, us; boundary = :h_out => :h_in,
        backend = BACKEND)
    x = collect(range(-0.3, 0.5; length = length(joined_coordinates(joined))))
    points = window_points(joined, x)
    h_end = prepare_query(built, plans[1], (:h_out,))(points[1])
    data = merge(ar1_rows(5:9), (; h_in = h_end))
    given = bind(AR1, data; conditioned = (:y, :h_in))
    given_built = build_kernel(given)
    names = coordinate_names(given_built.layout)
    @test !(:h_in in names)
    window2 = coordinate_names(assign_layout(plans[2]))
    u = [points[2][findfirst(==(n), window2)] for n in names]
    sampler = prepare_sampler(given_built, given, u; backend = BACKEND)
    g = zeros(length(u))
    sampler_value_and_gradient!(sampler, g, u)
    # The window's own steps have the joined gradient: their density reads the
    # boundary value whether it is fed or bound.
    gj = zeros(length(x))
    joined_value_and_gradient!(joined, gj, x)
    cs = joined_coordinates(joined)
    for (i, c) in enumerate(cs)
        c.window == 2 || continue
        @test gj[i] ≈ g[findfirst(==(window2[c.position]), names)]
    end
end

@testset "authored values by name" begin
    plan = bind(AR1, ar1_rows(1:4))
    built = build_kernel(plan)
    u = collect(range(-0.2, 0.3; length = built.layout.total))
    h_out, h = prepare_query(built, plan, (:h_out, :h))(u)
    @test h_out == h[end]
    @test h[1] == u[findfirst(==(:h_in), coordinate_names(built.layout))]
    @test_throws ContractValidationError prepare_query(built, plan, ())
    @test_throws ContractValidationError prepare_query(built, plan, (:likelihood,))
    # One name is its value itself, as one WANT of `ReactiveKernels.prepare` is.
    @test prepare_query(built, plan, (:h_out,))(u) == h_out
    j = prepare_query_ad(prepare_ad_pullback, built, plan, (:h_out,), BACKEND, 1.0, u)
    value, cotangent = ad_value_and_pullback(j, 1.0, u)
    @test value == h_out
    e = zeros(length(u)); e[end] = 1e-6
    fd = (prepare_query(built, plan, (:h_out,))(u .+ e) -
        prepare_query(built, plan, (:h_out,))(u .- e)) / 2e-6
    @test cotangent[end] ≈ fd rtol = 1e-6
end

@testset "boundary refusals" begin
    plans = [bind(AR1, ar1_rows(1:4)), bind(AR1, ar1_rows(5:9))]
    built = build_kernel(plans[1])
    us = [zeros(assign_layout(p).total) for p in plans]
    @test_throws ContractValidationError prepare_joined_windows(built, plans, us;
        boundary = :h_out => :nope, backend = BACKEND)
    @test_throws ContractValidationError prepare_joined_windows(built, plans, us;
        boundary = :nope => :h_in, backend = BACKEND)
    # The whole trajectory is not one end state per start.
    @test_throws ContractValidationError prepare_joined_windows(built, plans, us;
        boundary = :h => :h_in, backend = BACKEND)
    @test_throws ContractValidationError prepare_joined_windows(built, plans, us[1:1];
        boundary = :h_out => :h_in, backend = BACKEND)
    simplex = quote
        w ~ Dirichlet([1.0, 1.0, 1.0])
        tau ~ Exponential(1)
        out = w
        y .~ Normal.(w[1], tau)
    end
    sp = [bind(simplex, (; y = [0.1, 0.2])), bind(simplex, (; y = [0.3]))]
    sb = build_kernel(sp[1])
    @test_throws ContractValidationError prepare_joined_windows(sb, sp,
        [zeros(assign_layout(p).total) for p in sp]; boundary = :out => :w, backend = BACKEND)
end

@testset "joined windows as a LogDensityProblems target" begin
    whole = bind(AR1, ar1_rows(1:9))
    built = build_kernel(whole)
    plans = [bind(AR1, ar1_rows(1:4)), bind(AR1, ar1_rows(5:9))]
    joined = prepare_joined_windows(built, plans, [zeros(assign_layout(p).total) for p in plans];
        boundary = :h_out => :h_in, backend = BACKEND)
    problem = joined_problem(joined)
    n = length(joined_coordinates(joined))
    @test LogDensityProblems.dimension(problem) == n
    @test LogDensityProblems.capabilities(typeof(problem)) == LogDensityProblems.LogDensityOrder{1}()
    x = collect(range(-0.1, 0.2; length = n))
    v, g = LogDensityProblems.logdensity_and_gradient(problem, x)
    @test LogDensityProblems.logdensity(problem, x) ≈ v
    @test g ≈ last(joined_value_and_gradient!(joined, zeros(n), x))
end

end
