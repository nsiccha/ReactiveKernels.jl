using ReactiveKernels
using ReactiveKernelsPPL
using Distributions
using DifferentiationInterface
using Enzyme
using Test

# A `@scan` step may read any earlier entry of a carried array: an index that
# provably lies in `1:(t - 1)`, such as `a[s]` or `a[t - s]` under a generator
# `for s in 1:(t - 1)`, or a slice `a[(t - k):(t - 1)]`. Renewal equations and
# other full-history recurrences are stated as written, and lower to RK scans
# whose step reads the scan's own `history`.

function _hist_sampler(program, data)
    snapshot = deepcopy(data)
    plan = bind_data(lower_rkppl(program, Tuple(keys(data));
        conditioned = (:y,)), data)
    built = build_kernel(plan)
    q = prepare_sampler(built, plan, zeros(built.layout.total);
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    @test data == snapshot
    return (; plan, built, q)
end

# Value against the independent oracle at several points, and the native
# Enzyme Reverse gradient against central finite differences of the oracle.
function _hist_check(f, oracle; points = 3)
    n = f.built.layout.total
    for j in 1:points
        u = [0.15 * sin(3j + i) for i in 1:n]
        g = similar(u)
        value, _ = sampler_value_and_gradient!(f.q, g, u)
        @test value ≈ oracle(u) rtol = 1e-11
        h = cbrt(eps(Float64))
        fd = map(1:n) do i
            up, dn = copy(u), copy(u)
            up[i] += h
            dn[i] -= h
            (oracle(up) - oracle(dn)) / (2h)
        end
        @test g ≈ fd rtol = 1e-5 atol = 1e-7
    end
end

_hist_scans(spec) = count(e -> e.kind === :scan, recipe_inventory(spec))
_hist_structure(spec) = [(e.kind, e.depth) for e in recipe_inventory(spec)
    if e.kind !== :ordinary]

const _HIST_T = 14
const _HIST_SI = [exp(-k / 3) / 3 for k in 1:_HIST_T]
const _HIST_F = [exp(-k / 2) / 2 for k in 1:_HIST_T]
const _HIST_X = [0.1 * sin(t + j) for t in 1:_HIST_T, j in 1:2]
const _HIST_Y = [0.3 * sin(t) + 1 for t in 1:_HIST_T]

# A renewal equation with a susceptible-depletion factor, as in the Imperial
# College COVID-19 model: seeded infections, then each day's infections are a
# time-varying reproduction number times the convolution of all earlier
# infections with the serial interval, and expected deaths are the
# convolution of all earlier infections with the onset-to-death delay.
function _hist_renewal_program()
    quote
        mu ~ LogNormal(0, 0.3)
        a1 ~ Normal(0, 0.5)
        a2 ~ Normal(0, 0.5)
        i0 ~ Exponential(1)
        ifr ~ LogNormal(0, 0.2)
        @scan begin
            infections[1] = i0
            infections[2] = i0
            infections[3] = i0
            cumulative[1] = 0.0
            cumulative[2] = i0
            cumulative[3] = 2 * i0
            for t in 4:T
                cumulative[t] = cumulative[t - 1] + infections[t - 1]
                rt = mu * exp(-(X[t, 1] * a1 + X[t, 2] * a2))
                convolution = sum(infections[s] * SI[t - s] for s in 1:(t - 1))
                infections[t] = (1 - cumulative[t] / pop) * rt * convolution
            end
        end
        @scan begin
            deaths[1] = 1e-3 * infections[1]
            for t in 2:T
                deaths[t] = ifr * sum(infections[s] * f[t - s] for s in 1:(t - 1))
            end
        end
        y .~ Normal.(deaths, 0.5)
    end
end

function _hist_renewal_oracle(T, pop)
    return function (u)
        mu, a1, a2, i0, ifr = exp(u[1]), u[2], u[3], exp(u[4]), exp(u[5])
        lp = logpdf(LogNormal(0, 0.3), mu) + u[1] + logpdf(Normal(0, 0.5), a1) +
            logpdf(Normal(0, 0.5), a2) + logpdf(Exponential(1), i0) + u[4] +
            logpdf(LogNormal(0, 0.2), ifr) + u[5]
        infections = fill(i0, T)
        cumulative = [0.0, i0, 2i0, zeros(T - 3)...]
        for t in 4:T
            cumulative[t] = cumulative[t - 1] + infections[t - 1]
            rt = mu * exp(-(_HIST_X[t, 1] * a1 + _HIST_X[t, 2] * a2))
            convolution = 0.0
            for s in 1:(t - 1)
                convolution += infections[s] * _HIST_SI[t - s]
            end
            infections[t] = (1 - cumulative[t] / pop) * rt * convolution
        end
        deaths = zeros(T)
        deaths[1] = 1e-3 * infections[1]
        for t in 2:T, s in 1:(t - 1)
            deaths[t] += ifr * infections[s] * _HIST_F[t - s]
        end
        return lp + sum(logpdf.(Normal.(deaths, 0.5), _HIST_Y[1:T]))
    end
end

@testset "scan history: front end" begin
    capable(blk) = @test parse_scan_block(blk) isa ScanSpec
    capable(quote
        p[1] = 1.0
        for t in 2:T
            p[t] = r * sum(p[s] * w[t - s] for s in 1:(t - 1))
        end
    end)
    capable(quote                       # the other orientation
        p[1] = 1.0
        for t in 2:T
            p[t] = r * sum(w[s] * p[t - s] for s in 1:(t - 1))
        end
    end)
    capable(quote                       # a window and a literal seed position
        p[1] = 1.0
        p[2] = 1.0
        p[3] = 1.0
        for t in 4:T
            p[t] = -sum(p[(t - 3):(t - 1)]) + p[1]
        end
    end)
    capable(quote                       # a nested generator and a filter
        p[1] = 1.0
        for t in 2:T
            p[t] = sum(p[r] for s in 1:(t - 1) for r in 1:s if isodd(r))
        end
    end)
    sp = parse_scan_block(quote
        p[1] = 1.0
        q[1] = 0.0
        for t in 2:T
            q[t] = q[t - 1] + p[t - 1]
            p[t] = sum(p[s] for s in 1:(t - 1)) - q[t]
        end
    end)
    @test sp.maxlag == 1
    @test ReactiveKernelsPPL._scan_has_history(sp)

    reject(blk) = @test_throws SurfaceLoweringError parse_scan_block(blk)
    # refused: `p[t]` is read before the step writes it (P3)
    reject(quote
        p[1] = 1.0
        for t in 2:T
            p[t] = sum(p[s] for s in 1:t)
        end
    end)
    # refused: `p[0]` is out of bounds at every step (Julia BoundsError, P3)
    reject(quote
        p[1] = 1.0
        for t in 2:T
            p[t] = sum(p[s] for s in 0:(t - 1))
        end
    end)
    # refused: `p[t + 1]` is not written yet (P3)
    reject(quote
        p[1] = 1.0
        for t in 2:T
            p[t] = sum(p[s + 2] for s in 1:(t - 1))
        end
    end)
    # refused: `p[t - 3]` reads `p[-1]` at t = 2 (Julia BoundsError, P3)
    reject(quote
        p[1] = 1.0
        for t in 2:T
            p[t] = p[(t - 3):(t - 1)][1]
        end
    end)
    # refused: a generator variable rebinding the loop index or a carried
    # array would change what the step's own names mean (P3)
    reject(quote
        p[1] = 1.0
        for t in 2:T
            p[t] = sum(p[t] for t in 1:3)
        end
    end)
    reject(quote
        p[1] = 1.0
        for t in 2:T
            p[t] = sum(w[p] for p in 1:(t - 1))
        end
    end)
end

@testset "scan history: renewal equation and delay convolution" begin
    for T in (6, _HIST_T)
        data = Dict{Symbol,Any}(:y => _HIST_Y[1:T], :T => T, :SI => _HIST_SI,
            :f => _HIST_F, :X => _HIST_X, :pop => 80.0)
        f = _hist_sampler(_hist_renewal_program(), data)
        @test f.built.layout.total == 5
        _hist_check(f, _hist_renewal_oracle(T, 80.0))
    end
    # The recurrence and the convolution stay RK scans; the trajectory
    # length changes no structure (the step reads its own history).
    shapes = map((6, _HIST_T)) do T
        data = Dict{Symbol,Any}(:y => _HIST_Y[1:T], :T => T, :SI => _HIST_SI,
            :f => _HIST_F, :X => _HIST_X, :pop => 80.0)
        plan = bind_data(lower_rkppl(_hist_renewal_program(), Tuple(keys(data));
            conditioned = (:y,)), data)
        built = build_kernel(plan)
        @test _hist_scans(built.spec) == 3      # infections, cumulative, deaths
        code = string(kernel_expr(plan, built.layout))
        @test occursin("history = [", code)
        _hist_structure(built.spec)
    end
    @test allequal(shapes)
end

@testset "scan history: orientations, windows, innovations and centered scans" begin
    T = 10
    y = _HIST_Y[1:T]
    w = _HIST_SI
    # The convolution written over the lag, `p[t - s]`.
    f = _hist_sampler(quote
        r ~ LogNormal(0, 0.3)
        i0 ~ Exponential(1)
        @scan begin
            p[1] = i0
            for t in 2:T
                p[t] = r * sum(w[s] * p[t - s] for s in 1:(t - 1))
            end
        end
        y .~ Normal.(p, 1.0)
    end, Dict{Symbol,Any}(:y => y, :T => T, :w => w))
    _hist_check(f, function (u)
        r, i0 = exp(u[1]), exp(u[2])
        p = fill(i0, T)
        for t in 2:T
            p[t] = r * sum(w[s] * p[t - s] for s in 1:(t - 1))
        end
        logpdf(LogNormal(0, 0.3), r) + u[1] + logpdf(Exponential(1), i0) + u[2] +
            sum(logpdf.(Normal.(p, 1.0), y))
    end)

    # A window over the last three entries, with innovations (the seasonal
    # sum of a structural time-series model).
    f = _hist_sampler(quote
        s1 ~ LogNormal(0, 0.3)
        @scan begin
            h[1] ~ Normal(0, 1)
            h[2] ~ Normal(0, 1)
            h[3] ~ Normal(0, 1)
            for t in 4:T
                e ~ Normal(0, s1)
                h[t] = -sum(h[(t - 3):(t - 1)]) + e
            end
        end
        y .~ Normal.(h, 1.0)
    end, Dict{Symbol,Any}(:y => y, :T => T))
    _hist_check(f, function (u)
        s1, z = exp(u[1]), u[2:end]
        lp = logpdf(LogNormal(0, 0.3), s1) + u[1] + sum(logpdf.(Normal(), z[1:3]))
        h = [z[1:3]; zeros(T - 3)]
        for t in 4:T
            lp += logpdf(Normal(0, s1), z[t])
            h[t] = -sum(h[(t - 3):(t - 1)]) + z[t]
        end
        lp + sum(logpdf.(Normal.(h, 1.0), y))
    end)

    # An innovation scale reading the whole history.
    f = _hist_sampler(quote
        phi ~ Normal(0, 0.5)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                e ~ Normal(0, exp(0.1 * sum(h[s] for s in 1:(t - 1)) / (t - 1)))
                h[t] = phi * h[t - 1] + e
            end
        end
        y .~ Normal.(h, 1.0)
    end, Dict{Symbol,Any}(:y => y, :T => T))
    _hist_check(f, function (u)
        phi, z = u[1], u[2:end]
        lp = logpdf(Normal(0, 0.5), phi) + logpdf(Normal(), z[1])
        h = [z[1]; zeros(T - 1)]
        for t in 2:T
            lp += logpdf(Normal(0, exp(0.1 * sum(h[1:(t - 1)]) / (t - 1))), z[t])
            h[t] = phi * h[t - 1] + z[t]
        end
        lp + sum(logpdf.(Normal.(h, 1.0), y))
    end)

    # A centered scan (the state is sampled directly) reading its history.
    f = _hist_sampler(quote
        sigma ~ LogNormal(0, 0.3)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                h[t] ~ Normal(0.5 * sum(h[s] for s in 1:(t - 1)) / (t - 1), sigma)
            end
        end
        y .~ Normal.(h, 1.0)
    end, Dict{Symbol,Any}(:y => y, :T => T))
    _hist_check(f, function (u)
        sigma, h = exp(u[1]), u[2:end]
        lp = logpdf(LogNormal(0, 0.3), sigma) + u[1] + logpdf(Normal(), h[1])
        for t in 2:T
            lp += logpdf(Normal(0.5 * sum(h[1:(t - 1)]) / (t - 1), sigma), h[t])
        end
        lp + sum(logpdf.(Normal.(h, 1.0), y))
    end)

    # One carried array reads another's history in the same scan, beside a
    # bound-integer lag.
    f = _hist_sampler(quote
        r ~ LogNormal(0, 0.3)
        i0 ~ Exponential(1)
        @scan begin
            p[1] = i0
            p[2] = i0
            E[1] = 0.0
            E[2] = 0.0
            for t in 3:T
                p[t] = r * p[t - k]
                E[t] = sum(p[s] * w[t - s] for s in 1:(t - 1))
            end
        end
        y .~ Normal.(E, 1.0)
    end, Dict{Symbol,Any}(:y => y, :T => T, :w => w, :k => 2))
    _hist_check(f, function (u)
        r, i0 = exp(u[1]), exp(u[2])
        p = fill(i0, T)
        E = zeros(T)
        for t in 3:T
            p[t] = r * p[t - 2]
            E[t] = sum(p[s] * w[t - s] for s in 1:(t - 1))
        end
        logpdf(LogNormal(0, 0.3), r) + u[1] + logpdf(Exponential(1), i0) + u[2] +
            sum(logpdf.(Normal.(E, 1.0), y))
    end)
end

@testset "scan history: constant seeds" begin
    # A literal seed shares the trajectory vector with the steps (RK scan
    # `history = [seeds…]`), so native Enzyme reverse needs no runtime
    # activity.
    T = 9
    y = _HIST_Y[1:T]
    f = _hist_sampler(quote
        r ~ LogNormal(0, 0.3)
        @scan begin
            p[1] = 1.0
            p[2] = 0.5
            for t in 3:T
                p[t] = r * sum(p[s] for s in 1:(t - 1)) / t
            end
        end
        y .~ Normal.(p, 1.0)
    end, Dict{Symbol,Any}(:y => y, :T => T))
    _hist_check(f, function (u)
        r = exp(u[1])
        p = [1.0, 0.5, zeros(T - 2)...]
        for t in 3:T
            p[t] = r * sum(p[1:(t - 1)]) / t
        end
        logpdf(LogNormal(0, 0.3), r) + u[1] + sum(logpdf.(Normal.(p, 1.0), y))
    end)
end

# A `@scan` in a `@plate` cell runs one recurrence per index: an RK scan
# retained inside the plate cell.
_hist_window_columns(deaths, first_day, last_day) =
    [deaths[first_day[m]:last_day[m], m] for m in eachindex(first_day)]

@testset "scan history: a recurrence per plate index" begin
    T, M = 10, 3
    w = _HIST_SI
    y = [[0.3 * sin(t + m) + 1 for t in 1:T] for m in 1:M]
    f = _hist_sampler(quote
        r[1:M] .~ LogNormal.(0, 0.3)
        i0[1:M] .~ Exponential.(1)
        @plate for m in eachindex(y)
            @scan begin
                p[1] = i0[m]
                p[2] = 1.0
                for t in 3:T
                    p[t] = r[m] * sum(p[s] * w[t - s] for s in 1:(t - 1))
                end
            end
            loc[m] = p
            y[m] .~ Normal.(loc[m], 1.0)
        end
    end, Dict{Symbol,Any}(:y => y, :T => T, :w => w, :M => M))
    _hist_check(f, function (u)
        r, i0 = exp.(u[1:M]), exp.(u[(M + 1):(2M)])
        lp = sum(logpdf.(LogNormal(0, 0.3), r)) + sum(u[1:M]) +
            sum(logpdf.(Exponential(1), i0)) + sum(u[(M + 1):(2M)])
        for m in 1:M
            p = [i0[m], 1.0, zeros(T - 2)...]
            for t in 3:T
                p[t] = r[m] * sum(p[s] * w[t - s] for s in 1:(t - 1))
            end
            lp += sum(logpdf.(Normal.(p, 1.0), y[m]))
        end
        lp
    end)
    @test count(e -> e.kind === :scan, recipe_inventory(f.built.spec)) == 1
end

# The Imperial College COVID-19 model (PosteriorDB `covid19imperial_v2/v3`)
# in its natural spelling: per country, seeded infections, a renewal equation
# with susceptible depletion and an intervention-dependent reproduction
# number, and negative-binomial deaths from the delay convolution over each
# country's observation window, cut from the deaths matrix in the model.
function _hist_imperial_program()
    quote
        tau ~ Exponential(1 / 0.03)
        phi ~ truncated(Normal(0, 5), 0, Inf)
        kappa ~ truncated(Normal(0, 0.5), 0, Inf)
        alpha_hier[1:P] .~ Gamma.(0.1667, 1)
        mu[1:M] .~ truncated.(Normal.(3.28, kappa), 0, Inf)
        y0[1:M] .~ Exponential.(tau)
        ifr_noise[1:M] .~ truncated.(Normal.(1, 0.1), 0, Inf)
        alpha = alpha_hier .- log(1.05) / 6
        observed = _hist_window_columns(deaths, epidemic_start, last_day)
        @plate for m in eachindex(observed)
            @scan begin
                prediction[1] = y0[m]
                prediction[2] = y0[m]
                prediction[3] = y0[m]
                cumulative[1] = 0.0
                cumulative[2] = y0[m]
                cumulative[3] = 2 * y0[m]
                for t in 4:N2
                    cumulative[t] = cumulative[t - 1] + prediction[t - 1]
                    rt = mu[m] * exp(-sum(X[m, t, j] * alpha[j] for j in 1:P))
                    convolution = sum(prediction[s] * SI[t - s] for s in 1:(t - 1))
                    prediction[t] = (pop[m] - cumulative[t]) / pop[m] * rt * convolution
                end
            end
            @scan begin
                expected[1] = 1e-15 * prediction[1]
                for t in 2:N2
                    expected[t] = ifr_noise[m] *
                        sum(prediction[s] * f[t - s, m] for s in 1:(t - 1))
                end
            end
            window[m] = expected[epidemic_start[m]:last_day[m]]
            observed[m] .~ NegativeBinomial.(phi, phi ./ (phi .+ window[m]))
        end
    end
end

function _hist_imperial_data(M, N2, P)
    X = [t > 8 + 3j + m ? 1.0 : 0.0 for m in 1:M, t in 1:N2, j in 1:P]
    SI = [exp(-((k - 6.5) / 3)^2) for k in 1:N2]
    f = [exp(-((k - 12 + m) / 5)^2) / 50 for k in 1:N2, m in 1:M]
    deaths = [(3t + 7m) % 23 for t in 1:N2, m in 1:M]
    epidemic_start = [4 + m for m in 1:M]
    last_day = [N2 - m + 1 for m in 1:M]
    return (; M, P, N2, X, SI = SI ./ sum(SI), f, pop = [2e4 * (1 + m) for m in 1:M],
        epidemic_start, last_day, deaths)
end

function _hist_imperial_oracle(d)
    (; M, P, N2, X, SI, f, pop, epidemic_start, last_day, deaths) = d
    observed = _hist_window_columns(deaths, epidemic_start, last_day)
    return function (u)
        k = Ref(0)
        next() = u[k[] += 1]
        positive(prior) = (v = next(); x = exp(v); (x, logpdf(prior, x) + v))
        tau, lp = positive(Exponential(1 / 0.03))
        phi, l = positive(truncated(Normal(0, 5), 0, Inf)); lp += l
        kappa, l = positive(truncated(Normal(0, 0.5), 0, Inf)); lp += l
        ah = map(_ -> positive(Gamma(0.1667, 1)), 1:P)
        mu = map(_ -> positive(truncated(Normal(3.28, kappa), 0, Inf)), 1:M)
        y0 = map(_ -> positive(Exponential(tau)), 1:M)
        ifr = map(_ -> positive(truncated(Normal(1, 0.1), 0, Inf)), 1:M)
        lp += sum(last, ah) + sum(last, mu) + sum(last, y0) + sum(last, ifr)
        alpha = first.(ah) .- log(1.05) / 6
        for m in 1:M
            pred = fill(first(y0[m]), N2)
            cumulative = 2 * first(y0[m])
            for t in 4:N2
                cumulative += pred[t - 1]
                rt = first(mu[m]) * exp(-sum(X[m, t, j] * alpha[j] for j in 1:P))
                pred[t] = (pop[m] - cumulative) / pop[m] * rt *
                    sum(pred[s] * SI[t - s] for s in 1:(t - 1))
            end
            for (i, t) in enumerate(epidemic_start[m]:last_day[m])
                e = first(ifr[m]) * sum(pred[s] * f[t - s, m] for s in 1:(t - 1))
                lp += logpdf(NegativeBinomial(phi, phi / (phi + e)), observed[m][i])
            end
        end
        return lp
    end
end

@testset "scan history: the Imperial COVID-19 renewal model per country" begin
    d = _hist_imperial_data(3, 24, 2)
    data = Dict{Symbol,Any}(pairs(d))
    snapshot = deepcopy(data)
    plan = bind_data(lower_rkppl(_hist_imperial_program(), data;
        conditioned = (:observed,)), data)
    built = build_kernel(plan)
    q = prepare_sampler(built, plan, zeros(built.layout.total);
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    @test data == snapshot
    @test built.layout.total == 3 + 2 + 3 * 3
    _hist_check((; plan, built, q), _hist_imperial_oracle(d))
    # Both recurrences stay RK scans, retained inside the country plate: the
    # infections fold (which carries the cumulative sum in its window) and
    # the expected-deaths fold. Nothing reads `cumulative` itself, so its
    # own fold is pruned.
    @test count(e -> e.kind === :scan, recipe_inventory(built.spec)) == 2
end

@testset "scan history: arrays reading one another's history" begin
    # Capability gap: each fold produces one array's history, so two arrays
    # that read each other's earlier entries are not built yet.
    T = 6
    data = Dict{Symbol,Any}(:y => _HIST_Y[1:T], :T => T)
    program = quote
        r ~ LogNormal(0, 0.3)
        @scan begin
            p[1] = 1.0
            q[1] = 1.0
            for t in 2:T
                p[t] = r * sum(q[s] for s in 1:(t - 1)) / t
                q[t] = r * sum(p[s] for s in 1:(t - 1)) / t
            end
        end
        y .~ Normal.(p, 1.0)
    end
    built = try
        plan = bind_data(lower_rkppl(program, Tuple(keys(data));
            conditioned = (:y,)), data)
        build_kernel(plan)
    catch err
        err isa ContractValidationError || rethrow()
        nothing
    end
    @test_broken built !== nothing

    # Sampled seeds now use the ordinary packed cell-site layout.
    sampled = try
        lower_rkppl(quote
            s ~ Exponential(1)
            @plate for m in eachindex(y)
                @scan begin
                    h[1] ~ Normal(0, 1)
                    for t in 2:3
                        h[t] = h[t - 1]
                    end
                end
                loc[m] = h
                y[m] .~ Normal.(loc[m], s)
            end
        end, (:y,); conditioned = (:y,))
    catch err
        err isa SurfaceLoweringError || rethrow()
        nothing
    end
    @test sampled !== nothing
end
