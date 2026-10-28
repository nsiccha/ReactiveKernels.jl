module ObservedScanTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions
import DifferentiationInterface as DI
import Enzyme

# A `@scan` carried array whose name is observed data observes its entries
# (rkppl-use §5, §13): each seed's and step's density at its observed entry
# is the conditional density given the earlier ones, in `likelihood` and
# `:pointwise`; reads of the array read the data. Oracles are independent
# Distributions.jl recursions.

build(program, data; observed = (:y,), mod = Main) = begin
    plan = lower_rkppl(program, data; conditioned = observed, mod)
    bound = bind_data(plan, data)
    (; plan, bound, built = build_kernel(bound))
end

raw(f, u, name) = begin
    e = only(e for e in f.built.layout.entries if e.name === name)
    u[e.offset:(e.offset + e.size - 1)]
end

flat(x::AbstractVector{<:AbstractVector}) = reduce(vcat, x; init = Float64[])
flat(x) = x

# `oracle(u)` returns (; prior, likelihood, jac, pointwise).
function check(f, oracle; shifts = (0.0, 0.17), name = :y)
    n = f.built.layout.total
    initial = [0.11cos(i) for i in 1:n]
    sampler = prepare_sampler(f.built, f.bound, initial;
        backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
    prior = prepare_query(f.built, f.bound, :prior)
    likelihood = prepare_query(f.built, f.bound, :likelihood)
    pointwise = prepare_query(f.built, f.bound, :pointwise)
    data = deepcopy(f.bound.columns[name])
    for shift in shifts
        u = initial .+ shift
        expected = oracle(u)
        saved = copy(u)
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ expected.prior + expected.likelihood + expected.jac rtol = 1e-11
        h = cbrt(eps(Float64))
        posterior(x) = (o = oracle(x); o.prior + o.likelihood + o.jac)
        fd = map(eachindex(u)) do i
            up, dn = copy(u), copy(u)
            up[i] += h
            dn[i] -= h
            (posterior(up) - posterior(dn)) / (2h)
        end
        @test gradient ≈ fd rtol = 2e-6 atol = 2e-7
        @test Base.invokelatest(prior, u) ≈ expected.prior rtol = 1e-11 atol = 1e-12
        @test Base.invokelatest(likelihood, u) ≈ expected.likelihood rtol = 1e-11 atol = 1e-12
        pw = getproperty(Base.invokelatest(pointwise, u), name)
        @test length.(pw) == length.(expected.pointwise)
        @test flat(pw) ≈ flat(expected.pointwise) rtol = 1e-11 atol = 1e-12
        @test logjac(f.built.layout, u) ≈ expected.jac atol = 1e-12
        @test u == saved
    end
    @test f.bound.columns[name] == data
    return nothing
end

structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
    if e.kind !== :ordinary]

series(n, k = 0) = [0.4sin(0.9t + k) + 0.1t for t in 1:n]

ar_errors = quote
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    @scan begin
        y[1] ~ Normal(a + b * x[1], sigma / sqrt(1 - phi^2))
        for t in 2:T
            y[t] ~ Normal(a + b * x[t] + phi * (y[t - 1] - a - b * x[t - 1]), sigma)
        end
    end
end

function ar_oracle(f, y, x)
    u -> begin
        phi = 2 / (1 + exp(-only(raw(f, u, :phi)))) - 1
        ls = only(raw(f, u, :sigma)); sigma = exp(ls)
        a = only(raw(f, u, :a)); b = only(raw(f, u, :b))
        jac = log(2) + log(1 + phi) + log(1 - phi) - log(4) + ls
        prior = logpdf(Uniform(-1, 1), phi) + logpdf(Exponential(1), sigma) +
            logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b)
        m = a .+ b .* x
        pw = [logpdf(Normal(m[1], sigma / sqrt(1 - phi^2)), y[1]);
            [logpdf(Normal(m[t] + phi * (y[t - 1] - m[t - 1]), sigma), y[t])
                for t in 2:length(y)]]
        (; prior, likelihood = sum(pw), jac, pointwise = pw)
    end
end

@testset "an observed carried array observes each entry" begin
    for n in (2, 7, 40)
        y, x = series(n), [0.3cos(t) for t in 1:n]
        f = build(ar_errors, (; y, x))
        # Only the declared parameters: the observed trajectory has no coordinates.
        @test coordinate_names(f.built.layout) == [:phi, :sigma, :a, :b]
        check(f, ar_oracle(f, y, x))
        @test (:scan, 0) in structure(f.built.spec)
        # A supplied length is the scan's length too.
        g = build(ar_errors, (; y, x, T = n))
        check(g, ar_oracle(g, y, x); shifts = (0.05,))
    end
    # A model call observes it through `|`.
    m = @rkppl begin
        phi ~ Uniform(-1, 1)
        sigma ~ Exponential(1)
        @scan begin
            y[1] ~ Normal(0, sigma / sqrt(1 - phi^2))
            for t in 2:T
                y[t] ~ Normal(phi * y[t - 1], sigma)
            end
        end
    end
    y = series(9)
    bound = m() | (; y)
    built = build_kernel(bound)
    u = [0.2, -0.3]
    phi, sigma = 2 / (1 + exp(-u[1])) - 1, exp(u[2])
    expected = [logpdf(Normal(0, sigma / sqrt(1 - phi^2)), y[1]);
        [logpdf(Normal(phi * y[t - 1], sigma), y[t]) for t in 2:9]]
    @test Base.invokelatest(prepare_query(built, bound, :pointwise), u).y ≈ expected rtol = 1e-12
    # New observed values of the same length bind to the same build.
    y2 = series(9, 1.3)
    rebound = bind_data(bound, (; y = y2))
    expected2 = [logpdf(Normal(0, sigma / sqrt(1 - phi^2)), y2[1]);
        [logpdf(Normal(phi * y2[t - 1], sigma), y2[t]) for t in 2:9]]
    @test Base.invokelatest(prepare_query(built, rebound, :likelihood), u) ≈ sum(expected2) rtol = 1e-12
end

cell_ar_errors = quote
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    @plate for g in eachindex(y)
        @scan begin
            y[g][1] ~ Normal(mu[g][1], sigma / sqrt(1 - phi^2))
            for t in 2:length(y[g])
                y[g][t] ~ Normal(mu[g][t] + phi * (y[g][t - 1] - mu[g][t - 1]), sigma)
            end
        end
    end
end

@testset "observed scans per group in @plate cells" begin
    for lengths in ([3, 1, 5], [6, 2, 4, 1, 7, 3])
        y = [series(n, g) for (g, n) in enumerate(lengths)]
        mu = [[0.2g + 0.05t for t in 1:n] for (g, n) in enumerate(lengths)]
        f = build(cell_ar_errors, (; y, mu))
        @test coordinate_names(f.built.layout) == [:phi, :sigma]
        oracle(u) = begin
            phi = 2 / (1 + exp(-only(raw(f, u, :phi)))) - 1
            ls = only(raw(f, u, :sigma)); sigma = exp(ls)
            jac = log(2) + log(1 + phi) + log(1 - phi) - log(4) + ls
            prior = logpdf(Uniform(-1, 1), phi) + logpdf(Exponential(1), sigma)
            pw = map(y, mu) do yg, m
                [logpdf(Normal(m[1], sigma / sqrt(1 - phi^2)), yg[1]);
                    [logpdf(Normal(m[t] + phi * (yg[t - 1] - m[t - 1]), sigma), yg[t])
                        for t in 2:length(yg)]]
            end
            (; prior, likelihood = sum(sum, pw), jac, pointwise = pw)
        end
        check(f, oracle)
        # One recurrence per group, retained inside the group plate.
        @test (:plate, 0) in structure(f.built.spec)
        @test any(e -> e[1] === :scan && e[2] >= 1, structure(f.built.spec))
    end
    # The retained structure does not grow with the groups.
    shape(lengths) = structure(build(cell_ar_errors, (;
        y = [series(n, g) for (g, n) in enumerate(lengths)],
        mu = [zeros(n) for n in lengths])).built.spec)
    @test shape([2, 3]) == shape([4, 1, 6, 2, 5, 3, 2])
end

# A submodel whose body holds the scan and returns its carried array
# (rkppl-use §12): `y[g] ~ ar_gaps(…)` in a cell observes `y[g]` exactly as
# the hand-written cell scan `y[g][t] ~ …` does. AR(1) errors over irregular
# times: a gap of k steps has correlation phi^k.
@rkppl ar_gaps(mu, sigma, phi, time) = begin
    @scan begin
        y[1] ~ Normal(mu[1], sigma / sqrt(1 - phi^2))
        for t in 2:length(time)
            k = time[t] - time[t - 1]
            y[t] ~ Normal(mu[t] + phi^k * (y[t - 1] - mu[t - 1]),
                sigma * sqrt((1 - phi^(2k)) / (1 - phi^2)))
        end
    end
    return y
end

ar_gaps_call = quote
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    @plate for g in eachindex(y)
        y[g] ~ ar_gaps(mu[g], sigma, phi, time[g])
    end
end
ar_gaps_inline = quote
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    @plate for g in eachindex(y)
        @scan begin
            y[g][1] ~ Normal(mu[g][1], sigma / sqrt(1 - phi^2))
            for t in 2:length(time[g])
                k = time[g][t] - time[g][t - 1]
                y[g][t] ~ Normal(mu[g][t] + phi^k * (y[g][t - 1] - mu[g][t - 1]),
                    sigma * sqrt((1 - phi^(2k)) / (1 - phi^2)))
            end
        end
    end
end

gap_times(n, g) = cumsum([1.0 + mod(t + g, 3) for t in 1:n])

@testset "a submodel's observed carried array, called per cell" begin
    for lengths in ([3, 1, 5], [6, 2, 4, 1, 7, 3])
        y = [series(n, g) for (g, n) in enumerate(lengths)]
        mu = [[0.2g + 0.05t for t in 1:n] for (g, n) in enumerate(lengths)]
        time = [gap_times(n, g) for (g, n) in enumerate(lengths)]
        data = (; y, mu, time)
        f = build(ar_gaps_call, data; mod = @__MODULE__)
        twin = build(ar_gaps_inline, data)
        @test coordinate_names(f.built.layout) == [:phi, :sigma]
        oracle(u) = begin
            phi = 2 / (1 + exp(-only(raw(f, u, :phi)))) - 1
            ls = only(raw(f, u, :sigma)); sigma = exp(ls)
            jac = log(2) + log(1 + phi) + log(1 - phi) - log(4) + ls
            prior = logpdf(Uniform(-1, 1), phi) + logpdf(Exponential(1), sigma)
            pw = map(y, mu, time) do yg, m, tg
                [logpdf(Normal(m[1], sigma / sqrt(1 - phi^2)), yg[1]);
                    [logpdf(Normal(m[t] + phi^(tg[t] - tg[t - 1]) * (yg[t - 1] - m[t - 1]),
                        sigma * sqrt((1 - phi^(2(tg[t] - tg[t - 1]))) / (1 - phi^2))), yg[t])
                        for t in 2:length(yg)]]
            end
            (; prior, likelihood = sum(sum, pw), jac, pointwise = pw)
        end
        check(f, oracle)
        # The call expands into the hand-written cell scan: the same retained
        # structure and the same posterior.
        @test structure(f.built.spec) == structure(twin.built.spec)
        q, qt = (prepare_query(x.built, x.bound, :sampler) for x in (f, twin))
        for u in ([0.3, -0.2], [-1.1, 0.4])
            @test Base.invokelatest(q, u) ≈ Base.invokelatest(qt, u) rtol = 1e-12
        end
    end
end

# The same rule at top level: a single observed series.
@rkppl ar_steps(mu, sigma, phi) = begin
    @scan begin
        y[1] ~ Normal(mu[1], sigma / sqrt(1 - phi^2))
        for t in 2:T
            y[t] ~ Normal(mu[t] + phi * (y[t - 1] - mu[t - 1]), sigma)
        end
    end
    return y
end

@testset "a submodel's observed carried array, called at top level" begin
    called = quote
        phi ~ Uniform(-1, 1)
        sigma ~ Exponential(1)
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        m = a .+ b .* x
        y ~ ar_steps(m, sigma, phi)
    end
    for n in (2, 9)
        y, x = series(n), [0.3cos(t) for t in 1:n]
        f = build(called, (; y, x); mod = @__MODULE__)
        @test coordinate_names(f.built.layout) == [:phi, :sigma, :a, :b]
        check(f, ar_oracle(f, y, x))
        @test (:scan, 0) in structure(f.built.spec)
    end
end

# Without data, the returned carried array is the latent trajectory per
# index, as the hand-written per-index spelling `h[g][t] ~ …` is.
@rkppl walk(phi, s, len) = begin
    @scan begin
        h[1] ~ Normal(0, 1)
        for t in 2:len
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end
    return h
end
@rkppl scaled_walk(s, len) = begin
    sc = 2 * s
    @scan begin
        h[1] ~ Normal(0, 1)
        for t in 2:len
            z ~ Normal(0, 1)
            h[t] = h[t - 1] + sc * z
        end
    end
    return h
end

@rkppl rebinding_walk(s, len) = begin
    k = 2 * s
    @scan begin
        h[1] ~ Normal(0, 1)
        for t in 2:len
            k = h[t - 1]
            h[t] ~ Normal(k, s)
        end
    end
    return h
end

@testset "a submodel's latent carried array, called per cell" begin
    data = (; y = [series(4), series(1), series(3, 2)])
    pairs = (
        (quote
            phi ~ Normal(0, 0.7)
            s ~ Exponential(1)
            @plate for g in eachindex(y)
                h[g] ~ walk(phi, s, length(y[g]))
                y[g] .~ Normal.(h[g], 0.5)
            end
        end, quote
            phi ~ Normal(0, 0.7)
            s ~ Exponential(1)
            @plate for g in eachindex(y)
                @scan begin
                    h[g][1] ~ Normal(0, 1)
                    for t in 2:length(y[g])
                        h[g][t] ~ Normal(phi * h[g][t - 1], s)
                    end
                end
                y[g] .~ Normal.(h[g], 0.5)
            end
        end),
        # A body local read by the scan, with non-centered steps. The call
        # exposes the local per index (`h[g].sc`), so its twin is the
        # indexed output `sc[g]`, which the cell's scan step reads.
        (quote
            s ~ Exponential(1)
            @plate for g in eachindex(y)
                h[g] ~ scaled_walk(s, length(y[g]))
                y[g] .~ Normal.(h[g], 0.5)
            end
        end, quote
            s ~ Exponential(1)
            @plate for g in eachindex(y)
                sc[g] = 2 * s
                @scan begin
                    h[g][1] ~ Normal(0, 1)
                    for t in 2:length(y[g])
                        z ~ Normal(0, 1)
                        h[g][t] = h[g][t - 1] + sc[g] * z
                    end
                end
                y[g] .~ Normal.(h[g], 0.5)
            end
        end))
    # refused: single assignment (rkppl-use §9, §12 "a genuine
    # redefinition") — the cell and its scan's step both bind `k`.
    @test_throws SurfaceLoweringError build(quote
        s ~ Exponential(1)
        @plate for g in eachindex(y)
            h[g] ~ rebinding_walk(s, length(y[g]))
            y[g] .~ Normal.(h[g], 0.5)
        end
    end, data; mod = @__MODULE__)
    for (called, inline) in pairs
        f, twin = build(called, data; mod = @__MODULE__), build(inline, data)
        @test f.built.layout.total == twin.built.layout.total
        @test structure(f.built.spec) == structure(twin.built.spec)
        sampler(x) = prepare_sampler(x.built, x.bound, zeros(x.built.layout.total);
            backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
        q, qt = sampler(f), sampler(twin)
        for shift in (0.0, 0.3)
            u = [0.1sin(i) + shift for i in 1:f.built.layout.total]
            v, g = sampler_value_and_gradient!(q, similar(u), u)
            vt, gt = sampler_value_and_gradient!(qt, similar(u), u)
            @test v ≈ vt rtol = 1e-12
            @test g ≈ gt rtol = 1e-10
        end
    end
end

@testset "observed and latent carried arrays share one recurrence" begin
    # Stochastic volatility: a latent log-variance observed through y.
    sv = quote
        mu ~ Normal(0, 1)
        phi ~ Uniform(-1, 1)
        s ~ Exponential(1)
        @scan begin
            h[1] ~ Normal(mu, s / sqrt(1 - phi^2))
            y[1] ~ Normal(0, exp(h[1] / 2))
            for t in 2:T
                h[t] ~ Normal(mu + phi * (h[t - 1] - mu), s)
                y[t] ~ Normal(0, exp(h[t] / 2))
            end
        end
    end
    sv_parts(f, u) = (; mu = only(raw(f, u, :mu)),
        phi = 2 / (1 + exp(-only(raw(f, u, :phi)))) - 1, ls = only(raw(f, u, :s)))
    sv_terms(p, h, y) = begin
        s = exp(p.ls)
        prior = logpdf(Normal(p.mu, s / sqrt(1 - p.phi^2)), h[1]) +
            sum(logpdf(Normal(p.mu + p.phi * (h[t - 1] - p.mu), s), h[t]) for t in 2:length(h); init = 0.0)
        pw = [logpdf(Normal(0, exp(h[t] / 2)), y[t]) for t in eachindex(y)]
        (prior, pw)
    end
    for n in (3, 12)
        y = series(n)
        f = build(sv, (; y))
        # One coordinate per latent state, none for the observed array.
        @test f.built.layout.total == 3 + n
        hentry = only(e for e in f.built.layout.entries if e.kind === :scan)
        oracle(u) = begin
            p = sv_parts(f, u)
            h = u[hentry.offset:(hentry.offset + n - 1)]
            prior, pw = sv_terms(p, h, y)
            prior += logpdf(Normal(0, 1), p.mu) + logpdf(Uniform(-1, 1), p.phi) +
                logpdf(Exponential(1), exp(p.ls))
            jac = log(2) + log(1 + p.phi) + log(1 - p.phi) - log(4) + p.ls
            (; prior, likelihood = sum(pw), jac, pointwise = pw)
        end
        check(f, oracle)
    end
    # The same recurrence per group, with ragged series.
    cell_sv = quote
        mu ~ Normal(0, 1)
        phi ~ Uniform(-1, 1)
        s ~ Exponential(1)
        @plate for g in eachindex(y)
            @scan begin
                h[1] ~ Normal(mu, s / sqrt(1 - phi^2))
                y[g][1] ~ Normal(0, exp(h[1] / 2))
                for t in 2:length(y[g])
                    h[t] ~ Normal(mu + phi * (h[t - 1] - mu), s)
                    y[g][t] ~ Normal(0, exp(h[t] / 2))
                end
            end
        end
    end
    lengths = [4, 1, 3]
    y = [series(n, g) for (g, n) in enumerate(lengths)]
    f = build(cell_sv, (; y))
    @test f.built.layout.total == 3 + sum(lengths)
    seeds = only(e for e in f.built.layout.entries if occursin("seed", string(e.name)))
    steps = only(e for e in f.built.layout.entries if occursin("draw", string(e.name)))
    @test (seeds.size, steps.size) == (3, sum(lengths) - 3)
    oracle(u) = begin
        p = sv_parts(f, u)
        seed, step = u[seeds.offset:(seeds.offset + 2)], u[steps.offset:(steps.offset + steps.size - 1)]
        offsets = cumsum([0; lengths .- 1])
        prior, pw = 0.0, Vector{Float64}[]
        for g in eachindex(y)
            h = [seed[g]; step[(offsets[g] + 1):offsets[g + 1]]]
            pg, wg = sv_terms(p, h, y[g])
            prior += pg
            push!(pw, wg)
        end
        prior += logpdf(Normal(0, 1), p.mu) + logpdf(Uniform(-1, 1), p.phi) +
            logpdf(Exponential(1), exp(p.ls))
        jac = log(2) + log(1 + p.phi) + log(1 - p.phi) - log(4) + p.ls
        (; prior, likelihood = sum(sum, pw), jac, pointwise = pw)
    end
    check(f, oracle)
end

@testset "deterministic carried arrays, lags and history read the observed data" begin
    # ARMA(1,1): the residual is a deterministic carried array of the data.
    arma = quote
        mu ~ Normal(0, 1)
        phi ~ Normal(0, 0.5)
        theta ~ Normal(0, 0.5)
        sigma ~ Exponential(1)
        @scan begin
            y[1] ~ Normal(mu + phi * mu, sigma)
            err[1] = y[1] - (mu + phi * mu)
            for t in 2:T
                nu = mu + phi * y[t - 1] + theta * err[t - 1]
                y[t] ~ Normal(nu, sigma)
                err[t] = y[t] - nu
            end
        end
    end
    y = series(10)
    f = build(arma, (; y))
    @test coordinate_names(f.built.layout) == [:mu, :phi, :theta, :sigma]
    check(f, u -> begin
        mu, phi, theta, ls = u
        sigma = exp(ls)
        prior = logpdf(Normal(0, 1), mu) + logpdf(Normal(0, 0.5), phi) +
            logpdf(Normal(0, 0.5), theta) + logpdf(Exponential(1), sigma)
        nu = mu + phi * mu
        pw = [logpdf(Normal(nu, sigma), y[1])]
        err = y[1] - nu
        for t in 2:length(y)
            nu = mu + phi * y[t - 1] + theta * err
            push!(pw, logpdf(Normal(nu, sigma), y[t]))
            err = y[t] - nu
        end
        (; prior, likelihood = sum(pw), jac = ls, pointwise = pw)
    end)
    # AR(2) with two seeds, and a step reading the whole earlier history.
    ar2 = quote
        p1 ~ Normal(0, 0.5)
        p2 ~ Normal(0, 0.5)
        w ~ Normal(0, 0.5)
        s ~ Exponential(1)
        @scan begin
            y[1] ~ Normal(0, 2)
            y[2] ~ Normal(p1 * y[1], 2)
            for t in 3:T
                y[t] ~ Normal(p1 * y[t - 1] + p2 * y[t - 2] +
                    w * sum(y[k] for k in 1:(t - 1)) / (t - 1), s)
            end
        end
    end
    y = series(8)
    f = build(ar2, (; y))
    check(f, u -> begin
        p1, p2, w, ls = u
        s = exp(ls)
        prior = logpdf(Normal(0, 0.5), p1) + logpdf(Normal(0, 0.5), p2) +
            logpdf(Normal(0, 0.5), w) + logpdf(Exponential(1), s)
        pw = [logpdf(Normal(0, 2), y[1]); logpdf(Normal(p1 * y[1], 2), y[2]);
            [logpdf(Normal(p1 * y[t - 1] + p2 * y[t - 2] + w * sum(y[1:(t - 1)]) / (t - 1), s), y[t])
                for t in 3:length(y)]]
        (; prior, likelihood = sum(pw), jac = ls, pointwise = pw)
    end)
end

@testset "observations keep their stated support and normalizer" begin
    positive = quote
        phi ~ Normal(0, 0.5)
        s ~ Exponential(1)
        @scan begin
            y[1] ~ Exponential(1)
            for t in 2:T
                y[t] ~ truncated(Normal(phi * y[t - 1], s), 0, Inf)
            end
        end
    end
    y = abs.(series(6)) .+ 0.1
    f = build(positive, (; y))
    check(f, u -> begin
        phi, ls = u
        s = exp(ls)
        pw = [logpdf(Exponential(1), y[1]);
            [logpdf(truncated(Normal(phi * y[t - 1], s), 0, Inf), y[t]) for t in 2:6]]
        (; prior = logpdf(Normal(0, 0.5), phi) + logpdf(Exponential(1), s),
            likelihood = sum(pw), jac = ls, pointwise = pw)
    end)
    # An observed entry outside the stated support has zero density.
    for bad in (1, 4)
        z = copy(y)
        z[bad] = -0.2
        g = build(positive, (; y = z))
        pw = Base.invokelatest(prepare_query(g.built, g.bound, :pointwise), [0.1, 0.2]).y
        @test pw[bad] == -Inf
        @test all(isfinite, pw[setdiff(1:6, bad)])
    end
end

@testset "a prediction design leaves the observed array absent" begin
    fit = build(ar_errors, (; y = series(6), x = collect(0.1:0.1:0.6), T = 6))
    design = bind_data(fit.bound, (; x = collect(0.2:0.1:0.7), T = 6))
    u = [0.1, -0.2, 0.3, 0.05]
    expected = logpdf(Uniform(-1, 1), 2 / (1 + exp(-0.1)) - 1) +
        logpdf(Exponential(1), exp(-0.2)) + logpdf(Normal(0, 1), 0.3) + logpdf(Normal(0, 1), 0.05)
    @test Base.invokelatest(prepare_query(fit.built, design, :prior), u) ≈ expected rtol = 1e-12
    for preset in (:likelihood, :pointwise, :sampler)
        err = try
            prepare_query(fit.built, design, preset); nothing
        catch e
            e
        end
        @test err isa ContractValidationError && occursin("`y`", sprint(showerror, err))
    end
    # Per group: the design reads the groups from another array.
    cell = quote
        phi ~ Uniform(-1, 1)
        sigma ~ Exponential(1)
        @plate for g in eachindex(mu)
            @scan begin
                y[g][1] ~ Normal(mu[g][1], sigma / sqrt(1 - phi^2))
                for t in 2:length(mu[g])
                    y[g][t] ~ Normal(mu[g][t] + phi * (y[g][t - 1] - mu[g][t - 1]), sigma)
                end
            end
        end
    end
    mu = [[0.1, 0.2, 0.3], [0.4, 0.5]]
    fitted = build(cell, (; y = [series(3), series(2)], mu))
    design = bind_data(fitted.bound, (; mu))
    @test Base.invokelatest(prepare_query(fitted.built, design, :prior), [0.1, -0.2]) ≈
        logpdf(Uniform(-1, 1), 2 / (1 + exp(-0.1)) - 1) + logpdf(Exponential(1), exp(-0.2)) rtol = 1e-12
    err = try
        prepare_query(fitted.built, design, :likelihood); nothing
    catch e
        e
    end
    @test err isa ContractValidationError && occursin("`y`", sprint(showerror, err))
end

@testset "without data, the same block is a latent trajectory" begin
    # Top level: a prior-only recurrence of a supplied length binds with no
    # observation rows, and samples the trajectory.
    m = @rkppl begin
        phi ~ Uniform(-1, 1)
        sigma ~ Exponential(1)
        @scan begin
            y[1] ~ Normal(0, sigma / sqrt(1 - phi^2))
            for t in 2:T
                y[t] ~ Normal(phi * y[t - 1], sigma)
            end
        end
    end
    for T in (1, 5)
        built = build_kernel(m(; T))
        @test coordinate_names(built.layout) == [:phi, :sigma, (Symbol("y.", t) for t in 1:T)...]
        u = [0.2, -0.1, (0.3sin(t) for t in 1:T)...]
        phi, sigma, y = 2 / (1 + exp(-u[1])) - 1, exp(u[2]), u[3:end]
        expected = logpdf(Uniform(-1, 1), phi) + logpdf(Exponential(1), sigma) +
            logpdf(Normal(0, sigma / sqrt(1 - phi^2)), y[1]) +
            sum(logpdf(Normal(phi * y[t - 1], sigma), y[t]) for t in 2:T; init = 0.0)
        @test Base.invokelatest(prepare_query(built, bind_data(m(; T)), :prior), u) ≈ expected rtol = 1e-12
    end
end

@testset "without data, the per-index spelling is a latent trajectory" begin
    # `h[g][t] ~ …` with `h` not data is the cell-local carried array the
    # plate returns as `h[g]` (rkppl-use §5, "A recurrence per group").
    indexed = quote
        phi ~ Normal(0, 0.7)
        s ~ Exponential(1)
        @plate for g in eachindex(y)
            @scan begin
                h[g][1] ~ Normal(0, 1)
                for t in 2:length(y[g])
                    h[g][t] ~ Normal(phi * h[g][t - 1], s)
                end
            end
            y[g] .~ Normal.(h[g], 0.5)
        end
    end
    local_spelling = quote
        phi ~ Normal(0, 0.7)
        s ~ Exponential(1)
        @plate for g in eachindex(y)
            @scan begin
                h[1] ~ Normal(0, 1)
                for t in 2:length(y[g])
                    h[t] ~ Normal(phi * h[t - 1], s)
                end
            end
            loc[g] = h
            y[g] .~ Normal.(loc[g], 0.5)
        end
    end
    data = (; y = [series(4), series(1), series(3, 2)])
    a, b = build(indexed, data), build(local_spelling, data)
    @test a.built.layout.total == b.built.layout.total == 2 + 8
    for shift in (0.0, 0.3)
        u = [0.1sin(i) + shift for i in 1:a.built.layout.total]
        qa = prepare_query(a.built, a.bound, :sampler)
        qb = prepare_query(b.built, b.bound, :sampler)
        @test Base.invokelatest(qa, u) ≈ Base.invokelatest(qb, u) rtol = 1e-12
    end
end

module ObservedScanTypeStability
import ReactiveKernels
include(joinpath(pkgdir(ReactiveKernels), "test", "type_stability_audit.jl"))
end
using .ObservedScanTypeStability.TypeStabilityAudit: findings, ad_call_findings

@testset "observed scan samplers are concretely typed" begin
    # RK principle review `…/2026-10-12T20-09-11-062-114x8kz` (type-stability).
    lengths = [4, 1, 3]
    for (program, data) in (
            (ar_errors, (; y = series(7), x = collect(0.1:0.1:0.7))),
            (cell_ar_errors, (; y = [series(n, g) for (g, n) in enumerate(lengths)],
                mu = [zeros(n) for n in lengths])))
        f = build(program, data)
        u = [0.05 * sin(i) for i in 1:f.built.layout.total]
        q = @test_logs min_level = Base.CoreLogging.Warn prepare_sampler(
            f.built, f.bound, u; backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
        @test isfinite(q(u))
        @test isempty(findings(q.kernel, (Vector{Float64},)))
        @test isempty(ad_call_findings(q.ad, u))
    end
end

@testset "observed arrays are written by `~` and hold one entry per index" begin
    # refused: an assignment would replace an observed value; data classify
    # a statement on an observed name as an observation (rkppl-use §9)
    assigned = quote
        s ~ Exponential(1)
        @scan begin
            y[1] = 0.0
            for t in 2:T
                y[t] ~ Normal(y[t - 1], s)
            end
        end
    end
    @test_throws SurfaceLoweringError lower_rkppl(assigned, (; y = series(4)); conditioned = (:y,))
    # refused: the recurrence writes T entries; observed data of another
    # length is not that recurrence (rkppl-use §5)
    @test_throws ContractValidationError build(ar_errors, (; y = series(5), x = zeros(5), T = 4))
    @test_throws ContractValidationError build(ar_errors, (; y = series(3), x = zeros(5), T = 5))
    # A one-entry array fails where the scan reads past it.
    short = build(ar_errors, (; y = [0.3], x = zeros(5), T = 5))
    @test_throws BoundsError Base.invokelatest(
        prepare_query(short.built, short.bound, :likelihood), zeros(4))
    # refused: observed values go through `|` / `condition`; a call keyword
    # pins (conditioning is not call syntax, user decision `18h1h54`)
    m = @rkppl begin
        s ~ Exponential(1)
        @scan begin
            y[1] ~ Normal(0, s)
            for t in 2:T
                y[t] ~ Normal(y[t - 1], s)
            end
        end
    end
    @test_throws SurfaceLoweringError build_kernel(m(; y = series(4)))
    mismatched = quote
        phi ~ Uniform(-1, 1)
        sigma ~ Exponential(1)
        @plate for g in eachindex(y)
            @scan begin
                y[g][1] ~ Normal(0, sigma)
                for t in 2:n[g]
                    y[g][t] ~ Normal(phi * y[g][t - 1], sigma)
                end
            end
        end
    end
    f = build(mismatched, (; y = [series(3), series(2)], n = [3, 1]))
    q = prepare_query(f.built, f.bound, :likelihood)
    @test_throws DimensionMismatch Base.invokelatest(q, [0.1, 0.2])
    # Not built yet: a missing entry would need its successors' densities
    # marginalized over it; binding names the gap.
    y = Union{Missing,Float64}[series(5)...]
    y[3] = missing
    @test_broken try
        build(ar_errors, (; y, x = zeros(5)))
        true
    catch e
        e isa ContractValidationError && occursin("missing entries", sprint(showerror, e)) || rethrow()
        false
    end
end

end # module
