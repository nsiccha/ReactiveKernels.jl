module MarginalizedScanTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, LinearAlgebra
import DifferentiationInterface as DI
import Enzyme
import Random, Statistics

# `marginalize(model, :x)` integrates a linear-Gaussian latent `@scan` out of
# the density exactly (a Kalman filter). Oracles are independent: the dense
# joint Gaussian of the observations (covariance assembled entry by entry
# from the stated recurrence), its sequential conditionals for the pointwise
# densities, and Gaussian conditioning for the latent draws.

structure(spec) = [(e.kind, e.depth) for e in recipe_inventory(spec)
    if e.kind !== :ordinary]

# Mean and covariance of `x` from `x[1] ~ Normal(m0, sqrt(P0))`,
# `x[t] = a[t] * x[t - 1] + c[t] + Normal(0, sqrt(q[t]))`.
function latent_moments(m0, P0, a, c, q)
    n = length(a)
    mean = zeros(n)
    cov = zeros(n, n)
    mean[1], cov[1, 1] = m0, P0
    for t in 2:n
        mean[t] = a[t] * mean[t - 1] + c[t]
        for s in 1:(t - 1)
            cov[t, s] = cov[s, t] = a[t] * cov[t - 1, s]
        end
        cov[t, t] = a[t]^2 * cov[t - 1, t - 1] + q[t]
    end
    return mean, cov
end

# The joint Gaussian of `y = offset .+ loading .* x .+ noise .* z`.
function observation_moments(lat, offset, loading, noise)
    mean, cov = lat
    n = length(mean)
    L = loading .* ones(n)
    mu = offset .+ L .* mean
    S = L .* cov .* L' + Diagonal((noise .* ones(n)) .^ 2)
    return mu, Symmetric(Matrix(S))
end

# log p(y[t] | y[1:t-1]) for each t, from the joint Gaussian.
function sequential_logpdfs(mu, S, y)
    map(eachindex(y)) do t
        t == 1 && return logpdf(Normal(mu[1], sqrt(S[1, 1])), y[1])
        p = 1:(t - 1)
        w = S[p, p] \ S[p, t]
        logpdf(Normal(mu[t] + w' * (y[p] - mu[p]), sqrt(S[t, t] - S[t, p]' * w)), y[t])
    end
end

# x | y[present] from the joint Gaussian of (x, y).
function latent_posterior(lat, offset, loading, noise, y; present = eachindex(y))
    mx, Sx = lat
    n = length(mx)
    L = Matrix(Diagonal(loading .* ones(n)))[present, :]
    my = (offset .+ loading .* mx)[present]
    Sy = L * Sx * L' + Diagonal(((noise .* ones(n)) .^ 2)[present])
    K = Sx * L' / Sy
    return mx + K * (y[present] - my), Symmetric(Sx - K * L * Sx)
end

# The reporter's spelling: per-group OU deviations in one series, reset where a
# group starts (`continues[t] = 0`), non-centred innovations, irregular gaps.
const OU = @rkppl begin
    mu ~ Normal(0, 1)
    noise ~ Exponential(0.3)
    ou_sd ~ Exponential(0.2)
    ou_range ~ LogNormal(1, 0.5)
    level[1:N] .~ Normal.(mu, 0.5)
    @scan begin
        ou[1] ~ Normal(0, 1)
        for t in 2:R
            e ~ Normal(0, 1)
            r = continues[t] * exp(-gap[t] / ou_range)
            ou[t] = r * ou[t - 1] + sqrt(1 - r * r) * e
        end
    end
    y .~ Normal.(level[area] .+ ou_sd .* ou, noise)
end

function ou_data(N, T; seed = 3)
    rng = Random.Xoshiro(seed)
    R = N * T
    area = repeat(1:N; inner = T)
    continues = [t % T == 1 || T == 1 ? 0.0 : 1.0 for t in 1:R]
    gap = 0.5 .+ rand(rng, R)
    y = 1.0 .+ 0.3 .* randn(rng, R)
    return (; inputs = (; N, R, area, continues, gap), y)
end

# Sampler value and Enzyme Reverse gradient against the oracle density
# (`prior + likelihood + jac` at `u`), and the separate queries.
function check_density(built, plan, oracle; points = 3, seed = 1)
    n = built.layout.total
    rng = Random.Xoshiro(seed)
    sampler = prepare_sampler(built, plan, zeros(n);
        backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
    likelihood = prepare_query(built, plan, :likelihood)
    pointwise = prepare_query(built, plan, :pointwise)
    for _ in 1:points
        u = 0.4 .* randn(rng, n)
        expected = oracle(u)
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ expected.total rtol = 1e-10
        @test Base.invokelatest(likelihood, u) ≈ expected.likelihood rtol = 1e-10
        @test Base.invokelatest(pointwise, u).y ≈ expected.pointwise rtol = 1e-9 atol = 1e-11
        h = cbrt(eps(Float64))
        fd = map(1:n) do i
            up, dn = copy(u), copy(u)
            up[i] += h
            dn[i] -= h
            (oracle(up).total - oracle(dn).total) / (2h)
        end
        @test gradient ≈ fd rtol = 2e-6 atol = 2e-7
    end
end

ou_plan(d) = marginalize(OU(; d.inputs...), :ou) | (; y = d.y)

# `present`: the observed entries of `d.y` (the others are missing).
function ou_oracle(built, d; present = eachindex(d.y))
    (; N, R, area, continues, gap) = d.inputs
    function (u)
        c = constrain(built.layout, u)
        prior = logpdf(Normal(0, 1), c.mu) + logpdf(Exponential(0.3), c.noise) +
            logpdf(Exponential(0.2), c.ou_sd) + logpdf(LogNormal(1, 0.5), c.ou_range) +
            sum(logpdf.(Normal(c.mu, 0.5), c.level))
        a = continues .* exp.(-gap ./ c.ou_range)
        lat = latent_moments(0.0, 1.0, a, zeros(R), 1 .- a .^ 2)
        mu, S = observation_moments(lat, c.level[area], c.ou_sd, c.noise)
        P = present
        pointwise = zeros(R)
        pointwise[P] = sequential_logpdfs(mu[P], S[P, P], d.y[P])
        likelihood = logpdf(MvNormal(mu[P], Symmetric(S[P, P])), d.y[P])
        jac = logjac(built.layout, u)
        (; total = prior + likelihood + jac, likelihood, pointwise)
    end
end

@testset "the reporter's OU deviations integrate out exactly" begin
    for (N, T) in ((1, 2), (3, 1), (2, 5), (3, 9))
        d = ou_data(N, T; seed = N + T)
        plan = ou_plan(d)
        built = build_kernel(plan)
        # The sampler sees only the remaining parameters.
        @test coordinate_names(built.layout) ==
            [:mu, :noise, :ou_sd, :ou_range, (Symbol("level.$i") for i in 1:N)...]
        check_density(built, plan, ou_oracle(built, d))
    end
    # The model as written still samples its innovations.
    d = ou_data(2, 4)
    latent = build_kernel(OU(; d.inputs...) | (; y = d.y))
    @test latent.layout.total == 4 + 2 + d.inputs.R
end

@testset "the filter is one retained scan as the series grows" begin
    shape(N, T) = (d = ou_data(N, T); structure(build_kernel(ou_plan(d)).spec))
    small = shape(2, 3)
    @test (:scan, 0) in small
    @test small == shape(7, 40)
end

# Centred AR(1) with a drift and a sampled seed law; per-row loadings and
# per-row measurement scales (a data vector combined with a parameter).
const AR = @rkppl begin
    m0 ~ Normal(0, 1)
    s0 ~ Exponential(1)
    phi ~ Uniform(-1, 1)
    drift ~ Normal(0, 0.5)
    sigma ~ Exponential(0.5)
    noise ~ Exponential(0.5)
    @scan begin
        h[1] ~ Normal(m0, s0)
        for t in 2:T
            h[t] ~ Normal(phi * h[t - 1] + drift, sigma)
        end
    end
    y .~ Normal.(2.0 .- w .* h, sqrt.(noise^2 .+ se .^ 2))
end

# A random walk from a fixed start: its step is a local over the innovation,
# and the observation reads it through a definition.
const WALK = @rkppl begin
    a ~ Normal(0, 1)
    s ~ Exponential(1)
    noise ~ Exponential(1)
    @scan begin
        x[1] = 0.0
        for t in 2:T
            z ~ Normal(0, 1)
            step = s * z
            x[t] = x[t - 1] + step
        end
    end
    mu = a .+ x
    y .~ Normal.(mu, noise)
end

@testset "centred, drifting and random-walk spellings" begin
    rng = Random.Xoshiro(5)
    for n in (2, 6, 15)
        y = cumsum(0.3 .* randn(rng, n))
        w = 0.5 .+ rand(rng, n)
        se = 0.1 .+ 0.2 .* rand(rng, n)
        plan = marginalize(AR(; w, se), :h) | (; y)
        built = build_kernel(plan)
        @test coordinate_names(built.layout) == [:m0, :s0, :phi, :drift, :sigma, :noise]
        check_density(built, plan, function (u)
            c = constrain(built.layout, u)
            prior = logpdf(Normal(0, 1), c.m0) + logpdf(Exponential(1), c.s0) +
                logpdf(Uniform(-1, 1), c.phi) + logpdf(Normal(0, 0.5), c.drift) +
                logpdf(Exponential(0.5), c.sigma) + logpdf(Exponential(0.5), c.noise)
            lat = latent_moments(c.m0, c.s0^2, fill(c.phi, n), fill(c.drift, n),
                fill(c.sigma^2, n))
            mu, S = observation_moments(lat, 2.0, -w, sqrt.(c.noise^2 .+ se .^ 2))
            likelihood = logpdf(MvNormal(mu, S), y)
            (; total = prior + likelihood + logjac(built.layout, u), likelihood,
                pointwise = sequential_logpdfs(mu, S, y))
        end)

        plan = marginalize(WALK(), :x) | (; y)
        built = build_kernel(plan)
        @test coordinate_names(built.layout) == [:a, :s, :noise]
        check_density(built, plan, function (u)
            c = constrain(built.layout, u)
            prior = logpdf(Normal(0, 1), c.a) + logpdf(Exponential(1), c.s) +
                logpdf(Exponential(1), c.noise)
            lat = latent_moments(0.0, 0.0, ones(n), zeros(n), fill(c.s^2, n))
            mu, S = observation_moments(lat, c.a, 1.0, c.noise)
            likelihood = logpdf(MvNormal(mu, S), y)
            (; total = prior + likelihood + logjac(built.layout, u), likelihood,
                pointwise = sequential_logpdfs(mu, S, y))
        end)
    end
end

@testset "exact draws of the integrated-out trajectory" begin
    d = ou_data(2, 5; seed = 11)
    plan = ou_plan(d)
    built = build_kernel(plan)
    q = prepare_marginalized_latents(built, plan)
    u = 0.3 .* cos.(1:built.layout.total)
    c = constrain(built.layout, u)
    (; R, area, continues, gap) = d.inputs
    a = continues .* exp.(-gap ./ c.ou_range)
    filtered = q(u)
    @test keys(filtered) == (:ou,)
    @test filtered.ou.coef[2:end] ≈ a[2:end]
    @test filtered.ou.innovation_var[2:end] ≈ 1 .- a[2:end] .^ 2
    lat = latent_moments(0.0, 1.0, a, zeros(R), 1 .- a .^ 2)
    mean, cov = latent_posterior(lat, c.level[area], c.ou_sd, c.noise, d.y)
    # The last filtered state is the last smoothed state.
    @test filtered.ou.mean[end] ≈ mean[end]
    @test filtered.ou.var[end] ≈ cov[end, end]
    rng = Random.Xoshiro(2)
    n = 40_000
    draws = reduce(hcat, (marginalized_latent_draws(q, u, rng).ou for _ in 1:n))
    @test size(draws) == (R, n)
    @test all(abs.(vec(Statistics.mean(draws; dims = 2)) .- mean) .< 5 .* sqrt.(diag(cov) ./ n))
    @test Statistics.cov(draws; dims = 2) ≈ cov rtol = 0.05 atol = 0.02 * maximum(diag(cov))
    # A seeded generator reproduces the draws.
    @test marginalized_latent_draws(q, u, Random.Xoshiro(3)) ==
        marginalized_latent_draws(q, u, Random.Xoshiro(3))
end

# A missing entry of the observed vector is a step that only predicts: the
# filter reads the response's presence, and the likelihood skips the entry.
@testset "missing entries are steps that only predict" begin
    d = ou_data(2, 6; seed = 5)
    absent = [1, 4, 5, 9]
    y = Vector{Union{Missing,Float64}}(d.y)
    y[absent] .= missing
    present = setdiff(eachindex(y), absent)
    plan = marginalize(OU(; d.inputs...), :ou) | (; y)
    built = build_kernel(plan)
    @test coordinate_names(built.layout) == coordinate_names(build_kernel(ou_plan(d)).layout)
    check_density(built, plan, ou_oracle(built, d; present))
    # One program serves every absence pattern, none included, and every length.
    shape(y, d = d) = structure(build_kernel(marginalize(OU(; d.inputs...), :ou) | (; y)).spec)
    @test shape(y) == shape(Vector{Union{Missing,Float64}}(d.y))
    grown = ou_data(5, 12; seed = 6)
    ygrown = Vector{Union{Missing,Float64}}(grown.y)
    ygrown[2:7:end] .= missing
    @test shape(y) == shape(ygrown, grown)
    q = prepare_marginalized_latents(built, plan)
    u = 0.3 .* cos.(1:built.layout.total)
    c = constrain(built.layout, u)
    (; R, area, continues, gap) = d.inputs
    a = continues .* exp.(-gap ./ c.ou_range)
    lat = latent_moments(0.0, 1.0, a, zeros(R), 1 .- a .^ 2)
    mean, cov = latent_posterior(lat, c.level[area], c.ou_sd, c.noise, d.y; present)
    rng = Random.Xoshiro(4)
    n = 40_000
    draws = reduce(hcat, (marginalized_latent_draws(q, u, rng).ou for _ in 1:n))
    @test all(abs.(vec(Statistics.mean(draws; dims = 2)) .- mean) .< 5 .* sqrt.(diag(cov) ./ n))
    @test Statistics.cov(draws; dims = 2) ≈ cov rtol = 0.05 atol = 0.02 * maximum(diag(cov))
end

# ── Data augmentation: censored rows as bounded latents ───────────────────
#
# Rows below a detection limit and rounded rows enter the filter at latent
# values bounded by their own ranges (per-element bounds, RKPPL 0.1.179); the
# observed vector merges them with the measured rows in time order. The
# source without `marginalize` is the augmented latent model itself.

const CENSORED = @rkppl begin
    mu ~ Normal(0, 1)
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    noise ~ Exponential(0.5)
    @scan begin
        x[1] ~ Normal(0, sigma / sqrt(1 - phi^2))
        for t in 2:T
            x[t] ~ Normal(phi * x[t - 1], sigma)
        end
    end
    gap[1:K] .~ restricted.(Flat.(), 0, Inf)
    position[1:J] .~ restricted.(Flat.(), 0, 1)
    value = vcat(measured, limit .- gap, lo .+ (hi .- lo) .* position)[order]
    value .~ Normal.(mu .+ x, noise)
end

# `T` rows: every third below its limit, every fifth rounded to [lo, hi].
function censored_data(T; seed = 2)
    rng = Random.Xoshiro(seed)
    y = cumsum(0.4 .* randn(rng, T))
    below = [t for t in 1:T if t % 3 == 0]
    rounded = [t for t in 1:T if t % 5 == 0 && !(t in below)]
    seen = setdiff(1:T, below, rounded)
    limit = y[below] .+ 0.2 .+ 0.3 .* rand(rng, length(below))
    lo = floor.(y[rounded])
    inputs = (; T, K = length(below), J = length(rounded), measured = y[seen], limit, lo,
        hi = lo .+ 1, order = invperm(vcat(seen, below, rounded)))
    return (; inputs, seen, below, rounded)
end

censored_plan(d) = marginalize(CENSORED(; d.inputs...), :x)

# The augmented model's density with `x` integrated out: the Gaussian
# marginal of the merged vector, the priors, and the bounded latents' Jacobians.
function censored_oracle(built, d)
    (; T, measured, limit, lo, hi, order) = d.inputs
    function (u)
        c = constrain(built.layout, u)
        prior = logpdf(Normal(0, 1), c.mu) + logpdf(Uniform(-1, 1), c.phi) +
            logpdf(Exponential(1), c.sigma) + logpdf(Exponential(0.5), c.noise)
        value = vcat(measured, limit .- c.gap, lo .+ (hi .- lo) .* c.position)[order]
        _, Sx = latent_moments(0.0, c.sigma^2 / (1 - c.phi^2), fill(c.phi, T), zeros(T),
            fill(c.sigma^2, T))
        S = Symmetric(Sx + c.noise^2 * I)
        logpdf(MvNormal(fill(c.mu, T), S), value) + prior + logjac(built.layout, u)
    end
end

# The sampler's density and gradient against the oracle and its finite
# differences.
function check_total(built, plan, oracle; points = 3, seed = 1)
    n = built.layout.total
    rng = Random.Xoshiro(seed)
    sampler = prepare_sampler(built, plan, zeros(n);
        backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
    for _ in 1:points
        u = 0.4 .* randn(rng, n)
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ oracle(u) rtol = 1e-10
        h = cbrt(eps(Float64))
        fd = map(1:n) do i
            up, dn = copy(u), copy(u)
            up[i] += h
            dn[i] -= h
            (oracle(up) - oracle(dn)) / (2h)
        end
        @test gradient ≈ fd rtol = 2e-6 atol = 2e-7
    end
end

@testset "censored rows enter the filter at bounded latent values" begin
    d = censored_data(16)
    plan = censored_plan(d)
    built = build_kernel(plan)
    # The layout keeps the augmentation and drops the recurrence.
    latent = coordinate_names(build_kernel(CENSORED(; d.inputs...)).layout)
    @test coordinate_names(built.layout) ==
        filter(n -> !occursin(r"(^|_)x\.", String(n)), latent)
    @test any(n -> occursin(r"(^|_)x\.", String(n)), latent)
    check_total(built, plan, censored_oracle(built, d))
    # With `measured` bound as an input, nothing is conditioned: the merged
    # vector's term is a factor, as in the source without `marginalize`.
    let u = 0.2 .* cos.(1:built.layout.total), latent = bind_data(CENSORED(; d.inputs...))
        lbuilt = build_kernel(latent)
        lu = 0.2 .* cos.(1:lbuilt.layout.total)
        for (b, p, v) in ((built, plan, u), (lbuilt, latent, lu))
            @test Base.invokelatest(prepare_query(b, p, :likelihood), v) == 0
            @test isempty(Base.invokelatest(prepare_query(b, p, :pointwise), v))
        end
    end
    # One program as rows and censored rows grow.
    shape(d) = structure(build_kernel(censored_plan(d)).spec)
    @test shape(d) == shape(censored_data(41; seed = 3))
    # Exact draws of `x` given the parameters and the latent values.
    q = prepare_marginalized_latents(built, plan)
    u = 0.3 .* cos.(1:built.layout.total)
    c = constrain(built.layout, u)
    (; T, measured, limit, lo, hi, order) = d.inputs
    value = vcat(measured, limit .- c.gap, lo .+ (hi .- lo) .* c.position)[order]
    lat = latent_moments(0.0, c.sigma^2 / (1 - c.phi^2), fill(c.phi, T), zeros(T),
        fill(c.sigma^2, T))
    mean, cov = latent_posterior(lat, c.mu, 1.0, c.noise, value)
    rng = Random.Xoshiro(4)
    n = 40_000
    draws = reduce(hcat, (marginalized_latent_draws(q, u, rng).x for _ in 1:n))
    @test all(abs.(vec(Statistics.mean(draws; dims = 2)) .- mean) .< 5 .* sqrt.(diag(cov) ./ n))
    @test Statistics.cov(draws; dims = 2) ≈ cov rtol = 0.05 atol = 0.02 * maximum(diag(cov))
end

# The measured rows conditioned (`| (; measured)`): their entries' one-step-
# ahead predictive densities are the likelihood, in the order of `measured`,
# and the censored rows' are prior factors.
censored_observed_plan(d) =
    marginalize(CENSORED(; Base.structdiff(d.inputs, NamedTuple{(:measured,)})...), :x) |
    (; d.inputs.measured)

@testset "conditioned measured rows are the likelihood of a censored series" begin
    d = censored_data(16)
    plan = censored_observed_plan(d)
    built = build_kernel(plan)
    # The same coordinates and density as with `measured` bound as an input.
    @test coordinate_names(built.layout) == coordinate_names(build_kernel(censored_plan(d)).layout)
    total = censored_oracle(built, d)
    (; T, measured, limit, lo, hi, order) = d.inputs
    function split(u)
        c = constrain(built.layout, u)
        value = vcat(measured, limit .- c.gap, lo .+ (hi .- lo) .* c.position)[order]
        lat = latent_moments(0.0, c.sigma^2 / (1 - c.phi^2), fill(c.phi, T), zeros(T),
            fill(c.sigma^2, T))
        mu, S = observation_moments(lat, c.mu, 1.0, c.noise)
        terms = sequential_logpdfs(mu, S, value)
        (; total = total(u), likelihood = sum(terms[d.seen]), pointwise = terms[d.seen])
    end
    check_total(built, plan, total)
    likelihood = prepare_query(built, plan, :likelihood)
    pointwise = prepare_query(built, plan, :pointwise)
    prior = prepare_query(built, plan, :prior)
    for u in (0.3 .* cos.(1:built.layout.total), 0.4 .* sin.(1:built.layout.total))
        expected = split(u)
        @test Base.invokelatest(likelihood, u) ≈ expected.likelihood rtol = 1e-10
        pw = Base.invokelatest(pointwise, u)
        @test keys(pw) == (:measured,)
        @test pw.measured ≈ expected.pointwise rtol = 1e-9 atol = 1e-11
        @test Base.invokelatest(prior, u) ≈
            expected.total - expected.likelihood - logjac(built.layout, u) rtol = 1e-10
    end
    # Without `marginalize`, the measured rows' Normal densities given `x`.
    latent = CENSORED(; Base.structdiff(d.inputs, NamedTuple{(:measured,)})...) |
        (; measured)
    lbuilt = build_kernel(latent)
    lu = 0.2 .* cos.(1:lbuilt.layout.total)
    c = constrain(lbuilt.layout, lu)
    terms = logpdf.(Normal.(c.mu .+ c.x[d.seen], c.noise), measured)
    @test Base.invokelatest(prepare_query(lbuilt, latent, :likelihood), lu) ≈ sum(terms) rtol = 1e-10
    @test Base.invokelatest(prepare_query(lbuilt, latent, :pointwise), lu).measured ≈ terms rtol = 1e-10
    # One program as rows and censored rows grow.
    shape(d) = structure(build_kernel(censored_observed_plan(d)).spec)
    @test shape(d) == shape(censored_data(41; seed = 3))
end

# ── Rows observing the state at their steps ───────────────────────────────
#
# Several sensors read one series at irregular steps: a step may have several
# rows or none. The filter runs over events (each step, then its rows), and
# each row's pointwise density is its predictive given the rows before it in
# that order.

const GATHERED = @rkppl begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    bias[1:S] .~ Normal.(0, 1)
    noise ~ Exponential(0.5)
    @scan begin
        x[1] ~ Normal(0, sigma / sqrt(1 - phi^2))
        for t in 2:T
            x[t] ~ Normal(phi * x[t - 1], sigma)
        end
    end
    y .~ Normal.(bias[sensor] .+ x[step], noise .* scale)
end

function gathered_data(T, N; S = 3, seed = 4)
    rng = Random.Xoshiro(seed)
    step = rand(rng, 1:T, N)
    step[1:2] .= 2                       # one step with several rows, for sure
    inputs = (; T, S, step, sensor = rand(rng, 1:S, N), scale = 0.5 .+ rand(rng, N))
    return (; inputs, y = randn(rng, N))
end

gathered_plan(d) = marginalize(GATHERED(; d.inputs...), :x) | (; d.y)

function gathered_moments(c, d)
    (; T, step, sensor, scale) = d.inputs
    lat = latent_moments(0.0, c.sigma^2 / (1 - c.phi^2), fill(c.phi, T), zeros(T),
        fill(c.sigma^2, T))
    L = [step[i] == t ? 1.0 : 0.0 for i in eachindex(step), t in 1:T]
    mu = c.bias[sensor] .+ L * lat[1]
    S = Symmetric(L * lat[2] * L' + Diagonal((c.noise .* scale) .^ 2))
    return lat, L, mu, S
end

function gathered_oracle(built, d)
    function (u)
        c = constrain(built.layout, u)
        prior = logpdf(Uniform(-1, 1), c.phi) + logpdf(Exponential(1), c.sigma) +
            sum(logpdf.(Normal(0, 1), c.bias)) + logpdf(Exponential(0.5), c.noise)
        _, _, mu, S = gathered_moments(c, d)
        p = sortperm(d.inputs.step)
        pointwise = zeros(length(d.y))
        pointwise[p] = sequential_logpdfs(mu[p], S[p, p], d.y[p])
        likelihood = logpdf(MvNormal(mu, S), d.y)
        (; total = prior + likelihood + logjac(built.layout, u), likelihood, pointwise)
    end
end

@testset "rows observe the state at their steps" begin
    d = gathered_data(7, 12)
    @test any(t -> !(t in d.inputs.step), 1:7)       # a step without rows
    plan = gathered_plan(d)
    built = build_kernel(plan)
    @test !any(n -> occursin(r"(^|_)x\.", String(n)), coordinate_names(built.layout))
    check_density(built, plan, gathered_oracle(built, d))
    # One program as steps, rows and sensors per step grow.
    shape(d) = structure(build_kernel(gathered_plan(d)).spec)
    @test shape(d) == shape(gathered_data(30, 75; seed = 9))
    # Exact draws: x given every row.
    q = prepare_marginalized_latents(built, plan)
    u = 0.3 .* cos.(1:built.layout.total)
    c = constrain(built.layout, u)
    lat, L, mu, S = gathered_moments(c, d)
    K = lat[2] * L' / S
    mean, cov = lat[1] + K * (d.y - mu), Symmetric(lat[2] - K * L * lat[2])
    rng = Random.Xoshiro(5)
    n = 40_000
    draws = reduce(hcat, (marginalized_latent_draws(q, u, rng).x for _ in 1:n))
    @test all(abs.(vec(Statistics.mean(draws; dims = 2)) .- mean) .< 5 .* sqrt.(diag(cov) ./ n))
    @test Statistics.cov(draws; dims = 2) ≈ cov rtol = 0.05 atol = 0.02 * maximum(diag(cov))
    # Rows do not say how many steps there are; missing rows are not built.
    unsized = try
        (; S, step, sensor, scale) = d.inputs
        marginalize(GATHERED(; S, step, sensor, scale), :x) | (; d.y)
        "no error"
    catch e
        sprint(showerror, e)
    end
    @test occursin("the length `T` of its `@scan` is not supplied", unsized)
    ym = Vector{Union{Missing,Float64}}(d.y)
    ym[3] = missing
    @test occursin("without missing entries", try
        marginalize(GATHERED(; d.inputs...), :x) | (; y = ym)
        "no error"
    catch e
        sprint(showerror, e)
    end)
end

# ── A vector state ────────────────────────────────────────────────────────
#
# A scan carrying several arrays, or reading one at a lag above one, filters
# its window as one state. The oracle pushes unit vectors through the stated
# recurrence: the latents are an affine map of independent standard normals.

function affine_moments(rec, nz)
    base = rec(zeros(nz))
    M = reduce(hcat, (rec(Float64.(1:nz .== k)) .- base for k in 1:nz))
    return base, M * M'
end

# The density of `y ~ Normal(offset(c) + H(c) * latent, noise)` for a latent
# stack `rec(c)(z)`; `latent` holds its moments for the draws.
function vector_oracle(built, y, nz, rec, H, offset, prior)
    function (u)
        c = constrain(built.layout, u)
        m, S = affine_moments(rec(c), nz)
        Hc = H(c)
        mu = offset(c) .+ Hc * m
        Sy = Symmetric(Hc * S * Hc' + c.noise^2 * I)
        (; total = prior(c) + logpdf(MvNormal(mu, Sy), y) + logjac(built.layout, u),
            likelihood = logpdf(MvNormal(mu, Sy), y),
            pointwise = sequential_logpdfs(mu, Sy, y), latent = (m, S), noise = c.noise)
    end
end

# A local linear trend: the observation reads the level.
const TREND = @rkppl begin
    s_level ~ Exponential(1)
    s_slope ~ Exponential(0.5)
    noise ~ Exponential(1)
    @scan begin
        level[1] ~ Normal(0, 2)
        slope[1] ~ Normal(0, 1)
        for t in 2:T
            level[t] ~ Normal(level[t - 1] + slope[t - 1], s_level)
            slope[t] ~ Normal(slope[t - 1], s_slope)
        end
    end
    y .~ Normal.(level, noise)
end
trend_rec(c, T) = z -> begin
    level, slope = zeros(T), zeros(T)
    level[1], slope[1] = 2 * z[1], z[2]
    for t in 2:T
        level[t] = level[t - 1] + slope[t - 1] + c.s_level * z[2t - 1]
        slope[t] = slope[t - 1] + c.s_slope * z[2t]
    end
    vcat(level, slope)
end
trend_prior(c) = logpdf(Exponential(1), c.s_level) + logpdf(Exponential(0.5), c.s_slope) +
    logpdf(Exponential(1), c.noise)

# An AR(2) seeded at two entries, the second reading the first.
const AR2 = @rkppl begin
    phi1 ~ Uniform(-1, 1)
    phi2 ~ Uniform(-0.5, 0.5)
    sigma ~ Exponential(1)
    noise ~ Exponential(1)
    @scan begin
        x[1] ~ Normal(0, 1)
        x[2] ~ Normal(phi1 * x[1], sigma)
        for t in 3:T
            x[t] ~ Normal(phi1 * x[t - 1] + phi2 * x[t - 2], sigma)
        end
    end
    y .~ Normal.(0.5 .+ x, noise)
end
ar2_rec(c, T) = z -> begin
    x = zeros(T)
    x[1] = z[1]
    x[2] = c.phi1 * x[1] + c.sigma * z[2]
    for t in 3:T
        x[t] = c.phi1 * x[t - 1] + c.phi2 * x[t - 2] + c.sigma * z[t]
    end
    x
end
ar2_prior(c) = logpdf(Uniform(-1, 1), c.phi1) + logpdf(Uniform(-0.5, 0.5), c.phi2) +
    logpdf(Exponential(1), c.sigma) + logpdf(Exponential(1), c.noise)

# Non-centred, seeded at two entries per array, a deterministic array: an
# AR(2) cycle `c` read in the same step by a level, and a clock `v` the
# observation reads with a per-entry loading.
const CYCLE = @rkppl begin
    phi1 ~ Uniform(-1, 1)
    phi2 ~ Uniform(-0.5, 0.5)
    sigma ~ Exponential(1)
    tau ~ Exponential(1)
    b ~ Normal(0, 1)
    noise ~ Exponential(1)
    @scan begin
        c[1] ~ Normal(0, 1)
        level[1] ~ Normal(c[1], 1)
        v[1] = 0.0
        c[2] ~ Normal(phi1 * c[1], sigma)
        level[2] ~ Normal(level[1] + c[2], tau)
        v[2] = 1.0
        for t in 3:T
            e ~ Normal(0, 1)
            c[t] = phi1 * c[t - 1] + phi2 * c[t - 2] + sigma * e
            level[t] ~ Normal(level[t - 1] + c[t], tau)
            v[t] = v[t - 1] + 1
        end
    end
    y .~ Normal.(level .+ b .* w .* v, noise)
end
cycle_rec(p, T) = z -> begin
    c, level, v = zeros(T), zeros(T), collect(0.0:(T - 1))
    c[1] = z[1]
    level[1] = c[1] + z[2]
    c[2] = p.phi1 * c[1] + p.sigma * z[3]
    level[2] = level[1] + c[2] + p.tau * z[4]
    for t in 3:T
        c[t] = p.phi1 * c[t - 1] + p.phi2 * c[t - 2] + p.sigma * z[2t - 1]
        level[t] = level[t - 1] + c[t] + p.tau * z[2t]
    end
    vcat(c, level, v)
end
cycle_prior(p) = logpdf(Uniform(-1, 1), p.phi1) + logpdf(Uniform(-0.5, 0.5), p.phi2) +
    logpdf(Exponential(1), p.sigma) + logpdf(Exponential(1), p.tau) +
    logpdf(Normal(0, 1), p.b) + logpdf(Exponential(1), p.noise)

# One draw read by two rows (snag rkppl-marginaliz-a49e44e4): a process at
# the start `x0` and the end `x1` of each row's interval, `x1` reading `x0`'s
# draw of the same seed and step, every scale compound. Each row names its own
# share of an innovation: `x0`'s innovation reaches `x1`'s row as `phi * …`.
const SHARED = @rkppl begin
    sd ~ Exponential(1)
    range ~ LogNormal(0, 1)
    noise ~ Exponential(1)
    phi = exp(-1 / range)
    beta = (1 - phi) * range / (1 + phi)
    @scan begin
        x0[1] ~ Normal(0, 2 * sd)
        x1[1] ~ Normal(phi * x0[1], sd * sqrt(1 - phi * phi))
        for t in 2:T
            rho = exp(-gap[t] / range)
            x0[t] ~ Normal(rho * x1[t - 1], sd * sqrt(1 - rho * rho))
            x1[t] ~ Normal(phi * x0[t], sd * sqrt(1 - phi * phi))
        end
    end
    y .~ Normal.(beta .* x0 .+ beta .* x1, noise)
end
shared_rec(c, T, gap) = z -> begin
    phi = exp(-1 / c.range)
    x0, x1 = zeros(T), zeros(T)
    x0[1] = 2 * c.sd * z[1]
    x1[1] = phi * x0[1] + c.sd * sqrt(1 - phi^2) * z[2]
    for t in 2:T
        rho = exp(-gap[t] / c.range)
        x0[t] = rho * x1[t - 1] + c.sd * sqrt(1 - rho^2) * z[2t - 1]
        x1[t] = phi * x0[t] + c.sd * sqrt(1 - phi^2) * z[2t]
    end
    vcat(x0, x1)
end
shared_beta(c) = (phi = exp(-1 / c.range); (1 - phi) * c.range / (1 + phi))
shared_prior(c) = logpdf(Exponential(1), c.sd) + logpdf(LogNormal(0, 1), c.range) +
    logpdf(Exponential(1), c.noise)

function vector_cases(T; seed = 11)
    rng = Random.Xoshiro(seed)
    y = randn(rng, T)
    w = collect(range(0.1, 1; length = T))
    gap = collect(range(0.5, 2; length = T))
    E = Matrix(1.0I, T, T)
    return (;
        trend = (plan = marginalize(TREND(; T), :level, :slope) | (; y), y, nz = 2T,
            rec = c -> trend_rec(c, T), H = c -> [E zeros(T, T)], offset = c -> 0.0,
            prior = trend_prior, keys = (:level, :slope)),
        ar2 = (plan = marginalize(AR2(; T), :x) | (; y), y, nz = T,
            rec = c -> ar2_rec(c, T), H = c -> E, offset = c -> 0.5, prior = ar2_prior,
            keys = (:x,)),
        cycle = (plan = marginalize(CYCLE(; T, w), :c, :level, :v) | (; y), y, nz = 2T,
            rec = c -> cycle_rec(c, T), H = c -> [zeros(T, T) E Diagonal(c.b .* w)],
            offset = c -> 0.0, prior = cycle_prior, keys = (:c, :level, :v)),
        shared = (plan = marginalize(SHARED(; T, gap), :x0, :x1) | (; y), y, nz = 2T,
            rec = c -> shared_rec(c, T, gap), H = c -> shared_beta(c) .* [E E],
            offset = c -> 0.0, prior = shared_prior, keys = (:x0, :x1)))
end

@testset "a vector state integrates out as one filter" begin
    for (name, case) in pairs(vector_cases(8))
        built = build_kernel(case.plan)
        @test !any(n -> any(k -> occursin(Regex("^$k\\."), String(n)), case.keys),
            coordinate_names(built.layout))
        oracle = vector_oracle(built, case.y, case.nz, case.rec, case.H, case.offset,
            case.prior)
        check_density(built, case.plan, oracle)
        # Exact draws: every array given every observation.
        q = prepare_marginalized_latents(built, case.plan)
        u = 0.3 .* cos.(1:built.layout.total)
        o = oracle(u)
        c = constrain(built.layout, u)
        H = case.H(c)
        m, S = o.latent
        K = S * H' / Symmetric(H * S * H' + o.noise^2 * I)
        mean = m + K * (case.y - (case.offset(c) .+ H * m))
        cov = Symmetric(S - K * H * S)
        rng = Random.Xoshiro(5)
        n = 40_000
        draws = [marginalized_latent_draws(q, u, rng) for _ in 1:n]
        @test propertynames(first(draws)) == case.keys
        X = reduce(hcat, (vcat((d[k] for k in case.keys)...) for d in draws))
        @test all(abs.(vec(Statistics.mean(X; dims = 2)) .- mean) .<
            5 .* sqrt.(max.(diag(cov), 0) ./ n) .+ 1e-9)
        @test Statistics.cov(X; dims = 2) ≈ cov rtol = 0.05 atol = 0.02 * maximum(diag(cov))
    end
    # One program as the series grows.
    shape(T) = map(c -> structure(build_kernel(c.plan).spec), vector_cases(T))
    @test shape(5) == shape(40)
    # Refused: a subset of the arrays, and a vector state observed otherwise.
    d = vector_cases(8)
    named(model, names...; data...) = try
        build_kernel(marginalize(model, names...) | (; data...))
        "no error"
    catch e
        sprint(showerror, e)
    end
    w = collect(range(0.1, 1; length = 8))
    @test occursin("also carries `level`, `v`; a recurrence carrying several arrays",
        named(CYCLE(; T = 8, w), :c; y = d.cycle.y))
    gaps = Vector{Union{Missing,Float64}}(d.trend.y)
    gaps[2] = missing
    @test occursin("admits `missing` entries", named(TREND(; T = 8), :level, :slope; y = gaps))
    gathered = RKPPLModel(quote
        s ~ Exponential(1)
        @scan begin
            level[1] ~ Normal(0, 1)
            slope[1] ~ Normal(0, 1)
            for t in 2:T
                level[t] ~ Normal(level[t - 1] + slope[t - 1], s)
                slope[t] ~ Normal(slope[t - 1], s)
            end
        end
        y .~ Normal.(level[step], 1)
    end, @__MODULE__)
    @test occursin("gathers entries of a carried array",
        named(gathered(; T = 8, step = [1, 2, 2, 5]), :level, :slope; y = [0.1, 0.2, 0.3, 0.4]))
    defined = RKPPLModel(quote
        s ~ Exponential(1)
        @scan begin
            level[1] ~ Normal(0, 1)
            slope[1] ~ Normal(0, 1)
            for t in 2:T
                level[t] ~ Normal(level[t - 1] + slope[t - 1], s)
                slope[t] ~ Normal(slope[t - 1], s)
            end
        end
        v = vcat(m, m)
        v .~ Normal.(level, 1)
    end, @__MODULE__)
    unbound = try
        build_kernel(bind_data(marginalize(defined(; T = 4, m = [0.1, 0.2]), :level, :slope)))
        "no error"
    catch e
        sprint(showerror, e)
    end
    @test occursin("its observation `v` is not bound data", unbound)
    # The draws read a group's layout back from its generated names.
    dunder = RKPPLModel(quote
        s ~ Exponential(1)
        @scan begin
            a__b[1] ~ Normal(0, 1)
            c[1] ~ Normal(0, 1)
            for t in 2:T
                a__b[t] ~ Normal(a__b[t - 1], s)
                c[t] ~ Normal(c[t - 1], s)
            end
        end
        y .~ Normal.(a__b .+ c, 1)
    end, @__MODULE__)
    @test occursin("must not hold `__`", named(dunder(; T = 8), :a__b, :c; y = d.trend.y))
end

# ── A recurrence held in a submodel ───────────────────────────────────────
#
# A submodel's `@scan` integrates out by its author path (`z.x`); the program
# is the one its top-level spelling lowers to, and the draws nest under the
# call (`draws.z.x`), as `constrain` nests sampled locals.

@rkppl submodel_walk(s) = begin
    @scan begin
        x[1] ~ Normal(0, 1)
        for t in 2:T
            x[t] ~ Normal(x[t - 1], s)
        end
    end
    x
end

@rkppl submodel_trend(s_level, s_slope) = begin
    @scan begin
        level[1] ~ Normal(0, 2)
        slope[1] ~ Normal(0, 1)
        for t in 2:T
            level[t] ~ Normal(level[t - 1] + slope[t - 1], s_level)
            slope[t] ~ Normal(slope[t - 1], s_slope)
        end
    end
    level
end

const SUB_WALK = @rkppl begin
    s ~ Exponential(1)
    noise ~ Exponential(1)
    z ~ submodel_walk(s)
    y .~ Normal.(z, noise)
end
const TOP_WALK = @rkppl begin
    s ~ Exponential(1)
    noise ~ Exponential(1)
    @scan begin
        x[1] ~ Normal(0, 1)
        for t in 2:T
            x[t] ~ Normal(x[t - 1], s)
        end
    end
    y .~ Normal.(x, noise)
end
const SUB_TREND = @rkppl begin
    s_level ~ Exponential(1)
    s_slope ~ Exponential(0.5)
    noise ~ Exponential(1)
    z ~ submodel_trend(s_level, s_slope)
    y .~ Normal.(z, noise)
end

submodel_plans(T; y = randn(Random.Xoshiro(2), T)) = (;
    walk = (marginalize(SUB_WALK(; T), Symbol("z.x")) | (; y),
        marginalize(TOP_WALK(; T), :x) | (; y), (:x,)),
    trend = (marginalize(SUB_TREND(; T), Symbol("z.level"), Symbol("z.slope")) | (; y),
        marginalize(TREND(; T), :level, :slope) | (; y), (:level, :slope)))

@testset "a recurrence held in a submodel integrates out by its path" begin
    for (name, (sub, top, keys)) in pairs(submodel_plans(7))
        bs, bt = build_kernel(sub), build_kernel(top)
        @test coordinate_names(bs.layout) == coordinate_names(bt.layout)
        u = 0.3 .* cos.(1:bs.layout.total)
        @test Base.invokelatest(prepare_query(bs, sub, :sampler), u) ==
            Base.invokelatest(prepare_query(bt, top, :sampler), u)
        sampler(b, p) = prepare_sampler(b, p, zeros(b.layout.total);
            backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
        gs = sampler_value_and_gradient!(sampler(bs, sub), zeros(length(u)), u)
        gt = sampler_value_and_gradient!(sampler(bt, top), zeros(length(u)), u)
        @test gs[2] ≈ gt[2] rtol = 1e-12
        ds = marginalized_latent_draws(prepare_marginalized_latents(bs, sub), u, Random.Xoshiro(4))
        dt = marginalized_latent_draws(prepare_marginalized_latents(bt, top), u, Random.Xoshiro(4))
        @test propertynames(ds) == (:z,) && propertynames(ds.z) == keys
        @test all(ds.z[k] == dt[k] for k in keys)
        @test structure(bs.spec) == structure(build_kernel(submodel_plans(30)[name][1]).spec)
    end
    @test occursin("`z.q` names no local of a submodel call", try
        build_kernel(marginalize(SUB_WALK(; T = 4), Symbol("z.q")) | (; y = zeros(4)))
        "no error"
    catch e
        sprint(showerror, e)
    end)
end

# Each form the Kalman filter does not cover is refused at lowering, naming
# the statement.
function refusal(body, x, data, y)
    m = RKPPLModel(body, @__MODULE__)
    try
        marginalize(m(; data...), x) | (; y)
    catch e
        e isa SurfaceLoweringError || rethrow()
        return sprint(showerror, e)
    end
    return "no error"
end

@testset "forms that are not linear-Gaussian are refused" begin
    y, data = [0.1, -0.2, 0.4, 0.3], (; T = 4)
    statements(ex) = Meta.isexpr(ex, :block) ?
        Any[a for a in ex.args if !(a isa LineNumberNode)] : Any[ex]
    scan(step; seed = :(x[1] ~ Normal(0, 1))) = quote
        s ~ Exponential(1)
        @scan begin
            $seed
            for t in 2:T
                $(statements(step)...)
            end
        end
    end
    observed(body, obs = :(y .~ Normal.(x, 1))) =
        Expr(:block, statements(body)..., statements(obs)...)
    for (step, expected) in (
            (quote; e ~ StudentT(3); x[t] = x[t - 1] + s * e; end, "not an unbounded Normal draw"),
            (:(x[t] = tanh(x[t - 1]) + s), "is not linear"),
            (:(x[t] ~ Normal(x[t - 1], exp(x[t - 1]))), "reads `x` or an innovation"),
            (quote; e ~ Normal(0, 1); x[t] = x[t - 1] * e; end, "is not linear"),
            (:(x[t] ~ truncated(Normal(x[t - 1], s), 0, Inf)), "not an unbounded Normal draw"))
        msg = refusal(observed(scan(step)), :x, data, y)
        @test occursin("marginalize(`x`)", msg)
        @test occursin(expected, msg)
    end
    # A second carried array left out, and a seed that is not Normal.
    two = quote
        @scan begin
            x[1] ~ Normal(0, 1)
            v[1] = 0.0
            for t in 2:T
                v[t] = v[t - 1] + 1
                x[t] ~ Normal(x[t - 1], 1)
            end
        end
        y .~ Normal.(x, 1)
    end
    @test occursin("also carries `v`", refusal(two, :x, data, y))
    @test occursin("seed `x[1]` is not an unbounded Normal draw", refusal(observed(
        scan(:(x[t] ~ Normal(x[t - 1], 1)); seed = :(x[1] ~ Exponential(1)))), :x, data, y))
    # Reads other than one Normal observation.
    base = scan(:(x[t] ~ Normal(x[t - 1], s)))
    for (obs, expected) in (
            (:(y .~ Normal.(exp.(x), 1)), "it is read by"),
            (:(y .~ Normal.(x, exp.(x))), "it is read by"),
            (:(y .~ Laplace.(x, 1)), "it is read by"),
            (quote; y .~ Normal.(x, 1); m = sum(x); end, "every read of `x`"),
            (quote; mu = 2 .* x; y .~ Normal.(mu, 1); z = mu .+ 1; end, "must be read by its observation alone"))
        msg = refusal(observed(base, obs), :x, data, y)
        @test occursin(expected, msg)
    end
    @test occursin("no top-level `@scan` and no `@plate` cell `@scan` carries `q`",
        refusal(observed(base), :q, data, y))
    # The observation must be bound data.
    unobserved = try
        bind_data(marginalize(RKPPLModel(observed(base), @__MODULE__)(; T = 4), :x))
        "no error"
    catch e
        sprint(showerror, e)
    end
    @test occursin("is neither observed data nor a definition", unobserved)
    # A defined observed vector does not say the scan's length: it is supplied.
    unsized = try
        bind_data(marginalize(RKPPLModel(quote
            s ~ Exponential(1)
            g[1:2] .~ restricted.(Flat.(), 0, Inf)
            @scan begin
                x[1] ~ Normal(0, 1)
                for t in 2:T
                    x[t] ~ Normal(x[t - 1], s)
                end
            end
            v = vcat(m, 1 .- g)
            v .~ Normal.(x, 1)
        end, @__MODULE__)(; m = [0.1, 0.2]), :x))
        "no error"
    catch e
        sprint(showerror, e)
    end
    @test occursin("the length `T` of its `@scan` is not supplied", unsized)
    # A plan is already lowered.
    d = ou_data(1, 3)
    @test_throws SurfaceLoweringError marginalize(OU(; d.inputs...) | (; y = d.y), :ou)
    @test_throws ArgumentError marginalize(OU)
end

# ── One recurrence per `@plate` index ─────────────────────────────────────
#
# The groups are independent given the parameters, so the oracle is the sum
# of each group's dense joint Gaussian, and the pointwise densities are each
# group's sequential conditionals.

# Per-group OU deviations in the cell (rkppl-use §5 "A recurrence per
# group"): irregular gaps per group, a per-group level, a per-entry covariate
# in the offset, and the latent returned through a per-index output.
const CELL_OU = @rkppl begin
    noise ~ Exponential(0.3)
    ou_sd ~ Exponential(0.5)
    ou_range ~ LogNormal(1, 0.5)
    b ~ Normal(0, 1)
    level[1:G] .~ Normal.(0, 1)
    @plate for g in eachindex(y)
        @scan begin
            ou[1] ~ Normal(0, 1)
            for t in 2:length(y[g])
                e ~ Normal(0, 1)
                r = exp(-gap[g][t] / ou_range)
                ou[t] = r * ou[t - 1] + sqrt(1 - r * r) * e
            end
        end
        loc[g] = level[g] .+ b .* x[g] .+ ou_sd .* ou
        y[g] .~ Normal.(loc[g], noise)
    end
end

# Centred AR(1) per group: the seed reads per-group data, the loading is a
# per-group parameter and the scale is per entry.
const CELL_AR = @rkppl begin
    phi ~ Uniform(-1, 1)
    drift ~ Normal(0, 0.5)
    sigma ~ Exponential(0.5)
    noise ~ Exponential(0.5)
    lam[1:G] .~ LogNormal.(0, 0.3)
    @plate for g in eachindex(y)
        @scan begin
            h[1] ~ Normal(m0[g], 1.5)
            for t in 2:length(y[g])
                h[t] ~ Normal(phi * h[t - 1] + drift, sigma)
            end
        end
        y[g] .~ Normal.(lam[g] .* h, sqrt.(noise^2 .+ se[g] .^ 2))
    end
end

function cell_data(lengths; seed = 1)
    rng = Random.Xoshiro(seed)
    G = length(lengths)
    y = [cumsum(0.4 .* randn(rng, n)) for n in lengths]
    gap = [0.3 .+ rand(rng, n) for n in lengths]
    x = [randn(rng, n) for n in lengths]
    se = [0.1 .+ 0.2 .* rand(rng, n) for n in lengths]
    m0 = randn(rng, G)
    return (; G, y, gap, x, se, m0)
end

# The oracle over groups: `group(c, g)` gives each group's latent moments and
# its observation's offset, loading and noise.
function cell_oracle(built, prior, group, y)
    function (u)
        c = constrain(built.layout, u)
        likelihood, pointwise = 0.0, Vector{Float64}[]
        for g in eachindex(y)
            lat, offset, loading, noise = group(c, g)
            mu, S = observation_moments(lat, offset, loading, noise)
            likelihood += logpdf(MvNormal(mu, S), y[g])
            push!(pointwise, sequential_logpdfs(mu, S, y[g]))
        end
        (; total = prior(c) + likelihood + logjac(built.layout, u), likelihood, pointwise)
    end
end

cell_ou_plan(d) = marginalize(CELL_OU(; d.G, d.gap, d.x), :ou) | (; d.y)
function cell_ou_oracle(built, d)
    prior(c) = logpdf(Exponential(0.3), c.noise) + logpdf(Exponential(0.5), c.ou_sd) +
        logpdf(LogNormal(1, 0.5), c.ou_range) + logpdf(Normal(0, 1), c.b) +
        sum(logpdf.(Normal(0, 1), c.level))
    function group(c, g)
        n = length(d.y[g])
        a = exp.(-d.gap[g] ./ c.ou_range)
        lat = latent_moments(0.0, 1.0, a, zeros(n), 1 .- a .^ 2)
        return lat, c.level[g] .+ c.b .* d.x[g], c.ou_sd, c.noise
    end
    return cell_oracle(built, prior, group, d.y)
end

@testset "a recurrence per @plate index integrates out exactly" begin
    for lengths in ([1], [3, 1], [2, 5, 4], [6, 1, 3, 8])
        d = cell_data(lengths; seed = sum(lengths))
        plan = cell_ou_plan(d)
        built = build_kernel(plan)
        @test coordinate_names(built.layout) == [:noise, :ou_sd, :ou_range, :b,
            (Symbol("level.$i") for i in 1:d.G)...]
        check_density(built, plan, cell_ou_oracle(built, d))

        plan = marginalize(CELL_AR(; d.G, d.se, d.m0), :h) | (; d.y)
        built = build_kernel(plan)
        @test coordinate_names(built.layout) == [:phi, :drift, :sigma, :noise,
            (Symbol("lam.$i") for i in 1:d.G)...]
        check_density(built, plan, cell_oracle(built,
            c -> logpdf(Uniform(-1, 1), c.phi) + logpdf(Normal(0, 0.5), c.drift) +
                logpdf(Exponential(0.5), c.sigma) + logpdf(Exponential(0.5), c.noise) +
                sum(logpdf.(LogNormal(0, 0.3), c.lam)),
            function (c, g)
                n = length(d.y[g])
                lat = latent_moments(d.m0[g], 1.5^2, fill(c.phi, n), fill(c.drift, n),
                    fill(c.sigma^2, n))
                return lat, 0.0, c.lam[g], sqrt.(c.noise^2 .+ d.se[g] .^ 2)
            end, d.y))
    end
    # The model as written still samples each group's seed and innovations.
    d = cell_data([3, 2])
    @test build_kernel(CELL_OU(; d.G, d.gap, d.x) | (; d.y)).layout.total == 4 + 2 + 5
end

@testset "the cell filter is one retained scan in one plate as groups grow" begin
    shape(lengths) = (d = cell_data(lengths); structure(build_kernel(cell_ou_plan(d)).spec))
    small = shape([2, 3])
    @test small == shape([7, 40, 1, 12, 9])
end

@testset "exact draws of each index's integrated-out trajectory" begin
    d = cell_data([4, 1, 6]; seed = 9)
    plan = cell_ou_plan(d)
    built = build_kernel(plan)
    q = prepare_marginalized_latents(built, plan)
    u = 0.3 .* cos.(1:built.layout.total)
    c = constrain(built.layout, u)
    filtered = q(u)
    @test keys(filtered) == (:ou,)
    @test length(filtered.ou.mean) == d.G
    rng = Random.Xoshiro(4)
    n = 40_000
    draws = [marginalized_latent_draws(q, u, rng).ou for _ in 1:n]
    for g in 1:d.G
        a = exp.(-d.gap[g] ./ c.ou_range)
        @test filtered.ou.coef[g][2:end] ≈ a[2:end]
        lat = latent_moments(0.0, 1.0, a, zeros(length(a)), 1 .- a .^ 2)
        mean, cov = latent_posterior(lat, c.level[g] .+ c.b .* d.x[g], c.ou_sd, c.noise, d.y[g])
        @test filtered.ou.mean[g][end] ≈ mean[end]
        @test filtered.ou.var[g][end] ≈ cov[end, end]
        G = reduce(hcat, (draw[g] for draw in draws))
        @test size(G) == (length(d.y[g]), n)
        @test all(abs.(vec(Statistics.mean(G; dims = 2)) .- mean) .< 5 .* sqrt.(diag(cov) ./ n))
        @test Statistics.cov(G; dims = 2) ≈ cov rtol = 0.05 atol = 0.02 * maximum(diag(cov))
    end
    @test marginalized_latent_draws(q, u, Random.Xoshiro(3)) ==
        marginalized_latent_draws(q, u, Random.Xoshiro(3))
end

@testset "cell forms the filter does not cover are refused" begin
    y = [[0.1, -0.2, 0.4], [0.3, 0.2]]
    cellmodel(cell; before = :(s ~ Exponential(1))) = quote
        $before
        @plate for g in eachindex(y)
            @scan begin
                h[1] ~ Normal(0, 1)
                for t in 2:length(y[g])
                    h[t] ~ Normal(h[t - 1], s)
                end
            end
            $(cell.args...)
        end
    end
    # The per-index output is read beside the observation.
    msg = refusal(Expr(:block, cellmodel(quote
        loc[g] = h
        y[g] .~ Normal.(loc[g], 1)
    end).args..., :(total = sum(sum, loc))), :h, (;), y)
    @test occursin("must be read by its observation alone", msg)
    # A per-index trajectory observed entry by entry is a different model.
    @test occursin("every read of `h`", refusal(cellmodel(quote
        y[g] .~ Normal.(h, 1)
        z[g] = h[1]
    end), :h, (;), y))
    # A loading drawn inside the cell is not in the draws' plate.
    @test occursin("drawn inside the `@plate` cell", refusal(cellmodel(quote
        w ~ Normal(0, 1)
        y[g] .~ Normal.(w .* h, 1)
    end), :h, (;), y))
    # A whole program vector read in a cell is not placed.
    @test occursin("cannot tell whether", refusal(cellmodel(quote
        y[g] .~ Normal.(v .+ h, 1)
    end; before = quote
        s ~ Exponential(1)
        v[1:3] .~ Normal.(0, 1)
    end), :h, (;), y))
    # A missing entry in a cell's series is not a predict-only step yet.
    @test occursin("missing entry in a `@plate` cell", refusal(cellmodel(quote
        y[g] .~ Normal.(h, 1)
    end), :h, (;), [Union{Missing,Float64}[0.1, missing, 0.4], [0.3, 0.2]]))
    # A vector state is filtered at top level only.
    @test occursin("a vector state in a `@plate` cell is not marginalized yet", refusal(quote
        s ~ Exponential(1)
        @plate for g in eachindex(y)
            @scan begin
                h[1] ~ Normal(0, 1)
                k[1] ~ Normal(0, 1)
                for t in 2:length(y[g])
                    h[t] ~ Normal(h[t - 1] + k[t - 1], s)
                    k[t] ~ Normal(k[t - 1], s)
                end
            end
            y[g] .~ Normal.(h, 1)
        end
    end, :h, (;), y))
    # Rows observing a cell's state at their steps are not an event filter yet.
    @test occursin("gathered rows in a `@plate` cell", refusal(cellmodel(quote
        y[g] .~ Normal.(h[step], 1)
    end), :h, (; step = [1, 2, 3]), y))
end

# ── `jax_source` of a marginalized program ────────────────────────────────
#
# The module is standalone: every data definition, including the observed
# scan's length check, is translated (`UNTRANSLATED = {}`). Its log density
# and `jax.grad` match the native sampler and its finite differences.

include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))
include("jax_ppl_data.jl")

# An observed AR(1) series whose length is the response's own: the check
# without a supplied length.
const OBSERVED_AR = @rkppl begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    @scan begin
        y[1] ~ Normal(0, sigma / sqrt(1 - phi^2))
        for t in 2:T
            y[t] ~ Normal(phi * y[t - 1], sigma)
        end
    end
end

@testset "jax_source of a marginalized program" begin
    d, c = ou_data(2, 4), cell_data([3, 1, 5])
    gaps = Vector{Union{Missing,Float64}}(d.y)
    gaps[[1, 3]] .= missing
    plans = (; toplevel = ou_plan(d), cell = cell_ou_plan(c),
        missing_entries = marginalize(OU(; d.inputs...), :ou) | (; y = gaps),
        censored = bind_data(censored_plan(censored_data(9))),
        censored_observed = censored_observed_plan(censored_data(9)),
        gathered = gathered_plan(gathered_data(5, 8)),
        trend = vector_cases(6).trend.plan, ar2 = vector_cases(6).ar2.plan,
        shared = vector_cases(6).shared.plan,
        submodel = submodel_plans(6).trend[1],
        cell_ar = marginalize(CELL_AR(; c.G, c.se, c.m0), :h) | (; c.y),
        observed = OBSERVED_AR() | (; y = [0.1, -0.3, 0.4, 0.2]))
    for (name, plan) in pairs(plans)
        built = build_kernel(plan)
        src = jax_source(built, plan)
        @test occursin("UNTRANSLATED = {}", string(src))
        if !rk_jax_available()
            @info "RK_JAX_PYTHON is not set: JAX parity for marginalized $name skipped"
            @test_skip false
            continue
        end
        query = prepare_query(built, plan, :sampler)
        u = 0.2 .* randn(Random.Xoshiro(7), built.layout.total)
        r = rk_jax_evaluate(src, "log_density", (u, jax_raw_data(plan, src)...);
            grad = true, prepare = true)
        @test r.value ≈ Base.invokelatest(query, u) rtol = 1e-10
        reference = rk_fd_gradient(u -> Base.invokelatest(query, u), u)
        @test maximum(abs.(r.grad .- reference) ./ max.(1, abs.(reference))) < 1e-5
    end
end

@testset "marginalize composes with binding, observing and merge" begin
    d = ou_data(2, 4)
    reference = ou_plan(d)
    built = build_kernel(reference)
    u = 0.2 .* sin.(1:built.layout.total)
    density(plan) = Base.invokelatest(prepare_query(build_kernel(plan), plan, :likelihood), u)
    expected = density(reference)
    @test density(marginalize(OU, :ou)(; d.inputs...) | (; y = d.y)) == expected
    @test density(marginalize(OU | (; y = d.y), :ou)(; d.inputs...)) == expected
    # A merge keeps the marginalization; the merged prior changes the prior only.
    merged = Base.merge(marginalize(OU, :ou), :(noise ~ Exponential(0.5)))
    plan = merged(; d.inputs...) | (; y = d.y)
    @test coordinate_names(build_kernel(plan).layout) == coordinate_names(built.layout)
    @test density(plan) == expected
    @test_throws SurfaceLoweringError marginalize(marginalize(OU, :ou), :ou)
end

# ── Named filter values ───────────────────────────────────────────────────
#
# `marginalize(…; predictive, filtered)` names each observed entry's
# one-step-ahead predictive law and the filtered state moments as ordinary
# values of the program (snag `marginalize-the-5a1c5f81`: a consumer moving a
# hand-written Kalman filter onto `marginalize` keeps its `mu` /
# `predictive_sd`). Oracles: the joint Gaussians above, the location and
# variance of each sequential conditional, and Gaussian conditioning of the
# latent on the entries up to each step.

# Each entry's predictive location and variance given the present entries
# before it, from the joint Gaussian `(mu, S)` of `y`.
function sequential_moments(mu, S, y; present = eachindex(y))
    loc, var = zeros(length(y)), zeros(length(y))
    for t in eachindex(y)
        p = [s for s in present if s < t]
        if isempty(p)
            loc[t], var[t] = mu[t], S[t, t]
        else
            w = S[p, p] \ S[p, t]
            loc[t] = mu[t] + w' * (y[p] - mu[p])
            var[t] = S[t, t] - S[t, p]' * w
        end
    end
    return loc, var
end

# The latent stack `(m, S)` given `y[P]`, observed as `offset + H * stack`
# plus noise of per-entry variance `nv`.
function condition_stack(m, S, H, offset, nv, y, P)
    isempty(P) && return m, Matrix(S)
    n = size(H, 1)
    Hp = H[P, :]
    Sy = Symmetric(Hp * S * Hp' + Diagonal((nv .* ones(n))[P]))
    K = S * Hp' / Sy
    return m + K * (y[P] - ((offset .* ones(n))[P] + Hp * m)), Matrix(S - K * Hp * S)
end

# Per step `t`, the mean and variance of stack entry `index(t)` given the
# entries `observed(t)`.
function filtered_moments(m, S, H, offset, nv, y, T; index = identity, observed = t -> 1:t)
    post = [condition_stack(m, S, H, offset, nv, y, collect(Int, observed(t))) for t in 1:T]
    return [p[1][index(t)] for (t, p) in enumerate(post)],
        [p[2][index(t), index(t)] for (t, p) in enumerate(post)]
end

named_values(built, plan, names, u) = Base.invokelatest(prepare_query(built, plan, names), u)
# The density queries of two plans agree exactly at `u`.
same_density(b1, p1, b2, p2, u) = all(preset -> isequal(
    Base.invokelatest(prepare_query(b1, p1, preset), u),
    Base.invokelatest(prepare_query(b2, p2, preset), u)), (:sampler, :likelihood, :pointwise))

const OU_NAMED = (predictive = (location = :pred_loc, variance = :pred_var, scale = :pred_sd),
    filtered = (mean = :ou_mean, variance = :ou_var))
const OU_NAMES = (:pred_loc, :pred_var, :pred_sd, :ou_mean, :ou_var)

@testset "named predictive and filtered values of a top-level filter" begin
    d = ou_data(2, 5)
    (; R, area, continues, gap) = d.inputs
    gaps = Vector{Union{Missing,Float64}}(d.y)
    gaps[[1, 4, 7]] .= missing
    for (y, present) in ((d.y, collect(eachindex(d.y))), (gaps, findall(!ismissing, gaps)))
        plain = marginalize(OU(; d.inputs...), :ou) | (; y)
        plan = marginalize(OU(; d.inputs...), :ou; OU_NAMED...) | (; y)
        built, pbuilt = build_kernel(plan), build_kernel(plain)
        @test coordinate_names(built.layout) == coordinate_names(pbuilt.layout)
        yv = coalesce.(y, 0.0)
        for u in (0.3 .* sin.(1:built.layout.total),
                0.4 .* randn(Random.Xoshiro(3), built.layout.total))
            # Naming values leaves the density as it was.
            @test same_density(built, plan, pbuilt, plain, u)
            c = constrain(built.layout, u)
            a = continues .* exp.(-gap ./ c.ou_range)
            lat = latent_moments(0.0, 1.0, a, zeros(R), 1 .- a .^ 2)
            mu, S = observation_moments(lat, c.level[area], c.ou_sd, c.noise)
            loc, var = sequential_moments(mu, S, yv; present)
            fm, fv = filtered_moments(lat..., Matrix(c.ou_sd * I(R)), c.level[area],
                c.noise^2, yv, R; observed = t -> filter(<=(t), present))
            v = named_values(built, plan, OU_NAMES, u)
            @test v[1] ≈ loc rtol = 1e-10 atol = 1e-12
            @test v[2] ≈ var rtol = 1e-10
            @test v[3] ≈ sqrt.(var) rtol = 1e-10
            @test v[4] ≈ fm rtol = 1e-9 atol = 1e-11
            @test v[5] ≈ fv rtol = 1e-9 atol = 1e-12
        end
    end
    # One program as the series grows, as without names.
    shape(N, T) = let s = ou_data(N, T)
        structure(build_kernel(marginalize(OU(; s.inputs...), :ou; OU_NAMED...) | (; s.y)).spec)
    end
    @test shape(2, 3) == shape(4, 9)
    # Names compose with binding, observing, `condition` and `merge`.
    named = marginalize(OU, :ou; predictive = (location = :pred_loc,))
    reference = marginalize(OU(; d.inputs...), :ou; predictive = (location = :pred_loc,)) |
        (; y = d.y)
    built = build_kernel(reference)
    u = 0.2 .* cos.(1:built.layout.total)
    expected = named_values(built, reference, (:pred_loc,), u)
    for plan in (named(; d.inputs...) | (; y = d.y), condition(named(; d.inputs...); y = d.y),
            marginalize(OU | (; y = d.y), :ou; predictive = (location = :pred_loc,))(; d.inputs...),
            Base.merge(named, :(noise ~ Exponential(0.5)))(; d.inputs...) | (; y = d.y))
        @test named_values(build_kernel(plan), plan, (:pred_loc,), u) == expected
    end
end

@testset "named values of a filter observing a defined vector" begin
    d = censored_data(16)
    (; T, measured, limit, lo, hi, order) = d.inputs
    plain = censored_plan(d)
    plan = marginalize(CENSORED(; d.inputs...), :x;
        predictive = (location = :value_loc, scale = :value_sd),
        filtered = (mean = :x_mean, variance = :x_var))
    built, pbuilt = build_kernel(plan), build_kernel(plain)
    for u in (0.3 .* cos.(1:built.layout.total), 0.4 .* randn(Random.Xoshiro(5), built.layout.total))
        @test same_density(built, plan, pbuilt, plain, u)
        c = constrain(built.layout, u)
        value = vcat(measured, limit .- c.gap, lo .+ (hi .- lo) .* c.position)[order]
        lat = latent_moments(0.0, c.sigma^2 / (1 - c.phi^2), fill(c.phi, T), zeros(T),
            fill(c.sigma^2, T))
        loc, var = sequential_moments(fill(c.mu, T), lat[2] + c.noise^2 * I, value)
        fm, fv = filtered_moments(lat..., Matrix(1.0I, T, T), c.mu, c.noise^2, value, T)
        v = named_values(built, plan, (:value_loc, :value_sd, :x_mean, :x_var), u)
        @test v[1] ≈ loc rtol = 1e-10
        @test v[2] ≈ sqrt.(var) rtol = 1e-10
        @test v[3] ≈ fm rtol = 1e-9 atol = 1e-11
        @test v[4] ≈ fv rtol = 1e-9 atol = 1e-12
    end
end

@testset "named values of rows observing the state at their steps" begin
    d = gathered_data(7, 12)
    (; T, step, sensor, scale) = d.inputs
    plain = gathered_plan(d)
    plan = marginalize(GATHERED(; d.inputs...), :x;
        predictive = (location = :row_loc, variance = :row_var),
        filtered = (mean = :x_mean, variance = :x_var)) | (; d.y)
    built, pbuilt = build_kernel(plan), build_kernel(plain)
    for u in (0.3 .* cos.(1:built.layout.total), 0.4 .* randn(Random.Xoshiro(6), built.layout.total))
        @test same_density(built, plan, pbuilt, plain, u)
        c = constrain(built.layout, u)
        lat, L, mu, S = gathered_moments(c, d)
        # Each row given the rows before it in event order (steps, then rows).
        p = sortperm(step)
        loc, var = zeros(length(d.y)), zeros(length(d.y))
        loc[p], var[p] = sequential_moments(mu[p], S[p, p], d.y[p])
        # Each step given every row observing it or an earlier step.
        fm, fv = filtered_moments(lat..., L, c.bias[sensor], (c.noise .* scale) .^ 2, d.y, T;
            observed = t -> findall(<=(t), step))
        v = named_values(built, plan, (:row_loc, :row_var, :x_mean, :x_var), u)
        @test v[1] ≈ loc rtol = 1e-10
        @test v[2] ≈ var rtol = 1e-10
        @test v[3] ≈ fm rtol = 1e-9 atol = 1e-11
        @test v[4] ≈ fv rtol = 1e-9 atol = 1e-12
    end
end

@testset "named values of a recurrence per @plate index" begin
    d = cell_data([3, 1, 5]; seed = 4)
    plain = cell_ou_plan(d)
    plan = marginalize(CELL_OU(; d.G, d.gap, d.x), :ou;
        predictive = (location = :y_loc, scale = :y_sd),
        filtered = (mean = :ou_mean, variance = :ou_var)) | (; d.y)
    built, pbuilt = build_kernel(plan), build_kernel(plain)
    for u in (0.3 .* cos.(1:built.layout.total), 0.4 .* randn(Random.Xoshiro(7), built.layout.total))
        @test same_density(built, plan, pbuilt, plain, u)
        c = constrain(built.layout, u)
        v = named_values(built, plan, (:y_loc, :y_sd, :ou_mean, :ou_var), u)
        @test length.(v[1]) == length.(d.y)
        for g in 1:d.G
            n = length(d.y[g])
            a = exp.(-d.gap[g] ./ c.ou_range)
            lat = latent_moments(0.0, 1.0, a, zeros(n), 1 .- a .^ 2)
            offset = c.level[g] .+ c.b .* d.x[g]
            mu, S = observation_moments(lat, offset, c.ou_sd, c.noise)
            loc, var = sequential_moments(mu, S, d.y[g])
            fm, fv = filtered_moments(lat..., Matrix(c.ou_sd * I(n)), offset, c.noise^2,
                d.y[g], n)
            @test v[1][g] ≈ loc rtol = 1e-10
            @test v[2][g] ≈ sqrt.(var) rtol = 1e-10
            @test v[3][g] ≈ fm rtol = 1e-9 atol = 1e-11
            @test v[4][g] ≈ fv rtol = 1e-9 atol = 1e-12
        end
    end
end

@testset "named values of a vector state, its seeds included" begin
    T = 8
    cases = vector_cases(T)
    y = cases.trend.y
    w = collect(range(0.1, 1; length = T))
    E = Matrix(1.0I, T, T)
    named = (;
        trend = (marginalize(TREND(; T), :level, :slope;
                predictive = (location = :y_loc, variance = :y_var),
                filtered = (level = (mean = :level_mean, variance = :level_var),
                    slope = (mean = :slope_mean,))) | (; y),
            (:y_loc, :y_var, :level_mean, :level_var, :slope_mean),
            c -> trend_rec(c, T), 2T, c -> [E zeros(T, T)], c -> 0.0,
            # (role, stack entry of step t) per filtered name
            ((:mean, identity), (:var, identity), (:mean, t -> T + t))),
        # An AR(2): the filtered value at its first entry reads the seed window.
        ar2 = (marginalize(AR2(; T), :x; predictive = (scale = :y_sd,),
                filtered = (mean = :x_mean, variance = :x_var)) | (; y),
            (:y_sd, :x_mean, :x_var), c -> ar2_rec(c, T), T, c -> E, c -> 0.5,
            ((:mean, identity), (:var, identity))),
        # Three arrays seeded at two entries, one of them deterministic.
        cycle = (marginalize(CYCLE(; T, w), :c, :level, :v;
                filtered = (c = (mean = :c_mean,), v = (variance = :v_var,))) | (; y),
            (:c_mean, :v_var), c -> cycle_rec(c, T), 2T,
            c -> [zeros(T, T) E Diagonal(c.b .* w)], c -> 0.0,
            ((:mean, identity), (:var, t -> 2T + t))))
    plains = (; trend = cases.trend.plan, ar2 = cases.ar2.plan, cycle = cases.cycle.plan)
    for (name, (plan, names, rec, nz, H, offset, filtered)) in pairs(named)
        built, pbuilt = build_kernel(plan), build_kernel(plains[name])
        u = 0.3 .* cos.(1:built.layout.total)
        @test same_density(built, plan, pbuilt, plains[name], u)
        c = constrain(built.layout, u)
        m, S = affine_moments(rec(c), nz)
        Hc = H(c)
        mu = offset(c) .+ Hc * m
        loc, var = sequential_moments(mu, Hc * S * Hc' + c.noise^2 * I, y)
        v = named_values(built, plan, names, u)
        npred = length(names) - length(filtered)
        expected_pred = Dict(:y_loc => loc, :y_var => var, :y_sd => sqrt.(var))
        for k in 1:npred
            @test v[k] ≈ expected_pred[names[k]] rtol = 1e-9
        end
        for (k, (role, index)) in enumerate(filtered)
            fm, fv = filtered_moments(m, S, Hc, offset(c), c.noise^2, y, T; index)
            @test v[npred + k] ≈ (role === :mean ? fm : fv) rtol = 1e-8 atol = 1e-10
        end
    end
end

@testset "named values of a recurrence held in a submodel" begin
    T = 7
    y = randn(Random.Xoshiro(2), T)
    sub = marginalize(SUB_TREND(; T), Symbol("z.level"), Symbol("z.slope");
        predictive = (location = :y_loc,),
        filtered = Dict(Symbol("z.level") => (mean = :level_mean,))) | (; y)
    top = marginalize(TREND(; T), :level, :slope; predictive = (location = :y_loc,),
        filtered = (level = (mean = :level_mean,),)) | (; y)
    bs, bt = build_kernel(sub), build_kernel(top)
    u = 0.3 .* cos.(1:bs.layout.total)
    @test named_values(bs, sub, (:y_loc, :level_mean), u) ==
        named_values(bt, top, (:y_loc, :level_mean), u)
end

# Two recurrences, each with its own observation.
const TWO_WALKS = @rkppl begin
    s ~ Exponential(1)
    noise ~ Exponential(1)
    @scan begin
        x[1] ~ Normal(0, 1)
        for t in 2:T
            x[t] ~ Normal(x[t - 1], s)
        end
    end
    @scan begin
        z[1] ~ Normal(0, 1)
        for t in 2:T
            z[t] ~ Normal(z[t - 1], s)
        end
    end
    y .~ Normal.(x, noise)
    w .~ Normal.(z, noise)
end

@testset "named values: one call per recurrence, and refused names" begin
    T = 5
    y, w = randn(Random.Xoshiro(8), T), randn(Random.Xoshiro(9), T)
    plan = marginalize(marginalize(TWO_WALKS(; T), :x; predictive = (location = :x_loc,)), :z;
        predictive = (variance = :z_var,)) | (; y, w)
    built = build_kernel(plan)
    u = 0.3 .* cos.(1:built.layout.total)
    c = constrain(built.layout, u)
    lat = latent_moments(0.0, 1.0, ones(T), zeros(T), fill(c.s^2, T))
    S = lat[2] + c.noise^2 * I
    v = named_values(built, plan, (:x_loc, :z_var), u)
    @test v[1] ≈ first(sequential_moments(lat[1], S, y)) rtol = 1e-10 atol = 1e-12
    @test v[2] ≈ last(sequential_moments(lat[1], S, w)) rtol = 1e-10

    message(f) = try
        f()
        "no error"
    catch e
        e isa SurfaceLoweringError || rethrow()
        sprint(showerror, e)
    end
    d = ou_data(2, 3)
    lowered(; kw...) = () -> build_kernel(marginalize(OU(; d.inputs...), :ou; kw...) | (; d.y))
    # refused: the named value would shadow a parameter or an input of the model
    @test occursin("the value name `mu` is already used in the model",
        message(lowered(predictive = (location = :mu,))))
    @test occursin("the value name `gap` is already used in the model",
        message(lowered(filtered = (mean = :gap,))))
    # refused at the call: names RKPPL reserves, repeated names, unknown roles,
    # names that are not Symbols, and the integrated-out array's own name
    @test occursin("starts with `_rkppl`",
        message(() -> marginalize(OU, :ou; predictive = (location = :_rkppl_loc,))))
    @test occursin("the value name `a` is given twice",
        message(() -> marginalize(OU, :ou; predictive = (location = :a, scale = :a))))
    @test occursin("the value name `a` is given twice", message(() ->
        marginalize(marginalize(TWO_WALKS, :x; predictive = (location = :a,)), :z;
            predictive = (location = :a,))))
    @test occursin("`predictive` takes `location`, `variance`, `scale`; `mean` is not one",
        message(() -> marginalize(OU, :ou; predictive = (mean = :a,))))
    @test occursin("`filtered` takes `mean`, `variance`; `scale` is not one",
        message(() -> marginalize(OU, :ou; filtered = (scale = :a,))))
    @test occursin("`predictive.location` is the name of the value, a Symbol",
        message(() -> marginalize(OU, :ou; predictive = (location = "mu",))))
    @test occursin("`predictive` names values by role",
        message(() -> marginalize(OU, :ou; predictive = :mu)))
    @test occursin("`ou` is an array this call integrates out",
        message(() -> marginalize(OU, :ou; filtered = (mean = :ou,))))
    # refused: filtered values of a call with several arrays name their array,
    # and only arrays of the call
    @test occursin("this call integrates out several arrays", message(() ->
        marginalize(TREND, :level, :slope; filtered = (mean = :m,))))
    @test occursin("`filtered` names `z`, which this call does not integrate out",
        message(() -> marginalize(AR2, :x; filtered = (z = (mean = :m,),))))
    # refused: one call naming values for recurrences two statements observe
    @test occursin("integrates out recurrences observed by different statements",
        message(() -> build_kernel(marginalize(TWO_WALKS(; T), :x, :z;
            predictive = (location = :loc,)) | (; y, w))))
    # refused: two calls naming one recurrence's same value
    @test occursin("two `marginalize` calls name its predictive location", message(() ->
        build_kernel(marginalize(marginalize(TREND(; T), :level; predictive = (location = :a,)),
            :slope; predictive = (location = :b,)) | (; y))))
end

@testset "jax_source emits named filter values" begin
    isdefined(@__MODULE__, :rk_jax_available) ||
        include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))
    isdefined(@__MODULE__, :jax_raw_data) || include(joinpath(@__DIR__, "jax_ppl_data.jl"))
    d = ou_data(2, 4)
    gaps = Vector{Union{Missing,Float64}}(d.y)
    gaps[[2, 5]] .= missing
    g = gathered_data(5, 8)
    T = 6
    cases = (;
        toplevel = (marginalize(OU(; d.inputs...), :ou; OU_NAMED...) | (; d.y), OU_NAMES),
        missing_entries = (marginalize(OU(; d.inputs...), :ou; OU_NAMED...) | (; y = gaps),
            OU_NAMES),
        censored = (bind_data(marginalize(CENSORED(; censored_data(9).inputs...), :x;
            predictive = (location = :value_loc, scale = :value_sd),
            filtered = (mean = :x_mean,))), (:value_loc, :value_sd, :x_mean)),
        gathered = (marginalize(GATHERED(; g.inputs...), :x;
            predictive = (location = :row_loc,), filtered = (variance = :x_var,)) | (; g.y),
            (:row_loc, :x_var)),
        trend = (marginalize(TREND(; T), :level, :slope; predictive = (scale = :y_sd,),
            filtered = (level = (mean = :level_mean,),)) | (; y = vector_cases(T).trend.y),
            (:y_sd, :level_mean)))
    for (name, (plan, names)) in pairs(cases)
        built = build_kernel(plan)
        src = jax_source(built, plan; values = names)
        @test occursin("UNTRANSLATED = {}", string(src))
        if !rk_jax_available()
            @info "RK_JAX_PYTHON is not set: JAX parity of named values for $name skipped"
            @test_skip false
            continue
        end
        u = 0.2 .* randn(Random.Xoshiro(7), built.layout.total)
        native = named_values(built, plan, names, u)
        r = rk_jax_evaluate(src, "constrain", (u, jax_raw_data(plan, src)...); prepare = true).value
        for (k, nm) in enumerate(names)
            @test r[String(nm)] ≈ native[k] rtol = 1e-10 atol = 1e-12
        end
    end
end

end # module
