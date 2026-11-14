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
    # The merged vector reads parameters, so its term is a factor, as in the
    # source without `marginalize`: no likelihood or pointwise term.
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

function vector_cases(T; seed = 11)
    rng = Random.Xoshiro(seed)
    y = randn(rng, T)
    w = collect(range(0.1, 1; length = T))
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
            offset = c -> 0.0, prior = cycle_prior, keys = (:c, :level, :v)))
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
        gathered = gathered_plan(gathered_data(5, 8)),
        trend = vector_cases(6).trend.plan, ar2 = vector_cases(6).ar2.plan,
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

end # module
