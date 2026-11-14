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

# x | y from the joint Gaussian of (x, y).
function latent_posterior(lat, offset, loading, noise, y)
    mx, Sx = lat
    n = length(mx)
    L = Diagonal(loading .* ones(n))
    my = offset .+ L * mx
    Sy = L * Sx * L + Diagonal((noise .* ones(n)) .^ 2)
    K = Sx * L / Sy
    return mx + K * (y - my), Symmetric(Sx - K * L * Sx)
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

function ou_oracle(built, d)
    (; N, R, area, continues, gap) = d.inputs
    function (u)
        c = constrain(built.layout, u)
        prior = logpdf(Normal(0, 1), c.mu) + logpdf(Exponential(0.3), c.noise) +
            logpdf(Exponential(0.2), c.ou_sd) + logpdf(LogNormal(1, 0.5), c.ou_range) +
            sum(logpdf.(Normal(c.mu, 0.5), c.level))
        a = continues .* exp.(-gap ./ c.ou_range)
        lat = latent_moments(0.0, 1.0, a, zeros(R), 1 .- a .^ 2)
        mu, S = observation_moments(lat, c.level[area], c.ou_sd, c.noise)
        pointwise = sequential_logpdfs(mu, S, d.y)
        likelihood = logpdf(MvNormal(mu, S), d.y)
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
    # A second carried array, a lag above one, and a seed that is not Normal.
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
    lag2 = quote
        @scan begin
            x[1] ~ Normal(0, 1)
            x[2] ~ Normal(0, 1)
            for t in 3:T
                x[t] ~ Normal(x[t - 2], 1)
            end
        end
        y .~ Normal.(x, 1)
    end
    @test occursin("seeded at 2 entries", refusal(lag2, :x, data, y))
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
    @test occursin("no top-level `@scan` carries `q`",
        refusal(observed(base), :q, data, y))
    # The observation must be bound data.
    unobserved = try
        bind_data(marginalize(RKPPLModel(observed(base), @__MODULE__)(; T = 4), :x))
        "no error"
    catch e
        sprint(showerror, e)
    end
    @test occursin("is not observed data", unobserved)
    # A recurrence in a plate cell is not marginalized yet.
    cell = quote
        s ~ Exponential(1)
        @plate for g in eachindex(y)
            @scan begin
                h[1] ~ Normal(0, 1)
                for t in 2:length(y[g])
                    h[t] ~ Normal(h[t - 1], s)
                end
            end
            loc[g] = h
            y[g] .~ Normal.(loc[g], 1)
        end
    end
    @test occursin("inside a `@plate` cell", refusal(cell, :h, (;), [[0.1, 0.2], [0.3]]))
    # A plan is already lowered.
    d = ou_data(1, 3)
    @test_throws SurfaceLoweringError marginalize(OU(; d.inputs...) | (; y = d.y), :ou)
    @test_throws ArgumentError marginalize(OU)
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
