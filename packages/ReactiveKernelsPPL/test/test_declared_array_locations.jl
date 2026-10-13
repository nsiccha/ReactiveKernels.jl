# A declared array is a value with its own axes. Read whole as a
# distribution argument (`y .~ Normal.(theta, sigma)` with
# `theta[axes(y, 1)] .~ …`), it pairs `theta[j]` with `y[j]` as Julia
# broadcasting does, exactly like its named twin `loc = theta`: naming a
# subexpression never changes legality or the density (standing @rkppl
# principle 3, standard-Julia semantics; snag rkppl-declared-a-812e7021).
# Independent Distributions oracles, native Enzyme reverse against finite
# differences, and parity with the named, gathered and `@plate` spellings.
using Distributions: Normal, Cauchy, Exponential, Beta, Poisson, Bernoulli,
    Binomial, truncated, logpdf, pdf
using DifferentiationInterface: AutoEnzyme
using Enzyme
using LogExpFunctions: logistic, logaddexp
using ReactiveKernels
using ReactiveKernelsPPL
using Test

const _DAL_Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const _DAL_SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
const _DAL_COUNTS = [3, 0, 5, 2, 1, 7, 4, 2]
const _DAL_BINARY = [1, 0, 1, 1, 0, 0, 1, 0]
const _DAL_TRIALS = fill(8, 8)

# Build and prepare the sampler at a fixed nonzero point; preparation must
# not warn (a location read in place is concretely typed).
function _dal_query(bound)
    built = build_kernel(bound)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    q = @test_logs min_level = Base.CoreLogging.Warn prepare_sampler(built,
        bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    g = similar(u)
    value, _ = sampler_value_and_gradient!(q, g, u)
    return (; built, u, q, value, g = copy(g))
end

function _dal_fd(q, u; h = cbrt(eps(Float64)))
    return map(eachindex(u)) do i
        e = zeros(length(u))
        e[i] = h
        (q(u .+ e) - q(u .- e)) / 2h
    end
end

# The posterior at the query's point from the constrained draws: the stated
# priors and likelihood, plus the transforms' log Jacobian.
_dal_oracle(r, density) =
    density(constrain(r.built.layout, r.u)) + logjac(r.built.layout, r.u)

function _dal_check(r, density)
    @test r.value ≈ _dal_oracle(r, density) rtol = 1e-12
    @test isapprox(r.g, _dal_fd(r.q, r.u); rtol = 1e-5, atol = 1e-7)
end

_dal_same(a, b) = a.value ≈ b.value && isapprox(a.g, b.g; rtol = 1e-12)

@testset "a declared array as the location (centered eight schools)" begin
    direct = @rkppl begin
        mu ~ Normal(0, 5)
        tau ~ truncated(Cauchy(0, 5), 0, Inf)
        theta[axes(y, 1)] .~ Normal.(mu, tau)
        y .~ Normal.(theta, sigma)
    end
    named = @rkppl begin
        mu ~ Normal(0, 5)
        tau ~ truncated(Cauchy(0, 5), 0, Inf)
        theta[axes(y, 1)] .~ Normal.(mu, tau)
        loc = theta
        y .~ Normal.(loc, sigma)
    end
    gathered = @rkppl begin
        mu ~ Normal(0, 5)
        tau ~ truncated(Cauchy(0, 5), 0, Inf)
        theta[axes(y, 1)] .~ Normal.(mu, tau)
        y .~ Normal.(theta[rows], sigma)
    end
    looped = @rkppl begin
        mu ~ Normal(0, 5)
        tau ~ truncated(Cauchy(0, 5), 0, Inf)
        @plate for j in eachindex(y)
            theta[j] ~ Normal(mu, tau)
            y[j] ~ Normal(theta[j], sigma[j])
        end
    end
    y, sigma = _DAL_Y, _DAL_SIGMA
    r = _dal_query(direct(; sigma) | (; y))
    _dal_check(r, nt -> logpdf(Normal(0, 5), nt.mu) +
        logpdf(truncated(Cauchy(0, 5), 0, Inf), nt.tau) +
        sum(logpdf.(Normal.(nt.mu, nt.tau), nt.theta)) +
        sum(logpdf.(Normal.(nt.theta, sigma), y)))
    @test coordinate_names(r.built.layout) ==
        [:mu, :tau, (Symbol("theta.", j) for j in 1:8)...]
    @test _dal_same(r, _dal_query(named(; sigma) | (; y)))
    @test _dal_same(r, _dal_query(gathered(; sigma, rows = collect(1:8)) | (; y)))
    @test _dal_same(r, _dal_query(looped(; sigma) | (; y)))
    # A plan lowered from names alone binds to the same program.
    ast = :(begin
        mu ~ Normal(0, 5)
        tau ~ truncated(Cauchy(0, 5), 0, Inf)
        theta[axes(y, 1)] .~ Normal.(mu, tau)
        y .~ Normal.(theta, sigma)
    end)
    unbound = lower_rkppl(ast, (:y, :sigma); conditioned = (:y,))
    @test _dal_same(r, _dal_query(bind_data(unbound, (; y, sigma))))
end

@testset "a declared array under a link and in a bare slot" begin
    counts, binary, n = _DAL_COUNTS, _DAL_BINARY, _DAL_TRIALS
    log_rate = @rkppl begin
        a[axes(counts, 1)] .~ Normal.(0, 1)
        counts .~ Poisson.(exp.(a))
    end
    _dal_check(_dal_query(log_rate() | (; counts)), nt ->
        sum(logpdf.(Normal(0, 1), nt.a)) +
        sum(logpdf.(Poisson.(exp.(nt.a)), counts)))
    logit = @rkppl begin
        a[axes(binary, 1)] .~ Normal.(0, 1)
        binary .~ Bernoulli.(logistic.(a))
    end
    _dal_check(_dal_query(logit() | (; binary)), nt ->
        sum(logpdf.(Normal(0, 1), nt.a)) +
        sum(logpdf.(Bernoulli.(logistic.(nt.a)), binary)))
    # Written bare in a rate or probability slot, the array is that rate or
    # probability, as its named twin `r = lam` is.
    rate = @rkppl begin
        lam[axes(counts, 1)] .~ Exponential.(1)
        counts .~ Poisson.(lam)
    end
    named_rate = @rkppl begin
        lam[axes(counts, 1)] .~ Exponential.(1)
        r = lam
        counts .~ Poisson.(r)
    end
    bare = _dal_query(rate() | (; counts))
    _dal_check(bare, nt -> sum(logpdf.(Exponential(1), nt.lam)) +
        sum(logpdf.(Poisson.(nt.lam), counts)))
    @test _dal_same(bare, _dal_query(named_rate() | (; counts)))
    probability = @rkppl begin
        p[axes(binary, 1)] .~ Beta.(2, 2)
        binary .~ Bernoulli.(p)
    end
    _dal_check(_dal_query(probability() | (; binary)), nt ->
        sum(logpdf.(Beta(2, 2), nt.p)) + sum(logpdf.(Bernoulli.(nt.p), binary)))
    trials = @rkppl begin
        p[axes(counts, 1)] .~ Beta.(2, 2)
        counts .~ Binomial.(n, p)
    end
    _dal_check(_dal_query(trials(; n) | (; counts)), nt ->
        sum(logpdf.(Beta(2, 2), nt.p)) + sum(logpdf.(Binomial.(n, nt.p), counts)))
    inflated = @rkppl begin
        p[axes(counts, 1)] .~ Beta.(2, 2)
        counts .~ ZeroInflatedBinomial.(n, p, 0.2)
    end
    zib(k, m, q) = k == 0 ? log(0.2 + 0.8 * pdf(Binomial(m, q), 0)) :
        log(0.8) + logpdf(Binomial(m, q), k)
    _dal_check(_dal_query(inflated(; n) | (; counts)), nt ->
        sum(logpdf.(Beta(2, 2), nt.p)) + sum(zib.(counts, n, nt.p)))
end

@testset "a declared array as a mixture component location and a scale" begin
    y = _DAL_Y
    mixture = @rkppl begin
        s ~ Exponential(1)
        theta[axes(y, 1)] .~ Normal.(0, 10)
        y .~ MixtureModel.(vcat.(Normal.(theta, s), Normal.(0, s)), Ref([0.3, 0.7]))
    end
    _dal_check(_dal_query(mixture() | (; y)), nt ->
        logpdf(Exponential(1), nt.s) + sum(logpdf.(Normal(0, 10), nt.theta)) +
        sum(logaddexp.(log(0.3) .+ logpdf.(Normal.(nt.theta, nt.s), y),
            log(0.7) .+ logpdf.(Normal(0, nt.s), y))))
    scale = @rkppl begin
        mu ~ Normal(0, 5)
        s[axes(y, 1)] .~ Normal.(0, 1)
        y .~ Normal.(mu, exp.(s))
    end
    named_scale = @rkppl begin
        mu ~ Normal(0, 5)
        s[axes(y, 1)] .~ Normal.(0, 1)
        sd = exp.(s)
        y .~ Normal.(mu, sd)
    end
    r = _dal_query(scale() | (; y))
    _dal_check(r, nt -> logpdf(Normal(0, 5), nt.mu) +
        sum(logpdf.(Normal(0, 1), nt.s)) + sum(logpdf.(Normal.(nt.mu, exp.(nt.s)), y)))
    @test _dal_same(r, _dal_query(named_scale() | (; y)))
end

@testset "Julia broadcasting decides a declared location's axes" begin
    y, sigma = _DAL_Y, _DAL_SIGMA
    # A one-entry array stretches over every observation.
    single = @rkppl begin
        theta[1:1] .~ Normal.(0, 10)
        y .~ Normal.(theta, sigma)
    end
    _dal_check(_dal_query(single(; sigma) | (; y)), nt ->
        logpdf(Normal(0, 10), only(nt.theta)) +
        sum(logpdf.(Normal.(only(nt.theta), sigma), y)))
    # Labels play no part: an eight-level array pairs by position.
    by_level = @rkppl begin
        theta[levels(g)] .~ Normal.(0, 10)
        y .~ Normal.(theta, sigma)
    end
    _dal_check(_dal_query(by_level(; sigma, g = [3, 1, 2, 8, 7, 6, 5, 4]) | (; y)),
        nt -> sum(logpdf.(Normal(0, 10), nt.theta)) +
            sum(logpdf.(Normal.(nt.theta, sigma), y)))
    # A two-axis array locates a matrix response entry by entry.
    Y = reshape(collect(1.0:8.0), 2, 4)
    matrix = @rkppl begin
        s ~ Exponential(1)
        Z[axes(Y, 1), axes(Y, 2)] .~ Normal.(0, 1)
        Y .~ Normal.(Z, s)
    end
    _dal_check(_dal_query(matrix() | (; Y)), nt ->
        logpdf(Exponential(1), nt.s) + sum(logpdf.(Normal(0, 1), nt.Z)) +
        sum(logpdf.(Normal.(nt.Z, nt.s), Y)))
    # refused: Julia cannot broadcast lengths 8 and 3 together (standing
    # @rkppl principle 3, standard-Julia semantics)
    short = @rkppl begin
        theta[1:3] .~ Normal.(0, 10)
        y .~ Normal.(theta, sigma)
    end
    @test_throws DimensionMismatch begin
        bound = short(; sigma) | (; y)
        built = build_kernel(bound)
        Base.invokelatest(prepare_query(built, bound, :sampler),
            zeros(built.layout.total))
    end
end
