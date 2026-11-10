using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme

# DistributionKernels' LogNormal statistics called from a model, as BRM emits
# them for `mean`/`var` of a (one-sided truncated) `LogNormal` and its
# `logcdf`/`logccdf`. `var()` calls a prepared child that owns a plate inside
# a lazy arm. Enzyme infers in a cache of its own, where that call inferred
# `Any`: an observation plate reading `var` failed static Reverse with
# `EnzymeRuntimeActivityError`, and a plate cell computing it gave a
# `Vector{Any}` cell buffer (snags `native-lognormal-a666a8e1`,
# `native-truncated-9d7d9eda`; ReactiveKernels 0.1.149).
module LogNormalStatisticsReverse
using ReactiveKernels, ReactiveKernelsPPL, SpecialFunctions
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    lower_truncated_lognormal, upper_truncated_lognormal, lognormal

@kernel lower_mean(location, scale, bound) = begin
    return lower_truncated_lognormal(location, scale, bound).mean()
end
@kernel lower_var(location, scale, bound) = begin
    return lower_truncated_lognormal(location, scale, bound).var()
end
@kernel upper_mean(location, scale, bound) = begin
    return upper_truncated_lognormal(location, scale, bound).mean()
end
@kernel upper_var(location, scale, bound) = begin
    return upper_truncated_lognormal(location, scale, bound).var()
end
@kernel tail_logccdf(location, scale, bound) = begin
    return lognormal(location, scale).logccdf(bound)
end
@kernel tail_logcdf(location, scale, bound) = begin
    return lognormal(location, scale).logcdf(bound)
end

ordinary = @rkppl begin
    mu ~ Normal(0.0, 1.0)
    log_sigma ~ Normal(0.0, 0.5)
    m = lower_mean(mu, exp.(log_sigma), 0.0)
    v = lower_var(mu, exp.(log_sigma), 0.0)
    logmass = tail_logccdf(mu, exp.(log_sigma), 0.8)
    logcomplement = tail_logcdf(mu, exp.(log_sigma), 0.8)
    y .~ Normal.(m, sqrt.(v .+ 0.1))
    survived .~ BernoulliLogit.(logmass .- logcomplement)
end

lower = @rkppl begin
    shift ~ Normal(0.0, 1.0)
    log_scale ~ Normal(0.0, 0.5)
    log_bound ~ Normal(0.0, 0.3)
    @plate for i = eachindex(y)
        location = baseline[i] + shift
        scale = sigma[i] * exp(log_scale)
        limit = bound[i] * exp(log_bound)
        m_cell = lower_mean(location, scale, limit)
        v_cell = lower_var(location, scale, limit)
        logmass_cell = tail_logccdf(location, scale, limit)
        logcomplement_cell = tail_logcdf(location, scale, limit)
        y_location[i] = m_cell
        y_scale[i] = sqrt(v_cell + noise[i] ^ 2)
        survived_logit[i] = logmass_cell - logcomplement_cell
    end
    y .~ Normal.(y_location, y_scale)
    survived .~ BernoulliLogit.(survived_logit)
end

upper = @rkppl begin
    shift ~ Normal(0.0, 1.0)
    log_scale ~ Normal(0.0, 0.5)
    log_bound ~ Normal(0.0, 0.3)
    @plate for i = eachindex(y)
        location = baseline[i] + shift
        scale = sigma[i] * exp(log_scale)
        limit = bound[i] * exp(log_bound)
        m_cell = upper_mean(location, scale, limit)
        v_cell = upper_var(location, scale, limit)
        logmass_cell = tail_logcdf(location, scale, limit)
        logcomplement_cell = tail_logccdf(location, scale, limit)
        y_location[i] = m_cell
        y_scale[i] = sqrt(v_cell + noise[i] ^ 2)
        survived_logit[i] = logmass_cell - logcomplement_cell
    end
    y .~ Normal.(y_location, y_scale)
    survived .~ BernoulliLogit.(survived_logit)
end

# Independent 256-bit references: the one-sided truncated LogNormal moments,
# the LogNormal tails and the densities of the three models.
big_normlogpdf(mu, sd, x) = -log(sd) - log(2 * big(pi)) / 2 - ((x - mu) / sd)^2 / 2
big_logistic_log(eta) = -log1p(exp(-eta))
function moments(mu, s, b, side)
    side === :ordinary && return (exp(mu + s^2 / 2), exp(2mu + s^2) * expm1(s^2))
    z = (log(b) - mu) / s
    tail(x) = erfc((side === :upper ? -x : x) / sqrt(big(2))) / 2
    first = exp(mu + s^2 / 2) * tail(z - s) / tail(z)
    second = exp(2mu + 2s^2) * tail(z - 2s) / tail(z)
    (first, second - first^2)
end
big_logccdf(mu, s, b) = log(erfc((log(b) - mu) / (s * sqrt(big(2)))) / 2)
big_logcdf(mu, s, b) = log(erfc(-(log(b) - mu) / (s * sqrt(big(2)))) / 2)

function ordinary_density(u, y)
    setprecision(BigFloat, 256) do
        mu, ls = big(u[1]), big(u[2])
        s = exp(ls)
        m, v = moments(mu, s, big(0.0), :ordinary)
        eta = big_logccdf(mu, s, big(0.8)) - big_logcdf(mu, s, big(0.8))
        value = big_normlogpdf(0, 1, mu) + big_normlogpdf(0, big(0.5), ls)
        for yi in y
            value += big_normlogpdf(m, sqrt(v + big(0.1)), big(yi)) + big_logistic_log(eta)
        end
        Float64(value)
    end
end

function plate_density(u, data, side)
    setprecision(BigFloat, 256) do
        shift, ls, lb = big(u[1]), big(u[2]), big(u[3])
        value = big_normlogpdf(0, 1, shift) + big_normlogpdf(0, big(0.5), ls) +
            big_normlogpdf(0, big(0.3), lb)
        for i in eachindex(data.y)
            location = big(data.baseline[i]) + shift
            scale = big(data.sigma[i]) * exp(ls)
            limit = big(data.bound[i]) * exp(lb)
            m, v = moments(location, scale, limit, side)
            above, below = big_logccdf(location, scale, limit), big_logcdf(location, scale, limit)
            eta = side === :lower ? above - below : below - above
            value += big_normlogpdf(m, sqrt(v + big(data.noise[i])^2), big(data.y[i])) +
                big_logistic_log(eta)
        end
        Float64(value)
    end
end

central(f, u; h = 1e-6) = map(eachindex(u)) do j
    up, down = copy(u), copy(u)
    up[j] += h
    down[j] -= h
    (f(up) - f(down)) / 2h
end
end

@testset "LogNormal statistics under static Enzyme Reverse" begin
    M = LogNormalStatisticsReverse
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    function check(plan, oracle, points)
        built = build_kernel(plan)
        for u in points
            before = copy(u)
            q = prepare_sampler(built, plan, u; backend)
            gradient = similar(u)
            value, _ = sampler_value_and_gradient!(q, gradient, u)
            @test value ≈ oracle(u) rtol = 1e-11
            @test gradient ≈ M.central(oracle, u) rtol = 1e-6 atol = 1e-7
            @test u == before
        end
    end

    @testset "ordinary moments and tails read by observation rows" begin
        y = [1.4, 2.1, 0.6]
        plan = M.ordinary() | (; y, survived = ones(Int, 3))
        check(plan, u -> M.ordinary_density(u, y), ([0.1, -0.2], [0.0, 0.0], [-0.3, 0.4]))
    end

    @testset "$side truncated moments in a plate cell with an active bound" for
            (side, model) in ((:lower, M.lower), (:upper, M.upper))
        inputs = (; baseline = [-3.0, 0.2, 1.7], sigma = [0.2, 0.7, 1.3],
            bound = [1.0, 0.8, 2.0], noise = [0.1, 0.2, 0.3])
        observed = (; y = [1.1, 0.7, 1.8], survived = ones(Int, 3))
        data = merge(inputs, observed)
        saved = deepcopy(data)
        plan = model(; inputs...) | observed
        check(plan, u -> M.plate_density(u, data, side),
            (zeros(3), [0.15, -0.1, 0.08], [-0.2, 0.1, -0.15]))
        @test isequal(merge(inputs, observed), saved)
    end
end
