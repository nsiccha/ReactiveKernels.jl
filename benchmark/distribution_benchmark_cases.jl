module DistributionBenchmarkCases

using Random

export NORMAL_SIZES, SCALAR_GALLERY_FAMILIES, SCALAR_GALLERY_SIZES,
    DISTRIBUTION_GRADIENT_FAMILIES, STRUCTURED_SIZES, normal_parameters,
    normal_observations, scalar_family_inputs, mvn_inputs

const NORMAL_SIZES = (1, 1_000, 10_000, 30_000, 100_000, 1_000_000)
const SCALAR_GALLERY_FAMILIES = (
    "cauchy_location_scale",
    "laplace_location_scale",
    "bernoulli_logit",
    "lognormal_logscale",
    "exponential_logscale",
    "geometric_logit",
    "uniform_bounded",
    "poisson_lograte",
    "gamma_shape_rate",
    "beta_shapes",
    "binomial_logit",
)
const DISTRIBUTION_GRADIENT_FAMILIES = (
    "cauchy_location_scale",
    "laplace_location_scale",
    "bernoulli_logit",
    "lognormal_logscale",
    "exponential_logscale",
    "geometric_logit",
    "uniform_bounded",
)
const SCALAR_GALLERY_SIZES = (1_000, 100_000)
const STRUCTURED_SIZES = (4, 16, 64, 128)

normal_parameters() = (; location = 0.3, scale = 1.2)
normal_observations(n::Integer) = randn(n)

scalar_family_inputs(family::AbstractString, n::Integer) =
    scalar_family_inputs(Val(Symbol(family)), n)

function scalar_family_inputs(::Val{:cauchy_location_scale}, n::Integer)
    location, scale = -0.3, 1.1
    x = [location + scale * (0.04i - 2.0) for i in 1:n]
    (; x, location, scale)
end

function scalar_family_inputs(::Val{:laplace_location_scale}, n::Integer)
    location, scale = 0.2, 0.8
    x = [location + scale * sin(0.019i) for i in 1:n]
    (; x, location, scale)
end

function scalar_family_inputs(::Val{:bernoulli_logit}, n::Integer)
    logit = -0.7
    observed = [isodd(i * 5) for i in 1:n]
    (; observed, logit)
end

function scalar_family_inputs(::Val{:lognormal_logscale}, n::Integer)
    location, log_scale = 0.2, log(0.9)
    x = [exp(location + exp(log_scale) * 0.7sin(0.013i)) for i in 1:n]
    (; x, location, log_scale)
end

function scalar_family_inputs(::Val{:exponential_logscale}, n::Integer)
    log_scale = log(1.3)
    x = [0.05 + abs(sin(0.017i)) + 0.002(i % 11) for i in 1:n]
    (; x, log_scale)
end

function scalar_family_inputs(::Val{:geometric_logit}, n::Integer)
    logitp = 0.4
    observed = [mod(i * 5, 9) for i in 0:(n - 1)]
    (; observed, logitp)
end

function scalar_family_inputs(::Val{:uniform_bounded}, n::Integer)
    lower, upper = -1.0, 2.0
    x = collect(range(lower + 0.01, upper - 0.01; length = n))
    (; x, lower, upper)
end

function scalar_family_inputs(::Val{:poisson_lograte}, n::Integer)
    log_rate = log(2.5)
    observed = [mod(i * 7, 9) for i in 0:(n - 1)]
    (; observed, log_rate)
end

function scalar_family_inputs(::Val{:gamma_shape_rate}, n::Integer)
    shape, log_rate = 2.0, log(1.5)
    x = [0.05 + 2.5abs(sin(0.021i)) + 0.001(i % 13) for i in 1:n]
    (; x, shape, log_rate)
end

function scalar_family_inputs(::Val{:beta_shapes}, n::Integer)
    a, b = 2.0, 5.0
    x = [0.01 + 0.98 * (0.5 + 0.5sin(0.017i)) for i in 1:n]
    (; x, a, b)
end

function scalar_family_inputs(::Val{:binomial_logit}, count::Integer)
    trials, logit = 10, 0.2
    observed = [mod(i * 3, trials + 1) for i in 0:(count - 1)]
    (; observed, n = trials, logit)
end

function mvn_inputs(n::Integer)
    μ = collect(range(-0.4, 0.6; length = n))
    x = μ .+ [0.7sin(0.31i) - 0.2cos(0.17i) for i in 1:n]
    chol = zeros(n, n)
    for i in 1:n
        chol[i, i] = 0.9 + 0.002i
        for j in 1:(i - 1)
            chol[i, j] = 0.04sin(0.13i + 0.29j) / sqrt(n)
        end
    end
    (; x, μ, chol)
end

end # module DistributionBenchmarkCases
