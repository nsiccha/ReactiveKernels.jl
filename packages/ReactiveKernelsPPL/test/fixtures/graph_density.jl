module GraphDensityFixtures
using ReactiveKernels, ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
import Distributions

# A caller-owned scalar law; no statistical family is added to the producer.
@kernel threshold_score(value, mu, add, prop, lower, upper) = begin
    sigma = sqrt(add^2 + (mu*prop)^2)
    score = if value <= lower
        log(normal(mu, sigma).cdf(lower))
    elseif value >= upper
        log(normal(mu, sigma).ccdf(upper))
    else
        normal(mu, sigma).logpdf(value)
    end
    return score
end
const alias = threshold_score

@kernel scalar_score(value, mu, scale) = begin
    return normal(mu, scale).logpdf(value)
end
@kernel score_object() = begin
    density(value::Float64, mu::Float64, scale::Float64)::Float64 =
        normal(mu, scale).logpdf(value)
end
const extracted_score = extract(score_object; want=:density)
const density_alias = LogDensity
numerical_score(value, mu; scale=1.0) = -log(2pi)/2-log(scale)-(value-mu)^2/(2scale^2)

function fixture(n; head=:threshold_score, rowwise=false)
    x = [0.2sin(i) for i in 1:n]
    lower = rowwise ? [0.1+0.01i for i in 1:n] : 0.15
    upper = rowwise ? [0.9+0.01i for i in 1:n] : 0.95
    y = zeros(n)
    for i in eachindex(y)
        lo, hi = rowwise ? (lower[i], upper[i]) : (lower, upper)
        y[i] = (lo-0.1, lo, (lo+hi)/2, hi, hi+0.1)[mod1(i, 5)]
    end
    data = (;x, y, lower, upper)
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        add ~ Exponential(1)
        prop ~ Exponential(1)
        mu = a .+ b .* x
        y .~ LogDensity.($head, mu, add, prop, lower, upper)
    end
    bound = bind_data(lower_rkppl(ast, data; mod=@__MODULE__, conditioned=(:y,)), data)
    built = build_kernel(bound)
    return (;data, bound, built, u=[0.2, -0.3, log(0.7), log(0.25)])
end

function oracle(fx, u)
    a, b, add, prop = u[1], u[2], exp(u[3]), exp(u[4])
    points = map(eachindex(fx.data.y)) do i
        mu = a+b*fx.data.x[i]
        sigma = hypot(add, mu*prop)
        lo = fx.data.lower isa Number ? fx.data.lower : fx.data.lower[i]
        hi = fx.data.upper isa Number ? fx.data.upper : fx.data.upper[i]
        law = Distributions.Normal(mu, sigma)
        value = fx.data.y[i]
        value <= lo ? Distributions.logcdf(law, lo) :
            value >= hi ? Distributions.logccdf(law, hi) : Distributions.logpdf(law, value)
    end
    prior = -log(2pi)-(a^2+b^2)/2-add-prop
    return (;points, prior, likelihood=sum(points; init=0.0),
        posterior=sum(points; init=0.0)+prior+u[3]+u[4])
end

end
