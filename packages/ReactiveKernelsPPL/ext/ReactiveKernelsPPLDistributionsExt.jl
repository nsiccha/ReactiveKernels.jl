module ReactiveKernelsPPLDistributionsExt

import Distributions
import ReactiveKernelsPPL
import ReactiveKernelsDistributionKernels: DistributionKernelSources
using LogExpFunctions: logistic

# Distribution objects use their ordinary public logpdf. Parameter geometry
# is a separate capability; it is never inferred by evaluating active args.
ReactiveKernelsPPL.sampling_logdensity(d::Distributions.Distribution, value) =
    Distributions.logpdf(d, value)

# Structural mappings consume authored arguments without evaluating them.
# Defaults and Gamma's scale parameter follow the resolved Julia constructor.
_graph_call(name, args...) = Expr(:call, GlobalRef(DistributionKernelSources, name), args...)
_float_argument(x) = Expr(:call, GlobalRef(Base, :Float64), x)
_checked_argument(x, nonnegative=false) = _graph_call(:_checked_distribution_parameter, x, nonnegative)

function _location_scale_graph(name, args, nonnegative)
    length(args) <= 2 || return nothing
    location = isempty(args) ? 0.0 : args[1]
    scale = length(args) < 2 ? 1.0 : args[2]
    _graph_call(name, _float_argument(location), _checked_argument(scale, nonnegative))
end

ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.Normal}, args::Tuple) =
    _location_scale_graph(:normal, args, true)
ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.LogNormal}, args::Tuple) =
    _location_scale_graph(:lognormal, args, true)
ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.Cauchy}, args::Tuple) =
    _location_scale_graph(:cauchy, args, false)
ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.Laplace}, args::Tuple) =
    _location_scale_graph(:laplace, args, false)
ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.Logistic}, args::Tuple) =
    _location_scale_graph(:logistic, args, false)
ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.Gumbel}, args::Tuple) =
    _location_scale_graph(:gumbel, args, false)

# Positive-parameter families share constructor validation, but their arities
# and defaults remain those of the resolved Distributions.jl constructor.
function _positive_pair_graph(name, args)
    length(args) <= 2 || return nothing
    first = isempty(args) ? 1.0 : args[1]
    second = length(args) < 2 ? 1.0 : args[2]
    _graph_call(name, _checked_argument(first), _checked_argument(second))
end
function ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.Weibull}, args::Tuple)
    # The one-argument Julia constructor does not check its shape. Preserve
    # that call's behavior; the two-argument constructor validates both ports.
    length(args) == 1 && return _graph_call(:_unchecked_unit_weibull, _float_argument(only(args)))
    _positive_pair_graph(:weibull, args)
end
ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.InverseGamma}, args::Tuple) =
    _positive_pair_graph(:inverse_gamma, args)
ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.InverseGaussian}, args::Tuple) =
    _positive_pair_graph(:inverse_gaussian, args)

function ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.TDist}, args::Tuple)
    length(args) == 1 || return nothing
    _graph_call(:student_t, _checked_argument(only(args)), 0.0, 1.0)
end
function ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.Rayleigh}, args::Tuple)
    length(args) <= 1 || return nothing
    _graph_call(:rayleigh, _checked_argument(isempty(args) ? 1.0 : only(args)))
end
function ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.Chi}, args::Tuple)
    length(args) == 1 || return nothing
    _graph_call(:chi, _checked_argument(only(args)))
end
function ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.Chisq}, args::Tuple)
    length(args) == 1 || return nothing
    _graph_call(:chi_squared, _checked_argument(only(args)))
end
function ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.FDist}, args::Tuple)
    length(args) == 2 || return nothing
    _graph_call(:f_dist, _checked_argument(args[1]), _checked_argument(args[2]))
end
function ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.BetaPrime}, args::Tuple)
    length(args) <= 2 || return nothing
    alpha = isempty(args) ? 1.0 : args[1]
    beta = length(args) < 2 ? alpha : args[2]
    _graph_call(:beta_prime, _checked_argument(alpha), _checked_argument(beta))
end

function ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.Gamma}, args::Tuple)
    length(args) <= 2 || return nothing
    shape = isempty(args) ? 1.0 : args[1]
    scale = length(args) < 2 ? 1.0 : args[2]
    _graph_call(:gamma, Expr(:parameters,
        Expr(:kw, :shape, _checked_argument(shape)),
        Expr(:kw, :scale, _checked_argument(scale))))
end
function ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.Exponential}, args::Tuple)
    length(args) <= 1 || return nothing
    _graph_call(:exponential, _checked_argument(isempty(args) ? 1.0 : only(args)))
end
function ReactiveKernelsPPL.sampling_distribution_graph(::Type{Distributions.DirichletMultinomial}, args::Tuple)
    length(args) == 2 || return nothing
    _graph_call(:dirichlet_multinomial, args...)
end

# ── Laws for posterior-predictive draws ──────────────────────────────────
# Each observation's law cell (`ReactiveKernelsPPL._law_fields`) names the
# RK distribution object its density evaluates and that object's arguments:
# positional ones as `_1, _2, …`, keyword HAVE ports by name. Each method
# below builds the Distributions.jl law with the same density; the evidence
# wrapper then applies the response's censoring or truncation. This is the
# one seam a later move away from Distributions.jl replaces.

function ReactiveKernelsPPL._predictive_law(cell::NamedTuple)
    law = _rk_law(cell.law, cell)
    return hasproperty(cell, :evidence) ? _evidence(cell.evidence, law, cell) : law
end

_rk_law(::Val{name}, cell) where {name} = throw(ArgumentError(
    "[query] there is no Distributions.jl law for the RK distribution object " *
    "`$name`, so its observations cannot be drawn yet"))

# A positional argument, or the first of the keyword ports present.
function _arg(cell, position::Int, ports::Symbol...)
    p = Symbol(:_, position)
    hasproperty(cell, p) && return getproperty(cell, p)
    for port in ports
        hasproperty(cell, port) && return getproperty(cell, port)
    end
    return nothing
end

_rk_law(::Val{:normal}, c) = Distributions.Normal(c._1, c._2)
_rk_law(::Val{:lognormal}, c) = Distributions.LogNormal(c._1, c._2)
_rk_law(::Val{:cauchy}, c) = Distributions.Cauchy(c._1, c._2)
_rk_law(::Val{:laplace}, c) = Distributions.Laplace(c._1, c._2)
_rk_law(::Val{:logistic}, c) = Distributions.Logistic(c._1, c._2)
_rk_law(::Val{:student_t}, c) = c._2 + c._3 * Distributions.TDist(c._1)
_rk_law(::Val{:exponential}, c) = Distributions.Exponential(c._1)
_rk_law(::Val{:uniform}, c) = Distributions.Uniform(c._1, c._2)
_rk_law(::Val{:beta}, c) = Distributions.Beta(c._1, c._2)
_rk_law(::Val{:inverse_gamma}, c) = Distributions.InverseGamma(c._1, c._2)
_rk_law(::Val{:gumbel}, c) = Distributions.Gumbel(c._1, c._2)
_rk_law(::Val{:rayleigh}, c) = Distributions.Rayleigh(c._1)
_rk_law(::Val{:chi}, c) = Distributions.Chi(c._1)
_rk_law(::Val{:chi_squared}, c) = Distributions.Chisq(c._1)
_rk_law(::Val{:inverse_gaussian}, c) = Distributions.InverseGaussian(c._1, c._2)
_rk_law(::Val{:beta_prime}, c) = Distributions.BetaPrime(c._1, c._2)
_rk_law(::Val{:f_dist}, c) = Distributions.FDist(c._1, c._2)

# `gamma(shape, rate)`: Distributions.jl takes the scale.
function _rk_law(::Val{:gamma}, c)
    shape = _arg(c, 1, :shape)
    rate = _arg(c, 2, :rate)
    scale = rate !== nothing ? inv(rate) :
        hasproperty(c, :log_rate) ? exp(-c.log_rate) : c.scale
    return Distributions.Gamma(shape, scale)
end

function _rk_law(::Val{:weibull}, c)
    theta = _arg(c, 2, :theta)
    theta === nothing && (theta = exp(c.log_theta))
    return Distributions.Weibull(_arg(c, 1, :k), theta)
end

function _rk_law(::Val{:poisson}, c)
    rate = _arg(c, 1, :rate)
    return Distributions.Poisson(rate === nothing ? exp(c.log_rate) : rate)
end

_probability(c, position) = (p = _arg(c, position, :p); p === nothing ? logistic(c.logit) : p)

_rk_law(::Val{:bernoulli}, c) = Distributions.Bernoulli(_probability(c, 1))
_rk_law(::Val{:binomial}, c) = Distributions.Binomial(_arg(c, 1, :n), _probability(c, 2))
_rk_law(::Val{:beta_binomial}, c) = Distributions.BetaBinomial(c._1, c._2, c._3)
_rk_law(::Val{:negative_binomial}, c) = Distributions.NegativeBinomial(c._1, c._2)

# `negative_binomial2(mu, phi)`: mean mu, variance mu + mu^2 / phi.
function _rk_law(::Val{:negative_binomial2}, c)
    mu, phi = _arg(c, 1, :mu), _arg(c, 2, :phi)
    return Distributions.NegativeBinomial(phi, phi / (phi + mu))
end

# Zero inflation is a structural zero with probability zi.
_zero_inflated(count, zi) =
    Distributions.MixtureModel([Distributions.Dirac(0), count], [zi, 1 - zi])

function _rk_law(::Val{:zero_inflated_poisson}, c)
    rate = _arg(c, 1, :rate)
    rate === nothing && (rate = exp(c.log_rate))
    return _zero_inflated(Distributions.Poisson(rate), _arg(c, 2, :zi))
end

function _rk_law(::Val{:zero_inflated_binomial}, c)
    n, p = _arg(c, 1, :n), _arg(c, 2, :p)
    p === nothing && (p = logistic(c.logit))
    return _zero_inflated(Distributions.Binomial(n, p), _arg(c, 3, :zi))
end

_rk_law(::Val{:zero_inflated_binomial_logit_zi}, c) = _zero_inflated(
    Distributions.Binomial(c.n, logistic(c.logit)), logistic(c.zi_logit))

# RKPPL's `censored` is the clamp law and `truncated` renormalizes the base
# law on the inclusive bounds: Distributions.jl's `censored` and `truncated`.
_evidence(::Val{:censored}, law, c) = Distributions.censored(law;
    lower = get(c, :lower, nothing), upper = get(c, :upper, nothing))
_evidence(::Val{:truncated}, law, c) = Distributions.truncated(law;
    lower = get(c, :lower, nothing), upper = get(c, :upper, nothing))

end
