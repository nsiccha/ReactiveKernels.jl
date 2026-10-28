using Test, ReactiveKernelsPPL
import ReactiveKernels
import Distributions as EvidenceD

function _evidence_query(expr, data, q)
    bound = bind_data(lower_rkppl(expr, data; conditioned=(:y,)), data)
    built = build_kernel(bound)
    kernel = prepare_query(built, bound, :sampler)
    u = unconstrain(built.layout, q)
    return bound, built, kernel, u
end

_evidence_invlogit(a) = 1 / (1 + exp(-a))
const _EVIDENCE_FAMILY_CASES = [
    (:normal, :(Normal.(eta, 1.3)), a -> EvidenceD.Normal(a, 1.3), -1.0, 2.0, [0.1, 0.7]),
    (:student, :(StudentT.(5.0, eta, 1.3)), a -> a + 1.3EvidenceD.TDist(5), -1.0, 2.0, [0.1, 0.7]),
    (:lognormal, :(LogNormal.(eta, 0.7)), a -> EvidenceD.LogNormal(a, 0.7), 0.2, 2.4, [0.4, 1.2]),
    (:gamma, :(Gamma.(2.0, exp.(eta) ./ 2.0)), a -> EvidenceD.Gamma(2, exp(a)/2), 0.2, 2.4, [0.4, 1.2]),
    (:beta, :(Beta.(logistic.(eta) .* 4.0, (1 .- logistic.(eta)) .* 4.0)), a -> EvidenceD.Beta(4*_evidence_invlogit(a), 4*(1-_evidence_invlogit(a))), 0.1, 0.9, [0.3, 0.7]),
    (:weibull, :(Weibull.(2.0, exp.(eta))), a -> EvidenceD.Weibull(2, exp(a)), 0.2, 2.4, [0.4, 1.2]),
    (:exponential, :(Exponential.(exp.(eta))), a -> EvidenceD.Exponential(exp(a)), 0.2, 2.4, [0.4, 1.2]),
    (:inversegaussian, :(InverseGaussian.(exp.(eta), 2.0)), a -> EvidenceD.InverseGaussian(exp(a), 2), 0.2, 2.4, [0.4, 1.2]),
    (:vonmises, :(VonMises.(eta, 2.0)), a -> EvidenceD.VonMises(a, 2), -1.0, 1.0, [-0.4, 0.5]),
    (:bernoulli_logit, :(Bernoulli.(logistic.(eta))), a -> EvidenceD.Bernoulli(_evidence_invlogit(a)), 0.2, 1.0, [1]),
    (:bernoulli_probit, :(Bernoulli.(normcdf.(eta))), a -> EvidenceD.Bernoulli(EvidenceD.cdf(EvidenceD.Normal(), a)), 0.2, 1.0, [1]),
    (:bernoulli_cloglog, :(Bernoulli.(cexpexp.(eta))), a -> EvidenceD.Bernoulli(-expm1(-exp(a))), 0.2, 1.0, [1]),
    (:poisson, :(Poisson.(exp.(eta))), a -> EvidenceD.Poisson(exp(a)), 0.2, 3.6, [1, 2]),
    (:binomial_logit, :(Binomial.(5, logistic.(eta))), a -> EvidenceD.Binomial(5, _evidence_invlogit(a)), 0.2, 3.6, [1, 2]),
    (:binomial_probit, :(Binomial.(5, normcdf.(eta))), a -> EvidenceD.Binomial(5, EvidenceD.cdf(EvidenceD.Normal(), a)), 0.2, 3.6, [1, 2]),
    (:binomial_cloglog, :(Binomial.(5, cexpexp.(eta))), a -> EvidenceD.Binomial(5, -expm1(-exp(a))), 0.2, 3.6, [1, 2]),
    (:nb1, :(NegativeBinomial.(exp.(eta), 0.4)), a -> EvidenceD.NegativeBinomial(exp(a), 0.4), 0.2, 3.6, [1, 2]),
    (:nb2, :(NegativeBinomial2.(exp.(eta), 2.0)), a -> EvidenceD.NegativeBinomial(2, 2/(2+exp(a))), 0.2, 3.6, [1, 2]),
    (:betabinomial, :(BetaBinomial2.(5, logistic.(eta), 4.0)), a -> EvidenceD.BetaBinomial(5, 4*_evidence_invlogit(a), 4*(1-_evidence_invlogit(a))), 0.2, 3.6, [1, 2]),
    (:zip, :(ZeroInflatedPoisson.(exp.(eta), 0.2)), a -> EvidenceD.MixtureModel([EvidenceD.Dirac(0), EvidenceD.Poisson(exp(a))], [0.2, 0.8]), 0.2, 3.6, [1, 2]),
    (:hurdle, :(HurdlePoisson.(exp.(eta), 0.2)), a -> EvidenceD.MixtureModel([EvidenceD.Dirac(0), EvidenceD.truncated(EvidenceD.Poisson(exp(a)), 1, Inf)], [0.2, 0.8]), 0.2, 3.6, [1, 2]),
]
