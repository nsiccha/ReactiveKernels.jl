module SeedsCenteredExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export SEEDS_CENTERED_COUNTS, SEEDS_CENTERED_TOTALS,
       SEEDS_CENTERED_X1, SEEDS_CENTERED_X2
export build_seeds_centered_model_graph, demo
export SEEDS_CENTERED_MODEL_SOURCE, evaluate_seeds_centered_model_source

# posteriordb `seeds_data-seeds_centered_model` — the BUGS "seeds" binomial-logit
# GLMM (Crowder 1978) with narrow N(0,1) fixed-effect priors, a half-Cauchy(0,1)
# scale, and a MEAN-CENTERED per-plate random effect `b = c - mean(c)` (the raw
# effect `c ~ Normal(0, sigma)` carries the prior; the centered `b` enters the
# linear predictor). I = 21 plates; 2x2 design (seed type x1, root extract x2)
# with interaction. The full real dataset (I = 21) is embedded verbatim.
const SEEDS_CENTERED_COUNTS = [
    10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3,
]
const SEEDS_CENTERED_TOTALS = [
    39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7,
]
const SEEDS_CENTERED_X1 = Float64[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]
const SEEDS_CENTERED_X2 = Float64[
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
]

const SEEDS_CENTERED_MODEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy, binomial
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              counts::Vector{Int},
              totals::Vector{Int},
              x1::Vector{Float64},
              x2::Vector{Float64}) = begin
    # Stan's declared unconstrained order: (alpha0, alpha1, alpha12, alpha2,
    # c[1..I], log_sigma); D = I + 5. NOTE alpha12 precedes alpha2. Only sigma is
    # constrained (real<lower=0>); the four fixed effects and the raw effect c are
    # unconstrained. Slice without scalar indexing so the same prepared kernel
    # stays traceable as a Reactant tensor program.
    n_obs::Int = length(unconstrained) - 5
    alpha0::Float64 = sum(view(unconstrained, 1:1))
    alpha1::Float64 = sum(view(unconstrained, 2:2))
    alpha12::Float64 = sum(view(unconstrained, 3:3))
    alpha2::Float64 = sum(view(unconstrained, 4:4))
    c::AbstractVector{Float64} = view(unconstrained, 5:n_obs + 4)
    u_sigma::Float64 = sum(view(unconstrained, n_obs + 5:n_obs + 5))

    # sigma = exp(u_sigma); log|dsigma/du_sigma| = u_sigma (Stan's `lb_constrain`).
    # The four fixed effects and c are unconstrained (identity, zero Jacobian).
    sigma::Float64 = exp(u_sigma)
    log_jacobian::Float64 = u_sigma

    parameters = (; alpha0, alpha1, alpha12, alpha2, c, sigma)
    (parameters, log_jacobian::Float64) =
        ((; alpha0, alpha1, alpha12, alpha2, c, sigma), u_sigma)
    (alpha0::Float64, alpha1::Float64, alpha12::Float64, alpha2::Float64,
     c::AbstractVector{Float64}, sigma::Float64) =
        (parameters.alpha0, parameters.alpha1, parameters.alpha12,
         parameters.alpha2, parameters.c, parameters.sigma)

    # Priors: alpha0, alpha1, alpha2, alpha12 ~ Normal(0, 1); sigma ~ Cauchy(0, 1)
    # (half-Cauchy — the plain cauchy_lpdf density term plus the exp transform
    # Jacobian; Stan drops the truncation log2 constant, so value/gradient parity
    # hold); c ~ Normal(0, sigma). All proper, so each contributes a real term.
    alpha0_prior::Float64 = normal(0.0, 1.0).logpdf(alpha0)
    alpha1_prior::Float64 = normal(0.0, 1.0).logpdf(alpha1)
    alpha2_prior::Float64 = normal(0.0, 1.0).logpdf(alpha2)
    alpha12_prior::Float64 = normal(0.0, 1.0).logpdf(alpha12)
    sigma_prior::Float64 = cauchy(0.0, 1.0).logpdf(sigma)
    fixed_prior::Float64 =
        alpha0_prior + alpha1_prior + alpha2_prior + alpha12_prior + sigma_prior

    # Raw random-effect prior: c_j ~ Normal(0, sigma). sigma rides the plate as a
    # shared scalar arg (a scalar plate arg broadcasts across cells).
    c_pointwise = plate(c, sigma) do cc, s
        normal(0.0, s).logpdf(cc)
    end
    c_prior::Float64 = sum(c_pointwise)
    prior::Float64 = fixed_prior + c_prior

    # Transformed parameter: b = c - mean(c) (mean over the I elements — a scalar
    # reduction over c's own length, so it is derivable from `parameters` alone).
    # Named node; the likelihood recomputes (c - c_mean) inline.
    c_mean::Float64 = sum(c) / length(c)
    b = c .- c_mean

    # Transformed parameter: the logit-scale linear predictor with interaction
    # (Stan's transformed-data x1x2 = x1 .* x2 recomputed inline as x1*x2), using
    # the centered random effect (c - c_mean).
    logit_p = plate(x1, x2, c, c_mean, alpha0, alpha1, alpha2, alpha12) do xx1, xx2, cc, cm, a0, a1, a2, a12
        a0 + a1 * xx1 + a2 * xx2 + a12 * (xx1 * xx2) + (cc - cm)
    end

    # Likelihood: counts_j ~ Binomial_logit(totals_j, eta_j). The linear predictor
    # is recomputed inline (buffer-free fused total); the Binomial endpoint takes
    # the success probability, so the logit link is `logistic(.)`.
    pointwise = plate(counts, totals, x1, x2, c, c_mean, alpha0, alpha1, alpha2, alpha12) do cnt, nt, xx1, xx2, cc, cm, a0, a1, a2, a12
        binomial(nt, logistic(a0 + a1 * xx1 + a2 * xx2 + a12 * (xx1 * xx2) + (cc - cm))).logpdf(cnt)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the per-plate germination probabilities p = inv_logit(eta).
    p = plate(logit_p) do lp
        logistic(lp)
    end

    return posterior
end

q = vcat([0.2, 0.1, -0.05, 0.03], fill(0.0, length(SEEDS_CENTERED_COUNTS)), [log(0.4)])
counts = SEEDS_CENTERED_COUNTS
totals = SEEDS_CENTERED_TOTALS
x1 = SEEDS_CENTERED_X1
x2 = SEEDS_CENTERED_X2

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :counts, :totals, :x1, :x2),
    want = requested_nodes)

output = density_kernel(q, counts, totals, x1, x2)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :seeds_centered_model_posterior,
    origin = "posteriordb seeds_centered_model — binomial-logit GLMM with a mean-centered per-plate random effect",
    inputs = (; q, counts, totals, x1, x2),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    cauchy_object = cauchy,
    binomial_object = binomial,
)
"""

function evaluate_seeds_centered_model_source()
    _evaluate_ppl_source(SEEDS_CENTERED_MODEL_SOURCE, @__MODULE__; bindings = (
        :SEEDS_CENTERED_COUNTS, :SEEDS_CENTERED_TOTALS,
        :SEEDS_CENTERED_X1, :SEEDS_CENTERED_X2,
    ))
end

const _SEEDS_CENTERED_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _SEEDS_CENTERED_GRAPH_TEMPLATE[] =
        evaluate_seeds_centered_model_source().model
    nothing
end

"""
    build_seeds_centered_model_graph()

Build the posteriordb `seeds_centered_model` (a binomial-logit GLMM with a 2x2
interaction design and a MEAN-CENTERED per-plate random effect `b = c - mean(c)`)
as a declarative `ReactiveKernels.KernelSpec`. The four fixed effects reuse the
`Normal(0, 1)` endpoint; the scale is a half-`Cauchy(0, 1)` (density term plus
the exact `exp` transform Jacobian); the raw effect prior is `c ~ Normal(0,
sigma)`; and the centered effect `b = c - mean(c)` enters the logit predictor,
recomputed inline in the likelihood. The transform Jacobian, priors, transformed
`b`/`logit_p`, pointwise/summed likelihood, densities, posterior, and the
germination probabilities `p` are separate named nodes.
"""
function build_seeds_centered_model_graph()
    compose(_SEEDS_CENTERED_GRAPH_TEMPLATE[])
end

function demo()
    model = build_seeds_centered_model_graph()
    q = vcat([0.2, 0.1, -0.05, 0.03],
             fill(0.0, length(SEEDS_CENTERED_COUNTS)), [log(0.4)])
    posterior_plan = plan(model;
                          have = (:unconstrained, :counts, :totals, :x1, :x2),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, SEEDS_CENTERED_COUNTS, SEEDS_CENTERED_TOTALS,
                                SEEDS_CENTERED_X1, SEEDS_CENTERED_X2)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module SeedsCenteredExample

if abspath(PROGRAM_FILE) == @__FILE__
    SeedsCenteredExample.demo()
end
