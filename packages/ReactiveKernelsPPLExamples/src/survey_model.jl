module SurveyModelExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export SURVEY_NS, SURVEY_LC, SURVEY_SK, SURVEY_M, SURVEY_LOG1NMAX
export SURVEY_NMAX, SURVEY_K
export build_survey_model_graph, demo
export SURVEY_MODEL_SOURCE, evaluate_survey_model_source

# posteriordb `Survey_data-Survey_model` — "Inferring a return rate and the number
# of surveys from observed returns" (Lee & Wagenmakers). The number of surveys n
# is a DISCRETE latent, so Stan marginalizes it: the likelihood is a mixture over
# n = 1..nmax,
#   target += log_sum_exp_n( log(1/nmax) + binomial_lpmf(k | n, theta) ),
# where k is the vector of per-group observed returns and binomial_lpmf(k|n,theta)
# = Σᵢ [ logC(n,kᵢ) + kᵢ·log(theta) + (n-kᵢ)·log(1-theta) ]. Only n ≥ nmin = max(k)
# has support; n < nmin contributes log(0) = -Inf (each such term has some kᵢ > n).
# theta ∈ [0,1] has an implicit uniform(0,1) prior (no density term).
#
# The per-n log-binomial-coefficient sum LC[n] = Σᵢ logC(n,kᵢ) is DATA-only, so it
# is precomputed once (via log-factorials — C(500,27) overflows Int64 — with
# LC[n] = -Inf for n < nmin). Then the whole marginalization is a data-parallel
# `log_sum_exp` over the length-nmax vector of per-n log-probabilities. Real data
# is nmax=500, m=5, k=[16,18,22,25,27] (full, embedded).
const SURVEY_NMAX = 500
const SURVEY_K = [16, 18, 22, 25, 27]
const SURVEY_M = length(SURVEY_K)

# log(x!) via log-factorial (Base only; = lgamma(x+1)); log C(n,k) with the
# out-of-support (k>n) case as -Inf, matching binomial_lpmf(kᵢ|n,·) = log(0).
_survey_logfact(x::Int) = x <= 1 ? 0.0 : sum(log, 2:x)
function _survey_lchoose(n::Int, k::Int)
    k > n && return -Inf
    _survey_logfact(n) - _survey_logfact(k) - _survey_logfact(n - k)
end

# Transformed-data vectors/scalars (mirror Stan's `transformed data` + the
# n-independent parts of `binomial_lpmf`):
const SURVEY_NS = Float64.(1:SURVEY_NMAX)                       # the n values 1..nmax
const SURVEY_LC = Float64[sum(_survey_lchoose(n, ki) for ki in SURVEY_K)
                          for n in 1:SURVEY_NMAX]              # Σᵢ logC(n,kᵢ); -Inf for n<nmin
const SURVEY_SK = Float64(sum(SURVEY_K))                        # Σᵢ kᵢ
const SURVEY_LOG1NMAX = -log(Float64(SURVEY_NMAX))             # log(1/nmax)

const SURVEY_MODEL_SOURCE = raw"""
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              ns::Vector{Float64},
              lc::Vector{Float64},
              sk::Float64,
              m::Int,
              log_1_nmax::Float64) = begin
    # q = (u_theta); dim = 1. theta ∈ [0,1] via the logistic transform, with the
    # `lub_constrain` Jacobian log|dtheta/du| = -log1pexp(-u) - log1pexp(u).
    u_theta::Float64 = sum(view(unconstrained, 1:1))
    theta::Float64 = logistic(u_theta)
    log_jacobian::Float64 = -log1pexp(-u_theta) - log1pexp(u_theta)
    # log(theta) and log(1-theta) straight from the unconstrained value (no round trip).
    logtheta::Float64 = -log1pexp(-u_theta)
    log1mtheta::Float64 = -log1pexp(u_theta)

    parameters = (; theta)
    (parameters, log_jacobian::Float64) =
        ((; theta), -log1pexp(-u_theta) - log1pexp(u_theta))
    theta::Float64 = parameters.theta

    # theta ~ uniform(0,1) — implicit, prior density is a dropped constant.
    log_prior::Float64 = 0.0

    # Per-n log-probability of the marginalization mixture:
    #   lp_parts[n] = log(1/nmax) + binomial_lpmf(k | n, theta)
    #              = log_1_nmax + LC[n] + SK·log(theta) + (n·m - SK)·log(1-theta).
    # LC[n] = -Inf for n < nmin, so those cells are -Inf (zero probability). The n
    # value, LC[n] ride the plate per-cell; the theta-dependent scalars + constants
    # ride as shared plate args.
    lp_parts = plate(ns, lc, sk, m, logtheta, log1mtheta, log_1_nmax) do n, l, s, mm, lt, l1t, c0
        c0 + l + s * lt + (n * mm - s) * l1t
    end

    # Marginal likelihood = log_sum_exp over n, max-stabilized (matches Stan's
    # `log_sum_exp`): mx = maxₙ lp_parts; Σₙ exp(lp_parts - mx) then mx + log(·).
    # The -Inf cells contribute exp(-Inf)=0 to the sum.
    mx::Float64 = maximum(lp_parts)
    exp_parts = plate(lp_parts, mx) do lp, m0
        exp(lp - m0)
    end
    likelihood::Float64 = mx + log(sum(exp_parts))

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.0]
ns = SURVEY_NS
lc = SURVEY_LC
sk = SURVEY_SK
m = SURVEY_M
log_1_nmax = SURVEY_LOG1NMAX

requested_nodes = (:parameters, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :ns, :lc, :sk, :m, :log_1_nmax),
    want = requested_nodes,
    bound = (; ns = SURVEY_NS, lc = SURVEY_LC, sk = SURVEY_SK,
              m = SURVEY_M, log_1_nmax = SURVEY_LOG1NMAX))

output = density_kernel(q)
parameters, likelihood, posterior = output
@assert isfinite(posterior)

docs_example = (;
    name = :survey_posterior,
    origin = "posteriordb Survey_model — inferring return rate + survey count (discrete-n marginalization)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
)
"""

function evaluate_survey_model_source()
    _evaluate_ppl_source(SURVEY_MODEL_SOURCE, @__MODULE__; bindings = (
        :SURVEY_NS, :SURVEY_LC, :SURVEY_SK, :SURVEY_M, :SURVEY_LOG1NMAX,
    ))
end

const _SURVEY_MODEL_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _SURVEY_MODEL_GRAPH_TEMPLATE[] = evaluate_survey_model_source().model
    nothing
end

"""
    build_survey_model_graph()

Build the posteriordb `Survey_model` (inferring a return rate `theta` and the
discrete number of surveys `n` from observed per-group returns `k`) as a
declarative `ReactiveKernels.KernelSpec`. The discrete latent `n` is marginalized:
the likelihood is a `log_sum_exp` over `n = 1..nmax` of
`log(1/nmax) + binomial_lpmf(k | n, theta)`, with the out-of-support cells
(`n < max(k)`) carrying `-Inf`. `theta ∈ [0,1]` uses the logistic transform with
its Jacobian (implicit uniform prior). The per-`n` log-binomial-coefficient sums
are precomputed data; the marginalization is a max-stabilized data-parallel
reduction. Named nodes for the `parameters` NamedTuple, the marginal likelihood,
densities and the posterior.
"""
function build_survey_model_graph()
    compose(_SURVEY_MODEL_GRAPH_TEMPLATE[])
end

function demo()
    model = build_survey_model_graph()
    q = [0.0]
    posterior_kernel = prepare(model;
        have = (:unconstrained, :ns, :lc, :sk, :m, :log_1_nmax), want = :posterior,
        bound = (; ns = SURVEY_NS, lc = SURVEY_LC, sk = SURVEY_SK,
                  m = SURVEY_M, log_1_nmax = SURVEY_LOG1NMAX))
    println("Survey unconstrained log posterior = ", posterior_kernel(q))
    nothing
end

end # module SurveyModelExample

if abspath(PROGRAM_FILE) == @__FILE__
    SurveyModelExample.demo()
end
