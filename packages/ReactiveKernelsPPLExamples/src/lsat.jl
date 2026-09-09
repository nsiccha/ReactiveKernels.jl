module LsatExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export LSAT_RESPONSE, LSAT_RESP_FLAT, LSAT_STUDENT_IDX, LSAT_QUESTION_IDX
export build_lsat_graph, demo
export LSAT_SOURCE, evaluate_lsat_source

# posteriordb `lsat_data-lsat_model` — the classic BUGS "LSAT" Rasch / 1-PL IRT
# model (Section 6 of the Law School Admission Test analysis): each of T = 5
# binary questions is answered by N students; r[k,j] ~ Bernoulli_logit(beta *
# theta[j] - alpha[k]) with a per-student ability theta[j] ~ Normal(0, 1), a
# per-question difficulty alpha[k] ~ Normal(0, 100), and a shared discrimination
# beta ~ Normal(0, 100), beta > 0.
#
# The full real dataset is N = 1000 students collapsed into R = 32 distinct
# response patterns (2^5) via a `culm` cumulative-count vector. Embedded here is
# a faithfully-shaped REPRESENTATIVE subset: exactly ONE student per distinct
# response pattern, i.e. N = 32 students spanning every one of the 32 possible
# response patterns (the real response matrix, verbatim). This keeps the full
# IRT response space and per-student/per-question structure without shipping
# 1000 students; the per-pattern frequency weighting (the only thing dropped) is
# what `culm` encoded.
const LSAT_RESPONSE = [
    [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1],
    [0, 0, 1, 0, 0], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, 0], [0, 1, 0, 1, 1],
    [0, 1, 1, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0], [1, 0, 0, 1, 1],
    [1, 0, 1, 0, 0], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0], [1, 0, 1, 1, 1],
    [1, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 0, 1, 0], [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 0], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1],
]

const _LSAT_N = length(LSAT_RESPONSE)      # 32 students (one per pattern)
const _LSAT_T = length(LSAT_RESPONSE[1])   # 5 questions

# Flatten the 2-D (question k, student j) grid to 1-D, question-outer /
# student-inner, matching Stan's `for (k in 1:T) r[k] ~ bernoulli_logit(...)`.
# Each flat cell carries its response and its 1-based (student, question) gather
# indices, so RK reads theta[student] and alpha[question] by concrete index.
const LSAT_RESP_FLAT = Int[LSAT_RESPONSE[j][k] for k in 1:_LSAT_T for j in 1:_LSAT_N]
const LSAT_STUDENT_IDX = Int[j for k in 1:_LSAT_T for j in 1:_LSAT_N]
const LSAT_QUESTION_IDX = Int[k for k in 1:_LSAT_T for j in 1:_LSAT_N]

const LSAT_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, binomial
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              student_idx::Vector{Int},
              question_idx::Vector{Int},
              response::Vector{Int}) = begin
    # Stan's declared unconstrained order: (alpha[1..T], theta[1..N], log_beta);
    # dim = T + N + 1. LSAT has T = 5 questions (structural constant of the
    # test), so n_students = dim - 6. alpha and theta are unconstrained; only
    # beta is constrained (real<lower=0>).
    n_students::Int = length(unconstrained) - 6
    alpha::AbstractVector{Float64} = view(unconstrained, 1:5)
    theta::AbstractVector{Float64} = view(unconstrained, 6:n_students + 5)
    u_beta::Float64 = unconstrained[n_students + 6]

    # beta = exp(log_beta); log|dbeta/dlog_beta| = log_beta (Stan's `lb_constrain`).
    beta::Float64 = exp(u_beta)
    log_jacobian::Float64 = u_beta

    parameters = (; alpha, theta, beta)
    (parameters, log_jacobian::Float64) = ((; alpha, theta, beta), u_beta)
    (alpha::AbstractVector{Float64}, theta::AbstractVector{Float64}, beta::Float64) =
        (parameters.alpha, parameters.theta, parameters.beta)

    # Priors: alpha_k ~ Normal(0, 100), theta_j ~ Normal(0, 1),
    # beta ~ Normal(0, 100) (half-normal via beta>0; Stan adds the plain
    # normal_lpdf and drops the log2 constant, so gradient/value parity hold).
    alpha_pointwise = plate(alpha) do a
        normal(0.0, 100.0).logpdf(a)
    end
    alpha_prior::Float64 = sum(alpha_pointwise)
    theta_pointwise = plate(theta) do t
        normal(0.0, 1.0).logpdf(t)
    end
    theta_prior::Float64 = sum(theta_pointwise)
    beta_prior::Float64 = normal(0.0, 100.0).logpdf(beta)
    prior::Float64 = alpha_prior + theta_prior + beta_prior

    # 2-D IRT structure flattened to 1-D via concrete-index GATHERS done OUTSIDE
    # any plate: each (question k, student j) cell reads theta[j] and alpha[k].
    # A traced integer index does not lower, so docs_example binds both indices;
    # then the gathers trace as concrete gathers and the Reactant path sees only
    # the float parameters + the integer response observation.
    theta_flat = theta[student_idx]
    alpha_flat = alpha[question_idx]

    # Transformed parameter: the logit-scale linear predictor beta*theta - alpha
    # (named node + generated quantity).
    logit_p = plate(theta_flat, alpha_flat, beta) do th, al, be
        be * th - al
    end

    # Likelihood: r[k,j] ~ Bernoulli_logit(beta*theta[j] - alpha[k]). Bernoulli
    # is Binomial(1, p), so the Int 0/1 response uses the shared Binomial
    # endpoint with the logit link applied inline (buffer-free fused total).
    pointwise = plate(response, theta_flat, alpha_flat, beta) do r, th, al, be
        binomial(1, logistic(be * th - al)).logpdf(r)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantities: the centered difficulties a[k] = alpha[k] -
    # mean(alpha) (mean over T = 5 questions rides the plate as a shared arg).
    mean_alpha::Float64 = sum(alpha) / 5.0
    a = plate(alpha, mean_alpha) do al, ma
        al - ma
    end

    return posterior
end

q = vcat(0.1 .* collect(1:5), _LSAT_THETA_Q, [0.3])
student_idx = LSAT_STUDENT_IDX
question_idx = LSAT_QUESTION_IDX
response = LSAT_RESP_FLAT

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
# Bind the integer gather indices so `a.inputs` excludes them and the Reactant
# path traces only the float parameters + the integer response (spec addendum).
density_kernel = prepare(model;
    have = (:unconstrained, :student_idx, :question_idx, :response),
    want = requested_nodes,
    bound = (; student_idx, question_idx))

output = density_kernel(q, response)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :lsat_posterior,
    origin = "posteriordb lsat_model — Rasch / 1-PL IRT model (representative N=32 subset)",
    inputs = (; q, response),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    binomial_object = binomial,
)
"""

# Fixed, reproducible ability draws for the docs-example `q` (no RNG in the
# source string, so the authored source stays a pure declaration); bound in.
const _LSAT_THETA_Q = 0.05 .* Float64[
    0.9, -0.4, 0.1, 0.7, -1.1, 0.3, -0.2, 0.8, -0.6, 0.5,
    1.2, -0.3, 0.4, -0.9, 0.2, 0.6, -0.7, 1.0, -0.1, 0.35,
    -0.5, 0.15, 0.45, -0.85, 0.25, 0.55, -0.65, 0.95, -0.15, 0.4,
    -0.45, 0.2,
]

function evaluate_lsat_source(; model_only::Bool = false)
    _evaluate_ppl_source(LSAT_SOURCE, @__MODULE__; bindings = (
        :LSAT_STUDENT_IDX, :LSAT_QUESTION_IDX, :LSAT_RESP_FLAT,
        :_LSAT_THETA_Q,
    ), model_only)
end

const _LSAT_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LSAT_GRAPH_TEMPLATE[] = evaluate_lsat_source(; model_only = true).model
    nothing
end

"""
    build_lsat_graph()

Build the posteriordb `lsat_model` (a Rasch / 1-PL IRT model,
`r[k,j] ~ Bernoulli_logit(beta*theta[j] - alpha[k])`) as a declarative
`ReactiveKernels.KernelSpec`. The 2-D question x student response grid is
flattened to 1-D and assembled by two concrete-index gathers (theta[student],
alpha[question]) outside any plate; `beta` uses the exact `exp` support
transform and Jacobian; the Normal priors and the Bernoulli (= Binomial(1, .))
likelihood reuse the shared endpoints. The transform Jacobian, priors,
transformed `logit_p`, pointwise/summed likelihood, densities, posterior, and
the centered difficulties `a` are separate named nodes. Uses a representative
N = 32 subset (one student per response pattern).
"""
function build_lsat_graph()
    compose(_LSAT_GRAPH_TEMPLATE[])
end

function demo()
    model = build_lsat_graph()
    q = vcat(0.1 .* collect(1:5), _LSAT_THETA_Q, [0.3])
    posterior_plan = plan(model;
                          have = (:unconstrained, :student_idx, :question_idx, :response),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, LSAT_STUDENT_IDX, LSAT_QUESTION_IDX, LSAT_RESP_FLAT)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module LsatExample

if abspath(PROGRAM_FILE) == @__FILE__
    LsatExample.demo()
end
