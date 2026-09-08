module DiamondsExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export DIAMONDS_X, DIAMONDS_Y, DIAMONDS_PRIOR_ONLY
export build_diamonds_graph, demo
export DIAMONDS_SOURCE, evaluate_diamonds_source

# posteriordb `diamonds-diamonds` — a brms 2.10.0 Gaussian linear regression of
# log-price on 24 population-level effects (the `diamonds` dataset from ggplot2),
# with the brms centered-intercept parameterization. The generated Stan centers
# the design matrix in `transformed data`: it drops the intercept column and
# subtracts each remaining column's mean, so the model samples a temporary
# `Intercept` on the centered scale plus `b[1..Kc]`, and reports the actual
# population intercept `b_Intercept = Intercept - dot(means_X, b)` as a generated
# quantity. The likelihood is the fused `normal_id_glm_lpdf(Y | Xc, Intercept, b,
# sigma)` (i.e. Yᵢ ~ Normal(Intercept + Xcᵢ·b, sigma)).
#
# Priors (brms defaults, PROPER, so every constant shows in value parity):
#   b         ~ Normal(0, 1)                                  (each of Kc = 24)
#   Intercept ~ Student_t(3, 8, 10)
#   sigma     ~ Student_t(3, 0, 10) truncated to sigma > 0    (the `- student_t_lccdf(0 | 3, 0, 10)`
#              normalization is the constant log(2), since the t is symmetric about 0)
#
# Real, FULL data (N = 5000, K = 25) loaded from posteriordb via PosteriorDB.jl —
# X is the raw brms design matrix whose first column is the all-ones intercept, so
# the column-drop + centering is a DATA-ONLY prefix authored IN-GRAPH and hoisted
# by binding the raw `X` port (`bound = X`), exercising RK's partial evaluation.
let d = _posteriordb_data("diamonds-diamonds")
    global const DIAMONDS_X = Float64.(d["X"])
    global const DIAMONDS_Y = Float64.(d["Y"])
    global const DIAMONDS_PRIOR_ONLY = Int(d["prior_only"])   # 0 for this posterior
end

const DIAMONDS_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, student_t

@kernel model(unconstrained::Vector{Float64},
              X::Matrix{Float64},
              Y::Vector{Float64},
              prior_only::Int) = begin
    # Data-only centering prefix (Stan's `transformed data`). It reads only `X`,
    # so binding the raw design-matrix port runs the column-drop + centering once
    # at preparation and hoists the centered matrix `Xc` and column means into the
    # residual kernel as constants. Xc drops the intercept column (column 1 of the
    # brms design matrix) and subtracts each remaining column's mean.
    n_obs::Int = size(X, 1)
    Xnoint::Matrix{Float64} = X[:, 2:size(X, 2)]
    col_means::Matrix{Float64} = sum(Xnoint; dims = 1) ./ n_obs
    Xc::Matrix{Float64} = Xnoint .- col_means
    means_X::Vector{Float64} = vec(col_means)

    # Unconstrained layout q = (b[1..Kc], Intercept, log_sigma), matching the Stan
    # parameter declaration order `vector[Kc] b; real Intercept; real<lower=0> sigma`.
    # Only sigma has a support transform (sigma = exp(log_sigma)); b and the centered
    # Intercept are unconstrained reals.
    n_coef::Int = length(unconstrained) - 2
    b::AbstractVector{Float64} = view(unconstrained, 1:n_coef)
    Intercept::Float64 = unconstrained[n_coef + 1]
    log_sigma::Float64 = unconstrained[n_coef + 2]
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    # Two producers for the same `parameters` port and inverse edges exposing its
    # components — the constrain-only producer omits the Jacobian; the joint producer
    # emits it (the log|dsigma/du| = log_sigma change of variables).
    parameters = (; b, Intercept, sigma)
    (parameters, log_jacobian::Float64) = ((; b, Intercept, sigma), log_sigma)
    (b::AbstractVector{Float64}, Intercept::Float64, sigma::Float64) =
        (parameters.b, parameters.Intercept, parameters.sigma)

    # Priors (all PROPER, propto=false constants included). The half-Student-t on
    # sigma adds the explicit brms normalization `- student_t_lccdf(0 | 3, 0, 10)`
    # = -log(1/2) = log(2) (the Student-t is symmetric about its location 0).
    b_pointwise = plate(b) do coefficient
        normal(0.0, 1.0).logpdf(coefficient)
    end
    b_prior::Float64 = sum(b_pointwise)
    intercept_prior::Float64 = student_t(3.0, 8.0, 10.0).logpdf(Intercept)
    sigma_prior::Float64 = log(2.0) + student_t(3.0, 0.0, 10.0).logpdf(sigma)
    log_prior::Float64 = b_prior + intercept_prior + sigma_prior

    # Fitted mean on the centered scale: μ = Intercept + Xc·b (the fused
    # normal_id_glm linear predictor). Consumes the centered design once.
    mu::Vector{Float64} = Intercept .+ Xc * b

    # Likelihood: Yᵢ ~ Normal(μᵢ, sigma). The Stan model adds it only when the
    # data flag `prior_only` is 0 (`if (!prior_only) target += normal_id_glm_lpdf`),
    # so the data-directed choice is a bound port: `included_likelihood` SELECTS the
    # likelihood when prior_only == 0 and 0 otherwise, via `ifelse`. Binding
    # prior_only makes only that SELECTION a compile-time constant (the posterior
    # data has prior_only = 0, so the likelihood is always selected); `ifelse`
    # evaluates BOTH arguments, so the likelihood itself is still computed — the
    # bound flag chooses which contribution enters the density, it does not prune
    # the likelihood computation.
    pointwise = plate(Y, mu, sigma) do y, m, s
        normal(m, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)
    included_likelihood::Float64 = ifelse(prior_only == 0, likelihood, 0.0)

    constrained_logdensity::Float64 = log_prior + included_likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the actual (uncentered) population intercept,
    # b_Intercept = Intercept - dot(means_X, b) (Stan's `generated quantities`).
    b_Intercept::Float64 = Intercept - sum(means_X .* b)

    return posterior
end

q = vcat(fill(0.05, 24), 8.0, log(0.6))
X = DIAMONDS_X
Y = DIAMONDS_Y
prior_only = DIAMONDS_PRIOR_ONLY

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :X, :Y, :prior_only),
    want = requested_nodes,
    bound = (; X, prior_only))

output = density_kernel(q, Y)
parameters, log_prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :diamonds_posterior,
    origin = "posteriordb diamonds — brms Gaussian regression with a centered design matrix",
    inputs = (; q, Y),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    student_t_object = student_t,
)
"""

function evaluate_diamonds_source(; model_only::Bool = false)
    # Bind only the data. The authored source imports the reusable Normal and
    # Student-t endpoints itself and authors the centering prefix inline.
    _evaluate_ppl_source(DIAMONDS_SOURCE, @__MODULE__; bindings = (
        :DIAMONDS_X, :DIAMONDS_Y, :DIAMONDS_PRIOR_ONLY,
    ), model_only)
end

const _DIAMONDS_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _DIAMONDS_GRAPH_TEMPLATE[] = evaluate_diamonds_source(; model_only = true).model
    nothing
end


"""
    build_diamonds_graph()

Build the posteriordb `diamonds` model (a brms Gaussian linear regression with a
centered design matrix) as a declarative `ReactiveKernels.KernelSpec`. The
intercept-column drop and per-column centering are a data-only prefix (they read
only the raw design matrix `X`), so binding that port through the public `bound`
kwarg of `prepare` runs them once at preparation and hoists the centered matrix
`Xc` into the residual kernel as a constant. `sigma` has the `exp` support
transform with its Jacobian, the coefficient/Intercept/sigma priors reuse the
shared Normal and Student-t endpoints (the half-Student-t on sigma carries the
explicit `log(2)` normalization), and the fitted mean, pointwise/summed
likelihood, densities, unconstrained posterior, and the generated-quantity
population intercept `b_Intercept` are separate named nodes.
"""
function build_diamonds_graph()
    compose(_DIAMONDS_GRAPH_TEMPLATE[])
end

function demo()
    model = build_diamonds_graph()
    q = vcat(fill(0.05, 24), 8.0, log(0.6))
    posterior_plan = plan(model;
                          have = (:unconstrained, :X, :Y, :prior_only),
                          want = (:log_prior, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_prior, likelihood, posterior =
        prepare(posterior_plan)(q, DIAMONDS_X, DIAMONDS_Y, DIAMONDS_PRIOR_ONLY)
    println("log prior + log likelihood = ", log_prior, " + ", likelihood,
            " = ", posterior)
    nothing
end

end # module DiamondsExample

if abspath(PROGRAM_FILE) == @__FILE__
    DiamondsExample.demo()
end
