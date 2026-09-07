module RatsModelExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export RATS_RAT, RATS_X, RATS_Y, RATS_XBAR
export build_rats_model_graph, demo
export RATS_MODEL_SOURCE, evaluate_rats_model_source

# posteriordb `rats_data-rats_model` — the classic BUGS "rats" hierarchical
# linear growth curve (30 rats weighed at 5 ages). Each rat has its own intercept
# alpha_j and slope beta_j drawn from population Normals; the three scale
# parameters have FLAT improper priors. The full real dataset is N = 30 rats over
# Npts = 150 points; embedded here verbatim is a faithfully-shaped representative
# subset — the first N = 8 rats across all 5 ages (Npts = 40), re-indexed 1..8,
# with the real ages x ∈ {8, 15, 22, 29, 36} and xbar = 22 (their exact mean).
const RATS_RAT = Int[
    1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7,
    8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6,
    7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
]
const RATS_X = Float64[
    8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
    15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
    22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
    29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0,
    36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0,
]
const RATS_Y = Float64[
    151.0, 145.0, 147.0, 155.0, 135.0, 159.0, 141.0, 159.0,
    199.0, 199.0, 214.0, 200.0, 188.0, 210.0, 189.0, 201.0,
    246.0, 249.0, 263.0, 237.0, 230.0, 252.0, 231.0, 248.0,
    283.0, 293.0, 312.0, 272.0, 280.0, 298.0, 275.0, 297.0,
    320.0, 354.0, 328.0, 297.0, 323.0, 331.0, 305.0, 338.0,
]
const RATS_XBAR = 22.0

const RATS_MODEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              rat::Vector{Int},
              x::Vector{Float64},
              y::Vector{Float64},
              xbar::Float64) = begin
    # Stan's declared unconstrained order: (alpha[1..N], beta[1..N], mu_alpha,
    # mu_beta, log_sigma_y, log_sigma_alpha, log_sigma_beta); D = 2N + 5. The
    # per-rat intercept vector comes FIRST, then the per-rat slope vector. alpha,
    # beta, mu_alpha, mu_beta are unconstrained; the three scales use the exp
    # support transform. Slice without scalar indexing so the same kernel stays
    # traceable as a Reactant program.
    n_rats::Int = div(length(unconstrained) - 5, 2)
    alpha::AbstractVector{Float64} = view(unconstrained, 1:n_rats)
    beta::AbstractVector{Float64} = view(unconstrained, n_rats + 1:2 * n_rats)
    mu_alpha::Float64 = sum(view(unconstrained, 2 * n_rats + 1:2 * n_rats + 1))
    mu_beta::Float64 = sum(view(unconstrained, 2 * n_rats + 2:2 * n_rats + 2))
    log_sigma_y::Float64 = sum(view(unconstrained, 2 * n_rats + 3:2 * n_rats + 3))
    log_sigma_alpha::Float64 = sum(view(unconstrained, 2 * n_rats + 4:2 * n_rats + 4))
    log_sigma_beta::Float64 = sum(view(unconstrained, 2 * n_rats + 5:2 * n_rats + 5))

    # sigma = exp(log_sigma); log|dsigma/dlog_sigma| = log_sigma for each of the
    # three FLAT-prior scales (Stan `real<lower=0>` with no `~` statement — only
    # the transform Jacobian contributes, NO density term).
    sigma_y::Float64 = exp(log_sigma_y)
    sigma_alpha::Float64 = exp(log_sigma_alpha)
    sigma_beta::Float64 = exp(log_sigma_beta)
    log_jacobian::Float64 = log_sigma_y + log_sigma_alpha + log_sigma_beta

    parameters =
        (; alpha, beta, mu_alpha, mu_beta, sigma_y, sigma_alpha, sigma_beta)
    (parameters, log_jacobian::Float64) =
        ((; alpha, beta, mu_alpha, mu_beta, sigma_y, sigma_alpha, sigma_beta),
         log_sigma_y + log_sigma_alpha + log_sigma_beta)
    (alpha::AbstractVector{Float64}, beta::AbstractVector{Float64},
     mu_alpha::Float64, mu_beta::Float64, sigma_y::Float64,
     sigma_alpha::Float64, sigma_beta::Float64) =
        (parameters.alpha, parameters.beta, parameters.mu_alpha,
         parameters.mu_beta, parameters.sigma_y, parameters.sigma_alpha,
         parameters.sigma_beta)

    # Population priors: mu_alpha ~ Normal(0, 100), mu_beta ~ Normal(0, 100). The
    # three scales are FLAT improper (only the exp Jacobian above, no density).
    mu_alpha_prior::Float64 = normal(0.0, 100.0).logpdf(mu_alpha)
    mu_beta_prior::Float64 = normal(0.0, 100.0).logpdf(mu_beta)
    fixed_prior::Float64 = mu_alpha_prior + mu_beta_prior

    # Hierarchical priors (explicit density sums): alpha_j ~ Normal(mu_alpha,
    # sigma_alpha), beta_j ~ Normal(mu_beta, sigma_beta). The shared hyper-scalars
    # ride each plate as broadcast args.
    alpha_pointwise = plate(alpha, mu_alpha, sigma_alpha) do a, m, s
        normal(m, s).logpdf(a)
    end
    alpha_prior::Float64 = sum(alpha_pointwise)
    beta_pointwise = plate(beta, mu_beta, sigma_beta) do b, m, s
        normal(m, s).logpdf(b)
    end
    beta_prior::Float64 = sum(beta_pointwise)
    prior::Float64 = fixed_prior + alpha_prior + beta_prior

    # Transformed data: the age centered on its mean, x - xbar (a deterministic
    # function of data, so no Jacobian). `xbar` is a scalar data port; this
    # broadcast is done OUTSIDE any plate.
    x_centered = x .- xbar

    # Hierarchical integer-array GATHERS, done OUTSIDE any plate: the per-rat
    # intercept and slope gathered by the concrete data index `rat`. The
    # docs_example binds `rat` at preparation so the Reactant path traces only
    # floats + the parameter vector.
    alpha_rat = alpha[rat]
    beta_rat = beta[rat]

    # Transformed parameter / fitted mean: mu = alpha[rat] + beta[rat]*(x - xbar).
    # Named node sharing the gathered intercept/slope and centered age with the
    # likelihood (structural CSE), so only param scalars ride the plate.
    mu = alpha_rat .+ beta_rat .* x_centered

    # Likelihood: y[n] ~ Normal(alpha[rat_n] + beta[rat_n]*(x_n - xbar), sigma_y).
    # The gathered intercept/slope and centered age enter as batched vectors and
    # only sigma_y rides as a shared scalar arg (matching the radon reference).
    pointwise = plate(y, x_centered, alpha_rat, beta_rat, sigma_y) do yy, xc, a, b, s
        normal(a + b * xc, s).logpdf(yy)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: alpha0 = mu_alpha - xbar * mu_beta (the population
    # intercept back at age 0).
    alpha0::Float64 = mu_alpha - xbar * mu_beta

    return posterior
end

q = vcat(240.0 .+ collect(1.0:8.0), 6.0 .+ 0.1 .* collect(1.0:8.0),
         [240.0, 6.0, log(6.0), log(10.0), log(0.5)])
rat = RATS_RAT
x = RATS_X
y = RATS_Y
xbar = RATS_XBAR

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
# Bind the integer gather index (spec addendum): the Reactant path then traces
# only the float inputs + the parameter vector.
density_kernel = prepare(model;
    have = (:unconstrained, :rat, :x, :y, :xbar),
    want = requested_nodes,
    bound = (; rat))

output = density_kernel(q, x, y, xbar)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :rats_model_posterior,
    origin = "posteriordb rats_model — hierarchical linear growth curve (per-rat intercept + slope)",
    inputs = (; q, x, y, xbar),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_rats_model_source()
    _evaluate_ppl_source(RATS_MODEL_SOURCE, @__MODULE__; bindings = (
        :RATS_RAT, :RATS_X, :RATS_Y, :RATS_XBAR,
    ))
end

const _RATS_MODEL_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _RATS_MODEL_GRAPH_TEMPLATE[] = evaluate_rats_model_source().model
    nothing
end

"""
    build_rats_model_graph()

Build the posteriordb `rats_model` (a BUGS hierarchical linear growth curve with
a per-rat intercept alpha_j and slope beta_j) as a declarative
`ReactiveKernels.KernelSpec`. The population priors `mu_alpha`/`mu_beta ~
Normal(0, 100)` and the hierarchical `alpha_j ~ Normal(mu_alpha, sigma_alpha)`,
`beta_j ~ Normal(mu_beta, sigma_beta)` reuse the shared Normal endpoint; the
three scales `sigma_y`/`sigma_alpha`/`sigma_beta` have FLAT improper priors so
only their exact `exp` Jacobians contribute; and the per-rat intercept/slope are
gathered by the concrete `rat` index outside any plate, with the mean
`alpha[rat] + beta[rat]*(x - xbar)` recomputed inline in the likelihood. The
transform Jacobian, prior, gathered/combined mean `mu`, pointwise/summed
likelihood, densities, posterior, and the generated quantity
`alpha0 = mu_alpha - xbar*mu_beta` are named nodes.
"""
function build_rats_model_graph()
    compose(_RATS_MODEL_GRAPH_TEMPLATE[])
end

function demo()
    model = build_rats_model_graph()
    q = vcat(240.0 .+ collect(1.0:8.0), 6.0 .+ 0.1 .* collect(1.0:8.0),
             [240.0, 6.0, log(6.0), log(10.0), log(0.5)])
    posterior_plan = plan(model;
                          have = (:unconstrained, :rat, :x, :y, :xbar),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, RATS_RAT, RATS_X, RATS_Y, RATS_XBAR)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module RatsModelExample

if abspath(PROGRAM_FILE) == @__FILE__
    RatsModelExample.demo()
end
