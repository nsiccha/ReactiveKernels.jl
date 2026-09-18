module Bym2OffsetOnlyExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export BYM2_NODE1, BYM2_NODE2, BYM2_Y, BYM2_E, BYM2_SCALING_FACTOR
export build_bym2_offset_only_graph, demo
export BYM2_SOURCE, evaluate_bym2_offset_only_source

# posteriordb `traffic_accident_nyc-bym2_offset_only` — the Morris/Riebler BYM2
# spatial-Poisson model (Besag-York-Mollié 2), applied to an NYC traffic-accident
# case study (N = 1921 census areas, N_edges = 5461 adjacencies). Each area's
# log-rate is an offset `log_E` plus intercept `beta0` plus a convolved spatial +
# heterogeneous random effect
#   convolved_re = sqrt(1 - rho)*theta + sqrt(rho/scaling_factor)*phi,
# scaled by `sigma`; `phi` carries an ICAR prior `-0.5*Σ (phi[node1] - phi[node2])²`
# and a soft sum-to-zero constraint `sum(phi) ~ Normal(0, 0.001*N)`. `rho ∈ [0,1]`
# is the spatial/heterogeneous mixing proportion, `sigma > 0` the overall scale.
# `log_E = log(E)` is Stan transformed data, authored as an in-graph node over the
# bound exposure so bound partial-evaluation folds it. Real data (full) from
# posteriordb `traffic_accident_nyc-bym2_offset_only`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("traffic_accident_nyc-bym2_offset_only")
    global const BYM2_NODE1 = Int.(d["node1"])
    global const BYM2_NODE2 = Int.(d["node2"])
    global const BYM2_Y = Int.(d["y"])
    global const BYM2_E = Float64.(d["E"])
    global const BYM2_SCALING_FACTOR = Float64(d["scaling_factor"])
end

const BYM2_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, beta, poisson
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              node1::Vector{Int},
              node2::Vector{Int},
              y::Vector{Int},
              E::Vector{Float64},
              scaling_factor::Float64) = begin
    # Stan's declared unconstrained order: (beta0, sigma, rho, theta[1..N],
    # phi[1..N]); dim = 2N + 3. beta0/theta/phi are unconstrained; sigma > 0
    # (log transform), rho ∈ [0,1] (logit transform).
    N::Int = (length(unconstrained) - 3) ÷ 2
    beta0::Float64 = unconstrained[1]
    u_sigma::Float64 = unconstrained[2]
    u_rho::Float64 = unconstrained[3]
    theta::AbstractVector{Float64} = view(unconstrained, 4:N + 3)
    phi::AbstractVector{Float64} = view(unconstrained, N + 4:2N + 3)

    sigma::Float64 = exp(u_sigma)
    rho::Float64 = logistic(u_rho)
    jac_sigma::Float64 = u_sigma
    jac_rho::Float64 = -log1pexp(-u_rho) - log1pexp(u_rho)
    log_jacobian::Float64 = jac_sigma + jac_rho

    parameters = (; beta0, sigma, rho, theta, phi)
    (parameters, log_jacobian::Float64) =
        ((; beta0, sigma, rho, theta, phi), u_sigma - log1pexp(-u_rho) - log1pexp(u_rho))

    # Transformed data: log_E = log(E) (bound → folds by partial evaluation).
    log_E = plate(E) do e
        log(e)
    end

    # Transformed parameter: convolved random effect (variance ≈ 1 per component).
    convolved_re = plate(theta, phi, rho, scaling_factor) do t, p, r, sf
        sqrt(1 - r) * t + sqrt(r / sf) * p
    end

    # Priors: beta0 ~ Normal(0,1); sigma ~ Normal(0,1) (half via >0 support);
    # rho ~ Beta(0.5,0.5); thetaⱼ ~ Normal(0,1).
    beta0_prior::Float64 = normal(0.0, 1.0).logpdf(beta0)
    sigma_prior::Float64 = normal(0.0, 1.0).logpdf(sigma)
    rho_prior::Float64 = beta(0.5, 0.5).logpdf(rho)
    theta_pointwise = plate(theta) do t
        normal(0.0, 1.0).logpdf(t)
    end
    theta_prior::Float64 = sum(theta_pointwise)

    # ICAR prior on phi: target += -0.5 * dot_self(phi[node1] - phi[node2]).
    # node1/node2 are bound adjacency indices → gathers of the parameter phi.
    phi1 = phi[node1]
    phi2 = phi[node2]
    icar_pointwise = plate(phi1, phi2) do a, b
        (a - b)^2
    end
    icar_prior::Float64 = -0.5 * sum(icar_pointwise)

    # Soft sum-to-zero: sum(phi) ~ Normal(0, 0.001*N).
    sum_phi::Float64 = sum(phi)
    s2z_scale::Float64 = 0.001 * N
    s2z_prior::Float64 = normal(0.0, s2z_scale).logpdf(sum_phi)

    prior::Float64 =
        beta0_prior + sigma_prior + rho_prior + theta_prior + icar_prior + s2z_prior

    # Likelihood: yᵢ ~ Poisson_log(log_Eᵢ + beta0 + convolved_reᵢ * sigma).
    eta = plate(log_E, convolved_re, beta0, sigma) do le, cr, b, s
        le + b + cr * s
    end
    pointwise = plate(y, eta) do yy, e
        poisson(; log_rate = e).logpdf(yy)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: mu = exp(eta), the fitted mean counts.
    mu = plate(eta) do e
        exp(e)
    end

    return posterior
end

n_areas = length(BYM2_Y)
q = zeros(2 * n_areas + 3)
node1 = BYM2_NODE1
node2 = BYM2_NODE2
y = BYM2_Y
E = BYM2_E
scaling_factor = BYM2_SCALING_FACTOR

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :node1, :node2, :y, :E, :scaling_factor),
    want = requested_nodes,
    bound = (; node1, node2, y, E, scaling_factor))

output = density_kernel(q)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :bym2_offset_only_posterior,
    origin = "posteriordb bym2_offset_only — BYM2 spatial-Poisson (ICAR + heterogeneous convolution)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    beta_object = beta,
    poisson_object = poisson,
)
"""

function evaluate_bym2_offset_only_source(; model_only::Bool = false)
    _evaluate_ppl_source(BYM2_SOURCE, @__MODULE__; bindings = (
        :BYM2_NODE1, :BYM2_NODE2, :BYM2_Y, :BYM2_E, :BYM2_SCALING_FACTOR,
    ), model_only)
end

const _BYM2_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _BYM2_GRAPH_TEMPLATE[] = evaluate_bym2_offset_only_source(; model_only = true).model
    nothing
end

"""
    build_bym2_offset_only_graph()

Build the posteriordb `bym2_offset_only` model (the BYM2 spatial-Poisson
convolution of an ICAR spatial effect `phi` and a heterogeneous effect `theta`,
`convolved_re = sqrt(1-rho)*theta + sqrt(rho/scaling_factor)*phi`) as a
declarative `ReactiveKernels.KernelSpec`. `sigma > 0` (log) and `rho ∈ [0,1]`
(logit) carry their exact transform Jacobians; `beta0/sigma ~ Normal(0,1)`,
`rho ~ Beta(0.5,0.5)`, `theta ~ Normal(0,1)`; `phi` has the ICAR pairwise-
difference prior `-0.5*Σ(phi[node1]-phi[node2])²` (bound-index gathers) plus the
soft sum-to-zero `sum(phi) ~ Normal(0, 0.001*N)`. The offset `log_E = log(E)` is
an in-graph node over the bound exposure; the Poisson-log likelihood reuses the
shared endpoint. The transform Jacobian, `convolved_re`, priors, `eta`,
pointwise/summed likelihood, densities, posterior, and `mu = exp(eta)` are
separate named nodes.
"""
function build_bym2_offset_only_graph()
    compose(_BYM2_GRAPH_TEMPLATE[])
end

function demo()
    model = build_bym2_offset_only_graph()
    q = zeros(2 * length(BYM2_Y) + 3)
    posterior_kernel = prepare(model;
        have = (:unconstrained, :node1, :node2, :y, :E, :scaling_factor),
        want = :posterior,
        bound = (; node1 = BYM2_NODE1, node2 = BYM2_NODE2, y = BYM2_Y,
                   E = BYM2_E, scaling_factor = BYM2_SCALING_FACTOR))
    println("BYM2 unconstrained log posterior = ", posterior_kernel(q))
    nothing
end

end # module Bym2OffsetOnlyExample

if abspath(PROGRAM_FILE) == @__FILE__
    Bym2OffsetOnlyExample.demo()
end
