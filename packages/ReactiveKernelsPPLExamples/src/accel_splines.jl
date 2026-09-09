module AccelSplinesExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export ACCEL_Y, ACCEL_XS, ACCEL_ZS_1_1, ACCEL_XS_SIGMA, ACCEL_ZS_SIGMA_1_1,
    ACCEL_PRIOR_ONLY
export build_accel_splines_graph, demo
export ACCEL_SPLINES_SOURCE, evaluate_accel_splines_source

# posteriordb `mcycle_splines-accel_splines` — a brms-generated penalized-spline
# regression of the motorcycle-acceleration data, with a spline for BOTH the mean
# and the (log-linked) residual scale. Each spline enters as a fixed basis-function
# design matrix (`Zs_*`) times standardized coefficients `zs_*` scaled by a
# half-Student-t `sds_*`; the linear-effect design matrices `Xs_*` carry flat
# `bs_*`. The design-matrix products are genuine in-graph data→parameter
# transformations. Real full data (N = 133, 38 knots for each spline) from
# posteriordb, loaded via PosteriorDB.jl.
let d = _posteriordb_data("mcycle_splines-accel_splines")
    _mat(x) = x isa AbstractMatrix ? Float64.(x) :
              reduce(vcat, [permutedims(Float64.(r)) for r in x])
    global const ACCEL_Y = Float64.(d["Y"])
    global const ACCEL_XS = _mat(d["Xs"])
    global const ACCEL_ZS_1_1 = _mat(d["Zs_1_1"])
    global const ACCEL_XS_SIGMA = _mat(d["Xs_sigma"])
    global const ACCEL_ZS_SIGMA_1_1 = _mat(d["Zs_sigma_1_1"])
    global const ACCEL_PRIOR_ONLY = Int(d["prior_only"])
end

const ACCEL_SPLINES_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, student_t

@kernel model(unconstrained::Vector{Float64},
              Y::Vector{Float64},
              Xs::Matrix{Float64},
              Zs_1_1::Matrix{Float64},
              Xs_sigma::Matrix{Float64},
              Zs_sigma_1_1::Matrix{Float64},
              prior_only::Int) = begin
    # Basis widths come from the design matrices, so the unconstrained layout is
    # data-generic. Stan declaration order:
    #   [Intercept, bs, zs_1_1, log sds_1_1,
    #    Intercept_sigma, bs_sigma, zs_sigma_1_1, log sds_sigma_1_1].
    Ks::Int = size(Xs, 2)
    knots_1::Int = size(Zs_1_1, 2)
    Ks_sigma::Int = size(Xs_sigma, 2)
    knots_sigma_1::Int = size(Zs_sigma_1_1, 2)

    Intercept::Float64 = unconstrained[1]
    bs::AbstractVector{Float64} = view(unconstrained, 2:(1 + Ks))
    zs_1_1::AbstractVector{Float64} =
        view(unconstrained, (2 + Ks):(1 + Ks + knots_1))
    log_sds_1_1::Float64 = unconstrained[2 + Ks + knots_1]
    Intercept_sigma::Float64 = unconstrained[3 + Ks + knots_1]
    bs_sigma::AbstractVector{Float64} =
        view(unconstrained, (4 + Ks + knots_1):(3 + Ks + knots_1 + Ks_sigma))
    zs_sigma_1_1::AbstractVector{Float64} =
        view(unconstrained,
             (4 + Ks + knots_1 + Ks_sigma):(3 + Ks + knots_1 + Ks_sigma + knots_sigma_1))
    log_sds_sigma_1_1::Float64 =
        unconstrained[4 + Ks + knots_1 + Ks_sigma + knots_sigma_1]

    # Only the two spline standard deviations carry a `<lower=0>` support.
    sds_1_1::Float64 = exp(log_sds_1_1)
    sds_sigma_1_1::Float64 = exp(log_sds_sigma_1_1)
    log_jacobian::Float64 = log_sds_1_1 + log_sds_sigma_1_1

    # Actual spline coefficients (Stan transformed parameters).
    s_1_1::Vector{Float64} = sds_1_1 .* zs_1_1
    s_sigma_1_1::Vector{Float64} = sds_sigma_1_1 .* zs_sigma_1_1

    parameters = (; Intercept, bs, s_1_1, sds_1_1,
                    Intercept_sigma, bs_sigma, s_sigma_1_1, sds_sigma_1_1)

    # Linear predictors: an intercept plus the linear-effect and spline designs.
    # The residual scale uses a log link, so `sigma = exp(.)`.
    mu::Vector{Float64} = Intercept .+ Xs * bs .+ Zs_1_1 * s_1_1
    sigma_linpred::Vector{Float64} =
        Intercept_sigma .+ Xs_sigma * bs_sigma .+ Zs_sigma_1_1 * s_sigma_1_1
    sigma::Vector{Float64} = exp.(sigma_linpred)

    # Priors including all constants (propto = false). The half-Student-t priors
    # on the spline sds add the `-student_t_lccdf(0 | 3, 0, 36) = +log(2)`
    # truncation normalizer (symmetric Student-t, so lccdf(0) = log(1/2)); the
    # linear effects `bs`, `bs_sigma` are Stan-flat.
    Intercept_prior::Float64 = student_t(3.0, -13.0, 36.0).logpdf(Intercept)
    zs_1_1_pointwise = plate(zs_1_1) do z
        normal(0.0, 1.0).logpdf(z)
    end
    zs_1_1_prior::Float64 = sum(zs_1_1_pointwise)
    sds_1_1_prior::Float64 = student_t(3.0, 0.0, 36.0).logpdf(sds_1_1) + log(2.0)
    Intercept_sigma_prior::Float64 = student_t(3.0, 0.0, 10.0).logpdf(Intercept_sigma)
    zs_sigma_1_1_pointwise = plate(zs_sigma_1_1) do z
        normal(0.0, 1.0).logpdf(z)
    end
    zs_sigma_1_1_prior::Float64 = sum(zs_sigma_1_1_pointwise)
    sds_sigma_1_1_prior::Float64 =
        student_t(3.0, 0.0, 36.0).logpdf(sds_sigma_1_1) + log(2.0)
    prior::Float64 = Intercept_prior + zs_1_1_prior + sds_1_1_prior +
                     Intercept_sigma_prior + zs_sigma_1_1_prior + sds_sigma_1_1_prior

    # Likelihood: Yᵢ ~ Normal(muᵢ, sigmaᵢ), gated by the `prior_only` data flag.
    # The `ifelse` below evaluates the obs branch EAGERLY, so guard the likelihood
    # scale: when prior_only=1 selects the likelihood away, a saturating linear
    # predictor can drive sigma → 0 (underflow), making `normal(·, 0).logpdf` NaN
    # and poisoning the reverse gradient of the unselected branch. `sigma_ll` is
    # exactly `sigma` when the likelihood is active (prior_only=0), and a harmless
    # constant when it is not — keeping the eagerly-evaluated obs term finite.
    sigma_ll::Vector{Float64} = ifelse.(prior_only == 0, sigma, 1.0)
    obs_pointwise = plate(Y, mu, sigma_ll) do y, m, s
        normal(m, s).logpdf(y)
    end
    obs_ll::Float64 = sum(obs_pointwise)
    likelihood::Float64 = ifelse(prior_only == 0, obs_ll, 0.0)

    posterior::Float64 = prior + likelihood + log_jacobian
    return posterior
end

q = zeros(82)
Y = ACCEL_Y
Xs = ACCEL_XS
Zs_1_1 = ACCEL_ZS_1_1
Xs_sigma = ACCEL_XS_SIGMA
Zs_sigma_1_1 = ACCEL_ZS_SIGMA_1_1
prior_only = ACCEL_PRIOR_ONLY

requested_nodes = (:parameters, :prior, :likelihood, :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :Y, :Xs, :Zs_1_1, :Xs_sigma, :Zs_sigma_1_1, :prior_only),
    want = requested_nodes,
    bound = (; Y, Xs, Zs_1_1, Xs_sigma, Zs_sigma_1_1, prior_only))

output = density_kernel(q)
parameters, prior, likelihood, log_jacobian, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :accel_splines_posterior,
    origin = "posteriordb accel_splines — brms penalized-spline regression (mean and log-scale)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    student_t_object = student_t,
)
"""

function evaluate_accel_splines_source(; model_only::Bool = false)
    _evaluate_ppl_source(ACCEL_SPLINES_SOURCE, @__MODULE__; bindings = (
        :ACCEL_Y, :ACCEL_XS, :ACCEL_ZS_1_1, :ACCEL_XS_SIGMA,
        :ACCEL_ZS_SIGMA_1_1, :ACCEL_PRIOR_ONLY,
    ), model_only)
end

const _ACCEL_SPLINES_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _ACCEL_SPLINES_GRAPH_TEMPLATE[] = evaluate_accel_splines_source(; model_only = true).model
    nothing
end

"""
    build_accel_splines_graph()

Build the posteriordb `accel_splines` model (a brms penalized-spline regression
with a spline for both the mean and the log-linked residual scale) as a
declarative `ReactiveKernels.KernelSpec`. The two spline standard deviations
carry log/exp support transforms with the summed log-Jacobian; the mean and
log-scale linear predictors are in-graph data→parameter transformations
(intercept + linear-effect design + basis-function design times scaled
coefficients). The Normal and Student-t endpoints are reused from
`ReactiveKernelsDistributionKernels`; the half-Student-t priors on the spline
sds add the `+log(2)` truncation normalizer. The prior, transformed spline
coefficients, linear predictors, pointwise log-likelihood, likelihood reduction
(gated by the live `prior_only` flag), transform Jacobian, and total density are
separate named nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_accel_splines_graph()
    compose(_ACCEL_SPLINES_GRAPH_TEMPLATE[])
end

function demo()
    model = build_accel_splines_graph()
    q = zeros(82)

    density_plan = plan(model;
        have = (:unconstrained, :Y, :Xs, :Zs_1_1, :Xs_sigma, :Zs_sigma_1_1, :prior_only),
        want = (:prior, :likelihood, :log_jacobian, :posterior))
    println(explain(density_plan))
    prior, likelihood, log_jacobian, posterior = prepare(density_plan)(
        q, ACCEL_Y, ACCEL_XS, ACCEL_ZS_1_1, ACCEL_XS_SIGMA, ACCEL_ZS_SIGMA_1_1,
        ACCEL_PRIOR_ONLY)
    println("log prior + log likelihood + log Jacobian")
    println("= ", prior, " + ", likelihood, " + ", log_jacobian)
    println("= log posterior = ", posterior)

    nothing
end

end # module AccelSplinesExample

if abspath(PROGRAM_FILE) == @__FILE__
    AccelSplinesExample.demo()
end
