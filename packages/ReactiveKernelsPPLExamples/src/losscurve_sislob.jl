module LosscurveSislobExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export LOSSCURVE_GROWTHMODEL_ID, LOSSCURVE_COHORT_ID, LOSSCURVE_T_IDX,
    LOSSCURVE_T_VALUE, LOSSCURVE_PREMIUM, LOSSCURVE_LOSS
export build_losscurve_sislob_graph, demo
export LOSSCURVE_SISLOB_SOURCE, evaluate_losscurve_sislob_source

# posteriordb `loss_curves-losscurve_sislob` — a hierarchical insurance
# loss-development ("chain-ladder") curve. Each cohort's ultimate loss ratio LRᵢ
# is drawn hierarchically (lognormal around a shared μ_LR / sd_LR), and the
# expected loss at development time t is `LRᵢ · premiumᵢ · gf(t)`, where the
# growth factor gf(t) ∈ (0, 1) is a Weibull or log-logistic CDF-shaped curve
# selected by the `growthmodel_id` data flag. The observation-noise scale is
# cohort-specific (`loss_sd · premiumᵢ`). The growth-factor transform is a
# genuine in-graph data→parameter transformation (a named node), so bound
# partial evaluation can hoist the data-only parts. Real full data (n_data = 55,
# n_cohort = 10, n_time = 10) from posteriordb, loaded via PosteriorDB.jl.
let d = _posteriordb_data("loss_curves-losscurve_sislob")
    global const LOSSCURVE_GROWTHMODEL_ID = Int(d["growthmodel_id"])
    global const LOSSCURVE_COHORT_ID = Int.(d["cohort_id"])
    global const LOSSCURVE_T_IDX = Int.(d["t_idx"])
    global const LOSSCURVE_T_VALUE = Float64.(d["t_value"])
    global const LOSSCURVE_PREMIUM = Float64.(d["premium"])
    global const LOSSCURVE_LOSS = Float64.(d["loss"])
end

const LOSSCURVE_SISLOB_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, lognormal
using LogExpFunctions: log1pexp

@kernel model(unconstrained::Vector{Float64},
              growthmodel_id::Int,
              cohort_id::Vector{Int},
              t_idx::Vector{Int},
              t_value::Vector{Float64},
              premium::Vector{Float64},
              loss::Vector{Float64}) = begin
    n_cohort::Int = length(premium)

    # Unconstrained layout, in Stan declaration order:
    #   [log ω, log θ, log LR(1:n_cohort), μ_LR, log sd_LR, log loss_sd].
    # Every parameter but μ_LR carries a `<lower=0>` support, so its authoritative
    # unconstrained value is the log of the constrained one.
    log_omega::Float64 = unconstrained[1]
    log_theta::Float64 = unconstrained[2]
    log_LR::AbstractVector{Float64} = view(unconstrained, 3:(2 + n_cohort))
    mu_LR::Float64 = unconstrained[3 + n_cohort]
    log_sd_LR::Float64 = unconstrained[4 + n_cohort]
    log_loss_sd::Float64 = unconstrained[5 + n_cohort]

    omega::Float64 = exp(log_omega)
    theta::Float64 = exp(log_theta)
    LR::Vector{Float64} = exp.(log_LR)
    sd_LR::Float64 = exp(log_sd_LR)
    loss_sd::Float64 = exp(log_loss_sd)

    # The support transforms are the elementwise log/exp maps; the total log
    # Jacobian is the sum of the unconstrained values that carry a `<lower=0>`.
    log_jacobian::Float64 =
        log_omega + log_theta + sum(log_LR) + log_sd_LR + log_loss_sd

    parameters = (; omega, theta, LR, mu_LR, sd_LR, loss_sd)

    # In-graph growth-factor transform, per development time. The `growthmodel_id`
    # data flag stays a live bound port and selects Weibull (id == 1) or
    # log-logistic per cell; both closed forms are authored, so no branch is
    # silently specialized away. BOTH forms are written so the eagerly evaluated
    # but UNSELECTED branch cannot poison the reverse gradient of the selected
    # branch at extreme omega (measured at omega=1000, theta=max(t_value)):
    # Weibull's `(t/theta)^omega` stays <= 1 through the data there, and
    # log-logistic is the algebraically-equal `1/(1 + (theta/t)^omega)` written
    # as `exp(-log1pexp(omega*log(theta/t)))` — the naive ratio form overflows
    # `(theta/t)^omega` to Inf, whose forward reciprocal stays finite but whose
    # REVERSE chain rule produces 0*Inf = NaN adjoints.
    gf_weibull::Vector{Float64} = 1.0 .- exp.(-(t_value ./ theta) .^ omega)
    gf_loglogistic::Vector{Float64} =
        exp.(-log1pexp.(omega .* log.(theta ./ t_value)))
    gf::Vector{Float64} = ifelse.(growthmodel_id == 1, gf_weibull, gf_loglogistic)

    # Per-datum expected loss lmᵢ = LR[cohortᵢ] · premium[cohortᵢ] · gf[timeᵢ].
    # The cohort/time indices are integer-array gathers.
    premium_by_datum::Vector{Float64} = premium[cohort_id]
    lm::Vector{Float64} = LR[cohort_id] .* premium_by_datum .* gf[t_idx]

    # Priors. LR is drawn hierarchically around (μ_LR, sd_LR); the remaining
    # positive scales carry lognormal priors and μ_LR a normal prior.
    mu_LR_prior::Float64 = normal(0.0, 0.5).logpdf(mu_LR)
    sd_LR_prior::Float64 = lognormal(0.0, 0.5).logpdf(sd_LR)
    LR_pointwise = plate(LR, mu_LR, sd_LR) do lr, m, s
        lognormal(m, s).logpdf(lr)
    end
    LR_prior::Float64 = sum(LR_pointwise)
    loss_sd_prior::Float64 = lognormal(0.0, 0.7).logpdf(loss_sd)
    omega_prior::Float64 = lognormal(0.0, 0.5).logpdf(omega)
    theta_prior::Float64 = lognormal(0.0, 0.5).logpdf(theta)
    prior::Float64 = mu_LR_prior + sd_LR_prior + LR_prior +
                     loss_sd_prior + omega_prior + theta_prior

    # Likelihood: lossᵢ ~ Normal(lmᵢ, loss_sd · premium[cohortᵢ]).
    obs_scale::Vector{Float64} = loss_sd .* premium_by_datum
    pointwise = plate(loss, lm, obs_scale) do y, m, s
        normal(m, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    posterior::Float64 = prior + likelihood + log_jacobian
    return posterior
end

q = zeros(15)
growthmodel_id = LOSSCURVE_GROWTHMODEL_ID
cohort_id = LOSSCURVE_COHORT_ID
t_idx = LOSSCURVE_T_IDX
t_value = LOSSCURVE_T_VALUE
premium = LOSSCURVE_PREMIUM
loss = LOSSCURVE_LOSS

requested_nodes = (:parameters, :prior, :likelihood, :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :growthmodel_id, :cohort_id, :t_idx,
            :t_value, :premium, :loss),
    want = requested_nodes,
    bound = (; growthmodel_id, cohort_id, t_idx, t_value, premium, loss))

output = density_kernel(q)
parameters, prior, likelihood, log_jacobian, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :losscurve_sislob_posterior,
    origin = "posteriordb losscurve_sislob — hierarchical insurance loss-development curve",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    lognormal_object = lognormal,
)
"""

function evaluate_losscurve_sislob_source(; model_only::Bool = false)
    _evaluate_ppl_source(LOSSCURVE_SISLOB_SOURCE, @__MODULE__; bindings = (
        :LOSSCURVE_GROWTHMODEL_ID, :LOSSCURVE_COHORT_ID, :LOSSCURVE_T_IDX,
        :LOSSCURVE_T_VALUE, :LOSSCURVE_PREMIUM, :LOSSCURVE_LOSS,
    ), model_only)
end

const _LOSSCURVE_SISLOB_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOSSCURVE_SISLOB_GRAPH_TEMPLATE[] =
        evaluate_losscurve_sislob_source(; model_only = true).model
    nothing
end

"""
    build_losscurve_sislob_graph()

Build the posteriordb `losscurve_sislob` model (a hierarchical insurance
loss-development curve) as a declarative `ReactiveKernels.KernelSpec`. The
positive scales (`omega`, `theta`, `LR`, `sd_LR`, `loss_sd`) carry log/exp
support transforms with the summed log-Jacobian; the growth factor `gf` is an
in-graph data→parameter transformation whose `growthmodel_id` flag remains a
live bound port selecting Weibull or log-logistic per cell. The Normal and
LogNormal endpoints are reused from `ReactiveKernelsDistributionKernels`; the
prior, growth factor, per-datum mean, pointwise log-likelihood, likelihood
reduction, transform Jacobian, and total density are separate named nodes, and
the constrained parameters are a plain NamedTuple.
"""
function build_losscurve_sislob_graph()
    compose(_LOSSCURVE_SISLOB_GRAPH_TEMPLATE[])
end

function demo()
    model = build_losscurve_sislob_graph()
    q = zeros(15)

    density_plan = plan(model;
        have = (:unconstrained, :growthmodel_id, :cohort_id, :t_idx,
                :t_value, :premium, :loss),
        want = (:prior, :likelihood, :log_jacobian, :posterior))
    println(explain(density_plan))
    prior, likelihood, log_jacobian, posterior = prepare(density_plan)(
        q, LOSSCURVE_GROWTHMODEL_ID, LOSSCURVE_COHORT_ID, LOSSCURVE_T_IDX,
        LOSSCURVE_T_VALUE, LOSSCURVE_PREMIUM, LOSSCURVE_LOSS)
    println("log prior + log likelihood + log Jacobian")
    println("= ", prior, " + ", likelihood, " + ", log_jacobian)
    println("= log posterior = ", posterior)

    nothing
end

end # module LosscurveSislobExample

if abspath(PROGRAM_FILE) == @__FILE__
    LosscurveSislobExample.demo()
end
