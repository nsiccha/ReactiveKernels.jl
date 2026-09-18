module ProphetExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export PROPHET_T, PROPHET_CAP, PROPHET_Y, PROPHET_T_CHANGE, PROPHET_X,
    PROPHET_SIGMAS, PROPHET_TAU, PROPHET_TREND_INDICATOR, PROPHET_S_A, PROPHET_S_M
export build_prophet_graph, demo
export PROPHET_SOURCE, PROPHET_LOGISTIC_TREND_REFERENCE_SOURCE, evaluate_prophet_source

# posteriordb `rstan_downloads-prophet` — Facebook Prophet's forecasting model: a
# piecewise-trend (`k` base rate plus per-changepoint adjustments `delta`) with
# additive/multiplicative seasonality regressors (`X` with additive `s_a` /
# multiplicative `s_m` indicators). The changepoint incidence matrix `A[i,j] =
# 1{t_i ≥ t_change_j}` is a data-only structure computed in-graph. Stan's model
# supports a LINEAR (`trend_indicator == 0`) or LOGISTIC (`== 1`) trend; the
# authoritative `rstan_downloads` data selects the LINEAR trend.
#
# CAPABILITY (linear trend only). This graph implements the linear-trend density,
# the mode the authoritative rstan_downloads data selects (trend_indicator == 0).
# `build_prophet_graph` validates the trend flag and THROWS an explicit
# unsupported-mode error for the logistic trend rather than returning a wrong or
# sentinel density (Stan assigns the logistic trend a finite density, so it is a
# real mode). The logistic trend is UNIMPLEMENTED AND UNVERIFIED HERE — a
# separately-scoped deliverable, out of scope for this batch. Its natural
# `logistic_gamma` recurrence threads a first-order recurrence over two co-varying
# per-step sequences (bound `t_change[i]` and the parameter-derived rate ratio);
# that recurrence now authors directly with RK's lockstep multi-sequence
# `scan(t_change, r; init = …)` — the scan gap that previously blocked it (snag
# `scan-single-iter-896551a2`) was fixed and landed in canonical `main` (see
# `docs/src/scan.md`). Implementing and verifying the logistic mode (e.g. against
# the same `.stan` instantiated with constructed logistic-mode data) is a separate
# reviewed item, not done here. Real full data (T = 1169, K = 34, S = 25) from
# posteriordb.
let d = _posteriordb_data("rstan_downloads-prophet")
    _mat(x) = x isa AbstractMatrix ? Float64.(x) :
              reduce(vcat, [permutedims(Float64.(r)) for r in x])
    global const PROPHET_T = Float64.(d["t"])
    global const PROPHET_CAP = Float64.(d["cap"])
    global const PROPHET_Y = Float64.(d["y"])
    global const PROPHET_T_CHANGE = Float64.(d["t_change"])
    global const PROPHET_X = _mat(d["X"])
    global const PROPHET_SIGMAS = Float64.(d["sigmas"])
    global const PROPHET_TAU = Float64(d["tau"])
    global const PROPHET_TREND_INDICATOR = Int(d["trend_indicator"])
    global const PROPHET_S_A = Float64.(d["s_a"])
    global const PROPHET_S_M = Float64.(d["s_m"])
end

const PROPHET_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, laplace

@kernel model(unconstrained::Vector{Float64},
              t::Vector{Float64},
              t_change::Vector{Float64},
              X::Matrix{Float64},
              sigmas::Vector{Float64},
              tau::Float64,
              s_a::Vector{Float64},
              s_m::Vector{Float64},
              y::Vector{Float64}) = begin
    S::Int = length(t_change)
    K::Int = size(X, 2)

    # Unconstrained layout, in Stan declaration order:
    #   [k, m, delta(1:S), log sigma_obs, beta(1:K)].
    k::Float64 = unconstrained[1]
    m::Float64 = unconstrained[2]
    delta::AbstractVector{Float64} = view(unconstrained, 3:(2 + S))
    log_sigma_obs::Float64 = unconstrained[3 + S]
    beta::AbstractVector{Float64} = view(unconstrained, (4 + S):(3 + S + K))

    # Only sigma_obs carries a `<lower=0>` support.
    sigma_obs::Float64 = exp(log_sigma_obs)
    log_jacobian::Float64 = log_sigma_obs

    parameters = (; k, m, delta, sigma_obs, beta)

    # Changepoint incidence matrix A[i, j] = 1{t_i ≥ t_change_j}: a data-only
    # structure (t and t_change are bound), an in-graph comparison-mask matvec.
    A::Matrix{Float64} = (t .>= t_change') .* 1.0

    # Linear trend: (k + A·delta)·t + (m + A·(−t_change·delta)).
    Ad::Vector{Float64} = A * delta
    td::Vector{Float64} = t_change .* delta
    trend::Vector{Float64} = (k .+ Ad) .* t .+ (m .- A * td)

    # Seasonality: multiplicative (1 + X·(beta·s_m)) and additive X·(beta·s_a).
    seasonal_mult::Vector{Float64} = 1.0 .+ X * (beta .* s_m)
    seasonal_add::Vector{Float64} = X * (beta .* s_a)
    mean_response::Vector{Float64} = trend .* seasonal_mult .+ seasonal_add

    # Priors.
    k_prior::Float64 = normal(0.0, 5.0).logpdf(k)
    m_prior::Float64 = normal(0.0, 5.0).logpdf(m)
    delta_pointwise = plate(delta, tau) do d, b
        laplace(0.0, b).logpdf(d)
    end
    delta_prior::Float64 = sum(delta_pointwise)
    sigma_obs_prior::Float64 = normal(0.0, 0.5).logpdf(sigma_obs)
    beta_pointwise = plate(beta, sigmas) do b, s
        normal(0.0, s).logpdf(b)
    end
    beta_prior::Float64 = sum(beta_pointwise)
    prior::Float64 = k_prior + m_prior + delta_prior + sigma_obs_prior + beta_prior

    # Likelihood: yᵢ ~ Normal(mean_responseᵢ, sigma_obs) (linear trend).
    obs_pointwise = plate(y, mean_response, sigma_obs) do yi, mi, s
        normal(mi, s).logpdf(yi)
    end
    likelihood::Float64 = sum(obs_pointwise)

    posterior::Float64 = prior + likelihood + log_jacobian
    return posterior
end

q = zeros(62)
t = PROPHET_T
t_change = PROPHET_T_CHANGE
X = PROPHET_X
sigmas = PROPHET_SIGMAS
tau = PROPHET_TAU
s_a = PROPHET_S_A
s_m = PROPHET_S_M
y = PROPHET_Y

requested_nodes = (:parameters, :prior, :likelihood, :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :t, :t_change, :X, :sigmas, :tau, :s_a, :s_m, :y),
    want = requested_nodes,
    bound = (; t, t_change, X, sigmas, tau, s_a, s_m, y))

output = density_kernel(q)
parameters, prior, likelihood, log_jacobian, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :prophet_posterior,
    origin = "posteriordb prophet — Facebook Prophet piecewise-trend forecasting (linear trend)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    laplace_object = laplace,
)
"""

# STAN REFERENCE PSEUDOCODE of the LOGISTIC trend (the unimplemented mode). This
# is NOT a runnable RK reproducer and is NEITHER evaluated NOR lowered — it
# records Stan's `logistic_gamma`/`logistic_trend` so the intended full model and
# the exact recurrence shape `build_prophet_graph` refuses are documented in one
# place. The `logistic_gamma` recurrence threads `m_pr`; each step reads BOTH the
# bound `t_change[i]` and the parameter-derived rate ratio `1 - k_s[i]/k_s[i+1]`.
# A NATURAL RK authoring of that sequential recurrence now exists via the landed
# lockstep multi-sequence `scan(t_change, r; init = m)` (snag
# `scan-single-iter-896551a2` resolved; see `docs/src/scan.md`) — but the logistic
# mode is not implemented or verified here (see the module CAPABILITY note).
const PROPHET_LOGISTIC_TREND_REFERENCE_SOURCE = raw"""
# === STAN reference pseudocode (NOT runnable RK; documentation only) ===
# k_s = append_row(k, k + cumulative_sum(delta))     # length S+1 segment rates
# --- logistic_gamma: piecewise-continuity offsets (SEQUENTIAL over changepoints) ---
# m_pr = m
# for i in 1:S
#     gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i+1])
#     m_pr = m_pr + gamma[i]                          # accumulates: genuine recurrence
# end
# --- logistic_trend ---
# logistic_trend = cap .* inv_logit((k + A*delta) .* (t - (m + A*gamma)))
# mean_response = logistic_trend .* (1 .+ X*(beta.*s_m)) .+ X*(beta.*s_a)
#
# Natural RK authoring of `logistic_gamma` (now available via lockstep scan):
#   gamma = scan(t_change, r; init = m) do carry, tc, rr   # r = 1 .- k_s[1:S] ./ k_s[2:S+1]
#       next = carry + (tc - carry) * rr
#       (next, next - carry)                               # (new m_pr, gamma[i])
#   end
"""

function evaluate_prophet_source(; model_only::Bool = false)
    _evaluate_ppl_source(PROPHET_SOURCE, @__MODULE__; bindings = (
        :PROPHET_T, :PROPHET_T_CHANGE, :PROPHET_X, :PROPHET_SIGMAS,
        :PROPHET_TAU, :PROPHET_S_A, :PROPHET_S_M, :PROPHET_Y,
    ), model_only)
end

const _PROPHET_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _PROPHET_GRAPH_TEMPLATE[] = evaluate_prophet_source(; model_only = true).model
    nothing
end

"""
    build_prophet_graph(; trend_indicator = PROPHET_TREND_INDICATOR)

Build the posteriordb `prophet` model (Facebook Prophet's piecewise-trend
forecasting model) as a declarative `ReactiveKernels.KernelSpec`, for the LINEAR
trend that the authoritative `rstan_downloads` data selects (`trend_indicator ==
0`). The changepoint incidence matrix `A` is an in-graph data-only comparison-mask
matvec; the linear trend and additive/multiplicative seasonality are in-graph
data→parameter transformations. Only `sigma_obs` carries a log/exp support
transform (the log-Jacobian). The Normal and Laplace (double-exponential)
endpoints are reused from `ReactiveKernelsDistributionKernels`.

This is a LINEAR-TREND-ONLY capability. The `trend_indicator` flag is validated
against the supported mode; the logistic trend (`trend_indicator == 1`) — a real
mode Stan assigns a finite density — is UNIMPLEMENTED and UNVERIFIED here (a
separately-scoped item), so a request for it `throw`s an explicit `ArgumentError`
rather than returning a wrong or sentinel density. Its natural recurrence is
documented as Stan pseudocode in `PROPHET_LOGISTIC_TREND_REFERENCE_SOURCE`; it now
authors via RK's landed lockstep multi-sequence `scan` (snag
`scan-single-iter-896551a2` resolved; see `docs/src/scan.md`).

The prior, changepoint matrix, trend, seasonality, pointwise log-likelihood,
likelihood reduction, transform Jacobian, and total density are separate named
nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_prophet_graph(; trend_indicator::Integer = PROPHET_TREND_INDICATOR)
    trend_indicator == 0 || throw(ArgumentError(
        "prophet: trend_indicator=$trend_indicator is not supported — this deliverable " *
        "implements only the LINEAR trend (trend_indicator=0). The logistic trend is a " *
        "real (finite-density) Stan mode but is UNIMPLEMENTED and UNVERIFIED here (a " *
        "separately-scoped item). Its natural `logistic_gamma` recurrence is documented in " *
        "PROPHET_LOGISTIC_TREND_REFERENCE_SOURCE and now authors via RK's lockstep " *
        "multi-sequence scan (snag scan-single-iter-896551a2 resolved; see docs/src/scan.md)."))
    compose(_PROPHET_GRAPH_TEMPLATE[])
end

function demo()
    model = build_prophet_graph()
    q = zeros(62)

    density_plan = plan(model;
        have = (:unconstrained, :t, :t_change, :X, :sigmas, :tau, :s_a, :s_m, :y),
        want = (:prior, :likelihood, :log_jacobian, :posterior))
    println(explain(density_plan))
    prior, likelihood, log_jacobian, posterior = prepare(density_plan)(
        q, PROPHET_T, PROPHET_T_CHANGE, PROPHET_X, PROPHET_SIGMAS,
        PROPHET_TAU, PROPHET_S_A, PROPHET_S_M, PROPHET_Y)
    println("log prior + log likelihood + log Jacobian")
    println("= ", prior, " + ", likelihood, " + ", log_jacobian)
    println("= log posterior = ", posterior)

    nothing
end

end # module ProphetExample

if abspath(PROGRAM_FILE) == @__FILE__
    ProphetExample.demo()
end
