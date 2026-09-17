module StateSpaceStochasticExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export STATE_SPACE_Y, STATE_SPACE_X, STATE_SPACE_W
export build_state_space_stochastic_graph, demo
export STATE_SPACE_STOCHASTIC_SOURCE, evaluate_state_space_stochastic_source

# posteriordb `uk_drivers-state_space_stochastic_level_stochastic_seasonal` — a
# structural time-series (dynamic linear model) of UK monthly driver deaths:
# a stochastic (random-walk) LEVEL `mu`, a stochastic SEASONAL component whose
# 12-month contributions sum to ≈0, and two regression covariates (`beta·x`,
# `lambda·w`). The genuine sequential structure lives in the transition priors —
# muₜ ~ Normal(muₜ₋₁, σ₂) and the seasonal window sum — evaluated over the free
# state vectors, so they are authored as vectorized in-graph reductions over
# adjacent slices / a trailing-window matvec. The level is a bounded vector with
# data-directed bounds (mean(y) ± 3·sd(y)); the three scales are a
# positive_ordered[3]. Real full data (n = 192 months) from posteriordb.
let d = _posteriordb_data("uk_drivers-state_space_stochastic_level_stochastic_seasonal")
    global const STATE_SPACE_Y = Float64.(d["y"])
    global const STATE_SPACE_X = Float64.(d["x"])
    global const STATE_SPACE_W = Float64.(d["w"])
end

const STATE_SPACE_STOCHASTIC_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, student_t
using LogExpFunctions: log1pexp

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Float64},
              x::Vector{Float64},
              w::Vector{Float64}) = begin
    n::Int = length(y)

    # Data-only bounds for the level: mean(y) ± 3·sd(y) (Stan `sd` uses n − 1).
    ybar::Float64 = sum(y) / n
    ysd::Float64 = sqrt(sum((y .- ybar) .^ 2) / (n - 1))
    lower::Float64 = ybar - 3.0 * ysd
    upper::Float64 = ybar + 3.0 * ysd
    log_span::Float64 = log(upper - lower)

    # Unconstrained layout, in Stan declaration order:
    #   [mu(1:n), seasonal(1:n), beta, lambda, sigma(1:3)].
    mu_unc::AbstractVector{Float64} = view(unconstrained, 1:n)
    seasonal::AbstractVector{Float64} = view(unconstrained, (n + 1):(2 * n))
    beta::Float64 = unconstrained[2 * n + 1]
    lambda::Float64 = unconstrained[2 * n + 2]
    sigma_unc1::Float64 = unconstrained[2 * n + 3]
    sigma_unc2::Float64 = unconstrained[2 * n + 4]
    sigma_unc3::Float64 = unconstrained[2 * n + 5]

    # Level: a bounded (lower, upper) vector, so muᵢ = L + (U − L)·logistic(zᵢ)
    # with the stable logistic σ(z) = exp(−log1pexp(−z)). Its per-element
    # log-Jacobian is log(U − L) + log σ(zᵢ) + log σ(−zᵢ).
    mu::Vector{Float64} = lower .+ (upper - lower) .* exp.(-log1pexp.(-mu_unc))
    mu_jacobian::Float64 =
        n * log_span - sum(log1pexp.(-mu_unc)) - sum(log1pexp.(mu_unc))

    # Scales: a positive_ordered[3] — σ₁ = exp(z₁), σₖ = σₖ₋₁ + exp(zₖ) — so the
    # scales are positive and increasing; its log-Jacobian is Σ zₖ.
    sigma1::Float64 = exp(sigma_unc1)
    sigma2::Float64 = sigma1 + exp(sigma_unc2)
    sigma3::Float64 = sigma2 + exp(sigma_unc3)
    sigma_jacobian::Float64 = sigma_unc1 + sigma_unc2 + sigma_unc3
    sigma::Vector{Float64} = vcat(sigma1, sigma2, sigma3)

    log_jacobian::Float64 = mu_jacobian + sigma_jacobian

    parameters = (; mu, seasonal, beta, lambda, sigma)

    # Transformed parameter: yhat = mu + beta·x + lambda·w.
    yhat::Vector{Float64} = mu .+ beta .* x .+ lambda .* w

    # Stochastic level: muₜ ~ Normal(muₜ₋₁, σ₂) for t = 2..n (a random walk over
    # the free level vector — a vectorized density over adjacent slices).
    level_pointwise = plate(mu[2:n], mu[1:(n - 1)], sigma2) do curr, prev, s
        normal(prev, s).logpdf(curr)
    end
    level_lp::Float64 = sum(level_pointwise)

    # Stochastic seasonal: seasonalₜ ~ Normal(−Σ_{k=t−11}^{t−1} seasonalₖ, σ₁)
    # for t = 12..n. Equivalently the 12-term trailing window
    #   rₜ = seasonalₜ + Σ_{k=t−11}^{t−1} seasonalₖ = Σ_{k=t−11}^{t} seasonalₖ
    # is Normal(0, σ₁). The trailing-window sum is a banded 0/1 matvec built from
    # an index-difference mask (data-only structure over the free seasonal
    # vector), sliced to the valid range t = 12..n.
    idx = 1:n
    window::Matrix{Float64} = ((idx .- idx' .>= 0) .* (idx .- idx' .<= 11)) .* 1.0
    season_resid::Vector{Float64} = window * seasonal
    seasonal_pointwise = plate(season_resid[12:n], sigma1) do r, s
        normal(0.0, s).logpdf(r)
    end
    seasonal_lp::Float64 = sum(seasonal_pointwise)

    # Observation: yᵢ ~ Normal(yhatᵢ + seasonalᵢ, σ₃).
    obs_mean::Vector{Float64} = yhat .+ seasonal
    obs_pointwise = plate(y, obs_mean, sigma3) do yi, m, s
        normal(m, s).logpdf(yi)
    end
    obs_lp::Float64 = sum(obs_pointwise)

    # Scale prior: σ ~ Student-t(4, 0, 1) at each ordered scale.
    sigma_prior::Float64 = student_t(4.0, 0.0, 1.0).logpdf(sigma1) +
                           student_t(4.0, 0.0, 1.0).logpdf(sigma2) +
                           student_t(4.0, 0.0, 1.0).logpdf(sigma3)

    prior::Float64 = level_lp + seasonal_lp + sigma_prior
    posterior::Float64 = prior + obs_lp + log_jacobian
    return posterior
end

q = zeros(389)
y = STATE_SPACE_Y
x = STATE_SPACE_X
w = STATE_SPACE_W

requested_nodes = (:parameters, :level_lp, :seasonal_lp, :obs_lp, :sigma_prior,
                   :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :y, :x, :w),
    want = requested_nodes,
    bound = (; y, x, w))

output = density_kernel(q)
parameters, level_lp, seasonal_lp, obs_lp, sigma_prior, log_jacobian, posterior = output
@assert posterior ≈ level_lp + seasonal_lp + sigma_prior + obs_lp + log_jacobian

docs_example = (;
    name = :state_space_stochastic_posterior,
    origin = "posteriordb state_space_stochastic_level_stochastic_seasonal — UK-drivers structural DLM",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    student_t_object = student_t,
)
"""

function evaluate_state_space_stochastic_source(; model_only::Bool = false)
    _evaluate_ppl_source(STATE_SPACE_STOCHASTIC_SOURCE, @__MODULE__; bindings = (
        :STATE_SPACE_Y, :STATE_SPACE_X, :STATE_SPACE_W,
    ), model_only)
end

const _STATE_SPACE_STOCHASTIC_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _STATE_SPACE_STOCHASTIC_GRAPH_TEMPLATE[] =
        evaluate_state_space_stochastic_source(; model_only = true).model
    nothing
end

"""
    build_state_space_stochastic_graph()

Build the posteriordb `state_space_stochastic_level_stochastic_seasonal` model
(a UK-drivers structural dynamic linear model) as a declarative
`ReactiveKernels.KernelSpec`. The level `mu` is a bounded vector with
data-directed bounds (mean(y) ± 3·sd(y)) whose per-element logit Jacobian is
summed; the three scales are a positive_ordered[3] with the summed log-Jacobian.
The stochastic-level random walk and the stochastic-seasonal trailing-window sum
are the genuine sequential structure, authored as vectorized in-graph reductions
(adjacent-slice plate and a banded 0/1 matvec). The Normal and Student-t
endpoints are reused from `ReactiveKernelsDistributionKernels`. The transforms,
transition log-densities, observation log-density, scale prior, transform
Jacobian, and total density are separate named nodes, and the constrained
parameters are a plain NamedTuple.
"""
function build_state_space_stochastic_graph()
    compose(_STATE_SPACE_STOCHASTIC_GRAPH_TEMPLATE[])
end

function demo()
    model = build_state_space_stochastic_graph()
    q = zeros(389)

    density_plan = plan(model;
        have = (:unconstrained, :y, :x, :w),
        want = (:level_lp, :seasonal_lp, :obs_lp, :sigma_prior, :log_jacobian, :posterior))
    println(explain(density_plan))
    level_lp, seasonal_lp, obs_lp, sigma_prior, log_jacobian, posterior =
        prepare(density_plan)(q, STATE_SPACE_Y, STATE_SPACE_X, STATE_SPACE_W)
    println("level + seasonal + obs + sigma_prior + log Jacobian")
    println("= ", level_lp, " + ", seasonal_lp, " + ", obs_lp, " + ",
            sigma_prior, " + ", log_jacobian)
    println("= log posterior = ", posterior)

    nothing
end

end # module StateSpaceStochasticExample

if abspath(PROGRAM_FILE) == @__FILE__
    StateSpaceStochasticExample.demo()
end
