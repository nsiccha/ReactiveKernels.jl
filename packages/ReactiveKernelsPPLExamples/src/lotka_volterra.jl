module LotkaVolterraExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export LOTKA_TS, LOTKA_Y_INIT, LOTKA_Y
export build_lotka_volterra_graph, demo
export LOTKA_VOLTERRA_SOURCE, evaluate_lotka_volterra_source

# posteriordb `hudson_lynx_hare-lotka_volterra`: the Hudson Bay lynx/hare
# two-state predator--prey system, integrated with the model's explicitly
# configured Stan RK45 controls. All measurement times and observations come
# from the full real PosteriorDB dataset.
let d = _posteriordb_data("hudson_lynx_hare-lotka_volterra")
    global const LOTKA_TS = Float64.(d["ts"])
    global const LOTKA_Y_INIT = Float64.(d["y_init"])
    global const LOTKA_Y = Float64.(d["y"])
end

const LOTKA_VOLTERRA_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, lognormal
using SciMLBase
using SciMLSensitivity
import OrdinaryDiffEqLowOrderRK: DP5

function lotka_rhs!(du, u, p, t)
    du[1] = (p[1] - p[2] * u[2]) * u[1]
    du[2] = (-p[3] + p[4] * u[1]) * u[2]
    return nothing
end

# Build the problem once outside the differentiated model callable. Active
# initial state and parameters are supplied to solve at call time; constructing
# or remaking an ODEProblem inside AD stores those active values in the problem
# object and triggers Enzyme runtime-activity analysis.
const LOTKA_PROBLEM = SciMLBase.ODEProblem(
    lotka_rhs!, [30.0, 4.0], (0.0, last(LOTKA_TS)),  # Stan's initial time is 0
    [0.55, 0.028, 0.80, 0.024],
)

function lotka_solve(u0, p, ts)
    sol = SciMLBase.solve(
        LOTKA_PROBLEM, DP5();
        u0 = u0,
        p = p,
        saveat = ts,
        reltol = 1e-5,
        abstol = 1e-3,
        maxiters = 500,
    )
    return reduce(hcat, sol.u)'
end

@kernel model(unconstrained::Vector{Float64},
              ts::Vector{Float64},
              y_init::Vector{Float64},
              y::Matrix{Float64}) = begin
    # Stan declaration order: theta[4], z_init[2], sigma[2], all lower=0 and
    # therefore represented in BridgeStan's unconstrained space by exp(q).
    theta_unc::AbstractVector{Float64} = view(unconstrained, 1:4)
    z_init_unc::AbstractVector{Float64} = view(unconstrained, 5:6)
    sigma_unc::AbstractVector{Float64} = view(unconstrained, 7:8)
    theta::Vector{Float64} = exp.(theta_unc)
    z_init::Vector{Float64} = exp.(z_init_unc)
    sigma::Vector{Float64} = exp.(sigma_unc)
    log_jacobian::Float64 = sum(unconstrained)
    parameters = (; theta, z_init, sigma)

    trajectory::Matrix{Float64} = lotka_solve(z_init, theta, ts)
    log_trajectory::Matrix{Float64} = log.(trajectory)

    prior_theta::Float64 =
        normal(1.0, 0.5).logpdf(theta[1]) +
        normal(0.05, 0.05).logpdf(theta[2]) +
        normal(1.0, 0.5).logpdf(theta[3]) +
        normal(0.05, 0.05).logpdf(theta[4])
    prior_scales::Float64 =
        lognormal(-1.0, 1.0).logpdf(sigma[1]) +
        lognormal(-1.0, 1.0).logpdf(sigma[2])
    prior_initial::Float64 =
        lognormal(log(10.0), 1.0).logpdf(z_init[1]) +
        lognormal(log(10.0), 1.0).logpdf(z_init[2])
    log_prior::Float64 = prior_theta + prior_scales + prior_initial

    initial_likelihood::Float64 =
        lognormal(log(z_init[1]), sigma[1]).logpdf(y_init[1]) +
        lognormal(log(z_init[2]), sigma[2]).logpdf(y_init[2])
    prey_likelihood_pointwise = plate(view(y, :, 1), view(log_trajectory, :, 1), sigma[1]) do observation, log_mean, scale
        lognormal(log_mean, scale).logpdf(observation)
    end
    predator_likelihood_pointwise = plate(view(y, :, 2), view(log_trajectory, :, 2), sigma[2]) do observation, log_mean, scale
        lognormal(log_mean, scale).logpdf(observation)
    end
    observation_likelihood::Float64 =
        sum(prey_likelihood_pointwise) + sum(predator_likelihood_pointwise)

    constrained_logdensity::Float64 =
        log_prior + initial_likelihood + observation_likelihood
    posterior::Float64 = constrained_logdensity + log_jacobian
    return posterior
end

q = [0.0, log(0.05), 0.0, log(0.05), log(30.0), log(4.0), -1.0, -1.0]
ts = LOTKA_TS
y_init = LOTKA_Y_INIT
y = LOTKA_Y

requested_nodes = (:parameters, :trajectory, :log_prior, :initial_likelihood,
                   :observation_likelihood, :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :ts, :y_init, :y),
    want = requested_nodes,
    bound = (; ts, y_init, y))

output = density_kernel(q)
parameters, trajectory, log_prior, initial_likelihood, observation_likelihood,
    log_jacobian, posterior = output
@assert posterior ≈ log_prior + initial_likelihood + observation_likelihood + log_jacobian

docs_example = (;
    name = :lotka_volterra_posterior,
    origin = "posteriordb hudson_lynx_hare-lotka_volterra — adaptive RK45",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    lognormal_object = lognormal,
)
"""

function evaluate_lotka_volterra_source(; model_only::Bool = false)
    _evaluate_ppl_source(LOTKA_VOLTERRA_SOURCE, @__MODULE__; bindings = (
        :LOTKA_TS, :LOTKA_Y_INIT, :LOTKA_Y,
    ), model_only)
end

const _LOTKA_VOLTERRA_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOTKA_VOLTERRA_GRAPH_TEMPLATE[] =
        evaluate_lotka_volterra_source(; model_only = true).model
    nothing
end

"""
    build_lotka_volterra_graph()

Build the posteriordb Lotka--Volterra model as a declarative
`ReactiveKernels.KernelSpec`. The eight positive parameters use Stan's exact
log transforms and Jacobian. The adaptive DP5 solve is a natural opaque graph
node; its `ODEProblem` is constructed outside AD and active `u0`/`p` are passed
to `solve` at call time.
Accepted evidence path under decision `008vhy5`: native primal/value only.
Ordinary-Reverse gradient is unsupported pending core snag
`plain-enzyme-rev-3dc5d563` (prepared-kernel activity-analysis gap; the
identical math as a plain function differentiates fine), and compiled
Reactant primal/gradient is unsupported pending the separate survey
(investigation todo `2026-09-17T14-41-05-845-1k4a6ep`).
"""
function build_lotka_volterra_graph()
    compose(_LOTKA_VOLTERRA_GRAPH_TEMPLATE[])
end

function demo()
    model = build_lotka_volterra_graph()
    q = [0.0, log(0.05), 0.0, log(0.05), log(30.0), log(4.0), -1.0, -1.0]
    kernel = prepare(model;
        have = (:unconstrained, :ts, :y_init, :y),
        want = (:parameters, :trajectory, :posterior),
        bound = (; ts = LOTKA_TS, y_init = LOTKA_Y_INIT, y = LOTKA_Y))
    parameters, trajectory, posterior = kernel(q)
    println("theta = ", parameters.theta)
    println("z_init = ", parameters.z_init)
    println("sigma = ", parameters.sigma)
    println("trajectory at t = ", LOTKA_TS[end], ": ", trajectory[end, :])
    println("unconstrained posterior = ", posterior)
    return nothing
end

end # module LotkaVolterraExample

if abspath(PROGRAM_FILE) == @__FILE__
    LotkaVolterraExample.demo()
end
