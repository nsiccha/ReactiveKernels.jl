module SIRExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export SIR_T, SIR_Y0, SIR_STOI_HAT, SIR_B_HAT
export build_sir_graph, demo
export SIR_SOURCE, evaluate_sir_source

# posteriordb `sir-sir`: a four-state SIR model with environmental bacteria,
# integrated with Stan's unconfigured RK45 defaults on the full real outbreak
# counts and bacterial measurements.
let d = _posteriordb_data("sir-sir")
    global const SIR_T = Float64.(d["t"])
    global const SIR_Y0 = Float64.(d["y0"])
    global const SIR_STOI_HAT = Int.(d["stoi_hat"])
    global const SIR_B_HAT = Float64.(d["B_hat"])
end

const SIR_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    cauchy, lognormal, poisson
using SciMLBase
using SciMLSensitivity
import OrdinaryDiffEqLowOrderRK: DP5

function sir_rhs!(du, u, p, t)
    beta, kappa, gamma, xi, delta = p
    force_of_infection = beta * u[4] / (u[4] + kappa)
    du[1] = -force_of_infection * u[1]
    du[2] = force_of_infection * u[1] - gamma * u[2]
    du[3] = gamma * u[2]
    du[4] = xi * u[2] - delta * u[4]
    return nothing
end

# As with Lotka–Volterra, the problem is a constant outside AD. The bound raw
# initial state y0 and the transformed theta vector are supplied to solve.
const SIR_PROBLEM = SciMLBase.ODEProblem(
    sir_rhs!, [10000.0, 0.0, 0.0, 10000.0],
    (0.0, last(SIR_T)), [0.1, 1.0e6, 0.1, 1.0, 1.0],
)

function sir_solve(y0, p, t)
    sol = SciMLBase.solve(
        SIR_PROBLEM, DP5();
        u0 = y0,
        p = p,
        saveat = t,
        reltol = 1e-6,
        abstol = 1e-6,
        maxiters = 1_000_000,
    )
    return reduce(hcat, sol.u)'
end

@kernel model(unconstrained::Vector{Float64},
              t::Vector{Float64},
              y0::Vector{Float64},
              stoi_hat::Vector{Int},
              B_hat::Vector{Float64}) = begin
    # Stan parameter order: beta, gamma, xi, delta; all are lower=0 exp transforms.
    beta::Float64 = exp(unconstrained[1])
    gamma::Float64 = exp(unconstrained[2])
    xi::Float64 = exp(unconstrained[3])
    delta::Float64 = exp(unconstrained[4])
    log_jacobian::Float64 = sum(unconstrained)
    parameters = (; beta, gamma, xi, delta)

    # Stan derives this fixed transformed-data scalar in the model. Keep it as
    # a named graph node and include it in the natural solver parameter vector.
    kappa::Float64 = 1.0e6
    theta::Vector{Float64} = vcat(beta, kappa, gamma, xi, delta)
    trajectory::Matrix{Float64} = sir_solve(y0, theta, t)

    log_prior::Float64 =
        cauchy(0.0, 2.5).logpdf(beta) +
        cauchy(0.0, 1.0).logpdf(gamma) +
        cauchy(0.0, 25.0).logpdf(xi) +
        cauchy(0.0, 1.0).logpdf(delta)

    # Stan observes Poisson decrements of the susceptible state. The first
    # decrement is y0[1] - y[1,1]; every later one is y[n-1,1] - y[n,1].
    n_times::Int = length(stoi_hat)
    susceptible::AbstractVector{Float64} = view(trajectory, :, 1)
    incident_rates::Vector{Float64} = vcat(
        y0[1] - susceptible[1],
        view(susceptible, 1:(n_times - 1)) - view(susceptible, 2:n_times),
    )
    stoichiometric_pointwise = plate(stoi_hat, incident_rates) do count, rate
        poisson(rate).logpdf(count)
    end
    bacteria_pointwise = plate(B_hat, view(trajectory, :, 4)) do observation, state
        lognormal(log(state), 0.15).logpdf(observation)
    end
    likelihood::Float64 = sum(stoichiometric_pointwise) + sum(bacteria_pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    posterior::Float64 = constrained_logdensity + log_jacobian
    return posterior
end

q = [log(0.1), log(0.1), log(1.0), log(1.0)]
t = SIR_T
y0 = SIR_Y0
stoi_hat = SIR_STOI_HAT
B_hat = SIR_B_HAT

requested_nodes = (:parameters, :trajectory, :log_prior, :likelihood,
                   :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :t, :y0, :stoi_hat, :B_hat),
    want = requested_nodes,
    bound = (; t, y0, stoi_hat, B_hat))

output = density_kernel(q)
parameters, trajectory, log_prior, likelihood, log_jacobian, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :sir_posterior,
    origin = "posteriordb sir-sir — adaptive RK45",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    cauchy_object = cauchy,
    poisson_object = poisson,
    lognormal_object = lognormal,
)
"""

function evaluate_sir_source(; model_only::Bool = false)
    _evaluate_ppl_source(SIR_SOURCE, @__MODULE__; bindings = (
        :SIR_T, :SIR_Y0, :SIR_STOI_HAT, :SIR_B_HAT,
    ), model_only)
end

const _SIR_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _SIR_GRAPH_TEMPLATE[] = evaluate_sir_source(; model_only = true).model
    nothing
end

"""
    build_sir_graph()

Build the posteriordb SIR model as a declarative `ReactiveKernels.KernelSpec`.
The four positive parameters use Stan's exp transforms and Jacobian; the fixed
`kappa=1e6` transformed-data value and every real data input remain named graph
inputs. The unconfigured Stan RK45 defaults are mapped to explicit DP5 controls
`1e-6`, `1e-6`, and `1_000_000` through a constant prebuilt problem.
"""
function build_sir_graph()
    compose(_SIR_GRAPH_TEMPLATE[])
end

function demo()
    model = build_sir_graph()
    q = [log(0.1), log(0.1), log(1.0), log(1.0)]
    kernel = prepare(model;
        have = (:unconstrained, :t, :y0, :stoi_hat, :B_hat),
        want = (:parameters, :trajectory, :posterior),
        bound = (; t = SIR_T, y0 = SIR_Y0, stoi_hat = SIR_STOI_HAT,
                 B_hat = SIR_B_HAT))
    parameters, trajectory, posterior = kernel(q)
    println("parameters = ", parameters)
    println("final state = ", trajectory[end, :])
    println("unconstrained posterior = ", posterior)
    return nothing
end

end # module SIRExample

if abspath(PROGRAM_FILE) == @__FILE__
    SIRExample.demo()
end
