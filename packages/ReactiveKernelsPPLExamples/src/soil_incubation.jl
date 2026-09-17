module SoilIncubationExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export SOIL_T0, SOIL_TOTAL_C_T0, SOIL_TS, SOIL_ECO2MEAN
export build_soil_incubation_graph, demo
export SOIL_INCUBATION_SOURCE, evaluate_soil_incubation_source

# posteriordb `soil_carbon-soil_incubation`: a two-pool soil-carbon model with
# feedback, integrated with Stan's unconfigured RK45 defaults on the full real
# evolved-CO2 measurements. The data file additionally carries `eCO2sd`, which
# the Stan data block does not declare, so it is not a model input.
let d = _posteriordb_data("soil_carbon-soil_incubation")
    global const SOIL_T0 = Float64(d["t0"])
    global const SOIL_TOTAL_C_T0 = Float64(d["totalC_t0"])
    global const SOIL_TS = Float64.(d["ts"])
    global const SOIL_ECO2MEAN = Float64.(d["eCO2mean"])
end

const SOIL_INCUBATION_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, cauchy, beta
using LogExpFunctions: logistic, log1pexp
using SciMLBase
using SciMLSensitivity
import OrdinaryDiffEqLowOrderRK: DP5

function soil_rhs!(du, u, p, t)
    k1, k2, alpha21, alpha12 = p
    du[1] = -k1 * u[1] + alpha12 * k2 * u[2]
    du[2] = -k2 * u[2] + alpha21 * k1 * u[1]
    return nothing
end

# As with the other dynamics models, the problem is a constant outside AD.
# Here BOTH the initial pool state (through the active gamma partition) and
# the transformed theta vector are active values supplied to solve at call
# time; constructing or remaking an ODEProblem inside AD stores those active
# values in the problem object and triggers Enzyme runtime-activity analysis.
# The exemplar state is the gamma = 0.5 partition of the real total carbon.
const SOIL_PROBLEM = SciMLBase.ODEProblem(
    soil_rhs!, [3.85, 3.85], (SOIL_T0, last(SOIL_TS)), [0.1, 0.1, 0.1, 0.1],
)

function soil_solve(c_t0, p, ts)
    sol = SciMLBase.solve(
        SOIL_PROBLEM, DP5();
        u0 = c_t0,
        p = p,
        saveat = ts,
        reltol = 1e-6,
        abstol = 1e-6,
        maxiters = 1_000_000,
    )
    return reduce(hcat, sol.u)'
end

@kernel model(unconstrained::Vector{Float64},
              ts::Vector{Float64},
              total_c_t0::Float64,
              eco2_mean::Vector{Float64}) = begin
    # Stan parameter order: k1, k2, alpha21, alpha12 lower=0 exp transforms;
    # gamma in [0, 1] is a logistic transform; sigma lower=0 exp.
    k1::Float64 = exp(unconstrained[1])
    k2::Float64 = exp(unconstrained[2])
    alpha21::Float64 = exp(unconstrained[3])
    alpha12::Float64 = exp(unconstrained[4])
    gamma_unc::Float64 = unconstrained[5]
    gamma::Float64 = logistic(gamma_unc)
    sigma::Float64 = exp(unconstrained[6])
    log_jacobian::Float64 = unconstrained[1] + unconstrained[2] +
        unconstrained[3] + unconstrained[4] + unconstrained[6] -
        log1pexp(-gamma_unc) - log1pexp(gamma_unc)
    parameters = (; k1, k2, alpha21, alpha12, gamma, sigma)

    c_t0::Vector{Float64} = vcat(gamma * total_c_t0, (1.0 - gamma) * total_c_t0)
    theta::Vector{Float64} = vcat(k1, k2, alpha21, alpha12)
    trajectory::Matrix{Float64} = soil_solve(c_t0, theta, ts)

    # Stan's evolved CO2 is the initial total minus the summed pools.
    pool1::AbstractVector{Float64} = view(trajectory, :, 1)
    pool2::AbstractVector{Float64} = view(trajectory, :, 2)
    evolved::Vector{Float64} = total_c_t0 .- pool1 .- pool2

    log_prior::Float64 =
        normal(0.0, 1.0).logpdf(k1) +
        normal(0.0, 1.0).logpdf(k2) +
        normal(0.0, 1.0).logpdf(alpha21) +
        normal(0.0, 1.0).logpdf(alpha12) +
        beta(10.0, 1.0).logpdf(gamma) +
        cauchy(0.0, 1.0).logpdf(sigma)

    evolution_pointwise = plate(eco2_mean, evolved, Ref(sigma)) do observation, mu, s
        normal(mu, s).logpdf(observation)
    end
    likelihood::Float64 = sum(evolution_pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    posterior::Float64 = constrained_logdensity + log_jacobian
    return posterior
end

q = [log(0.1), log(0.1), log(0.1), log(0.1), 0.0, log(1.0)]
ts = SOIL_TS
total_c_t0 = SOIL_TOTAL_C_T0
eco2_mean = SOIL_ECO2MEAN

requested_nodes = (:parameters, :trajectory, :log_prior, :likelihood,
                   :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :ts, :total_c_t0, :eco2_mean),
    want = requested_nodes,
    bound = (; ts, total_c_t0, eco2_mean))

output = density_kernel(q)
parameters, trajectory, log_prior, likelihood, log_jacobian, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :soil_incubation_posterior,
    origin = "posteriordb soil_carbon-soil_incubation — adaptive RK45",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    cauchy_object = cauchy,
    beta_object = beta,
)
"""

function evaluate_soil_incubation_source(; model_only::Bool = false)
    _evaluate_ppl_source(SOIL_INCUBATION_SOURCE, @__MODULE__; bindings = (
        :SOIL_T0, :SOIL_TOTAL_C_T0, :SOIL_TS, :SOIL_ECO2MEAN,
    ), model_only)
end

const _SOIL_INCUBATION_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _SOIL_INCUBATION_GRAPH_TEMPLATE[] =
        evaluate_soil_incubation_source(; model_only = true).model
    nothing
end

"""
    build_soil_incubation_graph()

Build the posteriordb soil-carbon incubation model as a declarative
`ReactiveKernels.KernelSpec`. The four pool rates and sigma use Stan's exp
transforms, gamma uses the logistic transform with its exact Jacobian, and
every real data input remains a named graph input. The initial pool state is
an active in-graph partition passed to `solve` as `u0`; the unconfigured Stan
RK45 defaults are mapped to explicit DP5 controls `1e-6`, `1e-6`, and
`1_000_000` through a constant prebuilt problem.
"""
function build_soil_incubation_graph()
    compose(_SOIL_INCUBATION_GRAPH_TEMPLATE[])
end

function demo()
    model = build_soil_incubation_graph()
    q = [log(0.1), log(0.1), log(0.1), log(0.1), 0.0, log(1.0)]
    kernel = prepare(model;
        have = (:unconstrained, :ts, :total_c_t0, :eco2_mean),
        want = (:parameters, :trajectory, :posterior),
        bound = (; ts = SOIL_TS, total_c_t0 = SOIL_TOTAL_C_T0,
                 eco2_mean = SOIL_ECO2MEAN))
    parameters, trajectory, posterior = kernel(q)
    println("parameters = ", parameters)
    println("final state = ", trajectory[end, :])
    println("unconstrained posterior = ", posterior)
    return nothing
end

end # module SoilIncubationExample

if abspath(PROGRAM_FILE) == @__FILE__
    SoilIncubationExample.demo()
end
