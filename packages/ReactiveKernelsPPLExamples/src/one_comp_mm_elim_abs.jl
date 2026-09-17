module OneCompMMElimAbsExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export ONECOMP_T0, ONECOMP_D, ONECOMP_V, ONECOMP_TIMES, ONECOMP_C0, ONECOMP_C_HAT
export build_one_comp_mm_elim_abs_graph, demo
export ONE_COMP_MM_ELIM_ABS_SOURCE, evaluate_one_comp_mm_elim_abs_source

# posteriordb `one_comp_mm_elim_abs-one_comp_mm_elim_abs`: a one-compartment
# pharmacokinetic model with first-order absorption and Michaelis–Menten
# elimination, integrated with Stan's unconfigured BDF defaults on the full
# real concentration measurements. Stan packs the dose and compartment volume
# as real data `x_r = {D, V}` entering the RHS; they are named module data
# constants captured outside AD, exactly like the prebuilt problem itself.
# The transformed-data initial concentration is hardcoded `C0 = {0.0}`.
let d = _posteriordb_data("one_comp_mm_elim_abs-one_comp_mm_elim_abs")
    global const ONECOMP_T0 = Float64(d["t0"])
    global const ONECOMP_D = Float64(d["D"])
    global const ONECOMP_V = Float64(d["V"])
    global const ONECOMP_TIMES = Float64.(d["times"])
    global const ONECOMP_C0 = [0.0]
    global const ONECOMP_C_HAT = Float64.(d["C_hat"])
end

const ONE_COMP_MM_ELIM_ABS_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    cauchy, lognormal
using SciMLBase
using SciMLSensitivity
import OrdinaryDiffEqBDF: FBDF

function onecomp_rhs!(du, u, p, t)
    k_a, K_m, V_m = p
    dose = t > 0 ? exp(-k_a * t) * ONECOMP_D * k_a / ONECOMP_V : 0.0
    elim = (V_m / ONECOMP_V) * u[1] / (K_m + u[1])
    du[1] = dose - elim
    return nothing
end

# As with Lotka–Volterra and SIR, the problem is a constant outside AD. The
# bound raw initial concentration and the transformed theta vector are supplied
# to solve at call time; constructing or remaking an ODEProblem inside AD
# stores those active values in the problem object and triggers Enzyme
# runtime-activity analysis. Stan's unconfigured BDF defaults are
# reltol = abstol = 1e-10 with max_num_steps = 1e8.
const ONECOMP_PROBLEM = SciMLBase.ODEProblem(
    onecomp_rhs!, [0.0], (ONECOMP_T0, last(ONECOMP_TIMES)), [1.0, 1.0, 1.0],
)

function onecomp_solve(c0, p, times)
    sol = SciMLBase.solve(
        ONECOMP_PROBLEM, FBDF();
        u0 = c0,
        p = p,
        saveat = times,
        reltol = 1e-10,
        abstol = 1e-10,
        maxiters = 100_000_000,
    )
    return reduce(hcat, sol.u)'
end

@kernel model(unconstrained::Vector{Float64},
              times::Vector{Float64},
              c0::Vector{Float64},
              c_hat::Vector{Float64}) = begin
    # Stan parameter order: k_a, K_m, V_m, sigma; all are lower=0 exp transforms.
    k_a::Float64 = exp(unconstrained[1])
    K_m::Float64 = exp(unconstrained[2])
    V_m::Float64 = exp(unconstrained[3])
    sigma::Float64 = exp(unconstrained[4])
    log_jacobian::Float64 = sum(unconstrained)
    parameters = (; k_a, K_m, V_m, sigma)

    theta::Vector{Float64} = vcat(k_a, K_m, V_m)
    trajectory::Matrix{Float64} = onecomp_solve(c0, theta, times)

    log_prior::Float64 =
        cauchy(0.0, 1.0).logpdf(k_a) +
        cauchy(0.0, 1.0).logpdf(K_m) +
        cauchy(0.0, 1.0).logpdf(V_m) +
        cauchy(0.0, 1.0).logpdf(sigma)

    # Stan observes lognormal concentrations around the single compartment
    # state. No clamping: a nonpositive solve keeps Stan's exact -Inf shape.
    predicted::AbstractVector{Float64} = view(trajectory, :, 1)
    concentration_pointwise = plate(c_hat, predicted, Ref(sigma)) do observation, mu, s
        lognormal(log(mu), s).logpdf(observation)
    end
    likelihood::Float64 = sum(concentration_pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    posterior::Float64 = constrained_logdensity + log_jacobian
    return posterior
end

q = [log(1.0), log(1.0), log(1.0), log(1.0)]
times = ONECOMP_TIMES
c0 = ONECOMP_C0
c_hat = ONECOMP_C_HAT

requested_nodes = (:parameters, :trajectory, :log_prior, :likelihood,
                   :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :times, :c0, :c_hat),
    want = requested_nodes,
    bound = (; times, c0, c_hat))

output = density_kernel(q)
parameters, trajectory, log_prior, likelihood, log_jacobian, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :one_comp_mm_elim_abs_posterior,
    origin = "posteriordb one_comp_mm_elim_abs-one_comp_mm_elim_abs — adaptive BDF",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    cauchy_object = cauchy,
    lognormal_object = lognormal,
)
"""

function evaluate_one_comp_mm_elim_abs_source(; model_only::Bool = false)
    _evaluate_ppl_source(ONE_COMP_MM_ELIM_ABS_SOURCE, @__MODULE__; bindings = (
        :ONECOMP_T0, :ONECOMP_D, :ONECOMP_V, :ONECOMP_TIMES, :ONECOMP_C0,
        :ONECOMP_C_HAT,
    ), model_only)
end

const _ONE_COMP_MM_ELIM_ABS_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _ONE_COMP_MM_ELIM_ABS_GRAPH_TEMPLATE[] =
        evaluate_one_comp_mm_elim_abs_source(; model_only = true).model
    nothing
end

"""
    build_one_comp_mm_elim_abs_graph()

Build the posteriordb one-compartment Michaelis–Menten model as a declarative
`ReactiveKernels.KernelSpec`. The four positive parameters use Stan's exp
transforms and Jacobian; the dose/volume `x_r` data stay named constants
outside AD, and every real measurement input remains a named graph input. The
unconfigured Stan BDF defaults are mapped to explicit FBDF controls `1e-10`,
`1e-10`, and `100_000_000` through a constant prebuilt problem.
"""
function build_one_comp_mm_elim_abs_graph()
    compose(_ONE_COMP_MM_ELIM_ABS_GRAPH_TEMPLATE[])
end

function demo()
    model = build_one_comp_mm_elim_abs_graph()
    q = [log(1.0), log(1.0), log(1.0), log(1.0)]
    kernel = prepare(model;
        have = (:unconstrained, :times, :c0, :c_hat),
        want = (:parameters, :trajectory, :posterior),
        bound = (; times = ONECOMP_TIMES, c0 = ONECOMP_C0,
                 c_hat = ONECOMP_C_HAT))
    parameters, trajectory, posterior = kernel(q)
    println("parameters = ", parameters)
    println("final state = ", trajectory[end, :])
    println("unconstrained posterior = ", posterior)
    return nothing
end

end # module OneCompMMElimAbsExample

if abspath(PROGRAM_FILE) == @__FILE__
    OneCompMMElimAbsExample.demo()
end
