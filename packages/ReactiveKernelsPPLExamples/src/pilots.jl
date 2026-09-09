module PilotsExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export PILOTS_GROUP_ID, PILOTS_SCENARIO_ID, PILOTS_Y
export build_pilots_graph, demo
export PILOTS_SOURCE, evaluate_pilots_source

# posteriordb `pilots-pilots` — the Gelman & Hill "pilots" two-way crossed
# random-effects model: N = 40 observations, each with a group (n_groups = 5)
# and a scenario (n_scenarios = 8), and a continuous outcome y. The mean is
# y_hat[i] = a[group_id[i]] + b[scenario_id[i]] (two integer-array gathers). The
# full real dataset (N = 40) is embedded verbatim.
# Real data (full) from posteriordb `pilots-pilots`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("pilots-pilots")
    global const PILOTS_GROUP_ID = Int.(d["group_id"])
    global const PILOTS_SCENARIO_ID = Int.(d["scenario_id"])
    global const PILOTS_Y = Float64.(d["y"])
end

const PILOTS_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              group_id::Vector{Int},
              scenario_id::Vector{Int},
              y::Vector{Float64}) = begin
    # Stan's declared unconstrained order: (a[1..n_groups], b[1..n_scenarios],
    # mu_a, mu_b, sigma_a, sigma_b, sigma_y); dim = n_groups + n_scenarios + 5.
    # n_groups = 5 and n_scenarios = 8 are structural constants of this dataset. a,
    # b, mu_a, mu_b are unconstrained; the three sigmas are real<lower=0,
    # upper=100>.
    n_groups::Int = 5
    n_scenarios::Int = 8
    a::AbstractVector{Float64} = view(unconstrained, 1:n_groups)
    b::AbstractVector{Float64} = view(unconstrained, n_groups + 1:n_groups + n_scenarios)
    u_mu_a::Float64 = unconstrained[n_groups + n_scenarios + 1]
    u_mu_b::Float64 = unconstrained[n_groups + n_scenarios + 2]
    u_sigma_a::Float64 = unconstrained[n_groups + n_scenarios + 3]
    u_sigma_b::Float64 = unconstrained[n_groups + n_scenarios + 4]
    u_sigma_y::Float64 = unconstrained[n_groups + n_scenarios + 5]

    # sigma ∈ [0, 100]: scaled-logit interval transform sigma = 100*logistic(u);
    # log|dsigma/du| = log(100) - log1pexp(-u) - log1pexp(u). mu_a/mu_b identity.
    mu_a::Float64 = u_mu_a
    mu_b::Float64 = u_mu_b
    sigma_a::Float64 = 100.0 * logistic(u_sigma_a)
    sigma_b::Float64 = 100.0 * logistic(u_sigma_b)
    sigma_y::Float64 = 100.0 * logistic(u_sigma_y)
    jac_sigma_a::Float64 = log(100.0) - log1pexp(-u_sigma_a) - log1pexp(u_sigma_a)
    jac_sigma_b::Float64 = log(100.0) - log1pexp(-u_sigma_b) - log1pexp(u_sigma_b)
    jac_sigma_y::Float64 = log(100.0) - log1pexp(-u_sigma_y) - log1pexp(u_sigma_y)
    log_jacobian::Float64 = jac_sigma_a + jac_sigma_b + jac_sigma_y

    parameters = (; a, b, mu_a, mu_b, sigma_a, sigma_b, sigma_y)
    (parameters, log_jacobian::Float64) =
        ((; a, b, mu_a, mu_b, sigma_a, sigma_b, sigma_y),
         jac_sigma_a + jac_sigma_b + jac_sigma_y)
    (a::AbstractVector{Float64}, b::AbstractVector{Float64},
     mu_a::Float64, mu_b::Float64,
     sigma_a::Float64, sigma_b::Float64, sigma_y::Float64) =
        (parameters.a, parameters.b, parameters.mu_a, parameters.mu_b,
         parameters.sigma_a, parameters.sigma_b, parameters.sigma_y)

    # Hyperpriors: mu_a ~ Normal(0, 1), mu_b ~ Normal(0, 1). The three sigmas
    # have NO `~` statement (implicit uniform over [0, 100]); an unwritten prior
    # on a bounded parameter is flat and contributes 0 (only its transform
    # Jacobian enters, above), so the prior term below excludes them.
    mu_a_prior::Float64 = normal(0.0, 1.0).logpdf(mu_a)
    mu_b_prior::Float64 = normal(0.0, 1.0).logpdf(mu_b)

    # Group / scenario random effects: a_j ~ Normal(10*mu_a, sigma_a),
    # b_k ~ Normal(10*mu_b, sigma_b). The shared location/scale ride the plate
    # as scalar args (10*mu is computed as a node so the cell body stays inline).
    mu_a10::Float64 = 10.0 * mu_a
    mu_b10::Float64 = 10.0 * mu_b
    a_pointwise = plate(a, mu_a10, sigma_a) do aa, m, s
        normal(m, s).logpdf(aa)
    end
    b_pointwise = plate(b, mu_b10, sigma_b) do bb, m, s
        normal(m, s).logpdf(bb)
    end
    a_prior::Float64 = sum(a_pointwise)
    b_prior::Float64 = sum(b_pointwise)
    prior::Float64 = mu_a_prior + mu_b_prior + a_prior + b_prior

    # Transformed parameter: y_hat[i] = a[group_id[i]] + b[scenario_id[i]]. The
    # two integer-array gathers are done OUTSIDE any plate (a traced integer
    # index does not lower; docs_example binds both indices). This is the named
    # transformed-parameter / generated-quantity node.
    a_gathered = a[group_id]
    b_gathered = b[scenario_id]
    y_hat = plate(a_gathered, b_gathered) do ai, bi
        ai + bi
    end

    # Likelihood: y[i] ~ Normal(y_hat[i], sigma_y). sigma_y rides the plate as a
    # shared scalar arg; y_hat is the gathered mean vector.
    pointwise = plate(y, y_hat, sigma_y) do yi, yh, s
        normal(yh, s).logpdf(yi)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = vcat(0.1 .* collect(1:5), -0.05 .* collect(1:8), [0.1, -0.1, 0.0, 0.0, 0.0])
group_id = PILOTS_GROUP_ID
scenario_id = PILOTS_SCENARIO_ID
y = PILOTS_Y

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
# Bind the integer gather indices so `a.inputs` excludes them and the Reactant
# path traces only the float inputs + the parameter vector (spec addendum).
density_kernel = prepare(model;
    have = (:unconstrained, :group_id, :scenario_id, :y),
    want = requested_nodes,
    bound = (; group_id, scenario_id, y))

output = density_kernel(q)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :pilots_posterior,
    origin = "posteriordb pilots — two-way crossed random-effects Gaussian model",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_pilots_source(; model_only::Bool = false)
    _evaluate_ppl_source(PILOTS_SOURCE, @__MODULE__; bindings = (
        :PILOTS_GROUP_ID, :PILOTS_SCENARIO_ID, :PILOTS_Y,
    ), model_only)
end

const _PILOTS_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _PILOTS_GRAPH_TEMPLATE[] = evaluate_pilots_source(; model_only = true).model
    nothing
end

"""
    build_pilots_graph()

Build the posteriordb `pilots` model (a two-way crossed random-effects Gaussian
model, `y_hat[i] = a[group_id[i]] + b[scenario_id[i]]`) as a declarative
`ReactiveKernels.KernelSpec`. The three `sigma ∈ [0, 100]` parameters use exact
scaled-logit interval transforms and Jacobians and carry an implicit-uniform
(flat, zero-contribution) prior; the group/scenario random-effect priors and the
Gaussian likelihood reuse the shared Normal endpoint, and the per-observation
mean is assembled by two integer-array gathers outside any plate. The transform
Jacobian, priors, gathered `y_hat`, pointwise/summed likelihood, densities and
posterior are named nodes.
"""
function build_pilots_graph()
    compose(_PILOTS_GRAPH_TEMPLATE[])
end

function demo()
    model = build_pilots_graph()
    q = vcat(0.1 .* collect(1:5), -0.05 .* collect(1:8), [0.1, -0.1, 0.0, 0.0, 0.0])
    posterior_plan = plan(model;
                          have = (:unconstrained, :group_id, :scenario_id, :y),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, PILOTS_GROUP_ID, PILOTS_SCENARIO_ID, PILOTS_Y)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module PilotsExample

if abspath(PROGRAM_FILE) == @__FILE__
    PilotsExample.demo()
end
