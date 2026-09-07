module ARMA11Example

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export ARMA_SERIES
export build_arma11_graph, demo
export ARMA11_SOURCE, evaluate_arma11_source

# A ReactiveKernels port of the `arma11` model from posteriordb
# (posterior `arma-arma11`): a scalar ARMA(1, 1) time series. The interesting
# structure is the *sequential* one-step-ahead error recursion carried inside
# the log density — a stateful computation, unlike the pointwise GLM examples.

# The real 200-point series from posteriordb (`arma` data).
const ARMA_SERIES = [0.731977, 0.662415, 0.945948, 0.901509, 1.006875, 0.946637,
    0.948778, 0.781079, 0.613273, 0.534024, 0.401598, 0.320459, 0.017771,
    0.019702, -0.229683, -0.380992, -0.401708, -0.72995, -0.726304, -1.021445,
    -0.967538, -0.859425, -1.205409, -0.966899, -0.857342, -1.000173, -0.939406,
    -0.855041, -0.700544, -0.399891, -0.51578, -0.099021, -0.052063, 0.225023,
    0.278993, 0.194437, 0.376614, 0.542823, 0.644386, 0.578309, 0.776933,
    0.556517, 0.58704, 0.734169, 0.684086, 0.514809, 0.640156, 0.404432,
    0.367594, 0.304009, 0.111691, 0.008163, 0.013685, -0.17159, -0.021198,
    0.081083, -0.21757, -0.147447, -0.153075, -0.160525, -0.296323, -0.155072,
    -0.054796, -0.110654, 0.007436, -0.053195, -0.006439, 0.260951, 0.126465,
    0.136788, 0.036766, 0.034216, 0.106868, -0.107268, -0.129762, -0.039305,
    -0.228579, -0.259555, -0.351538, -0.265314, -0.403232, -0.588314, -0.386214,
    -0.445297, -0.513062, -0.436614, -0.574873, -0.440405, -0.313912, -0.19292,
    -0.276975, -0.158547, -0.105033, 0.080951, 0.196393, 0.424559, 0.60433,
    0.589595, 0.66023, 0.611304, 0.926863, 0.653265, 0.892154, 1.035382,
    1.033097, 0.993893, 0.964193, 0.730898, 0.555726, 0.649464, 0.399487,
    0.131351, 0.092127, -0.02398, -0.126541, -0.490735, -0.523514, -0.663709,
    -0.597087, -0.633145, -0.908637, -0.753392, -1.119828, -1.041987, -0.961722,
    -0.834669, -0.732266, -0.738515, -0.521619, -0.359525, -0.573124, -0.291007,
    -0.038611, 0.062588, 0.105103, 0.373114, 0.400512, 0.582664, 0.688843,
    0.607633, 0.750171, 0.724275, 0.704799, 0.482801, 0.730943, 0.444734,
    0.381957, 0.298012, 0.360173, 0.262207, 0.195215, 0.260634, -0.036351,
    -0.083412, 0.022241, -0.152055, -0.307458, -0.137477, -0.172826, -0.329838,
    -0.362642, -0.347819, -0.244646, -0.181609, -0.068722, 0.05008, -0.118369,
    -0.10796, 0.015245, -0.048397, 0.034671, 0.018905, 0.039958, 0.043508,
    -0.259214, 0.034084, -0.25472, -0.23441, -0.407578, -0.549465, -0.341984,
    -0.417517, -0.537901, -0.503191, -0.498666, -0.402078, -0.509743, -0.622694,
    -0.26258, -0.32625, -0.431907, -0.315292, -0.125547, 0.122771, 0.167974,
    0.367001, 0.618939, 0.636397, 0.633471, 0.78147]

const ARMA11_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

@kernel model(unconstrained::Vector{Float64},
              series::Vector{Float64}) = begin
    # q = (μ, φ, θ, log_σ). One-element reductions extract the packed scalars
    # without scalar indexing, keeping the kernel Reactant-traceable.
    μ::Float64 = sum(view(unconstrained, 1:1))
    φ::Float64 = sum(view(unconstrained, 2:2))
    θ::Float64 = sum(view(unconstrained, 3:3))
    log_σ::Float64 = sum(view(unconstrained, 4:4))

    # Only σ has a support transform. Either σ or log_σ may be the authoritative
    # HAVE value; supplying both cuts both edges.
    log_σ::Float64 = log(σ)
    σ::Float64 = exp(log_σ)
    log_jacobian::Float64 = log_σ

    # Two producers for `parameters`, and inverse edges so the constrained
    # NamedTuple is also an authoritative input boundary.
    parameters = (; μ, φ, θ, σ)
    (parameters, log_jacobian::Float64) = ((; μ, φ, θ, σ), log_σ)
    (μ::Float64, φ::Float64, θ::Float64, σ::Float64) =
        (parameters.μ, parameters.φ, parameters.θ, parameters.σ)

    # The latent one-step-ahead errors are the sequential heart of the model:
    #   ν₁ = μ + φ·μ (err₀ ≡ 0), errₜ = yₜ − νₜ,
    #   νₜ = μ + φ·y_{t-1} + θ·err_{t-1}   (t ≥ 2).
    # This is the NATURAL authoring, expressed with the `scan` primitive: the
    # carry threads (y_{t-1}, err_{t-1}), seeded (μ, 0) so the unified step
    #   νₜ = μ + φ·carry.y_prev + θ·carry.err_prev
    # reproduces ν₁ = μ + φ·μ at t = 1 (y₀ ≡ μ, err₀ ≡ 0). `scan` lowers this
    # sequential recurrence to a `stablehlo.while` carry loop — so unlike the raw
    # `for`/`err[t-1]` authoring (forbidden scalar indexing of a traced array),
    # this natural form DOES lower through Reactant. The likelihood/density reduce
    # it directly; `errors_closed` below is an independent vectorized cross-check.
    errors::Vector{Float64} =
        scan(series, Ref(μ), Ref(φ), Ref(θ);
             init = (; y_prev = μ, err_prev = 0.0)) do carry, y, m, f, t
            ν = m + f * carry.y_prev + t * carry.err_prev
            e = y - ν
            ((; y_prev = y, err_prev = e), e)
        end

    # The SAME errors as a vectorized closed form — kept as an INDEPENDENT
    # numerical cross-check on the `scan` lowering (a test asserts the two equal).
    # The recurrence errₜ = aₜ − θ·err_{t-1} (with aₜ = yₜ − μ − φ·y_{t-1}) is
    # linear, so errₜ = Σ_{k≤t} (−θ)^{t−k} aₖ — a lower-triangular Toeplitz matvec
    # err = L·a. It uses only vectorized ops (slice, vcat, broadcast, a `.>=`
    # comparison mask, matmul), and also lowers through Reactant.
    errors_closed::Vector{Float64} = let
        T = length(series)
        y_lag = vcat(μ, series[1:(T - 1)])                  # y_{t-1}, with y₀ ≡ μ
        a = series .- μ .- φ .* y_lag
        Δ = (0:(T - 1)) .- (0:(T - 1))'                     # Δ[t, k] = t − k
        L = (Δ .>= 0) .* ((-θ) .^ max.(Δ, 0))               # lower-tri Toeplitz
        L * a
    end

    # Log prior: μ ~ Normal(0, 10), φ, θ ~ Normal(0, 2), σ ~ HalfCauchy(2.5).
    # The half-Cauchy folds the reusable Cauchy endpoint with the log(2)
    # truncation constant.
    μ_prior::Float64 = normal(0.0, 10.0).logpdf(μ)
    φ_prior::Float64 = normal(0.0, 2.0).logpdf(φ)
    θ_prior::Float64 = normal(0.0, 2.0).logpdf(θ)
    σ_cauchy::Float64 = cauchy(0.0, 2.5).logpdf(σ)
    σ_prior::Float64 = log(2.0) + σ_cauchy
    prior::Float64 = μ_prior + φ_prior + θ_prior + σ_prior

    # One authored likelihood plate over the latent errors: errₜ ~ Normal(0, σ).
    # It reduces the NATURAL sequential `errors` (the `scan` form), so the whole
    # density compiles through Reactant off the natural recursion; the scalar σ
    # broadcasts across the error vector.
    pointwise = plate(errors, σ) do e, s
        normal(0.0, s).logpdf(e)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    density::Float64 = constrained_logdensity + log_jacobian

    # Deterministic one-step-ahead point forecast ν_{T+1} = μ + φ·y_T + θ·err_T,
    # read off the constrained parameters and the last recursion error.
    forecast::Float64 =
        parameters.μ + parameters.φ * series[end] + parameters.θ * errors[end]

    return density
end

q = [0.0, 0.9, -0.2, log(0.15)]
series = ARMA_SERIES

requested_nodes = (:prior, :log_jacobian, :pointwise, :likelihood, :density)
density_kernel = prepare(model;
    have = (:unconstrained, :series),
    want = requested_nodes)

output = density_kernel(q, series)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :arma11_density,
    origin = "Inline ARMA(1,1) with a sequential error recursion — posteriordb arma11",
    inputs = (; q, series),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    cauchy_object = cauchy,
)
"""

function evaluate_arma11_source()
    # Bind only the data. The authored source imports the reusable Normal and
    # Cauchy distribution objects itself and contains the complete PPL assembly,
    # including the inline error recursion, with no helper evaluator.
    _evaluate_ppl_source(ARMA11_SOURCE, @__MODULE__; bindings = (
        :ARMA_SERIES,
    ))
end

# Evaluate the authored source from `__init__`, after package precompilation has
# closed the module. `build_arma11_graph` clones this runtime template so every
# caller gets an independent mutable graph without crossing a fresh `Core.eval`
# world-age boundary inside its own compiled function.
const _ARMA11_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _ARMA11_GRAPH_TEMPLATE[] = evaluate_arma11_source().model
    nothing
end

"""
    build_arma11_graph()

Build the posteriordb ARMA(1, 1) model as a declarative
`ReactiveKernels.KernelSpec`. The latent one-step errors are computed by a
sequential recursion authored inline with the `scan` primitive and exposed as
their own port, so a query can ask for just the errors, the full density, or the
one-step forecast. `scan` lowers the recurrence to a `stablehlo.while` carry
loop, so the likelihood/density reduce the natural sequential `errors` directly
and the whole density lowers through Reactant; a vectorized closed-form
equivalent (`errors_closed`) is kept as an independent numerical cross-check. The Normal and Cauchy endpoints are reused from
`ReactiveKernelsDistributionKernels`; the half-Cauchy prior on `σ` folds the
Cauchy endpoint with the `log(2)` truncation constant. The support transform +
Jacobian, prior, pointwise log-likelihood, likelihood reduction, and total
density remain separate nodes, and constrained parameters are a plain NamedTuple.
"""
function build_arma11_graph()
    compose(_ARMA11_GRAPH_TEMPLATE[])
end

function demo()
    model = build_arma11_graph()
    q = [0.0, 0.9, -0.2, log(0.15)]

    println("Latent one-step errors only (density branches pruned):")
    errors_plan = plan(model; have = (:unconstrained, :series), want = :errors)
    println(explain(errors_plan))
    errors = prepare(errors_plan)(q, ARMA_SERIES)
    println("first five errors = ", errors[1:5])

    println("\nFull unconstrained-space log density:")
    density_plan = plan(model;
                        have = (:unconstrained, :series),
                        want = (:prior, :log_jacobian, :likelihood, :density,
                                :forecast))
    println(explain(density_plan))
    prior, log_jacobian, likelihood, density, forecast =
        prepare(density_plan)(q, ARMA_SERIES)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log density = ", density)
    println("one-step-ahead forecast for y[T+1] = ", forecast)

    nothing
end

end # module ARMA11Example

if abspath(PROGRAM_FILE) == @__FILE__
    ARMA11Example.demo()
end
