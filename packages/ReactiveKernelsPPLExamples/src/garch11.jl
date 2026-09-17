module GARCH11Example

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export GARCH11_Y, GARCH11_SIGMA1
export build_garch11_graph, demo
export GARCH11_SOURCE, evaluate_garch11_source

# A ReactiveKernels port of the `garch11` model from posteriordb
# (posterior `garch-garch11`): a Gaussian GARCH(1,1) with a mean `mu`. The
# interesting structure is the *sequential* conditional-standard-deviation
# recursion carried inside the log density — a stateful computation, like
# `arma11`, unlike the pointwise GLM examples.
#
# Stan parameter-block order is `mu`, `alpha0` (lower=0), `alpha1`
# (lower=0,upper=1), `beta1` (lower=0,upper=(1-alpha1)); the unconstrained vector
# is q = (mu, u_alpha0, u_alpha1, u_beta1), dim = 4. The support transforms are
# inlined: alpha0 via exp (Jacobian u_alpha0); alpha1 via the (0,1) logistic
# interval transform; beta1 via the (0, 1-alpha1) scaled-logistic transform whose
# UPPER BOUND depends on the already-constrained alpha1 (Stan constrains in
# declaration order, so beta1's Jacobian carries log(1-alpha1)). There are NO
# explicit priors — the only density terms are the likelihood and the transform
# Jacobian (Stan's improper-flat-on-the-constrained-support parametrization).

# Real data (full) from posteriordb `garch-garch11`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("garch-garch11")
    global const GARCH11_Y = Float64.(d["y"])
    global const GARCH11_SIGMA1 = Float64(d["sigma1"])
end

const GARCH11_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Float64},
              sigma1::Float64) = begin
    # q = (mu, u_alpha0, u_alpha1, u_beta1) — Stan parameter-block order. dim = 4.
    mu::Float64       = unconstrained[1]
    u_alpha0::Float64 = unconstrained[2]
    u_alpha1::Float64 = unconstrained[3]
    u_beta1::Float64  = unconstrained[4]

    # Support transforms + Jacobian. Stan constrains in declaration order, so
    # beta1's upper bound (1 - alpha1) uses the already-constrained alpha1, and
    # its Jacobian carries log(1 - alpha1):
    #   alpha0 ∈ (0, ∞)     : exp,             Jac += u_alpha0
    #   alpha1 ∈ (0, 1)      : logistic,        Jac += log α1 + log(1 − α1)
    #   beta1  ∈ (0, 1 − α1) : scaled logistic, Jac += log(1 − α1) + log w + log(1 − w)
    # The (0,1) Jacobian log α1 + log(1 − α1) is written straight from the
    # unconstrained value as −log1pexp(−u) − log1pexp(u) (no logistic→log round trip).
    alpha0::Float64 = exp(u_alpha0)
    alpha1::Float64 = logistic(u_alpha1)
    one_minus_alpha1::Float64 = 1.0 - alpha1
    beta1::Float64 = one_minus_alpha1 * logistic(u_beta1)

    jac_alpha0::Float64 = u_alpha0
    jac_alpha1::Float64 = -log1pexp(-u_alpha1) - log1pexp(u_alpha1)
    jac_beta1::Float64  = log(one_minus_alpha1) - log1pexp(-u_beta1) - log1pexp(u_beta1)
    log_jacobian::Float64 = jac_alpha0 + jac_alpha1 + jac_beta1

    # Constrained parameters as a NamedTuple, with inverse edges so it is also an
    # authoritative HAVE boundary (a deterministic query can start from the
    # constrained parameters, e.g. the one-step-ahead volatility forecast).
    parameters = (; mu, alpha0, alpha1, beta1)
    (mu::Float64, alpha0::Float64, alpha1::Float64, beta1::Float64) =
        (parameters.mu, parameters.alpha0, parameters.alpha1, parameters.beta1)

    # Sequential GARCH(1,1) conditional-sd recursion:
    #   σ₁ = sigma1 (data);  σₜ = sqrt(α0 + α1·(y_{t−1} − μ)² + β1·σ_{t−1}²)  (t ≥ 2).
    # Authored with `scan`: the carry is σ_{t−1} (seeded σ₁ = sigma1) and the
    # per-step input is y_{t−1} (the lagged series, derived in-graph). `scan`
    # lowers this recurrence to a stablehlo.while carry loop, so the whole density
    # compiles through Reactant off the natural recursion — a raw scalar-indexed
    # `for`/`σ[t-1]` loop over a traced array would not.
    T::Int = length(y)
    y_lag::Vector{Float64} = y[1:(T - 1)]                 # y_{t−1}, in-graph shape prep
    sigma_tail::Vector{Float64} =
        scan(y_lag, Ref(mu), Ref(alpha0), Ref(alpha1), Ref(beta1);
             init = sigma1) do sprev, yprev, m, a0, a1, b1
            st = sqrt(a0 + a1 * (yprev - m)^2 + b1 * sprev^2)
            (st, st)
        end
    sigma::Vector{Float64} = vcat(sigma1, sigma_tail)      # σ₁ .. σ_T

    # Likelihood y[t] ~ Normal(μ, σ[t]), t = 1..T (σ₁ = sigma1). One authored plate
    # zips (y, σ) with μ broadcast; the scalar Normal endpoint is reused.
    pointwise = plate(y, sigma, mu) do yt, st, m
        normal(m, st).logpdf(yt)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = likelihood           # no prior term
    posterior::Float64 = likelihood + log_jacobian

    # Deterministic one-step-ahead conditional sd σ_{T+1}, read off the last state.
    # Explicit `T` index (not `end`) so `forecast_sigma` also lowers through
    # Reactant — `end` on a traced result does not resolve in the traced program.
    forecast_sigma::Float64 =
        sqrt(alpha0 + alpha1 * (y[T] - mu)^2 + beta1 * sigma[T]^2)

    return posterior
end

q = [0.0, log(0.1), 0.0, 0.0]
y = GARCH11_Y
sigma1 = GARCH11_SIGMA1

requested_nodes = (:parameters, :log_jacobian, :sigma, :pointwise, :likelihood,
                   :posterior, :forecast_sigma)
# The iterated series `y` enters as a TRACED argument (only the scalar `sigma1`
# is bound), so the `scan` recursion lowers to a `stablehlo.while` carry loop
# under Reactant instead of unrolling over the fixed-length series.
density_kernel = prepare(model;
    have = (:unconstrained, :y, :sigma1),
    want = requested_nodes,
    bound = (; sigma1))

output = density_kernel(q, y)
parameters, log_jacobian, sigma, pointwise, likelihood, posterior, forecast_sigma =
    output
@assert likelihood ≈ sum(pointwise)
@assert posterior ≈ likelihood + log_jacobian
@assert isfinite(posterior)
@assert length(sigma) == length(y)
@assert sigma[1] == sigma1
@assert all(>(0.0), sigma)

docs_example = (;
    name = :garch11_density,
    origin = "posteriordb garch11 — Gaussian GARCH(1,1) with a sequential conditional-sd recursion",
    inputs = (; q, y),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_garch11_source(; model_only::Bool = false)
    # The source imports the reusable Normal endpoint itself and contains the
    # complete PPL assembly including the inline sd recursion. `model_only=true`
    # (used by `__init__`) stops as soon as `model` is defined — cheap load.
    _evaluate_ppl_source(GARCH11_SOURCE, @__MODULE__; bindings = (
        :GARCH11_Y, :GARCH11_SIGMA1,
    ), model_only)
end

# Evaluate the authored source from `__init__`, after package precompilation has
# closed the module. `build_garch11_graph` clones this runtime template so every
# caller gets an independent mutable graph without crossing a fresh `Core.eval`
# world-age boundary inside its own compiled function (a lazy build that
# `Core.eval`s the source and then `prepare`s in the same function body trips a
# "method too new" world-age error).
const _GARCH11_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _GARCH11_GRAPH_TEMPLATE[] = evaluate_garch11_source(; model_only = true).model
    nothing
end

"""
    build_garch11_graph()

Build the posteriordb `garch11` model as a declarative
`ReactiveKernels.KernelSpec`: a Gaussian GARCH(1,1) with mean `mu`. The
conditional standard deviations are computed by a sequential recursion authored
inline with the `scan` primitive (carry σ_{t−1}, per-step input the lagged
series), so the density lowers through Reactant off the natural recursion. The
support transforms + Jacobian (exp for `alpha0`, the (0,1) logistic for
`alpha1`, and the (0, 1−alpha1) scaled logistic for `beta1` — whose upper bound
depends on `alpha1`, matching Stan's declaration-order constraining), the
per-observation/summed likelihood, and the total density remain separate nodes;
`sigma`, the constrained `parameters`, and a one-step-ahead volatility forecast
`forecast_sigma` are selectable. There are no explicit priors (Stan's
improper-flat parametrization), so the density is likelihood + Jacobian.
"""
function build_garch11_graph()
    compose(_GARCH11_GRAPH_TEMPLATE[])
end

function demo()
    model = build_garch11_graph()
    q = [0.0, log(0.1), 0.0, 0.0]
    posterior_kernel = prepare(model;
        have = (:unconstrained, :y, :sigma1), want = :posterior,
        bound = (; sigma1 = GARCH11_SIGMA1))
    println("garch11 unconstrained log posterior = ", posterior_kernel(q, GARCH11_Y))
    nothing
end

end # module GARCH11Example

if abspath(PROGRAM_FILE) == @__FILE__
    GARCH11Example.demo()
end
