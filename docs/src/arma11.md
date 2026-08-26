# Declarative PPL kernel: ARMA(1, 1) time series

This example ports the `arma11` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior `arma-arma11`)
into the same declarative-`@kernel` style as the
[eight-schools example](eight-schools.md). Its distinctive structure is a
**sequential recursion**: the latent one-step-ahead errors are computed by
walking the series in order, and that stateful computation lives inside the log
density.

The complete runnable source is
[`examples/arma11.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/arma11.jl).
It models a scalar series `y₁,…,y_T`:

```math
\begin{aligned}
\nu_1 &= \mu + \phi\mu, \qquad \nu_t = \mu + \phi\, y_{t-1} + \theta\, \varepsilon_{t-1}, \\
\varepsilon_t &= y_t - \nu_t, \qquad \varepsilon_t \sim \operatorname{Normal}(0, \sigma), \\
\mu &\sim \operatorname{Normal}(0, 10),\quad \phi, \theta \sim \operatorname{Normal}(0, 2),\quad \sigma \sim \operatorname{HalfCauchy}(2.5).
\end{aligned}
```

The unconstrained vector is `(μ, φ, θ, log_σ)`; only `σ` needs a support
transform, so `σ = exp(log_σ)` and the optional log absolute Jacobian
determinant is `log_σ`.

```text
unconstrained ──► split ──► μ, φ, θ, log_σ ──► σ ──► constrained parameters
   │                                    │              │
series ─────────────────────────────────┴─► errors (recursion) ──► pointwise ─► likelihood
   │                                                   │
   └───────────────────────────────► one-step-ahead forecast

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The latent errors are a **first-class named port**: a query can ask for just the
`errors`, the full `density`, or the one-step-ahead `forecast`, and the planner
computes the recursion once and shares it.

The panel below is one coherent, build-executed artifact. **Raw input** is the
exact source that builds and runs the query, **Generated kernel** is
`code_expr(density_kernel)` from that execution, and **Compute DAG** is the live
colored `visualize(density_plan)` component.

```@eval
Main.ReactiveKernelsDocs.execute_example(@__MODULE__, raw"""
@kernel model(unconstrained::UnconstrainedParameters,
              series::RealVector) = begin
    (μ::Real, φ::Real, θ::Real, log_σ::Real) =
        ARMA11Example.split_unconstrained(unconstrained)
    σ::Real = ARMA11Example.positive_scale(log_σ)
    parameters::ARMAParameters = ARMA11Example.assemble_parameters(μ, φ, θ, σ)
    log_jacobian::Real = ARMA11Example.log_abs_det_jacobian(log_σ)

    errors::RealVector = ARMA11Example.arma_errors(parameters, series)
    prior::Real = ARMA11Example.log_prior(parameters)
    pointwise::RealVector = ARMA11Example.pointwise_log_likelihood(errors, parameters)
    likelihood::Real = ARMA11Example.sum_log_likelihood(pointwise)
    density::Real = ARMA11Example.total_log_density(
        prior, log_jacobian, likelihood,
    )
    forecast::Real = ARMA11Example.one_step_forecast(parameters, series, errors)
    return density
end

q = (0.0, 0.9, -0.2, log(0.15))
series = ARMA_SERIES

density_kernel = prepare(model;
    have = (:unconstrained, :series),
    want = (:prior, :log_jacobian, :pointwise, :likelihood, :density))

output = density_kernel(q, series)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :arma11_density,
    origin = "compact @kernel model (build executed) — posteriordb arma11",
    inputs = (; q, series),
    kernel = density_kernel,
    output,
)
"""; setup = Main.ReactiveKernelsDocs.setup_arma11!)
```

Because the errors are their own port, asking only for them prunes every density
recipe — just the transforms and the recursion run:

```julia
errors_kernel = prepare(model;
    have = (:unconstrained, :series),
    want = :errors)
errors = errors_kernel(q, series)
```

The numeric graph ports are typed at a `Real` boundary, so the prepared density
kernel differentiates cleanly through reverse-mode AD
(`DifferentiationInterface` with the Enzyme backend) — the error recursion
included:

```julia
using DifferentiationInterface
import Enzyme

enzyme_backend = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)

density_only = prepare(model;
    have = (:unconstrained, :series),
    want = :density)
logdensity(qv) = density_only(Tuple(qv), series)
gradient = DifferentiationInterface.gradient(logdensity, enzyme_backend, collect(q))
```

The one-step-ahead forecast is a deterministic generated quantity; planning it
from a constrained boundary keeps the recursion (the forecast needs the last
error) but drops the prior, likelihood, and total-density recipes:

```julia
forecast_kernel = prepare(model;
    have = (:parameters, :series),
    want = :forecast)
forecast = forecast_kernel(parameters, series)
```

Run the walkthrough from the repository root:

```sh
julia --project=. examples/arma11.jl
```
