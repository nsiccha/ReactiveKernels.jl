# Declarative PPL kernel: ARMA(1, 1) time series

```@eval
Main.ReactiveKernelsDocs.render_review_status(:frozen_ppl)
```

This example ports the `arma11` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior `arma-arma11`)
into the same declarative-`@kernel` style as the
[eight-schools example](eight-schools.md). Its distinctive structure is a
**sequential recursion**: the latent one-step-ahead errors are computed by
walking the series in order, and that stateful computation lives inside the log
density.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/arma11.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/arma11.jl).
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

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected
plan, and the **Compute DAG** (`visualize(density_plan)`). The exact compiled AST
remains available as `code_expr(density_kernel)`.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :ARMA11Example, :ARMA11_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_arma11!,
)
```

Because the errors are their own port, asking only for them prunes every density
recipe — just the transforms and the recursion run:

```julia
errors_kernel = prepare(model;
    have = (:unconstrained, :series),
    want = :errors)
errors = errors_kernel(q, series)
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
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.ARMA11Example.demo()'
```
