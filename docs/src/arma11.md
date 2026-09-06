# Declarative PPL kernel: ARMA(1, 1) time series

This example ports the `arma11` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior `arma-arma11`)
into the same declarative-`@kernel` style as the
[eight-schools example](eight-schools.md): the model is authored inline in one
source string, the Normal and Cauchy endpoints are reused from the shared
distribution objects (the `HalfCauchy(2.5)` prior on `σ` folds the Cauchy
endpoint with the `log(2)` truncation constant), and the constrained parameters
are a plain NamedTuple. Its distinctive structure is a **sequential recursion**:
the latent one-step-ahead errors are computed by walking the series in order, and
that stateful computation is authored inline in the log density as one named
graph node — irreducibly sequential, so it is not a plate. The example also
carries the *vectorized closed form* of that recursion (`errors_closed`) as a
second node, which the density reduces so the model lowers through Reactant (see
the Reactant section below).

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
unconstrained ──► μ, φ, θ, log_σ ──► σ ──► constrained parameters
   │                                              │
series ──┬─► errors (sequential recursion) ──────► one-step-ahead forecast
         │        (native reference; parity-checked against errors_closed)
         └─► errors_closed (vectorized Toeplitz) ─► pointwise ─► likelihood
                   (lowers through Reactant)

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The latent errors are a **first-class named port** in two forms: the sequential
`errors` (native reference; the forecast reruns it) and the vectorized
`errors_closed` (the density's error source, which lowers). A query can ask for
either, the full `density`, or the one-step-ahead `forecast`; a test asserts
`errors_closed ≈ errors`.

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

## Reactant

This model shows a **side-by-side**: the natural sequential recursion and its
vectorized closed-form equivalent. The recursion (`errors`) reads `series[t-1]`
and `err[t-1]` element by element, and XLA disallows scalar indexing of a traced
array, so that node does **not** lower — `@compile` of the raw `errors` throws.

`errors_closed` is the exact vectorized equivalent (a test asserts
`errors_closed ≈ errors`): the linear recurrence `εₜ = aₜ − θ·ε_{t-1}` has the
closed form `ε = L·a` with `L` a lower-triangular Toeplitz of powers of `−θ`,
built from only vectorized ops (slice, `vcat`, broadcast, `matmul`). The
likelihood and density reduce `errors_closed`, so the **whole density compiles
and executes through the public Reactant boundary with value parity** —
`test/test_ppl_examples_reactant.jl` `@compile`s it and asserts the compiled
result matches native, and separately asserts the raw sequential node still
throws.

A sequential-scan (`stablehlo.while`) lowering would let the *natural* recursion
lower directly, with no reformulation; until such a lowering exists on the
declarative `@kernel` surface, the vectorized closed form is arma11's Reactant
path.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.ARMA11Example.demo()'
```
