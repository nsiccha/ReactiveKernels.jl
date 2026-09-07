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
graph node — irreducibly sequential, so it is not a plate. It is authored with
the [`scan` primitive](scan.md), which threads a carry through the series and
lowers the recurrence to a `stablehlo.while` loop, so the natural sequential form
**lowers through Reactant directly** (see the Reactant section below). The example
also carries the *vectorized closed form* of that recursion (`errors_closed`) as
a second node, kept as an independent numerical cross-check.

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
series ──┬─► errors (scan recursion) ──┬─────────► one-step-ahead forecast
         │   (lowers via stablehlo.while)└─► pointwise ─► likelihood
         └─► errors_closed (vectorized Toeplitz, independent cross-check)

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The latent errors are a **first-class named port**: the sequential `errors`,
authored with [`scan`](scan.md), which the density reduces and which the forecast
reruns. A query can ask for just the errors, the full `posterior`, or the
one-step-ahead `forecast`. The vectorized `errors_closed` is kept as an
independent numerical cross-check; a test asserts `errors ≈ errors_closed`.

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

The natural sequential recursion **lowers through Reactant directly**. `errors`
is authored with the [`scan` primitive](scan.md): the carry threads
`(y_{t-1}, ε_{t-1})`, seeded `(μ, 0)`, and each step computes
`νₜ = μ + φ·carry.y_prev + θ·carry.err_prev` and `εₜ = yₜ − νₜ`. `scan` lowers
this to a `stablehlo.while` carry loop — one traced loop, no unrolling and no
per-step scalar indexing of a traced array — so the likelihood and density reduce
the natural `errors` and the **whole density compiles and executes through the
public Reactant boundary with value parity**.
`test/test_ppl_examples_reactant.jl` `@compile`s the density and asserts the
compiled result matches native; it also compiles the standalone `errors` kernel,
asserts its MLIR contains `stablehlo.while` (proving it did not unroll), and
matches native.

`errors_closed` is an independent numerical cross-check (a test asserts
`errors ≈ errors_closed`): the linear recurrence `εₜ = aₜ − θ·ε_{t-1}` has the
closed form `ε = L·a` with `L` a lower-triangular Toeplitz of powers of `−θ`,
built from only vectorized ops (slice, `vcat`, broadcast, `matmul`), and it also
lowers through Reactant. Before `scan` existed this closed form was arma11's only
Reactant path, because the raw `for`/`err[t-1]` recursion could not lower; it is
kept as a second, structurally different derivation of the same errors.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.ARMA11Example.demo()'
```
