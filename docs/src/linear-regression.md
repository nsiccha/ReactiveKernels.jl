# Declarative PPL kernel: linear regression

`ReactiveKernels` has no built-in probabilistic-programming semantics. Like the
[eight-schools example](eight-schools.md), this one assembles those semantics
manually from ordinary pure Julia recipes, while leaving the graph planner
responsible only for selecting the computation required by a particular
`have`/`want` query.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/linear_regression.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/linear_regression.jl).
It implements a simple Gaussian regression

```math
\begin{aligned}
\alpha &\sim \operatorname{Normal}(0, 10), \\
\beta &\sim \operatorname{Normal}(0, 10), \\
\sigma &\sim \operatorname{HalfNormal}(5), \\
y_i &\sim \operatorname{Normal}(\alpha + \beta x_i, \sigma).
\end{aligned}
```

The unconstrained vector is `(α, β, log_σ)`. Only `σ` needs a support transform,
so `σ = exp(log_σ)` and the optional log absolute Jacobian determinant is
`log_σ`.

```text
unconstrained
  ├─ split ──► α, β, log_σ ──► σ = exp(log_σ) ──► constrained parameters
  │                                   │
  │                                   ├─► log prior
  │                                   ├─► pointwise log likelihood ─► log likelihood
  │                                   └─► new-observation prediction
  └─ log_σ ──► log Jacobian

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

These remain separate named ports: pointwise terms are a first-class port, so
returning them together with the scalar density shares the likelihood
computation rather than repeating it.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected
plan, and the **Compute DAG** (`visualize(density_plan)`). The exact compiled AST
remains available as `code_expr(density_kernel)`.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :LinearRegressionExample, :LINEAR_REGRESSION_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_linear_regression!,
)
```

Asking only for constrained parameters selects just the split, positive-scale,
and assembly recipes; the Jacobian and every density recipe disappear:

```julia
constrain_kernel = prepare(model;
    have = :unconstrained,
    want = :parameters)
parameters = constrain_kernel(q)
```

Generated quantities can start at an already-constrained boundary. In this
query, planning removes the unconstrained transform, Jacobian, prior,
likelihood reduction, and total-density recipes:

```julia
generated_kernel = prepare(model;
    have = (:parameters, :new_predictor, :prediction_innovation),
    want = :prediction)

prediction = generated_kernel(parameters, 3.0, -1.0)
```

Prediction takes a standard-normal innovation as an input rather than drawing a
random number inside a recipe. That preserves the graph's purity: a caller can
draw a fresh innovation, replay a fixed one, or batch them using the same
prepared kernel without hiding an effect from the planner.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.LinearRegressionExample.demo()'
```
