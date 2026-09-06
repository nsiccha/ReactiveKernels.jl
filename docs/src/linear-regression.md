# Declarative PPL kernel: linear regression

`ReactiveKernels` has no built-in probabilistic-programming semantics. Like the
[eight-schools example](eight-schools.md), this one assembles those semantics
inside a single authored model kernel: the transform, priors, and likelihood are
written out directly, the Normal density is reused from the shared distribution
objects, and the graph planner is left responsible only for selecting the
computation a particular `have`/`want` query needs.

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
`log_σ`. The prior on `σ` reuses the shared `normal` endpoint folded with the
`log(2)` half-normal truncation constant rather than re-authoring a scale
density.

```text
unconstrained ──► α, β, log_σ ──► σ = exp(log_σ) ──► parameters (α, β, σ)
  │                                   │
  │                                   ├─► log prior  (Normal + Normal + HalfNormal)
  │                                   ├─► pointwise plate ─► log likelihood
  │                                   └─► new-observation prediction
  └─ log_σ ──► log Jacobian

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The likelihood is one authored `plate`. Its pointwise terms are a first-class
port, so a query that asks for the summed density alone fuses the reduction with
no output buffer, while a query that also asks for the pointwise vector shares
the same traversal instead of repeating it.

The panel below shows three views of this model: **Raw input** (the exact
executed source), a readable **Generated kernel** derived from the executed
kernel and selected plan, and the **Compute DAG** (`visualize(density_plan)`).
The exact compiled AST remains available as `code_expr(density_kernel)`.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :LinearRegressionExample, :LINEAR_REGRESSION_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_linear_regression!,
)
```

Asking only for constrained parameters selects just the packed-scalar extraction
and the positive-scale transform; the Jacobian and every density recipe
disappear:

```julia
constrain_kernel = prepare(model;
    have = :unconstrained,
    want = :parameters)
parameters = constrain_kernel(q)
```

Generated quantities can start at an already-constrained boundary. In this
query, planning removes the unconstrained transform, Jacobian, prior, likelihood
reduction, and total-density recipes, leaving only the prediction arithmetic:

```julia
generated_kernel = prepare(model;
    have = (:parameters, :new_predictor, :prediction_innovation),
    want = :prediction)

prediction = generated_kernel(parameters, 3.0, -1.0)
```

The constrained parameters and the prediction are plain `NamedTuple`s, not custom
types. Prediction takes a standard-normal innovation as an input rather than
drawing a random number inside a recipe. That preserves the graph's purity: a
caller can draw a fresh innovation, replay a fixed one, or batch them using the
same prepared kernel without hiding an effect from the planner.

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with no model-specific lowering path. `test/test_linear_regression_reactant.jl`
converts the packed unconstrained boundary to Reactant arrays, `@compile`s the
prepared density kernel, and asserts the compiled result matches the native
evaluation. The small static likelihood plate lowers as per-lane scalar recipes
with a scalar reduction — there is no opaque or externalized operation in the
compiled module.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.LinearRegressionExample.demo()'
```
