# Declarative PPL kernel: partial evaluation of a data-only prefix

Many Bayesian workflows preprocess the data once — standardizing predictors,
forming sufficient statistics, factorizing a fixed covariance — and then evaluate
the log density thousands of times inside a sampler. When that preprocessing is
written *inside* the model, it depends only on the data, so it should run once at
preparation rather than on every density call. ReactiveKernels does exactly that
through the public `bound` keyword: binding a data port at preparation runs the
data-only subgraph once and hoists its result into the residual kernel as a
constant.

This example is a standardized linear regression. The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/bound_regression.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/bound_regression.jl):

```math
\begin{aligned}
\alpha &\sim \operatorname{Normal}(0, 10),\quad \beta_j \sim \operatorname{Normal}(0, 5),\quad \sigma \sim \operatorname{HalfNormal}(5), \\
z &= \operatorname{standardize}(X),\qquad y_i \sim \operatorname{Normal}(\alpha + (z\beta)_i,\ \sigma).
\end{aligned}
```

The predictors `X` are standardized (centered and scaled by their column
statistics) inside the model, but `z = standardize(X)` reads only `raw_predictors`
— a data-only prefix. The unconstrained vector is `(α, β₁, β₂, log_σ)`; only `σ`
has a support transform.

The panel below shows the full model prepared normally, so the standardization is
part of the displayed kernel:

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :BoundRegressionExample, :BOUND_REGRESSION_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_bound_regression!,
)
```

## Hoisting the data-only prefix with `bound`

Binding `raw_predictors` at preparation runs the standardization once and hoists
the standardized design matrix into the prepared kernel as a constant. The
residual kernel then takes only the unconstrained vector and the responses:

```julia
bound_kernel = prepare(model;
    have = (:unconstrained, :raw_predictors, :responses),
    want = :density,
    bound = (; raw_predictors))

# `raw_predictors` has left the runtime signature; it is no longer passed.
bound_kernel(q, responses)          # == the unbound kernel's density
```

The optimization is not cosmetic. The unbound kernel recomputes the standardized
design matrix on every call; the bound kernel does not, so its residual call
allocates strictly less. `bound` is a general Plan → Plan partial-evaluation
pre-pass — the same mechanism `prepare`, `prepare_ad`, and the Reactant boundary
all consume — not a regression-specific shortcut: any subgraph that reads only
bound ports is hoisted, whatever the model.

## Reactant

Both the plain and the `bound` kernels compile and execute through the public
Reactant boundary with value parity — `test/test_ppl_examples_reactant.jl`
`@compile`s each and asserts the compiled result matches native. The bound
kernel's hoisted standardized design matrix re-enters the compiled program as a
constant, exactly as it does natively.

Run the walkthrough — it prints the plain and bound kernels' input signatures and
densities — from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.BoundRegressionExample.demo()'
```
