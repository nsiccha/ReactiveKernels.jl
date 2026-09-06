# Declarative PPL kernel: correlated (MvNormal) regression

This example is a generalized-least-squares regression: the residuals are not
independent but share one fixed, correlated covariance. It is authored in the
same inline style as the [eight-schools example](eight-schools.md), and its
lesson is the multivariate-Normal likelihood's **authoritative HAVE
parametrizations** — the same modeled density is reachable through the
covariance, its Cholesky factor, the precision, or the precision Cholesky
factor, and the planner prunes whichever factorizations a query did not supply.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/mvnormal_regression.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/mvnormal_regression.jl).
It models a response vector `y` with a correlated Gaussian likelihood

```math
\begin{aligned}
\beta_k &\sim \operatorname{Normal}(0, 10), \\
y &\sim \operatorname{MvNormal}(X\beta,\ \Sigma),
\end{aligned}
```

where `X` is the design matrix and `Σ` is a fixed AR(1)-structured residual
covariance. The coefficients `β` are unconstrained, so there is no support
transform; the interest is entirely in the likelihood.

```text
unconstrained (β) ──► parameters ──► mean = X·β ─┐
                          │                       ├─► MvNormal.logpdf(y) ─► likelihood
                    prior plate (Normalₖ) ─► prior │      ▲   ▲   ▲   ▲
                          │                        covariance chol precision precision_chol
prior + likelihood ──► log density                 (one authoritative parametrization; the rest prune)
```

The `mean = X·β` linear predictor is a single node shared across every
parametrization, so it is computed once. The shared `mvnormal` distribution
object carries four producers for the log-determinant and quadratic form — one
per parametrization — and HAVE authority selects the one the query supplies.

The panel below shows three views of this model: **Raw input** (the exact
executed source, using the covariance parametrization), a readable **Generated
kernel**, and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :MVNormalRegressionExample, :MVNORMAL_REGRESSION_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_mvnormal_regression!,
)
```

The same density is reachable through any single parametrization. Each query
prunes the recipes and input ports of the three it does not use:

```julia
# Whichever ONE of these the query supplies is authoritative; the others are
# pruned from the plan and are not required inputs.
covariance_kernel = prepare(model;
    have = (:unconstrained, :predictors, :responses, :covariance),
    want = :density)

cholesky_kernel = prepare(model;
    have = (:unconstrained, :predictors, :responses, :chol),
    want = :density)

precision_kernel = prepare(model;
    have = (:unconstrained, :predictors, :responses, :precision),
    want = :density)
```

Supplying a covariance that is not positive definite fails the internal Cholesky
with a `PosDefException` rather than returning a silently wrong density.

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value parity, for **every** parametrization —
`test/test_ppl_examples_reactant.jl` `@compile`s the density kernel through the
covariance, Cholesky, precision, and precision-Cholesky HAVE routes and asserts
each compiled result matches its native evaluation. The `mvnormal` object's
Cholesky factorizations and triangular solves lower through XLA.

Run the walkthrough — it prints the density through three parametrizations — from
the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.MVNormalRegressionExample.demo()'
```
