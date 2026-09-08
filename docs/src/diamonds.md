# Declarative PPL kernel: diamonds (brms centered regression)

This example ports the `diamonds` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`diamonds-diamonds`) into the declarative-`@kernel` style. It is a brms 2.10.0
Gaussian linear regression of log-price on 24 population-level effects (the
`diamonds` dataset from ggplot2), authored on the FULL real data (N = 5000,
K = 25) loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/diamonds.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/diamonds.jl).
brms uses a **centered** parameterization:

```math
\begin{aligned}
Y_i &\sim \operatorname{Normal}(\text{Intercept} + X^c_i \cdot b,\; \sigma), \\
b_j &\sim \operatorname{Normal}(0, 1), \quad j = 1..24, \\
\text{Intercept} &\sim \operatorname{Student\_t}(3, 8, 10), \\
\sigma &\sim \operatorname{Student\_t}(3, 0, 10)\ \text{truncated to } \sigma > 0.
\end{aligned}
```

The design matrix is centered in Stan's `transformed data`: it drops the
all-ones intercept column and subtracts each remaining column's mean, giving
`Xc`. That step reads only `X`, so it is authored as a **data-only prefix** and
hoisted by binding the raw design-matrix port (`bound = (; X, prior_only)`) — RK
runs the column-drop and centering once at preparation and folds `Xc` into the
residual kernel as a constant. The likelihood is added only when the data flag
`prior_only` is 0 (`ifelse(prior_only == 0, likelihood, 0.0)`), a data-directed
branch that the bound `prior_only` port folds to a constant.

The unconstrained vector is `(b[1..24], Intercept, log_sigma)`. Only `sigma`
needs a support transform (`sigma = exp(log_sigma)`, Jacobian `log_sigma`); the
half-Student-t on `sigma` carries the explicit brms normalization
`-student_t_lccdf(0 | 3, 0, 10) = log 2`. The population intercept is reported as
a generated quantity `b_Intercept = Intercept - dot(means_X, b)`.

```text
unconstrained ──► b, Intercept, log_sigma ──► sigma ──► constrained parameters
X ──► Xc, means_X (data-only prefix, hoisted)          │
  │                                                    ├─► log prior
  │                        Intercept + Xc·b = μ ───────┼─► pointwise log likelihood ─► log likelihood
  │                                                    └─► b_Intercept (generated)
  └─ log_sigma ──► log Jacobian

log prior + log Jacobian + (prior_only ? 0 : log likelihood) ──► unconstrained log density
```

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected plan,
and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :DiamondsExample, :DIAMONDS_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_diamonds!,
)
```

The generated-quantity population intercept starts from an already-constrained
boundary, so planning removes the transforms, Jacobian, prior, and likelihood:

```julia
gq_kernel = prepare(model;
    have = (:unconstrained, :X),
    want = :b_Intercept,
    bound = (; X = DIAMONDS_X))
b_Intercept = gq_kernel(q)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/batch1_gate.jl` `@compile`s
both the primal and the gradient and asserts they match the native evaluation
and the reference `.stan` (via BridgeStan). The centering prefix, the fused
`normal_id_glm` mean, and the reused `normal`/`student_t` endpoints all lower
cleanly.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.DiamondsExample.demo()'
```
