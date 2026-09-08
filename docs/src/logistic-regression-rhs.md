# Declarative PPL kernel: logistic regression (regularized horseshoe)

This example ports the `logistic_regression_rhs` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`ovarian-logistic_regression_rhs`) into the declarative-`@kernel` style. It is
Bayesian logistic regression with the REGULARIZED HORSESHOE (Finnish horseshoe)
prior of Piironen & Vehtari (2017), authored on the FULL real `ovarian`
microarray data (n = 54 samples, d = 1536 genes) loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/logistic_regression_rhs.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/logistic_regression_rhs.jl).
The regularized horseshoe puts a global–local scale hierarchy on the `d`
coefficients and shrinks the largest toward a slab of width `c`:

```math
\begin{aligned}
c &= \text{slab\_scale}\,\sqrt{\text{caux}}, \qquad
\tilde\lambda = \sqrt{\frac{c^2\,\lambda^2}{c^2 + \tau^2\,\lambda^2}}, \qquad
\beta = z \odot \tilde\lambda \cdot \tau, \\
y_i &\sim \operatorname{Bernoulli\_logit}(\beta_0 + x_i \cdot \beta).
\end{aligned}
```

Priors (`nu_global = nu_local = 1` for `ovarian`, so the half-t's are
half-Cauchy): `z ~ Normal(0,1)`, `λ ~ Student_t(nu_local, 0, 1)` truncated `> 0`,
`τ ~ Student_t(nu_global, 0, scale_global·2)` truncated `> 0`,
`caux ~ Inverse_Gamma(slab_df/2, slab_df/2)`, `β_0 ~ Normal(0, scale_icept)`.
The half-t priors carry no explicit `student_t_lccdf` normalization here (Stan
drops it), so the truncation is applied by the exp-transform Jacobian alone.

The unconstrained vector is `(β_0, z[1..d], log_τ, log_λ[1..d], log_caux)`. The
three positive scales use the `exp` support transform (Jacobians `log_τ`,
`Σ log_λ`, `log_caux`); `β_0` and `z` are identity. The design matrix `x`, the
0/1 outcomes `y`, and the six prior hyper-scalars are bound data ports, so the
same graph serves the sibling `prostate-logistic_regression_rhs` (d = 5966) by
binding its data. The likelihood uses the natural logit HAVE route
(`bernoulli(; logit = f)`), so no `logistic → logit` round trip enters the kernel.

```text
unconstrained ──► β0, z, log_τ, log_λ, log_caux ──► τ, λ, caux ──► c, λ̃, β
  │                                                  │          └─► f = β0 + x·β ─► Bernoulli-logit plate ─► log likelihood
  │                                                  └─► log prior (z, λ, τ, caux, β0)
  └─ (log_τ, log_λ, log_caux) ──► log Jacobian
```

The panel below shows **Raw input** (the source), the **Generated kernel**, and
the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :LogisticRegressionRHSExample, :LOGISTIC_RHS_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_logistic_regression_rhs!,
)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/batch1_gate.jl` `@compile`s
both the primal and the gradient and asserts they match the native evaluation and
the reference `.stan`. The horseshoe transforms (`lambda_tilde`, the `exp`
scales), the `d = 1536` linear predictor, and the logit-route Bernoulli
likelihood all lower cleanly.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.LogisticRegressionRHSExample.demo()'
```
