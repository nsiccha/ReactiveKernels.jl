# Declarative PPL kernel: GLMM1 (per-site Poisson-log GLMM)

This example ports the `GLMM1_model` from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`GLMM_data-GLMM1_model`) into the declarative-`@kernel` style. It is a
hierarchical Poisson-log GLMM (BPA ch. 6, Kéry & Schaub) with `nsite = 235`
per-site random effects and `nobs = 2072` observed counts, authored on the FULL
real data loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/glmm1_model.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/glmm1_model.jl).

```math
\begin{aligned}
\alpha_j &\sim \operatorname{Normal}(\mu_\alpha,\; \sigma_\alpha), \quad j = 1..\text{nsite}, \\
\mu_\alpha &\sim \operatorname{Normal}(0, 10), \qquad \sigma_\alpha \in [0, 5]\ \text{(implicit uniform)}, \\
\text{obs}_i &\sim \operatorname{Poisson\_log}\!\big(\log\lambda_{\text{obsyear}_i,\, \text{obssite}_i}\big).
\end{aligned}
```

Stan sets `log_lambda = rep_matrix(alpha', nyear)`, so every year-row equals
`alpha'`. `obsyear_i` occurs in the likelihood when it selects
`log_lambda[obsyear_i, obssite_i]`, but that value equals `alpha[obssite_i]`
for ANY year, so the year dependence cancels algebraically; the other missingness
inputs feed generated quantities. The faithful graph therefore binds only the RAW
`obs` and `obssite` ports and gathers the site effect `alpha[obssite]` in-graph —
no materialized `nyear × nsite` broadcast. Only `sigma_alpha` needs a support
transform (scaled logit onto `[0, 5]`, with its interval Jacobian and NO prior
density term).

```text
unconstrained ──► alpha, mu_alpha, u_sigma ──► sigma_alpha ──► constrained parameters
obssite (raw) ─────────────────► alpha[obssite] = log-rate ─► pointwise Poisson_log ─► log likelihood
obs (raw) ─────────────────────────────────────────────────┘
             alpha, mu_alpha, sigma_alpha ─► log prior;   u_sigma ─► log Jacobian

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :GLMM1ModelExample, :GLMM1_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_glmm1!,
)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/batch_latent_gate.jl`
`@compile`s both the primal and the gradient and asserts they match the native
evaluation and the reference `.stan` (via BridgeStan, `propto = false`,
`jacobian = true`). The bound-index site-effect gather lowers cleanly.
Evidence is bounded to six reference-finite native points (Reactant uses their
first point) under BridgeStan 2.9 / Stan 2.39; finite-point parity is not an
all-input proof.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.GLMM1ModelExample.demo()'
```
