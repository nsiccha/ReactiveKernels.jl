# Declarative PPL kernel: gpcm_latent_reg_irt (GPCM ordinal IRT + latent regression)

This example ports the `gpcm_latent_reg_irt` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`timssAusTwn_irt-gpcm_latent_reg_irt`, from Furr's edstan case studies) into the
declarative-`@kernel` style. It is a Generalized Partial Credit ordinal
item-response model with a latent ability regression. Authored on the FULL real
data (`I = 11` items, `J = 500` persons, `N = 5500` ordinal responses, `K = 5`
covariates) loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/gpcm_latent_reg_irt.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/gpcm_latent_reg_irt.jl).

```math
\begin{aligned}
L_{n,v} &= v\,(\theta_{jj_n}\alpha_{ii_n}) - \sum_{s\le v}\beta_{\text{seg}(ii_n),\,s}, \quad
p(y_n) = \operatorname{softmax}_v(L_{n,v})\big|_{v = y_n}, \\
\alpha_i &\sim \operatorname{LogNormal}(1, 1), \quad \beta \sim \operatorname{Normal}(0, 3)\ \text{over all } \textstyle\sum_i m_i\ \text{steps}, \\
\lambda_k &\sim \operatorname{Student\_t}(3, 0, 1), \quad \theta_j \sim \operatorname{Normal}\!\big((W_{\text{adj}}\lambda)_j,\ 1\big).
\end{aligned}
```

Raw long-form `ii`, `jj`, `y` (ordinal), the covariate matrix `W`, and the
required item count `I` are the data HAVEs. Every model-specific design is
derived **in-graph** from data bound at `prepare`:

- The ragged per-item category structure — `m[i]` (each item's max category),
  the segment positions, and the category-validity masks — is derived from the
  bound `y`/`ii`.
- The covariate design `W_adj` (`obtain_adjustments`, same verbatim Stan
  transcription as `2pl_latent_reg_irt`) and the sum-to-zero step-difficulty map
  are folded by that `prepare`/`bound` specialization.

The per-observation GPCM categorical is **data-generic**: an `N×(M+1)` logit
matrix (`M = max category`, read from the data when `prepare` binds it) built from a
concrete-index **matrix gather** `beta[POSIDX]` plus a constant cumulative-sum
matmul, with invalid categories masked to `−Inf` and a numerically-stable
row-wise `logsumexp` normalizer. The category count adapts to the data, so there
is no fixed-`K` unrolling. The unconstrained vector is `(u_alpha[1..I],
beta_free[1..Σm−1], theta[1..J], lambda_adj[1..K])`, with
`alpha = exp.(u_alpha)` and Jacobian `sum(u_alpha)`.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected plan,
and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :GpcmLatentRegIrtExample, :GPCM_LR_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_gpcm_latent_reg_irt!,
)
```

## Reactant

On the focused gate's finite BridgeStan-valid probes, the exact authored graph
compiles and executes through the public Reactant boundary with value and
gradient parity — primal and gradient both `@compile` and match native/the
reference `.stan` (via BridgeStan, `propto = false`, `jacobian = true`) to
machine precision. The concrete-index
matrix gather, the cumulative-sum matmul, the masked row-wise `logsumexp`, and
the reused `LogNormal`/`Normal`/`Student-t` endpoints all lower cleanly.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.GpcmLatentRegIrtExample.demo()'
```
