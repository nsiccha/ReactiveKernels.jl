# Declarative PPL kernel: grsm_latent_reg_irt (rating-scale ordinal IRT + latent regression)

This example ports the `grsm_latent_reg_irt` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`science_irt-grsm_latent_reg_irt`) into the declarative-`@kernel` style. It is
a rating-scale (GRSM) ordinal item-response model with a latent ability
regression. Authored on the FULL real data (`I = 7` items, `J = 392` persons,
`N = 2744` ordinal responses, `K = 1` covariate) loaded through
PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/grsm_latent_reg_irt.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/grsm_latent_reg_irt.jl).

```math
\begin{aligned}
\mathrm{unsummed} &= [0,\ \theta_{jj_n}\alpha_{ii_n} - \beta_{ii_n} - \kappa_1,\ \dots,\ \theta_{jj_n}\alpha_{ii_n} - \beta_{ii_n} - \kappa_m], \\
p(y_n) &= \operatorname{softmax}\!\big(\operatorname{cumsum}(\mathrm{unsummed})\big)\big|_{y_n + 1}, \quad m = \max(y), \\
\alpha_i &\sim \operatorname{LogNormal}(1, 1), \quad \beta_i \sim \operatorname{Normal}(0, 3), \quad \kappa_s \sim \operatorname{Normal}(0, 3), \\
\lambda_k &\sim \operatorname{Student\_t}(3, 0, 1), \quad \theta_j \sim \operatorname{Normal}\!\big((W_{\text{adj}}\lambda)_j,\ 1\big).
\end{aligned}
```

Both the item difficulties (`β = [β_free; −Σβ_free]`, `I − 1` free) and the
shared rating-scale steps (`κ = [κ_free; −Σκ_free]`, `m − 1` free) carry
Stan's global sum-to-zero constraint, applied in-graph through folded constant
design matrices — exactly the `append_row(x_free, −sum(x_free))` of the
reference model.

Raw long-form `ii`, `jj`, `y` (ordinal), the covariate matrix `W`, and the
required item count `I` are the data HAVEs. Every model-specific design is
derived **in-graph** from data bound at `prepare`:

- The shared category count `m = max(y)` (`m + 1` response categories for
  every observation — a rating scale has no ragged per-item structure).
- The covariate design `W_adj` (`obtain_adjustments`, the same verbatim Stan
  transcription as `gpcm_latent_reg_irt`, including the upstream
  operator-precedence quirk) and both sum-to-zero maps are folded by that
  `prepare`/`bound` specialization.

The per-observation rating-scale categorical is **data-generic**: a uniform
`N×(m+1)` logit matrix whose column `v` is
`v·(θ[jj]·α[ii] − β[ii]) − Σ_{s≤v} κ[s]` — the cumulative-sum form of Stan's
`rsm` function, with a numerically-stable row-wise `logsumexp` normalizer and
a column-major gather of the observed-response cell. The category count
adapts to the data, so there is no fixed-`m` unrolling. The unconstrained
vector is `(u_alpha[1..I], beta_free[1..I−1], kappa_free[1..m−1],
theta[1..J], lambda_adj[1..K])`, with `alpha = exp.(u_alpha)` and Jacobian
`sum(u_alpha)` — the only non-identity transform.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected plan,
and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :GrsmLatentRegIrtExample, :GRSM_LR_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_grsm_latent_reg_irt!,
)
```

## Reactant

On the focused gate's finite BridgeStan-valid probes, the exact authored graph
compiles and executes through the public Reactant boundary on the **all-bound
query** — data bound, only the unconstrained vector traced — with primal and
compiled-gradient parity to the native density and to the reference `.stan`
(machine precision; `propto = false`, `jacobian = true`). The
**traced-stream query** (raw data ports left free and traced) is an explicit
UNSUPPORTED boundary at this pin: integer data ports (`ii`, `jj`, `y`, `I`)
do not lower as traced streams (traced-boolean `TypeError`; complete
diagnostic retained by the gate). Per-axis support is recorded by the
structured gate's per-model accounting (`benchmark/structured_gate.jl`).

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.GrsmLatentRegIrtExample.demo()'
```
