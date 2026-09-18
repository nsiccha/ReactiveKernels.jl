# Declarative PPL kernel: 2pl_latent_reg_irt (2PL IRT + latent regression)

This example ports the `2pl_latent_reg_irt` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`fims_Aus_Jpn_irt-2pl_latent_reg_irt`, from Furr's edstan case studies) into the
declarative-`@kernel` style. It is a 2PL item-response model in which each
person's ability is regressed on person-level covariates. Authored on the FULL
real data (`I = 14` items, `J = 500` persons, `N = 7000` long-form responses,
`K = 4` covariates) loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/2pl_latent_reg_irt.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/2pl_latent_reg_irt.jl).

```math
\begin{aligned}
y_n &\sim \operatorname{Bernoulli\_logit}\!\big(\alpha_{ii_n}\,\theta_{jj_n} - \beta_{ii_n}\big), \\
\alpha_i &\sim \operatorname{LogNormal}(1, 1), \quad
\beta \sim \operatorname{Normal}(0, 3)\ \text{over all } I\ \text{difficulties}, \\
\lambda_k &\sim \operatorname{Student\_t}(3, 0, 1), \quad
\theta_j \sim \operatorname{Normal}\!\big((W_{\text{adj}}\lambda)_j,\ 1\big).
\end{aligned}
```

Raw long-form `ii`, `jj`, `y`, the covariate matrix `W`, and the required item
count `I` are the data HAVEs.
Two data-only designs are derived **in-graph** from the bound data, so partial
evaluation folds them:

- `W_adj` — Stan's `transformed data` covariate centering/scaling
  (`obtain_adjustments`) — is authored as a named recipe over the bound `W`. The
  upstream Stan has an operator-precedence quirk in its "column takes only two
  values" test, so the range branch is dead and the scale is always `2·sd` for
  `k ≥ 2`; this is transcribed verbatim for exact BridgeStan parity.
- The sum-to-zero difficulties `beta = [beta_free; −Σ beta_free]` are written as
  a folded constant design-matrix multiply `beta = S·beta_free` (a form that
  lowers cleanly under Reactant).

The unconstrained vector is `(u_alpha[1..I], beta_free[1..I−1], theta[1..J],
lambda_adj[1..K])`; `alpha = exp.(u_alpha)` is the only positive-constrained
block, with Jacobian `sum(u_alpha)`.
The likelihood reuses the direct logit-HAVE `Bernoulli` endpoint.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected plan,
and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :TwoplLatentRegIrtExample, :TWOPL_LR_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_2pl_latent_reg_irt!,
)
```

## Reactant

On the focused gate's finite BridgeStan-valid probes, the exact authored graph
compiles and executes through the public Reactant boundary with value and
gradient parity — primal and gradient both `@compile` and match native/the
reference `.stan` (via BridgeStan, `propto = false`, `jacobian = true`) to
machine precision. The folded covariate
design, the constant-matrix sum-to-zero map, the `W_adj·lambda` regression mean,
and the reused endpoints all lower cleanly.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.TwoplLatentRegIrtExample.demo()'
```
