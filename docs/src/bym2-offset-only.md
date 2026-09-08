# Declarative PPL kernel: BYM2 (spatial-Poisson convolution)

This example ports the `bym2_offset_only` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`traffic_accident_nyc-bym2_offset_only`) into the declarative-`@kernel` style. It
is the Morris/Riebler BYM2 spatial-Poisson model (Besag-York-Mollié 2) on an NYC
traffic-accident case study (`N = 1921` census areas, `N_edges = 5461`
adjacencies), authored on the FULL real data loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/bym2_offset_only.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/bym2_offset_only.jl).

```math
\begin{aligned}
\text{convolved\_re} &= \sqrt{1-\rho}\,\theta + \sqrt{\rho/\text{scaling\_factor}}\,\phi, \\
y_i &\sim \operatorname{Poisson\_log}\!\big(\log E_i + \beta_0 + \text{convolved\_re}_i\,\sigma\big), \\
\beta_0,\ \sigma &\sim \operatorname{Normal}(0, 1), \qquad \rho \sim \operatorname{Beta}(0.5, 0.5), \qquad \theta \sim \operatorname{Normal}(0, 1), \\
\text{target} &\mathrel{+}= -\tfrac{1}{2}\textstyle\sum (\phi_{\text{node1}} - \phi_{\text{node2}})^2, \qquad \textstyle\sum\phi \sim \operatorname{Normal}(0,\ 0.001 N).
\end{aligned}
```

The graph binds only the RAW `node1`/`node2` adjacency, `y`, `E`, and
`scaling_factor` ports. The transformed-data offset `log_E = log(E)` is authored
as an in-graph node over the bound exposure (it folds under partial evaluation);
the ICAR pairwise-difference prior gathers the spatial effect through
`phi[node1]`/`phi[node2]`; the soft sum-to-zero constraint uses `sum(phi)`.
`sigma > 0` (log) and `rho ∈ [0, 1]` (logit) carry their exact transform
Jacobians.

```text
unconstrained ──► beta0, u_sigma, u_rho, theta, phi ──► sigma, rho ──► constrained parameters
E (raw) ──► log_E (in-graph)                                          │
node1,node2 (raw) ──► phi[node1]-phi[node2] ──► ICAR prior           ├─► log prior
theta, phi, rho ──► convolved_re ─┐                                  │
log_E + beta0 + convolved_re·sigma = eta ─► pointwise Poisson_log ───┴─► log likelihood
y (raw) ──────────────────────────┘         u_sigma, u_rho ─► log Jacobian

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :Bym2OffsetOnlyExample, :BYM2_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_bym2_offset_only!,
)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/batch_latent_gate.jl`
`@compile`s both the primal and the gradient and asserts they match the native
evaluation and the reference `.stan` (via BridgeStan, `propto = false`,
`jacobian = true`). The bound-index ICAR gathers and the fused `eta` lower
cleanly.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.Bym2OffsetOnlyExample.demo()'
```
