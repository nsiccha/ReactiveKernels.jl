# Declarative PPL kernel: hier_2pl (hierarchical 2PL IRT)

This example ports the `hier_2pl` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior `sat-hier_2pl`,
the mc-stan `hierarchical_2pl` case study) into the declarative-`@kernel` style.
It is a hierarchical 2PL item-response model whose item parameters (log
discrimination and difficulty) are drawn from a correlated bivariate Normal.
Authored on the FULL real data (`I = 32` items, `J = 600` persons, `N = 19200`
long-form responses) loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/hier_2pl.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/hier_2pl.jl).

```math
\begin{aligned}
y_n &\sim \operatorname{Bernoulli\_logit}\!\big(\alpha_{ii_n}(\theta_{jj_n} - \beta_{ii_n})\big),
  \quad \alpha_i = e^{\xi_{1i}},\ \beta_i = \xi_{2i}, \\
\xi_i &\sim \operatorname{MultiNormalCholesky}\!\big(\mu,\ \operatorname{diag}(\tau)\,L_\Omega\big),
  \quad L_\Omega \sim \operatorname{LKJCholesky}(4), \\
\theta_j &\sim \operatorname{Normal}(0, 1), \quad
\mu_1 \sim \operatorname{Normal}(0,1),\ \mu_2 \sim \operatorname{Normal}(0,5),\ \tau_k \sim \operatorname{Exponential}(0.1).
\end{aligned}
```

Raw long-form `ii`, `jj`, `y` (plus the item/person counts) are the data HAVEs.
The correlation is a `cholesky_factor_corr[2]` reconstructed from a single
unconstrained value `w`: `L21 = tanh(w)`, `L22 = sqrt(1 − L21²)`, with the exact
Jacobian `log(1 − L21²)`. The `LKJ(4)` Cholesky density for `K = 2` is the
analytic `(2η−2)·log(L22) − logB(1/2, 4)`, and the normalizer `logB(1/2, 4)`
collapses to the exact base-`log` constant `log(6/6.5625)` (the `√π` from
`Γ(1/2)` cancels `Γ(4.5) = 6.5625·√π`). The hierarchical item prior
`ξ_i ~ MultiNormalCholesky(μ, diag(τ)·L_Ω)` is authored as a fused bivariate
2-D quadratic-form plate over the item axis. The unconstrained vector is
`(theta[1..J], xi1[1..I], xi2[1..I], mu[2], u_tau[1..2], w)`;
`tau = exp.(u_tau)`, and only those exp coordinates and `w` (the correlation)
carry Jacobians.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected plan,
and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :Hier2plExample, :HIER_2PL_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_hier_2pl!,
)
```

## Reactant

On the focused gate's finite BridgeStan-valid probes, the exact authored graph
compiles and executes through the public Reactant boundary with value and
gradient parity — primal and gradient both `@compile` and match native/the
reference `.stan` (via BridgeStan, `propto = false`, `jacobian = true`) to
machine precision. The `tanh`
correlation reconstruction, the fused bivariate `multi_normal_cholesky` plate,
and the reused `Normal`/`Exponential`/`Bernoulli` endpoints all lower cleanly.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.Hier2plExample.demo()'
```
