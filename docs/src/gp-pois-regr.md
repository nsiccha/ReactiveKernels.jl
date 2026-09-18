# Declarative PPL kernel: latent Gaussian-process Poisson regression

This is the posteriordb `gp_pois_regr` model: a Poisson regression with a
one-dimensional **non-centered latent** Gaussian process (exponential-quadratic
covariance). It is the counterpart to the [marginal GP regression](gp-regr.md) —
here the latent field is kept explicit and pushed through a Poisson-log
likelihood. Full real data (`N = 11`) is loaded through `PosteriorDB.jl`.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/gp_pois_regr.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/gp_pois_regr.jl).
It models the counts `k` as

```math
\begin{aligned}
\rho &\sim \operatorname{Gamma}(25, 4), \quad
\alpha \sim \operatorname{Normal}^{+}(0, 2), \quad
\tilde f \sim \operatorname{Normal}(0, 1), \\
f &= \operatorname{chol}\!\Big(\alpha^2 \exp\!\big(-\tfrac{(x_i-x_j)^2}{2\rho^2}\big) + 10^{-10} I\Big)\, \tilde f, \\
k &\sim \operatorname{PoissonLog}(f).
\end{aligned}
```

The two positive scale parameters use Stan's `log` transform with its exact
Jacobian; `f_tilde` is unconstrained, so `q = (log ρ, log α, \tilde f)`. The
non-centered latent GP `f = L·f_tilde` applies the parameter-dependent Cholesky
**factor** to the standard-normal draw, and the likelihood consumes `f` through
the natural `log_rate` HAVE route of the shared `poisson` endpoint.

```text
unconstrained (log ρ, log α, f_tilde) ──► ρ, α ──► covariance ──► cholesky.L
        │                                                              │
   priors (Gamma, Normal, Normal(f_tilde)) ─► log_prior     f = L·f_tilde
        │                                                              │
  log_prior + likelihood + log|J| ◄── Σ poisson_log(kₙ | fₙ) ◄─────────┘
```

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :GPPoisRegrExample, :GP_POIS_REGR_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_gp_pois_regr!,
)
```

The native primal and the plain-Enzyme reverse gradient match BridgeStan's
reference `.stan` (`propto = false`, `jacobian = true`) to machine precision on
the full data.

## Reactant

The authored graph compiles and executes its **primal** through Reactant with
value parity — the parameter-dependent covariance, its Cholesky factor, and the
`f = L·f_tilde` matvec lower through XLA (design input `x` stays bound, counts
`k` bound, only the unconstrained draw traced).

The **compiled Reactant reverse gradient does not lower for this shape**: XLA /
EnzymeMLIR has no adjoint for the dense Cholesky primitive `stablehlo.cholesky`
(the latent-GP factor). This is an upstream limitation, documented in the
`reactivekernels-use` skill §7f — the supported AD path is the native primal
plus the native plain-Enzyme reverse gradient (both machine-precision accurate
against Stan), together with the Reactant primal.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.GPPoisRegrExample.demo()'
```
