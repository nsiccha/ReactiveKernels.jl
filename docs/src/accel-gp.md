# Declarative PPL kernel: approximate GP (HSGP, distributional)

This is the posteriordb `accel_gp` model (posterior `mcycle_gp-accel_gp`): the
`brms`-generated **Hilbert-space approximate Gaussian process** (HSGP) for the
motorcycle-acceleration data. It is a *distributional* model — a latent GP on
**both** the mean and the log-standard-deviation of a Normal response. Its
lesson is that an approximate/basis GP is **pure dense matrix-vector products
with no covariance matrix and no Cholesky**, so — unlike the exact GPs
([marginal](gp-regr.md), [latent Poisson](gp-pois-regr.md)) — it lowers all
four acceptance axes through Reactant, gradient included. Full real data
(`N = 133`) is loaded through `PosteriorDB.jl`.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/accel_gp.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/accel_gp.jl).
Each GP is the basis expansion

```math
\operatorname{gp}(X, \operatorname{sdgp}, \ell, z) = X\,\big(\sqrt{s} \odot z\big), \quad
s_m = \operatorname{sdgp}^2 \sqrt{2\pi}\,\ell \, \exp\!\big(-\tfrac{1}{2}\ell^2 \lambda_m^2\big),
```

where `X` are the Laplacian eigenfunctions and `√λ` are the square roots of the
Laplacian eigenvalues (the brms `spd_cov_exp_quad`, `D = 1`). The mean is
`μ = Intercept + gp₁`, the standard
deviation `σ = exp(Intercept_σ + gp_σ)`, and `Y ~ Normal(μ, σ)`. The four
positive scale parameters (`sdgp`, `lscale` on each GP) use Stan's `log`
transform with its exact Jacobian; the priors reuse the shared `student_t`
(intercepts, half-Student-t marginal SDs), `inverse_gamma` (length-scales), and
`normal` (latent coefficients) endpoints.
The Stan `prior_only` 0/1 data flag is validated at raw-data entry and exposed
through `accel_gp_posterior_want`. Its strict Boolean argument selects
`:prior_only_posterior` (the prior and Jacobian only; planning that node does
not compute the likelihood) or the likelihood-including `:posterior`.

```text
unconstrained ──► Intercept, sdgp, lscale, zgp (×2, mean + log-sd) ──► spectral density s
        │                                                                        │
   priors (Student-t, half-Student-t, inv-Gamma, Normal) ─► log_prior    μ, σ = f(X·(√s ⊙ z))
        │                                                                        │
  log_prior + likelihood + log|J| ◄──────────────── Σ Normal(Yₙ | μₙ, σₙ) ◄──────┘
```

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :AccelGPExample, :ACCEL_GP_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_accel_gp!,
)
```

The native primal and the plain-Enzyme reverse gradient match BridgeStan's
reference `.stan` (`propto = false`, `jacobian = true`) to machine precision on
the full data.

## Reactant

Because the HSGP is only dense matrix-vector products, the exact authored graph
lowers **all four** acceptance axes through Reactant: primal (all data bound,
only the unconstrained draw traced), and — unlike the dense-Cholesky GPs — the
**compiled reverse gradient**, which matches BridgeStan to machine precision.
This is the approximate-GP case that avoids the `stablehlo.cholesky` /
`stablehlo.triangular_solve` adjoint gap documented in `reactivekernels-use`
§7f.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.AccelGPExample.demo()'
```
