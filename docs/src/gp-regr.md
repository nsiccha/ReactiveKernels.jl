# Declarative PPL kernel: Gaussian-process regression (marginal)

This is the posteriordb `gp_regr` model (posterior `gp_pois_regr-gp_regr`): a
one-dimensional Gaussian-process regression with the exponential-quadratic
covariance function, the latent GP analytically marginalized. Its lesson is a
**dense, parameter-dependent Cholesky built inside the graph**, evaluated on the
full real posteriordb data (`N = 11`) loaded through `PosteriorDB.jl`.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/gp_regr.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/gp_regr.jl).
It models the response `y` as

```math
\begin{aligned}
\rho &\sim \operatorname{Gamma}(25, 4), \quad
\alpha \sim \operatorname{Normal}^{+}(0, 2), \quad
\sigma \sim \operatorname{Normal}^{+}(0, 1), \\
K_{ij} &= \alpha^2 \exp\!\left(-\tfrac{(x_i - x_j)^2}{2\rho^2}\right) + \sigma\,\delta_{ij}, \\
y &\sim \operatorname{MultiNormalCholesky}(0,\ \operatorname{chol}(K)).
\end{aligned}
```

The three positive scale parameters enter through Stan's `log` transform with
the exact `lower=0` Jacobian, so the unconstrained draw is
`q = (log ρ, log α, log σ)` — identical to BridgeStan's. The squared-distance
design is an **in-graph node** whose only input is the bound `x`, so `bound=`
partial evaluation folds it to a compile-time constant; the covariance, its
Cholesky factor, the gamma/normal priors, the `multi_normal_cholesky`
log-likelihood and the unconstrained posterior are all separate named nodes.

```text
unconstrained (log ρ, log α, log σ) ──► ρ, α, σ ──► covariance K ──► cholesky(K)
        │                                                                  │
   priors (Gamma, Normal) ─► log_prior          multi_normal_cholesky(y|0,L) ─► likelihood
        │                                                                  │
  log_prior + likelihood + log|J| ─────────────────────────► posterior
```

The panel below shows three views of this model: **Raw input** (the exact
executed source), a readable **Generated kernel**, and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :GPRegrExample, :GP_REGR_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_gp_regr!,
)
```

The native primal log-density and the plain-Enzyme reverse gradient match
BridgeStan's reference `.stan` (`propto = false`, `jacobian = true`) to machine
precision on the full data.

## Reactant

The exact authored graph compiles and executes its **primal** log-density
through the public Reactant boundary with value parity — the covariance,
Cholesky, and the `multi_normal_cholesky` whitening lower through XLA. The
*response* is passed traced (as with the `mvnormal` object), which lets the
triangular solve lower; the design input `x` stays bound so the squared-distance
node folds.

The **compiled Reactant reverse gradient does not lower for this shape**: XLA /
EnzymeMLIR has no adjoint for the dense Cholesky primitive
`stablehlo.triangular_solve` (the marginal-GP solve). This is an upstream
limitation, documented in the `reactivekernels-use` skill §7f — the supported AD
path here is the native primal plus the native plain-Enzyme reverse gradient
(both machine-precision accurate against Stan), together with the Reactant
primal. The approximate/basis GP (see [Approximate GP (HSGP)](accel-gp.md)) has
no Cholesky and lowers all of primal, Reactant primal, and Reactant gradient.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.GPRegrExample.demo()'
```
