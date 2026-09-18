# Declarative PPL kernel: kronecker_gp (Kronecker-structured GP over a grid)

This example ports the `kronecker_gp` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`synthetic_grid_RBF_kernels-kronecker_gp`) into the declarative-`@kernel`
style. A Gaussian process over a 2-D grid factors as a Kronecker product of an
RBF-kernel margin (`var1`, `bw1`) and a correlation margin (the Cholesky factor
`L` of a correlation matrix with an LKJ(2) prior), plus `sigma1` observation
noise. Both margins are diagonalized **exactly and in-graph** through
`eigen(Symmetric(·))` — Stan's `eigenvectors_sym` / `eigenvalues_sym` — so the
marginal likelihood is a contraction against the product eigenspectrum without
materializing the Kronecker product (Stan's `kron_mvprod`):

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/kronecker_gp.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/kronecker_gp.jl).

```math
\begin{aligned}
L_{n,v}^{\text{eig}} &: \quad
\text{likelihood} = -\tfrac{1}{2}\sum\!\Big(y \odot \tfrac{(Q_1^{\!\top}\!\otimes Q_2^{\!\top})\,y}{R_2 R_1^{\!\top} + \sigma_1}\Big)
-\tfrac{1}{2}\sum \log\!\big(R_2 R_1^{\!\top} + \sigma_1\big), \\
\text{var}_1 &\sim \operatorname{LogNormal}(0, 1), \quad
\text{bw}_1 \sim \operatorname{Cauchy}(0, 2.5), \\
\sigma_1 &\sim \operatorname{LogNormal}(0, 1), \quad
L \sim \operatorname{LKJ-Corr-Cholesky}(2).
\end{aligned}
```

The data-derived squared-distance matrix `xd` (`-(x1[i] - x1[j])^2`) is
computed **in-graph** from the bound locations `x1`, exactly Stan's
transformed-data loop. The reference data is 30×30 (`n1 = n2 = 30`, the
model's data contract).

The `cholesky_factor_corr[30]` constraining transform — Stan's
`cholesky_corr_constrain` — is authored in-graph in its exact upstream form:
elementwise `tanh` (with its `log(1 − tanh²)` Jacobian terms) filling the
strict lower triangle row-major, the `0.5·log(1 − sum_sqs)` partial-sum
Jacobian terms, and unit diagonal. The lower-bound transforms for `var1`,
`bw1` (`exp`) and `sigma1` (`1e-5 + exp`) carry their own exact log-Jacobian
terms. The unconstrained vector is
`(var1, bw1, L_free[435], sigma1)` in Stan's declared order.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected plan,
and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :KroneckerGpExample, :KRON_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_kronecker_gp!,
)
```

## Automatic differentiation and Reactant boundaries

This page owns the model's native-Enzyme and Reactant support boundary
explicitly (the same per-model disclosure policy as the GP pages):

- **Native value** — hard-asserted against the reference `.stan` via BridgeStan
  (`propto = false`, `jacobian = true`) on multiple reference-valid probes.
- **Native gradient** — ordinary `AutoEnzyme(mode = Enzyme.Reverse)` through
  the graph containing two dense symmetric eigendecompositions; per-axis
  support is recorded by the structured gate with the complete diagnostic
  retained for any unsupported axis.
- **Reactant axes** — the all-bound and traced-stream compiled queries, and
  the Reactant-compiled gradient, are measured per axis in the isolated
  Reactant phase of `benchmark/structured_gate.jl`. Because dense symmetric
  eigendecomposition lowering through Reactant/EnzymeMLIR is not established
  at this pin, this model's Reactant axes run in measured mode: each axis is
  recorded as PASS or UNSUPPORTED with its complete retained diagnostic —
  never asserted in advance and never silently skipped.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.KroneckerGpExample.demo()'
```
