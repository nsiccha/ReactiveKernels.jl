# Declarative PPL kernel: nn_rbm1b (neural-network softmax classifier)

This example ports the `nn_rbm1b` single-hidden-layer neural-network softmax
classifier from [posteriordb](https://github.com/stan-dev/posteriordb) into the
declarative-`@kernel` style, in the RBM prior parametrization of Lampinen &
Vehtari (2001). The graph is **data-generic** — the hidden-unit count `J`, the
class count `K`, and the data `x`, `y` are HAVE ports — so ONE authored source
certifies **both** posteriordb definitions:

| posterior | dataset | N | M (pixels) | K | J | unconstrained dim |
|---|---|---|---|---|---|---|
| `mnist_100-nn_rbm1bJ10` (rendered default below) | `mnist_100` | 100 | 784 | 10 | 10 | 7951 |
| `mnist-nn_rbm1bJ100` | `mnist` | 60000 | 784 | 10 | 100 | 79411 |

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/nn_rbm.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/nn_rbm.jl).
This is **not** the existing `mnist_logistic` model (that is a plain
multinomial-logistic classifier with a zero-logit reference and no hidden layer).

```math
\begin{aligned}
\sigma^2_\alpha &\sim \operatorname{Inv\text{-}Gamma}(\nu_\alpha/2,\ \nu_\alpha s^2_{0,\alpha}/2), &
\sigma^2_\beta &\sim \operatorname{Inv\text{-}Gamma}(\nu_\beta/2,\ \nu_\beta s^2_{0,\beta}/2), \\
\alpha_{1,j},\ \beta_{1,c} &\sim \operatorname{Normal}(0, 1), &
\alpha_{ij} &\sim \operatorname{Normal}(0, \sigma_\alpha),\quad
\beta_{jc} \sim \operatorname{Normal}(0, \sigma_\beta), \\
H &= \tanh(x\,\alpha + \alpha_1), &
v_n &= \bigl[\,1;\ (H\beta + \beta_1)_n\,\bigr], \\
y_n &\sim \operatorname{Categorical\text{-}logit}(v_n). &&
\end{aligned}
```

The prior scales `s^2_{0,\alpha} = (0.05/M^2)^2`, `s^2_{0,\beta} = (0.05/J^2)^2`
(with `\nu = 0.5`) are transformed data — functions of the fixed dimensions only —
so they are authored inline and fold to constants. Class 1 is the **reference**
with a fixed logit of `1` (the `.stan` `append_col(ones, ...)` column), which is
why the whole-vector `categorical_logit` is used rather than the zero-reference
`categorical_logit_ref`.

The unconstrained vector is Stan's declaration order (matrices column-major):
`(u_α, u_β, vec(α), vec(β), α₁, β₁)`. The packed variance coordinates are
`u_α` and `u_β`; each passes through `exp` to `σ²` and then `sqrt` to `σ`, with
Jacobian `u_α + u_β`. The weights are unconstrained
Normal. `to_vector(α) ~ Normal(0, σ_α)` and `to_vector(β) ~ Normal(0, σ_β)` are
authored as whole-vector reductions (one normalization each), while `α₁`, `β₁`
reuse the shared `normal` endpoint and the two variances reuse `inverse_gamma`.

```text
unconstrained ─► u_α, u_β ─► exp ─► σ²_α, σ²_β ─► sqrt ─► σ_α, σ_β ───┐ (Jacobian: u_α + u_β)
    │                                                ├─► log prior
    ├─► α, β, α₁, β₁ ──────────────────────────────┐ │
x ──┴─► tanh(x·α + α₁) = H ─► [1; H·β + β₁] = v ────┼─┴─► categorical_logit(vₙ).logpdf(yₙ)
                                                    └────► summed log likelihood

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The panel below renders the **rendered default** — `mnist_100-nn_rbm1bJ10` (the
small committed example, executed on the FULL `mnist_100` dataset): **Raw input**
(the source), a readable **Generated kernel** derived from the executed kernel and
selected plan, and the **Compute DAG**. `mnist-nn_rbm1bJ100` binds the same
authored source to full MNIST with `J = 100`.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :NNRBMExample, :NN_RBM_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_nn_rbm!,
)
```

Named nodes make alternate boundaries cheap: `want = :pointwise` cuts the same
plate to the per-observation log-likelihoods; `want = :prior` or `:log_jacobian`
prunes the likelihood entirely.

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/batch_textnn_gate.jl`
`@compile`s both the primal and the gradient and asserts they match the native
evaluation and the reference `.stan` (via BridgeStan). For
`mnist_100-nn_rbm1bJ10` the data matrix is **bound** for the Reactant compile as
for the native axes. For `mnist-nn_rbm1bJ100` the 376,320,000-byte design matrix
`x` exceeds the observed Reactant-0.2.285 constant-embedding threshold
(104857600 bytes), so it is passed as a **traced runtime input** (still inactive
in the gradient) — a different data-entry boundary with an identical graph,
value, and parameter gradient (reactivekernels-use §7e).

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.NNRBMExample.demo()'
```
