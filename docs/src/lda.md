# Declarative PPL kernel: LDA (latent Dirichlet allocation, ldaK2 / ldaK5)

This example ports the `ldaK2` / `ldaK5` latent Dirichlet allocation models from
[posteriordb](https://github.com/stan-dev/posteriordb) into the
declarative-`@kernel` style, with the discrete per-word topic assignment
marginalized analytically (a stable `log_sum_exp` over topics), exactly as the
`.stan` model block writes it. The graph is **data-generic** — the topic count
`K` and the two Dirichlet concentration vectors `alpha` (length `K`), `beta`
(length `V`) are HAVE ports — so ONE authored source certifies **both**
posteriordb definitions:

| posterior | dataset | V (vocab) | M (docs) | N (word instances) | K | unconstrained dim |
|---|---|---|---|---|---|---|
| `three_men1-ldaK2` (rendered default below) | `three_men1` | 249 | 6 | 4999 | 2 | 502 |
| `prideprejudice_chapter-ldaK5` | `prideprejudice_chapter` | 1495 | 61 | 32877 | 5 | 7714 |

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/lda.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/lda.jl).
For `ldaK2` the `.stan` computes `alpha = ones(K)`, `beta = ones(V)` in transformed
data, so they are bound as the corresponding ones-vectors; for `ldaK5` they are
data (`alpha = beta = 0.1`).

```math
\begin{aligned}
\theta_m &\sim \operatorname{Dirichlet}(\alpha),\ m = 1..M, &
\phi_k &\sim \operatorname{Dirichlet}(\beta),\ k = 1..K, \\
\log p(w) &= \sum_{n=1}^{N} \operatorname{log\_sum\_exp}_{k}\!\Bigl(
  \log \theta_{\mathrm{doc}_n,\,k} + \log \phi_{k,\,w_n} \Bigr). &&
\end{aligned}
```

Each simplex uses the **Stan-2.39 inverse-ILR transform**
`x = softmax(sum_to_zero_constrain(y))`, whose log-absolute-Jacobian is
`sum(log x) + 0.5·log(dim)` (this is **not** the older stick-breaking transform).
`sum_to_zero_constrain` is the fixed linear map `z = Wstz · y`; the sum-to-zero
basis `Wstz` (a function of the simplex dimension only) is built **in-graph** from
the bound dimension, so ONLY raw data is bound. The unconstrained vector is Stan's
declaration order: `theta` first (M blocks of `K-1`), then `phi` (K blocks of
`V-1`), column-major.

```text
unconstrained ─► ThetaU, PhiU ─► Wstz·U = Z ─► softmax per column ─► log θ, log φ  (Jacobian: Σ log x + ½ log dim per simplex)
                                                       │
alpha, beta ─────────────────────────────────────────┼─► Dirichlet(α/β).logpdf per column ─► log prior
                                                       │
doc, w ─► per-word gather indices ─► log θ[docₙ,·] + log φ[·,wₙ] = γ ─► log_sum_exp(γ) ─► summed log likelihood

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The panel below renders the **rendered default** — `three_men1-ldaK2` (the small
committed example, executed on the FULL `three_men1` dataset): **Raw input** (the
source), a readable **Generated kernel** derived from the executed kernel and
selected plan, and the **Compute DAG**. `prideprejudice_chapter-ldaK5` binds the
same authored source to the larger vocabulary with `K = 5` and data-supplied
concentrations.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :LDAExample, :LDA_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_lda!,
)
```

Named nodes make alternate boundaries cheap: `want = :pointwise` cuts the same
plate to the per-word marginal log-likelihoods; `want = :prior` or `:log_jacobian`
prunes the likelihood entirely.

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/batch_textnn_gate.jl`
`@compile`s both the primal and the gradient and asserts they match the native
evaluation and the reference `.stan` (via BridgeStan). The in-graph sum-to-zero
basis construction (comparison masks + matmul, all data-only) lowers cleanly, and
the per-word integer gathers and the marginalizing `log_sum_exp` compile without
special handling.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.LDAExample.demo()'
```
