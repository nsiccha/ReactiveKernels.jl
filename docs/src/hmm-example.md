# Declarative PPL kernel: HMM (forward algorithm)

This example ports the `hmm_example` model from
[posteriordb](https://github.com/stan-dev/posteriordb)
(posterior `hmm_example-hmm_example`) into the declarative-`@kernel` style: a
2-state unit-variance Gaussian hidden Markov model whose likelihood is the
marginal over the discrete state path, computed by the **sequential forward
algorithm**. Unlike the [ARMA](arma11.md)/[GARCH](garch11.md) recursions (a
scalar carry), the forward algorithm threads a **K-vector belief-state carry**,
authored with the [`scan` primitive](scan.md) and lowering to a
`stablehlo.while` loop — so the natural sequential form **lowers through Reactant
directly**.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/hmm_example.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/hmm_example.jl).
For observations `y₁,…,y_N` and `K` states:

```math
\gamma_1[k] = \operatorname{Normal}(y_1 \mid \mu_k, 1), \qquad
\gamma_t[k] = \operatorname{logsumexp}_j\!\big(\gamma_{t-1}[j] + \log\theta_{j,k}\big)
             + \operatorname{Normal}(y_t \mid \mu_k, 1),
```

and the marginal log-likelihood is `logsumexp(γ_N)`. The transition rows are two
simplexes `θ₁, θ₂` and the means are `positive_ordered`, with the separated prior
`μ₁ ~ Normal(3,1)`, `μ₂ ~ Normal(10,1)`. The unconstrained vector is
`(θ₁_free, θ₂_free, μ_free)`. The Stan 2.39 support transforms are matched
exactly — the **inverse-ILR simplex** (`z = softmax(sum_to_zero_constrain(free))`,
Jacobian `sum(log z) + ½ log K`), authored here as a single linear map so it
lowers through Reactant, and `positive_ordered` for `μ` (Jacobian `sum(u)`).

```text
unconstrained ─► θ₁, θ₂ (inverse-ILR simplex), μ (positive_ordered) + Jacobian
   │                    │                          │
y ─┴─► γ₁ ─► scan over the belief state (K-vector carry) ─► likelihood = logsumexp(γ_N)
        (lowers via stablehlo.while)
prior + likelihood + log Jacobian ──► unconstrained log density
```

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel**, and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :HmmExampleExample, :HMM_EXAMPLE_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_hmm_example!,
)
```

## Reactant

The K-vector-carry forward algorithm **lowers through Reactant directly on the
traced raw-series query**: the walkthrough entry passes `y` as a traced argument
and binds only the scalar state count `K`. The belief state `γ` is a `scan`
carry; each step forms the `K×K` accumulator `γ[j] + logθ[j,k]` and reduces it
with a `logsumexp` over the previous state, so `scan` lowers the recurrence to a
`stablehlo.while` carry loop with a K-vector loop-carried value (indices use the
explicit `T`, never `end`, which does not resolve on traced results). The
all-bound query (host `y` partial-evaluated) is a separate, also-supported
native boundary; it unrolls the fixed-length recursion instead of emitting the
loop. The translation was verified against the reference `.stan` via BridgeStan
(`propto=false, jacobian=true`): native value and gradient, and the
Reactant-compiled primal and gradient, all match to machine precision.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.HmmExampleExample.demo()'
```
