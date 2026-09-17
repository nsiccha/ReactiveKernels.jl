# Declarative PPL kernel: input-output HMM

This example ports the `iohmm_reg` model from
[posteriordb](https://github.com/stan-dev/posteriordb)
(posterior `iohmm_reg_simulated-iohmm_reg`, the stancon18 model) into the
declarative-`@kernel` style: an input-output hidden Markov model whose `K`-state
transition probabilities and emission means **both depend on a per-observation
input vector** `u_t ∈ ℝ^M`. Like the other HMMs the likelihood is the
**sequential forward algorithm** over a K-vector belief carry — here with
*input-dependent* per-step transition and emission — authored with the
[`scan` primitive](scan.md) over `eachrow` of a combined per-step matrix, and
**lowers through Reactant directly**.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/iohmm_reg.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/iohmm_reg.jl).
For observations `y₁,…,y_T`, inputs `u_t`, and `K` states:

```math
A_t = \operatorname{softmax}_j(u_t \cdot w_j), \qquad
\gamma_t[j] = \operatorname{logsumexp}_i\!\big(\gamma_{t-1}[i] + \log A_t[i]\big)
             + \operatorname{Normal}(y_t \mid u_t \cdot b_j, \sigma_j),
```

with the marginal log-likelihood `logsumexp(γ_T)`. A **faithful detail** from the
Stan source is preserved exactly: the transition term enters the accumulator as
`logA[t][i]` — indexed by the *previous* state `i` only, the same for every
current state `j` (`A_t` is a single softmax vector, not a `K×K` matrix).

The parameters are `pi1` (simplex), the transition regressors `w[j]` and mean
regressors `b[j]` (unconstrained `M`-vectors), and per-state `σ`, with priors
`w, b ~ Normal(0,5)` and `σ ~ Normal(0,3)`. The unconstrained vector is
`(pi1_free, w, b, σ_free)`; the transforms are the inverse-ILR simplex for `pi1`
and `exp` for `σ` (`w, b` unconstrained). The input-dependent transition and
emission designs `u·w` and `u·b` are built inside the graph from the raw input
matrix.

```text
unconstrained ─► pi1 (inverse-ILR simplex), w, b, σ (exp) + Jacobian
   │                    │
u ─┴─► logA = softmax(u·w),  logoblik = N(y | u·b, σ)  (in-graph designs)
        │
        └─► γ₁ ─► scan(eachrow) over the belief state (K-vector carry) ─► logsumexp(γ_T)
prior + likelihood + log Jacobian ──► unconstrained log density
```

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel**, and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :IohmmRegExample, :IOHMM_REG_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_iohmm_reg!,
)
```

## Reactant

The input-dependent forward algorithm **compiles and executes through Reactant
on the all-bound query, with a disclosed lowering gap.** The per-step transition
and emission rows vary with `u_t`, so they are precomputed as `(T-1)×K` matrices
and the `scan` iterates their rows via `scan(eachrow(...))`, threading the
K-vector belief state. Because the per-step scan inputs are K-vectors, this
model cannot use the `stablehlo.while` scan lowering (which gathers only 1-D
traced sequences): Reactant compiles the correct unrolled program instead —
value/gradient parity holds, at the cost of a program that grows with `T`.
Iterating a traced matrix directly fails Reactant's scalar-indexing boundary,
so the row form stays the natural authoring. The translation was verified
against the reference `.stan` via BridgeStan (`propto=false, jacobian=true`):
native value and gradient, and the Reactant-compiled primal and
gradient, all match to machine precision.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.IohmmRegExample.demo()'
```
