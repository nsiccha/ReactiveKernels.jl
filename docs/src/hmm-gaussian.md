# Declarative PPL kernel: Gaussian HMM (K states)

This example ports the `hmm_gaussian` model from
[posteriordb](https://github.com/stan-dev/posteriordb)
(posterior `hmm_gaussian_simulated-hmm_gaussian`, the stancon18 model) into the
declarative-`@kernel` style: a `K`-state Gaussian hidden Markov model with a full
initial distribution `pi1`, a `K×K` transition matrix `A`, `ordered` means and
per-state standard deviations. Like [hmm_example](hmm-example.md) the likelihood
is the **sequential forward algorithm** over a K-vector belief carry, authored
with the [`scan` primitive](scan.md), and **lowers through Reactant directly**.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/hmm_gaussian.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/hmm_gaussian.jl).
For observations `y₁,…,y_T` and `K` states:

```math
\gamma_t[j] = \operatorname{logsumexp}_i\!\big(\gamma_{t-1}[i] + \log A_{i,j}\big)
             + \operatorname{Normal}(y_t \mid \mu_j, \sigma_j),
```

with the marginal log-likelihood `logsumexp(γ_T)`. There are no explicit priors.
A **faithful detail** from the Stan source is preserved exactly: at `t = 1` Stan
writes `logalpha[1] = log(pi1) + normal_lpdf(y[1] | mu, sigma)`, where the
vectorized `normal_lpdf(real | vector, vector)` returns the *sum over all K
states*, so the full emission sum is added uniformly to every initial state.

The unconstrained vector is `(pi1_free, A_free, μ_free, σ_free)`. The Stan 2.39
support transforms are matched exactly — the **inverse-ILR simplex** for `pi1`
and each of the `K` transition rows (constrained together through the shared
basis), `ordered` for `μ` (Jacobian `sum(v[2:K])`), and `exp` for `σ`.

```text
unconstrained ─► pi1, A (K inverse-ILR simplexes), μ (ordered), σ (exp) + Jacobian
   │                                                   │
y ─┴─► γ₁ (t=1 emission-sum quirk) ─► scan (K-vector carry) ─► likelihood = logsumexp(γ_T)
        (lowers via stablehlo.while)
likelihood + log Jacobian ──► unconstrained log density
```

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel**, and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :HmmGaussianExample, :HMM_GAUSSIAN_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_hmm_gaussian!,
)
```

## Reactant

The K-vector-carry forward algorithm **lowers through Reactant directly on the
traced raw-series query**: the walkthrough entry passes `y` as a traced argument
and binds only the scalar state count `K`. The `K` transition simplexes are
constrained at once by reshaping the free block and mapping it through the
shared inverse-ILR basis (a matmul), and the belief state is a `scan` carry
whose per-step update is a `K×K` `logsumexp` over the previous state, so `scan`
lowers the recurrence to a `stablehlo.while` carry loop (indices use the
explicit `T`, never `end`, which does not resolve on traced results). The
all-bound query (host `y` partial-evaluated) is a separate, also-supported
native boundary; it unrolls the fixed-length recursion instead of emitting the
loop. The translation was verified against the reference `.stan` via BridgeStan
(`propto=false, jacobian=true`): native value and gradient, and the
Reactant-compiled primal and gradient, all match to machine precision.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.HmmGaussianExample.demo()'
```
