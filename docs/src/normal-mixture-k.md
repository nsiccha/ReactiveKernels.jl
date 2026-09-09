# Declarative PPL kernel: normal mixture (marginalized simplex)

This example ports the `normal_mixture_k` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`normal_5-normal_mixture_k`) into the declarative-`@kernel` style. It is a
`K`-component univariate normal mixture with unknown mixing weights, means and
scales; the discrete component label is marginalized analytically, so no discrete
parameter appears. It is authored on the FULL real data (N = 1701, K = 5) loaded
through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/normal_mixture_k.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/normal_mixture_k.jl):

```math
\begin{aligned}
y_n &\sim \sum_{k=1}^{K} \theta_k\, \operatorname{Normal}(\mu_k, \sigma_k), \\
\theta &\sim \operatorname{Simplex}(K)\ (\text{uniform}), \quad
\mu_k \sim \operatorname{Normal}(0, 10), \quad
\sigma_k \sim \operatorname{Uniform}(0, 10),
\end{aligned}
```

with the per-observation likelihood written as a K-way log-sum-exp
`log p(y_n) = \operatorname{logsumexp}_k(\log\theta_k + \operatorname{Normal\_lpdf}(y_n \mid \mu_k, \sigma_k))`.

The model is authored as a **natural K-dimensional graph with `K` a bound data
port** (`int<lower=1> K` in the `.stan` data block), not a `K = 5` unrolling:
binding `K` folds every K-dependent shape at preparation, and the same spec
serves any K (the regression tests exercise a small `K = 3` alongside the real
`K = 5`). The unconstrained vector is `(θ_free[1..K-1], μ[1..K], σ_free[1..K])`.

Two coordinates carry only their transform Jacobian (no prior density):

- `θ` uses Stan 2.39's default simplex transform, the inverse
  isometric-log-ratio `θ = softmax(sum_to_zero_constrain(tu))`.
  `sum_to_zero_constrain` is a fixed linear map of the `K-1` free values,
  authored in-graph as the `K × (K-1)` contrast matrix
  `A[i,j] = [j ≥ i] - j·[j = i-1]` built from the bound `K`; its Jacobian is
  `Σ_k log θ_k + 0.5·log K`.
- `σ_k = 10·logistic(su_k)` lands in `(0, 10)` with the interval Jacobian
  `log 10 - log1pexp(-su) - log1pexp(su)`.

```text
unconstrained ──► tu, μ, σ_free (K-dependent views, bound K)
  │      tu ──► w ──► s = A·w ──► log θ, θ  ──► log prior (μ) 
  │                    │                   └─► log Jacobian (θ)
  │      σ_free ──► σ ─┴──────────────────────► log Jacobian (σ)
  └─ y, μ, σ, log θ ──► N×K log-density ──► per-obs logsumexp ─► log likelihood
```

The panel below shows **Raw input** (the source), the **Generated kernel**, and
the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :NormalMixtureKExample, :NORMAL_MIXTURE_K_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_normal_mixture_k!,
)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/batch1_gate.jl` `@compile`s
both the primal and the gradient and asserts they match the native evaluation and
the reference `.stan`. The in-graph contrast-matrix simplex map, the interval
scale transform, and the whole N×K marginalized likelihood all lower cleanly.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.NormalMixtureKExample.demo()'
```
