# Declarative PPL kernel: dogs (correlated avoidance learning)

This example ports the `dogs_nonhierarchical` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`dogs-dogs_nonhierarchical`) into the declarative-`@kernel` style. It is the
Solomon–Wynne avoidance-learning model in its CORRELATED per-dog form, authored
on the FULL real data (J = 30 dogs × T = 25 trials) loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/dogs_nonhierarchical.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/dogs_nonhierarchical.jl).
Each dog `j` has its own multiplicative learning rates `a_j, b_j ∈ (0,1)`, drawn
from a bivariate logit-normal with a shared mean, shared scales and a correlation:

```math
\begin{aligned}
(\operatorname{logit} a_j,\ \operatorname{logit} b_j) &= \mu + z_j \cdot \operatorname{diag}(\sigma)\, L', \\
p_{j,t} &= a_j^{\,\text{prev\_shock}_{j,t}}\; b_j^{\,\text{prev\_avoid}_{j,t}}, \qquad
y_{j,t} \sim \operatorname{Bernoulli}(p_{j,t}),
\end{aligned}
```

where `prev_shock`/`prev_avoid` are the running counts of prior shocks/avoids
BEFORE trial `t` (both 0 at `t = 1`). The non-centered parameterization uses
`z ~ Normal(0,1)` (a `J×2` matrix) and a `cholesky_factor_corr[2]` `L`. Priors:
`μ_k ~ Logistic(0,1)`, `σ_k ~ Normal(0,1)` truncated to `> 0`,
`L ~ lkj_corr_cholesky(2)`.

**Raw `y` is the ONLY data HAVE.** The running-count design is derived entirely
in-graph from the bound `y` matrix: the fixed structural operator
`C[s,t] = (s < t)` is built in-graph from the declared trial count (a
strict-upper-triangular mask), and `prev_shock = y·C`, `prev_avoid = (1-y)·C`,
the log-probability matrix, and the per-dog scaling all live in named graph
nodes, so binding `y` (`bound = (; y)`) hoists the whole data-only prefix.

Two transforms carry a Jacobian: each `σ_k = exp(u)` (half-normal, Stan drops the
`log 2`), and `L` from a single unconstrained `w` via
`L21 = tanh(w), L22 = sqrt(1 - L21²)` (Jacobian `log(1 - L21²)`); the `K=2` LKJ
log-density is the analytic `2·log(L22) - log(4/3)`.

```text
y ──► C = (1:T .< 1:T'), prev_shock = y·C, prev_avoid = (1-y)·C   (data-only prefix, hoisted)
unconstrained ──► μ, u_σ, w, z1, z2 ──► σ, L21, L22 ──► per-dog log a, log b
  │                                        │           └─► log p = prev_shock·log a + prev_avoid·log b
  │                                        └─► log prior (μ, σ, LKJ, z)
  └─ (u_σ, w) ──► log Jacobian              vec(log p), vec(y) ──► Bernoulli plate ─► log likelihood
```

At `t = 1` both counts are 0, so `p = 1` exactly with a zero-gradient exponent;
the Bernoulli endpoint's branch-selecting `logpdf` keeps that boundary finite
(the `benchmark/batch1_gate.jl` support-boundary probe puts `z = 0` to exercise
it against Stan).

The panel below shows **Raw input** (the source), the **Generated kernel**, and
the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :DogsNonhierarchicalExample, :DOGS_NH_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_dogs_nonhierarchical!,
)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/batch1_gate.jl` `@compile`s
both the primal and the gradient and asserts they match the native evaluation and
the reference `.stan`. The in-graph triangular operator, the `y·C` running-count
matrices, the per-dog broadcast, and the multiplicative Bernoulli likelihood all
lower cleanly.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.DogsNonhierarchicalExample.demo()'
```
