# Efficient Gibbs sampling on ReactiveKernels — proof of concept

Investigates whether ReactiveKernels can support efficient Gibbs sampling,
prompted by the spike-and-slab Turing.jl thread
(<https://discourse.julialang.org/t/spike-and-slab-implementation-using-turing-jl/132816>).

**Answer: yes, and the machinery already exists in the core.** A Gibbs sampler
needs, for each variable block, the *full conditional* — which depends only on
that block's Markov blanket. ReactiveKernels' two core capabilities line up
exactly:

1. **have→want planning** (`plan` / `prepare`) computes *only* the recipes a
   requested conditional needs — you slice any conditional out of one joint
   model graph, with no separate model per block and no hand-derived "which
   terms matter".
2. **provenance-aware invalidation** (`ReactiveState` + `set!` / `get!`)
   recomputes a cached quantity only when a value in its *actual* selected-plan
   dependency set changed. Updating one block invalidates exactly its dependents
   — the Markov blanket — automatically, with no hand-written caching.

## Files

| File | What it shows |
|---|---|
| `incremental_provenance.jl` | The bare mechanism: `set!`-ing one source recomputes only the terms whose Markov blanket contains it; unrelated terms stay cached. Instrumented with op-execution counters. |
| `ssvs_gibbs.jl` | A complete, correct SSVS spike-and-slab Gibbs sampler (George & McCulloch 1993) built on `ReactiveState`. Recovers a known sparse truth, is bit-identical to a hand-written reference, and measures the per-sweep recompute. |
| `setup.jl` | Builds the local example env (develops the repo, adds `Distributions`, `StableRNGs`). |

## Run

```sh
julia --startup-file=no benchmark/gibbs_poc/setup.jl
julia --startup-file=no --project=benchmark/gibbs_poc benchmark/gibbs_poc/incremental_provenance.jl
julia --startup-file=no --project=benchmark/gibbs_poc benchmark/gibbs_poc/ssvs_gibbs.jl
```

## Model (the spike-and-slab family from the thread)

```
ω        ~ Beta(a0, b0)                       inclusion probability
z_j      ~ Bernoulli(ω)                       inclusion indicator, j = 1..p
β_j | z_j ~ Normal(0, z_j ? τ²_slab : τ²_spike)
σ²       ~ InverseGamma(a_σ, b_σ)
y | β,σ² ~ Normal(X β, σ² I)
```

Every full conditional is conjugate, so this is a pure Gibbs sampler. The four
blocks have sharply different Markov blankets:

| block | conditional | blanket | cost |
|---|---|---|---|
| `β`  | Normal(P⁻¹rhs, P⁻¹), P = XᵀX/σ² + diag(1/τ²_z) | data suff-stats, σ², z | O(p²)/O(p³) |
| `σ²` | InverseGamma(a_σ+N/2, b_σ+SSR/2)               | β, data suff-stats     | O(p²) |
| `z`  | independent Bernoulli(logistic(log-odds_j))    | ω, β  (**not the data**) | O(p) |
| `ω`  | Beta(a0+Σz, b0+p−Σz)                            | Σz                     | O(p) |

The whole model is authored **once** as a have→want graph; data sufficient
statistics (`XᵀX`, `Xᵀy`, `yᵀy`) and hyperparameters are fixed source values,
each parameter block is a source value, and every conditional-input quantity is
a recipe. The Gibbs loop just `set!`s each block after drawing it and `get!`s
the next block's conditional inputs.

## Measured results (`ssvs_gibbs.jl`, `p = 30`, `N = 200`, 5 true active)

- **Correctness** — recovered active set `[1,2,8,11,15]` exactly (PIP = 1.0 on
  every true-active coefficient, max PIP 0.033 off-active); β posterior means
  match the truth.
- **Identity** — β, z, σ², ω chains are **bit-identical** to an independent
  hand-written conjugate Gibbs sampler sharing the RNG. The reactive graph
  changes only *what* is recomputed, never the numbers.
- **Efficiency** — measured op executions per sweep:
  - `β_prec` (O(p²)) 1×, `SSR` (O(p²)) 1× — each expensive quantity recomputes
    once per sweep, only because its inputs actually changed.
  - The `z` and `ω` updates recompute **zero** data-touching quantities — the
    planner's Markov blanket automatically excludes the data.
  - A provenance-blind sampler that re-scores the whole model at each block
    (what a generic PPL does) executes **8** expensive ops/sweep vs
    ReactiveKernels' **2**; the gap widens with every additional cheap block.

## Where the incremental cache pays off most

In this fully-coupled regression every block touches shared suff-stats, so the
headline win is the **slicing** (2 vs 8). The provenance *cache* adds
cross-update reuse whenever a term's inputs are stable between requests — which
is dramatic for models with **locality** (state-space / Markov-chain latents,
Markov random fields, hierarchical models with many local terms), where a
single-site or local-block update leaves almost every other term's cached
conditional inputs valid. `incremental_provenance.jl` demonstrates that
mechanism directly.
