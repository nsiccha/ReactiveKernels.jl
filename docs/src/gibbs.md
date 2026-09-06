# Gibbs sampling on reactive kernels

A Gibbs sampler needs, for each block of variables, its **full conditional** —
which depends only on that block's Markov blanket. ReactiveKernels' two core,
public capabilities line up with that exactly, so Gibbs is expressible directly
on the graph API with no new sampler machinery:

- **have→want planning** (`plan` / `prepare`) slices *any* conditional out of one
  joint model graph — only the recipes that conditional needs are computed. No
  separate model per block, no hand-derived "which terms matter".
- **`ReactiveState` provenance invalidation** (`set!` / `get!`) recomputes only
  the changed block's Markov blanket — automatically, with no hand-written
  caching. `set!`-ing one block bumps its version; a cached value is reused iff
  every recorded dependency (the *actual* selected-plan have-leaves) still
  matches.

The two worked examples below live under `benchmark/` and are authored with the
ordinary `Graph` / `value!` / `add!` recipe API. The kernels are shown in full,
read from the source at build time, so the syntax and the interaction are visible
and cannot drift from the code that runs.

## Spike-and-slab variable selection (a conjugate Gibbs sampler)

`benchmark/gibbs_poc/` implements the George & McCulloch (1993) SSVS spike-and-slab
model — the family behind the
[Turing.jl spike-and-slab discussion](https://discourse.julialang.org/t/spike-and-slab-implementation-using-turing-jl/132816).
Every full conditional is conjugate, so it is a pure Gibbs sampler. The whole
model is authored **once** as a have→want graph: the data sufficient statistics
(`XᵀX`, `Xᵀy`, `yᵀy`) and hyperparameters are fixed source values, each parameter
block (`β`, `z`, `ω`, `σ²`) is a source value, and every conditional-input
quantity is a recipe:

```@eval
Main.GibbsDocs.render_ssvs_graph()
```

The Gibbs loop `set!`s each block after drawing it and `get!`s the next block's
conditional inputs. Because the `z` and `ω` conditionals depend only on `ω` / `β`
and `Σz` — not on the data — the reactive layer recomputes **zero** data-touching
quantities when those blocks update; the planner's Markov blanket excludes the
data automatically. Measured: exact recovery of the sparse active set, chains
**bit-identical** to an independent hand-written conjugate Gibbs sampler under the
same RNG (the reactive graph changes only *what* is recomputed, never the
numbers), and 2 data-touching O(p²) ops per sweep versus 8 for a provenance-blind
re-score.

## CGGibbs on a logistic GLM, and an honest head-to-head with NUTS

`benchmark/cggibbs/` reproduces the "compute graph Gibbs" (CGGibbs) idea of
[*Is Gibbs sampling faster than Hamiltonian Monte Carlo on GLMs?* (arXiv 2410.03630)](https://arxiv.org/abs/2410.03630)
for Bayesian logistic regression: cache the linear predictor `η = Xθ` so a
coordinate update is O(nd), not O(nd²), with slice sampling within each
coordinate (GLMs aren't conjugate). The per-coordinate conditional log-density is
authored as a reactive kernel that is prepared once and evaluated by the slice
sampler:

```@eval
Main.GibbsDocs.render_cggibbs_conditional()
```

### An honest note on "reactive caching = CGGibbs for free"

Block-level incrementality (distinct Markov blankets, as in the SSVS `z` / `ω`
blocks) *is* free from the pure reactive provenance above. Coordinate-level
CGGibbs on a dense GLM is different: ReactiveKernels recomputes at **recipe
granularity**, so a dense `η = Xθ` recipe costs O(nd) per coordinate — the naive
cost. The rank-1 `η` cache is therefore **authored** (an explicit buffer here; the
stateful `mutate!` / `touch!` layer is the declarative route). ReactiveKernels
supplies the authoring surface CGGibbs's hand-rolled caching maps onto cleanly; it
does not make a naive dense-GLM Gibbs O(d) automatically.

### The measured result

For a **faithful single-site slice-within-Gibbs**, benchmarked against NUTS
(AdvancedHMC, the same setup the NUTS pages use) on bulk-ESS/second over the same
logistic posterior, NUTS wins across every regime tried:

- at `n=500, d=20` (weak correlation): NUTS **4172** vs CGGibbs **465** ESS/s
  (NUTS ~9×);
- at `n=60, d=200, ρ=0.6`: NUTS **275** vs CGGibbs **~41** ESS/s (NUTS ~6.7×);
- at `n=200, d=100, ρ=0.6`: NUTS **114** vs CGGibbs **~4** ESS/s (NUTS ~25×).

CGGibbs's ESS/second is a **stable rate** (independent of chain length — more of
its cheap sweeps buy proportional ESS *and* time), and it converges very slowly
on correlated posteriors. **NUTS wins on ESS/second by 6–25×**, robustly, even
with CGGibbs given long chains. The O(d) caching is real (a 2.6–3.1× per-sweep
speedup over the naive recompute) but per-sweep *mixing* is the binding
constraint.

This does **not** refute the paper — its reported wins likely rely on adaptation,
blocking, or specific dataset structure not implemented here — but it does refute
the easy story that a reactive compute graph makes Gibbs beat HMC for free. What
ReactiveKernels unambiguously delivered is making the whole comparison **easy to
author, faithful, and rigorous to measure**. The runnable benchmark, its drivers,
and the full write-up are under `benchmark/cggibbs/`.

## A higher-level interface

The `ReactiveKernelsPPLExamples` package layers a concise declarative interface
over these hand-authored kernels — `gibbs(...)` / `Gibbs`, with automatic
conditional derivation, conjugacy detection, and Metropolis-within-Gibbs
fallback (`packages/ReactiveKernelsPPLExamples/src/ppl_gibbs.jl`). The kernels on
this page are the layer it lowers to.
