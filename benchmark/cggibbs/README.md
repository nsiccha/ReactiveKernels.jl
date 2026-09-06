# CGGibbs on ReactiveKernels — logistic regression (arXiv 2410.03630)

Reproduces the model + sampler of *"Is Gibbs sampling faster than Hamiltonian
Monte Carlo on GLMs?"* (arXiv 2410.03630) on ReactiveKernels, per user decision
`ReactiveKernels:sampling:gibbs/decisions/2026-09-06T19-03-53-605-1i097kg` (GO).

The paper's **CGGibbs** ("compute graph Gibbs") makes a coordinate Gibbs sweep
**O(nd)** instead of **O(nd²)** by caching the linear predictor `η = Xθ` and
updating it rank-1 when one `θⱼ` changes, with **slice sampling** within each
coordinate (GLMs aren't conjugate).

Per the user's steer, the rk kernels here are **authored directly** (not via the
`@ppl` macro) — the `@ppl` macro is a future concise front-end that should lower
*to* kernels shaped like these.

## Status

- **Increment 1** (`cggibbs.jl`): correct CGGibbs coordinate slice sampler with the
  cached linear predictor, on synthetic logistic data. The per-coordinate
  conditional log-density is authored as an rk graph/kernel.
- **Increment 2** (`cggibbs_vs_nuts.jl`): the paper's headline head-to-head —
  CGGibbs vs **NUTS** (AdvancedHMC, repo-faithful setup) on bulk-ESS/second. See
  **Head-to-head** below — the honest result is that single-site CGGibbs *loses*
  to NUTS in this synthetic regime.

Not yet: an optimized (non-allocating) conditional eval + blocking, the reusable
`ReactiveKernelsSamplers` extraction, and the paper's real datasets (see **Next**).

## Run

```sh
julia --startup-file=no benchmark/cggibbs/setup.jl
julia --startup-file=no --project=benchmark/cggibbs benchmark/cggibbs/cggibbs.jl
julia --startup-file=no --project=benchmark/cggibbs benchmark/cggibbs/cggibbs_vs_nuts.jl
```

## Measured (this environment)

- **Correctness** — well-identified case (n=2000, d=12, 4 active): posterior means
  track the truth (true ±1.2 → post-means 1.12…−1.19), max |β̂| off-active 0.12.
- **Cache is provably correct** — the cached (CGGibbs) and naive-recompute chains
  are **bit-identical** under the same RNG: caching changes only *cost*, never the
  numbers.
- **Scaling** — median ms/sweep (n=400), CGGibbs vs naive `η=Xθ` recompute; the
  ratio grows with `d`, confirming O(nd) vs O(nd²):

  | d | CGGibbs | naive | naive/CGGibbs |
  |--:|--:|--:|--:|
  | 16 | 0.61 | 0.64 | 1.05× |
  | 64 | 2.52 | 2.68 | 1.06× |
  | 256 | 8.35 | 10.77 | 1.29× |
  | 1024 | 32.1 | 67.0 | 2.09× |
  | 2048 | 55.4 | 172.2 | 3.11× |

  The win is modest at small `d` because the shared per-coordinate conditional
  evaluation (an allocating rk kernel call, O(n)) dominates the cache saving;
  it grows as the O(nd²) predictor recompute takes over.

## Head-to-head vs NUTS — bulk-ESS/second (`cggibbs_vs_nuts.jl`)

The paper's actual question. NUTS is AdvancedHMC 0.8.6 with the repo's setup
(`MultinomialTS` + `GeneralisedNoUTurn` + `StanHMCAdaptor`); ESS is
`MCMCDiagnosticTools` bulk ESS; both sample the same logistic posterior
(2000 draws, 1000 warmup, weakly-correlated synthetic X).

| n | d | sampler | min ESS | sec | min-ESS/s | vs NUTS |
|--:|--:|--|--:|--:|--:|--:|
| 500 | 20 | NUTS | 2494 | 0.6 | 4172 | — |
| 500 | 20 | CGGibbs | 822 | 1.8 | 465 | 0.11× |
| 500 | 100 | NUTS | 917 | 1.2 | 776 | — |
| 500 | 100 | CGGibbs | 196 | 7.4 | 26 | 0.03× |
| 400 | 300 | NUTS | 1669 | 46 | 36 | — |
| 400 | 300 | CGGibbs | 6 | 10 | 0.6 | ⚠ not converged |

**Honest reading — single-site CGGibbs loses to NUTS here.** Two distinct causes,
one algorithmic and one implementational:

- **Mixing (algorithmic).** Coordinate-at-a-time Gibbs mixes worse than NUTS on
  correlated logistic posteriors (≈3× fewer ESS at d=20), and at d=300 it
  collapses (ESS 6, posterior means disagree with NUTS by ~4 → not converged).
  The remedy is **blocking** (update correlated groups jointly), which the paper's
  winning regimes rely on.
- **Speed (implementational).** CGGibbs is also ~3× slower per unit here because
  the conditional eval is an allocating rk kernel. An optimized (non-allocating)
  eval would roughly close the *speed* gap at low d — at d=20, 822 ESS in ~0.18 s
  would be ≈NUTS — but not the mixing gap.

So this is **not** "RK makes Gibbs beat HMC". It shows RK can run a faithful
head-to-head, and that a naive single-site CGGibbs on weakly-correlated data is
the wrong end of the paper's story. Whether CGGibbs wins requires the paper's
regime (real GLM datasets), an optimized eval, and blocking — the next steps.

## Honest note on "RK gives CGGibbs for free"

The earlier assessment (brief `2026-09-06T18-26-43-851-ksq7f2`) said RK's reactive
layer *is* CGGibbs. Refined by building it:

- **Block-level** incrementality (different Markov blankets per block — e.g. the
  SSVS PoC's `z`/`ω` not touching the data) **is** free from rk's pure reactive
  provenance (`set!`/`get!`).
- **Coordinate-level** CGGibbs on a dense GLM is different: rk's pure reactive
  layer recomputes at **recipe granularity**, so a dense `η = Xθ` recipe is O(nd)
  *per coordinate* — i.e. the naive cost, not the rank-1 update. So the rank-1 `η`
  cache is **authored** (here as an explicit buffer). The declarative route inside
  rk is the **stateful `mutate!`/`touch!` layer** (a rank-1 in-place update as a
  compiled reactive transition) — the next thing to demonstrate.

So RK doesn't make a naive dense-GLM Gibbs O(d) automatically; it gives the
authoring surface (the conditional kernel + the stateful incremental cache) that
CGGibbs's hand-rolled caching maps onto cleanly.

## Where this fits the kernel split (decision `1n5b20l`, Option 1)

The **slice-within-Gibbs sweep** is the model-agnostic kernel → to be extracted
into the new `packages/ReactiveKernelsSamplers`. The **conditional log-density
kernel** is the model-coupling (here logistic; from the PPL layer in general).

## Next

Reordered by what the head-to-head showed matters most:

1. **Blocking** — the algorithmic gap. Update correlated coefficient groups
   jointly (a block conditional handle, HMC/slice within each block) so mixing is
   competitive at higher d. This is the model-agnostic kernel work.
2. **Optimize the conditional eval** — rk non-allocating / prepared path, and the
   stateful `mutate!` rank-1 `η` cache; closes the *speed* gap.
3. **The paper's real datasets** (gene-expression, colon cancer 62×2000, Guyon) —
   the regimes where CGGibbs is reported to win; re-run the head-to-head there.
4. Extract the generic Gibbs kernel (sweep + within-block transitions +
   conjugate-draw) into `packages/ReactiveKernelsSamplers` (decision `1n5b20l`).
5. Optionally add **Stan NUTS** via BridgeStan as a further baseline.
