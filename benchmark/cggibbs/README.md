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

**Increment 1 (this commit):** correct CGGibbs coordinate slice sampler with the
cached linear predictor, on synthetic logistic data. The per-coordinate
conditional log-density is authored as an rk graph/kernel (prepared once,
evaluated by the slice sampler). Not yet: the reusable-package extraction, the
real datasets, and the Stan-NUTS comparison (see **Next**).

## Run

```sh
julia --startup-file=no benchmark/cggibbs/setup.jl
julia --startup-file=no --project=benchmark/cggibbs benchmark/cggibbs/cggibbs.jl
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

1. Sharpen the conditional eval (rk non-allocating / prepared path) and show the
   stateful `mutate!` rank-1 `η` cache.
2. Extract the generic Gibbs kernel (sweep + slice-within-Gibbs + conjugate-draw)
   into `packages/ReactiveKernelsSamplers`.
3. The paper's 8 public datasets (newsgroups, gene-expression, colon cancer,
   Guyon).
4. Benchmark vs **Stan NUTS** (provisioned BridgeStan): ESS/sec, time-to-median-
   ESS-100.
