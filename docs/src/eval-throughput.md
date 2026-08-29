# Evaluation throughput vs Turing.jl

This page compares **evaluation** throughput — how fast a log density is
evaluated, differentiated, and turned into a generated quantity — between a
ReactiveKernels `@kernel` and the equivalent `Turing.jl` model. It is **not** a
sampler benchmark: it measures per-call primal / gradient / generated-quantity
cost, not NUTS adaptation, draws, wall time, or ESS. For the sealed-native NUTS
artifact and its separate, work-normalized inner-loop receipt see
[NUTS sampling](nuts.md); end-to-end sampler speed against AdvancedHMC/DynamicHMC
remains unmeasured and nothing here should be read as such.

Both sides evaluate the **same** model — an iid Normal log density of a position
vector `x` given fixed `(μ, log σ)` — and every value is parity-checked against
the closed form before timing. Three modes are timed, each **with and without
Reactant** compilation:

- **`primal`** — the scalar log density. ReactiveKernels evaluates
  `prepare(model; want = :logdensity)`; Turing uses `LogDensityProblems.logdensity`.
- **`gradient`** — ∂ log density / ∂x, via one shared reverse-mode Enzyme backend
  through DifferentiationInterface on both sides.
- **`gq`** — the pointwise log densities (a vector). ReactiveKernels evaluates
  `prepare(model; want = :pointwise)`; Turing uses `DynamicPPL.generated_quantities`.

The gradient uses one shared reverse-mode Enzyme backend through
DifferentiationInterface on both sides; the benchmark asserts that the Enzyme
backend is actually the one exercised (a correct gradient alone does not prove
which backend produced it). Reactant device transfers and compilation happen
outside the timed region; only the compiled call is timed, synchronously.

## Reactant support

ReactiveKernels lowers a log density to a straight-line array program, so
Reactant can `@compile` the primal and the generated quantities and Enzyme can
differentiate it under Reactant. Turing's DynamicPPL evaluation re-runs the model
rather than being a straight-line array program, so it is **not
Reactant-traceable**; those cells are reported unsupported rather than faked.

## Results

Median per-call time on a **log time axis — lower (shorter) is better**. Each bar
is read straight from the static receipt
[`benchmark/receipts/eval-throughput-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/eval-throughput-v1.toml)
(default CPU backend, `Float64`) while the docs build, so the picture cannot drift
from the numbers. `Turing + Reactant` is unsupported for every mode and is
omitted.

```@eval
Main.ReactiveKernelsDocs.eval_throughput_chart()
```

## What the numbers say

- **ReactiveKernels native is the fastest path at every size and mode**, roughly
  2.5–5.7× faster than the equivalent Turing evaluation. The fused straight-line
  kernel avoids the per-call model-evaluation overhead of DynamicPPL.
- **Reactant is a net loss for small CPU workloads.** Its per-call device
  dispatch/sync is a large fixed cost (~3 µs), so at `n = 16` the compiled path
  is ~30–80× slower than native. This is expected: Reactant targets large XLA
  fusion and accelerators, not tiny CPU kernels.
- **That fixed cost amortizes as the work grows.** By `n = 4096` the Reactant
  gradient (≈11.6 µs) is within ~1.2× of RK native and **faster than the Turing
  native gradient** (≈27.2 µs); the trend points to Reactant winning at larger
  sizes or on a GPU. This receipt uses the default CPU backend and does not
  measure an accelerator.
- Numbers are one machine's CPU receipt; treat the **ratios**, not the absolute
  nanoseconds, as the portable result.

## Reproduce

```julia
julia --startup-file=no benchmark/eval_throughput_comparison.jl \
  --output=benchmark/receipts/eval-throughput-v1.toml
```

The script provisions a fresh, pinned environment (Reactant 0.2.278, Turing
0.47.1, DynamicPPL 0.42.6, Enzyme 0.13.199, DifferentiationInterface 0.7.21) that
never enters ReactiveKernels' root dependencies. A quick smoke run:
`RK_EVAL_SIZES=16,64 RK_EVAL_ROUNDS=20 julia --startup-file=no benchmark/eval_throughput_comparison.jl`.
