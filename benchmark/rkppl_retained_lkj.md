# Retained LKJ measurements

Declared LKJ factors now retain one transform/Jacobian/prior program as K
changes. At K=16 the measured native gradient was about 24 times faster and
allocated about 25 times fewer bytes than the opening implementation.
Build allocations fell from 711 MB to 46 MB. Small K retains a fixed
preparation cost: K=2 build allocations increased from 24 MB to 46 MB.

## Method

Synthetic public data, Julia 1.10.12, Enzyme 0.13.209,
DifferentiationInterface 0.7.21 and Reactant 0.2.290 on the shared CPU host
`gordito`, 2026-10-03. No CPU affinity or isolated-host timing guarantee.
`rkppl_retained_lkj.jl` uses three Normal observations with
`mu = L[K,1] .* x`, eta=2, and `u[i] = 0.2sin(i)`.
The density and in-place gradient are warmed, then timed with the minimum
of five batches of 1,000 calls; allocation columns measure one warmed call.
A preliminary K=2 row warms common machinery and is excluded from these
comparisons. Preparation measurements are single calls, include their
allocations, and exclude subsequent evaluation/JIT. First-gradient time
includes first evaluation/JIT after preparation and depends on cache history.
The complete exact measurements, including query and AD preparation bytes
and first-density time, are in `rkppl_retained_lkj.csv`.

Opening code: `78321b3fcbe3bfbc3f668dc57e4ef12d00697b69` in a pristine
scratch checkout and resolved consumer environment. That surface rejected a
shape-only matrix alongside observations. The benchmark's
`LKJ_PLAN_BASELINE=1` path lowers literal K, replaces the two declaration axes
with `size(M,2)` before binding, and executes the opening transform unchanged.
Retained code: `ce9feab47149b9ea41fa5b2bc9bde7855744ba98`, based on integrated
canonical `48bc85695997aa5e124c9162def6a319f2ff15f0`.

Run with a resolved consumer environment containing ReactiveKernelsPPL,
ReactiveKernels, Enzyme and DifferentiationInterface:

```sh
julia --startup-file=no --project=/path/to/consumer-env benchmark/rkppl_retained_lkj.jl
```

## Hot native evaluation

| K | Recipes before / retained | Density µs before / retained | Gradient µs before / retained | Density bytes before / retained | Gradient bytes before / retained |
|---:|---:|---:|---:|---:|---:|
| 2 | 12 / 10 | 0.498 / 0.097 | 1.836 / 0.303 | 448 / 240 | 2064 / 1680 |
| 4 | 22 / 10 | 0.742 / 0.295 | 2.853 / 0.446 | 1072 / 384 | 3536 / 1968 |
| 8 | 66 / 10 | 3.236 / 0.922 | 43.525 / 1.350 | 4672 / 944 | 54224 / 3088 |
| 16 | 250 / 10 | 12.171 / 4.045 | 164.321 / 6.888 | 17376 / 3344 | 199072 / 7888 |

## Preparation and first evaluation

| K | Build s before / retained | Build MB before / retained | Query prepare s before / retained | AD prepare s before / retained | First gradient s before / retained |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.313 / 0.453 | 24.0 / 45.6 | 0.464 / 0.472 | 0.233 / 0.326 | 0.170 / 0.286 |
| 4 | 0.505 / 0.438 | 39.8 / 45.6 | 0.898 / 0.486 | 0.372 / 0.295 | 0.661 / 0.284 |
| 8 | 2.033 / 0.505 | 204.1 / 45.7 | 0.994 / 0.489 | 0.950 / 0.310 | 4.548 / 0.414 |
| 16 | 6.320 / 0.588 | 710.8 / 45.7 | 2.663 / 0.583 | 3.482 / 0.303 | 149.272 / 0.256 |

## Structure and acceptance

Prepared RK recipe counts are constant at 10 for this model. Primal
StableHLO operation inventories match at K=2/4/8. Under the default optimized
reverse pipeline, K=8/16 both contain 456 StableHLO operations, 13 retained
while loops and 8 conditional regions. Tensor shapes and tape capacities
specialize; factor-entry bodies do not replicate in the lowering.

The transform keeps each original left-associated product. Its inner loops
use prepared bounds and lazy triangular guards, so nested reverse tapes have
fixed capacity and unused partials remain unread. Packing follows Stan's
column blocks; normalization constants are computed at preparation.
The arithmetic remains cubic to preserve product association. No extra
`Val{K}` dispatch, generated shape implementation, cache or derivative rule
was added. The measured ordinary preparation/JIT path supports that choice.
Legacy structural-margin LKJ blocks retain their existing scalar graph.

Final acceptance passed 331 assertions: 227 native/conditioned LKJ checks,
60 compiled/axis/reverse-structure checks, and 44 authored-loop checks.
Tests cover both matrix size axes, literal declarations, exact native
transform/Jacobian parity, Distributions density oracles, independent
closed-form gradients, and finite differences of a weighted sum over every
factor entry. The focused array/value/LKJ regression run also passed 383
assertions while preserving 11 existing capability pins.

K=1 has zero coordinates, `[1.0;;]`, compiled density, and native empty
gradients. Reactant 0.2.290 cannot export its empty AD buffer (`tensor.empty`)
for compiled reverse; one acceptance pin records that exact existing boundary.
Canonical `repro_reactant_empty_gradient.jl` isolates it without RK imports,
and `docs/src/constraints.md` documents it. The reduced `only_enzyme` pipeline
also leaves unsized tapes; correctness and reverse scaling use the default
pipeline that compiles successfully.

Durable command receipts: final acceptance/retained measurement
`kb-run-compact.dxHHLf` (exit 0, 265 s), and focused regressions/opening
measurement `kb-run-compact.bhURSK` (exit 0, 599 s), in the assigned managed
scratch directory. The source and CSV above preserve the reviewable result.
