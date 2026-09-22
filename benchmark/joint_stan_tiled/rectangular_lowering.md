# Rectangular recurrence lowering

The grouped PPL emitter already emits a constant number of Julia statements,
but its host loops expand once per subject and operation during tracing. The
first implementation moves those recurrences behind a general RK runtime
boundary, `_rectangular_fold(step, init, columns, shared, marker)`.

**Status:** joint primal proof only; PK reverse compilation is blocked in
Enzyme/MLIR. The PK adapter is disabled by default. Set `RK_RECTANGULAR=1` in
`bench_rkppl_reactant.jl` to opt in for measurements. This is not a supported
sampler configuration or a demonstrated runtime improvement. TGI nadir uses the
new path automatically and passes reverse parity. CPU fusion stays enabled.

## Representation and lowering

`columns` is a tuple of equally sized, typed vectors: a rectangular table
without forcing integer indices and floating point data to share an element
type. `step(carry, row, shared...)` returns the next carry. The return value is
the final carry, including any fixed-size output buffers the caller allocated.

The native method runs an ordinary Julia loop. The Reactant extension promotes
host columns, carry storage, and shared arrays to traced values before a single
fixed-trip `@trace for`. The induction variable always advances by one toward
the static row count. Early completion is represented by inactive state updates;
it never changes the trip count. Scalar wrapper copies preserve separate carry
identities. `_recurrence_branch` retains lazy branches so an inactive singular
or overflowing transition is not evaluated.

Logical lengths that vary at runtime must fit a fixed capacity: include an
active column or compare the row index with the logical length, and freeze
inactive carry updates. The present PK adapter instead packs the known bound
ragged schedule exactly, so it executes no padding rows. The primitive requires
one-based vector columns and rejects inconsistent lengths before lowering.

This slice needs no `Recipe`, `Plan`, or authored-scan IR changes: the existing
grouped cell is a recipe operation and the experimental adapter calls the runtime
boundary when its parameters are traced. `scan` continues returning scalar per-step outputs on its
existing Reactant paths. A future authored final-carry surface can lower to the
same boundary; no public syntax is added here.

## PK and TGI adapters

PK uses a flat operation table with subject id, reset flag, and concentration/AUC
destination indices. Unequal subject lengths therefore need neither padded
operations nor dynamic slices. The carry is the three compartment amounts,
cumulative administered amount, and a fixed output buffer. Read destinations
preserve each subject's `[concentration; AUC]` ordering. Subject parameters are
gathered by subject id; event bioavailability is gathered by global operation id.

Repeated-dose segments retain the original affine binary-power calculation in
a second fixed-trip loop. Its capacity is the bit length of the largest bound
dose count; exhausted exponents freeze their state. The loop body count is
independent of subjects and operations. The small fixed 3×3 and 4×4 tuple math
remains scalar code to preserve the existing arithmetic order.

TGI nadir uses one row per assessment and a reset flag for the first assessment
of each nonempty subject. Empty segments add no rows. Each step stores the
previous nadir before incorporating the current assessment.

Bound shapes still determine compilation shapes and tape capacity. Changing a
bound schedule requires preparing/compiling again. The aim is constant program
structure, not a shape-polymorphic executable or constant total tape memory.

## Acceptance

`test/test_rectangular_fold_reactant.jl` checks fixed loop count, primal parity,
native/traced reverse parity, lazy branches, and input preservation.
`packages/ReactiveKernelsPPL/test/test_pk_rectangular.jl` checks unequal subject
ranges and repeated-dose segments. `test_nadir_rectangular.jl` checks empty
segments, output-before-update semantics, and reverse parity.

Joint K=1 then K=3 measurements must report unoptimized StableHLO loop/operation
counts, compile wall time and peak process RSS, synchronized resident-input
evaluation time, and primal/reverse parity against native execution. Default
CPU fusion is the acceptance path. No K=10 measurement precedes stable K=3.

## Reverse blocker

Tracked in [ReactiveKernels #13](https://github.com/nsiccha/ReactiveKernels.jl/issues/13),
including the full standalone reproducer and environment setup.

Reactant 0.2.285 reports `had set op which was not a direct descendant` while
processing an `enzyme.set` in the PK recurrence. The joint K=1 reverse process
exited 1 at 1.98 GiB peak RSS; it did not exhaust memory. The standalone
`repro_rectangular_reverse.jl` reproduces the same failure with two subjects,
six operations, and no PPL generator or likelihood. Generic fold reverse,
nadir reverse, and smaller nested-loop/conditional controls compile correctly;
the evidence does not establish a general nested-loop limitation.

The forward recurrence is expressible without a new public RK primitive.
The outstanding requirement is a working reverse of this fixed-trip, lazy
conditional recurrence with indexed parameter reads and fixed output storage.
A backend fix or a separately validated reverse rule is needed before enabling
the PK adapter. K=3 reverse and K=10 remain unattempted; fused gradient runtime
and full AD parity are unknown.

## Measured on strato2, 2026-09-22

Julia 1.10.11, Reactant 0.2.285, Enzyme 0.13.204, Reactant_jll 0.0.407+0;
CPU backend, default fusion, synchronized compilation and resident inputs.
Machine-readable receipts are in `rectangular_results.json`.

| Metric | K=1 primal | K=3 primal |
| --- | ---: | ---: |
| Unoptimized HLO while / if count | 2 / 5 | 2 / 5 |
| Unoptimized HLO operation occurrences | 10,476 | 9,911 |
| HLO text bytes | 803,439 | 791,773 |
| Cold HLO trace (s) | 105.28 | 118.53 |
| Subsequent compile (s) | 12.8 | 14.0 |
| Whole process wall (s) | 196.70 | 224.26 |
| Peak process RSS (GiB) | 1.81 | 1.76 |
| Reactant / native primal evaluation (ms) | 0.1994 / 0.0204 | 0.5230 / 0.0364 |
| Relative primal error at seeded point | 1.882e-16 | 8.755e-16 |

Both runs also pass parity at a second point. The small variation in total
operation count does not grow with K. This is evidence for bounded recurrence
structure at these two shapes, not a proof of constant HLO size for every model.
Cold HLO tracing warms Julia before the reported compilation; **12.8/14.0 s are
not standalone cold compile times**. Peak RSS and whole-process wall include
native model construction, tracing, compilation, and evaluation.

K=1 reverse exits 1 after 389.94 s whole-process wall at 2,071,772 KiB peak RSS.
Native Enzyme evaluation is 0.1739 ms, but no compiled gradient exists to time or
compare. The primal is still about 10–14 times slower than native primal here.
These measurements therefore do not establish the requested fused-equivalent
runtime outcome.

Reproduce the successful primal runs in an environment developing this root and
its `ReactiveKernelsPPL` and `ReactiveKernelsDistributionKernels` subpackages:

```sh
TILE_K=1 RK_RECTANGULAR=1 RK_HLO=1 RK_PRIMAL=1 RK_GRAD=0 REPS=50 \
  /usr/bin/time -v julia --startup-file=no --project=<env> \
  benchmark/joint_stan_tiled/bench_rkppl_reactant.jl
# Then repeat with TILE_K=3 after K=1 succeeds.
```

Run `repro_rectangular_reverse.jl` directly in the same environment for the
focused expected failure. Do not launch the full joint reverse or K=3 reverse
again until that reproducer compiles and its gradient matches a native oracle.

`repro_rectangular_reverse_standalone.jl` contains the recurrence helpers inline
and needs only the public PK math. It was also verified against the original RK
baseline `9b129af9fd63106a34a43e35deae4d91a17a9181`: same diagnostic, exit 1,
81.71 s whole-process wall, 1,342,016 KiB peak RSS, including first-use loading.
This version can be shared upstream without publishing the experimental branch.

The final branch acceptance batches pass 582 assertions covering the generic
fold, lazy branches, nadir reverse, existing authored scan, native joint parity,
PK cells, and joint emitter native/compiled AD. The MutatingFunctions extension
also loads successfully. These checks validate the supported paths; the PK
reverse reproducer remains an expected failure, outside the test suite.
