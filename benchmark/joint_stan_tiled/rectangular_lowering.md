# Rectangular recurrence lowering

The grouped PPL emitter already emits a constant number of Julia statements,
but its host loops expand once per subject and operation during tracing. The
first implementation moves those recurrences behind a general RK runtime
boundary, `_rectangular_fold(step, init, columns, shared, marker)`.

**Status:** joint primal proof on released toolchains. The PK reverse needs two
Enzyme-JAX fixes that are not released yet (see "Reverse status"). With both,
on the CI-built Reactant_jll of Enzyme-JAX PR #3241, the joint K=1 and K=3
gradients match native. The PK adapter is disabled by default. Set
`RK_RECTANGULAR=1` in `bench_rkppl_reactant.jl` to opt in for measurements. This is not a supported
sampler configuration or a demonstrated runtime improvement. Ordinary traced PK
calls now fail explicitly with the issue link; they never fall back to the
unrolled host recurrence. Native PK execution is unchanged. Both direct and
segmented TGI nadir calls use the retained path. CPU fusion stays enabled.

## Required semantics

The repository-wide [core constraints](../../docs/src/constraints.md) apply:
retain data-derived iteration and required lazy control flow. Here this includes
subject/operation counts and the binary-power capacity computed from maximum
dose count. The proposed predicated step with a data-derived unrolled power
chain was withdrawn and has not been adopted.

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

TGI interval and observation likelihoods use the same lazy scalar branch
boundary. Empty intervals return `-Inf` without evaluating `log_diff_exp`;
no dummy operands are substituted. Category and censoring decisions likewise
evaluate only their selected likelihood.

Bound shapes still determine compilation shapes and tape capacity. Changing a
bound schedule requires preparing/compiling again. The aim is constant program
structure, not a shape-polymorphic executable or constant total tape memory.

## Acceptance

`test/test_rectangular_fold_reactant.jl` checks fixed loop count, primal parity,
native/traced reverse parity, lazy branches, and input preservation.
`packages/ReactiveKernelsPPL/test/test_pk_rectangular.jl` checks unequal subject
ranges and repeated-dose segments. `test_nadir_rectangular.jl` checks empty
segments, output-before-update semantics, and reverse parity.

The follow-up constraint tests also cover direct helper calls. Traced array
wrappers expose their parent storage to RK's general backend-marker discovery,
so views cannot silently choose the native host loop. Direct PK cells use the
same experimental adapter as grouped calls; without the diagnostic opt-in,
both reject compilation with issue #13 instead of tracing an unrolled fallback.
Standalone affine powers retain their binary-power loop as well.

| Focused unoptimized StableHLO check | Data sizes | Loops | Operation occurrences |
| --- | --- | ---: | ---: |
| Ragged PK recurrence | 2 / 6 subjects, repeated operation schedule | 2 / 2 | 5,934 / 5,934 |
| Standalone affine power | exponents 3 / 31 | 1 / 1 | 303 / 303 |
| Direct TGI nadir | 3 / 9 assessments | 1 / 1 | 63 / 63 |

`test_tgi_control_flow.jl` checks lazy interval branches and their reverse at
ordinary, empty, reversed and clamped intervals, plus direct nadir and traced
views. Nadir gradient comparisons avoid ties at the running minimum, where the
derivative is undefined. The ordinary TGI fixture retains its native/reference
and compiled-gradient checks for all observation families.

The constraint-repair acceptance passes 858 assertions in four ordered batches:
TGI (256), PK cells and retained powers (270), focused control-flow plus core
fold tests (45), and joint emission/parity/default compilation checks (287).
Both Reactant and MutatingFunctions extensions were loaded. Process receipts
and package versions are recorded in `constraint_results.json`; these are
acceptance costs, not new joint-model performance measurements.

Joint K=1 then K=3 measurements must report unoptimized StableHLO loop/operation
counts, compile wall time and peak process RSS, synchronized resident-input
evaluation time, and primal/reverse parity against native execution. Default
CPU fusion is the acceptance path. No K=10 measurement precedes stable K=3.

## Reverse status

Two Enzyme-JAX defects stood between the retained PK recurrence and a correct
compiled gradient. Both are backend fixes; neither is in a Reactant_jll release
yet, so the PK adapter stays opt-in (`_rectangular_pk_enabled`). Both fixes
together pass on a real CI-built binary (see "Reverse acceptance on the #3241
CI build").

1. **Reverse fails to compile** (`had set op which was not a direct
   descendant`): [ReactiveKernels #13](https://github.com/nsiccha/ReactiveKernels.jl/issues/13),
   reproducer `repro_reactant_while_reverse.jl`. Fixed by EnzymeAD/Enzyme-JAX
   #3240 (the if/case removers erase the branch ops they hoisted). The PR's
   CI-built Reactant_jll passes that reproducer.
2. **Reverse compiles but is silently wrong** once (1) is fixed: the reverse of
   a `stablehlo.if`/`stablehlo.case` read its result adjoint in the branches
   without zeroing it. An if nested in a branch of another if inside a loop —
   the PK `count > 1` branch inside the read/dose branch — then reuses an
   earlier iteration's adjoint. The primal is exact; subject 1's
   rate-parameter derivatives were 6–59% off in `repro_rectangular_reverse.jl`.
   Reproducer `repro_nested_if_reverse.jl` (Reactant and Enzyme only:
   -11.208 instead of -17.273). Fixed by zeroing the adjoints in front of the
   reverse op, as the `scf.if` reverse already does: EnzymeAD/Enzyme-JAX
   #3241 (stacked on #3240; commit `85c34a10`, the same change as the earlier
   local `a3c08614` with a project-neutral test).

The measurements under "Preliminary reverse measurements" below were taken
with fix (1) compiled and fix (2) applied to the post-AD IR by emulation. The
section "Reverse acceptance on the #3241 CI build" repeats the parity checks
with both fixes compiled. Neither is a released toolchain.

### History: the original blocker

Tracked in [ReactiveKernels #13](https://github.com/nsiccha/ReactiveKernels.jl/issues/13),
with an RK-free upstream reproducer in `repro_reactant_while_reverse.jl`.

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
focused check. On a released toolchain it still fails as described above; with
both fixes its gradient matches native (see "Reverse status").

`repro_reactant_while_reverse.jl` reduces the same diagnostic to one scalar
conditional sum: a traced loop over six values, with a lazy branch that skips
the first. It imports only Reactant and Enzyme. There are no RK helpers, PK
equations, operation tables, nested loops, or output buffers. Primal compilation
returns the correct sum; reverse compilation fails before producing a gradient.
This replaces the larger RK-dependent example, which was not a minimal upstream
reproducer.

The exact file was verified in fresh public-only environments containing no RK
package or developed path. Reactant 0.2.285 / Reactant_jll 0.0.407+0 exits 1 after
24.90 s at 1,123,152 KiB peak RSS. Reactant 0.2.286 / Reactant_jll 0.0.408+0 also
fails with the same diagnostic (22.39 s; 1,123,292 KiB). Both use Enzyme 0.13.204
and Julia 1.10.11. Related three-element conditional and six-element branch-free
controls compile with correct gradients. These controls narrow the observed
failure; they do not establish that every loop containing a branch fails.

To share the reproducer, copy that single Julia file and create a public-only
environment; no RK checkout or experimental branch is required:

```sh
julia --startup-file=no --project=mwe-env -e 'using Pkg; Pkg.add([
    PackageSpec(name="Reactant", version="0.2.286"),
    PackageSpec(name="Enzyme", version="0.13.204"),
    PackageSpec(name="Reactant_jll", version="0.0.408")])'
julia --startup-file=no --project=mwe-env repro_reactant_while_reverse.jl
```

The initial fold/nadir slice passed 582 assertions covering the generic
fold, lazy branches, nadir reverse, existing authored scan, native joint parity,
PK cells, and joint emitter native/compiled AD. The MutatingFunctions extension
also loaded successfully. The later constraint-repair acceptance is the
858-assertion result above. These checks validate the supported paths; the PK
reverse reproducer remains an expected failure, outside the test suite.

## Reverse acceptance on the #3241 CI build, 2026-09-23

Not a released toolchain. Setup:
- Reactant.jl `main` (0.2.287) with Enzyme 0.13.204, Julia 1.10.11.
- `libReactantExtra` from the `Build Reactant_jll` CI run of Enzyme-JAX PR #3241 (run `35803008630`, head `85c34a10` = #3240 + fix 2, both compiled).
- CPU backend, default fusion, synchronized, `RK_RECTANGULAR=1`. Record: `rectangular_results.json` key `ci_binary_3241_2026_09_23`.

| Check | Result |
| --- | --- |
| `repro_nested_if_reverse.jl` | -17.273417152590273 (exact -17.27341715259027) |
| `repro_rectangular_reverse.jl` | all 10 gradient entries match native |
| Joint K=1 gradient parity vs native (max rel) | 5.6e-14 |
| Joint K=3 gradient parity vs native (max rel) | 1.3e-14 |
| Joint K=1 / K=3 primal relative error | 1.9e-16 / 1.5e-16 |

strato2 was heavily shared during these runs (native gradient 0.78–0.82 ms instead of 0.20/0.49 ms), so this section records correctness only. The runtime numbers remain those below.

## Same-arithmetic exponential hoisting, 2026-09-24

The rectangular path now computes `exp(A * dt)` for every row that needs it
before entering the retained recurrence. The rows are array lanes through the
existing `_pk_expm3`, so Padé selection, LU pivoting, squaring, and per-entry
operation order are unchanged. The sequential loop only selects the row's nine
matrix entries and applies the resulting 3×3 matrix to the carried state.
Repeated-dose segments use the same table for `exp(A * interval)`.

The row sets come from the bound schedule and preserve the existing lazy
branches: rows whose propagation or repeated-dose branch is not taken are not
evaluated. This is a layout change, not data-derived unrolling or a replacement
of the recurrence by superposition.

The focused two-subject, 48-operation benchmark on the #3241 CI binary above
measured:

| Default CPU fusion | Before hoisting | Hoisted | Native |
| --- | ---: | ---: | ---: |
| primal per row | 2.57 µs | 0.95–1.07 µs | 0.34–0.36 µs |
| gradient per row | 241.4 µs | 4.73–5.23 µs | 2.33–2.35 µs |

Thus hoisting improves the compiled primal by about 2.5–2.7× and the compiled
gradient by about 46–51×. At this short schedule the compiled path remains
about 2–3× slower than native; it is not evidence that native execution should
be removed. The retained HLO stayed structurally invariant when the test
schedule grew from two to six subjects (28,531 unoptimized StableHLO operation
occurrences in both cases).

Correctness checks covered the original reproducer, single doses, and repeated
doses over two subjects. Compiled gradients matched native Enzyme to
`9.1e-15` or better with no NaNs. The targeted rectangular suite passed 17/17,
including a same-time dose/read schedule where both exponential tables are
absent.

The optional `xla_cpu_use_multi_output_fusion` flag improved the hoisted primal
to 0.73 µs/row, but its gradient aborted inside XLA symbolic-map composition
on this old CI binary (`GetNumDims() == other.GetNumResults()`, 2 vs 1). Default
fusion is the validated path; MOF is neither required nor enabled.

## Preliminary reverse measurements (emulated fix), 2026-09-22

Not a released toolchain. Setup:
- Reactant.jl `main` (0.2.287) with Enzyme 0.13.204, Julia 1.10.11.
- `libReactantExtra` from the `Build Reactant_jll` CI run of Enzyme-JAX PR #3240 (fix 1, compiled).
- Fix 2 applied by rewriting the post-AD IR exactly as the patched reverse emits it. Each reverse `stablehlo.if`/`case` reads and zeroes its result adjoints before branching; the rest of the shipped pipeline then runs.
- CPU backend, default fusion, synchronized, resident inputs, `RK_RECTANGULAR=1`, the same seeded point as above.

| Metric | K=1 | K=3 |
| --- | ---: | ---: |
| Unoptimized HLO while / if count | 2 / 10 | 2 / 10 |
| Unoptimized HLO operation occurrences | 9,463 | 9,463 |
| Primal compile (s, after HLO trace) | 13.0 | 12.4 |
| Relative primal error | 1.9e-16 | 1.5e-16 |
| Value+gradient compile (s) | 118.2 | 114.5 |
| Gradient parity vs native Enzyme (max rel) | 5.6e-14 | 1.3e-14 |
| Peak process RSS (GiB) | 5.4 | 5.3 |
| Reactant / native primal evaluation (ms) | 0.193 / 0.0145 | 0.460 / (noisy) |
| Reactant / native gradient evaluation (ms) | 7.98 / 0.197 | 39.4 / 0.495 |

Findings:
- **Control:** with fix 1 only (the shipped CI library, no IR rewrite), the K=1 gradient compiles (159 s) but fails parity at max rel 1.1e-2. Fix 2 is required for the joint model, not only for the PK reproducer.
- **Invariance:** the unoptimized HLO is the same size at K=1 and K=3, and compile time and memory do not grow with K. The unrolled program's gradient compile was killed at 13.3–14.2 GB.
- **Runtime:** the compiled gradient is 40–80× slower than native Enzyme on CPU, so fused runtime is still not established.

The focused PK reverse over seven schedules also matches native to ≤ 6.2e-15:
- the reproducer;
- dose counts 2/3/5/8;
- three ragged subjects with interleaved dose segments;
- a subject without doses.

Its compiled gradient takes 2.2–4.2 ms against 29–150 µs native. Across those schedules the unoptimized primal HLO is the same (5,948 ops, 2 while, 5 if; one op fewer when no dose segment exists). The optimized primal and gradient are identical across subject and operation counts. They change only with the static power-bit capacity, because the optimizer unrolls that small constant-trip loop.
