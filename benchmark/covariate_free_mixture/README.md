# Covariate-free whole-event mixture allocations

Investigation dated 2026-11-13. Production acceptance across three Enzyme
builds and focused ragged-event integration is complete.

This diagnostic removes the previous-capture covariate from PosteriorDB's
CaptureMtbh model. It is a different model, used to isolate an allocation
regression; it is not a replacement spelling of the original model. The
public fixture contains 146 events, five positions and the reporter's fixed
constrained point (seed 468).

All sampler comparisons use the default nonallocating sampler and ordinary
Enzyme Reverse, with `native_forward_batch=false`. The candidate changes the
shared Bernoulli-logit product-mixture law. A singleton parameter row uses
exact integer success/failure counts; zero-count terms remain lazy. Full
matrices iterate their own row extent and retain per-position evaluation.
Both shape guards remain. There are no custom derivative rules or activity
annotations in the candidate.

The diagnostic itself supplies full parameter rows. Its allocation repair
comes from iterating that row extent directly, without selecting a repeated
row index inside every position. The count expression preserves supported
singleton rows while allowing this iteration scheme. The separate singleton
benchmark measures the additional benefit of reusing Bernoulli log masses.

## Run

Use a resolved Julia consumer environment containing ReactiveKernels,
ReactiveKernelsPPL, ReactiveKernelsDistributionKernels, Enzyme,
DifferentiationInterface, LogExpFunctions, JSON and Chairmarks. Select exact
package/compiler pins in that environment before running; this driver does
not resolve dependencies. Native counting requires Linux/glibc and a C
compiler. Run under the host's normal compute reservation mechanism.
The focused test runner also needs the distribution packages listed in the
DistributionKernels and PPL test environments.

```sh
bench="$PWD/benchmark/covariate_free_mixture"
gcc -std=c11 -O2 -shared -fPIC "$bench/count-allocators.c" -o /tmp/count-allocators.so
gcc -std=c11 -O2 -fno-builtin "$bench/check-allocators.c" /tmp/count-allocators.so -o /tmp/check-allocators
LD_PRELOAD=/tmp/count-allocators.so /tmp/check-allocators
LD_PRELOAD=/tmp/count-allocators.so julia --project=/path/to/env "$bench/plain-rows-write.jl"
EVENT_COUNTS=1,11,146 DUMP_PREFIX=/tmp/mixture \
  LD_PRELOAD=/tmp/count-allocators.so julia --project=/path/to/env "$bench/paired-counts.jl"
LD_PRELOAD=/tmp/count-allocators.so julia --project=/path/to/env "$bench/counts-law-bench.jl"
julia --project=/path/to/env "$bench/focused-tests.jl"
```

`baseline-law.jl` preserves the law at source `f91621d6472664039d6b498247bc97b58a0931f6`
under a separate name. The candidate is the actual production law. The paired
driver prepares independent models, checks
the complete gradients at the fixture point and two changed points, then
alternates baseline/candidate timing order for 20 rounds. It also reports
retained constants, Julia heap bytes, native allocator calls, emitted
structure and shared-host load. Source hashes identify the measured inputs.
`DUMP_PREFIX` saves actual `code_expr` displays using the standard expression
printer with only line-number nodes removed.

`counts-law-bench.jl` compares bound singleton rows, bound full rows and
unbound full rows at 5, 51 and 501 positions. `counts-law.jl` is the historical
prototype retained to identify the earlier results; production drivers do
not load it.
The exact earlier drivers are stored as `results/prototype-*-driver.jl.txt`;
their canonical baseline refers to the pre-edit source named in those results.

The counter measures calls to malloc, calloc, realloc and non-null free on
the armed thread, and requested bytes by allocator. It does not cover every
native allocator. A realloc call need not move storage; requested realloc
bytes are not newly allocated bytes. The C fixture checks exact calls
1/1/1/2 and requests 17/12/33 bytes. Julia heap accounting is separate.

## Production validation

The production edit is based on `0987481070ee9b08b738ef5e201bd2bf843a0d47`
(RK 0.1.225). The measured law source has SHA-256
`2d969a4a5e4a2ac790e82b7f06e94ad1e2cc79535b8b2fb58064405312adcc77`.
On current development Enzyme, the five focused DistributionKernels files
pass 408 checks, including 115 new independent-oracle/structure checks and
158 existing row-law checks. Three inspection helpers now descend into actual
`_PlateProgram.ast` bodies; no existing assertion was relaxed. The weight-log
helper initially failed two checks because it skipped those bodies, then
passed all 65 checks in its file after the traversal repair. The 115 new
checks also pass on old development and registered Enzyme. The PPL whole-event invariants
suite passes 50 checks with one existing expected broken case for invariant
tables whose input is a view. Its ordinary reverse gradients agree with an
independent Julia likelihood, including an empty event collection.
The existing ragged logit/structural-zero testset passes all 19 checks,
including three parameter points and its high-precision likelihood/derivative
oracle. `ragged-logit-tests.jl` selects that testset without changing its model
or assertions. Current development therefore has 477 passing checks and one
existing expected broken case; old development and registered Enzyme each
have 115 passing additional checks, for 707 passes across the three builds.
The ragged run's exit-0 receipt is `results/production-ragged-receipt.txt`.

The production full-model pair passes on all three compiler builds at N=1,
11 and 146, including two changed parameter points and identical complete
gradients. These are actual production results, separate from the earlier
prototype measurements below. Times are microseconds, medians of 20 warmed
sample minima with alternating baseline/candidate order:

| Enzyme line | Baseline calls, N=1/11/146 | Candidate calls, N=1/11/146 | N=146 time | Ratio |
| --- | --- | --- | --- | ---: |
| old development | 14 / 24 / 159 | 15 / 15 / 15 | 23.000 → 19.815 | 0.862 |
| current development | 14 / 18 / 22 | 15 / 15 / 15 | 21.155 → 19.840 | 0.938 |
| registered | 28 / 108 / 1,188 | 31 / 31 / 31 | 43.275 → 18.955 | 0.438 |

Calls are equal malloc/free counts, with zero calloc/realloc. Requested
malloc bytes at N=146 change from 23,600 → 26,520 on old development,
52,808 → 26,520 on current development and 37,344 → 34,424 on registered
Enzyme. Retained constants decrease from 31,012 → 25,658 bytes.
The old build therefore trades 12.4% more requested bytes for fewer calls.
Constant call counts do not imply constant storage, and the candidate does
not recover the historical 12-call floor exactly. At N=1, the old and
registered pairs are respectively 5.1% and 2.5% slower; current development
is 6.0% faster. This is a workload-specific improvement.

Emitted structure stays at 1,653 nodes and 13 loops for the candidate,
versus 1,197 nodes and nine loops for the baseline, across all three N.
Actual one-event and 146-event displays are retained as
`results/production-current-{baseline,counts}-N{1,146}-program.txt`.
Both harness calls still report 96 Julia heap bytes; their source remains
unattributed. These measurements describe warmed repeated evaluation,
not preparation or compilation time. Raw pairs are in
`results/production-{old,current,registry}-paired.txt`.

Validation was performed as separate focused invocations, not as one run of
`focused-tests.jl` or the full repository suite. The first wider current run
was explicitly stopped while compiling the unchanged probability-law model
at the beginning of `test_whole_event_mixture.jl`; its aggregate exit status
is 1, with that interrupted stage and the original inspection-helper failures
retained in `results/production-current-focused-interrupted.txt` and
`results/production-current-receipt.txt`. The helper's corrected file and the
whole-event invariants suite subsequently passed. The unselected probability
and occupancy files are not claimed as verified. The old/registered batch
completed with exit 0 (`results/production-other-receipt.txt`).

The actual production law's direct-gradient controls pass all value and
gradient comparisons. Times are microseconds, medians of 12 alternating
rounds of warmed sample minima:

| Rows and binding | Positions | Baseline | Candidate | Ratio |
| --- | ---: | ---: | ---: | ---: |
| singleton, bound data | 5 | 0.2082 | 0.1251 | 0.601 |
| singleton, bound data | 51 | 1.5112 | 0.1172 | 0.078 |
| singleton, bound data | 501 | 14.785 | 0.1144 | 0.008 |
| full rows, bound data | 5 | 0.2062 | 0.2207 | 1.070 |
| full rows, bound data | 51 | 1.5436 | 1.5664 | 1.015 |
| full rows, bound data | 501 | 22.17 | 22.34 | 1.008 |
| full rows, unbound data | 5 | 0.3226 | 0.3260 | 1.010 |
| full rows, unbound data | 51 | 1.5568 | 1.5445 | 0.992 |
| full rows, unbound data | 501 | 18.045 | 14.060 | 0.779 |

Both bound forms trade two malloc/free calls for four (32 → 36 requested
bytes), with zero calloc/realloc and zero Julia heap bytes. The smallest
bound full-row case is about 7% slower, a 15 ns increase. Unbound full rows
keep four malloc/free calls, 36 requested bytes and 48 observed Julia heap
bytes in both harness calls. Full-row gradients are identical; singleton
maximum absolute error is 4.41e-13 at 501 positions. These direct-law results
are distinct from the full-model allocator regression.

## Prototype results before the production edit

Julia 1.13.1, gordito, one CPU reservation under shared-host load:

| Enzyme line | Enzyme.jl | libEnzyme |
| --- | --- | --- |
| old development | `1ed42fb9984ff3f49a8adcba7732382e0f22c9bb` | `eea6a012bc319b5a20d68a945f816dfc0312e4ab` |
| current development | `7832d492b16bad8862de8fb0d0dcf7200af3a30d` | `e316a076ef438970e339876079acd02ce48d3d21` |
| registered control | Enzyme 0.13.219 | Enzyme_jll 0.0.307+0 |

The registered control uses Enzyme tree
`49a2f5cefe1810e44e898377f4de6e1ec7b97c19`, EnzymeCore 0.8.27 tree
`d88c803e26799193db42e47f6e1e28eeb0322e4d`, Enzyme_jll tree
`5464e555af2b36b35ac10ceda173b35a5eee63ef`, and library artifact
`e1daa6b140cf75505b106c5ec781e57c2beaae0a` (LLVM 20). Supporting developed
packages are MutatingFunctions `02eb981b9770e767f47a7fda287fcbfb9531d587`
and OutputSignatures `f971381f760e91986b3a728a719341e7e7761f55`.

With old Enzyme fixed, the historical source regression is between
`769df47f6155c24ea5e9a54a5b37e9e8010d8497` (12 malloc/free at N=146) and
`b78ce392f5d536b0f6d7a10f45e21b227a8b9ad2` (307). The latter merge introduced
nested bound-cell caching and singleton-row support. Unconstrained points,
values and complete gradients are identical across the bisect points.
The identical point/value/full-gradient dump is retained once as
`results/historical-point-value-gradient.txt`; the three original file hashes
are recorded in `results/historical-gradient-hashes.txt` (all SHA-256
`b7b480f2557ec74e20ca94e766327b9b83c47ed3aab4a660e86da047e644d8be`).

At the refreshed source `95b1ebba1f6ebb90010d2e523f5cc5be216922e2`, RK 0.1.224
and RKPPL 0.1.174, the baseline differs from that historical 307:

| Enzyme line | N=1 | N=11 | N=146 | calloc/realloc |
| --- | ---: | ---: | ---: | --- |
| old development | 14 | 24 | 159 | zero |
| current development | 14 | 18 | 22 | zero |

Entries are equal malloc/free counts. Both baselines retain nine emitted
loops and 1,197 counted expression nodes across these event counts. The
registered control executes successfully at N=146 with 1,188 malloc/free;
its maximum full-gradient difference from the historical reference is
1.78e-15. It is not an unsupported execution case.

The counts candidate gives 15 malloc/free, zero calloc/realloc and 26,520
requested malloc bytes at N=146 on both development builds. Its complete
point, value and gradient are byte-identical to the reference.
At that N the old baseline requests 23,600 bytes and the current baseline
52,808 bytes: the candidate trades 12.4% more requested bytes on the old
build for fewer allocator calls, while reducing bytes on the current build.
Constant call counts would not imply constant storage as the model grows.

Acceptance on refreshed source `f91621d6472664039d6b498247bc97b58a0931f6`
passes all 158 existing row-law checks and 80 additional value, ordinary
reverse derivative and ownership comparisons. The latter reuse prepared
derivatives across empty observations with NaN/Inf logits, saturated logits,
and singleton/full/empty row shapes. These tests exercise the candidate
under a separate name before the production law was changed.

The paired old-Enzyme run on that refreshed source passes original and
changed-point value comparisons at rtol/atol 2e-13; full gradients are
identical at every tested point. Native malloc/free
calls are 15 at all three event counts, with zero calloc/realloc calls.
At N=146, alternating gradient medians are 23.03 us baseline and 19.9495 us
candidate (ratio 0.86624). Retained constants decrease from 31,012 to 25,658
bytes. Candidate emission stays at 1,653 nodes and 13 loops at every N;
baseline emission stays at 1,197 nodes and nine loops. Both paired harness
calls report 96 Julia heap bytes; this observation remains unattributed and
is not a claim about intrinsic allocations in the sampler kernel.

The current-Enzyme paired run also gives 15 malloc/free calls at all three
sizes, zero calloc/realloc, exact full gradients at every tested point and
the same retained/emitted sizes. At N=146, alternating medians are
21.105 us baseline and 20.515 us candidate (ratio 0.97204), a modest timing
improvement. The timing statistic is the median of 20 warmed sample minima,
with alternating measurement order and host load retained in the raw logs.

The registered-Enzyme paired run also passes, with identical full gradients
at every tested point. Its baseline malloc/free counts are 28, 108 and
1,188; the candidate stays at 31. Both have zero calloc/realloc calls.
At N=146, alternating medians are 43.915 us baseline and 19.0345 us candidate
(ratio 0.43344). Requested malloc bytes decrease from 37,344 to 34,424.
At N=1 the candidate adds three calls and measures about 1.3% slower, so
this is a workload-specific improvement rather than a universal speed claim.

The current-Enzyme singleton-only prototype benchmark, with bound Boolean
observations and weights, measures 0.2012 → 0.1191 us at 5 positions,
1.5283 → 0.1172 us at 51, and 14.055 → 0.1073 us at 501. It trades two
malloc/free calls for four (32 → 36 requested bytes); both variants report
zero Julia heap bytes. Maximum gradient discrepancy is 4.41e-13 at 501
positions. These timings do not describe the full-row diagnostic above.

## Rejected controls and mechanism

Suppressing only redundant primitive bound-vector read caches reduces
retained constants but adds one malloc/free call at every measured N on
both development builds. Alternating timing shows no useful speedup. It is
not a standalone fix. Restoring strict full-row semantics together with
that cache change recovers 12 calls, but removes supported singleton rows;
it is an isolation control only.

Separating singleton/full-row work into prepared child kernels materializes
intermediates and worsens the old-build count to 740. Choosing the loop
extent conditionally gives 160. A constant-stride row range gives 159 old /
22 current in the full model. None is an accepted alternative.

`plain-rows.jl` and `plain-rows-write.jl` isolate the shape/store interaction
in ordinary Julia, without RK. Both check values and analytical gradients.
Without later workspace writes, the ordinary indexed loop allocates no
native storage. Adding later differentiable writes makes it allocate N+2
malloc/free calls on old Enzyme (3, 13, 148); current Enzyme uses 3, 7, 11.
The strict-row control stays at two. The stride-with-write control makes
6N realloc calls on both builds, which a malloc/free-only counter misses.
The later-write suite passes 54 checks on each build.

## Scope

The owner has a separate parked follow-up for singleton transcendental work
across all product-mixture laws and nested duplicate-cell grouping. This
investigation addresses the reported Bernoulli-logit allocation regression;
it does not establish completion of that broader optimization task.
