# Joint derivative staging: bounded comparison

This experiment compares selecting a primal and its active cotangents together
with selecting the reverse branch first. The private implementation is
`56241c299c7e1d16e019f287dd3a449b576d88f8`, based on `6358fa81`. It is a proposal,
not a selected production architecture. It changes general `DerivativeRule`
staging; it does not change the plate-specific reverse adapter.

## Selection and execution

The four-recipe fixture gives independent primal and derivative producers cost
4 each. A joint producer costs 5 and its contraction costs 1. Reverse-only
planning chooses cost 4, then separately needs the cost-4 primal. Joint planning
chooses total declared cost 6. The prototype preserves those chosen producers
when splitting at seed availability. If the selected primal needs the seed, it
uses the existing independent staging instead.

The focused joint, multi-output and general derivative suites passed 483 checks
in the initial combined run. A later matched comparison passed the 165 joint
checks on each Enzyme environment. These overlapping counts establish the
tested selection and numerical behavior, not globally optimal runtime.

## Warm work and storage

`derivative_joint_staging_tradeoffs.jl` compares a shared exponential, an array
of exponentials and cheap squared-reduction partials. The independent control
has joint cost 10, while the sharing arm has joint cost 5. Both run the same
prototype; this isolates producer choice rather than claiming a version
speedup. Array arms also compare ordinary Enzyme Reverse of the native primal.

Each process uses 12 rotating paired rounds, 60 samples per arm and one
evaluation per sample. Ratios below are medians of the paired round ratios;
less than 1 favors sharing. Julia 1.10.12, strato2, one CPU budget, one BLAS
thread; registered Enzyme 0.13.219 and the recorded dogfood 0.13.220 build with
libEnzyme `4990aa111a76fc2b0bedd247fa5dfab8957c8677`.

| Workload and measured path | Length | Registered | Dogfood |
| --- | ---: | ---: | ---: |
| Scalar exponential, explicit forward/reverse stage loops | 1,000 | 0.485 | 0.486 |
| Scalar exponential, explicit forward/reverse stage loops | 100,000 | 0.510 | 0.518 |
| Array exponential, ordinary Reverse through rule adapter | 1,000 | 0.580 | 0.595 |
| Array exponential, ordinary Reverse through rule adapter | 100,000 | 0.479 | 0.474 |
| Array square, ordinary Reverse through rule adapter | 1,000 | 1.630 | 1.601 |
| Array square, ordinary Reverse through rule adapter | 100,000 | 1.136 | 1.147 |

The corrected scalar callers allocate zero Julia bytes in both arms and use
preallocated value, tape and gradient arrays. The initial scalar driver failed
to specialize its callable; its allocation/timing results are superseded by
the corrected driver in `552609b23`.

At length 1,000 the independent array adapter allocates 8,144 Julia bytes in
two allocations, versus 16,272 bytes in three allocations for sharing. At
100,000 the corresponding observations are 800,064 bytes/three allocations and
1,600,112 bytes/five allocations. The ordinary native control reports zero
Julia allocations; that does not account for possible native backend storage.
Sharing saves expensive arithmetic here but loses for cheap partials despite
its lower declared graph cost.

Bootstrap summaries resample the 12 paired round ratios, 20,000 draws with seed
20261113. Their intervals describe within-run variation on a shared host, not
independent-machine replication. Exact manifests, raw times and the summary
script are included in the evidence archives below.

## Retention correctness

`derivative_joint_staging_alias.jl` authors an equivalent joint producer whose
named partial aliases its input. An ordinary Reverse caller evaluates the
rule, overwrites that input, then returns the earlier value. At lengths 3 and
17 the joint arm returns a zero gradient instead of `2x`; the independent and
explicit-copy arms pass. Empty inputs pass. Both recorded Enzyme environments
reproduce this result. Recursive stage type audits are empty, so inference
alone does not establish ownership safety.

The adapter classifies residuals using graph names. A newly named intermediate
is not proof of fresh storage. The unchanged reverse-first staging avoids this
particular cost-5 choice, while the proposed selection exposes it. This
counterexample rules out adopting the current prototype unchanged.

`derivative_residual_snapshots.jl` is a separate conservative comparison. It
tests direct aliases, views, tuples, named tuples, records, nested arrays and a
view of the returned primal. Its snapshot mode deep-copies the entire retained
tuple together, preserving internal sharing. This is a benchmark-only method
replacement, not a production ownership policy.

The first run varied the joint cost to 2, which also makes unchanged staging
retain the partial. Both old and proposed staging then recorded 14 wrong
ordinary input-overwrite gradients and two wrong returned-primal-view gradients.
The unchanged retention helper has identical source bytes at `6358fa81`,
`aa82c44e1` and `95b1ebba1`. The defect therefore predates joint selection.

Whole-tuple copying passed every numerical, retained-type and shared-alias
assertion, but the run finished **146 pass / 5 fail**: five recursive audit
assertions reported inlined Base copying internals. Their frames identify
`Array`, `IdDict` and `deepcopy`; one mutable-record case also reports
`deepcopy_internal::Any`. These failures remain in the original log.

`derivative_snapshot_audit.jl` isolates the same finding in a plain caller of
Base's `deepcopy`. Preserving that Base call boundary produces no audit finding,
while a nonconcrete user helper behind the same boundary is still reported.
All nine attribution checks passed. This follows the documented audit scope
which treats Julia's own callees as leaves; it does not make Base's internals
concrete. The `snapshot_leaf` mode passed all 151 numerical, ownership and
scoped audit checks on registered Enzyme, including the eight container forms
at lengths 0, 3 and 17. No audit implementation or policy was changed.

## Costs with conservative snapshots

The registered-Enzyme follow-up applies whole-tuple copying to both adapter
arms, including direct input residuals which a more precise ownership analysis
could avoid copying. The explicit stage-loop arms do not call retention and
therefore are not the snapshot-inclusive comparison. The following uses the
ordinary Reverse adapter arms, with the same 12 paired rounds as above.

| Family | Length | Shared / independent | Within-run 95% interval | Shared / native ordinary AD |
| --- | ---: | ---: | ---: | ---: |
| Array exponential | 1,000 | 0.632 | 0.605–0.655 | 0.703 |
| Array exponential | 100,000 | 0.500 | 0.495–0.508 | 0.509 |
| Array square | 1,000 | 1.330 | 1.166–1.368 | 5.147 |
| Array square | 100,000 | 1.213 | 1.175–1.239 | 2.466 |

At length 1,000, independent/shared adapters allocate 16,640/24,768 Julia
bytes in 7/8 allocations. At 100,000 they allocate 1,600,480/2,400,528 bytes
in 9/11 allocations. The native control still reports zero Julia allocations.
The snapshot-inclusive exponential sharing gain survives this conservative
copying strategy; cheap square partials still favor recomputation. These are
controlled producer-selection comparisons, not a general production speedup.
The second Enzyme environment passed the same 151 ownership checks and 203
joint/multi-output checks. Its first broader-regression process stopped while
loading the missing scratch dependency `ChainRulesCore`, before costs ran.
That failed run is retained below; installed-candidate verification with the
required dependency and the remaining costs are separate work.

The installed candidate uses the same tuple snapshot plus a bits-only fast
path. Its registered run passes 722 checks: 239 new mutation/retention checks
and the existing joint, multi-output and general derivative suites. This is a
completed phase of fleet `489475552ac94380`; the overall run is still active.
The registered scalar ordinary-Reverse comparison reports zero Julia allocations
and a 30 ns median for independent, shared and native arms. At that timing
resolution it establishes no scalar adapter speed advantage. The explicit
forward/reverse stage-loop gains above must not be transferred to that result.

## Preparation and evidence limits

The initial preparation observations shared compiler caches between arms and
cannot establish an independent first-call saving. Two subsequent preparation
drivers failed before measurement: first on a relative include path, then on
assigning the exported `value` name at top level. Neither failed run establishes
a preparation cost. The corrected `derivative_joint_preparation.jl` now measures
graph construction, rule construction and first forward/reverse stages in
separate fresh processes, excluding package loading and driver setup. It calls
the timed operations through `invokelatest` so target compilation occurs inside
the timer. The following are individual shared-host observations, not repeated
cold-start distributions or measurements of first ordinary-AD compilation.

| Family | Existing staging | Joint staging |
| --- | ---: | ---: |
| Scalar exponential | 8.776 s | 9.302 s |
| Array exponential | 8.957 s | 9.054 s |
| Array square | 9.760 s | 8.903 s |

Compilation accounts for almost all of these measured intervals. They do not
show a consistent preparation advantage for either strategy.

The diagnostics below preserve failures as well as successful phases. A
diagnostic exit 0 means the wrong-gradient result was recorded with passing
controls, not that the alias arm passed correctness.

- Original paired array measurements: fleet `172e7adfe3ef408e`, exit 0;
  [sources, environments and raw measurements](http://localhost:4200/code/download?path=/home/niko/.local/state/kb-agents/uploads/1d74c9050d9d9c57c5f78f81a40e42b5d64c1635a1815c33e8faa9a2f348f433.tgz).
- Corrected scalar measurements: fleet `8bb2159a6a694415`, combined exit 1 due
  to separate environment/preparation phases;
  [complete retained evidence](http://localhost:4200/code/download?path=/home/niko/.local/state/kb-agents/uploads/2e0da92ca83b716773057729640721446ad11f7b202bf11d14a55b6d53b845b9.tgz).
- Matched baseline/joint alias attribution and landed PPL checks: fleet
  `f111b12289c44ea3`, combined exit 1 due to the preparation name collision;
  [logs, manifests, exact failed driver and terminal receipt](http://localhost:4200/code/download?path=/home/niko/.local/state/kb-agents/uploads/662c084cbbcc49411ba998b338b38ec5f38ff192ac17ad5404ac225122272b13.tgz).
- Existing/joint ownership controls, first snapshot experiment and baseline
  preparation: fleet `1a1e38dfe0314cda`, exit 1 on the five audit assertions;
  [exact sources and full retained evidence](http://localhost:4200/code/download?path=/home/niko/.local/state/kb-agents/uploads/a026b3374cd443f8e500ccd86fe0402c3955e03e5ce69b8c9fa47aa84a2646e5.tgz).
- Independent joint preparation, snapshot acceptance, audit attribution and
  registered snapshot-inclusive costs: fleet `287b0545ab4c4558`, exit 0,
  605 seconds;
  [exact sources, typed IR, manifests, raw rounds, summary and terminal receipt](http://localhost:4200/code/download?path=/home/niko/.local/state/kb-agents/uploads/db20ada0d0f59502be1f4cf098390991cc80c5473105de0bb7c3a145f791529f.tgz).
- Dogfood snapshot and partial regression verification: fleet
  `384fa7c25ec248db`, exit 1, 334 seconds, missing `ChainRulesCore`;
  [successful phases, loader failure, exact sources and terminal receipt](http://localhost:4200/code/download?path=/home/niko/.local/state/kb-agents/uploads/868e12a94eeca7395c3987ff0c8964ef831a9e1742383e522ad60c152bb0dd91.tgz).

The remaining work includes installed-source repair verification, scalar
ordinary-adapter costs and a readonly design-matrix comparison. Conservative
copying can duplicate readonly data and whole parents of small views; it is not
an ownership analysis. The prototype does not account for these measured
storage costs when selecting recipes. No general preference follows from the
additive recipe costs alone.
