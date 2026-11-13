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
replacement, not a production ownership policy. Its validation and cost
measurements are tracked separately from the completed results above.

## Preparation and evidence limits

The initial preparation observations shared compiler caches between arms and
cannot establish an independent first-call saving. Two subsequent preparation
drivers failed before measurement: first on a relative include path, then on
assigning the exported `value` name at top level. Neither failed run establishes
a preparation cost. A corrected, separately timed process comparison is in
progress; its timer covers graph construction, rule construction and first
forward/reverse stages, excluding package loading and driver setup.

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

The remaining comparison is whether ownership-safe retention preserves useful
sharing gains after its preparation, allocation and memory costs. No general
preference follows from the additive recipe costs alone.
