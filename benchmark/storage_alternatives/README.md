# Storage placement diagnostics

This compares source exposure and algebra before buffer planning with
reconsideration of already planned materializations. The assigned recovery
base is `0a6b45626c23c20425de836b99215de97d317a9f`. These drivers do not replace
production methods or author derivative rules. Gradient runs use ordinary
`Enzyme.Reverse`, with the optional plate reverse path disabled.

| Arm | Mechanism | Main tradeoff |
| --- | --- | --- |
| `control` | Existing nonallocating preparation | Preserves the established execution/storage order. |
| `early` | Expose fused recipes, rewrite algebra and reconsider arrays, then plan buffers | More source algebra is visible, but exposure must integrate with the decomposition grammar. Raw deferral fails mutation tests. |
| `late` | Expose owned materialization steps, reconsider arrays, restore survivors and rewrite | Reuses the existing plan, but must maintain cache ownership and recognize which storage can be deferred. Raw deferral fails mutation tests. |
| `early_snapshot` | Early placement with snapshots of eligible dense floating operands | Preserves the measured operand mutation cases, at the cost of copying data. |
| `late_snapshot` | Late placement with the same operand snapshots | Same semantic obligation; appended existing copy steps reuse owned buffers. |

The snapshot arms are narrow structural experiments, not a general safety
proof. Copying operands does not establish that delaying a producer preserves
its global reads, effects or exceptions. Unsupported operand shapes retain
their materialization. A separate diagnostic asks for an unconditional
compiler effect-free result for each recipe; its current refusal even for
the read-only example means it supplies no demonstrated optimization coverage.
The fallback is existing execution, not a restriction on client programs.

## Drivers

- `smoke.jl`: small primal/ordinary-Reverse check of all arms.
- `compare.jl`: four mechanism workloads, lengths 0/1/17/1000, changed runtime
  inputs, full call-graph type audits, emitted structural counts, warm timings,
  and direct/global/private-intermediate mutation controls.
- `edges.jl OUTPUT`: BigFloat tail oracle, escaped outputs, required lazy
  guards and exceptions, empty and ragged data, and type audits.
- `natural.jl ARCHIVE INPUTS OUTPUT`: includes the supplied unchanged private
  model source and checks four frozen points for each of two LogisticRHS
  models. `extract_natural.py frozen-inputs.json INPUTS` writes runtime binary
  arrays; neither model source nor private inputs is embedded in this tree.
- `cold.jl MODE OUTPUT.toml`: one arm per fresh Julia process; separately
  measures planning, preparation, first primal, AD preparation and first
  gradient, including bytes and GC time. Package loading is outside these
  measurements. Run each arm in its own process and rotate order between rounds.
- `callable_effects.jl OUTPUT.toml`: bounded check of a producer reading
  mutable global state while a later reader changes it.

Run from the repository root in an environment resolved with this RK checkout,
MutatingFunctions, Enzyme, DifferentiationInterface, BenchmarkTools and the
in-tree distribution/PPL packages. Host compute-token requirements still
apply. The recorded experiments use Julia 1.10.12 and the shared package depot.

`STORAGE_RESULTS`, `STORAGE_CASES`, `STORAGE_SIZES` and `STORAGE_SNAPSHOTS`
filter the mechanism driver. For example:

```sh
STORAGE_RESULTS=/path/to/results STORAGE_SIZES=0,17,1000 \
  julia --startup-file=no --project=/path/to/resolved-env \
  benchmark/storage_alternatives/compare.jl
```

## Reading the evidence

The multi-arm drivers catch individual failures and retain their evidence.
**Exit zero alone is not acceptance:** inspect the `failures` arrays and
`preserved` fields. The deliberately unsafe raw arms remain in the comparison
so that performance cannot conceal their semantic failures.

Warm timing order rotates through three rounds. Report medians together with
the raw rounds; these runs share a busy host and do not establish a universal
speed ranking. Preparation and first-gradient times in the multi-arm drivers
share a compiler process and have order effects. Use the separate-process
`cold.jl` observations when comparing preparation costs. Neither timing is a
pure compiler-time measurement: it includes the named stage's ordinary work.

AST node and loop counts are explicitly structural diagnostics, obtained by
expanding reachable plate programs. Constant counts across runtime sizes are
evidence against data-length-dependent source replication, not a substitute
for primal, derivative, access-safety and mutation checks. Any expanded
structural dump is distinct from the actual `code_expr` of a prepared kernel.

Results under `results/` carry source hashes and environment/library identity.
The initial `mechanisms-20261113.json` predates integration fixes that route
exposed materializations and late snapshots through existing reusable slots;
its incidental allocations must not be attributed to placement itself.

## Measured comparison, 2026-11-13

The v3 corpus is `results/comparison-v3-20261113.json`. On Julia 1.10.12,
Enzyme overlay `7832d492b16bad8862de8fb0d0dcf7200af3a30d` and libEnzyme
`e316a076ef438970e339876079acd02ce48d3d21` (LLVM15), it records 80 arithmetic
cases, 55 edge cases and 10 natural-model arms with no validation failures.
The 20 separate mutation controls deliberately retain the raw early/late
failures. Both snapshot arms and control preserve those mutation cases.

Warm gradient medians in microseconds (median of three round medians):

| Workload | Control | Early snapshot | Late snapshot |
| --- | ---: | ---: | ---: |
| Stored, n=1000 | 23.18 | 20.58 | 20.06 |
| Explicit algebra, n=1000 | 30.08 | 30.12 | 30.22 |
| Retained producer, n=1000 | 23.26 | 23.47 | 22.31 |
| Logarithmic, n=1000 | 16.80 | 14.71 | 14.82 |
| LogisticRHSOvarian | 65.69 | 65.96 | 66.88 |
| LogisticRHSProstate | 335.82 | 337.23 | 341.67 |

All five arms allocate zero warm gradient bytes in this corpus. The natural
models use their original source, dimensions 3075/11935 and four frozen
reference points each; maximum relative gradient errors are 1.22e-15 and
1.55e-14. Raw early/late medians are 57.33/59.39 microseconds for Ovarian
and 279.05/284.46 for Prostate, but their mutation failures rule them out as
general replacements. Snapshot variants have no demonstrated natural-model
speed benefit here. No full-model rewrite is inferred from the mechanism toys.

The separate-process corpus `results/cold-20261113.json` has two observations
per arm, with forward/reverse execution order, on the stored n=1000 workload:

| Arm | Preparation, seconds | AD preparation, seconds | First gradient, seconds | Sum of those stages, seconds |
| --- | ---: | ---: | ---: | ---: |
| Control | 6.21–7.08 | 0.87–0.91 | 3.93–4.04 | 11.05–11.99 |
| Early | 5.79–6.47 | 1.07–1.42 | 3.66–3.70 | 10.51–11.58 |
| Early snapshot | 6.50–6.73 | 1.22–1.43 | 4.53–4.68 | 12.25–12.84 |
| Late | 6.97–7.72 | 0.82–0.98 | 3.35–3.71 | 11.65–11.89 |
| Late snapshot | 6.92–7.20 | 0.80–0.81 | 4.47–4.73 | 12.46–12.47 |

First-gradient allocations are about 207 MiB for control/raw arms and 321 MiB
for snapshot arms; these are one-time preparation/JIT allocations, not warm
per-gradient costs or retained heap size. Early snapshot preparation allocates
about 390 MiB versus 424 MiB for late snapshot. Two observations on a shared
host establish scale and expose the fixed-order bias in multi-arm timings;
they do not establish a statistically robust ranking.

The callable-effects probe also passes all five arms: the runtime total-function
guard retains the custom global-reading producer. This is a bounded control,
not evidence that arbitrary delayed callables are safe.

The current recommendation is to retain production storage. Early placement
offers broader algebra visibility and somewhat lighter preparation here,
but needs integration with the decomposition grammar and currently duplicates
planner logic in this diagnostic. Late placement can reuse the established
plan and target known storage steps, but builds and revisits that plan and
must preserve cache ownership. Both need the same snapshot/effect semantics;
neither earns a general performance preference from these natural workloads.
