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
