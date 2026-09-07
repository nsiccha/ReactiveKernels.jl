# Faithful posteriordb benchmark checkpoint

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This page reports the current reproducible benchmark checkpoint for the **82 faithful
posteriordb model modules** implemented with idiomatic ReactiveKernels distribution-kernel
graphs. It is a checkpoint, not a claim that every compiler backend is complete: native
ReactiveKernels, upstream Turing, and reference Stan are measured here; the per-model
Reactant pass remains separate and its cells say so explicitly.

Every comparator receives the **same complete posteriordb dataset**, parameterization,
priors, supports, and Jacobians. Reference Stan is compiled from the posteriordb model;
Turing comes from the pinned upstream DynamicPPL posteriordb catalog; ReactiveKernels uses
the committed rich `@kernel` model. Values, gradients, and support probes are checked before
any timing is accepted. A failure stays visible in the table as its exact diagnostic.

```@eval
Main.ReactiveKernelsDocs.render_all80_native_checkpoint_summary()
```

## Protocol

- Primal and value-plus-gradient cells are median wall-clock nanoseconds; lower is faster.
- HMC uses the same fixed mechanics on both sides: multinomial HMC, **16 leapfrog steps per
  transition**, step size `0.03`, one untimed warmup, and six timed rounds. The number of
  transitions is chosen once per model from the slower measured gradient to target about
  0.5 seconds per round, bounded at 4–1,000, and is shared by RK and AHMC. The receipt records
  that count. Cells report median microseconds per transition; this is throughput, not ESS or
  adaptation quality.
- Native timings run in a Julia subprocess that never loads Reactant, preventing Reactant's
  compiler state from perturbing native compilation and timing. The pinned environment can
  load Turing, Mooncake, Enzyme, AdvancedHMC, BridgeStan, PosteriorDB, and Reactant together;
  the process split is for measurement isolation, not dependency compatibility.
- A declared additive density offset is accepted only when derived from the Stan/Turing
  sources and constant across evaluation points. Gradients and support must still match.

The complete machine-readable checkpoint is
[`all80-native-checkpoint-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/all80-native-checkpoint-v1.toml).

## All model rows

The three tables are sortable. `not run in this native checkpoint` is intentionally not a
Reactant performance claim. `unavailable` means the correctness or differentiation gate
failed and includes the diagnostic that prevented a valid timing.

```@eval
Main.ReactiveKernelsDocs.render_all80_native_checkpoint()
```

## Reading the results

This table is deliberately not summarized as “ReactiveKernels always wins.” Some rich RK
graphs are substantially faster than the comparison, some are ties, and some lose. The
`Structural difference` column records important authoring differences such as a vectorized
Stan primitive versus a per-observation RK plate. Point medians that are close should be read
as ties unless a repeated-distribution analysis establishes separation.

The HMC numbers can be much larger than a single gradient call because each transition
performs 16 leapfrog steps. Adaptive repetition prevents a slow model from spending minutes
to establish needless decimal places while cheap models still receive up to 1,000
transitions per round. The checkpoint is crash-safe and writes each completed row
immediately.

## What remains

The Reactant phase must still execute each operation for each model. Its publication rule is
numeric timing where lowering succeeds and the exact lowering error where it does not; the
benchmark never substitutes a hand-rewritten “Reactant-friendly” density. Known native
value/gradient failures likewise remain explicit until their owning kernel or AD limitation
is corrected and the affected rows are replayed.

The older hand-written flat-density and small-subset experiments answered useful compiler
questions, but they are not substitutes for this full-data, rich-kernel comparison. Their
receipts remain in the repository as historical artifacts; this page's table and linked
checkpoint receipt are the current source of truth.
