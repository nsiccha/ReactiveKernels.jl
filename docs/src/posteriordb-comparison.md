# Faithful posteriordb benchmark checkpoint

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This page reports the current reproducible benchmark checkpoint for the **82 faithful
posteriordb model modules** implemented with idiomatic ReactiveKernels distribution-kernel
graphs, measured against the natural-source base (`f1e8b83`, whose modules source their
complete real data through `PosteriorDB.jl` rather than hand-inlined arrays). It is a
checkpoint, not a claim that every compiler backend is complete: native ReactiveKernels,
upstream Turing, reference Stan, and the per-model Reactant lowering are all measured here.
Each Reactant cell is a numeric timing where the faithful graph lowers, or the exact
lowering/AD diagnostic where it does not — never a hand-rewritten "Reactant-friendly"
density. The failures are kept explicit precisely so they can be improved or fixed.

On this base the load-independent result is strong: **all 82 native RK graphs produce finite
primal, gradient, and native-HMC values**, and **81 of 82 pass full parity** with reference
Stan and upstream Turing — the single non-pass, `GLMM_Poisson`, is a Turing prior-support
non-equivalence, not an RK defect (see the reading guide). **All 82 graphs also lower fully
through Reactant** (primal, gradient, and the compiled HMC loop) with no remaining lowering or
AD diagnostic. The native defects that were open on the earlier frozen base are resolved by the
inherited core compiler fixes, not by the data naturalization.

Every comparator receives the **same complete posteriordb dataset**, parameterization,
priors, supports, and Jacobians. Reference Stan is compiled from the posteriordb model;
Turing comes from the pinned upstream DynamicPPL posteriordb catalog; ReactiveKernels uses
the committed rich `@kernel` model. Values, gradients, and support probes are checked before
any timing is accepted. A failure stays visible in the table as its exact diagnostic.

```@eval
Main.ReactiveKernelsDocs.render_all80_native_checkpoint_summary()
```

## At a glance

The clean, load-independent core of this checkpoint is **correctness** and **Reactant/AD
lowering coverage** — which faithful graphs are parity-verified and which compile through
Reactant — reported **separately from timing**. This entire run was measured under sustained
observed host load (competing julia processes at ~377 of 383 telemetry samples), so **no
timing here was measured under certified isolation** (see the **Measurements & evidence**
section below). Every timing is therefore directional-only, never a performance verdict; the
coverage and correctness below do not depend on host load and are the source of truth.

Coverage first — how much of the faithful portfolio lowers through Reactant, and where each
model records a numeric result versus an exact lowering/AD diagnostic. On this base **all 82
lower for all three operations** (primal, gradient, and the compiled HMC loop):

```@eval
Main.ReactiveKernelsDocs.render_all80_reactant_coverage_plot()
```

Single-evaluation speedup (RK vs reference Stan / upstream Turing), read under the whole-run
observed-load caveat above — directional only; the one confirmed gross-workload mismatch (`Mb`)
is marked, and `GLMM`'s RK/Turing ratio is excluded (non-equivalent Turing support):

```@eval
Main.ReactiveKernelsDocs.render_all80_speedup_plot()
```

Reactant compiled-HMC-loop throughput versus native RK (same load caveats; throughput only,
**not verified-matched** — see the reading guide):

```@eval
Main.ReactiveKernelsDocs.render_all80_reactant_hmc_plot()
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

The complete machine-readable receipt (native ∪ Reactant) is
[`all80-benchmark-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/all80-benchmark-v1.toml);
the native-only frozen checkpoint
[`all80-native-checkpoint-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/all80-native-checkpoint-v1.toml)
remains committed as historical provenance.

## All model rows

The three tables are sortable. A `RK + Reactant` cell is a numeric timing where the faithful
graph lowers through Reactant, or the exact lowering/AD diagnostic where it does not (sorted to
the bottom). `unavailable` means the correctness or differentiation gate failed and includes the
diagnostic that prevented a valid timing.

```@eval
Main.ReactiveKernelsDocs.render_all80_native_checkpoint()
```

## Reading the results

This checkpoint is deliberately **not** summarized as “ReactiveKernels always wins.” Read every
comparison through these categories (curated with `ReactiveKernels:performance`); a
numerically/parity-valid ratio is **not** automatically a matched-workload verdict:

- **Gross-workload mismatch — `Mb` only (confirmed).** The RK graph evaluates a rich O(M)
  per-individual plate while the comparator precomputes O(1) sufficient statistics — marked
  distinctly, not a matched comparison. No other model is positively certified same-workload;
  the rest of capture-recapture (`Mt`/`Mth`/`Mtbh`) and `Survey` are **unclassified** on this axis.
- **Preprocessing / source-form deltas.** Eight wells models (`wells_dist100`, `wells_dist100ars`,
  `wells_interaction`, `wells_interaction_c`, `wells_dae`, `wells_dae_c`, `wells_dae_inter`,
  `wells_daae_c`) and the `mesquite` / `log-height` / `kidscore-Z` named-transform set differ by
  where preprocessing is placed (plus endpoint costs), not by a sufficient-statistic collapse.
- **`wells_dist` — the historical endpoint defect is structurally removed.** The natural-source
  graph evaluates the likelihood directly on the logit scale (`bernoulli(; logit = η).logpdf`);
  `logistic` appears only in a generated-quantity `p`, outside the posterior. The old
  probability→logit saturation roundtrip that produced a non-finite density is therefore gone,
  and the model passes native parity (its multi-point gradient residual `~1e-10`, larger than the
  `~1e-15` typical, is not by itself evidence of a lurking endpoint defect — a targeted replay of
  the historical failing point is the remaining numerical closure check).
- **External-preprocessing registry binds** (dogs, LSAT, capture-recapture, …). An
  authoring / partial-evaluation coverage issue; being external does not by itself establish
  unequal timed work against Turing. (`Survey`, previously listed here, now lowers fully through
  Reactant — its historical Enzyme AD process-abort no longer reproduces on this base.)
- **Turing support mismatch — `GLMM_Poisson`.** Upstream Turing declares `beta2 ~ Uniform(-10, 20)`
  where reference Stan uses the `uniform(-10, 10)` prior, so RK/Stan is a valid comparison but
  RK/Turing is non-equivalent and its ratio is excluded.
- **HMC throughput is not verified-matched.** A configured 16 leapfrog steps is not an established
  *executed* 16; all AHMC/RK throughput stays not-verified-matched pending gradient-call /
  executed-step counters. HMC medians are large because each transition runs 16 leapfrog steps.
- **What the "0 remaining RK defects" improvement is, and is not.** Relative to the earlier frozen
  base, the count of flagged comparison rows dropped from roughly twenty to one. That is *comparison
  certification*, not "twenty compiler bugs fixed", and decomposes into distinct causes: about
  eleven were undeclared reference-normalizer offsets, now *declared* (a reporting change, not a
  fix); inherited core compiler fixes address the marker (`2ba4a639`), authored-scan
  (`d3cc5878`/`bc17`), and Bernoulli-AD cases; and `wells_dist` additionally carries the
  direct-logit source change above. Data naturalization is separate again — the earlier benchmark
  already bound real posteriordb data, so loading it through `PosteriorDB.jl` is not itself a
  correctness change.
- **The parity gate is a multi-point finite probe.** Each model is gated at three in-support draws
  (`draws = 3`): the value residual is taken across all three and the gradient error is the maximum
  over them. Finite probing at a few points cannot prove all-input correctness, but it is not a
  single-point check.

Point medians that are close should be read as ties. The whole-run observed-load timing provenance
is in the next section.

## Measurements & evidence (timing provenance)

Timing on this shared host was **not** measured under certified isolation. Unlike earlier
partial passes, this checkpoint was produced by a **single end-to-end run** (native then
Reactant) under **sustained observed load**: 20-second-cadence telemetry recorded a competing
julia process at **377 of 383 samples** across the whole run. So the honest statement is uniform
— *every* timing on this page is an observed-load measurement, not a quiet-window one.

- **No adjusted or synthetic times are substituted.** The raw measured values live in the
  machine-readable receipt exactly as recorded, with this provenance. This is not a claim that
  every absolute timing is inflated by a fixed factor; differential interference between backends
  is unknown, so ratios near parity in particular should be read as ties.
- **The ranking plots conservatively omit four historically-contended rows**
  (`earnings-earn_height`, `dogs-dogs_hierarchical`, `wells_data-wells_dist`,
  `GLMM_Poisson_data-GLMM_Poisson_model`); their structural/value/diagnostic results stand and
  their raw timings remain in the receipt. Because the whole run is now uniformly observed-load,
  whether to keep that four-row omission, drop the timing rankings entirely, or show all 82 under
  one caveat is a presentation choice under review.

A certified-clean timing pass remains blocked on a genuinely-quiet window on this shared host,
which was not available. The run is crash-safe and writes each completed row immediately. Until a
clean window exists, **correctness and Reactant coverage are the source of truth here and timing
is provenance-only.**

## What remains

On this base the native value/gradient/support defects that were open on the earlier frozen base
are resolved — each by its inherited fix canonical, not by any workaround in the model sources
(`dogs_hierarchical` → `1349092`, GLMM marker-six → `2ba4a639`, arma11 authored-scan →
`bc17`/`d3cc5878`, `wells_dist` → the direct-logit source form). The Reactant phase, run once for
numerical/diagnostic **coverage**, now lowers all three operations for all 82 graphs with no
remaining lowering or AD diagnostic; the mechanism that records an exact diagnostic (rather than a
workaround) stays in place for the next model or backend change that does fail.

Three things genuinely remain. **(1)** A certified-clean **timing** pass — the only thing that
would turn these observed-load timings into a defensible performance ranking — still awaits a
genuinely-quiet window on this shared host. **(2)** A **matched same-transition** HMC comparison
(same `T`, `q`, metric, leapfrog count, step size on both backends) is a tracked investigation;
the throughput figures here are a capability probe, not a matched ranking. **(3)** The receipt is
currently **run-log-certified** for source identity (the harness's `source_identity()` digest and
its `assert_resume_source!` check exist but are not yet wired into resume verification, which
validates schema and phase only); stamping one frozen process-start identity into the receipt is a
separate hardening task. Until the clean timing window exists, correctness and coverage are the
source of truth here and timing is provenance-only.

The older hand-written flat-density and small-subset experiments answered useful compiler
questions, but they are not substitutes for this full-data, rich-kernel comparison. Their
receipts remain in the repository as historical artifacts; this page's tables, plots, and the
linked aggregated benchmark receipt are the current source of truth.
