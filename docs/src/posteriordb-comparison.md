# Faithful posteriordb benchmark checkpoint

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This page reports the current reproducible benchmark checkpoint for the **82 faithful
posteriordb model modules** implemented with idiomatic ReactiveKernels distribution-kernel
graphs. It is a checkpoint, not a claim that every compiler backend is complete: native
ReactiveKernels, upstream Turing, reference Stan, and the per-model Reactant lowering are all
measured here. Each Reactant cell is a numeric timing where the faithful graph lowers, or the
exact lowering/AD diagnostic where it does not — never a hand-rewritten "Reactant-friendly"
density. The failures are kept explicit precisely so they can be improved or fixed.

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
Reactant — reported **separately from timing**. Timing on this shared host could not be
measured under certified isolation (see the **Measurements & evidence** section below);
the four known-contended rows are **omitted** from the ranking below, and the remaining
timings are un-audited for host isolation — read directionally, never as a performance verdict.

Coverage first — how much of the faithful portfolio lowers through Reactant, and where each
model records a numeric result versus an exact lowering/AD diagnostic:

```@eval
Main.ReactiveKernelsDocs.render_all80_reactant_coverage_plot()
```

Single-evaluation speedup (RK vs reference Stan / upstream Turing), with the known-contended
rows omitted and the caveats above; the one confirmed gross-workload mismatch (`Mb`) is marked,
and `GLMM`'s RK/Turing ratio is excluded (non-equivalent Turing support):

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
- **Endpoint / numerical defect — `wells_dist`.** A real RK defect (logistic saturation to p=1 →
  non-finite density at an in-support draw where reference Stan is finite), not a preprocessing or
  workload excuse; bound to the T3 direct-logit cleanup.
- **External-preprocessing registry binds** (dogs, LSAT, capture-recapture, Survey, …). An
  authoring / partial-evaluation coverage issue; being external does not by itself establish
  unequal timed work against Turing.
- **Turing support mismatch — `GLMM_Poisson`.** Upstream Turing declares `beta2 ~ Uniform(-10, 20)`
  where reference Stan uses the `uniform(-10, 10)` prior, so RK/Stan is a valid comparison but
  RK/Turing is non-equivalent and its ratio is excluded.
- **HMC throughput is not verified-matched.** A configured 16 leapfrog steps is not an established
  *executed* 16; all AHMC/RK throughput stays not-verified-matched pending gradient-call /
  executed-step counters. HMC medians are large because each transition runs 16 leapfrog steps.

Point medians that are close should be read as ties. Timing provenance — including which rows are
omitted for known contention — is in the next section.

## Measurements & evidence (timing provenance)

Timing on this shared host was **not** measured under certified isolation. Two distinct classes:

- **Known-contended — OMITTED from every ranking/ratio.** The four rows re-measured after the
  gate-correctness fixes — `earnings-earn_height`, `dogs-dogs_hierarchical`, `wells_data-wells_dist`,
  `GLMM_Poisson_data-GLMM_Poisson_model` — were each measured while a competing julia process ran on
  the host (5-second-cadence telemetry observed competing load at ~119 of 132 samples during the
  run). Their structural/value/diagnostic results **stand**; their timing values are quarantined
  from the plots and ratios. The raw contended values remain in the machine-readable receipt with
  this provenance — **no adjusted or synthetic times are substituted**. This is not a claim that
  every absolute timing is inflated; differential interference between backends is unknown.
- **Un-audited — shown, not clean-certified.** The remaining rows were measured in an earlier pass
  whose host isolation was not independently verified. They are neither silently recertified clean
  nor automatically implicated by the later contention; they carry their own un-audited provenance
  and should be read directionally.

A certified-clean timing pass remains blocked on a genuinely-quiet host window, which was not
available on this shared host during the checkpoint. The checkpoint is crash-safe and writes each
completed row immediately.

## What remains

The Reactant phase is run once for the numerical/diagnostic **coverage** (which faithful graphs
lower and which record an exact lowering or AD diagnostic) — reported separately from timing, since
that pass also runs under the same observed host load. Those diagnostics — not workarounds in the
model sources — are the handles for the next round of generic compiler fixes: as each lands, the
affected rows are replayed and the aggregated receipt is regenerated. Known native value / gradient /
support defects are surfaced the same way, each bound to its fix canonical (dogs_hierarchical →
`1349092`, GLMM marker-six → `2ba4a639`, arma11 authored-scan → `bc17`/`d3cc5878`, wells_dist →
T3 direct-logit), and upgraded to numbers once the owning kernel or AD limitation is corrected.

A certified-clean **timing** pass — the only thing that would turn the omitted/un-audited timings
into a defensible performance ranking — still awaits a genuinely-quiet window on this shared host.
Until then, correctness and coverage are the source of truth here and timing is provenance-only.

The older hand-written flat-density and small-subset experiments answered useful compiler
questions, but they are not substitutes for this full-data, rich-kernel comparison. Their
receipts remain in the repository as historical artifacts; this page's tables, plots, and the
linked aggregated benchmark receipt are the current source of truth.
