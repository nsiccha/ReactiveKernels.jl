# Joint PK+QT+TGI: Stan (BridgeStan) vs RKPPL speed comparison

Tiled subject-replica scaling of the W4 joint oracle fixture
(`joint_pk_qt_tgi_brm1`, continuous config). All SB index arrays are
within-subject relative (per-subject plate cells in `joint.stan`), so
tiling replicates each subject's ragged segment with no index remap.

## Files

- `tile.py K` — tile `stan_data.json` K times → `stan_data_tiledK.json`
  (replica-major subject order; mirrors `_tile_*` in the RKPPL script).
- `bench_stan.jl` — `TILE_K=K [PROPTO=0/1] julia --project=<bridgestan-env>
  bench_stan.jl`; needs `joint_model.so` next to the oracle `joint.stan`
  (BridgeStan compiles it on first construct) and the tiled JSONs.
- `bench_rkppl_tiled.jl` — `TILE_K=K julia --project=<kb-ppl-test-env>
  bench_rkppl_tiled.jl` from `packages/ReactiveKernelsPPL`.
- `bench_rkppl_reactant.jl` — `TILE_K=K [RK_GRAD=1] [RK_OPT=default|
  no_slice_slice] julia --project=<env with Reactant+Enzyme+DI>
  bench_rkppl_reactant.jl`: full-program `Reactant.@compile` of the same
  posterior (primal) and `compile_ad_value_and_gradient` (Enzyme-through-
  Reactant), parity-checked against the native kernels at the same point;
  prints one `RESULT {…}` line whose `reactant_*` fields merge into a
  `results.json` cell.
- The tiling helpers (`tiled_columns(K)`) live in
  `packages/ReactiveKernelsPPL/test/parity/joint_tiling.jl`, shared with the
  Reactant ladder tests so tests and benchmark bind identical columns.
- `results.json` — steady-state per-call means (20 reps) behind the
  KB brief plot, with per-cell repeat runs where taken.

## Conditions (2026-09-21 run)

- Single-threaded both sides (model has no `reduce_sum`/`map_rect`).
- Same-distribution random points (`0.1*randn`, fixed seed), fresh dims per K.
- Contended box (load 11-14 on 8 CPUs): sub-2x differences are noise.
- RKPPL K=30 gradient: one-time Enzyme `prepare_ad` did not return in
  15 min (bind/build/eval all complete); Stan K=30 grad ~8 ms.

## Reactant rows (2026-09-22 run, `bench_rkppl_reactant.jl`)

- Program: RKPPL at `467a70a` — grouped cell assignments emit one
  subject-batched statement each, so `kernel_expr` has 209 statements at
  every K (was 230/419/839 at K=1/5/10).  Native eval/grad unchanged.
- K=1 primal `Reactant.@compile` of the sampler-cut posterior: parity
  1.9e-16 vs native at two points; compile 211–227 s at 3.8 GB peak RSS;
  eval 0.03–0.17 ms device-resident (noise on a contended box; native
  0.016–0.018 ms at n_obs=7).
- K=3 primal (idle host, 2026-09-22): parity 7.3e-16 vs native at two
  points; compile 206.6 s at 8.6 GB peak RSS; eval 0.026 ms (native
  0.033 ms).  The XLA program still grows with K — Reactant traces the
  runtime subject loop unrolled (bound `op_ends`, exact values,
  `reactivekernels-use` §7c): 3.8 GB at K=1, 8.6 GB at K=3, so K=10
  extrapolates to ~25 GB and was not attempted (the contended-box K=3
  attempt on 2026-09-21 was SIGTERMed at 6.8 GB).  An O(1) XLA program
  needs the per-subject recurrence as a traced loop over a rectangular
  op table (root RK feature, not in this run).
- Gradient (`compile_ad_value_and_gradient`): six K=1 attempts, none
  returned.  Attempts 1–5 (contended box; unrolled and batched programs,
  default and `no_slice_slice` pipelines) were SIGTERMed by kb-earlyoom
  after 9–12 min at 4.2–5.95 GB RSS (snag `strato2-earlyoom-cf96ec60`).
  Attempt 6 on an idle host (13 GB available) grew to 13.3 GB RSS in
  8.3 min before the kill — the reverse compile's own requirement, more
  than 3.5× the primal compile of the same program — filed as snag
  `reactant-ad-comp-c1319307` on ReactiveKernels.  The compiled AD path
  itself is proven on the tiny model in `test_reactant_joint.jl`.
