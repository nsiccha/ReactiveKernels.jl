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
- `results.json` — steady-state per-call means (20 reps) behind the
  KB brief plot, with per-cell repeat runs where taken.

## Conditions (2026-09-21 run)

- Single-threaded both sides (model has no `reduce_sum`/`map_rect`).
- Same-distribution random points (`0.1*randn`, fixed seed), fresh dims per K.
- Contended box (load 11-14 on 8 CPUs): sub-2x differences are noise.
- RKPPL K=30 gradient: one-time Enzyme `prepare_ad` did not return in
  15 min (bind/build/eval all complete); Stan K=30 grad ~8 ms.
