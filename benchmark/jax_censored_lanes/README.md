# Censored rows in JAX modules: lane partitions and the fraction's stop

Snag `jax-source-censo-ca7d7cd5` (RK 0.1.277). A censored Student-t response
is one plate whose cell branches on its data, `y <= lower ? log tail :
log density`, inside an arm of the law's parameter-support guard. Its JAX
module was one `jax.vmap` of that cell, and under `vmap` the batched
`jax.lax.cond` evaluates both arms on every row: the lower tail's
incomplete-beta continued fraction ran on every row although only the
censored rows take it.

`jax_source` with `data` now partitions such a plate by its data-only
branches (`src/jax_lanes.jl`): `prepare_data` returns each decision pattern's
cells, and the functions run one `vmap` per pattern, so the tail runs only
on the censored rows.

## The emitted modules

`emit.jl` writes a generic censored Student-t regression (12,245 rows, 115 of
them censored at a detection limit, `-Inf` elsewhere; seeded) twice: as
`jax_source` emits it (`partitioned.py`) and with the partition switched off
in that process only (`whole.py`, the module before RK 0.1.277). It also
writes the raw data and native Julia log densities and Enzyme gradients at
four points (`inputs.json`). `measure.py` checks every module it finds under
`OUT/*/` against the native reference of its own directory, then times
`jax.jit(jax.vmap(jax.value_and_grad(log_density)))` over batches of points.

```sh
julia --project=ENV benchmark/jax_censored_lanes/emit.jl OUT/<tree> [ROWS] [CENSORED]
python benchmark/jax_censored_lanes/measure.py OUT [batches] > OUT/measurements.json
```
