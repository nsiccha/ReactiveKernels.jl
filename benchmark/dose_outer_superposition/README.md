# Dose-outer superposition cell

A plate cell that sums a few weighted, shifted unit responses per observation,

```julia
plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
    sum((w[i] * get(u, dose_row(t, p, dose_table(t, p)[i]), 0.0) for i in eachindex(w));
        init = 0.0)
end
```

runs dose-outer in the native lowering from RK `5a077572`: one pass over the
observations per dose, each observation accumulating in the authored dose
order, split at the window where the lag is in range when the lag advances by
one per observation (checked per call). See `docs/src/batched.md` for the
conditions. `run.jl` measures it against the same cell through an alias of
`get` that the lowering does not recognize (the observation-outer cell loop),
the filtered spelling, and the released in-place hand loop, on a lattice plan
(`t - shift`) and on stored per-observation lag rows, plus the consumer's
32-lane position-batched read. Every RK form is asserted bitwise equal to the
cell loop before it is timed.

```sh
julia --project=<environment with ReactiveKernels> run.jl
```

## strato2, 2026-10-01

Julia 1.10.11, x86_64-linux-gnu, one thread, host load average ≈ 6 during the
run; minimum over 300 calls; bytes for one call. RK at `5a077572`.

| shape | `get` cell (dose-outer) | `get` cell, cell loop | filtered cell | hand loop |
| --- | --- | --- | --- | --- |
| lattice, 3 doses × 16321 obs | 8.7 µs / 130672 B | 35.58 µs / 130672 B | 31.66 µs / 130672 B | 7.18 µs / 130672 B |
| stored rows, 3 doses × 16321 obs | 22.74 µs / 130672 B | 98.92 µs / 130672 B | — | 24.88 µs / 130672 B |
| lattice, 3 doses × 6529 obs | 3.66 µs / 52336 B | 14.07 µs / 52336 B | 12.48 µs / 52336 B | 2.72 µs / 52336 B |
| stored rows, 3 doses × 6529 obs | 8.86 µs / 52336 B | 37.68 µs / 52336 B | — | 9.8 µs / 52336 B |
| lattice, 14 doses × 6529 obs | 9.25 µs / 52336 B | 45.06 µs / 52336 B | 42.26 µs / 52336 B | 7.68 µs / 52336 B |
| stored rows, 14 doses × 6529 obs | 37.63 µs / 52336 B | 66.36 µs / 52336 B | — | 38.22 µs / 52336 B |
| lattice, 14 doses × 16321 obs | 22.68 µs / 130672 B | 116.17 µs / 130672 B | 113.12 µs / 130672 B | 21.59 µs / 130672 B |
| stored rows, 14 doses × 16321 obs | 103.23 µs / 130672 B | 182.27 µs / 130672 B | — | 106.26 µs / 130672 B |

| 32 lanes, 14 doses × 6529 obs | `get` cell (dose-outer) | `get` cell, cell loop |
| --- | --- | --- |
| lattice | 446.36 µs / 3355696 B | 1688.83 µs / 3355696 B |
| stored rows | 1482.3 µs / 3355600 B | 2298.76 µs / 3355600 B |

These are microkernel and batched-read measurements on x86-64. They are not a
measurement of a full consumer request, and ARM64 was not measured here.
