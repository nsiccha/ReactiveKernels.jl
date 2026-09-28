# Two-compartment transit reverse-gradient comparison

This benchmark compares three paths for the same scalar: the sum of central
unit-dose amounts on a deterministic 1401-lag grid. The five differentiated
linear coordinates are `[k10, k12, k21, rate, shape]`.

- `RK-Enzyme`: reverse differentiation through the transparent recurrence.
- `RK-rule-Enzyme`: the same numerical expression, with analytic partials
  retained by one RK mathematical graph and contracted by its generated rule.
- `Stan-BDF`: actual Stan `ode_bdf_tol`, with the Bruno posterior's RHS,
  separate ODE/source parameter arguments, zero initial state, unit dose,
  relative/absolute tolerances `1e-6`, and `max_num_steps=10000`.

The Stan extraction follows `stan/varyingsource3.stan`, lines 2154–2212, at
Bruno commit `896137dd5853a08810e311a921c0c4a3c84e0331`. It retains the source's
`log(t + 1e-16)`. The reduced Stan model has identity parameter coordinates,
no priors, and no transform Jacobian. This measures one unit solve, rather
than a complete posterior gradient.

Both RK paths expose the series convergence tolerance and Watson term count.
The generated rule binds these numerical controls during preparation and
keeps the lag grid and PK parameters as its runtime inputs.
The default strict setting is `1e-15` with eight Watson terms. The benchmark
also measures 24 terms to reduce the separate asymptotic truncation error;
a small series tolerance alone does not guarantee floating-point accuracy
of the whole expression. Looser settings are measured as candidates,
without choosing a production approximation policy.

## Reproduce

Use a consumer environment containing local ReactiveKernelsPPL, Enzyme,
DifferentiationInterface, and their dependencies. Build with an installed
BridgeStan checkout and a disk-backed scratch output directory:

```sh
BRIDGESTAN=/path/to/bridgestan bash benchmark/transit_twocmt/build.sh /scratch/transit-build
julia --startup-file=no --threads=1 --project=/path/to/consumer-env \
  benchmark/transit_twocmt/gradients.jl \
  /scratch/transit-build/unit_bdf_model.so /scratch/transit-results.tsv
```

On managed KB hosts, acquire a compute token around each command. The shared
library is called through BridgeStan's C API from the same Julia process.
Preparation and warmup precede timing; one sample batches five complete
reverse-gradient calls. The script takes 60 samples per path, shuffles path
order each round, and reports medians. Julia and BLAS each use one thread.

Gradient error is `max(abs(g - reference)) / max(abs(reference))`, where the
reference is Stan BDF at `1e-12` tolerances. The script checks convergence
against a second Stan solve at `1e-10`. `matched` means error no larger than
production Stan BDF's error on the same case. This is an empirical gradient
criterion, not a global error bound. The full five-coordinate gradients and
absolute errors are saved in the TSV.

`Julia_allocated_bytes` counts allocations visible to Julia; it does not count
Stan's C++ heap. Primal-only calls and compilation time are not timed.

## Measured results, 2026-09-28

The complete run on strato2 exited `0`; all 292 focused assertions passed
before the Stan build and benchmark. Source commit: `474c1b26`. The run used
Julia 1.10.11, Enzyme 0.13.205, DifferentiationInterface 0.7.21, BridgeStan
2.9.0, Stan/stanc 2.39.0, and the `znver3` CPU target. C++ used `-O3`;
Stan threads, MPI, and OpenCL were disabled. The 1401 lags span
`0.031383334164729604` to `665.7492378410009`.

| Case | k10 | k12 | k21 | rate | shape |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1 | 0.08 | 0.15 | 0.05 | 0.20 | 1.20 |
| P2 | 0.50 | 0.30 | 0.40 | 0.05 | 1.05 |
| P3-shape8 | 0.08 | 0.15 | 0.05 | 0.30 | 8.00 |

At `series_rtol=1e-15`, `watson_terms=24`, median complete gradient times
in microseconds were:

| Case | RK + Enzyme | RK rule + Enzyme | Stan BDF | Rule speedup over Stan |
| --- | ---: | ---: | ---: | ---: |
| P1 | 3030.83 | 554.31 | 1145.29 | 2.07× |
| P2 | 1670.24 | 408.43 | 952.83 | 2.33× |
| P3-shape8 | 1863.18 | 368.49 | 954.62 | 2.59× |

The generated rule is 4.1–5.5× faster than transparent Enzyme at this
setting. Relative gradient differences from the tight Stan reference are
`5.32e-11`, `1.74e-10`, and `6.30e-11`, respectively. Production Stan's
differences are `1.06e-6`, `4.50e-5`, and `4.29e-6`. The default eight-term
setting is also recorded in the raw results.

The fastest tested setting meeting production Stan's measured accuracy
on each case used four Watson terms:

| Case | Series tolerance | RK + Enzyme (μs) | RK rule + Enzyme (μs) | Stan BDF (μs) | RK relative error | Rule speedup over Stan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | 1e-7 | 2474.66 | 426.60 | 1145.29 | 2.45e-7 | 2.68× |
| P2 | 1e-6 | 1225.28 | 249.42 | 952.83 | 6.48e-6 | 3.82× |
| P3-shape8 | 1e-6 | 1476.68 | 265.57 | 954.62 | 3.37e-6 | 3.59× |

The `1e-5` setting failed the accuracy criterion on all three cases;
`1e-6` also failed on P1. These per-case choices are measurements, not a
production setting or a global guarantee. The plain Enzyme path remains
slower than Stan on every case, including these matched settings.

The `1e-10` to `1e-12` Stan reference changes were `2.40e-9`, `8.83e-9`,
and `3.72e-9`. They are small relative to production Stan's error, but they
do not certify floating-point exactness of the RK result. Tight RK/reference
differences above are numerical comparisons, not absolute error bounds.
Shared-host timing medians and this unit-solve scope also do not establish
full-posterior performance.

All 39 rows, including absolute errors, allocations, and full gradients, are
in [results-strato2-20260928.tsv](results-strato2-20260928.tsv). The complete
test/build/run receipt, with trailing whitespace normalized, is
[receipt-strato2-20260928.txt](receipt-strato2-20260928.txt).
