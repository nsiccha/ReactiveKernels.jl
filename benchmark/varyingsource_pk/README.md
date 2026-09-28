# Emitted varying-source PK slice versus actual Stan BDF

This compares complete reverse gradients of an emitted 17-coordinate PK
density with the equivalent Stan density in `pk_bdf.stan`. Both include the
normalized 2-D GP, treatment unit responses, ordered dose feedback,
bioavailability and volume, observation likelihood, all priors, and the sigma
transform Jacobian. The GP coordinates are already scaled coefficients with
independent standard-normal priors. This slice does not contain PD/placebos,
the twin's GP innovation transform, centered subject hierarchy, or censoring.

The public synthetic fixture has 3 subjects, 8 observations and 5 doses. One
subject has no doses; other cases include interleaved axes, duplicate reads,
equal-time doses and the deployed source-column convention. A second workload
replicates the fixture ten times: 30 subjects, 80 observations and 50 doses,
with the same 17 shared coordinates and priors. Three parameter points probe
different elimination/absorption rates, including baseline Gamma shape 8.

`model.jl` lowers and binds the RK surface, replaces the bound GP vector with
a concrete `:vector_normal` parameter, and verifies all coordinate names.
Stan receives the same schedule products and a checked canonical-port to RK
coordinate permutation. Its parameters are already unconstrained; sigma's
exponential transform and Jacobian are authored once in the model. BridgeStan
retains all density constants and adds no second transform Jacobian.

Stan uses `ode_bdf_tol` with the Bruno `896137dd` transit RHS, separate ODE
and source parameter groups, zero initial amounts, unit dose and step cap
10000. Production relative/absolute tolerances are both `1e-6`; reference runs
use `1e-10` and `1e-12`. The RK emitter uses its bound defaults
`series_rtol=1e-15, watson_terms=8`; native Enzyme reverse differentiates the
emitted density and ordinary GP/feedback arithmetic, using the existing
generated transit rule. Preparation and compilation are outside the timings.

Each row times density plus the full reverse gradient. After ten warmup calls,
60 deterministically shuffled rounds take five calls per sample. Julia and
BLAS use one thread. Medians on a shared compute host are measurements, not
portable performance guarantees. `Julia_allocated_bytes` excludes C++ heap
allocation and cannot compare total allocation between Julia and Stan.

Accuracy uses the `1e-12` Stan gradient as reference and reports the maximum
absolute difference divided by the reference's maximum absolute entry. The
driver checks `1e-10` to `1e-12` convergence and density/gradient parity before
timing. A row is accuracy-matched only when its relative gradient difference
is no larger than production Stan's measured difference. The raw table also
retains absolute errors, density, coordinates and complete gradient vectors.
These checks do not prove a global bound or settle production approximation
policy; Watson truncation error remains separate from series tolerance.

Reproduce in a scratch consumer environment with ReactiveKernelsPPL, Enzyme
and DifferentiationInterface, using a disk-backed build directory:

```sh
bash benchmark/varyingsource_pk/build.sh /path/to/scratch/build
julia --startup-file=no --threads=1 --project=/path/to/consumer-env \
  benchmark/varyingsource_pk/gradients.jl \
  /path/to/scratch/build/pk_bdf_model.so /path/to/results.tsv
```

`BRIDGESTAN` and `STANC_PATH` override the build script's installed defaults.
On a managed RK compute host, hold one compute token across the build and run.

## Recorded strato2 result — 2026-09-28

Julia 1.10.11, Enzyme 0.13.205, DifferentiationInterface 0.7.21,
BridgeStan 2.9.0 / Stan 2.39.0, `znver3`, one Julia/BLAS thread, C++ `-O3`.
Native implementation commit: `f00da66a1b619477c9ab0317e463c3e858b59768`.
The combined native-regression/corpus/build/comparison command exited 0 at
`2026-09-28T21:33:52Z` after 379 seconds. The
[run receipt](receipt-strato2-20260928.txt) retains all output with trailing
whitespace normalized; the original managed scratch log is retained as well.

| Point | Subjects | RK emitted reverse (μs) | Stan BDF reverse (μs) | Stan / RK |
| --- | ---: | ---: | ---: | ---: |
| P1 | 3 | 51.17 | 1212.65 | 23.70× |
| P2 | 3 | 33.90 | 846.62 | 24.97× |
| P3-shape8 | 3 | 36.62 | 530.18 | 14.48× |
| P1 | 30 | 182.97 | 9146.58 | 49.99× |
| P2 | 30 | 202.25 | 8668.95 | 42.86× |
| P3-shape8 | 30 | 187.18 | 5100.40 | 27.25× |

Every RK row is accuracy-matched. RK/reference gradient differences are
`4.75e-11`–`1.98e-10`; production Stan/reference differences are
`3.36e-6`–`1.44e-5`. Tight Stan `1e-10`→`1e-12` changes are
`2.20e-9`–`1.00e-8`. RK/reference density differences range from `1.18e-8`
to `7.65e-6` in absolute units. The
[12-row table](results-strato2-20260928.tsv) retains every density, gradient,
parameter point, accuracy measure and allocation observation. This is a PK
slice on the stated small synthetic grids; real-fit dimensions and the full
PK/PD posterior remain separate work.

Related checks passed: 381 focused transit/native-cell/emitter assertions,
then the native cell's 56 assertions including two added reverse checks with
constant GP coefficients and active slopes; 1954 existing contract/layout/generator/query/surface/
corpus assertions; 245 native grouped/event-LP assertions; and the updated
corpus's 233 assertions. Counts overlap where a targeted file was rerun.
The scratch-only driver excludes the Reactant seam and its macro wrapper;
the new cell makes no compiled-Reactant claim. No existing golden was changed.
