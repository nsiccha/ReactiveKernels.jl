# Full native varying-source PK/PD posterior versus original Stan

This compares the complete native emitted posterior gradient with the original
Bruno `896137dd` `varyingsource3.stan` program. The source is supplied by the
caller and SHA-256 checked by `build.sh`; it is not replaced by a simplified
Stan model. The reference variants change only the two BDF tolerance pairs.

The public synthetic fixture has 3 subjects, 10 observations and 7 doses. It
covers all three assays, all five vessels and four diet levels, equal-time
doses, duplicate PK reads, and a dose-free subject with an initial-only PBMC
observation. Dose rows stay grouped by subject, as required by the original
program and the BRM bridge. The native cell also supports interleaved dose
rows; separate native tests cover those. The larger workload repeats this
fixture ten times: 30 subjects, 100 observations and 70 doses.

Both sides include all 13 correlated centered subject margins, their LKJ(2)
correlation prior and exponential scale priors, population intercepts,
disease and sex/age/weight effects, three vessel/diet dose predictors with
monotonic simplexes, effectiveness GP innovations, both placebo HSGPs, six
assay observation scales, censoring and all transform Jacobians. The GP is
4×4; each placebo has ten basis terms. There are 232 unconstrained coordinates
with 3 subjects and 583 with 30. Julia's `Exponential` arguments are scales:
the original Stan rates 1.5 and 4 become scales `1/1.5` and `.25`.

Three deterministic parameter points vary coefficients and innovations. The
third raises the absorption mode to probe Gamma shape near eight. Covariates,
doses and responses are synthetic; this is not a benchmark of a fitted
patient dataset or its posterior samples.

`stan.jl` maps every original constrained parameter exactly once, checks its
unconstraining against the independent pure Julia coordinate map, and pulls
the complete Stan gradient back with ordinary Enzyme. Centered effects shift
by the population mean; covariate coefficients and stable-LKJ partials rescale;
the three simplexes transport from RK stick breaking into Stan's Helmert ILR.
The log-density difference is a parameter-independent constant:

```julia
lkj_logconst(13, 2.) - sum(log, PKPD_SCALE0) - 6log(.1) - 1.5log(3.)
```

The driver checks that constant and the likelihood independently, together
with full gradient parity. It checks BDF `1e-10`→`1e-12` convergence before
timing the production `1e-6` program. RK binds `series_rtol=1e-15,
watson_terms=8`; Watson truncation error remains separate. Accuracy reports
the largest absolute gradient difference divided by the reference gradient's
largest absolute entry. A timing is accuracy-matched only if RK's measured
error is no larger than production Stan's.

Each timed call includes density plus the full reverse gradient in that
engine's own coordinates. Preparation, compilation, coordinate transport and
generated quantities are outside timings. Stan retains all density constants
and includes its transform Jacobians. The raw table stores the complete
transported gradients. After ten warmup calls, 60 deterministically shuffled
rounds take five calls per sample, with one Julia/BLAS thread. Julia allocation
counts exclude C++ heap allocation. Shared-host timings are measurements, not
portable performance guarantees.

The driver retains one full batched cell call and one centered-prior call at
both subject counts. Its code-node check counts the bound integer grouping
level table once: preparation folds this data-only encoder. Raw expression
sizes include 3 or 30 level literals. The native cell and centered-prior tests
also verify identical LLVM structure at both group counts; the loops stay
dynamic.

Reproduce with a consumer environment containing ReactiveKernelsPPL, Enzyme,
DifferentiationInterface and their test dependencies:

```sh
bash benchmark/varyingsource_pkpd/build.sh \
  /path/to/Bruno/web-pkpd/stan/varyingsource3.stan /path/to/scratch/build
julia --startup-file=no --threads=1 --project=/path/to/consumer-env \
  benchmark/varyingsource_pkpd/gradients.jl /path/to/scratch/build \
  /path/to/results.tsv
```

Use the original file path in your Bruno checkout; the illustrative path above
may differ. `BRIDGESTAN` and `STANC_PATH` override the installed build defaults.
On managed compute hosts, hold one compute token across the build and run.
The requested full-data Mac benchmark remains separate from this synthetic
comparison. See [the consumer contract](../../docs/src/varyingsource-pkpd.md).

## Primal-only output selection, 2026-09-29

`primal.jl` runs the same six posterior points without loading Enzyme. It
checks the coordinate map, complete density and likelihood against the original
Stan binaries at all three tolerances. Schedule construction, binding and query
preparation precede the timed evaluations. Each measured call computes the full
unconstrained log density, including constants and transform Jacobians.

```sh
julia --startup-file=no --threads=1 --project=/path/to/consumer-env \
  benchmark/varyingsource_pkpd/primal.jl /path/to/ReactiveKernels.jl \
  /path/to/scratch/build /path/to/primal-results.tsv
```

The baseline is `475afe61`, published in `3cdba95c`; the new source is
`eaeef55d`. The new mathematical rule offers a value-only recipe and a joint
value/partials recipe. Generic rule lowering accepts both, allowing the planner
to select the value-only recipe for ordinary evaluation. No posterior-specific
optimizer or cache was added. The same-process primitive comparison and
operation-count regressions are documented in
[the transit benchmark](../transit_twocmt/README.md).

Full-posterior median microseconds on strato2:

| Subjects | Point | Baseline RK primal | New RK primal | Baseline / new | Production Stan in new run |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 93.059 | 79.095 | 1.18× | 456.293 |
| 3 | 2 | 70.261 | 52.584 | 1.34× | 408.091 |
| 3 | 3 | 64.087 | 46.997 | 1.36× | 252.770 |
| 30 | 1 | 397.857 | 273.266 | 1.46× | 3608.468 |
| 30 | 2 | 400.797 | 274.752 | 1.46× | 3710.373 |
| 30 | 3 | 398.910 | 275.001 | 1.45× | 2194.743 |

These before/after full-model timings are separate runs on a shared host;
the same-process paired primitive benchmark is the stronger attribution check.
Both full-model runs used ten warmups and sixty shuffled five-call samples,
one Julia/BLAS thread, Julia 1.10.11 and the same Stan binaries as below. Baseline
`TFqMQ8` exited 0 in 259 seconds; new run `cj5q9H` exited 0 in 246 seconds.
No Enzyme module was loaded in either run. All 24 old/new exported density values
and coordinate vectors are identical. New Julia allocations are 92,384 bytes
at 3 subjects and 193,296 at 30, down from 99,488 and 264,336. Stan's Julia
allocation counter excludes its C++ heap.

The [baseline table](primal-baseline-strato2-20260929.tsv) and
[new table](primal-joint-strato2-20260929.tsv) include medians, IQRs,
coordinates and density errors. Exact exits are in the
[baseline receipt](receipt-primal-baseline-strato2-20260929.txt) and
[new receipt](receipt-primal-joint-strato2-20260929.txt). The new driver differs
from the executed scratch driver only in its usage-message filename.

This removes derivative-only work from the primitive primal. Shared GP work at
that checkpoint still ran inside the posterior's subject loop; the following
section records its factoring into shared graph nodes. Bruno's app uses its own
mathematical graph and workload; its preparation-inclusive results are a
separate acceptance check.

The unchanged full-gradient driver also passed at `eaeef55d`: run `3Bgath`
exited 0 in 386 seconds. All twelve exported coordinate vectors, gradients,
reference gradients and densities are identical to the original gradient run.
All six RK cases retain the accuracy-matched verdict. Julia allocations remain
258,128/703,792 bytes at 3/30 subjects. Current RK medians are
274.84/312.34/256.10 μs at 3 subjects and 1713.78/1667.54/1344.21 μs at 30.
Stan's timings also changed between runs, so these are regression receipts,
not evidence for a gradient speedup from this change. See
[the new gradient table](gradients-joint-strato2-20260929.tsv),
[exact receipt](receipt-gradients-joint-strato2-20260929.txt), and
[artifact/source hashes](joint-validation.json).

## Shared GP mathematical dependencies

Runtime checkpoint `cd3ed22a` starts from published `ed98d70c`. The emitter
exposes effectiveness weights, its scalar normalization, the primary placebo
course and a CSF-only secondary course as ordinary RK graph nodes. The existing
planner shares live producers once per evaluation and folds fully bound
producers during preparation. Subject-specific GP predictors retain the general
reader. The same subject/grid traversal consumes these mathematical results;
no persistent cache or new AD adapter is introduced.

The full sampler has live GP parameters. Its GP preparation counters are zero;
the domain-independent fixture in `test/test_shared_preparation.jl` separately
checks once-per-preparation basis work, fully bound weights/normalization,
changed live inputs, rebinding and WANT pruning. It passes 33 assertions.
The shared reader retains no-dose and empty-course guards, and secondary
preparation visits only CSF times. Both reader forms retain identical LLVM
across 3/30-subject data, with runtime loop and branch instructions. Rebinding
also leaves the executable emitted graph size fixed. Opaque helper internals
remain ordinary Julia: the placebo sine basis still runs over the time grid
inside each course producer.

Primal-only counters instrument the owned mathematical functions after timing,
including view-backed innovation vectors. The actual full sampler calls are:

| Subjects | Baseline weights | Shared weights | Baseline normalization | Shared normalization | Baseline placebo courses | Shared placebo courses |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2 | 1 | 2 | 1 | 5 | 2 |
| 30 | 20 | 1 | 20 | 1 | 50 | 2 |

All three GP preparation counters are zero in both runs because these parameters
are live. Changing an unconstrained input preserves the evaluation counts.
See the [baseline counts](counts-shared-gp-baseline-strato2-20260929.tsv)
and [shared counts](counts-shared-gp-strato2-20260929.tsv).

All six full-posterior densities and coordinate vectors are identical to the
published joint baseline. The largest old/new gradient difference is
`1.7053025658242404e-13`; all six RK cases retain the accuracy-matched verdict
against the original three Stan tolerances. Gradient Julia allocations fall
from 258,128 to 221,616 bytes at 3 subjects and from 703,792 to 630,608 at 30.
The unchanged full gradient driver exits 0 in 741 seconds. See
[raw gradients](gradients-shared-gp-strato2-20260929.tsv) and the
[exact receipt](receipt-shared-gp-gradients-strato2-20260929.txt).

`shared_gp.jl` measures one initial query preparation plus first evaluation per
subject size, then the six steady posterior points. Model construction and
schedule binding are outside the initial measurement; bound producers, lowering,
native query creation and first-call generated-code compilation are inside.
Both runs visit 3 subjects before 30, so the second size shares compilation of
ordinary package methods. These are single initial measurements, not medians:

| Subjects | Baseline initial seconds | Shared initial seconds | Baseline Julia bytes | Shared Julia bytes |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 40.1958 | 60.6942 | 1,420,898,680 | 1,439,343,928 |
| 30 | 29.5208 | 39.3184 | 673,981,832 | 652,510,392 |

The prepared callable is then specialized once and sampled with three warmups
and 31 samples of five calls per point. No Enzyme module is loaded. This call
path differs from the older `primal-joint` table's per-call world-age barrier;
compare the following two same-protocol tables for attribution:

| Subjects | Point | Baseline steady μs | Shared steady μs | Baseline Julia bytes | Shared Julia bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 32.1404 | 46.6942 | 26,592 | 26,192 |
| 3 | 2 | 32.6490 | 31.7516 | 26,592 | 26,192 |
| 3 | 3 | 32.5170 | 31.4654 | 26,592 | 26,192 |
| 30 | 1 | 318.5976 | 296.2990 | 127,504 | 121,136 |
| 30 | 2 | 281.9350 | 284.5146 | 127,504 | 121,136 |
| 30 | 3 | 256.8946 | 278.7708 | 127,504 | 121,136 |

These separate runs experienced shared-host contention. Initial time rises and
steady timings move in both directions; the measurements do not establish a
wall-time speedup. Actual operation counts and allocation reductions establish
the removed repeated mathematical work. The raw tables retain IQRs, all
coordinates and densities. The comparison script checks those identities and
the complete old/new gradients against the published joint gradient baseline.

The [baseline primal table](primal-shared-gp-baseline-strato2-20260929.tsv)
and [shared primal table](primal-shared-gp-strato2-20260929.tsv) contain both
initial and steady measurements. Their exact receipts are
[baseline](receipt-shared-gp-baseline-primal-strato2-20260929.txt), exit 0/315s,
and [shared](receipt-shared-gp-primal-strato2-20260929.txt), exit 0/416s.
The [comparison output](shared-gp-comparison.json) records checked values,
gradient deltas, counts and before/after costs. Reproduce each primal run with
its own consumer environment resolved to the indicated repository root, then
compare the resulting artifacts:

```sh
julia --startup-file=no --threads=1 --project=/path/to/consumer-env \
  benchmark/varyingsource_pkpd/shared_gp.jl /path/to/ReactiveKernels.jl \
  /path/to/primal.tsv /path/to/counts.tsv
python3 benchmark/varyingsource_pkpd/compare_shared_gp.py \
  /path/to/primal-before.tsv /path/to/primal-after.tsv \
  /path/to/gradients-before.tsv /path/to/gradients-after.tsv \
  /path/to/counts-before.tsv /path/to/counts-after.tsv
```

Focused successful receipts cover
[bound/live emitted reverse](receipt-shared-gp-emitter-strato2-20260929.txt),
[shared values, lazy arms and subject-specific fallback](receipt-shared-gp-shared-tests-strato2-20260929.txt),
and [generic dependencies, raw/shared LLVM and both actual MutatingFunctions extension loads](receipt-shared-gp-core-loops-extensions-strato2-20260929.txt).
The [identity record](shared-gp-validation.json) records runtime, source and
Stan-binary hashes. Reactant acceptance and Bruno's real app workload remain
separate acceptance lanes.

## Original gradient results

The complete run exited 0 in 402 seconds at 2026-09-29T01:32:50Z on strato2
(AMD EPYC-Milan / Julia `znver3`). Julia 1.10.11, Enzyme 0.13.205,
DifferentiationInterface 0.7.21, Stan 2.39.0 and BridgeStan 2.9.0 were used;
C++ builds used `-O3`. See [the run receipt](receipt.txt),
[build receipts](build-receipt.txt), [raw coordinates and gradients](gradients.tsv),
and [verification notes](validation.md).

Median density-plus-reverse times in microseconds:

| Subjects | Coordinates | Point | RK emitted Enzyme | Stan BDF | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 232 | 1 | 334.54 | 2005.77 | 6.00× |
| 3 | 232 | 2 | 383.14 | 2083.15 | 5.44× |
| 3 | 232 | 3 | 357.83 | 1064.16 | 2.97× |
| 30 | 583 | 1 | 2166.16 | 21709.26 | 10.02× |
| 30 | 583 | 2 | 2118.66 | 20662.18 | 9.75× |
| 30 | 583 | 3 | 1511.73 | 5678.05 | 3.76× |

All six points matched production Stan's measured gradient accuracy.
The selected-source Gamma shape ranges on points 1/2 are 1.106–1.139.
Point 3 increases mode 56-fold because shape is `1 + rate*mode` with
baseline rate `1/8`; its realized ranges are 6.917–8.378 at 3 subjects
and 6.852–8.807 at 30. The driver reports and checks these ranges.

Normalized full-gradient differences:

| Subjects | Point | RK / reference | Production Stan / reference | BDF 1e-10 / 1e-12 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 1.530e-10 | 2.073e-6 | 1.249e-9 |
| 3 | 2 | 1.560e-10 | 1.938e-6 | 9.090e-10 |
| 3 | 3 | 7.485e-8 | 6.929e-4 | 2.244e-6 |
| 30 | 1 | 1.537e-10 | 1.788e-6 | 8.215e-10 |
| 30 | 2 | 1.565e-10 | 1.950e-6 | 6.387e-10 |
| 30 | 3 | 7.285e-8 | 6.251e-4 | 2.518e-6 |

The shape-eight cases have weaker reference convergence: the last column
exceeds the RK/reference difference. These are measured differences against
the tight reference, not a certified bound on true gradient error. Convergence
still improves by more than tenfold relative to production Stan on every
case. The density offsets and likelihoods also passed the independent checks.

Native calls allocated 258,128 Julia bytes at 3 subjects and 703,792 at 30.
The Stan C call allocated 208 Julia bytes; its C++ heap allocation is outside
that count. All exported post-timing gradients independently reproduce the
reported difference metrics.
