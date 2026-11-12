# Generated plate reverse with buffer reuse

`nonallocating_plate_reverse.jl` compares two synthetic Gaussian models: a
grouped regression with indexed latent effects, and a matrix regression. It
crosses ordinary/non-allocating preparation with generated plate reverse
disabled/enabled at runtime observation counts 2,000 and 100,000. Each row
checks its value and gradient against the ordinary, rule-disabled result.

Run it with this checkout and its nested PPL package developed in an
environment containing MutatingFunctions, DifferentiationInterface, Enzyme,
and BenchmarkTools:

```sh
julia --startup-file=no --project=/path/to/environment benchmark/nonallocating_plate_reverse.jl
```

`RK_COMPOSE_SIZES=2000` selects only the smaller runtime input. The benchmark
prints package paths and versions, backend library, preparation and first-call
costs, hot time and allocations, static eligibility, and inference findings.
After warming all four variants it records twelve paired rounds with rotating
and reversed execution order, including every timing sample.
The direct prepared-program audit must pass. The sampler audits also print
backend initialization and world-age boundaries rather than hiding them.

## Experiment before canonical integration

The composition implementation was recorded as `9e67adcf0`, based on
`42e134aa8`. The measurement process additionally loaded the exact adapter
`reverse` method from `8ba7c282c`, which removes captured-variable boxing.
That diagnostic replacement was not part of the composition commit. The
benchmark file now runs the installed implementation without replacements;
these historical measurements must not be attributed to an unpatched adapter.

Julia 1.10.12; Enzyme source `720f5124043473b215c0d87f8e5df2474e64bc0e`;
libEnzyme `4990aa111a76fc2b0bedd247fa5dfab8957c8677`, LLVM 15; gordito,
one Julia and BLAS thread. Time is the minimum of 30 samples with one
evaluation each. Shared-host scheduling makes these directional observations,
not stable speedup promises. All 16 value/gradient comparisons passed.

| Model | N | Ordinary, off (µs/B) | Ordinary, on (µs/B) | Reuse, off (µs/B) | Reuse, on (µs/B) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grouped | 2,000 | 11.950 / 0 | 3.520 / 96 | 8.750 / 0 | 3.730 / 96 |
| Matrix | 2,000 | 10.590 / 32,256 | 8.160 / 32,256 | 5.900 / 0 | 5.360 / 0 |
| Grouped | 100,000 | 625.168 / 0 | 214.889 / 96 | 479.129 / 0 | 177.999 / 96 |
| Matrix | 100,000 | 640.288 / 1,600,096 | 407.728 / 1,600,096 | 428.028 / 0 | 268.559 / 0 |

At 2,000 rows, reuse/rule-on preparation took 0.512 s for the grouped model
and 0.249 s for the matrix model; first gradients took 1.704 s and 1.929 s.
At 100,000 rows, after the compiler had seen the same model and argument
types, those preparations took 93.5 ms and 47.4 ms. These are ordered,
same-process measurements, not independent cold-start comparisons. Julia's
compilation counter does not include all backend compilation. They do not
establish negligible preparation cost or a globally optimal joint program.

The recursive gradient audit below the sampler's world-age boundary reported
one identical non-concrete Enzyme lazy-compilation initialization call in each
of the eight smaller cases. It reported no RK finding. The outer sampler
also reported its deliberate `invokelatest` boundary. These results are not
a claim that every reachable backend initialization path is inferred.

The final run exited 0 on 2026-11-12 at 19:13:26 UTC. Its correctness phase
passed 768 assertions: 537 composition, marker safety and changed-shape checks,
plus 231 existing summed-plate reverse checks. Before the marker extension,
the broader focused run passed 1,192 assertions with one existing broken test.
On registered Enzyme 0.13.219, the earlier broader run had scan activity,
mutability and nested-AD failures; only the scan failure was directly
reproduced against the original lowering in this lane. The successful result
uses the pinned backend build above, not the registered version.

## Scope

The implementation retains plate boundaries through cache analysis and
installs generated reverses after activity and ownership are determined.
It preserves the existing AD-owned cache copies. It does not implement a
joint tape/storage optimizer or choose save-versus-recompute plans. Static
eligibility is distinct from backend rule engagement. Tests inspect installed
rules, gradients at changed points, empty inputs, changed cache shapes,
caller ownership, and bounded emitted node/loop counts.

Canonical integration exposed a preparation-order limit: native direct reads
and factored sums emitted while the rule is disabled are not all covered by
the generated reverse. Prepare the kernel and AD in the enabled scope to
retain eligible loops. Enabling only at AD preparation can use backend
fallback. Changed-shape numerical checks retain that late-enable case;
it is not counted as evidence of generated-rule engagement. The report
describes the prepared representation and is independent of the flag at the
time the report is queried.

The first integrated run (`e8e0d055e` plus its recorded overlay), on strato2,
exited 1: 1,576 checks passed,
28 eligibility/engagement assertions failed, and one existing test was
broken. Those failures exposed the preparation-order distinction above;
the numerical checks passed. All 16 benchmark value/gradient comparisons
and all eight direct prepared-program inference audits passed. Without
the adapter boxing repair, opted-in grouped gradients allocated 1,680 B
(43 allocations) and matrix gradients 704 B (25 allocations) in the reuse
mode. Do not substitute the earlier diagnostic-patched numbers for this
unmodified-candidate result.

## Integrated composition and adapter verification

The follow-up used `1ea1cb291` (canonical base `c9f43b5d0`, composition,
and the exact `8ba7c282c` adapter repair) plus the corrected acceptance tests
and benchmark. It exited 0 at 20:31:27 UTC on 2026-11-12 after 1,643 seconds:
566 composition/marker/changed-shape assertions and 231 existing plate-reverse
assertions passed. All 16 benchmark comparisons passed, and all eight direct
prepared-program audits had zero findings. The broader gradient audits still
reported the same Enzyme lazy-initialization call described above.

This run used strato2, Julia 1.10.12 build `d93beab124c`, the same Enzyme and
libEnzyme pins above, and one CPU of time. The resolved OpenBLAS_jll version
was 0.3.23+5. Host load was 6.30/6.18/6.85 at admission and 5.55/6.57/6.58 at
release on eight logical CPUs. Its absolute times are not comparable with
the earlier gordito experiment.

Each table time is the median of twelve rounds, each itself the median of
30 one-evaluation samples. The ratio is the median of the twelve paired
rule-on/rule-off ratios. Both columns use buffer reuse.

| Model | N | Rule off (µs) | Rule on (µs) | Paired on/off | On faster | On bytes/allocations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Grouped | 2,000 | 3.825 | 4.879 | 1.281 | 0/12 | 96 / 2 |
| Grouped | 100,000 | 194.902 | 247.487 | 1.259 | 0/12 | 96 / 2 |
| Matrix | 2,000 | 4.519 | 4.586 | 1.028 | 4/12 | 0 / 0 |
| Matrix | 100,000 | 273.905 | 272.291 | 0.997 | 7/12 | 0 / 0 |

The paired ratio ranges were 1.102–2.042, 1.206–1.552, 0.756–1.170, and
0.865–1.062 respectively. The grouped case consistently favored the current
ordinary path; matrix timings were mixed. The adapter repair removes the
observed boxing allocations, but this composition increment does not establish
a speedup over the newer native optimizations. The sibling's fused-loop
factoring/direct-read changes were not included.

At N=2,000, reuse/rule-on preparation took 0.833 s (0.757 s recorded Julia
compilation) for grouped and 0.614 s (0.568 s compilation) for matrix. First
gradients took 1.872 s and 1.550 s. At N=100,000, after warming the same
types, preparation took 84.3 ms and 44.9 ms. Ordering and backend-compilation
limits still apply; these measurements do not show negligible preparation.
