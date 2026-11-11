# Weak-component mixture SMC

This synthetic benchmark measures posterior mode mass under sequential Monte
Carlo (SMC). It compares a single independently tuned mutation schedule with a
fixed composition of two independently tuned schedules. Both routes use the same
likelihood, priors, unconstrained coordinates and Gaussian reference.

## Reproduce

Use a consumer environment containing this checkout's ReactiveKernels,
ReactiveKernelsPPL and ReactiveKernelsDistributionKernels, plus ADTypes, Enzyme,
LogDensityProblems, SequentialMonteCarlo and WarmupHMC. The measured versions are
listed below. The program uses one Julia thread and explicit Xoshiro RNGs.

```sh
SMC_SEEDS=41,42,43 julia --startup-file=no --project=<consumer-environment> \
    benchmark/smc_weak_component/run.jl 512 8 single
SMC_SEEDS=41,42,43 julia --startup-file=no --project=<consumer-environment> \
    benchmark/smc_weak_component/run.jl 4096 8 two-pilot
```

Arguments are particle count, total moves per particle per temperature, and
`single` or `two-pilot`. Defaults are `4096 8 two-pilot`. An optional fourth
argument names an output directory for serialized schedules and SMC checkpoints;
otherwise the program only prints results. Keep that directory outside the
checkout. Checkpoint reuse requires compatible package versions and the same
model, binding and options. The benchmark is outside the documentation build.

## Target and checks

Twelve observations in two dimensions come in exact positive/negative pairs.
Two Gaussian components have ordered first-coordinate means, symmetric broad
mean priors, exchangeable exponential scales, Dirichlet(2,2) weights and positive
observation standard deviations. Component labels are marginalized in the
likelihood. All observations and prior settings are runtime model inputs.

Reflection of the means followed by component reversal preserves the target.
It exchanges the events that either ordered component carries more than half
the expected observation memberships. Their exact posterior probabilities are
therefore 0.5. The driver checks eight reflected density pairs and compares the
ordinary Enzyme gradient with centered finite differences at nine positions.
Finite differences are only a check; inference uses the ordinary AD gradient.

The output includes the event mass, weak-component mass, first mixture-weight
mean (exactly 0.5), mean midpoint and summed other-coordinate means (both zero),
and component log-scale differences (zero in each dimension). It also retains
temperature paths, ancestry, all mutation divergences and timing. These are
checks on specified quantities, not proof of complete posterior coverage.

## Frozen kernel composition

The reference is fitted to the same 1,024 prior-scale calibration positions
before any pilot. Each route uses the same 19 temperature knots,
`pilot_steps=256`, eight total NUTS moves per temperature and maximum tree depth
seven. The adaptive pilot's draw setting is a floor, not a cap on its work.
NUTS is the No-U-Turn Sampler used for each mutation transition.

`single` reproduces the original sequential adaptive pilot from means
`[-0.1, 0.1]`. `two-pilot` prepares two schedules with `grid_start=:independent`
from means `[0, 4]` and `[-4, 0]`, mixture weights `[0.8, 0.2]` and `[0.2, 0.8]`,
and corresponding occupied/empty-component scale starts. Every pilot RNG is
independent of the production RNG. No production cloud is used for tuning.

Each schedule supplies four moves. After `smc_step`, `rejuvenate_smc` applies the
other schedule at the same temperature while retaining weights and evidence.
Rejuvenation adopts that schedule for the next step, so the order alternates
A,B then B,A. Each temperature uses both schedules. The selection depends only
on the stage number. Composition of kernels preserving a common target also
preserves it; a position-dependent choice would require its own justification.

This uses existing public APIs. It does not add an automatic mode finder or
change SMC defaults. Distinct pilot starts must still be supplied and the
resulting finite-budget estimates checked on the quantities of interest.

## Diagnosis on the measured package line

The unchanged 512-particle reproduction produced exactly the original event
masses and divergence counts after updating the packages. Density symmetry
checks passed, and the maximum scaled gradient-check discrepancy was
`1.373e-6` over the nine checked positions.

The original seed-41 terminal pilot's two other-coordinate metric scales were
`0.00063075` and `0.08000002`. Reflecting a particle exchanges which component
needs the narrow scale. On 64 terminal positions and their equal-density
reflections, eight fixed terminal-kernel transitions per position gave:

| Step-size multiplier | Original: moved / 512 | Reflected: moved / 512 | Original divergences | Reflected divergences |
| --- | ---: | ---: | ---: | ---: |
| 1 | 502 | 8 | 8 | 476 |
| 1/4 | 504 | 8 | 8 | 407 |
| 1/16 | 507 | 31 | 6 | 345 |
| 1/64 | 512 | 509 | 0 | 6 |

Paired log densities differed by at most `4.264e-14`. This directly demonstrates
poor mutation in an equally probable geometry. It does not establish a density,
gradient or weight-bookkeeping defect. Shrinking the step by 64 increased the
original-side evaluation count from 3,960 to 65,024 in this audit.

The Gaussian reference is also asymmetric under reflection in the original
ordered coordinates: if `u1` is the left mean and `u2` is the log gap, reflection
maps them to `(-u1-exp(u2), u2)`. A valid reference need not share the posterior
symmetry. All reported composition comparisons retain that same reference, so
they do not isolate the benefit of changing it.

## Measurements

2026-11-11, a Linux fleet host, one CPU allocation and one Julia thread.
ReactiveKernels 0.1.169 / ReactiveKernelsPPL 0.1.159 at
`2cc9c5c342784a7a95edf44166572bdafb433bdb`, WarmupHMC 0.2.18 at
`c4d6885edefe8aa33acf02d3e571d224662825b4`, Enzyme 0.13.220 at
`e4641c6cbacc82c4cba760d440a973092e696563`, SequentialMonteCarlo 0.7.0,
Julia 1.10.12 (`09cc1b5`), libEnzyme build
`10018598600593e88b99090bb231755ad3ce5d53` for LLVM 15.

Every row uses production seeds 41/42/43 with separately prepared pilots.
The exact event mass is 0.5. Eight moves is the **total** per temperature,
including both schedules in the composed route.

| Route | Particles | Seed 41 | Seed 42 | Seed 43 | Production seconds, 41 / 42 / 43 |
| --- | ---: | ---: | ---: | ---: | --- |
| Original single pilot | 512 | 0.988281 | 0.955078 | 0.703125 | 39.8 / 39.0 / 44.0 |
| Same single-pilot kernels, quarter steps | 512 | 0.548828 | 0.787109 | 0.652344 | 169.8 / 170.5 / 178.6 |
| Two diverse pilots | 512 | 0.447266 | 0.246094 | 0.343750 | 49.8 / 44.1 / 52.2 |
| Two diverse pilots | 4,096 | 0.493408 | 0.493164 | 0.522461 | 383.5 / 308.1 / 310.4 |

The packaged driver's current-package single-pilot check at 4,096 particles,
seed 41, gave `0.98974609375` in 247.6 production seconds, exactly matching the
reporter's earlier-package result. Its final 155 divergences out of 32,768
transitions (0.47%) accompanied that incorrect mode mass. Its first-weight mean
was 0.868211 and mean midpoint 1.543998, versus exact values 0.5 and zero.
The reporter's other two large single-pilot results were 0.697021 / 0.614746
on RK 0.1.157 / RKPPL 0.1.154, WarmupHMC 0.2.15 and Enzyme 0.13.218; those two
seeds were not rerun at that budget on the current line.

One-seed controls at 512 particles gave 0.945313 with an independent-grid
single adaptive pilot, 0.392578 with a continuous pilot, and 0.482422 with the
original adaptive kernels' steps divided by 16. The quarter-step replication
shows why a favorable single seed is insufficient evidence for a remedy.

The three large composed runs reached temperature one after 20 tempering
stages. Weak-component masses were 0.996826 / 0.995850 / 0.993408. Mixture-weight
means were 0.495108 / 0.495364 / 0.516673; mean midpoints were -0.003997 /
0.062412 / 0.124565. Summed other-coordinate means were 0.000315 / -0.002691 /
-0.000984. Component log-scale differences were (-0.019433, 0.020297),
(-0.021097, 0.000061), and (-0.059422, -0.033214).

Remaining mutation divergences were 26.12% / 25.87% / 25.33%, with 278 / 253 /
277 surviving initial ancestors. Pilot times were 28.8 / 13.1 / 13.5 seconds,
including first-call compilation where applicable. The complete large-run
process exited zero after 1,154 seconds, including loading and diagnostic work.
Host load varied, so these are observed costs, not a controlled speed ranking.

The composed route at 4,096 particles supports the reported symmetry statistic
within 0.023 in these three independent clouds. It is a measured finite-budget
recipe, not a general posterior-accuracy bound. The high divergence count
remains an efficiency limitation. Neither the 512-particle composition nor the
quarter-step single-pilot route was reliable in this comparison. The successful
recipe changes pilot starts, grid initialization and kernel composition together;
it does not identify their individual contributions. Further targets and
observables require their own validation.
