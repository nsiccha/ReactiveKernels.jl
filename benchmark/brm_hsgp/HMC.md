# Exact motorcycle model: native and Reactant HMC

Reactant is **about 2.3× faster at 1,000 transitions per batch** in both measured
coordinate frames. These are RK implementations: native uses prepared Enzyme;
Reactant compiles Enzyme differentiation together with the existing authored
multinomial HMC loop. BRM/Turing and Stan are the previously verified density
references, not samplers in this comparison.

## Warmed throughput

Both backends use 16 leapfrog steps per transition and step size 0.03. Times
are median **microseconds per transition**, followed by the observed minimum
and maximum across nine timed batches. Ranges are observations, not confidence
intervals. Batches run through the state-preserving prepared interface,
including its state copies and final output snapshots. Reactant synchronizes
once per batch. RNG construction and host validation are outside timing.

| Frame | Transitions/batch | Native + Enzyme, μs | Reactant + Enzyme, μs | Native / Reactant |
|---|---:|---:|---:|---:|
| Noncentered | 4 | 239.47 [222.03–279.69] | 111.81 [71.71–765.53] | 2.14× |
| Noncentered | 100 | 144.11 [141.42–173.02] | 66.63 [55.30–216.41] | 2.16× |
| Noncentered | 1,000 | 200.22 [160.79–276.32] | 85.49 [66.53–118.24] | 2.34× |
| Selected partial | 4 | 162.92 [154.09–198.04] | 62.15 [60.24–65.17] | 2.62× |
| Selected partial | 100 | 160.53 [139.88–202.28] | 49.09 [47.47–55.54] | 3.27× |
| Selected partial | 1,000 | 158.57 [145.19–228.41] | 67.58 [49.09–95.91] | 2.35× |

For 1,000 transitions, this is **200.22 vs 85.49 ms** in NCP coordinates and
**158.57 vs 67.58 ms** in selected-partial coordinates, each doing 16,000
leapfrog steps. Median Julia allocations per batch are 203,810,048 bytes native
and 8,064 bytes Reactant. The latter excludes C++/device allocations; it is not
a total-memory comparison.

The host was shared, with each fresh process pinned to CPU6, Julia 1.10.11,
one Julia thread, and one BLAS thread. Variation is substantial, so these data
support the throughput advantage more strongly than a precise scaling curve.
The earlier three-batch-warmup pilot is retained below: its 1,000-transition
medians were 147.70/45.44 μs (NCP native/Reactant) and 137.07/53.61 μs (partial).
Short-batch startup drift motivated the final protocol: collect compilation
garbage once, then warm for at least one second and three batches. Timed rounds
include normal GC costs. The final table uses every round of that protocol.

## Preparation and first execution

These are seconds from the first NCP case in each independent process. Later
cases benefit from initialization in the same process.

| Stage | Native process | Reactant process |
|---|---:|---:|
| Bind data / prepare density | 2.681 | 3.168 |
| Prepare native Enzyme AD | 1.525 | 1.569 |
| First native value + gradient | 19.887 | 21.544 |
| Prepare HMC (includes Reactant compilation) | 14.754 | 66.741 |
| First four-transition batch | 0.959 | 0.382 |

Subsequent HMC preparations took 0.006–0.009 s native and 2.04–4.36 s Reactant.
Package imports and the explicit throughput warmup are additional costs, not
included in this stage table or the warmed throughput table.

## Shared work and validation

The new reusable helper is
[`HMCBenchmark.prepare_hmc` / `benchmark_hmc`](../sampler_transpiler/hmc_benchmark.jl).
The existing Eight Schools prepared consumer and timing driver now use it too.
It remains benchmark support outside the package API. The unchanged
[`position_multinomial_hmc_kernel.jl`](../sampler_transpiler/position_multinomial_hmc_kernel.jl)
owns momentum refreshes, random forward/backward trajectory splitting, leapfrog
integration, and streaming multinomial proposal selection. There is no separate
hand-written Reactant sampler or model.

The exact 44-coordinate motorcycle density, normalized priors, Jacobians, and
all 133 observations are unchanged from the 40,072-point BRM/Turing/Stan parity
verification. Data-only design work is folded once; AD preparation marks only
`q` active. Centeredness remains an input of the density but is held fixed in
the sampler's callback context. Each frame starts at its supplied posterior
column 5,000 and uses a fixed diagonal mass equal to inverse coordinate variance
over its supplied 10,000 draws. Both backends receive identical positions,
centeredness, mass, step size, and leapfrog/batch counts.

All 108 timed final positions and densities were finite, and all timed batches
moved. All 48 additional continued batches were finite and moved. Native uses
Xoshiro; Reactant uses ReactantRNG. No cross-backend trajectory equality gate
was imposed. The existing focused prepared-interface probe also passed all
43 checks, covering scalar/vector behavior, HMC continuation, replay, input
preservation, and invalid RNG/state contracts.

These are fixed-work HMC throughput measurements. There is no timed adaptation,
sample-history storage, acceptance diagnostic, ESS, or mixing-efficiency claim.
The fixed mass is posterior-informed benchmark setup. Compiling gradients into
the sampling loop avoids a standalone host/compiler boundary at every
leapfrog step; the earlier isolated-gradient latency does not predict this
whole-loop throughput.

## Receipts and reproduction

Measured implementation: `24e3b4f67a979e3159a0bcaca9389792b820c3ad`, clean worktree.
Reactant 0.2.284, Enzyme 0.13.204, DifferentiationInterface 0.7.21. The receipts
record source/data hashes, package versions and local source revisions, initial
positions, centeredness, metrics, timestamps, load averages, raw timings, GC,
Julia allocation bytes, and final-density/movement checks.

- [Native final receipt](../receipts/brm-hsgp-hmc-native-v1.toml)
- [Reactant final receipt](../receipts/brm-hsgp-hmc-reactant-v1.toml)
- [Native pilot](../receipts/brm-hsgp-hmc-native-pilot.toml)
- [Reactant pilot](../receipts/brm-hsgp-hmc-reactant-pilot.toml)

See [README.md](README.md#native-and-reactant-hmc) for setup and exact runner
arguments. Final commands used `HMC_BACKEND=native` and then `reactant`, with
`BRM_BUNDLE=$TMPDIR/brm-benchmark`, default work counts, and
`JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 6 julia --startup-file=no
--project=benchmark/brm_hsgp benchmark/brm_hsgp/hmc.jl`.

Both commands exited 0: native 71 s (`kb-run-compact.s4mVxG`), Reactant 141 s
(`kb-run-compact.P61x3Y`). The focused interface command was
`taskset -c 7 julia --startup-file=no --project=benchmark/sampler_transpiler
benchmark/sampler_transpiler/prepared_interface_probe.jl`, exit 0 in 240 s
(`kb-run-compact.HWPNAl`). No blanket `Pkg.test` or server restart was used.
