# Exact BRM motorcycle HSGP benchmark

The model is authored once in [`examples/brm_hsgp.jl`](../../examples/brm_hsgp.jl).
It uses ordinary `@kernel` recipes and an observation `plate`. `prepare(...;
bound=...)` folds the data-only basis and squared frequencies; the 44-vector
`q` and 40-vector of centeredness controls remain live. The benchmark separately
prepares native Enzyme AD, compiles the primal through Reactant, and compiles
value plus gradient through `compile_ad_value_and_gradient`. `compare.jl`
measures isolated evaluations; `hmc.jl` measures the authored HMC loop below.

## Native and Reactant HMC

The [measured comparison](HMC.md) includes full timings, preparation costs,
variation, validation, and raw receipts.

`hmc.jl` uses the shared `sampler_transpiler/hmc_benchmark.jl` helper, also used
by the Eight Schools HMC timing driver. It prepares the existing multinomial
HMC `@kernel` through `prepare_transpiled`; the algorithm and model source are
identical for both backends. Native calls prepared Enzyme gradients. Reactant
compiles the whole batch, including gradients, leapfrog steps, momentum draws,
and multinomial proposal selection, and synchronizes once per batch.

After preparing the environment and downloading the bundle as described below,
run each backend in a fresh process on the same CPU:

```sh
BRM_BUNDLE=BUNDLE_DIR HMC_BACKEND=native HMC_OUTPUT=native-hmc.toml \
  JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 6 \
  julia --startup-file=no --project=benchmark/brm_hsgp benchmark/brm_hsgp/hmc.jl
BRM_BUNDLE=BUNDLE_DIR HMC_BACKEND=reactant HMC_OUTPUT=reactant-hmc.toml \
  JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 6 \
  julia --startup-file=no --project=benchmark/brm_hsgp benchmark/brm_hsgp/hmc.jl
```

Defaults are matched batches of 4, 100, and 1,000 transitions, 16 leapfrog
steps per transition, step size 0.03, and nine timed rounds. `HMC_BATCHES`,
`HMC_STEPS`, `HMC_STEPSIZE`, `HMC_ROUNDS`, and `HMC_FRAMES` override these.
Both noncentered and selected-partial frames start at column 5,000 of their
supplied posterior bundle. Each frame uses the same fixed diagonal mass on both
backends: inverse coordinate variance over its 10,000 supplied draws. This is
posterior-informed benchmark setup, not timed adaptation.

Data-only design work is partially evaluated once. The density retains `(q,c)`
inputs and prepares AD with only `q` active; the sampler holds `c` fixed in its
callback context for the entire run. Changing this context requires preparing
a new sampler. Thus HMC measures fixed-frame sampling, whereas the isolated
evaluation comparison exposes centeredness at every executable call.

Receipts separate model/AD preparation, first native gradient, HMC preparation
(including Reactant compilation), first batch execution, and raw warmed batch
times. Compilation garbage is collected once, then additional untimed warmup
runs for at least one second and three batches; timed rounds include normal
GC costs. RNG construction and
finite-density checks are outside timing. The
prepared interface's state copies and output snapshots remain inside timing;
Julia allocation counts exclude C++/device runtime allocations. Timed rounds
start from the same initial position with distinct seeds. Four continued
batches additionally check finite, moving execution. Different RNG engines are
used, with no cross-backend trajectory equality requirement.

These runs retain only final positions and do not measure adaptation, history
storage, ESS, or effective samples per second. Equal transition/leapfrog work
supports a throughput comparison; it does not establish equal mixing.

## Exact target and coordinates

The data are all 133 rows of `MASS::mcycle`, time scaled to `[-1,1]` and
acceleration divided by its sample standard deviation. Each GP has 20 basis
functions on `(-1.5,1.5)`. There are no population intercepts. Both length scales
and both marginal SDs have `LogNormal(0,4)` priors, with zero lower bounds.
The likelihood is normalized Gaussian.

The RK coordinate order is

```
log(rho_mu), log(sd_mu), v_mu[1:20],
log(rho_logsigma), log(sd_logsigma), v_logsigma[1:20]
```

For each GP, `log_s[j] = log(sd) + log(rho)/2 + log(2pi)/4 - rho^2*(j*pi/3)^2/4`.
The standardized coefficient is `z[j] = v[j]*exp(-c[j]*log_s[j])`, and the
basis weight is `v[j]*exp((1-c[j])*log_s[j])`. The coordinate Jacobian is
`-sum(c .* log_s)`. All four positive-hyperparameter Jacobians are included.
Centeredness is inactive during differentiation with respect to `q`.

`compare.jl` loads the literal model definitions between the markers in BRM's
`research/adaptive_centering/reproduce.jl`, then independently builds `SBBRMI`
and `TuringBRMI`. It maps Stan coordinates using `BridgeStan.param_unc_names`
and native coordinates using DynamicPPL's range metadata. Partial coefficients
are named `beta_partial`; NCP coefficients are named `beta_raw`. Exact maps are
written to the receipt. `propto=false, jacobian=true` is explicit for Stan.

The native Turing value control is the generated `DynamicPPL.LogDensityFunction`.
Its gradient control is BRM's existing `adaptive_centering_problem` wrapper,
constructed from the generated NCP target. The source centeredness is set
through the wrapper's indexed reparametrization and its scoring-plan
synchronization callback; its target remains NCP. Each basis control is matched
to the DynamicPPL coordinate index. The separately generated fixed-partial
target is an independent value control, with its own coordinate permutation.
The wrapper combines the generated density with the supported analytic HSGP
gradient and Enzyme coordinate transport. Direct Enzyme differentiation of this DynamicPPL
model is a documented BRM limitation. Neither ForwardDiff nor ReverseDiff is
used for differentiation here.

## Reproduce

Use Julia 1.10 and the benchmark project. Reactant is pinned to 0.2.284.
BRM must contain `5b8c9c6b1c2c7dbd0e222194d41387f0e509381f`; the initial audited
snapshot was descendant `8dfe41253af3043482cb3270cf513b50a1de5437` (ancestry
counts `0 12`). The receipt records all measured package versions and SHAs.

BRM has unregistered dependencies. Prepare this consumer project using the
ecosystem's canonical resolver, supplying exact `Name=path=40hexsha` overlays
for the BRM and MutatingFunctions snapshots and `ReactiveKernels=<this repo>`.
For example, from this repository in a provisioned KB environment:

```sh
RESOLVE_ACCEPTANCE=1 bash -c 'set -euo pipefail
source /home/n/github/nsiccha/Claude/lib-repos.sh
source /home/n/github/nsiccha/Claude/lib-resolve.sh
julia --startup-file=no --project="$1" -e "$(resolve_script "$1" ReactiveKernels "$2" "ReactiveKernels=$2" "${@:3}")"
' _ "$PWD/benchmark/brm_hsgp" "$PWD" \
  "BayesianRegressionModels=<snapshot path>=<full SHA>" \
  "MutatingFunctions=<snapshot path>=<full SHA>"
```

Download and extract the supplied posterior bundle into a scratch directory.
Its KB source is `/code?path=/home/niko/.local/state/kb-agents/uploads/eaaf3c4129b24f53.targz&raw=1`.
The bundle contains 10,000 NCP and 10,000 selected-partial posterior positions,
each with all 44 coordinates. Expected SHA-256 hashes:

| File | SHA-256 |
|---|---|
| Bundle | `4512eb3425c7fd28cc79bd5b9b8626eec4871a7403a72c22c2cb359b00c36e37` |
| `noncentered.jls` | `9f0b2513dd9762d54360c26548322e6349fa4e834691d0a92ae210d951435c50` |
| `partial.jls` | `ad667cb50ddf28faff727cc73bdf22556ddaff54d5384dd4337556a22fca9247` |
| `centeredness.tsv` | `e882c5b7a906275bbff370291c686ba8284272fdf7aa5102bccba3323468e289` |
| `examples/data/mcycle.csv` | `b89a1e4eb0391a982b32be3e378df00e8593ff9971e9425e9c5d7929b74f9801` |

```sh
julia --startup-file=no --project=benchmark/brm_hsgp test/test_brm_hsgp_reactant.jl
julia --startup-file=no --project=benchmark/brm_hsgp \
  benchmark/brm_hsgp/compare.jl BUNDLE_DIR OUTPUT_DIR
```

A final optional positional argument limits posterior points for smoke runs;
these are never full acceptance receipts. `RK_HSGP_NATIVE=0` explicitly runs
only the Stan reference and writes `native_verified=false`; it cannot silently
convert a native failure into a passing comparison.

## Measurement boundaries

Preparation, first native gradient, Reactant primal compilation, Reactant
gradient compilation, and first compiled execution have separate seconds fields.
Warmed measurements use BenchmarkTools with `evals=1` and identical positions
for all evaluators, reporting median/minimum nanoseconds and Julia allocations.
BLAS has one thread. `reactant_resident` uses resident inputs and synchronous
execution; `reactant_host` includes transferring `q` and materializing the value
and gradient on the host. Centeredness stays resident between evaluations.
Backend-managed memory is outside Julia's allocation counter.

The `≤1.25× StanBlocks` target must be judged with the measurement boundary
stated. Compilation cost is never mixed into the warmed ratio. A native Turing
target whose support differs from Stan is an unsupported comparison, not an
opportunity to modify the model or discard posterior points.

## Current verified CPU run

[`../receipts/brm-hsgp-reactant-v2.toml`](../receipts/brm-hsgp-reactant-v2.toml)
records RK `0dfe2165f978b1fdb1332a4d089cb31dd8851d25` with the corrected BRM
`d137c326fa6a30cf173bf81fd6767e440bf025c0` and WarmupHMC
`ad1feb6a228b8e9437fa745ae19ad490a3af34b1`. All other measured versions and
source hashes are in the receipt.

```sh
KB_COMPACT_KEEP_LOG=1 kb-run-compact taskset -c 6 \
  julia --startup-file=no --project=benchmark/brm_hsgp \
  benchmark/brm_hsgp/compare.jl "$TMPDIR/brm-benchmark" "$TMPDIR/brm-comparison-native-v2"
```

Exit 0, elapsed 365 s, strato2 retained log `$TMPDIR/kb-run-compact.uNK6Mt`.
All 40,072 positions passed against both native Turing and Stan, including
the separately generated fixed-frame DynamicPPL values. No columns were dropped.
Maximum normalized value error is `9.094947017729282e-13` against either
reference. Maximum scaled gradient error is `1.2635231778851861e-11` against
native Turing and `1.5455363706851556e-11` against Stan.

Measurements now allow 100,000 samples and two seconds per evaluator,
`evals=1`. Native Enzyme is 1.02–1.20× StanBlocks, meeting the 1.25× target in
all four frames; resident Reactant is 1.50–1.98× and the host boundary is
3.42–4.43×. The initial 1,000-sample run overstated the native Enzyme gap.
This shared-host run is still a measurement, not a fixed cost guarantee.
The [documentation page](../../docs/src/brm-hsgp.md) renders the current table
directly from v2. [DIAGNOSTICS.md](DIAGNOSTICS.md) reports separate controls
for constant centeredness, allocations, synchronous calls, and dense projections.

The full run also corrected two consumer mistakes exposed after the BRM fix:
`turing_model_source` returns an expression, which must be rendered before
hashing; and the adaptive native wrapper starts from NCP, with partial
centering on its source side. Feeding it a generated `beta_partial` target
is outside that wrapper's supported contract.

## Initial CPU run (superseded)

[`../receipts/brm-hsgp-reactant-v1.toml`](../receipts/brm-hsgp-reactant-v1.toml)
is the unmodified runner output from RK
`214cec420a7c4c1d0a354f7febea7d6ae83f34cb`, on `strato2`, 2026-09-14:

```sh
RK_HSGP_NATIVE=0 KB_COMPACT_KEEP_LOG=1 kb-run-compact taskset -c 6 \
  julia --startup-file=no --project=benchmark/brm_hsgp \
  benchmark/brm_hsgp/compare.jl "$TMPDIR/brm-benchmark" "$TMPDIR/brm-comparison-full"
```

Exit 0, elapsed 353 s, retained log `$TMPDIR/kb-run-compact.Gw0SWX`.
All four frames in this initial receipt have 1,000 warmed samples per
path, `evals=1`, at the midpoint posterior column; medians are not averages over
10,000 different positions. Affinity is one CPU and BLAS one thread, on a shared
host. Minimum timings and allocations are retained alongside medians. The
resident value-and-gradient median is 1.59–2.97× StanBlocks, and the host result
boundary is 3.56–5.27×: neither meets the 1.25× target. This is an initial CPU
measurement, not a GPU result or an optimized performance ceiling.

Numerical checks cover all 20,000 original posterior draws, 20,000 additional
transported positions, and 72 adversarial positions. Maximum density error is
`9.094947017729282e-13`; maximum gradient error scaled componentwise by
`1+abs(Stan gradient)` is `1.5455363706851556e-11`. The full receipt keeps the
absolute errors as well: mixed/centered adversarial gradients can be around
`1e74`, with absolute error around `1e61` but scaled error below `1.8e-13`.
No columns were dropped. Both the separately compiled primal and compiled
value-plus-gradient are checked at every point.

Native Turing was explicitly unsupported in this initial receipt. At BRM
`8dfe41253af3043482cb3270cf513b50a1de5437`, initialization with `rho=0.2` fails
in `Bijectors.VectorBijectors.Untruncate` inside `_BRMConstrainedKernel` and
`_brm_turing_hsgp_term` with `DomainError(-0.005181297595301976)`. The generated
native target retains a length-scale floor near 0.2051813, while the independently
emitted Stan declares both length scales with `lower=0.0`. This is tracked as
BRM snag `turing-hsgp-expl-e60fccfe`. The failed default run exited 1 after
135 s (`$TMPDIR/kb-run-compact.rrsktz`). The literal model-definition hash is
identical at the bundle source, the requested source, and the measured BRM
snapshot. The v2 receipt above closes this verification boundary using the
corrected source and supported adaptive-wrapper construction.
