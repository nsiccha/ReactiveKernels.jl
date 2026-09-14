# Exact BRM motorcycle HSGP benchmark

The model is authored once in [`examples/brm_hsgp.jl`](../../examples/brm_hsgp.jl).
It uses ordinary `@kernel` recipes and an observation `plate`. `prepare(...;
bound=...)` folds the data-only basis and squared frequencies; the 44-vector
`q` and 40-vector of centeredness controls remain live. The benchmark separately
prepares native Enzyme AD, compiles the primal through Reactant, and compiles
value plus gradient through `compile_ad_value_and_gradient`. It does not sample.

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
which combines that density with the supported analytic HSGP gradient and
Enzyme coordinate transport. Direct Enzyme differentiation of this DynamicPPL
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
