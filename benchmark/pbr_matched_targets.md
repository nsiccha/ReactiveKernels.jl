# Matched PBR targets and shared scalar derivatives

The original PB/PBR investigation and RK nonlinear example did not measure
the same model. These experiments compare the same unconstrained targets,
then isolate a reusable scalar mechanism without a handwritten model gradient.

`pbr_matched_targets.jl` uses two design columns, no intercept, Normal(0,2)
coefficient priors, and either a Gaussian likelihood with LogNormal(0,1) scale
or a Bernoulli-logit likelihood. The Gaussian coordinate is log(scale), including
its transform Jacobian. Both targets omit parameter-independent constants as
PBR does. Numerical references are separate from every timed implementation.

Both packages prepare storage for the measured observation count and receive
caller-owned gradient outputs. RK also checks changed lengths on the same
prepared operation. Concrete function barriers keep reassigned outer-loop
variables out of the timing closures. Ordinary Enzyme reverse differentiates
the authored RK/native primal in every AD comparison; the local owned rules
derive their adapters from their pure value/partial graph.

The corrected first comparison passed 296 checks. At 2,048 observations,
Gaussian PBR took 1.84 microseconds and RK's nonallocating generated plate
took 1.91, both allocating zero bytes. At 8,192 they took 7.56 and 7.86.
These are workload-specific results on a shared host, not general parity.

## Bernoulli attribution

`pbr_softplus_attribution.jl` compares the original `log1pexp`, a stable
primal using `exp(-abs(x))`, and an owned scalar graph sharing that exponential
between value and derivative. It passed 74 checks. Each entry below is the
median of five round medians, with randomized order within each round.

| Implementation | 2,048 rows, microseconds | 8,192 rows, microseconds |
| --- | ---: | ---: |
| PBR in place | 17.280 | 70.000 |
| RK original | 31.960 | 124.115 |
| RK stable primal, ordinary AD | 30.130 | 150.699 |
| RK joint value/partial graph | 22.460 | 89.509 |
| Native Julia original, ordinary AD | 33.930 | 146.560 |
| Native stable primal, ordinary AD | 32.940 | 140.884 |
| Native joint graph | 24.030 | 110.355 |

RK and PBR allocate zero bytes here. The native controls allocate two arrays
(32,912 bytes at 2,048 rows; 131,216 at 8,192), six allocations in total.
There was substantial timing noise at 8,192: the stable-primal RK rounds range
from 117.790 to 199.819 microseconds. The joint graph's rounds range from
89.080 to 136.095. The repeatable conclusion is the value/partial sharing
gain; the simpler primal by itself did not establish a useful improvement.

Optional LLVM dumps in this experiment expose the wrapper and an indirect
AD call, not the differentiated loops. They do not establish scalar operation
counts or explain the entire remaining PBR gap.

## Wider local likelihood graph

`pbr_bernoulli_cell.jl` extends the joint graph to a complete Bernoulli cell,
including the observation partial. It passed 300 numerical and ownership
checks, including 512-bit references for extreme tails and ordinary reverse
with both arguments active. It avoids cancellation in an observed success's
positive-logit tail, but does not establish an overall performance gain.

| Implementation | 2,048 rows, microseconds | 8,192 rows, microseconds |
| --- | ---: | ---: |
| PBR | 16.520 | 66.450 |
| RK joint softplus | 22.190 | 88.629 |
| RK joint complete cell | 21.460 | 101.139 |
| RK complete cell, generated plate | 31.400 | 126.305 |
| Native complete cell | 24.430 | 109.505 |

The complete-cell variant remains a benchmark control. The smaller softplus
graph is the integration candidate. Production acceptance uses the existing
Bernoulli sign-selected logit formula and retains its independent lazy
probability-input route. `pbr_ppl_bernoulli.jl` measures the actual distribution
and PPL sampler paths, accounting for PPL's Normal prior constants.

## Reproduction

Julia 1.12.7, PBR `6c6d9300160be5432becd801ec602ad06d0e87f8`, Enzyme
`7832d492b16bad8862de8fb0d0dcf7200af3a30d`, libEnzyme
`e316a076ef438970e339876079acd02ce48d3d21` with LLVM18;
MutatingFunctions `02eb981b9770e767f47a7fda287fcbfb9531d587`,
OutputSignatures `f971381f760e91986b3a728a719341e7e7761f55`.
Runs used one standard compute token on gordito and BLAS with one thread.
Timing samples use BenchmarkTools with `evals=1`, up to 1,000 samples and
0.3 seconds per round. Package loading, preparation and first compilation
are outside the warm timings; this comparison makes no cold-latency claim.

The environment needs RK, its DistributionKernels/PPL packages, PBR,
MutatingFunctions, Enzyme, DifferentiationInterface and BenchmarkTools.
Run each `.jl` file directly in that resolved environment. PBR is only a
benchmark dependency. No production dependency on PBR was introduced.
