# Distribution log-density benchmark environment

This environment pins the packages used by
`benchmark/distributions_comparison.jl`, including ProbabilityMeasures at the
exact reviewed commit. From the repository root:

```sh
julia --startup-file=no --project=benchmark/distributions benchmark/distributions/setup.jl
julia --startup-file=no --project=benchmark/distributions \
  benchmark/distributions_comparison.jl \
  --output=benchmark/receipts/distribution-logdensity-v1.toml
julia --startup-file=no benchmark/receipts/validate_distributions.jl \
  benchmark/receipts/distribution-logdensity-v1.toml

julia --startup-file=no --project=benchmark/distributions \
  benchmark/scalar_distribution_gallery_comparison.jl \
  --output=benchmark/receipts/scalar-distribution-gallery-v1.toml
julia --startup-file=no benchmark/receipts/validate_scalar_gallery_distributions.jl \
  benchmark/receipts/scalar-distribution-gallery-v1.toml

julia --startup-file=no --project=benchmark/distributions \
  benchmark/structured_distributions_comparison.jl \
  --output=benchmark/receipts/structured-distribution-logdensity-v1.toml
julia --startup-file=no benchmark/receipts/validate_structured_distributions.jl \
  benchmark/receipts/structured-distribution-logdensity-v1.toml
```

The benchmark refuses a dirty ReactiveKernels tree by default. Reactant
compilation and host-to-device transfers are outside the timed execution
region; every compiled call uses `sync=true`, and both model parameters remain
tracked runtime values. Per-shape compilation diagnostics are retained in the
receipt; the first RK compile includes Reactant service startup and is not used
as a cross-library compile-time comparison.

The scalar Normal receipt also records a distinct invocation-amortization
protocol: the scalar RK kernel is lifted with `replica(...; batched = :x)` and
one compiled call evaluates 1, 16, or 256 independent observations. Whole-call
timing and host allocations are retained alongside time normalized per logical
evaluation. Override the exploratory counts with `RK_DENSITY_REPLICAS=1,4`;
published receipts use `1,16,256`.

The scalar-gallery benchmark compares the exact Exponential, Geometric, and
Uniform scalar sources after their generic `plate` lift. It records two vector
sizes per family, traces every model parameter under Reactant, and keeps an
explicit diagnostic for any public comparison path that does not compile.

The structured benchmark compares the exact build-executed MVN and stationary
AR(1) sources against the public multivariate-Normal interfaces in Distributions
and ProbabilityMeasures. The matched MVN timing uses the covariance-Cholesky
HAVE boundary; the same authored RK graph is separately acceptance-tested from
covariance, covariance Cholesky, precision, and precision Cholesky boundaries,
natively and under Reactant. The AR(1) baselines use its equivalent dense
multivariate Normal with a factor computed before timing; RK evaluates the
authored O(T) recurrence. Unsupported full-MVN Reactant paths are retained
with their compiler diagnostics rather than silently omitted.
