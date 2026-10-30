# Distribution log-density benchmark environment

This environment pins the packages used by
`benchmark/distributions_comparison.jl`, including ProbabilityMeasures at the
exact reviewed commit. From the repository root:

```sh
julia --startup-file=no --project=benchmark/distributions benchmark/distributions/setup.jl
julia --startup-file=no --project=benchmark/distributions \
  benchmark/distributions_comparison.jl --output=distribution-logdensity-v2.toml
julia --startup-file=no --project=benchmark/distributions \
  benchmark/scalar_distribution_gallery_comparison.jl \
  --output=scalar-distribution-gallery-v2.toml
julia --startup-file=no --project=benchmark/distributions \
  benchmark/structured_distributions_comparison.jl \
  --output=structured-distribution-logdensity-v2.toml

julia --startup-file=no benchmark/enzyme_variants.jl \
  --project=benchmark/distributions benchmark/distribution_gradients.jl \
  --output=benchmark/receipts/distribution-gradient-v1.toml
julia --startup-file=no benchmark/receipts/validate_distribution_gradients.jl \
  benchmark/receipts/distribution-gradient-v1.toml
```

The gradient benchmark runs once with the newest registered Enzyme and once
with the host's local Enzyme `dogfood` source; the project's `=0.13.199` pin
remains the historical receipt's version. The requested path receives the
registered receipt; `distribution-gradient-v1.enzyme-dogfood.toml` and
`distribution-gradient-v1.enzyme-variants.toml` are written beside it. Each
variant's environment is derived from this project, together with
ReactiveKernels' exact source dependencies. See `../ENZYME_VARIANTS.md`.

The three log-density scripts write v2 receipts with native cells only. The
checked-in v1 receipts (`distribution-logdensity-v1.toml`,
`scalar-distribution-gallery-v1.toml`,
`structured-distribution-logdensity-v1.toml`) also carry Reactant cells,
measured with ReactiveKernels' removed Reactant path; their validators in
`benchmark/receipts/` check those historical receipts.

The benchmark refuses a dirty ReactiveKernels tree by default. The Normal run
measures the authored return-only likelihood against the established
`plate(normal.logpdf; ...)` path, and the harness rejects an authored pointwise
output allocation before timing.

The scalar-gallery benchmark compares the public Cauchy, Laplace, Bernoulli,
LogNormal, Exponential, Geometric, Uniform, Poisson, Gamma, Beta, and Binomial
objects after their generic `plate` lift. Normal has its separate
shared-object/control benchmark. The gallery records two vector sizes per
family.

The structured benchmark compares the exact build-executed MVN source against
the public multivariate-Normal interfaces in Distributions and
ProbabilityMeasures. The matched timing uses the covariance-Cholesky HAVE
boundary. AR(1) is intentionally absent because neither comparison package
exposes a native AR(1) distribution.

The distribution-gradient run preserves its original seven-family corpus
(Cauchy through Uniform) and reuses the shared inputs and size inventory for
those families. The four newer families have analytic Enzyme coverage in the
package tests and native coverage in the scalar gallery; they are not
retroactively added to this historical zero-allocation receipt. The gradient
run measures prepared Enzyme reverse mode twice for vector active ports:
`ad_gradient` owns its returned vector, while `ad_value_and_gradient!` writes
into caller-owned storage. Bernoulli and Geometric differentiate their scalar
logit ports because integer observations cannot be active; their isbits scalar
gradients need no mutable destination. Preparation is excluded from every timed
region, analytic gradients gate every row, and the receipt retains timing plus
allocation bytes/counts for all five raw rounds.
