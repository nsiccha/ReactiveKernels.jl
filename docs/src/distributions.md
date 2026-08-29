# Native log densities as recipes

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

`ReactiveKernels` is not a distribution library. A log density is ordinary
arithmetic in a `@kernel`; `prepare` selects the requested have→want path and
compiles it. `Distributions.jl` is used only by the tests and benchmarks as an
independent oracle—the generated kernels below do not call it.

Write formulas from values already in hand. If the model parameter is `logσ`,
use it directly in the normalizer and derive `σ = exp(logσ)` only for the
standardized residual. RK plans the graph and reuses a repeated subexpression,
but it does no algebra — it will not cancel `log(exp(logσ))` for you.

Each panel below shows the **Raw input** (the source), the **Generated kernel**
(`code_expr`), and its **Compute DAG**.

## Continuous: Normal location and log scale

The Gaussian uses three transparent steps: scale, standardized residual, and
log density. The test suite checks the result against `Distributions.logpdf`.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.CONTINUOUS_SOURCE,
)
```

## Discrete: Bernoulli observation with a logit

The Bernoulli-logit recipe selects the stable `-log1pexp(∓logit)` expression for
the observed bit.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.DISCRETE_SOURCE,
)
```

## Batched: the same recipe with `plate`

`plate` marks observation ports as batched and adds the sum over observations.
Steps that depend only on shared parameters are lifted out of the loop: here
`exp(logσ)` runs once, while the residual and density run per observation. The
default returns the sum without building an intermediate vector; `reduce =
nothing` returns the per-observation values for LOO/WAIC. Several ports can be
batched together.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.VECTORIZED_SOURCE,
)
```

## More scalar families shared with ProbabilityMeasures

These compact recipes cover three qualitatively different shapes that are also
implemented by ProbabilityMeasures.jl: a heavy-tailed Cauchy, the nondifferentiable
peak of a Laplace density, and a positive-support LogNormal with an explicit
`-Inf` support result.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.CAUCHY_SOURCE,
)
```

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.LAPLACE_SOURCE,
)
```

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.LOGNORMAL_SOURCE,
)
```

The same pattern extends without family-specific batching code. Exponential is
authored from a log scale, Geometric from a success-probability logit, and
Uniform from dynamic endpoints. Each block defines one scalar formula and then
uses the ordinary `plate` API for independent observations.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.EXPONENTIAL_SOURCE,
)
```

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.GEOMETRIC_SOURCE,
)
```

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.UNIFORM_SOURCE,
)
```

### Added-family native and Reactant benchmark

The matched comparison uses the public vectorized log-density APIs in
Distributions and ProbabilityMeasures and the generic RK `plate` generated from
each scalar source above. Parameters are traced runtime inputs under Reactant;
compilation and host↔device transfers are excluded from execution timings.

```@eval
Main.ReactiveKernelsDocs.render_scalar_gallery_benchmarks()
```

Each cell is the median of five minimum-time measurements. Unsupported
Distributions + Reactant cells retain the constructor diagnostic instead of
silently disappearing. The checked-in
[gallery benchmark receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/scalar-distribution-gallery-v1.toml)
contains raw samples, allocations, support results, and exact package pins.

## Structured families: multivariate Normal and AR(1)

Non-scalar families use the same authoring idea, but they are not scalar
`plate`s. The coordinates of a multivariate observation—and the time steps of
an autoregressive series—are coupled inside one mathematical kernel.

The multivariate Normal is authored once. Alternative recipes produce the same
`half_logdet_cov` and `quadratic` ports from a covariance matrix, its Cholesky
factor, a precision matrix, or its Cholesky factor. `prepare` starts at whichever
representation is in HAVE and selects only that route to `logdensity`; callers
do not convert everything to one privileged parametrization first.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.MVNORMAL_SOURCE,
)
```

These rows are derived from the four `PreparedKernel`s built by that source. When
you start from a full matrix the plan includes the factorization; when you start
from an already-factored form the plan skips it.

```@eval
Main.ReactiveKernelsDocs.render_mvn_parametrization_plans(
    Main.DistributionExamples.MVNORMAL_SOURCE,
)
```

The stationary AR(1) kernel similarly treats the complete sequence as one
value. Its lagged residuals give an O(T) log density, including the stationary
initial-state term and an explicit `-Inf` result outside `abs(ϕ) < 1`.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.AR1_SOURCE,
)
```

Both examples then apply `replica(...; batched = :x)`: a matrix holds independent
vectors or independent series, one per column. The inner coordinate/time axis
stays coupled, while the new trailing axis runs the whole kernel once per column.

### Structured native and Reactant benchmark

The matched comparison below uses the covariance-Cholesky HAVE boundary and
public multivariate-Normal log-density APIs. It compares evaluation cost only,
not construction-time parametrization APIs.
For AR(1), Distributions and ProbabilityMeasures receive the mathematically
equivalent dense MVN with its Cholesky factor computed before timing; RK runs
the authored O(T) recurrence. Distribution construction, factorization,
compilation, and host↔device transfers are excluded from execution timings.

```@eval
Main.ReactiveKernelsDocs.render_structured_distribution_benchmarks()
```

Each cell is the median of five minimum-time measurements. The
[structured benchmark receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/structured-distribution-logdensity-v1.toml)
contains all raw samples, allocation observations, compiler diagnostics, and
exact package/commit pins; both tests and the docs build validate it.

The unsupported Reactant cells are measured compatibility results. At the
pinned versions, ProbabilityMeasures' full MVN uses scalar indexing of the
traced vector, while Distributions has no full-MVN `logpdf` method for a traced
array. The receipt retains both diagnostics.

## Scalar plate native and Reactant benchmark

The comparison below evaluates the same batched Normal log density with shared
location and scale parameters. The native columns use each library's idiomatic
vectorized public interface; the Reactant columns compile the corresponding
whole calculation with traced observations and traced shared parameters.

```@eval
Main.ReactiveKernelsDocs.render_distribution_benchmarks()
```

Each cell is the median of five minimum-time measurements, with Reactant work
synchronized before timing. Compilation and host↔device transfers are excluded
from execution times. The Reactant allocation cells report only the Julia host
wrapper observed by `@allocated`, not device memory. Native RK's reduction is
exactly zero-allocation; the other native public interfaces allocate their
vector of pointwise densities before summing it.

Distributions + Reactant is marked unsupported because its exact public
constructor rejects traced `μ` and `σ` values. First-shape compilation is
recorded in the receipt for diagnostics but is not compared: RK's first sample
includes Reactant service startup because it ran first.

The machine-readable inputs and all five raw samples are checked in as the
[benchmark receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/distribution-logdensity-v1.toml). The
docs build rejects a dirty/unpinned receipt or failed RK/ProbabilityMeasures
Reactant acceptance, and the test suite independently re-derives every median.
