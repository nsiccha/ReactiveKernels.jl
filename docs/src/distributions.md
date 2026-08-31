# Distribution kernels

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

Distribution kernels are ordinary transparent `@kernel` graphs. A standard
family owns operations such as `logpdf`, `cdf`, and `quantile`; one shared
location-scale kernel supplies translation, scaling, standardization, and the
inverse standardization route. `Distributions.jl` appears only in tests and
benchmarks as an independent oracle—the generated kernels below do not call it.

Each panel below shows the **Raw input** (the source), a readable **Generated
kernel** derived from the executed kernel and selected plan, and its **Compute
DAG**. The exact compiled AST remains available through `code_expr`.

## One location-scale graph, several families

The complete authoring surface is below. `standard_normal`, `standard_cauchy`,
and `standard_laplace` contain only their standard-family mathematics.
`location_scale` supplies the shared transformation once. The final three lines
bind concrete, immutable kernel objects—`normal` is graph data with named ports,
not a closure hidden behind a global function.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.CONTINUOUS_SOURCE,
)
```

The simple path stays simple: `prepare(normal.logpdf)` asks for the distinguished
result of that endpoint. Inside another `@kernel`, `normal.logpdf(x)` splices the
same transparent graph into the caller; it does not insert an opaque runtime
call.

Every named boundary remains selectable. For example, one plan can request both
the density and CDF:

```julia
joint = extract(normal;
    have = (:x, :location, :scale),
    want = (:logpdf, :cdf))
```

Both endpoints reuse the same `standardized(x)` node. `quantile(p)` follows the
explicit inverse edge `inv(standardized, z)`. The graph also contains both
`log_scale = log(scale)` and `scale = exp(log_scale)`, so callers may start from
either representation. When a caller supplies both, RK respects both HAVE
values: it neither recomputes nor validates either one.

Included child ports retain scoped names rather than colliding with the outer
result. For example, `normal[Symbol("standard.logpdf")]` addresses the standard
family term, while `normal.logpdf` addresses the adjusted location-scale
result; either can be passed to `extract`.

## Batched: the same endpoint with `plate`

`plate` lifts `normal.logpdf` over observations and sums the results. Work that
depends only on the shared location and scale stays outside the observation
loop. `reduce = nothing` selects the same per-observation values for LOO, WAIC,
or PSIS.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.VECTORIZED_SOURCE,
)
```

## Other standard location-scale families

Cauchy and Laplace use exactly the same `location_scale` graph. Their examples
only select the family object and prepare its `logpdf` endpoint; neither repeats
the transform or normalization plumbing.

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

## Discrete: Bernoulli observation with a logit

The Bernoulli-logit kernel selects the stable `-log1pexp(∓logit)` expression for
the observed bit.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.DISCRETE_SOURCE,
)
```

## Positive-support and bounded families

LogNormal, Exponential, Geometric, and Uniform illustrate shapes that are not a
location-scale lift of one of the standard objects above. Each remains a compact
mathematical kernel with an explicit support result.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.LOGNORMAL_SOURCE,
)
```

Exponential is authored from a log scale, Geometric from a
success-probability logit, and Uniform from dynamic endpoints. The ordinary
`plate` API handles independent observations without family-specific batching
code.

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

The multivariate Normal is authored once. Alternative graph paths produce the same
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
location and scale parameters. It includes both the shared `normal.logpdf`
object and a one-off RK formula as a matched control, so reuse overhead is
measured directly. The other native columns use each library's idiomatic
vectorized public interface; the Reactant columns compile the corresponding
whole calculation with traced observations and traced shared parameters.

```@eval
Main.ReactiveKernelsDocs.render_distribution_benchmarks()
```

### Amortizing the Reactant call boundary

For a tiny scalar density, most of the compiled-call time is fixed host/runtime
overhead. When observations are independent, the same scalar `PreparedKernel`
can be lifted with `replica(...; batched = :x)` and evaluated once over a vector.
The checked-in receipt measures 1, 16, and 256 independent one-observation
evaluations per compiled call:

```@eval
Main.ReactiveKernelsDocs.render_distribution_amortization()
```

The allocation count and bytes are for the whole host-side invocation, not for
each observation. Batching therefore avoids paying that wrapper once per logical
evaluation; it does not change the latency of an isolated one-observation call.

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
