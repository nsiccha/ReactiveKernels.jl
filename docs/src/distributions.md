# Native log densities as recipes

`ReactiveKernels` is not a distribution library. A log density is ordinary
arithmetic in a `@kernel`; `prepare` selects the requested have→want path and
compiles it. `Distributions.jl` is used only by the tests and benchmarks as an
independent oracle—the generated kernels below do not call it.

Write formulas from values already in hand. If the model parameter is `logσ`,
use it directly in the normalizer and derive `σ = exp(logσ)` only for the
standardized residual. RK performs graph planning and structural CSE, not
algebraic rewriting such as cancelling `log(exp(logσ))`.

Every panel is build-executed. **Raw input** is the complete authored path,
**Generated kernel** is its actual `code_expr`, and **Compute DAG** is the plan
that produced it.

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

`plate` marks observation ports as batched and generates the reduction. Recipes
depending only on shared parameters are hoisted: here `exp(logσ)` runs once,
while the residual and density run per observation. The default returns the
sum without an intermediate vector; `reduce = nothing` returns per-observation
values for LOO/WAIC. Multiple ports may be batched together.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.VECTORIZED_SOURCE,
)
```

The tests separately verify oracle parity, concrete `Float64` returns, invariant
hoisting, exact zero-allocation native reduction, and output-only allocation in
collect mode.

## More families shared with ProbabilityMeasures

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

These rows are derived from the four `PreparedKernel`s built by that exact
source. Matrix boundaries include a factorization recipe; pre-factorized
boundaries cut the graph after it.

```@eval
Main.ReactiveKernelsDocs.render_mvn_parametrization_plans(
    Main.DistributionExamples.MVNORMAL_SOURCE,
)
```

Native tests check all four results against the same `Distributions.MvNormal`
oracle and require concrete `Float64` inference. Reactant compiles all four
scalar kernels and all four whole-vector replica maps from the same source.

The stationary AR(1) kernel similarly treats the complete sequence as one
value. Its lagged residuals give an O(T) log density, including the stationary
initial-state term and an explicit `-Inf` result outside `abs(ϕ) < 1`.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.AR1_SOURCE,
)
```

Both examples then apply `replica(...; batched = :x)`: a matrix represents
independent vectors or independent series by columns. The inner coordinate/time
axis remains coupled, while the added trailing axis maps the whole authored
kernel. The native and Reactant tests compile both the scalar and replicated
forms from these exact source strings.

### Structured native and Reactant benchmark

The matched comparison below uses the covariance-Cholesky HAVE boundary and
public multivariate-Normal log-density APIs. The four-boundary table above is a
planner acceptance result; this timing table does not pretend that the three
libraries expose equivalent construction-time parametrization APIs.
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
constructor rejects traced `μ` and `σ` values. This is a compatibility result,
not an omitted measurement. First-shape compilation is recorded in the receipt
for diagnostics but is not compared: RK's first sample includes Reactant service
startup because it ran first.

The machine-readable inputs and all five raw samples are checked in as the
[benchmark receipt](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/distribution-logdensity-v1.toml). The
docs build rejects a dirty/unpinned receipt or failed RK/ProbabilityMeasures
Reactant acceptance, and the test suite independently re-derives every median.
