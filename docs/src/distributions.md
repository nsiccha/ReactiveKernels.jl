# Compositional Distributions.jl log densities

These executable examples sketch the boundary a future probabilistic-programming
layer can build on without adding probability semantics to `ReactiveKernels`
itself. `Distributions.jl` supplies distributions and `logpdf`; public
`@kernel` fragments expose named, typed ports; `compose` joins fragments; and a
`prepare` query selects the requested log-density computation.

`Distributions.jl` and `ForwardDiff.jl` are example, test, and documentation
dependencies only. They are not dependencies of the package runtime.

Each panel is produced from one build-executed artifact. **Raw input** retains
the literal compact source evaluated during this documentation build, alongside
the established origin, executed-input, and actual-output annotations.
**Generated kernel** is `code_expr` from that execution, and **Compute DAG** is
the live `visualize(plan)` component. **Compare all** moves those same three
views into a side-by-side dialog.

## Continuous: Normal location and log scale

The distribution-family fragment turns an unconstrained log scale into a
`Normal`; the observation fragment consumes that distribution. The example
compares both the scalar value and a real `ForwardDiff.gradient` over
`(x, μ, logσ)` directly with `Distributions.logpdf`.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.CONTINUOUS_SOURCE,
)
```

## Discrete: Bernoulli observation with a differentiable logit

The observation is deliberately discrete, so differentiation is only with
respect to its continuous logit parameter. The kernel value and that
one-dimensional `ForwardDiff` gradient are checked directly against a
`Bernoulli` reference.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.DISCRETE_SOURCE,
)
```

## Multivariate: shared regression coefficients

One coefficient vector feeds both an `MvNormal` prior and an `MvNormal`
likelihood. A third fragment sums the two selected terms. This makes the shared
parameter visible to the planner and demonstrates a multivariate composition
without proposing a model DSL or sampler API. The joint value and gradient with
respect to the shared coefficients are checked against direct `MvNormal`
calls.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.MULTIVARIATE_SOURCE,
)
```

The sources record `BenchmarkTools.@ballocated` measurements behind a function
barrier for both each prepared kernel and its equivalent direct
`Distributions.jl` call, then report their difference as prepared-composition
overhead. The scalar paths measure zero bytes on both sides. The multivariate
path reports equal, nonzero measurements because `MvNormal` construction and
linear algebra allocate in both the prepared and direct calls. The scalar
`Base.return_types` results are checked as exactly `Float64`, so allocation and
inference are separate claims rather than an `Any` result being mistaken for a
distribution cost. The package's separate non-allocating path remains
documented under [Non-allocating kernels](nonallocating.md).
