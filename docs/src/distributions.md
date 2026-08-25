# Native log densities as recipes

These executable examples build log densities the way a probabilistic-programming
layer on top of `ReactiveKernels` would want them: as ordinary `@kernel` recipes
of closed-form arithmetic. The **compute path contains no `Distributions.jl`
call** — each density is written out directly, `compose` joins the fragments,
and a `prepare` query lowers exactly the requested computation to a straight-line
kernel. That keeps the hot path free of distribution-object construction and
friendly to batching, lazy evaluation, and the non-allocating lowering under
[Non-allocating kernels](nonallocating.md).

`Distributions.jl` appears **only as an independent oracle**: it supplies a
reference value (and a reference allocation figure) that the native kernel is
checked against. It is an example/test/documentation dependency, never a
dependency of the package runtime and never part of the compute or gradient
path. Differentiation goes through `DifferentiationInterface` with the `Enzyme`
reverse-mode backend, and `Enzyme` differentiates the native recipes only —
never `Distributions.jl`.

Each panel is produced from one build-executed artifact. **Raw input** retains
the literal compact source evaluated during this documentation build, alongside
the established origin, executed-input, and actual-output annotations.
**Generated kernel** is `code_expr` from that execution — inspect it to confirm
the lowered kernel names no distribution library — and **Compute DAG** is the
live `visualize(plan)` component. **Compare all** moves those same three views
into a side-by-side dialog.

## Continuous: Normal location and log scale

The family fragment maps an unconstrained log scale to a positive scale; the
observation fragment evaluates the Gaussian log density `-½log2π - logσ - ½z²`
in closed form. The example checks both the scalar value and a real reverse-mode
`Enzyme` gradient over `(x, μ, logσ)`: the value against `Distributions.logpdf`,
and the gradient against the same closed-form density written as a plain
function, so that preparation is shown to preserve AD.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.CONTINUOUS_SOURCE,
)
```

## Discrete: Bernoulli observation with a differentiable logit

The family fragment computes the two outcome log-probabilities with the stable
`-log1pexp(∓logit)` form; the observation fragment selects the observed one.
The observation is deliberately discrete, so differentiation is only with
respect to the continuous logit. The kernel value is checked against a
`Bernoulli` reference and the one-dimensional `Enzyme` gradient against the
native closed-form density.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.DISCRETE_SOURCE,
)
```

## Multivariate: shared regression coefficients

One coefficient vector feeds both an isotropic-Gaussian prior and an
isotropic-Gaussian regression likelihood, each written as a native quadratic
form (no `MvNormal` object). A third fragment sums the two selected terms. This
makes the shared parameter visible to the planner and demonstrates a
multivariate composition without proposing a model DSL or sampler API. The joint
value is checked against direct `MvNormal` calls, and the gradient with respect
to the shared coefficients — held constant against the design, observations, and
scales via `DifferentiationInterface`'s `Constant` — against the native total.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.MULTIVARIATE_SOURCE,
)
```

The sources record `BenchmarkTools.@ballocated` measurements behind a function
barrier for both each prepared kernel and its equivalent direct
`Distributions.jl` oracle call. The scalar paths measure **zero bytes** on both
sides. The multivariate native path measures **strictly fewer bytes than the
`MvNormal` reference**, because the quadratic form avoids distribution-object
construction — reversing the situation of a wrapper that inherits the library's
allocations. The scalar `Base.return_types` results are checked as exactly
`Float64`, so allocation and inference are separate claims rather than an `Any`
result being mistaken for a distribution cost. The separate non-allocating path
that drives even the multivariate case to zero bytes is documented under
[Non-allocating kernels](nonallocating.md).

## The planner is domain-agnostic

None of this adds probability semantics to `ReactiveKernels`; the densities are
just ordinary recipes. The planner is equally happy to carry an *opaque* library
object across ports — a `Distributions.Normal` value passed from one fragment to
another and consumed by `logpdf` — if a future layer prefers to reuse an existing
distribution library rather than write densities natively. That interop works,
but it makes the library the compute engine and inherits its allocation profile,
which is why the examples above keep the arithmetic native.
