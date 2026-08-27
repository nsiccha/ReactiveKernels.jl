# Native log densities as recipes

These executable examples build log densities the way a probabilistic-programming
layer on top of `ReactiveKernels` would want them: as ordinary `@kernel` recipes
of closed-form arithmetic. The **compute path contains no `Distributions.jl`
call** — each density is written out directly, a `have`/`want` `prepare` query
lowers exactly the requested computation to a straight-line kernel, and `compose`
enters only where several recipes share a parameter (the multivariate case
below). That keeps the hot path free of distribution-object construction and
friendly to batching, lazy evaluation, and the non-allocating lowering under
[Non-allocating kernels](nonallocating.md).

Each density is written in terms of what you already **have**. When the log scale
`logσ` is a given, the density names it directly for the `-logσ` term and derives
`σ = exp(logσ)` only where the scale itself is needed — there is no `exp`-then-`log`
round trip. The planner does `have`/`want` min-cost planning and structural common
subexpression elimination, not algebraic identities like `log∘exp = id`, so the
minimal work comes from stating the recipe naturally, not from a post-hoc
simplification pass.

`Distributions.jl` appears **only as an independent oracle**: it supplies a
reference value (and a reference allocation figure) that the native kernel is
checked against. It is an example/test/documentation dependency, never a
dependency of the package runtime and never part of the compute path.

`ReactiveKernels` **plans and computes** the density; it provides **no AD of its
own** — no pullbacks — and AD is orthogonal to it, so nothing here differentiates.
Each kernel value is checked against the `Distributions.jl` oracle; that is the
only role the library plays.

Each panel is produced from one build-executed artifact. **Raw input** retains
the literal compact source evaluated during this documentation build, alongside
the established origin, executed-input, and actual-output annotations.
**Generated kernel** is `code_expr` from that execution — inspect it to confirm
the lowered kernel names no distribution library — and **Compute DAG** is the
live `visualize(plan)` component. **Compare all** moves those same three views
into a side-by-side dialog.

## Continuous: Normal location and log scale

A single recipe evaluates the Gaussian log density `-½log2π - logσ - ½z²` in
closed form from `have = (x, μ, logσ)`. Because `logσ` is a given, the `-logσ`
term uses it directly and `σ = exp(logσ)` is derived once, only for the
standardized residual `z = (x - μ)/σ` — no `log(exp(logσ))` round trip. The
scalar value is checked against `Distributions.logpdf`.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.CONTINUOUS_SOURCE,
)
```

## Discrete: Bernoulli observation with a logit

A single recipe selects between the two numerically stable outcome
log-probabilities `-log1pexp(∓logit)` on the observed bit. The kernel value is
checked against a `Bernoulli` reference.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.DISCRETE_SOURCE,
)
```

## Multivariate: shared regression coefficients

This is the case that actually needs `compose`. One coefficient vector feeds
both an isotropic-Gaussian prior and an isotropic-Gaussian regression
likelihood, each written as a native quadratic form (no `MvNormal` object), and
a third recipe sums the two terms. `compose` unifies the shared `coefficients`
port across those separately-authored recipes so the planner sees one parameter
feeding both densities — which is what `compose` is *for*, and why the scalar
examples above, being single recipes, use none. The joint value is checked
against direct `MvNormal` calls.

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
