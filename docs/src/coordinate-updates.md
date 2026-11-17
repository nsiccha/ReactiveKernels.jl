# Coordinate updates

`prepare_coordinate_conditional` prepares the conditional log density of one
packed coordinate given all others, for coordinate-wise samplers such as slice
sampling within Gibbs. The model is written as usual. When its density reads a
product `X * beta` of a constant matrix and a parameter vector, the conditional
of `beta[j]` keeps that product cached: a trial value moves it by one column,
`X[:, j] * (z - beta[j])`, so a trial costs O(n) instead of O(n d). This is the
compute-graph Gibbs construction of Luu et al. (2024, arXiv:2410.03630),
derived from the emitted program rather than written by hand.

```@example coordinates
using ReactiveKernelsPPL, Random

logistic_model = @rkppl begin
    beta[axes(X, 2)] .~ Normal.(0, sigma)
    y .~ Bernoulli.(logistic.(X * beta))
end
rng = Xoshiro(1)
X = randn(rng, 40, 5)
y = Int.(rand(rng, 40) .< 0.5)
plan = logistic_model(; X, sigma = 10.0) | (; y)
built = build_kernel(plan)
problem = prepare_coordinate_conditional(built, plan, zeros(5))
state = coordinate_state(problem, zeros(5))
coordinate_logdensity(state, 2, 0.5)
```

## Moving a coordinate

`coordinate_state(problem, u0)` holds a copy of the packed point and its cached
products; several states can share one prepared problem. The joint density must
be finite at the start.

- `coordinate_logdensity(state, k, z)` evaluates the conditional of coordinate
  `k` at `z` and leaves the state unchanged.
- `set_coordinate!(state, k, z)` moves coordinate `k` and updates each cached
  product by one column.
- `refresh_products!(state)` recomputes the products from the point and returns
  the largest absolute change, the rounding drift of the column updates.

A conditional keeps only the emitted density terms that read the moved
coordinate, so it equals the joint log density up to a constant. That constant
differs between coordinates: compare values of one coordinate only.

## Slice sweeps

`coordinate_slice_step!(rng, state, k)` updates one coordinate by a slice
sampling transition of its conditional: stepping out by doubling, shrinkage, and
the acceptance test for doubled intervals (Neal 2003). It leaves the
conditional, and therefore the joint posterior, invariant.
`coordinate_slice_sweep!(rng, state)` updates every coordinate in order. Both
return the number of conditional evaluations.

```@example coordinates
draws = map(1:500) do _
    coordinate_slice_sweep!(rng, state)
    copy(state.position)
end
(; evaluations_per_sweep = coordinate_slice_sweep!(rng, state),
   posterior_mean = sum(draws[251:end]) / 250,
   drift = refresh_products!(state))
```

The defaults are an initial width of 10 and at most 20 doublings, as in the
slice sampler of Pigeons.jl. `width` may also be a vector with one width per
coordinate. A sweep typically costs a few evaluations per coordinate.

## What is cached

- A product `A * w` is cached when `w` is a contiguous run of packed
  coordinates read whole, as an unconstrained vector parameter is, and `A` does
  not depend on any coordinate.
- A coordinate of such a vector evaluates only the terms that read the vector.
  A term summing a plate over the vector's cells, such as its prior or its
  Jacobian, is read at the moved coordinate's cell alone.
- Every other coordinate, a shared scale for example, evaluates all terms that
  read any coordinate.

Not covered yet: models with free integer parameters are refused, a product
over a transformed vector (a positive coefficient vector, for example) is
recomputed in full, and the fused `BernoulliLogitGLM` spelling hides its
product from the cache. Execution is native; there is no JAX coordinate
sampler.

## When it pays

Coordinate updates mix well when the coefficients are weakly correlated, and the
cached product makes each sweep cost about as much as a few full density
evaluations whatever the number of coefficients. On a Gaussian-prior logistic
regression of 62 gene-expression samples with 2,000 predictors (the colon data
of Luu et al.), slice sweeps reached an effective sample size per second within
a factor of two of WarmupHMC's NUTS on the same model, and ahead of it by the
paper's median-ESS summary. Many predictors alone are not enough: on a
synthetic design with 1,024 predictors and 128 observations, NUTS was about
seven times more efficient. With fewer coefficients than observations, or with
strongly correlated coefficients, NUTS through `prepare_sampler` was ten to a
hundred times more efficient. Start from NUTS, and measure coordinate updates
against it on the model at hand before switching.
