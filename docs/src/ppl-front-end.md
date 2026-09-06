# Declarative `@ppl` front-end + Gibbs (experimental)

```@eval
Main.ReactiveKernelsDocs.render_review_status(:review_pending_ppl)
```

`ReactiveKernels` is a general math authoring/encoding layer, not a
probabilistic-programming or sampling library. The examples in this satellite
package (`ReactiveKernelsPPLExamples`) assemble PPL semantics *on top of* the
core graph — most of them by hand (see [Eight schools](eight-schools.md)).

This page documents an **experimental** convenience layer built entirely on
ReactiveKernels' public surface (`@kernel` / `prepare` / `plan` / `plate` plus
the reusable distribution-kernel objects), with **nothing added to
ReactiveKernels core**:

- `PPLMacro.@ppl` — a StanBlocks-like declarative `~` model front-end that lowers
  to an ordinary `@kernel` graph exposing the canonical `PPLWorkflow` node set,
  so a macro-produced model is queried exactly like a hand-authored one.
- `PPLGibbs.gibbs` — a Gibbs-sampling layer over `@ppl` models, driven
  incrementally through `ReactiveState`.

Both are deliberately **unexported** and are **not** part of the consumer API
(`reactivekernels-use` does not mention them). Reach them only through the fully
qualified submodule paths, and do not depend on them until they are reviewed and
approved:

```julia
using ReactiveKernelsPPLExamples.PPLMacro: @ppl
using ReactiveKernelsPPLExamples.PPLGibbs: gibbs, Gibbs
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, beta, binomial
```

## Authoring a model with `@ppl`

An `@ppl` model is `name(data...) = begin … end` with `~` sampling statements.
The left-hand side of a parameter `~` is StanBlocks-faithful typed-LHS — `name`,
`name::real`, or `name::vector[size]` — and the right-hand side is one of the
reusable distribution objects. A `~` whose left-hand side is a declared *data*
argument is an observation (a likelihood term); every other `~` declares a
latent parameter.

```julia
@ppl linreg(predictors::Vector{Float64}, responses::Vector{Float64}) = begin
    alpha ~ normal(0.0, 10.0)
    beta  ~ normal(0.0, 10.0)
    sigma ~ positive(normal(0.0, 5.0))          # half-Normal(5): a support constraint
    responses ~ normal(alpha + beta * predictors, sigma)
end
```

The macro owns a small PPL AST plus a posterior-mode parameter/observation
analysis and lowers the model to a `KernelSpec`. It infers each parameter's
support from its family and authors the matching unconstrained↔constrained
transform and log-Jacobian, so the model is queried the usual way — pack the
unconstrained parameters, hand over the data, and ask for a workflow node:

```julia
prepare(linreg; have = (:unconstrained, :predictors, :responses),
        want = :posterior)(unconstrained, predictors, responses)

# or the committed sampler cut, exactly as for a hand-authored model:
prepare(linreg; have = (:unconstrained, :predictors, :responses),
        want = PPLWorkflow.workflow_wants(:sampler))(unconstrained, predictors, responses)
```

Supported surface (first cut):

- **Real** parameters (`normal` / `cauchy` / `laplace`) — identity transform.
- **Positive** parameters (`exponential` / `gamma` / `lognormal` /
  `inverse_gamma`) — log/exp transform. A `positive(<real family>)` combinator
  makes a real family a **half distribution** (e.g. `positive(cauchy(0, 5))` is a
  half-Cauchy).
- **Unit** parameters (`beta`) — logit/logistic transform.
- **Real vector** parameters — `name::vector[size] ~ dist(…)`; the size is a data
  argument or literal, the prior a summed per-element `plate`.
- **Discrete latents** (`bernoulli`) — a Gibbs-only latent with no unconstrained
  coordinate, carried as a `Bool` port (`z ~ bernoulli(ω)` or a vector
  `z::vector[p] ~ bernoulli(ω)`), used naturally as e.g. `ifelse(z, …)`.

This reproduces the hand-written `beta_binomial`, `poisson_gamma`,
`linear_regression`, and `eight_schools` example densities exactly.

## Sampling with `PPLGibbs.gibbs`

Each block's full conditional is a slice of the one joint `@ppl` graph, and a
`ReactiveState` drives the sweep: setting a block auto-invalidates only its
Markov blanket, so a block update recomputes only the terms that actually depend
on it — no hand-written caching. The named latents of the model are the blocking
vocabulary.

```julia
@ppl bb(successes::Vector{Int}, trials::Vector{Int}) = begin
    rate ~ beta(2.0, 2.0)
    successes ~ binomial(trials, rate)
end

res = gibbs(bb; blocks = [:rate], data = (; successes, trials),
            init = (; rate = 0.5), iters = 20_000, rng = MersenneTwister(1))
res.draws[:rate]        # the retained chain
res.accept_rate[:rate]  # per-block acceptance (1.0 for an exact draw)
```

The sampler chooses, per block:

- **Automatic conjugacy** (`conjugate = true`, default) — a block whose full
  conditional is a supported data-only conjugate pair is drawn in closed form
  (exact, always accepted): **Beta-Bernoulli**, **Beta-Binomial**,
  **Gamma-Poisson** (closed-form draws use `Random`-only samplers).
- **Discrete enumeration** — a discrete (`bernoulli`) latent is sampled by exact
  enumeration of its full conditional (scalar, or single-site per element for a
  vector block).
- **Support-aware random walk** — otherwise a Metropolis-within-Gibbs step; a
  `:positive` / `:unit` block (`support` kwarg) walks in log / logit space with
  the matching Metropolis-Hastings correction, so proposals respect the support.

Blocking and per-block sampler choice are explicit (`blocks`):

```julia
gibbs(model; blocks = Gibbs(:sigma, (:alpha, :beta)), …)  # group (alpha, beta) into one joint block
gibbs(model; blocks = Gibbs(:rate => :rw), …)             # force the random-walk sampler on :rate
```

A plain `Vector{Symbol}` is shorthand for one single-site block per latent.

## The ReactiveKernels / PPL boundary

The PPL-specific analysis lives here, not in ReactiveKernels: which statements
contribute to the prior versus the likelihood does not fall out of the graph
alone — it depends on the combination of model and conditioned data — and the
constrained↔unconstrained transforms are PPL-specific. ReactiveKernels supplies
the general graph, planning, `plate`, and `ReactiveState` machinery on which this
layer is built; it stays a general authoring/encoding layer and never becomes a
PPL or sampling library.
