# Discrete latent variables

A model with a discrete latent, such as a hidden Markov model's state path, a
mixture's component labels or a change point, is often sampled by Gibbs
sampling: a gradient-based sampler moves the continuous parameters while
another sampler moves the discrete latents, for example Turing's
`Gibbs(HMC(…, :m, :T), PG(…, :s))`, whose particle Gibbs step on the states
becomes sticky when its particles degenerate.

`@rkppl` takes the other exact route. The observation's law sums the discrete
latent out, so the sampler (NUTS on `prepare_sampler`, see
[Writing models with `@rkppl`](rkppl.md)) sees only the continuous parameters, on their exact marginal posterior. The
latents are then recovered per posterior draw: `prepare_discrete_latents`
computes their exact posterior given the parameters, and
`discrete_latent_draws` draws them from it. A parameter draw with its latent
draw is an exact draw from the joint posterior; no particle approximation and
no alternation between blocks is involved.

`prepare_discrete_latents` covers two summed-out latents:

- a whole-event hidden Markov observation, `y ~ LogDensity(law, …)` with one of
  the RK hidden Markov laws (`normal_hmm_logdensity`,
  `exponential_hmm_logdensity`, `hmm_logdensity` and their log-weight twins);
- a plain finite-mixture response, `y .~ MixtureModel.(…)`.

Any other summed-out latent is recovered as an authored value of the model,
as the change-point example below shows.

Each example on this page runs at docs-build time, exactly as displayed. The
same files are checked against independent references by ReactiveKernelsPPL's
`test_discrete_latents.jl`. They evaluate the latents at the data-generating
parameters; in a fit, use each posterior draw instead (see "Over posterior
draws" below).

## A hidden Markov model

The model of Turing's Bayesian HMM tutorial: three states, each emitting
around its own mean, with Dirichlet transition rows.

```@eval
Main.DiscreteLatentDocs.render_source("hmm.jl")
```

```@eval
Main.DiscreteLatentDocs.render_hmm()
```

`posterior.probabilities[t, k]` is the probability of state `k` at step `t`
given the whole sequence (forward-backward smoothing).
`posterior.log_filtered[t, k]` is its log probability given the first `t`
steps alone, and `posterior.log_transition` holds the log transition weights
the law reads. Those last two are the arguments of the path's law:
`discrete_latent_draws` draws the last state from its filtered probabilities,
then each earlier state given its successor (forward filtering, backward
sampling), with `rng` as the only source of randomness.

Each hidden Markov law has a state-posterior twin with the same arguments,
which the query calls on the observation's own arguments:
`normal_hmm_state_posterior` for `normal_hmm_logdensity`, and so on. The law
and its twin share one forward pass; the twin adds a backward history scan
over the (step, state) cells:

```@eval
Main.DiscreteLatentDocs.render_kernel("_hmm_log_weights_forward")
```

```@eval
Main.DiscreteLatentDocs.render_kernel("hmm_log_weights_state_posterior")
```

A law of another emission family (a module-visible `@kernel` that fills the
log emissions and returns `hmm_logdensity(…)`) has no twin the query knows. Author its
posterior as a value instead, `states = hmm_state_posterior(log_emissions,
transition, initial)`, and draw from it with `draw_hmm_states(rng, states)`.

## A Gaussian mixture

The one-dimensional form of Turing's Gaussian mixture tutorial, which samples
the assignments with particle Gibbs:

```@eval
Main.DiscreteLatentDocs.render_source("mixture.jl")
```

```@eval
Main.DiscreteLatentDocs.render_mixture()
```

`memberships[i, k]` is the probability that observation `i` came from
component `k`: the weighted component densities the likelihood sums,
normalized. The rows follow the response's order. Each observation's
component is drawn independently.

Mixtures with evidence (censoring, truncation), weights, missing entries,
ranges, packed rows or one array per `@plate` index are refused by
`prepare_discrete_latents`, naming the response; so are hidden Markov
observations inside `@plate` cells.

## A change point

A switch year is summed out by an ordinary model function. Its posterior is
then an authored value, which any query of the model can read:

```@eval
Main.DiscreteLatentDocs.render_source("changepoint.jl")
```

```@eval
Main.DiscreteLatentDocs.render_changepoint()
```

## Over posterior draws

With posterior draws `U` of the parameters (one per column, from any sampler
on `prepare_sampler(built, plan, u0)`), one latent draw per parameter draw
gives draws from the joint posterior:

```julia
using Random
rng = Xoshiro(1)
paths = [discrete_latent_draws(latents, u, rng).y for u in eachcol(U)]
state_probabilities = sum(latents(u).y.probabilities for u in eachcol(U)) ./ size(U, 2)
```

Averaging the conditional probabilities over the draws, as in the last line,
estimates each step's posterior state probabilities with less Monte Carlo
error than counting the drawn paths.

`discrete_latents_node(built)` names the graph value the query evaluates. WANT
it in a cut of `built.spec` and `vectorize` that cut over the draws; its leaves
are numeric arrays, stacked on a trailing draw axis, and
`discrete_latent_draws(value, rng; position = k)` draws the latents of draw `k`.
