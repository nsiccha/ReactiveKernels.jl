# Finite integer parameters

`Binomial`, `Bernoulli`, `Categorical` and `DiscreteUniform` declarations have
integer state. Their named values travel beside the real vector `u`; they
occupy no real coordinates and have no transformation Jacobian.
Binomial trials, DiscreteUniform endpoints and restriction bounds read data.
Categorical probabilities may read a sampled simplex.

```@example integer
using ReactiveKernelsPPL, Distributions, Random

count_model = @rkppl begin
    p ~ Beta(2, 3)
    k ~ Binomial(n, p)
    y .~ Normal.(k, 1.0)
end
count_plan = count_model(; n = 5) | (; y = [1.2, 2.1])
count_built = build_kernel(count_plan)
u = unconstrain(count_built.layout, (; p = 0.4, k = 2))
discrete = (; k = 2)
count_density = prepare_query(count_built, count_plan, :sampler)
(; density = count_density(u, discrete),
   parameters = constrain(count_built.layout, u, discrete))
```

`unconstrain` returns the continuous coordinates. Keep the integer values
separately. Pass them after `u` when calling a prepared query, a predictive
query or `predictive_laws`. For `prepare_sampler` and `prepare_query_ad`,
supply `discrete` as a preparation keyword; derivatives are with respect to
the real vector. `restore_draws(layout, U; discrete)` accepts either one
named integer state for every draw or a vector of states, one per column
of `U`, and preserves integer element types in the returned draws.
`constrain` copies integer arrays, so a saved draw is independent of later
changes to the supplied integer state.

Cell, partition, compact derivative, density-factor, slice, window and SMC
preparations also take `discrete`. They retain a copy of that state for the
prepared conditional query. Prepare again to change it: integer indices can
change the row selections and coordinate footprints those queries cache.
Nested submodels use nested names in the state, as in constrained draws.
Normalized `truncated` integer laws are not implemented yet; `restricted`
keeps the base mass on its declared integer interval.

`marginalize(model, :k)` sums out a finite integer declaration exactly.
The factor catalogue does not yet represent joint prior events after this
elimination; use the complete `:prior` query for the integrated prior.
The support bounds must come from data: for example Binomial trials `n`
or DiscreteUniform endpoints. The generated graph loops over that support
at runtime, so changing `n` does not duplicate its body in generated source.

```@example integer
marginal_plan = marginalize(count_model, :k)(; n = 5) | (; y = [1.2, 2.1])
marginal_built = build_kernel(marginal_plan)
marginal_density = prepare_query(marginal_built, marginal_plan, :sampler)
latents = prepare_discrete_latents(marginal_built, marginal_plan)
(; density = marginal_density(u),
   probabilities = latents(u).parameters.probabilities,
   draw = discrete_latent_draws(latents, u, Xoshiro(1)).parameters)
```

The sampler targets the log-sum-exp over the finite states. The latent
query returns their conditional posterior probabilities, and drawing takes
place in ordinary Julia with the supplied RNG. Several marginalized
declarations are recovered jointly. Marginal observations are dependent;
their pointwise terms use sequential predictive probabilities in observation
field and column-major order (including ragged cells), whose sum is the
marginal log likelihood. Scalar integer declarations can be eliminated;
integer arrays remain explicit runtime state.

For supported observation laws, `prepare_predictive` on the marginalized
model draws one joint integer state and then every response conditional on
that state. `predictive_laws` returns each entry's marginal mixture law;
sampling those laws separately would lose the shared-state dependence.
Use `predictive_draws`, `predictive_draws!` or the prepared query for a joint
draw. Like latent recovery, these queries condition on the supplied observed
data. They require those responses even if the original, conditional model's
predictive laws could be evaluated on an unobserved design.

If the model also has mixture or hidden Markov observation latents, the
integer latent query retains their conditional posterior columns under
`_conditional`, with a trailing joint-state axis. `discrete_latent_draws`
first draws the integer state, then those observation latents conditional on
it, returning the two namespaces as `parameters` and `observations`.
An empty restricted support has density `-Inf`; conditional draws and
pointwise queries are undefined and raise an error.

An authored value or `:observations` argument that reads an eliminated integer
has no single value at the remaining `u`. Query it on the conditional model
with explicit integer state. Independent named values remain available;
joint predictive and discrete-latent queries recover their conditional laws.

JAX execution is verified for all four families, including ordinary
gradients, changed data sizes, integer-dependent observation trials, and
Categorical matrices with empty axes. Pass named integer dictionaries beside
`u`. Marginalized models execute in JAX too, one integer or several summed
jointly: their density, ordinary gradients, recovered integer probabilities
and pointwise terms, empty observations included. A pointwise field that holds
nested arrays still raises `JAXEmissionError`.
