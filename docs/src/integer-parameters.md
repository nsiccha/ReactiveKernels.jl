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
marginal log likelihood. An integer array is summed out element by element,
as the mixture below shows.

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

For explicit scalar integer parameters, `prepare_gibbs` derives one finite
conditional per declaration and one continuous conditional over all real
coordinates. The driver draws the integers exactly, then takes one
WarmupHMC NUTS transition on the real coordinates. Load `WarmupHMC`,
`LogDensityProblems` and `Random` to enable the driver.

```@example integer
using WarmupHMC, LogDensityProblems, Enzyme, DifferentiationInterface

problem = prepare_gibbs(count_built, count_plan, u; discrete,
    backend = AutoEnzyme(; mode = Enzyme.Reverse))
state = initialize_gibbs(Xoshiro(42), problem, u; discrete)
draws = gibbs_draws(state, 1_000)
draws.parameters.p
draws.parameters.k
```

`initialize_gibbs` adapts the NUTS kernel during `warmup` sweeps (1000 by
default). Each warm-up sweep draws the integers exactly and then moves the
real coordinates with WarmupHMC's continuously adapting kernel, which follows
the continuous conditional as the integers change. The chain then continues
from the last warm-up state with that kernel fixed, so every saved sweep is an
ordinary Markov chain transition. `state.warmup` reports the warm-up sweeps,
their divergences and gradient evaluations. To supply the kernel instead,
pass `stepsize` and optionally `metric`, a scale `L` with covariance `L*L'`;
then no warm-up runs. `discard` only omits sweeps. `gibbs_step!(state)`
advances one sweep and returns its divergence flag. Save multiple chains and check R-hat, bulk and tail ESS, and divergences
before interpreting posterior summaries.

Each conditional omits whole emitted density terms that do not read its
block. For the RAG model, the integer conditional omits `p`'s prior, while
the continuous conditional omits the honest-claims factor. These conditional
log densities differ from the full joint by block-constant terms. The driver
checks that the initial state has finite full joint density.

The default visits scalar integers separately in declaration order. For
strongly coupled or deterministically linked integers, use a joint block,
for example `blocks = ((:k, :z),)`. Every free integer must appear exactly
once. A nested name is written `Symbol("child.k")`; the state still uses
nested named tuples. A joint block enumerates the Cartesian product, so
its cost grows with the product of support sizes. Single-site updates can
be unable to move between valid states when integers are deterministically
linked; joint updates can move them together.

The chain owns its position, integer values and RNG streams. Integer values
are stored as `Int`, including when the starting value has a narrower type.
Saved positions and constrained draws remain unchanged by later steps.
Prepare a separate problem for each concurrent chain. Purely discrete
models use the same driver without an AD backend, metric or step size.

Some integers can be summed out while the others are updated: on
`marginalize(model, :a)`, `prepare_gibbs` updates every remaining integer from
its collapsed conditional, which sums over `a`, and the continuous step
targets the density with `a` summed out. A model whose integers are all
marginalized has nothing left to update; sample its real coordinates with
NUTS instead.

```@example integer
collapsed = @rkppl begin
    a ~ Bernoulli(0.3)
    j ~ Binomial(4, 0.45)
    y ~ Normal(j + 2a, 1)
end
collapsed_plan = marginalize(collapsed(), :a) | (; y = 2.6)
collapsed_built = build_kernel(collapsed_plan)
collapsed_problem = prepare_gibbs(collapsed_built, collapsed_plan, Float64[];
    discrete = (; j = 0))
weights = only(collapsed_problem.integers).kernel(Float64[], (; j = 0)).logweights
exact = [pdf(Binomial(4, 0.45), j) *
    (0.7 * pdf(Normal(j, 1), 2.6) + 0.3 * pdf(Normal(j + 2, 1), 2.6)) for j in 0:4]
exp.(vec(weights) .- maximum(weights)) ./ sum(exp.(vec(weights) .- maximum(weights))) ≈
    exact ./ sum(exact)
```

An integer parameter indexes an array as Julia does, so a mixture assigns one
component per observation:

```@example integer
mixture = @rkppl begin
    w ~ Dirichlet(ones(K))
    mu[1:K] .~ Normal.(0, 5)
    z[axes(y, 1)] .~ Categorical(w)
    y .~ Normal.(mu[z], 1.0)
end
mixture_y = [-2.1, -1.9, 2.0, 2.2, 1.8]
mixture_plan = mixture(; K = 2) | (; y = mixture_y)
mixture_built = build_kernel(mixture_plan)
mixture_z = [1, 1, 2, 2, 2]
mixture_u = unconstrain(mixture_built.layout,
    (; w = [0.5, 0.5], mu = [-2.0, 2.0], z = mixture_z))
mixture_problem = prepare_gibbs(mixture_built, mixture_plan, mixture_u;
    discrete = (; z = mixture_z), backend = AutoEnzyme(; mode = Enzyme.Reverse))
mixture_state = initialize_gibbs(Xoshiro(1), mixture_problem, mixture_u;
    discrete = (; z = mixture_z), warmup = 200)
mixture_draws = gibbs_draws(mixture_state, 500)
sum(sort(mixture_draws.parameters.mu; dims = 1); dims = 2) ./ 500
```

The two components are exchangeable, so the chain may swap their labels; the
summary sorts each draw's means, which is label invariant.

Each declared integer array is a block of its own, after the scalar blocks.
Here every density term that reads `z` sums cells that read only their own
element, so the elements are conditionally independent: one evaluation of
those cells per support value gives every element's log weight, and all
elements are drawn together. A term that couples elements, such as one reading
`sum(z)`, makes the driver draw the elements one after another, each from its
exact conditional. Julia checks integer indices when the program runs, so an
index outside the indexed array raises a `BoundsError`.

Each element may have its own support when a bound is read per element:
`Binomial.(n, p)` with a vector of trials `n`, `DiscreteUniform.(lo, hi)`
with vector endpoints, or `Categorical.(ws)` over data that holds one
probability vector per element, which reads element `i`'s law from `ws[i]`, as
in Julia. Single-site updates enumerate each element's own range. The joint
update evaluates each value that some element can take once; an element whose
own support excludes that value keeps an in-support value in that evaluation
and gets probability zero:

```@example integer
trials = [2, 5, 1]
count_array = @rkppl begin
    k[axes(trials, 1)] .~ Binomial.(trials, 0.35)
    y .~ Normal.(k, 0.7)
end
counts_y = [1.2, 3.9, 0.1]
counts_plan = count_array(; trials) | (; y = counts_y)
counts_built = build_kernel(counts_plan)
counts_problem = prepare_gibbs(counts_built, counts_plan, Float64[];
    discrete = (; k = [0, 0, 0]))
counts_draws = gibbs_draws(initialize_gibbs(Xoshiro(3), counts_problem,
    Float64[]; discrete = (; k = [0, 0, 0])), 4_000)
# The second element's exact conditional over its own 0:5.
exact = [pdf(Binomial(5, 0.35), s) * pdf(Normal(s, 0.7), counts_y[2]) for s in 0:5]
frequency = [count(d -> d.k[2] == s, counts_draws.discrete) for s in 0:5] ./ 4_000
all(d -> all(0 .<= d.k .<= trials), counts_draws.discrete) &&
    maximum(abs.(frequency .- exact ./ sum(exact))) < 0.03
```

The same mixture can be written as a loop, one observation per cell. The
loop's integers are the same parameter as the declared array: densities,
gradients and Gibbs draws agree, and each cell reads only its own element, so
all elements are still drawn together:

```@example integer
loop_mixture = @rkppl begin
    w ~ Dirichlet(ones(K))
    mu[1:K] .~ Normal.(0, 5)
    @plate for i in eachindex(y)
        z[i] ~ Categorical(w)
        y[i] ~ Normal(mu[z[i]], 1.0)
    end
end
loop_plan = loop_mixture(; K = 2) | (; y = mixture_y)
loop_built = build_kernel(loop_plan)
loop_u = unconstrain(loop_built.layout,
    (; w = [0.5, 0.5], mu = [-2.0, 2.0], z = mixture_z))
prepare_query(loop_built, loop_plan, :sampler)(loop_u, (; z = mixture_z)) ≈
    prepare_query(mixture_built, mixture_plan, :sampler)(mixture_u, (; z = mixture_z))
```

A cell's `Categorical` reads its probabilities as one whole vector: a sampled
simplex (`w`) or bound data. Data arrays are indexed by integers too, so
known component centers read `y .~ Normal.(centers[z], s)`, and a scalar
integer reads one entry (`centers[k]`).

The assignments can instead be summed out, `marginalize(model, :z)`, when
every density term that reads `z` sums cells that read only their own
element, as in both spellings above. The density then holds each
observation's log-sum-exp over the components, at a cost linear in the
observations and the components, and equals the components written out as
`MixtureModel.(…)` in value and gradient. Each element's posterior is under
`parameters.elements`. A term that couples elements, such as one reading
`sum(z)`, is refused: summing it out would enumerate every joint assignment.
For each response entry, `predictive_laws` returns the mixture weighted by
that element's posterior. A joint `predictive_draws` call draws the element's
component once and uses it for every response that reads the element. In JAX,
request `values = (:pointwise, :_ppl_latents)` to recover the per-entry log
likelihoods and component probabilities alongside the density.

```@example integer
summed_plan = marginalize(mixture(; K = 2), :z) | (; y = mixture_y)
summed_built = build_kernel(summed_plan)
summed_u = unconstrain(summed_built.layout, (; w = [0.5, 0.5], mu = [-2.0, 2.0]))
assignments = prepare_discrete_latents(summed_built, summed_plan)
law = MixtureModel(Normal.([-2.0, 2.0], 1.0), [0.5, 0.5])
(; likelihood = prepare_query(summed_built, summed_plan, :likelihood)(summed_u),
   written_out = sum(y -> logpdf(law, y), mixture_y),
   probabilities = assignments(summed_u).parameters.elements.z.probabilities,
   draw = discrete_latent_draws(assignments, summed_u, Xoshiro(1)).parameters.z)
```

JAX execution is verified for all four families, including ordinary
gradients, changed data sizes, integer-dependent observation trials, and
Categorical matrices with empty axes. Pass named integer dictionaries beside
`u`. Marginalized models execute in JAX too, one integer or several summed
jointly: their density, ordinary gradients, recovered integer probabilities
and pointwise terms, empty observations included. A pointwise field that holds
nested arrays still raises `JAXEmissionError`.
