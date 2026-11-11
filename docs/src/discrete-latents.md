# Conditional latent states

This example simulates a hidden Markov chain with two states and a continuous observation at each step. The transition matrix favors staying in the current state. Each state has its own normal emission mean, with a shared known observation scale. The model sums over the latent path when evaluating its density. Afterward, a conditional-latent query recovers state probabilities at supplied values of the continuous parameters; no statistical fit runs in this documentation build.

## Data

A seeded random number generator makes the simulated observations reproducible. The state sequence is retained only as a diagnostic because it is known in this simulated dataset. The inference query receives the observations, not the generating states.

```@example latents
using ReactiveKernelsPPL, Random, AlgebraOfVega
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal_hmm_logdensity
using Main.Tutorials

rng = Xoshiro(2019)
transition = [0.90 0.10; 0.10 0.90]
means = [0.0, 2.0]
noise = 0.7
steps = 40
states = zeros(Int, steps)
states[1] = rand(rng, 1:length(means))
for t in 2:steps
    states[t] = min(searchsortedfirst(
        cumsum(transition[states[t - 1], :]), rand(rng)), length(means))
end
y = means[states] + noise * randn(rng, steps)
(; observations=y, generating_states=states)
```

The observation scale allows the emission distributions to overlap. A single observation can therefore leave uncertainty about its state even when the transition matrix and emission means are known. Neighboring observations provide further information through the transition model. The example keeps those parameters fixed at their generating values so that this conditional uncertainty is separate from uncertainty about the parameters themselves.

## Model

Each transition row has a Dirichlet prior. The emission means have normal priors. The distribution kernel used for the observations evaluates the marginalized hidden Markov likelihood, with a uniform initial-state distribution. The state count, prior settings and observation scale enter through the model binding.

```@example latents
model = @rkppl begin
    eachrow(T[1:K, 1:K]) .~ Dirichlet(concentration)
    m[1:K] .~ Normal.(mean_prior, mean_scale)
    y ~ LogDensity(normal_hmm_logdensity, m, noise, T, initial)
end

K = length(means)
bound = model(;
    K, concentration=ones(K) ./ K,
    mean_prior=means, mean_scale=1.0, noise,
    initial=ones(K) ./ K,
) | (; y)
built = build_kernel(bound)
u = unconstrain(built.layout, (; T=transition, m=means))
target = prepare_query(built, bound, :sampler)
target(u)
```

The sampler-facing vector contains continuous transition and emission parameters. It does not contain a discrete coordinate for every step. Summing out the path preserves a continuous density for those parameters while retaining a separate query for the conditional states.

The generated model view shows the program used for these queries. Its dependency graph and generated expressions come from the built model rather than a second authored representation of the hidden Markov calculation.

```@example latents
Tutorials.html(model_view(built; bound))
```

## Conditional probabilities

Prepare the discrete-latent query once, then evaluate it at the unconstrained parameter vector. The returned response field contains probabilities for every step and state, together with the filtered and transition quantities used for path draws.

```@example latents
latents = prepare_discrete_latents(built, bound)
posterior = latents(u).y
probabilities = posterior.probabilities
most_probable = [argmax(probabilities[t, :]) for t in eachindex(y)]
(; probabilities, most_probable)
```

These are conditional state probabilities given the complete observation sequence and the supplied parameter values. They are not probabilities averaged over a posterior distribution of transition matrices or emission means. An application with parameter draws can evaluate the same prepared query at those draws and combine the results using the averaging appropriate to its inference task.

The figure follows the probability of the second state across the sequence. Values between the extremes expose remaining state uncertainty. Abrupt changes can correspond to evidence for a transition, while observations near the overlap of the emission distributions can leave a less certain assignment. The figure is generated from the returned probability matrix.

```@example latents
plot = data((; step=collect(eachindex(y)), probability=probabilities[:, 2])) *
    mapping(:step => "Sequence step (index)",
            :probability => "Conditional probability of state 2") *
    visual(Lines; color="#2a78d6")
Tutorials.figure(plot, "conditional-states",
    "Conditional probability of state 2 given the complete observation sequence and the supplied transition and emission parameters.")
```

## Diagnostics

Each row must be a probability distribution. Check normalization, finiteness and nonnegativity before interpreting its largest entry. The displayed row-sum error quantifies the normalization check; the assertions make the build fail if the returned object violates the probability contract.

```@example latents
row_sum_error = maximum(abs, vec(sum(probabilities; dims=2)) .- 1)
minimum_probability = minimum(probabilities)
@assert all(isfinite, probabilities)
@assert minimum_probability >= 0
@assert row_sum_error < 1e-12
(; row_sum_error, minimum_probability)
```

A conditional draw samples an entire path, preserving dependencies between steps. Drawing each state's marginal independently would generally produce a different distribution over paths. The API therefore exposes a path-draw operation alongside the probability query.

```@example latents
path = discrete_latent_draws(latents, u, Xoshiro(1)).y
@assert length(path) == length(y)
@assert all(s -> 1 <= s <= K, path)
(; path, valid_states=all(s -> 1 <= s <= K, path))
```

A modal state at each step is a collection of marginal decisions. It need not be the most probable complete path, and it need not equal a sampled path. The two outputs answer different questions. Likewise, matching the generating states in this simulated dataset would not establish calibration across datasets; that would require a separate simulation study with an explicit design.

## Query lifetime

The prepared query is reusable for parameter vectors with the same layout. A changed observation sequence requires a compatible new binding and query preparation. Keep the layout attached to the built model: constructing a vector from an assumed parameter order risks evaluating a valid numerical vector with the wrong interpretation.

The [probabilistic-graph guide](rkppl.md) explains this binding and coordinate boundary. The [scan guide](scan.md) introduces sequential recurrences at the core graph level, while the [PPL reference](ppl-api.md) documents the conditional-latent result and path-draw interfaces. Distribution-specific latent queries may return different structures; use the contract of the selected observation distribution.

```@example latents
Tutorials.receipt("discrete-latents"; checks=(
    row_sum_error, minimum_probability,
    valid_states=all(s -> 1 <= s <= K, path),
))
```

## State labels

State labels identify rows and columns of the transition matrix and entries of the emission-mean vector. The probability column selected for the figure uses that same ordering. Relabeling states consistently leaves the observation model unchanged but changes which label appears on the plotted column. Keep this correspondence when combining results across parameter draws; an inconsistent permutation would average probabilities for different states.

This example avoids that ambiguity by evaluating a single supplied parameter point with named generating values. The retained simulated path helps explain the data construction, but the normalization and path-validity diagnostics do not depend on recovering that particular random path. They check the conditional-query contract on its own terms.

```@raw html
<details><summary>Reproduce</summary>
```

The seed and supplied parameters appear above. Run these shell commands from the repository root.

```sh
julia --project=docs docs/instantiate.jl
julia --project=docs docs/make.jl
```

Download the <a href="./receipts/discrete-latents.toml" download>execution receipt</a>. The [research README](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/docs-history/README.md) describes the location of historical experiments.

```@raw html
</details>
```
