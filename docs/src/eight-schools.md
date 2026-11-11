# Eight schools

The data are estimated treatment effects and their reported standard errors from the [public Stan eight-schools example](https://github.com/stan-dev/example-models/blob/master/misc/eight_schools/eight_schools.data.R), in test-score units. Each school has an unknown effect, with a common population location and scale. This page evaluates the centered hierarchical model at supplied parameter values. It does not fit a posterior. Keeping the parameter point explicit makes the density decomposition, support transformation and observation contributions directly checkable.

## Model

A normal population distribution connects the school effects. The population location has a normal prior, and the positive population scale has a half-Cauchy prior. The observed effect in each school has a normal likelihood with its reported standard error. Prior settings are bound inputs; changing those numerical values does not require authoring another model.

```@example schools
using ReactiveKernelsPPL, Distributions, AlgebraOfVega
using Main.Tutorials

y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
prior_location = 0.0
prior_scale = 5.0

model = @rkppl begin
    mu ~ Normal(prior_location, prior_scale)
    tau ~ truncated(Cauchy(prior_location, prior_scale), 0, Inf)
    theta[axes(y, 1)] .~ Normal.(mu, tau)
    y .~ Normal.(theta, sigma)
end

bound = model(; sigma, prior_location, prior_scale) | (; y)
built = build_kernel(bound)
point = (; mu=5.0, tau=3.0, theta=fill(5.0, length(y)))
u = unconstrain(built.layout, point)
(; coordinate_names=coordinate_names(built.layout), unconstrained=u)
```

The layout supplies the coordinate order and transformations. The positive population scale needs an unconstrained representation, while the location and school effects remain real-valued. Constructing coordinates through the layout keeps the example independent of a guessed packing order. The displayed vector is a parameter point, not a posterior draw or an estimate derived from the observations.

The generated model view exposes the actual program and its graph. The source of each model term remains available for inspection beside the queries that consume it.

```@example schools
Tutorials.html(model_view(built; bound))
```

## Density components

Prepare the sampler target and its named components separately. Each query selects a boundary from the generated graph. There is no independently maintained implementation of the prior or likelihood for these queries.

```@example schools
target = prepare_query(built, bound, :sampler)
prior = prepare_query(built, bound, :prior)
likelihood = prepare_query(built, bound, :likelihood)
jacobian = prepare_query(built, bound, :log_jacobian)
components = (
    prior=prior(u),
    likelihood=likelihood(u),
    log_jacobian=jacobian(u),
    target=target(u),
)
@assert components.target ≈
    components.prior + components.likelihood + components.log_jacobian
components
```

The log Jacobian belongs to the density in unconstrained coordinates. It is not another observation term. Comparing a constrained-space reference with the sampler target therefore requires adding that transformation term. Omitting it would test a different target even if the prior and likelihood themselves agreed.

The pointwise query retains the likelihood contribution for each observed school. Summing these values recovers the likelihood query. Because the observations are conditionally independent given the school effects, each entry corresponds to one observed effect and its reported uncertainty.

```@example schools
pointwise = prepare_query(built, bound, :pointwise)
contributions = pointwise(u).y
@assert sum(contributions) ≈ components.likelihood
(; contributions, sum=sum(contributions), likelihood=components.likelihood)
```

The figure displays these computed contributions. A lower value means that the observed effect receives a lower log density at this parameter point, taking its standard error into account. It is not a ranking of schools by treatment effectiveness, and it is not a posterior predictive check. The differences combine the standardized residual with the normalizing term associated with each uncertainty.

```@example schools
plot = data((; school=string.(eachindex(y)), contribution=contributions)) *
    mapping(:school => "School (index)",
            :contribution => "Log likelihood contribution (natural-log units)") *
    visual(BarPlot; color="#2a78d6")
Tutorials.figure(plot, "school-contributions",
    "Likelihood contributions at the supplied parameter point; each bar is one observed school, including its reported standard error.")
```

## Diagnostics

Use Distributions to evaluate the constrained-space prior and likelihood independently of the generated graph. The reference includes the population priors and every conditional school-effect density. The likelihood uses the supplied school effects and observation standard errors.

```@example schools
reference_prior =
    logpdf(Normal(prior_location, prior_scale), point.mu) +
    logpdf(truncated(Cauchy(prior_location, prior_scale), 0, Inf), point.tau) +
    sum(logpdf.(Normal.(point.mu, point.tau), point.theta))
reference_likelihood = sum(logpdf.(Normal.(point.theta, sigma), y))
prior_error = abs(components.prior - reference_prior)
likelihood_error = abs(components.likelihood - reference_likelihood)
@assert components.prior ≈ reference_prior
@assert components.likelihood ≈ reference_likelihood
(; prior_error, likelihood_error)
```

A coordinate round trip checks a different boundary. It verifies that converting the structured parameter point to unconstrained coordinates and back recovers the values used by the reference calculation. This check would catch a mismatched layout even when a density happened to remain finite.

```@example schools
recovered = constrain(built.layout, u)
roundtrip_error = maximum((
    abs(recovered.mu - point.mu),
    abs(recovered.tau - point.tau),
    maximum(abs, recovered.theta - point.theta),
))
@assert recovered.mu ≈ point.mu
@assert recovered.tau ≈ point.tau
@assert recovered.theta ≈ point.theta
(; recovered, roundtrip_error)
```

## Another parameter point

The prepared queries accept another vector with the same coordinate structure. Changing the population location changes its prior and the conditional prior for the school effects. Holding the school effects fixed leaves the observation likelihood unchanged. This distinction is visible in the returned values below.

```@example schools
other_point = merge(point, (; mu=8.0))
other_u = unconstrain(built.layout, other_point)
changed = (
    prior_before=prior(u), prior_after=prior(other_u),
    likelihood_before=likelihood(u), likelihood_after=likelihood(other_u),
)
@assert changed.likelihood_before == changed.likelihood_after
@assert changed.prior_before != changed.prior_after
changed
```

The [probabilistic-graph guide](rkppl.md) explains binding and query preparation, and the [differentiation guide](automatic-differentiation.md) introduces the active-input boundary. A sampling application would differentiate the complete unconstrained target and supply its own sampling algorithm and convergence checks. The evaluations here establish the target's components; they do not establish sampler performance or posterior accuracy.

```@example schools
Tutorials.receipt("eight-schools"; checks=(
    prior_error, likelihood_error, roundtrip_error,
    decomposition=components.target ≈
        components.prior + components.likelihood + components.log_jacobian,
))
```

## Interpretation of the boundary

The school effects connect the observation likelihood to the population parameters through their conditional prior. The graph therefore distinguishes two kinds of changes that can look similar in source code. Updating a school effect changes its observation contribution and its conditional prior. Updating only the population location changes the hierarchical prior while leaving the observation distribution at the supplied school effects unchanged. The preceding output checks the latter case directly.

Known standard errors are data in this model. They are not estimates produced by this program, and the model does not include a sampling distribution for those standard errors. A different scientific model could treat uncertainty estimates differently, but it would need a corresponding declaration and a different target. Similarly, a centered and a noncentered parameterization can describe the same constrained model while exposing different unconstrained coordinates and Jacobian terms.

The example uses a finite, interior parameter point so every displayed component can be compared numerically. This does not test all boundary behavior of the half-Cauchy transform. A production inference workflow should inspect its own initial points and diagnostics, especially when the population scale approaches its support boundary. Those checks belong to the sampling workflow; the present build checks the declared model and the consistency of its prepared queries.

```@raw html
<details><summary>Reproduce</summary>
```

Run from the repository root. All displayed components and the figure are generated during this build.

```sh
julia --project=docs docs/instantiate.jl
julia --project=docs docs/make.jl
```

Download the <a href="./receipts/eight-schools.toml" download>execution receipt</a>. Historical compiler and performance experiments are described in the [research README](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/docs-history/README.md).

```@raw html
</details>
```
