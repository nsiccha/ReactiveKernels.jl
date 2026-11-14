# Probabilistic graphs

Use ReactiveKernelsPPL to declare a model, bind its data and prepare a log-density query. The nested package supplies probabilistic authoring and coordinate transforms on top of the core graph compiler.

## Declaration and binding

A model is a Julia block of sampling statements and deterministic assignments. Inputs such as prior scales and observations are supplied when binding, so their values remain data.

```@example ppl
using ReactiveKernelsPPL

model = @rkppl begin
    a ~ Normal(prior_location, prior_scale)
    b ~ Normal(prior_location, prior_scale)
    sigma ~ Exponential(noise_scale)
    mu = a .+ b .* x
    y .~ Normal.(mu, sigma)
end

x = [-1.0, 0.0, 1.0]
y = [-0.8, 1.1, 2.9]
bound = model(; x, prior_location=0.0, prior_scale=2.0, noise_scale=1.0) | (; y)
built = build_kernel(bound)
u = unconstrain(built.layout, (; a=1.0, b=2.0, sigma=0.3))
density = prepare_query(built, bound, :sampler)
density(u)
```

The left side of `|` binds inputs. The right side observes declared variables and retains their densities as likelihood terms. An observed response whose element type admits `missing` is bound with a presence mask, however many entries are missing, so one program serves every absence pattern and the likelihood skips the absent entries. Supplying a declared variable directly as a model-call keyword pins it instead: that declaration and its density are removed. Choose the operation that expresses the intended probability model.

## Coordinates

The layout maps structured constrained parameters to a packed unconstrained vector. Use the layout conversion rather than constructing that vector from an assumed ordering.

```@example ppl
parameters = constrain(built.layout, u)
(; names=coordinate_names(built.layout), parameters)
```

The sampler target includes the prior, likelihood and support-transform log Jacobian. A likelihood query excludes the other terms, and a pointwise query retains observation contributions under their response names.

```@example ppl
likelihood = prepare_query(built, bound, :likelihood)
pointwise = prepare_query(built, bound, :pointwise)
contributions = pointwise(u).y
@assert sum(contributions) ≈ likelihood(u)
(; contributions, sum=sum(contributions), likelihood=likelihood(u))
```

## Inspection

A model view combines the generated program with graph inspection. It is derived from the built model and its binding, so it describes the program used by the queries above.

```@example ppl
using Main.Tutorials
Tutorials.html(model_view(built; bound))
```

`prepare_query` returns a raw prepared callable. Calls made from an already compiled Julia function may require a world-age barrier; `prepare_sampler` supplies the managed sampler boundary. See the [PPL reference](ppl-api.md) for derivative, predictive and conditional-latent queries.

## Integrating out a linear-Gaussian recurrence

A latent `@scan` whose steps are linear in the previous state with Normal innovations, observed through a Normal law whose location is affine in it, can be integrated out exactly. `marginalize(model, :x)` keeps the model source as written and lowers the recurrence and its observation into a Kalman filter. The sampler then has no coordinates for `x` or its innovations, and the likelihood is the exact marginal density of the observations given the remaining parameters.

```@example marginal
using ReactiveKernelsPPL, Random

model = @rkppl begin
    level ~ Normal(0, 2)
    sd ~ Exponential(1)
    length_scale ~ LogNormal(0, 1)
    noise ~ Exponential(0.5)
    @scan begin
        x[1] ~ Normal(0, 1)
        for t in 2:T
            e ~ Normal(0, 1)
            r = exp(-gap[t] / length_scale)
            x[t] = r * x[t - 1] + sqrt(1 - r * r) * e
        end
    end
    y .~ Normal.(level .+ sd .* x, noise)
end

gap = [1.0, 0.5, 2.0, 1.0, 0.7, 1.3]
y = [0.3, 0.5, -0.1, 0.0, 0.4, 0.2]
latent = build_kernel(model(; gap, T = length(y)) | (; y))
plan = marginalize(model(; gap, T = length(y)), :x) | (; y)
built = build_kernel(plan)
(; latent=coordinate_names(latent.layout), marginal=coordinate_names(built.layout))
```

Each step's coefficients may read parameters, data and the loop index, so irregular gaps and resets where a new series begins are covered. The pointwise query holds each observation's one-step-ahead predictive density. Draws of the integrated-out trajectory come afterward, one per parameter draw, by forward filtering and backward sampling:

```@example marginal
u = unconstrain(built.layout, (; level=0.2, sd=0.3, length_scale=1.5, noise=0.2))
latents = prepare_marginalized_latents(built, plan)
marginalized_latent_draws(latents, u, Xoshiro(1)).x
```

A `missing` entry of `y` is a step that only predicts: the filter carries the state through it without an update, and the likelihood has no term for it. The draws then condition on the present entries alone, and one program serves every pattern of missing entries:

```@example marginal
y_gaps = Union{Missing,Float64}[0.3, missing, -0.1, 0.0, missing, 0.2]
with_gaps = marginalize(model(; gap, T = length(y_gaps)), :x) | (; y = y_gaps)
built_gaps = build_kernel(with_gaps)
latents_gaps = prepare_marginalized_latents(built_gaps, with_gaps)
marginalized_latent_draws(latents_gaps, u, Xoshiro(1)).x
```

The observed vector may also be one the model computes, as long as it does not read `x`. Censored rows enter the filter this way (data augmentation). Each censored row's unknown value is a latent bounded by its own range (per-element bounds), and the measured rows join those values in time order. The scan's length is then supplied, and the measured rows are inputs:

```@example marginal
censored = @rkppl begin
    level ~ Normal(0, 2)
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    noise ~ Exponential(0.5)
    @scan begin
        x[1] ~ Normal(0, sigma / sqrt(1 - phi^2))
        for t in 2:T
            x[t] ~ Normal(phi * x[t - 1], sigma)
        end
    end
    gap[1:K] .~ restricted.(Flat.(), 0, Inf)    # a row below its limit: limit - gap
    value = vcat(measured, limit .- gap)[order]
    value .~ Normal.(level .+ x, noise)
end
measured = [0.3, 0.5, 0.0, 0.4]
limit = [-0.2, -0.3]                            # rows 3 and 6 fell below these
order = invperm([1, 2, 4, 5, 3, 6])
augmented = marginalize(censored(; T = 6, K = 2, measured, limit, order), :x)
coordinate_names(build_kernel(augmented).layout)
```

Rows may also observe the state at their steps, `x[step]` with `step` bound integer data, so that a step has several rows or none (several sensors, irregular sampling). The filter then runs over the steps and their rows in order, and the scan's length is supplied:

```@example marginal
sensors = @rkppl begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    bias[1:S] .~ Normal.(0, 1)
    noise ~ Exponential(0.5)
    @scan begin
        x[1] ~ Normal(0, sigma / sqrt(1 - phi^2))
        for t in 2:T
            x[t] ~ Normal(phi * x[t - 1], sigma)
        end
    end
    y .~ Normal.(bias[sensor] .+ x[step], noise)
end
step = [1, 1, 2, 4, 4, 5]                       # step 3 has no row
sensor = [1, 2, 1, 1, 2, 2]
readings = [0.2, 0.4, -0.1, 0.3, 0.1, 0.0]
rows = marginalize(sensors(; T = 5, S = 2, step, sensor), :x) | (; y = readings)
built_rows = build_kernel(rows)
u = zeros(built_rows.layout.total)
(; coordinates=coordinate_names(built_rows.layout),
    x=marginalized_latent_draws(prepare_marginalized_latents(built_rows, rows), u, Xoshiro(1)).x)
```

A recurrence in a `@plate` cell integrates out the same way, one filter per index, when the cell observes it through `y[g] .~ Normal.(offset .+ loading .* x, noise)` and the bound `y` holds one vector per index. The draws then hold one trajectory per index:

```@example marginal
grouped = @rkppl begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    noise ~ Exponential(0.5)
    level[1:G] .~ Normal.(0, 2)
    @plate for g in eachindex(y)
        @scan begin
            x[1] ~ Normal(0, 1)
            for t in 2:length(y[g])
                x[t] ~ Normal(phi * x[t - 1], sigma)
            end
        end
        y[g] .~ Normal.(level[g] .+ x, noise)
    end
end
ys = [[0.3, 0.5, -0.1], [0.0, 0.4], [0.2, 0.1, 0.6, 0.3]]
cells = marginalize(grouped(; G = length(ys)), :x) | (; y = ys)
built_cells = build_kernel(cells)
u = zeros(built_cells.layout.total)
latents = prepare_marginalized_latents(built_cells, cells)
(; coordinates=coordinate_names(built_cells.layout),
    x=marginalized_latent_draws(latents, u, Xoshiro(1)).x)
```

Recurrences carrying several arrays, a lag above one, recurrences inside nested `@plate` cells or submodels, missing entries or gathered rows in a `@plate` cell's series, and observations that are not Normal are refused with an error naming the statement.

## Adaptive partial centering

Select affine cells when setting up WarmupHMC. Declare their location and
log-scale as ordinary named values in the model, then pass those names to
`adaptive_centering_problem`. It is public but not exported, because
BayesianRegressionModels exports a different function of that name; import it
explicitly:

```@example adaptive_centering
using ReactiveKernelsPPL, WarmupHMC, LogDensityProblems, Enzyme
using ReactiveKernelsPPL: adaptive_centering_problem
using DifferentiationInterface: AutoEnzyme

model = @rkppl begin
    mu ~ Normal(0, mu_sd)
    sigma ~ Exponential(sigma_scale)
    z[1:N] .~ Normal.(0, 1)
    z_location = mu .+ offsets
    z_log_scale = log(sigma)
    total = z_location .+ exp(z_log_scale) .* z
    y .~ Normal.(total, se)
end
plan = model(; N = 4, offsets = [-0.2, 0.1, 0.3, -0.1],
    mu_sd = 2.0, sigma_scale = 1.0, se = 0.3) | (; y = [0.1, 0.2, 0.7, -0.2])
built = build_kernel(plan)
u0 = zeros(built.layout.total)
problem = prepare_sampler(built, plan, u0; backend = AutoEnzyme())
adaptive = adaptive_centering_problem(built, plan, problem, AutoEnzyme();
    cells = (z = (location = :z_location, log_scale = :z_log_scale),), u0)
@assert LogDensityProblems.logdensity(adaptive, u0) ==
    LogDensityProblems.logdensity(problem, u0)
(; dimension = LogDensityProblems.dimension(adaptive),
    logdensity = LogDensityProblems.logdensity(adaptive, u0))
```

Pass `adaptive` to WarmupHMC with nonlinear adaptation enabled. The selected
latent starts in its authored non-centered coordinates; the sampler can adapt
its centering and returns draws in the original model frame. A cell can set
`target = 1.0` for an authored centered coordinate and `source` for a different
initial sampler frame. Zero means non-centered; one means centered.

Each location or log-scale accepts either a `Real` constant shared by the
block or a symbol naming a scalar or latent-shaped array in the model graph.
For a zero-centered block, use
`cells = (z = (location = 0.0, log_scale = :z_log_scale),)`;
the zero needs no declaration in the model. Graph values and their derivatives
may depend on unselected latent parameters. The adapter selects identity-transformed real
coordinates, without changing their priors. For example, a real ICAR block
keeps its ICAR density and any soft sum constraint.

A cell's location and log-scale must not read its own latent. They may read
another selected cell's latent, as in a nested hierarchy where unit effects
`z_u` are centred about `unit_location = mu .+ tau_g .* z_g[group]` and the
group effects `z_g` about `mu`:
`cells = (z_g = (location = :mu, log_scale = :log_tau_g), z_u = (location = :unit_location, log_scale = :log_tau_u))`.
Such a read sees the other cell's model-frame values. The reads order the
cells into levels, which WarmupHMC maps in order, so re-centering a parent level
keeps the meaning of the coordinates centred on it. The adapter checks these
graph dependencies before sampling and refuses own reads and cycles. Nested
levels need RKPPL 0.1.185 and WarmupHMC 0.3.0 or later.

A latent random walk or AR(1) series observed only in parts has no good
fixed frame: written in its innovations, the step size falls as the series
grows; written in its states, it mixes badly where nothing is observed.
Mark its `@scan` with `markov_chains(model, :w)` before binding. The
recurrence qualifies when it is linear-Gaussian in its previous state, as
for `marginalize`, written centred or non-centred, at top level or in a
`@plate` cell. RKPPL lowers it in its states, and `adaptive_centering_problem`
gives it one WarmupHMC `MarkovChain` (WarmupHMC 0.3.1 or later), which adapts
how far to centre each time step. `cells` may then be omitted.

RKPPL 0.1.175 with WarmupHMC
0.2.21 or later prepares one joint endpoint kernel, sharing common graph work
across selected cells and their location/log-scale roles. It runs once per
vector and level on a map or inverse, and shares results across scoring
candidates. Within a level, cells are ordered by packed coordinate, regardless
of keyword order. Identity
maps skip endpoint evaluation. Ordinary AD differentiates the joint graph;
arrays remain read-only, without concatenation into a new endpoint buffer.
For Enzyme differentiation through cached matrix endpoints, use the
`const_array_reshape` repair (dogfood `9c4df614` or later); the acceptance
suite checks ordinary reverse differentiation and unchanged cached inputs.

The [Eight-schools example](eight-schools.md) checks the complete density decomposition and its coordinate transformation.
