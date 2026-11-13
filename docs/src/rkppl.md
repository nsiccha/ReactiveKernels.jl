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

Every selected affine quantity must remain invariant under the selected maps.
A location that reads another selected latent needs a coupled transform;
WarmupHMC's independent indexed maps do not supply that transport. The adapter
checks this graph dependency before sampling. RKPPL 0.1.175 with WarmupHMC
0.2.21 or later prepares one joint endpoint kernel, sharing common graph work
across selected cells and their location/log-scale roles. It runs once per
vector on a map or inverse, and shares results across scoring candidates.
Cells are ordered by packed coordinate, regardless of keyword order. Identity
maps skip endpoint evaluation. Ordinary AD differentiates the joint graph;
arrays remain read-only, without concatenation into a new endpoint buffer.

The [Eight-schools example](eight-schools.md) checks the complete density decomposition and its coordinate transformation.
