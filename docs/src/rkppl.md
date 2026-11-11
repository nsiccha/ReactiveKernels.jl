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

The left side of `|` binds inputs. The right side observes declared variables and retains their densities as likelihood terms. Supplying a declared variable directly as a model-call keyword pins it instead: that declaration and its density are removed. Choose the operation that expresses the intended probability model.

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

The [Eight-schools example](eight-schools.md) checks the complete density decomposition and its coordinate transformation.
