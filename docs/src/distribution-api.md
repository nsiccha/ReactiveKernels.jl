# Distribution kernels

`ReactiveKernelsDistributionKernels` exports the `DistributionKernelSources` module.
Its kernel objects expose named endpoints such as `logpdf`, `logcdf` and
`quantile`; the available endpoints depend on the distribution. These are
ReactiveKernels graphs, with explicit ports for observations and parameters.

## Kernel objects

Import the required object and call an endpoint inside an authored graph.
The prepared result below is checked against Distributions at the same inputs.

```@example distribution_reference
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using Distributions: Normal, logpdf

@kernel normal_density(x, location, scale) = begin
    density = normal(; location, scale).logpdf(x)
    return density
end
run_density = prepare(normal_density)
value = run_density(0.5, 0.0, 2.0)
reference = logpdf(Normal(0.0, 2.0), 0.5)
@assert value ≈ reference
(; value, reference)
```

The endpoint is part of the enclosing graph, so preparation can select its
inputs and outputs using the same boundary rules as an authored expression.
The [graph guide](graphs.md) explains that selection. The
[conditional-latent example](discrete-latents.md) uses a hidden Markov density
and its corresponding latent query.

## Law parameters

Each law's docstring, listed under "Law reference" below, opens with its call
signature, such as `gamma(shape, rate)`. A positional construction
binds the parameters in that order. A named construction binds any port the
law lists under "Named bindings", which selects another parameterization: here
the gamma law by its scale. Outside a kernel, a prepared endpoint takes the
parameters first and then the endpoint's own argument.

```@example distribution_reference
using ReactiveKernelsDistributionKernels.DistributionKernelSources: gamma
using Distributions: Gamma

@kernel gamma_by_rate(x, shape, rate) = gamma(shape, rate).logpdf(x)
@kernel gamma_by_scale(x, shape, θ) = gamma(; shape, scale = θ).logpdf(x)
by_rate = prepare(gamma_by_rate)(1.2, 2.5, 1.5)
by_scale = prepare(gamma_by_scale)(1.2, 2.5, 1 / 1.5)
outside = prepare(gamma.logpdf)(2.5, 1.5, 1.2)
reference = logpdf(Gamma(2.5, 1 / 1.5), 1.2)
@assert by_rate ≈ by_scale ≈ outside ≈ reference
(; by_rate, by_scale, outside, reference)
```

In the REPL, `?DistributionKernelSources.gamma` or
`@doc DistributionKernelSources.gamma` shows the docstring.
`Base.Docs.doc(gamma)` takes the value instead of the name, so it describes
the object's internal type.

## Exported bindings

The table is generated from the module loaded by this build. Source recipes
are Julia source strings used by the package's development examples; prefer
kernel objects and named density functions in application code. Links lead
to a definition when located, or to the module's source directory, at the
revision used for this build.

```@eval
Main.Tutorials.export_index(Main.ReactiveKernelsDistributionKernels.DistributionKernelSources)
```

## Law reference

```@autodocs
Modules = [ReactiveKernelsDistributionKernels.DistributionKernelSources]
Order = [:constant]
Public = true
Private = false
```

## State-path draws

```@autodocs
Modules = [ReactiveKernelsDistributionKernels.DistributionKernelSources]
Order = [:function]
Public = true
Private = false
```
