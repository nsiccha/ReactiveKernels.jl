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

## Exported bindings

The table is generated from the module loaded by this build. Source recipes
are Julia source strings used by the package's development examples; prefer
kernel objects and named density functions in application code. Links lead
to a definition when located, or to the module's source directory, at the
revision used for this build.

```@eval
Main.Tutorials.export_index(Main.ReactiveKernelsDistributionKernels.DistributionKernelSources)
```

## State-path draws

```@autodocs
Modules = [ReactiveKernelsDistributionKernels.DistributionKernelSources]
Order = [:function]
Public = true
Private = false
```
