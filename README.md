# ReactiveKernels.jl

ReactiveKernels prepares reusable Julia computations from the values you have to the values you want.

Install from a source checkout:

```sh
git clone https://github.com/nsiccha/ReactiveKernels.jl.git
cd ReactiveKernels.jl && julia --project=. deps/source_dependencies.jl
```

This example runs on the documentation landing page during every docs build:

```julia
using ReactiveKernels

@kernel affine(x, scale, offset) = begin
    value = scale * x + offset
end
f = prepare(affine)
f(2.0, 3.0, 1.0)
```

The prepared function returns the affine transformation's value. See the
[executed example and its output](https://nsiccha.github.io/ReactiveKernels.jl/dev/).

The documentation has four entry points:

- [Start](https://nsiccha.github.io/ReactiveKernels.jl/dev/): installation and a complete evaluation.
- [Guides](https://nsiccha.github.io/ReactiveKernels.jl/dev/graphs): graph boundaries, binding, batching and differentiation.
- [Examples](https://nsiccha.github.io/ReactiveKernels.jl/dev/bound-regression): evaluated graphs with visible results and diagnostics.
- [Reference](https://nsiccha.github.io/ReactiveKernels.jl/dev/api): exported interfaces and their contracts.

The nested [ReactiveKernelsPPL package](packages/ReactiveKernelsPPL) adds
probabilistic declarations, coordinate transformations and log-density queries.
Core ReactiveKernels does not select an inference algorithm.

Contributors must read the [core constraints](CONTRIBUTING.md) before
changing a lowering, extension or example. See [docs/README.md](docs/README.md)
for the documentation build and [benchmark/README.md](benchmark/README.md) for
research and performance experiments.
