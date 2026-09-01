# Automatic differentiation through Reactant

When the optional Reactant weak dependency is loaded, the same prepared
differentiation boundary can be compiled through Reactant/XLA.
`compile_ad_gradient` and `compile_ad_value_and_gradient` take a
[`PreparedADKernel`](@ref), reusing the native scalar-WANT, single-active-port,
and authored-HAVE-order validation.

This example is deliberately not executed during the docs build; the executable
authority is
[`test/test_ad_reactant.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/test/test_ad_reactant.jl).

```julia
using ReactiveKernels
using Reactant
import Enzyme
using DifferentiationInterface: AutoEnzyme

backend = AutoEnzyme(; mode = Enzyme.Reverse)
prepared = prepare_ad(
    objective, backend, parameters;
    data, active = :q, want = :density,
)

traced = map(Reactant.to_rarray, (parameters, data))
compiled_gradient = compile_ad_gradient(prepared, traced...)
gradient = compiled_gradient(traced...)

compiled_both = compile_ad_value_and_gradient(prepared, traced...)
value, gradient = compiled_both(traced...)
```

## Boundary

- A compiled gradient is available only where the primal kernel itself compiles
  through Reactant. The primal `@compile` error propagates unchanged.
- Host active arguments and incorrect HAVE arity raise clear errors rather than
  silently changing the differentiated boundary.
- `compile_ad_value_and_gradient` returns `(value, gradient)` in one compiled
  call; `compile_ad_gradient` returns only the gradient.
- Packed Eight Schools joint/prior paths remain outside the accepted boundary
  where their primal form hits Reactant's scalar-indexing rejection.

For native preparation and ownership rules, start with
[Prepared gradients](automatic-differentiation.md). For reviewed model-level
evidence, see [PPL automatic differentiation](ppl-ad.md).
