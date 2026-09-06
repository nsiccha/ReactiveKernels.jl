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

`ad_value_and_gradient(prepared, args...)` can also appear inside an enclosing
RK kernel or a function compiled by Reactant. With a traced scalar or array as
the active input, it stages the value and derivative into that enclosing
program. This lets a generated integrator use the model gradient without a
separate compiled-gradient call. The compiler selects the tensorized primal
body and the caller's DI backend; the native DI preparation remains available
for native calls. Bound data and the authored argument order are preserved.
Focused executable examples are in
[`test/test_ad_fused_reactant.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/test/test_ad_fused_reactant.jl).

## Boundary

- A compiled gradient is available only where the primal kernel itself compiles
  through Reactant. The primal `@compile` error propagates unchanged.
- Host active arguments and incorrect HAVE arity raise clear errors rather than
  silently changing the differentiated boundary.
- `compile_ad_value_and_gradient` returns `(value, gradient)` in one compiled
  call; `compile_ad_gradient` returns only the gradient.
- The packed Eight Schools model is covered by the
  [model-level Reactant AD measurements](eight-schools-reactant.md).

For native preparation and ownership rules, start with
[Prepared gradients](automatic-differentiation.md). For reviewed model-level
evidence, see [PPL automatic differentiation](ppl-ad.md).
