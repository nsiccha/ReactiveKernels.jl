# JAX source

Use `jax_source` to emit a Python module for a supported graph boundary. Julia performs source generation; the resulting module uses JAX at runtime.

## Emitted function

This build executes the emitter and prints its complete result. The documentation environment does not include Python JAX, so this page does not claim to execute the emitted Python.

```@example jax
using ReactiveKernels

@kernel quadratic(x::Float64, scale::Float64) = begin
    value::Float64 = scale * x^2
    return value
end

source = jax_source(quadratic; have=(:x, :scale), want=:value)
print(source)
```

Input annotations tell the emitter each port's rank. For untyped ports, supply the corresponding `types` keyword. The emitted input order follows the selected input boundary.

## Julia evaluation

The Julia callable evaluates the same authored mathematical expression. Its output establishes the reference value for a consumer testing the emitted module.

```@example jax
prepared = prepare(quadratic)
prepared(3.0, 2.0)
```

A PPL model can emit its sampler boundary through `jax_source(built, bound)`. The generated module includes data preparation and coordinate metadata. A top-level plate's cell recipe that reads only the plate's data and that the JAX functions cannot compute runs in `prepare_data` as a loop over the cells, and the plate reads its values as inputs of its own, one per cell. Inspect its untranslated-data declarations before treating it as a standalone raw-data pipeline; a definition without a host translation remains a required prepared input.

Only supported operations can be emitted. Translation errors identify an unsupported operation instead of substituting a different computation. JAX compilation, device placement and differentiation belong to the Python consumer.

The [Eight-schools example](eight-schools.md) explains the model and coordinate boundary that a PPL export must preserve.
