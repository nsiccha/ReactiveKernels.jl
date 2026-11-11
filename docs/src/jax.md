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

A PPL model can emit its sampler boundary through `jax_source(built, bound)`. The generated module includes data preparation and coordinate metadata. A top-level plate's cell recipe that reads only the plate's data and that the JAX functions cannot compute runs in `prepare_data` as a loop over the cells, and the plate reads its values as inputs of its own, one per cell. Inspect its untranslated-data declarations before treating it as a standalone raw-data pipeline; a definition without a host translation remains a required prepared input. A cell's vector argument arrives padded to the longest row, and an array it is broadcast into is padded along the same axes: reductions, also over `dims`, and `vec` read only the row's entries. A response whose element type admits `missing` is passed to `prepare_data` as the caller has it, with `None` for a missing entry; the module computes its presence mask, so one module serves every absence pattern.

Only supported operations can be emitted. Translation errors identify an unsupported operation instead of substituting a different computation. JAX compilation, device placement and differentiation belong to the Python consumer.

Unless `simplify = false`, each emitted recipe computes numerically fragile compositions in the stable form native preparation computes, together with the statements that compute its inputs, also from a lazy arm or a plate's cell: `log(exp(u))` is emitted as `u`, and `log1p(-p)` of `p = logistic(u)` reads `u`, so the value and `jax.grad` stay finite where `exp(u)` overflows. A law's term on a value computed outside the law (`Normal(0, s)`'s `log(s)` of `s = exp(a)`) is still translated as written: the law's endpoint is translated from its compiled lowered code, which the rewrites do not read yet.

The [Eight-schools example](eight-schools.md) explains the model and coordinate boundary that a PPL export must preserve.

## Helpers with loops and branches

A helper that a recipe calls is translated from its source when the emitter has it. An `@traceable` helper records its own source: a `for` loop becomes `jax.lax.fori_loop`, an `if` becomes `jax.lax.cond`, and `A[i] = v` becomes `A.at[i].set(v)`. A plain helper is translated from its lowered code, which must be straight-line.

A method that a package generates, such as one its macro writes into the caller's module, cannot carry `@traceable`. The generating package keeps the method's source as a `TraceableSource` (argument names, body, and the module whose globals the body reads) and hands it over through a provider. It registers the provider with `register_source_provider!`, once, from the `__init__` of the package extension that loads with ReactiveKernels. The emitter asks the provider for a call that has no other translation and no `@traceable` source, with the function and the `Tuple` type of the argument types. The provider answers `nothing` or the source of the method those arguments call natively.

The example below stands in for such a package. Its macro defines the method and keeps its source.

```@example jax
const GENERATED = Dict{Any,TraceableSource}()

macro generate(def)
    call, body = def.args
    names = Tuple(arg.args[1] for arg in call.args[2:end])
    source = TraceableSource(names, body, __module__)
    :($(esc(def)); GENERATED[$(esc(call.args[1]))] = $source)
end

@generate function decay_sum(x::Vector{Float64}, rate::Float64)
    total = zero(rate)
    for i in 1:3
        total = total + x[i] * exp(-i * rate)
    end
    if rate > 0
        total = total / rate
    end
    return total
end

provide_source(f, argument_types) = get(GENERATED, f, nothing)
register_source_provider!(provide_source)

@kernel decayed(x::Vector{Float64}, rate::Float64) = begin
    value = decay_sum(x, rate)
    return value
end

print(jax_source(decayed; have = (:x, :rate), want = :value))
```

The body follows the rules of an `@traceable` body: a `return` only at its end, and in the JAX functions no `break` or `continue` and a `for` over one range. `prepare_data` runs the body as host Python, which takes both. A provider's source is read only by `jax_source`; native execution runs the method itself.
