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

A PPL model can emit its sampler boundary through `jax_source(built, bound)`. The generated module includes data preparation and coordinate metadata. A top-level plate's cell recipe that reads only the plate's data and that the JAX functions cannot compute runs in `prepare_data` as a loop over the cells, and the plate reads its values as inputs of its own, one per cell. A data-only definition that calls a program-level `@kernel` with positional arguments, as Julia runs it (`prepare(spec)(args...)`), is computed in `prepare_data` by a host function of that kernel's plan. Inspect its untranslated-data declarations before treating it as a standalone raw-data pipeline; a definition without a host translation remains a required prepared input. A cell's vector argument arrives padded to the longest row, and an array it is broadcast into is padded along the same axes: reductions, also over `dims`, and `vec` read only the row's entries. A response whose element type admits `missing` is passed to `prepare_data` as the caller has it, with `None` for a missing entry; the module computes its presence mask, so one module serves every absence pattern.

A dotted cell latent, `v[i] .~ D.(args…)` with a length per index that data decide, has its dimensions, packed offsets, block table and argument entries computed in `prepare_data` when its arguments read data, scalar parameters or a shared array; an argument that reads a sampled array at the index, or an array of rank 2 or more at each index, raises `JAXEmissionError` for now.

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

## Package translations

A package extension can translate its own callable to an existing Python operation. These hooks run during source emission; they do not change Julia execution or register a derivative rule. Python differentiates the emitted primal with ordinary `jax.grad`.

Use `ReactiveKernels._jax_elementwise(::typeof(f))` for scalar calls and `f.(xs)`. Return a Python callable name, a `name => source` pair defining that module-level helper, or `nothing` to leave translation to the other paths. This does **not** register an ordinary array call `f(A)`: Julia's `exp(A)` is a matrix exponential, while `exp.(A)` applies the scalar exponential to each entry. Treating them alike would change the computation.

For a positional whole-function call, including an array-valued helper, define
`_jax_function_call(ctx::_JaxContext, ::typeof(my_function), args::Vector{_PyVal})`.
Qualify each RK name with `ReactiveKernels`; the executable example below shows
the complete method.

The underscored names in this section are the package extension interface. A method must include the context and vector annotations shown above so it is more specific than RK's generic method. It is selected before recorded or provider source and lowered-code translation, but after an applicable scalar/elementwise mapping. It translates positional calls; it does not itself register keyword handling.

Each `args[i]` describes one positional argument, **not** an actual Julia or Python value. `code` is its Python expression and `T` its Julia type. Preserve expressions with parentheses when composing operators; a value may be an expression rather than a bare variable. Return a `_PyVal`, not a string or `nothing`. The generic hook translates source; an override needing that fallback calls `_jax_traceable_or_lowered(ctx, f, args)`.

`_jax_result(f, args, expression)` infers the result's Julia type from `f` and the argument types, so an inferred matrix stays a matrix and subsequent matrix multiplication keeps its meaning. The Python expression must implement that native method's result and shape. Shapes and numerical inputs remain runtime expressions; the hook receives types, not sample data. For a whole-array call that is also elementwise and preserves the arrays' axes, `_jax_elementwise_result(f, args, expression)` additionally carries the padded-axis metadata.

For translations that change shape or ragged structure, the complete result constructor is `_PyVal(code, T, static, extent, limit, padded)`. Short constructors default `static=false`, `extent=""`, `limit=""`, and `padded=0`.

| Field | Meaning |
| --- | --- |
| `static::Bool` | Known as a Python value while JAX traces; it can decide Python control flow. A JAX array is not static merely because its inputs are constants. |
| `extent::String` | Valid length of a padded row, or lengths of a ragged collection; empty for an unpadded value. |
| `limit::String` | Static upper-bound expression of a data-dependent integer, for loops and allocation; empty when absent. |
| `padded::Int` | Bitmask of padded Julia axes, bit `k-1` for axis `k`. A padded vector's single axis is implicit; matrices need their axes recorded. |

`_jax_result` carries an input extent to an array result but does not derive new lengths or padding axes. A whole-array operation must account for padding in its mathematics and return metadata for its actual result; blindly copying an input's metadata is not correct for an arbitrary reduction, reshape, or matrix function. The matrix-exponential example below takes an ordinary dense matrix, not a padded ragged row.

### Host preparation and helpers

The same hook is used for `prepare_data`, with `ctx.host !== nothing`. RK rewrites `jnp.*` to `np.*`, `jsp.*` to `ssp.*` (SciPy special functions), and `jax.scipy.linalg.*` to `scipy.linalg.*`. `_jax_result` also uses these host spellings when all operands are static, keeping shape or literal arithmetic usable in Python guards. A translation requiring different host mathematics can branch on `ctx.host` and emit a NumPy/SciPy expression explicitly. Leftover device-only constructs such as `jax.lax` or `.at[...]` cannot form host preparation; the recipe stays in the device function or is reported as untranslated, according to the preparation boundary.

For a Python helper definition, call `_jax_helper!(ctx, name, source)` and use its **returned name** in the expression. It emits the helper once and, in host mode, emits a rewritten NumPy twin whose name has an `_np` suffix. Names must be unique across the emitted module; repeated names identify the same helper. In device mode with all-static operands, use `_host_helper!(ctx, name, source)` when the helper has a host spelling: `_jax_result` cannot inspect a separately defined helper's body to discover JAX calls. `_jax_elementwise` pairs handle this selection automatically. A custom helper must not mutate its inputs; RK cannot make arbitrary Python text obey ownership rules or translate an independently written derivative.

### Executable scalar and matrix example

This file defines owned Julia helpers, their mappings, and the kernels that use them. The `triple` helper has a loop, so its mapping is what makes the call translatable. The matrix exponential demonstrates whole-array mathematics distinct from an elementwise mapping.

```@example jax
using Markdown # hide
translation_file = joinpath(pkgdir(ReactiveKernels), "examples", "jax_translations.jl") # hide
include(translation_file) # hide
Markdown.parse("```julia\n" * read(translation_file, String) * "\n```") # hide
```

Emit scalar and matrix device calls, and a data-only matrix exponential in host preparation:

```@example jax
ex = JAXTranslationExample
scalar_source = jax_source(ex.mapped; have=(:x,), want=:total, types=(x=Float64,))
matrix_source = jax_source(ex.mapped_product; have=(:A, :v), want=:total)
host_source = jax_source(ex.exponential; have=(:scale, :A), want=:total, data=(:A,))
print(host_source)
```

After `write("matrix_model.py", host_source)`, run `python examples/jax_translations.py matrix_model.py` with a Python that has JAX and SciPy. This consumer checks the result and gradient against SciPy and checks that preparation returns NumPy arrays:

```@example jax
python_file = joinpath(pkgdir(ReactiveKernels), "examples", "jax_translations.py") # hide
Markdown.parse("```python\n" * read(python_file, String) * "\n```") # hide
```

The repository's JAX tests execute these examples with `RK_JAX_PYTHON` enabled: direct and wrapped calls, matrix rank, changed sizes, host preparation, and ordinary gradients against native Julia. The documentation build executes the Julia examples; it displays the Python consumer's source.
