# JAX source

Use `jax_source` to emit a Python module for a supported graph boundary. Julia performs source generation; the resulting module uses JAX at runtime.

## Emitted function

This build executes the emitter and prints its complete result. The documentation environment has no Python JAX, so the emitted Python is not executed here.

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

Typed `Float32` allocations and conversions keep `float32`, including arrays
computed by host `prepare_data`; `Float64` keeps `float64`. Prepared input
metadata reports the same dtype.

`fma` of `Float64` or `Float32` values, also broadcast, rounds once, as in
Julia; `jax.grad` gives `(y, x, 1)`. Integer `fma` and `muladd` are
`x * y + z`. Subnormals follow the backend.

## Julia evaluation

The Julia callable evaluates the same authored mathematical expression. Its output establishes the reference value for a consumer testing the emitted module.

```@example jax
prepared = prepare(quadratic)
prepared(3.0, 2.0)
```

A PPL model can emit its sampler boundary through `jax_source(built, bound)`. The generated module includes data preparation and coordinate metadata. A top-level plate's cell recipe that reads only the plate's data and that the JAX functions cannot compute runs in `prepare_data` as a loop over the cells, and the plate reads its values as inputs of its own, one per cell. A data-only definition that calls a program-level `@kernel` with positional arguments, as Julia runs it (`prepare(spec)(args...)`), is computed in `prepare_data` by a host function of that kernel's plan. Inspect its untranslated-data declarations before treating it as a standalone raw-data pipeline; a definition without a host translation remains a required prepared input. A cell's vector argument arrives padded to the longest row, and an array it is broadcast into is padded along the same axes: reductions, also over `dims`, and `vec` read only the row's entries. `reduce(hcat, cells)` of vectors or matrices keeps Julia's columns; like `stack`, it raises at run time unless every column fills the padded width (Julia's `DimensionMismatch`). A response whose element type admits `missing` is passed to `prepare_data` as the caller has it, with `None` for a missing entry; the module computes its presence mask, so one module serves every absence pattern.

A dotted cell latent, `v[i] .~ D.(args…)` with a length per index that data decide, has its dimensions, packed offsets, block table and argument entries computed in `prepare_data` when its arguments read data, scalar parameters or a shared array; an argument that reads a sampled array at the index, or an array of rank 2 or more at each index, raises `JAXEmissionError` for now.

Translation errors identify an unsupported operation instead of substituting a different computation. JAX compilation, device placement and differentiation belong to the Python consumer.

Unless `simplify = false`, each emitted recipe computes numerically fragile compositions in the stable form native preparation computes, together with the statements that compute its inputs, also from a lazy arm or a plate's cell: `log(exp(u))` is emitted as `u`, and `log1p(-p)` of `p = logistic(u)` reads `u`, so the value and `jax.grad` stay finite where `exp(u)` overflows. A law's term on a value computed outside the law (`Normal(0, s)`'s `log(s)` of `s = exp(a)`) is still translated as written: the law's endpoint is translated from its compiled lowered code, which the rewrites do not read yet.

The [Eight-schools example](eight-schools.md) explains the model and coordinate boundary that a PPL export must preserve.

## Helpers with loops and branches

A helper translates from recorded source when available: `@traceable` records it; `for` becomes `jax.lax.fori_loop`, `if` becomes `jax.lax.cond`, and `A[i] = v` becomes `A.at[i].set(v)`. Plain lowered helpers support straight-line code and terminal forward branches, including size guards. They can create immutable closures and fold fixed argument lists with `Base.afoldl`. Broadcasting `Ref(x)` passes `x` whole to each call. Captures remain runtime inputs; host `prepare_data` uses the same translation.

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

A package extension can map its own callable during source emission. Julia execution is unchanged; ordinary `jax.grad` differentiates the emitted primal without a registered derivative rule.

Use `ReactiveKernels._jax_elementwise(::typeof(f))` for scalar calls and `f.(xs)`. Return a Python callable name, `name => source` defining a module-level helper, or `nothing` for fallback translation. This does **not** register ordinary array calls: Julia's `exp(A)` is a matrix exponential; `exp.(A)` applies scalar exponentials.

For a positional whole-function call, including an array-valued helper, define
`_jax_function_call(ctx::_JaxContext, ::typeof(my_function), args::Vector{_PyVal})`.
Qualify RK names with `ReactiveKernels`, as in the executable example below.

These underscored names form the package extension interface. The context and vector annotations make the method more specific than RK's generic method. Selection precedes recorded/provider source and lowered-code translation, but follows applicable scalar/elementwise mappings. This hook registers positional calls, not keyword handling.

Each `args[i]` describes an argument: `code` is its Python expression, `T` its Julia type. Parenthesize expressions when composing operators. Return a `_PyVal`, not a string or `nothing`; call `_jax_traceable_or_lowered(ctx, f, args)` for the generic hook's source fallback.

`_jax_result(f, args, expression)` infers the native result type from `f` and the argument types, preserving matrix rank for subsequent multiplication. The expression must implement that native result and shape. Shapes and numerical inputs remain runtime expressions, not sampled data. For elementwise whole-array calls preserving the arrays' axes, `_jax_elementwise_result(f, args, expression)` additionally carries padded-axis metadata.

For changed shape or ragged structure, the full constructor is `_PyVal(code, T, static, extent, limit, padded)`. Short constructors default `static=false`, `extent=""`, `limit=""`, and `padded=0`.

| Field | Meaning |
| --- | --- |
| `static::Bool` | Known as a Python value while JAX traces; it can decide Python control flow. A JAX array is not static merely because its inputs are constants. |
| `extent::String` | Valid length of a padded row, or lengths of a ragged collection; empty for an unpadded value. |
| `limit::String` | Static upper-bound expression of a data-dependent integer, for loops and allocation; empty when absent. |
| `padded::Int` | Bitmask of padded Julia axes, bit `k-1` for axis `k`. A padded vector's single axis is implicit; matrices need their axes recorded. |

`_jax_result` carries an input extent to an array result without deriving new lengths or padding axes. Operations must account for padding and return their actual result metadata: copying input metadata cannot describe arbitrary reductions, reshapes or matrix functions. The matrix-exponential example takes an ordinary dense matrix, not a padded ragged row.

### Host preparation and helpers

The hook also serves `prepare_data`, where `ctx.host !== nothing`. RK rewrites `jnp.*` to `np.*`, `jsp.*` to `ssp.*` (SciPy special functions), and `jax.scipy.linalg.*` to `scipy.linalg.*`. `_jax_result` uses these host spellings for all-static operands too, keeping shape/literal arithmetic usable in Python guards. Branch on `ctx.host` if different host mathematics is needed. Leftover device-only constructs (`jax.lax`, `.at[...]`) cannot form host preparation: the recipe stays on device or is reported untranslated, according to the preparation boundary.

Define Python helpers with `_jax_helper!(ctx, name, source)` and use its **returned name**. It emits each name once; host mode emits a NumPy twin with an `_np` suffix. Names must be unique across the module. For all-static operands in device mode, use `_host_helper!(ctx, name, source)` when a host spelling exists: `_jax_result` cannot discover JAX calls inside separate helper bodies. `_jax_elementwise` pairs select automatically. Custom helpers must not mutate inputs; RK cannot enforce ownership on arbitrary Python text or translate independently written derivatives.

### Executable scalar and matrix example

This file defines owned helpers, mappings and kernels. The loop in `triple` requires its mapping for translation; matrix exponential demonstrates whole-array mathematics distinct from elementwise mapping.

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

After `write("matrix_model.py", host_source)`, run `python examples/jax_translations.py matrix_model.py` with a Python that has JAX and SciPy. This consumer checks the result and gradient against SciPy and that preparation returns NumPy arrays:

```@example jax
python_file = joinpath(pkgdir(ReactiveKernels), "examples", "jax_translations.py") # hide
Markdown.parse("```python\n" * read(python_file, String) * "\n```") # hide
```

The repository's JAX tests execute these examples with `RK_JAX_PYTHON` enabled: direct and wrapped calls, matrix rank, changed sizes, host preparation, and ordinary gradients against native Julia. The documentation build executes the Julia examples; it displays the Python consumer's source.
