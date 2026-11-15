# JAX translations

This guide continues [JAX source](jax.md): the Julia functions that host preparation translates, and how a package maps its own callables.

## Host preparation

Host preparation translates `findfirst(predicate, vector)`, including captured predicates such as `isequal(label)`, `Base.Fix1` and `Base.Fix2`. It searches in order, stops at the first match, and returns a one-based index or `nothing`. A positional kernel can therefore compute group indices with `plate(values) do value; Int(findfirst(isequal(value), levels)); end` inside the emitted module. Label values and the lengths of both vectors remain runtime inputs.

Host preparation also orders vectors of numbers or strings as Julia's `isless` does (-0.0 before 0.0, every NaN last): `sort`, `unique` (and `sort!`, `unique!` of a list built with `push!`), `searchsortedfirst` and `searchsortedlast` (one-based), and `sortperm(v)`, whose one-based positions keep equal values in ascending position. `invperm(p)` of a vector of integers is the inverse permutation and raises `ArgumentError: argument is not a permutation` as Julia does. A data-only definition such as `pos = invperm(vcat(rows_a, rows_b))` therefore runs in `prepare_data`. Keyword orders (`rev`, `by`, `lt`) have no translation. In the JAX functions these calls have none either: a recipe that also reads a parameter (`theta[sortperm(rows)]`) raises `JAXEmissionError` naming the function, so give the permutation a definition of its own that reads data alone.

`repeat(A, counts...)` tiles `A` `counts[d]` times along axis `d`, as Julia's `repeat(A; outer = counts)` does, and `repeat(A; inner, outer)` first repeats each entry `inner[d]` times along axis `d`; each keyword takes an integer or a tuple of them. Counts past `A`'s rank add trailing axes, as Julia's do (`repeat(v, 1, k)` of a vector is a matrix with `k` columns). Shapes are static while JAX traces, so the JAX functions need static counts (literals, or integers static to `jax.jit`); a count of data is computed in `prepare_data` when the recipe reads data alone. A categorical model's prior scale per coefficient term, repeated once per non-reference category (`repeat([intercept_scale, slope_scale], 1, k)`), therefore runs in `prepare_data`, and one module serves every category count. Julia's argument checks hold: a negative count, or with `inner` a count list shorter than `A`'s rank, raises with Julia's `ArgumentError` message. A recipe that repeats by a count the functions trace raises `JAXEmissionError` naming the recipe; a padded (ragged) value has no `repeat` translation.

A `Bool` array index (a `BitVector` or `Vector{Bool}`, such as `table[table[:, 2] .!= 1, 4]` or `values[present]`) selects the entries where it is true, as Julia's logical indexing does; it is not shifted as one-based positions. A mask of the array's whole shape (`A[A .> 0]`) selects in column-major order. `prepare_data` and other host code use a NumPy boolean mask, which fails on a mask of another length as Julia's `BoundsError` does; a list built with `push!` keeps the entries whose mask entry is true. In device code the count of selected entries depends on data, so the gather is padded to the mask's length: the selected entries come first, padding repeats the last of them, and reductions, `length` and broadcasts mask the padding as they do for a ragged row. A write through a mask writes only the selected entries. A device mask beside another kept dimension (`A[mask, :]`) and a mask of the groups of a ragged value raise `JAXEmissionError`, as does a single `Bool` index into an array, which Julia refuses.

## Statistics reductions

With Statistics loaded (`ReactiveKernelsStatisticsExt`, RK 0.1.260), `mean(x)`, `mean(f, x)`, `var(x)` and `std(x)` translate, with the keywords `dims` (one integer known while JAX traces, or `:`), and for `var` and `std` also `corrected` (a literal `true` or `false`) and `mean`. Each is the core `sum`, `length` and `size` translation it composes, `sum(x) / length(x)` and `sum(abs2.(x .- m)) / (n - corrected)`, so a padded row or a `dims` reduction reads only its valid entries and `prepare_data` takes NumPy's spelling. Over `dims` the reduced axis keeps length one, as in Julia. An empty argument, or one entry with `corrected`, gives Julia's NaN. Before, Statistics' own methods, which branch, were refused as "the function `_mean` has loops or branches".

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
| `padded::Int` | Bitmask of padded axes (bit `k-1` for axis `k`), implicit for a vector; for a tuple, of its fields with a length. |

`_jax_result` carries an input extent to an array result without deriving new lengths or padding axes. Operations must account for padding and return their actual result metadata: copying input metadata cannot describe arbitrary reductions, reshapes or matrix functions. The matrix-exponential example takes an ordinary dense matrix, not a padded ragged row.

### Host preparation and helpers

The hook also serves `prepare_data`, where `ctx.host !== nothing`. RK rewrites `jnp.*` to `np.*`, `jsp.*` to `ssp.*` (SciPy special functions), and `jax.scipy.linalg.*` to `scipy.linalg.*`. `_jax_result` uses these host spellings for all-static operands too, keeping shape/literal arithmetic usable in Python guards. Branch on `ctx.host` if different host mathematics is needed. Leftover device-only constructs (`jax.lax`, `.at[...]`) cannot form host preparation: the recipe stays on device or is reported untranslated, according to the preparation boundary.

Define Python helpers with `_jax_helper!(ctx, name, source)` and use its **returned name**. It emits each name once; host mode emits a NumPy twin with an `_np` suffix. Names must be unique across the module. For all-static operands in device mode, use `_host_helper!(ctx, name, source)` when a host spelling exists: `_jax_result` cannot discover JAX calls inside separate helper bodies. `_jax_elementwise` pairs select automatically. Custom helpers must not mutate inputs; RK cannot enforce ownership on arbitrary Python text or translate independently written derivatives.

### Executable scalar and matrix example

This file defines owned helpers, mappings and kernels. The loop in `triple` requires its mapping for translation; matrix exponential demonstrates whole-array mathematics distinct from elementwise mapping.

```@example jax
using ReactiveKernels, Markdown # hide
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
