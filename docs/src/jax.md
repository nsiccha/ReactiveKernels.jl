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

The module docstring starts with the generated line shown above. The `doc` keyword adds the author's description of the program after it: a string, or a binding's docstring as `@doc(NAME)`. A docstring written above `const NAME = …` belongs to the binding, not to the value `jax_source` receives, so the caller passes it. For a PPL model, `jax_source(built, bound; doc)` places it before the generated `Data:` and `Model documentation:` lists. Docstrings keep their text as written, backslashes and quotes included.

Typed `Float32` values keep `float32`, also in `prepare_data`; other floats
take `jax_source`'s `float_type` (`Float64`, or `Float32` without JAX's 64-bit
mode), as metadata reports.

`fma` of `Float64` or `Float32` values, also broadcast, rounds once, as in
Julia; `jax.grad` gives `(y, x, 1)`. Integer `fma` and `muladd` are
`x * y + z`. Subnormals follow the backend.

For `qr(A)`, `rk_cholesky_lower(A)` and `L \ b`, see
[factorizations and triangular solves](jax-translations.md#Factorizations-and-triangular-solves).
`diagm(v)` also constructs a diagonal matrix, and source loops or
comprehensions over `axes(A, d)` retain iteration along that dimension.
Prepared matrix cells retain their rank and row lengths, including in empty
plates. Matrix cells can vary along their first axis; trailing dimensions
must agree because the representation carries one extent per cell.
For raw empty matrix lists, call `pad_ragged([], rank=2)`.

For a ragged row index `r`, `M[r, :]` and `view(M, r, :)` keep every column.
`sum(M[r, :] * b)` counts each selected row, including repeats. In
`M[r, :]' * w` the contraction uses only the valid rows. A ragged column
index beside a kept axis and two array indices including a ragged one
remain unsupported.

Plate cells can return numerical tuples or named records, including nested
records, numeric arrays and [ragged row groups](jax-host.md#Groups-of-ragged-rows). Later plates can read those fields directly,
such as `r.prior + sum(r.pointwise.y)`. The emitted representation stacks each
leaf along the mapped axes and keeps ragged field lengths separate from the
record array's valid extent. Cartesian axes retain Julia's order, also through
`vec`/`reshape` of a record matrix (`vec`: valid records first), and empty
axes skip the cell body. These operations use ordinary JAX reverse differentiation.
Empty records written as `NamedTuple()`, `NamedTuple{()}(())` or `(;)` have the
same representation, including in nested results and kernels with no inputs.
Public named-record results use Python dictionaries, so an empty record is `{}`.
Record-valued plates can keep empty fields beside numerical leaves, including
in ragged cells. A plate of records with no numerical leaves remains unsupported.
A helper's `map` keeps each entry's shape and lengths: mapping records to a numerical field (`map(records) do r; r.nested.y end`) gives dense rows that indexing, `first`, `last`, `similar`, loops and `sum` read as arrays.

## Julia evaluation

The Julia callable evaluates the same authored mathematical expression. Its output establishes the reference value for a consumer testing the emitted module.

```@example jax
prepared = prepare(quadratic)
prepared(3.0, 2.0)
```

A PPL model can emit its sampler boundary through `jax_source(built, bound)`. The generated module includes data preparation and coordinate metadata. A top-level plate's cell recipe that reads only the plate's data and that the JAX functions cannot compute runs in `prepare_data` as a loop over the cells, and the plate reads its values as inputs of its own, one per cell. A branch in a top-level plate's cell whose condition reads only the plate's data, such as a censored row's `y <= lower`, also inside an arm of a parameter's support guard, is decided in `prepare_data`: for each pattern of such decisions it returns the cells that take it (`<value>_lanes_<k>`, 0-based) and the order that restores the plate (`<value>_lane_order`), and the functions run one plate per pattern over its cells. Each cell then evaluates only the arms it takes, as the Julia program does, instead of both arms of a batched `jax.lax.cond`. The patterns follow the cell's source, each with its plate even when no cell takes it, so one module serves every dataset. A data-only definition that calls a program-level `@kernel` with positional arguments, as Julia runs it (`prepare(spec)(args...)`), is computed in `prepare_data` by a host function of that kernel's plan. Inspect its untranslated-data declarations before treating it as a standalone raw-data pipeline; a definition without a host translation remains a required prepared input. A cell's vector argument arrives padded to the longest row, and an array it is broadcast into is padded along the same axes: `size`, reductions, also over `dims`, and `vec` read only the row's entries. Padded lengths and sizes retain bounds through `+`, `-` and `*`, including `(1 + length(parameters)) * length(row)`. Capacities follow input shapes during JAX tracing; emitted source serves different sizes. Ranges and arrays of equal valid lengths pad alike. Past its stop, a range to such a length repeats the stop, as padding repeats a row's last entry. `diff(x)` of a padded row has one valid entry fewer, and its padding repeats the last valid difference; a row with no difference (empty or one entry) has none to repeat. Two rows of a cell broadcast together (a feature row and its standard errors, `(x .- centres) ./ sqrt.(scales .^ 2 .+ s .^ 2)`) pad their result by Julia's broadcast length, the rows' common length or the other row's where one has length one; lengths Julia refuses to broadcast (`DimensionMismatch`) are not checked. Rows padding different axes by different lengths (`x .* s'`) are refused. `reduce(hcat, cells)` of vectors or matrices keeps Julia's columns; like `stack`, it raises at run time unless every column fills the padded width (Julia's `DimensionMismatch`). A response whose element type admits `missing` is passed to `prepare_data` as the caller has it, with `None` for a missing entry; the module computes its presence mask, so one module serves every absence pattern.

Data-only `Iterators.product` over numerical ranges or arrays runs in
`prepare_data`, including host loops and `collect`. For example,
`states = collect(Iterators.product(0:n, 0:m))` can supply a plate's states:
the support lengths remain runtime data, and the first axis advances
fastest, as in Julia. Collected products preserve empty axes and their
multidimensional shape. A collected product of homogeneous numerical
tuples crosses to JAX as one array per tuple field with that same shape;
compiled plates and indexing read those fields together. Increasing the
support lengths changes the arrays, not the emitted bodies.

Preparation also accepts recipes with no inputs. For example,
`states = collect(Iterators.product(0:1, 0:1))` runs in `prepare_data()` when
emitted with `data = ()` or `data = NamedTuple()`. Call it once and pass the
result to the functions as `f(x, **prepared)`. This applies to input-free
host operations generally, including chains of recipes; it requires no
dummy data dependency. Constants with a device translation remain in the
functions, and numerical parameters remain runtime inputs.

`prepare_data` takes the data by the names the model reads them, also a name Python reserves or the module's code uses (`class`, `lambda`, `id`). No parameter can have such a name, so `prepare_data` takes it through `**data`: call it by name, `prepare_data(**raw)`. The dict it returns, the functions' `**prepared` and the metadata (`DATA`, `INPUTS`, `PREPARED`, `UNTRANSLATED`) use the same names. A name that is no Python identifier (a submodel's `r.g`) is spelled `r_g` there, and `UNTRANSLATED` keys it as the binding does.

The functions take the prepared data as `prepare_data` returns them, host NumPy arrays, eagerly or under `jax.jit`, whether the data are closed over or passed as arguments. Each prepared array is converted to a JAX array on entry, so a plate cell can index it by its traced index. A prepared vector of numeric tuples of one element type, or of named numbers (a per-cell shape, `[(minimum(r), maximum(r)) for r in R]`), is the tuple of its columns, one array per field; the functions raise a `TypeError` for a list of its rows.

A dotted cell latent, `v[i] .~ D.(args…)` with a length per index that data decide, has its dimensions, packed offsets, block table and argument entries computed in `prepare_data` when its arguments read data, scalar parameters or a shared array; an argument that reads a sampled array at the index, or an array of rank 2 or more at each index, raises `JAXEmissionError` for now.

Translation errors identify an unsupported operation instead of substituting a different computation. JAX compilation, device placement and differentiation belong to the Python consumer.

## Further JAX guides

- [JAX helpers and submodels](jax-helpers.md): shared submodel functions, and helpers with loops and branches.
- [JAX host preparation](jax-host.md): data-only Julia in `prepare_data`.
- [JAX translations](jax-translations.md): package-defined translations.
- [JAX runtime and derivatives](jax-runtime.md), [ODE solves](jax-ode.md): runtime checks, derivative rules, stable rewrites, ODE bridges.

The [Eight-schools example](eight-schools.md) explains the model and coordinate boundary that a PPL export must preserve.
