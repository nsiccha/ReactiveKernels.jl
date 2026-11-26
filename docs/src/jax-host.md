# JAX host preparation

Companion to [JAX source](jax.md): the data-only Julia that `prepare_data` runs on the host, once per dataset.

## Data-only functions

Host `findfirst(predicate, vector)` returns the first matching one-based index or `nothing`. Captured predicates (`isequal(label)`, `Base.Fix1`, `Base.Fix2`) work in positional kernels too: `plate(values) do value; Int(findfirst(isequal(value), levels)); end`. Labels and vector lengths remain runtime inputs.

RK 0.1.286 adds `findall(vector)`/`findall(predicate, vector)`: every matching one-based position, possibly empty. Predicates run in order and return `Bool`. `Dict(zip(keys, values))`, including `eachindex(rows)` values, supports indexing, `get` and `haskey` in host plates. Zip is reusable and stops at its shortest input. Repeated keys keep their last value; equality follows `isequal` (NaNs, signed zero). These operations also translate in data-only `definitions`.

Host preparation also orders vectors of numbers, strings or tuples of them as Julia's `isless` does (-0.0 before 0.0, every NaN last): `sort`, `unique` (and `sort!`, `unique!` of a list built with `push!`), `searchsortedfirst` and `searchsortedlast` (one-based), and `sortperm(v)`, whose one-based positions keep equal values in ascending position. The two searches look for a number or a string, not a tuple. `invperm(p)` of a vector of integers is the inverse permutation and raises `ArgumentError: argument is not a permutation` as Julia does. A data-only definition such as `pos = invperm(vcat(rows_a, rows_b))` therefore runs in `prepare_data`. `sort`, `sort!` and `sortperm` take `by` as a projection onto tuple fields (`by = r -> (r[2], r[1])`); any other `by`, and `rev` and `lt`, have no translation. In the JAX functions these calls have none either: a recipe that also reads a parameter (`theta[sortperm(rows)]`) raises `JAXEmissionError` naming the function, so give the permutation a definition of its own that reads data alone.

`collect(zip(a, b))` is a vector of tuples whose fields keep their types: an integer label beside a float stays an integer. Tuples are dictionary keys, also in a generator (`Dict(key => i for (i, key) in enumerate(keys))`) and a broadcast lookup (`getindex.(Ref(lookup), keys)`); a `Set` takes a generator too. `all(predicate, values)` and `any(predicate, values)` call a `Bool` predicate in order and stop at the first element that decides the result. `==` and `!=` of two numerical arrays compare their shapes and values, integers exactly.

Host preparation keeps ordinary data-only Julia as written (RK 0.1.278). Typed comprehensions convert each element as `convert` does, so `Int[findfirst(==(k), levels) for k in keys]` raises when a key is absent; nested generators keep Julia's clause order. `push!` and `append!` grow any local vector, which becomes a fresh Python list, so an input is never written. `sizehint!`, `view`, `@view` and `@views` read as Julia's. An index that is `findfirst`'s result raises `ArgumentError: invalid index: nothing of type Nothing` when it is `nothing`. `issorted` and a scalar `isless` follow the `isless` order of `sort`. A `Set` holds `isequal` keys and is never iterated (Julia's hash order). `(; a, b) = nt` binds fields by name; an anonymous function is called where host code calls it (`filter!`).

`repeat(A, counts...)` tiles `A` `counts[d]` times along axis `d`, as Julia's `repeat(A; outer = counts)` does, and `repeat(A; inner, outer)` first repeats each entry `inner[d]` times along axis `d`; each keyword takes an integer or a tuple of them. Counts past `A`'s rank add trailing axes, as Julia's do (`repeat(v, 1, k)` of a vector is a matrix with `k` columns). Shapes are static while JAX traces, so the JAX functions need static counts (literals, or integers static to `jax.jit`); a count of data is computed in `prepare_data` when the recipe reads data alone. A categorical model's prior scale per coefficient term, repeated once per non-reference category (`repeat([intercept_scale, slope_scale], 1, k)`), therefore runs in `prepare_data`, and one module serves every category count. Julia's argument checks hold: a negative count, or with `inner` a count list shorter than `A`'s rank, raises with Julia's `ArgumentError` message. A recipe that repeats by a count the functions trace raises `JAXEmissionError` naming the recipe; a padded (ragged) value has no `repeat` translation.

A `Bool` array index (a `BitVector` or `Vector{Bool}`, such as `table[table[:, 2] .!= 1, 4]` or `values[present]`) selects the entries where it is true, as Julia's logical indexing does; it is not shifted as one-based positions. A mask of the array's whole shape (`A[A .> 0]`) selects in column-major order. `prepare_data` and other host code use a NumPy boolean mask, which fails on a mask of another length as Julia's `BoundsError` does; a list built with `push!` keeps the entries whose mask entry is true. In device code the count of selected entries depends on data, so the gather is padded to the mask's length: the selected entries come first, padding repeats the last of them, and reductions, `length` and broadcasts mask the padding as they do for a ragged row. A write through a mask writes only the selected entries. A leading device mask also selects matrix rows (`A[mask, :]`). A column mask beside a kept axis and a mask of the groups of a ragged value raise `JAXEmissionError`, as does a single `Bool` index into an array, which Julia refuses.

A data-only vector of records whose fields include vectors of a different length in each record, such as a per-subject design a data-only plate builds, crosses to the functions with each vector field padded as a ragged input is, and `<name>_lengths`: the tuple of those fields' lengths, `None` for a number field (RK 0.1.310). Pass both from `prepare_data` to the functions with `**prepared`.

## Groups of ragged rows

For `x::Vector{Vector{Vector{Int}}}` and other numerical vectors nested
three deep, `values, lengths = pad_ragged(x, nested=True)` pads both inner
axes. Pass `values, lengths` to the direct function. Unpacking
`group_lengths, row_lengths = lengths` gives the number of rows in each
group and the number of entries in each row. For example, `[[], [[]]]` has
group lengths `[0, 1]`. The second group contains one empty row.
With bound data, `prepare_data(x=x)` supplies the pair under `x_lengths`.
A plate can return a group of rows in a tuple or named tuple. For example,
if a cell returns `(weights, rows)`, a subsequent plate can read
`length(cell[2])` for the row count and `sum(cell[2][1])` for the sum of
the first row. An existing empty first row sums to zero.
A group can concatenate its rows with `reduce(vcat, rows)`. When a traced
index selects a group, the selected vector's length can vary within one
compiled function. Use a named WANT such as `(; flat = :flat)` to get
`flat` and `flat_lengths` under JIT. A plain vector is trimmed in eager
execution and requires its logical length to fill its storage under JIT.

## Local helpers

A local named helper with plain positional arguments (`add!(i, value) = push!(out, i * value)`) reads the locals of the code that defines it and writes their collections. It reads their values at each call, so a local reassigned before the call is read as reassigned. A helper that rebinds a captured local, has typed arguments or several methods, or is passed to a recorded helper while it captures locals, raises `JAXEmissionError`. Helpers that write their arguments in place are described in [JAX helpers and submodels](jax-helpers.md).

## Ragged row sums

When each row of a numerical plate sums an inner plate, `prepare_data` can
group entries across rows by conditions on the bound data. It records row and
column positions for the actual entries. The compiled function evaluates the
selected arms and sums the values in their original rows. Shared scalar
parameters and unresolved comparisons stay in the compiled function.
The compiled function can batch straight-line arithmetic across a group.
When a live condition guards that arithmetic, all visits and mapped chains
must select the same arm to take the parallel path. Mixed decisions use
the scalar cell. The same emitted module serves different row lengths and
branch decisions. Changing the number of entries in a group can trigger
recompilation by JAX.
