# Values a plate cell reads at its own position, computed at that position
# only (`plate_cell`, cells.jl; snag `rkppl-per-subjec-a3d6e6d8`, increment 1b),
# or at the several rows the cell gathers (snag `compact-cell-sam-d85d96ae`).
#
# A cell that captures a whole array and reads it at its batched element,
# `log_k[i]` in `plate(i, ...) do i ... exp(log_k[i]) end`, needs one row of
# it. The same is true when `log_k` is itself a batched input to the cell's
# plate. A cell whose batched element is an integer vector `r` reads the rows
# `log_w[r]`, and a cell that reads its rows through a shared index array,
# `log_w[rows[k]]` (`rows` read whole or through `vec`), reads `rows[k]`: one
# row or several. When that array's producer computes it row by row, the
# native lowering runs the producer's own operation on those rows instead: the
# producer is given views of its row-aligned operands at those rows, and the
# cell is given a `_CellWindow` holding the result, which answers reads of
# those rows and throws `BoundsError` for any other. The rewrite applies to
# these source shapes, read off the closure's native body
# (`_kernel_native_body`):
#
# - a dotted expression (one fused broadcast) whose operands are recipe
#   inputs, numeric literals, gathers `A[I]` / `A[I, 1, ...]` of an input by
#   an integer vector input `I` (integer literals after it), column reads
#   `A[:, 1, ...]`, products `A * w` of two inputs, and literal-position reads
#   `a[1]` of an array of numbers (read whole);
# - such a gather on its own (also as `_readonly_gather(A, I)`, the view RKPPL
#   reads a bare gathered definition through), and such a column read on its
#   own;
# - `hcat(a, b, ...)` of inputs: row `r` is the `hcat` of each operand's row
#   `r`;
# - a product `A * W...` (a source expression, or a bare call that keeps
#   Base's `*` as its operation) whose left factor `A` is an input array, a
#   column read or an `hcat`: row `r` of the product is `A[r:r, ...] * W...`.
#   Its full axes, which a dotted reader and the cell's linear reads need,
#   follow from `A`'s for a product of two operands whose right operand is an
#   array or a number (`_cell_product_shape`); any other product is read only
#   by gathers;
# - an authored plate: row `r` of its value is the plate over row `r` of each
#   batched argument, lowered natively over those one-row views at the reading
#   cell. A shared argument its body reads only as `v[q]`, `q` the element of
#   one batched integer argument, is a linear gather by that argument, as in a
#   cell (`_cell_capture_reader`); its other shared arguments are read whole.
#   Its full axes are its batched arguments' broadcast axes, as its loop
#   combines them.
#
# Row `r` of the dotted expression is the same expression over row `r` of each
# array operand (the operand itself when broadcasting extrudes it), and row
# `r` of a gather by `I` reads row `I[r]` of its source; rows `R` read rows
# `R`, and `I[R]`. Those rows are again computed at those rows when their
# producers have one of these shapes, and are otherwise computed whole and
# indexed. A value is computed only at rows when every reader of it reads rows
# this way, so the program computes no value twice in different ways. The
# cell's batched element must be provably an integer (a linear position) or an
# integer vector, and each gather index provably an integer vector
# (`_cell_static_type`). A recipe that holds one of these shapes inside one
# expression, such as a gather from an inline `vcat`, is first split into
# recipes of its own along its authored structure (`_cell_outlined_pushdowns`,
# below).
#
# Shapes and bounds keep the whole program's checks for this cell: each
# dotted expression's and `hcat`'s full shape is computed from its operands'
# axes (so mismatched operands still throw `DimensionMismatch`), the rows are
# checked against it, factor rows are bounds-checked views, and window reads
# check their row. Values at other rows are not computed, so an error that
# only another row raises (an out-of-range gather index elsewhere, a domain
# error in another subject's predictor) is not raised, as the cell itself does
# not run other cells. Everything else, including the operation table and
# `on_error = :ignore` twins, is unchanged.

"""
    _CellWindow(data, rows, shape)

Rows of a value read by one cell. `rows` is one row (an `Int`), or the rows a
gather by an integer vector reads (a `Vector{Int}`, in the order read, repeats
allowed). `data` holds the value at those rows, in that order along its first
axis, every later dimension whole; `shape` is the value's full axes, or
`nothing` when only row reads `v[I, j...]` are made of it. Reading any other
row throws `BoundsError`.
"""
struct _CellWindow{D,R<:Union{Int,Vector{Int}},S}
    data::D
    rows::R
    shape::S
end

# The position along `data`'s first axis of row `row`; `at` is the read. The
# lookup reads the window's rows only, so native reverse differentiation sees
# index arithmetic apart from the data (below).
@inline function _cell_local_row(w::_CellWindow{<:Any,Int}, row, at)
    row == w.rows || throw(BoundsError(w.rows, at))
    firstindex(w.data, 1)
end
@inline function _cell_local_row(w::_CellWindow{<:Any,Vector{Int}}, row, at)
    k = findfirst(==(row), w.rows)
    k === nothing && throw(BoundsError(w.rows, at))
    firstindex(w.data, 1) + k - 1
end
# The positions of rows `I`, computed before the data is read: a vector read
# is then one gather of the data at those positions. Reading the data row by
# row inside the position lookup instead (`[w[i, j] for i in I]`) made native
# Enzyme Reverse refuse the read with `EnzymeRuntimeActivityError`, the
# constant positions and the active data meeting in one loop.
@inline function _cell_local_rows(w::_CellWindow, I::AbstractVector{<:Integer}, at)
    positions = Vector{Int}(undef, length(I))
    for (k, i) in enumerate(I)
        positions[k] = _cell_local_row(w, i, at)
    end
    positions
end

# Whether gather index `I` reads exactly a vector value's window rows, in
# order: then the gather is the window's data.
@inline _cell_reads_rows(w::_CellWindow{<:Any,Int}, I) = false
@inline _cell_reads_rows(w::_CellWindow{<:Any,Vector{Int}}, I) =
    isone(firstindex(w.data, 1)) && I == w.rows

# Linear reads `v[q]`, as the cell and single-index gathers make them.
@inline function Base.getindex(w::_CellWindow{<:Any,<:Any,Tuple{}}, q::Integer)
    q == 1 || throw(BoundsError(w.rows, q))
    w.data[]
end
@inline function Base.getindex(w::_CellWindow{<:Any,<:Any,<:Tuple{Any,Vararg{Any}}},
                               q::Integer)
    position = Tuple(CartesianIndices(w.shape)[q])
    w.data[_cell_local_row(w, first(position), q), Base.tail(position)...]
end
@inline function Base.getindex(w::_CellWindow{<:Any,<:Any,<:Tuple},
                               I::AbstractVector{<:Integer})
    cells = CartesianIndices(w.shape)
    positions = [CartesianIndex(_cell_local_row(w, first(Tuple(cells[q])), q),
                                Base.tail(Tuple(cells[q]))...) for q in I]
    w.data[positions]
end
# A vector value with one-based axes: linear position `q` is row `q`.
@inline function Base.getindex(w::_CellWindow{<:Any,<:Any,<:Tuple{Base.OneTo}},
                               I::AbstractVector{<:Integer})
    _cell_reads_rows(w, I) && return w.data[:]
    w.data[_cell_local_rows(w, I, I)]
end

# Row reads `v[i, j...]` and gathers `v[I, j...]`.
@inline Base.getindex(w::_CellWindow, i::Integer, j::Integer, rest::Integer...) =
    w.data[_cell_local_row(w, i, (i, j, rest...)), j, rest...]
@inline function Base.getindex(w::_CellWindow, I::AbstractVector{<:Integer}, j::Integer,
                               rest::Integer...)
    _cell_reads_rows(w, I) && return w.data[:, j, rest...]
    w.data[_cell_local_rows(w, I, (I, j, rest...)), j, rest...]
end

# A read-only gather of a vector (`_readonly_gather`, core.jl, the view RKPPL
# reads a bare gathered definition through) reads the window's rows as the
# linear gather `w[I]` does; any other shape keeps `_readonly_gather`'s
# `MethodError`.
@inline _readonly_gather(w::_CellWindow{<:Any,<:Any,<:Tuple{Any}},
                         indices::AbstractVector{<:Integer}) = w[indices]

Base.show(io::IO, w::_CellWindow{<:Any,Int}) =
    print(io, "cell window at row ", w.rows, " of ", summary(w.data))
Base.show(io::IO, w::_CellWindow{<:Any,Vector{Int}}) =
    print(io, "cell window at rows ", w.rows, " of ", summary(w.data))

# The full broadcast axes of an operand (`Base.broadcastable` as `broadcasted`
# applies it); folded across operands by `_plate_broadcast_shape`.
@inline _cell_leaf_shape(arg) = axes(Base.broadcastable(arg))

# Row `row` of a broadcast operand: a one-row view, or the operand itself when
# broadcasting extrudes its first dimension. Always a `UnitRange` view (a
# union of range types defeats native Reverse type analysis). Rows `rows` are
# a view at those rows, and an extruded operand repeats its row.
@inline _cell_row_slice(arg, row) = arg
@inline _cell_row_slice(arg::Tuple, row::Integer) = length(arg) == 1 ? arg : (arg[row],)
@inline _cell_row_slice(arg::Tuple, rows::Vector{Int}) =
    [arg[length(arg) == 1 ? 1 : row] for row in rows]
@inline function _cell_row_slice(arg::AbstractArray{<:Any,N}, row::Integer) where {N}
    N == 0 && return arg
    axis = axes(arg, 1)
    position = length(axis) == 1 ? first(axis) : row
    view(arg, position:position, ntuple(_ -> Colon(), Val(N - 1))...)
end
@inline function _cell_row_slice(arg::AbstractArray{<:Any,N}, rows::Vector{Int}) where {N}
    N == 0 && return arg
    axis = axes(arg, 1)
    positions = length(axis) == 1 ? fill(Int(first(axis)), length(rows)) : rows
    view(arg, positions, ntuple(_ -> Colon(), Val(N - 1))...)
end

# The full axes of a matrix-vector product `A * v`.
@inline _cell_factor_shape(A::AbstractMatrix) = (axes(A, 1),)

# The full axes of a product `A * w` of two operands, from the full axes of its
# left factor, as `*` forms them: a matrix-vector product is a vector, a
# product with a matrix spans its columns, a scaling keeps the factor's axes.
@inline _cell_product_shape(factor::Tuple, w::AbstractVector) = (first(factor),)
@inline _cell_product_shape(factor::Tuple, w::AbstractMatrix) = (first(factor), axes(w, 2))
@inline _cell_product_shape(factor::Tuple, w::Number) = factor

# The full axes of a column read `A[:, j...]`: `A`'s first axis.
@inline _cell_column_shape(shape::Tuple) = (first(shape),)

# The full axes of `hcat(xs...)` from its operands' full axes (`()` for a
# number), as `hcat` forms them: the second dimension adds up, every other
# must agree (a missing trailing dimension counts as 1), or `DimensionMismatch`.
@generated function _cell_hcat_shape(shapes::Tuple)
    ranks = Int[fieldcount(T) for T in fieldtypes(shapes)]
    n = max(2, maximum(ranks; init = 0))
    size(i, d) = d <= ranks[i] ? :(length(shapes[$i][$d])) : 1
    checks = Any[]
    for i in 2:length(ranks), d in 1:n
        d == 2 && continue
        push!(checks, :($(size(1, d)) == $(size(i, d)) || throw(DimensionMismatch(
            string("hcat operands mismatch in dimension ", $d)))))
    end
    width = foldl((a, i) -> :($a + $(size(i, 2))), 2:length(ranks); init = size(1, 2))
    axes = Any[d == 2 ? :(Base.OneTo($width)) : :(Base.OneTo($(size(1, d)))) for d in 1:n]
    Expr(:block, checks..., Expr(:tuple, axes...))
end

# The full axes of `vcat(xs...)` from its operands' full axes (`()` for a
# number), as `vcat` forms them: the first dimension adds up, every other must
# agree (a missing trailing dimension counts as 1), or `DimensionMismatch`.
@generated function _cell_vcat_shape(shapes::Tuple)
    ranks = Int[fieldcount(T) for T in fieldtypes(shapes)]
    n = max(1, maximum(ranks; init = 0))
    size(i, d) = d <= ranks[i] ? :(length(shapes[$i][$d])) : 1
    checks = Any[]
    for i in 2:length(ranks), d in 2:n
        push!(checks, :($(size(1, d)) == $(size(i, d)) || throw(DimensionMismatch(
            string("vcat operands mismatch in dimension ", $d)))))
    end
    height = foldl((a, i) -> :($a + $(size(i, 1))), 2:length(ranks); init = size(1, 1))
    axes = Any[d == 1 ? :(Base.OneTo($height)) : :(Base.OneTo($(size(1, d)))) for d in 1:n]
    Expr(:block, checks..., Expr(:tuple, axes...))
end

# The rows before each `vcat` operand, and their total: `(0, h1, h1 + h2, …)`.
@generated function _cell_vcat_starts(shapes::Tuple)
    ranks = Int[fieldcount(T) for T in fieldtypes(shapes)]
    starts = Any[0]
    for (i, rank) in enumerate(ranks)
        push!(starts, :($(starts[end]) + $(rank == 0 ? 1 : :(length(shapes[$i][1])))))
    end
    Expr(:tuple, starts...)
end

# The rows of operand `k` that rows `rows` of a `vcat` read, local to it, in
# the order read: none when no read falls in it.
@inline _cell_vcat_part_rows(starts::Tuple, k::Int, row::Integer) =
    starts[k] < row <= starts[k + 1] ? Int[row - starts[k]] : Int[]
@inline _cell_vcat_part_rows(starts::Tuple, k::Int, rows::Vector{Int}) =
    Int[row - starts[k] for row in rows if starts[k] < row <= starts[k + 1]]

# Rows `rows` of a `vcat` from its operands' rows (`parts`, each holding the
# rows `_cell_vcat_part_rows` gave it, in order; a number operand is its own
# row), along the first axis in the order read, every later dimension whole.
@inline function _cell_vcat_rows(rows, starts::Tuple, parts::Tuple, shape::Tuple)
    out = Array{Base.promote_eltypeof(parts...)}(undef,
        rows isa Integer ? 1 : length(rows), map(length, Base.tail(shape))...)
    _cell_vcat_fill!(out, rows, starts, parts, 1)
end
@inline _cell_vcat_fill!(out, rows, starts, ::Tuple{}, k) = out
@inline function _cell_vcat_fill!(out, rows, starts, parts::Tuple, k)
    part = first(parts)
    read = 0
    for (i, row) in enumerate(rows isa Integer ? (rows,) : rows)
        starts[k] < row <= starts[k + 1] || continue
        read += 1
        _cell_vcat_row!(out, i, part, read)
    end
    _cell_vcat_fill!(out, rows, starts, Base.tail(parts), k + 1)
end
@inline function _cell_vcat_row!(out, i, part::Number, read)
    for J in CartesianIndices(Base.tail(axes(out)))
        out[i, J] = part
    end
    out
end
@inline function _cell_vcat_row!(out, i, part::AbstractArray, read)
    for J in CartesianIndices(Base.tail(axes(out)))
        out[i, J] = part[read, J]
    end
    out
end

# The rows a plate reads through its index array `w` at the positions `qs`
# its index argument holds there (`v[w[q]]`): each `w[q]` is one row or several.
@inline _cell_via_rows(w, qs) =
    Int[row for q in qs for row in (vec(w)[q] isa Integer ? (vec(w)[q],) : vec(w)[q])]

# The rows several gather indices read, joined in order.
@inline _cell_gather_rows(ats...) =
    Int[row for at in ats for row in (at isa Integer ? (at,) : at)]

# The rows several readers ask a value for, each once, in first-asked order.
@inline _cell_union_rows(ats...) = unique!(_cell_gather_rows(ats...))

# Rows `rows` of a value's window `data` computed at rows `union` (which hold
# them), in the order asked, every later dimension whole. A value without rows
# (a dotted expression of numbers is a number) reads the same at every row.
@inline _cell_take_rows(data::Number, ::Vector{Int}, rows) = data
@inline _cell_take_rows(data::AbstractArray{<:Any,0}, ::Vector{Int}, rows) = data
@inline function _cell_take_rows(data::AbstractArray{<:Any,N}, union::Vector{Int},
                                 rows) where {N}
    positions = Int[_cell_union_position(union, row)
                    for row in (rows isa Integer ? (rows,) : rows)]
    view(data, positions, ntuple(_ -> Colon(), Val(N - 1))...)
end
@inline function _cell_union_position(union::Vector{Int}, row)
    k = findfirst(==(row), union)
    k === nothing && throw(BoundsError(union, row))
    k
end

# Rows of a product's left factor; bounds-checked, never extruded.
@inline _cell_factor_rows(arg::AbstractArray{<:Any,N}, row::Integer) where {N} =
    view(arg, row:row, ntuple(_ -> Colon(), Val(N - 1))...)
@inline _cell_factor_rows(arg::AbstractArray{<:Any,N}, rows::Vector{Int}) where {N} =
    view(arg, rows, ntuple(_ -> Colon(), Val(N - 1))...)

# A dotted expression's rows must lie in its full first axis.
@inline function _cell_check_row(shape::Tuple, row::Integer)
    valid = isempty(shape) ? row == 1 : checkindex(Bool, first(shape), row)
    valid || throw(BoundsError(CartesianIndices(shape), row))
    row
end
@inline function _cell_check_row(shape::Tuple, rows::Vector{Int})
    foreach(row -> _cell_check_row(shape, row), rows)
    rows
end

# The rows of a computed operand that rows `row` of its reader need.
@inline _cell_leaf_row(shape::Tuple{}, row::Integer) = 1
@inline _cell_leaf_row(shape::Tuple, row::Integer) =
    length(first(shape)) == 1 ? Int(first(first(shape))) : row
@inline _cell_leaf_row(shape::Tuple, rows::Vector{Int}) =
    Int[_cell_leaf_row(shape, row) for row in rows]

# A row window still spans a plate's trailing axes. A multidimensional cell
# selects its remaining coordinates from that window; a vector row is already
# exactly the one-cell slice needed by the plate.
@inline _cell_batched_slice(value, cell) = value
@inline function _cell_batched_slice(value::AbstractArray{<:Any,N}, cell) where {N}
    N <= 1 && return value
    local_cell = CartesianIndex(Base.setindex(Tuple(cell), 1, 1))
    _plate_cell_slice(value, local_cell)
end

# The row holding linear position `q` of a value with full axes `shape`; the
# rows holding positions `q`, for a gather by an integer vector.
@inline function _cell_linear_row(shape::Tuple{}, q::Integer)
    q == 1 || throw(BoundsError(CartesianIndices(shape), q))
    1
end
@inline _cell_linear_row(shape::Tuple, q::Integer) =
    Int(first(Tuple(CartesianIndices(shape)[q])))
@inline _cell_linear_row(shape::Tuple, q::AbstractVector{<:Integer}) =
    Int[_cell_linear_row(shape, p) for p in q]

# The index a row of a gather index holds, or the indices its rows hold.
@inline _cell_gather_row(index, row::Integer) = Int(only(index))
@inline _cell_gather_row(index, rows::Vector{Int}) = Int[i for i in index]

# A range view's range from its parts `(lo, hi)` or `(lo, step, hi)`, and its
# full axes; the positions of its source that rows `rows` of the view read.
@inline _cell_range(parts::Tuple) = (:)(parts...)
@inline _cell_range_shape(parts::Tuple) = (Base.OneTo(length(_cell_range(parts))),)
@inline _cell_range_rows(parts::Tuple, row::Integer) = Int(_cell_range(parts)[row])
@inline _cell_range_rows(parts::Tuple, rows::Vector{Int}) =
    let range = _cell_range(parts)
        Int[range[row] for row in rows]
    end

# A reshape's full axes from its dimensions; the positions of its source
# vector that rows `rows` of the reshaped array read: row `r` holds positions
# `r + n * (j - 1)` for each trailing coordinate `j` (column-major), trailing
# coordinate outer and row inner, so the rows reshape back in that order
# (`_cell_reshape_window`).
@inline _cell_dims_shape(dims::Tuple) = map(Base.OneTo, dims)
@inline _cell_reshape_rows(dims::Tuple, row::Integer) =
    Int[row + first(dims) * (j - 1) for j in 1:prod(Base.tail(dims); init = 1)]
@inline _cell_reshape_rows(dims::Tuple, rows::Vector{Int}) =
    Int[row + first(dims) * (j - 1) for j in 1:prod(Base.tail(dims); init = 1) for row in rows]
@inline _cell_reshape_window(data, row::Integer, dims::Tuple) =
    reshape(data, 1, Base.tail(dims)...)
@inline _cell_reshape_window(data, rows::Vector{Int}, dims::Tuple) =
    reshape(data, length(rows), Base.tail(dims)...)

# ---------------------------------------------------------------- analysis

# How a source recipe computes its value row by row. `roles[j]` is how input
# `j` is read: `:leaf` (a broadcast operand), `:index` (a gather index, and
# possibly a broadcast operand), `:gather` / `:linear_gather` (the source of
# `A[I, j...]` / `A[I]`, by input `gather_index[j]`), `:column` (a broadcast
# operand read as a column `A[:, k...]`), `:factor_leaf` (the left factor of a
# broadcast operand `A * w`, `w` input `gather_index[j]`), `:factor` (a
# product's left factor), `:scalar` (an array of numbers read at literal
# positions, `a[1]`, so whole), `:range_source` (the array a `view(A, lo:hi)`
# or `A[lo:hi]` with literal bounds reads), `:reshape_source` (the vector a
# `reshape(v, n, k...)` with literal dimensions reads), or `:whole`. `leaves` are the inputs whose
# axes make up a pointwise value's full shape. `operand` is the input `v` of a
# product `A * v` of exactly two inputs (0 otherwise): with a matrix `A` and a
# vector `v`, its full axes are `(axes(A, 1),)`. `indices[j]` are every index
# input a gather source `j` is read by (`A[I]` and `A[J]` in one expression
# read `A` at the rows of both). `parts` are a `vcat`'s operands in order: an
# input position, or a number literal; an input read as a `vcat` operand has
# role `:part`. `vias[j]` is the shared index array an authored plate reads
# source `j` through (`v[w[q]]`, `q` the element of index input
# `gather_index[j]`), 0 for a direct read; that index array has role `:via`.
struct _CellForm
    kind::Symbol
    roles::Vector{Symbol}
    gather_index::Vector{Int}
    leaves::Vector{Int}
    operand::Int
    indices::Vector{Vector{Int}}
    parts::Vector{Any}
    vias::Vector{Int}
    # A `:range_view`'s literal range, or a `:reshape`'s literal dimensions.
    literal::Any
end
_CellForm(kind, roles, gather_index, leaves, operand, indices = Vector{Int}[
        role in (:gather, :linear_gather) ? [gather_index[j]] : Int[]
        for (j, role) in enumerate(roles)], parts = Any[]) =
    _CellForm(kind, roles, gather_index, leaves, operand, indices, parts,
        zeros(Int, length(roles)), nothing)
_CellForm(kind, roles, gather_index, leaves, operand, indices, parts, vias) =
    _CellForm(kind, roles, gather_index, leaves, operand, indices, parts, vias, nothing)

# Expression heads that bind no names; any other head declines the rewrite.
const _CELL_PLAIN_HEADS = (:call, :ref, :tuple, :., Symbol("'"), :vect, :hcat,
    :vcat, :row, :nrow, :ncat, :parameters, :kw, :comparison, :&&, :||, :if,
    :curly, :..., :string, :(::))

function _cell_native(r::Recipe; outlined::Bool = false)
    closure = _lane_source_closure(r; outlined)
    closure === nothing && return nothing
    _, params, mod = closure
    source = _lane_expand_dot(_lane_strip_lines(r.source), mod)
    source isa Expr && !_lane_has_macro(source) || return nothing
    native = _lane_strip_lines(_kernel_native_body(source, mod, Set{Symbol}(params)))
    (native, params, mod)
end

_cell_materialize(ex) = ex isa Expr && ex.head === :call && length(ex.args) == 2 &&
    ex.args[1] == GlobalRef(@__MODULE__, :_native_broadcast_materialize)
_cell_broadcasted(ex) = ex isa Expr && ex.head === :call && length(ex.args) >= 2 &&
    ex.args[1] == GlobalRef(Base, :broadcasted)

# Whether `ex` reads none of `params` and binds no names.
function _cell_free_of(ex, params)
    ex isa Symbol && return !(ex in params)
    ex isa Expr || return true
    ex.head in _CELL_PLAIN_HEADS || return false
    all(arg -> _cell_free_of(arg, params), ex.args)
end

# Every parameter read in `ex` (a plain expression), into `uses` as `role`.
function _cell_whole_uses!(uses, ex, position, role)
    if ex isa Symbol
        haskey(position, ex) && push!(uses[position[ex]], role)
        return true
    end
    ex isa Expr || return true
    ex.head in _CELL_PLAIN_HEADS || return false
    all(arg -> _cell_whole_uses!(uses, arg, position, role), ex.args)
end

function _cell_form(r::Recipe; outlined::Bool = false)
    length(r.outputs) == 1 && !r.effectful || return nothing
    r.op isa _AuthoredPlateOp && return _cell_plate_form(r)
    r.source isa Symbol && return _cell_identity_form(r)
    if r.op === Base.:* && length(r.inputs) >= 2
        # A bare exact call `A * B...` keeps Base's `*` itself as its
        # operation (`_kernel_operation_body`); it is called positionally.
        n = length(r.inputs)
        roles = fill(:whole, n)
        roles[1] = :factor
        return _CellForm(:product, roles, zeros(Int, n), Int[], n == 2 ? 2 : 0)
    end
    r.op isa _KernelSourceOp || return nothing
    analyzed = _cell_native(r; outlined)
    analyzed === nothing && return nothing
    native, params, mod = analyzed
    position = Dict{Symbol,Int}(name => j for (j, name) in enumerate(params))
    uses = [Set{Symbol}() for _ in params]
    # A source's gather indices, in first-read order, and whether its gathers
    # are linear (`A[I]`) or row reads (`A[I, k...]`), the same for every read.
    gathers = Dict{Int,Tuple{Vector{Int},Bool}}()
    products = Dict{Int,Int}()
    leaves = Int[]
    param(x) = x isa Symbol && haskey(position, x)
    # `_readonly_gather(A, I)`, RKPPL's view of a bare gathered definition: the
    # linear gather `A[I]`.
    readonly(ex) = ex isa Expr && ex.head === :call && length(ex.args) == 3 &&
        !(ex.args[1] in params) && _kernel_resolve_binding(mod, ex.args[1]) === _readonly_gather
    function gather!(ex)
        length(ex.args) >= 2 || return false
        source, index = ex.args[1], ex.args[2]
        haskey(position, source) && haskey(position, index) || return false
        all(i -> i isa Int, ex.args[3:end]) || return false
        linear = length(ex.args) == 2
        found, linear_reads = get!(() -> (Int[], linear), gathers, position[source])
        linear_reads == linear || return false
        position[index] in found || push!(found, position[index])
        push!(uses[position[source]], linear ? :linear_gather : :gather)
        push!(uses[position[index]], :index)
        push!(leaves, position[index])
        true
    end
    function operand!(ex)
        _cell_broadcasted(ex) && return lazy!(ex)
        if ex isa Symbol
            haskey(position, ex) || return false
            push!(uses[position[ex]], :leaf)
            push!(leaves, position[ex])
            return true
        end
        ex isa Union{Int,Float64,Bool} && return true
        ex isa Expr && ex.head === :ref && return gather!(ex) || column!(ex) || scalar!(ex)
        readonly(ex) && return gather!(Expr(:ref, ex.args[2], ex.args[3]))
        ex isa Expr && ex.head === :call && return product!(ex)
        false
    end
    # `A[:, k...]`: row `r` is `A[r, k...]`.
    function column!(ex)
        length(ex.args) >= 3 && param(ex.args[1]) && ex.args[2] === :(:) &&
            all(i -> i isa Int, ex.args[3:end]) || return false
        push!(uses[position[ex.args[1]]], :column)
        push!(leaves, position[ex.args[1]])
        true
    end
    # `a[1]`: one element, the same for every row.
    function scalar!(ex)
        length(ex.args) >= 2 && param(ex.args[1]) && all(i -> i isa Int, ex.args[2:end]) ||
            return false
        push!(uses[position[ex.args[1]]], :scalar)
        true
    end
    # `A * w`: row `r` is `A[r:r, ...] * w`.
    function product!(ex)
        length(ex.args) == 3 && _cell_base_product(ex.args[1], mod, params) &&
            param(ex.args[2]) && param(ex.args[3]) && ex.args[2] != ex.args[3] || return false
        factor, w = position[ex.args[2]], position[ex.args[3]]
        get!(products, factor, w) == w || return false
        push!(uses[factor], :factor_leaf)
        push!(uses[w], :whole)
        push!(leaves, factor)
        true
    end
    lazy!(ex) = _cell_free_of(ex.args[2], params) && all(operand!, ex.args[3:end])
    kind = :pointwise
    operand = 0
    parts = Any[]
    literal = nothing
    if _cell_materialize(native)
        lazy = native.args[2]
        _cell_broadcasted(lazy) && lazy!(lazy) || return nothing
    elseif native isa Expr && native.head === :ref && length(native.args) >= 3 &&
           native.args[2] === :(:) && native.args[1] isa Symbol &&
           haskey(position, native.args[1]) && all(i -> i isa Int, native.args[3:end])
        # A column read `A[:, j...]`: row `r` is `A[r, j...]`.
        kind = :columns
        push!(uses[position[native.args[1]]], :leaf)
        push!(leaves, position[native.args[1]])
    elseif native isa Expr && native.head === :ref &&
           (range_view = _cell_range_view(native, position, mod)) !== nothing
        # `A[lo:hi]` with literal or scalar-input bounds: a range view.
        kind = :range_view
        source, literal = range_view
        push!(uses[source], :range_source)
        push!(leaves, source)
        foreach(part -> part isa Int || push!(uses[part.input], :whole), literal)
    elseif native isa Expr && native.head === :ref
        gather!(native) || return nothing
    elseif readonly(native)
        gather!(Expr(:ref, native.args[2], native.args[3])) || return nothing
    elseif native isa Expr && native.head === :call && length(native.args) >= 2 &&
           native.args[1] == GlobalRef(@__MODULE__, :_native_hcat) &&
           all(arg -> arg isa Symbol && haskey(position, arg), native.args[2:end])
        # `hcat(a, b, ...)`: row `r` is the `hcat` of each operand's row `r`.
        # `leaves` keeps every operand in order, repeats included.
        kind = :hcat
        for arg in native.args[2:end]
            push!(uses[position[arg]], :leaf)
            push!(leaves, position[arg])
        end
    elseif native isa Expr && native.head === :call && length(native.args) >= 2 &&
           native.args[1] == GlobalRef(@__MODULE__, :_native_vcat) &&
           all(arg -> arg isa Number || arg isa Symbol && haskey(position, arg),
               native.args[2:end])
        # `vcat(a, b, ...)` of inputs and number literals: its rows are its
        # operands' rows, one operand after another, so a row of it is a row
        # of exactly one operand.
        kind = :vcat
        for arg in native.args[2:end]
            arg isa Symbol || continue
            push!(uses[position[arg]], :part)
            push!(leaves, position[arg])
        end
        parts = Any[arg isa Symbol ? position[arg] : arg for arg in native.args[2:end]]
    elseif native isa Expr && native.head === :call && length(native.args) >= 3 &&
           _cell_base_product(native.args[1], mod, params) &&
           native.args[2] isa Symbol && haskey(position, native.args[2])
        kind = :product
        push!(uses[position[native.args[2]]], :factor)
        length(native.args) == 3 && native.args[3] isa Symbol &&
            haskey(position, native.args[3]) && (operand = position[native.args[3]])
        all(arg -> _cell_whole_uses!(uses, arg, position, :whole), native.args[3:end]) ||
            return nothing
    elseif (range_view = _cell_range_view(native, position, mod)) !== nothing
        # `view(A, lo:hi)` or `A[lo:hi]`, the bounds integer literals or
        # scalar inputs (RKPPL's packed parameter blocks): row `r` of it is
        # row `(lo:hi)[r]` of `A`.
        kind = :range_view
        source, literal = range_view
        push!(uses[source], :range_source)
        push!(leaves, source)
        foreach(part -> part isa Int || push!(uses[part.input], :whole), literal)
    elseif (reshaped = _cell_reshape(native, position, mod)) !== nothing
        # `reshape(v, n, k...)`, the dimensions integer literals or scalar
        # inputs (RKPPL's matrix blocks): row `r` of it holds positions
        # `r + n * (j - 1)` of `v`.
        kind = :reshape
        source, literal = reshaped
        push!(uses[source], :reshape_source)
        push!(leaves, source)
        foreach(part -> part isa Int || push!(uses[part.input], :whole), literal)
    else
        return nothing
    end
    kind === :pointwise && isempty(leaves) && return nothing
    roles = Symbol[]
    for used in uses
        role = isempty(used) ? :whole :
            used == Set([:leaf]) ? :leaf :
            issubset(used, Set([:leaf, :index])) ? :index :
            length(used) == 1 ? only(used) : nothing
        role === nothing && return nothing
        push!(roles, role)
    end
    indices = Vector{Int}[haskey(gathers, j) ? first(gathers[j]) : Int[]
                          for j in eachindex(params)]
    gather_index = Int[isempty(indices[j]) ? get(products, j, 0) : first(indices[j])
                       for j in eachindex(params)]
    _CellForm(kind, roles, gather_index, kind in (:hcat, :vcat) ? leaves : unique!(leaves),
        operand, indices, parts, zeros(Int, length(roles)), literal)
end

# A literal of a range bound or a reshape dimension: an integer, or a scalar
# input read whole, by its position (`_cell_literal_parts`).
struct _CellInputLiteral
    input::Int
end

# `view(A, lo:hi)` / `Base.view(A, lo:hi)` / `A[lo:hi]` with `A` a parameter
# and the bounds (a step allowed) integer literals or inputs:
# `(A's position, the range's parts)`.
function _cell_range_view(native, position, mod)
    native isa Expr || return nothing
    if native.head === :ref && length(native.args) == 2
        source, range = native.args
    elseif native.head === :call && length(native.args) == 3 &&
           _cell_resolves_to(native.args[1], mod, position, Base.view)
        source, range = native.args[2], native.args[3]
    else
        return nothing
    end
    source isa Symbol && haskey(position, source) || return nothing
    range isa Expr && range.head === :call && range.args[1] === :(:) &&
        length(range.args) in (3, 4) || return nothing
    parts = _cell_literal_parts(range.args[2:end], position, source)
    parts === nothing && return nothing
    (position[source], parts)
end

# `reshape(v, n, k...)` with `v` a parameter and the dimensions integer
# literals or inputs: `(v's position, the dimensions' parts)`.
function _cell_reshape(native, position, mod)
    native isa Expr && native.head === :call && length(native.args) >= 4 &&
        _cell_resolves_to(native.args[1], mod, position, Base.reshape) || return nothing
    source = native.args[2]
    source isa Symbol && haskey(position, source) || return nothing
    parts = _cell_literal_parts(native.args[3:end], position, source)
    parts === nothing && return nothing
    (position[source], parts)
end

# Each of `exs` as an `Int` literal or the input it names (not `source`).
function _cell_literal_parts(exs, position, source)
    parts = Any[]
    for ex in exs
        if ex isa Int
            push!(parts, ex)
        elseif ex isa Symbol && haskey(position, ex) && ex !== source
            push!(parts, _CellInputLiteral(position[ex]))
        else
            return nothing
        end
    end
    Tuple(parts)
end

# Whether a callee, as the closure resolves it, is the function `f`.
function _cell_resolves_to(callee, mod, position, f)
    callee isa Symbol && haskey(position, callee) && return false
    callee isa GlobalRef && return isdefined(callee.mod, callee.name) &&
        getglobal(callee.mod, callee.name) === f
    mod isa Module || return false
    _kernel_resolve_binding(mod, callee) === f
end

# An identity `b::T = a` (a typed alias RK keeps as a recipe): its rows are
# its input's rows, converted to the declared type as the typed local the
# whole program assigns converts them (`_declare_typed_output!`). Only a
# declaration that converts no elements (`AbstractVector`, `Any`) qualifies:
# it holds for the rows exactly when it holds for the whole value.
function _cell_identity_form(r::Recipe)
    op = r.op
    op isa _KernelSourceOp && kernel_sourceop_form(op) === :fused &&
        op.ignored_throws !== nothing && length(r.inputs) == 1 || return nothing
    f = _kernel_native_source(_lane_source_callable(op.f))
    fieldcount(typeof(f)) == 0 || return nothing
    params = _lane_source_params(f, 1)
    params === nothing && return nothing
    name = only(params)
    # A gensym parameter is stored without its `##` prefix (`_lane_source_arguments`).
    r.source in (name, Symbol("##", name)) || return nothing
    T = valtype(only(r.outputs))
    T === Any || T isa Type && T <: AbstractArray && eltype(T) === Any || return nothing
    _CellForm(:identity, [:leaf], [0], [1], 0)
end

# An authored plate's value, row by row: its batched arguments are broadcast
# operands, a shared argument read only at one batched integer argument's
# element, or through a shared index array at it (`v[w[q]]`, as a total over a
# group of rows reads them), is gathered by it, and any other shared argument
# is read whole.
function _cell_plate_form(r::Recipe)
    op = r.op::_AuthoredPlateOp
    atomic = typeof(op).parameters[2]
    inner = op.kernel.plan
    ig = inner.graph
    roots = ValueId[canon_id(ig, v.id) for v in inner.have]
    length(roots) == length(r.inputs) && allunique(roots) || return nothing
    n = length(r.inputs)
    leaves = Int[k for k in 1:n if !(k in atomic)]
    isempty(leaves) && return nothing
    roles = Symbol[k in atomic ? :whole : :leaf for k in 1:n]
    gather_index = zeros(Int, n)
    vias = zeros(Int, n)
    wants = Set(canon_id(ig, w.id) for w in inner.want)
    for k in atomic
        roots[k] in wants && continue
        read = _cell_capture_reader(inner, roots, atomic, roots[k])
        read === nothing && continue
        b, via = read
        # An index array is read whole; it cannot also be gathered.
        via == 0 || roles[via] in (:whole, :via) || continue
        roles[k] = :linear_gather
        gather_index[k] = b
        roles[b] = :index
        via == 0 || (vias[k] = via; roles[via] = :via)
    end
    any(k -> roles[k] === :via && (gather_index[k] != 0 || vias[k] != 0), 1:n) &&
        return nothing
    f = _CellForm(:pointwise, roles, gather_index, leaves, 0)
    _CellForm(f.kind, f.roles, f.gather_index, f.leaves, f.operand, f.indices, f.parts,
        vias)
end

# `*` as the closure resolves it is Base's, or the native body's companion
# for it (`_native_mul`, core.jl).
function _cell_base_product(callee, mod, params)
    callee isa GlobalRef && return isdefined(callee.mod, callee.name) &&
        getglobal(callee.mod, callee.name) in (Base.:*, _native_mul)
    callee === :* && !(callee in params) && mod isa Module || return false
    isdefined(mod, :*) && getglobal(mod, :*) === Base.:*
end

# The positions `k` of the cell's shared arguments that its body reads only as
# `v[q]` or `v[w[q]]`, `q` the element of one batched argument `b` and `w`
# another shared argument, as `k => (b, w)` (`w = 0` for `v[q]`). The element
# `q`, or `w[q]`, is a linear position or an integer vector of them.
function _cell_capture_positions(r::Recipe, static_type)
    plate = (r.op::_AuthoredPlateCellOp).plate
    atomic = typeof(plate).parameters[2]
    inner = plate.kernel.plan
    ig = inner.graph
    roots = ValueId[canon_id(ig, v.id) for v in inner.have]
    captures = Dict{Int,Tuple{Int,Int}}()
    length(roots) == length(r.inputs) - 1 && allunique(roots) || return captures
    wants = Set(canon_id(ig, w.id) for w in inner.want)
    for k in atomic
        roots[k] in wants && continue
        read = _cell_capture_reader(inner, roots, atomic, roots[k])
        read === nothing && continue
        b, via = read
        lane = static_type(r.inputs[1 + b])
        if via == 0
            _cell_positions_type(lane) || continue
        else
            lane <: AbstractArray{<:Integer} &&
                _cell_positions_type(static_type(r.inputs[1 + via])) || continue
        end
        captures[k] = read
    end
    captures
end

# An array whose elements are linear positions, or integer vectors of them.
_cell_positions_type(T) =
    T <: AbstractArray{<:Integer} || T <: AbstractArray{<:AbstractVector{<:Integer}}

# How the plate body reads shared argument `cid`: only as `v[q]` or `v[w[q]]`,
# `q` the element of one batched argument `b`, `w` one other shared argument,
# the same way at every read. `(b, w)`, `w = 0` for `v[q]`, or `nothing`.
function _cell_capture_reader(inner::Plan, roots, atomic, cid)
    ig = inner.graph
    found = nothing
    root(params, recipe, name) = let j = findfirst(==(name), params)
        j === nothing ? nothing : findfirst(==(canon_id(ig, recipe.inputs[j].id)), roots)
    end
    # The shared argument an index array is: itself, or `vec` of it, which
    # reads the same at every linear position.
    function via_root(params, recipe, name)
        k = root(params, recipe, name)
        k === nothing || return k
        j = findfirst(==(name), params)
        j === nothing && return nothing
        cid = canon_id(ig, recipe.inputs[j].id)
        for producer in inner.recipes
            length(producer.outputs) == 1 &&
                canon_id(ig, only(producer.outputs).id) == cid || continue
            analyzed = _cell_native(producer)
            analyzed === nothing && return nothing
            native, vec_params, _ = analyzed
            native == Expr(:call, GlobalRef(Base, :vec), only(vec_params)) || return nothing
            return findfirst(==(canon_id(ig, only(producer.inputs).id)), roots)
        end
        nothing
    end
    for recipe in inner.recipes
        slots = findall(v -> canon_id(ig, v.id) == cid, collect(recipe.inputs))
        isempty(slots) && continue
        recipe.op isa _KernelSourceOp || return nothing
        analyzed = _cell_native(recipe)
        analyzed === nothing && return nothing
        native, params, _ = analyzed
        reads = Any[]
        _cell_capture_reads!(reads, native, Set(params[slots])) || return nothing
        for read in reads
            q, via = read isa Symbol ? (read, nothing) : (read.args[2], read.args[1])
            b = root(params, recipe, q)
            (b === nothing || b in atomic) && return nothing
            w = via === nothing ? 0 : via_root(params, recipe, via)
            (w === nothing || (w != 0 && !(w in atomic)) || w == b) && return nothing
            found === nothing && (found = (b, w))
            found == (b, w) || return nothing
        end
    end
    found
end

# Whether `ex` reads the names `captured` only as `v[q]` or `v[w[q]]`, with `q`
# and `w` names; each such `q`, or `w[q]`, is recorded.
function _cell_capture_reads!(reads, ex, captured)
    ex isa Symbol && return !(ex in captured)
    ex isa Expr || return true
    ex.head in _CELL_PLAIN_HEADS || return false
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] isa Symbol &&
       ex.args[1] in captured
        index = ex.args[2]
        name(x) = x isa Symbol && !(x in captured)
        name(index) || Meta.isexpr(index, :ref, 2) && all(name, index.args) ||
            return false
        push!(reads, index)
        return true
    end
    all(arg -> _cell_capture_reads!(reads, arg, captured), ex.args)
end

"""
    _CellPushdowns

The plan-wide pushdown decision for `_lower_with_ops`: `recipes` are the
producers computed only at their readers' rows (their table slots are kept and
recorded in `slots` as lowering reaches them: an operation's slot, or, for an
authored plate, the table length before its reserved slots, with the
argument values it is lowered with in `plates`). `captures[cell recipe id]`
maps each shared argument position read at the cell's own rows to the batched
argument holding those rows and the shared argument they are read through
(`_cell_capture_positions`); `batched[cell recipe id]` names the cell's
ordinary batched argument positions; `order[recipe id]` is its position in the
plan, producers first.
"""
struct _CellPushdowns
    graph::Graph
    pushed::Set{ValueId}
    recipes::Set{Int}
    producer::Dict{ValueId,Recipe}
    forms::Dict{Int,_CellForm}
    captures::Dict{Int,Dict{Int,Tuple{Int,Int}}}
    batched::Dict{Int,Set{Int}}
    slots::Dict{Int,Int}
    plates::Dict{Int,Any}
    order::Dict{Int,Int}
    # The locality record (`_cell_locality`, plate_partition.jl): every value
    # a reader asked for at rows, with `(reader recipe id, role)` per ask;
    # why a value asked for has no row form (`:no_form`, `:unshaped`,
    # `:operands`); and, for a value with a form that is still computed
    # whole, the reader that reads it whole or is itself computed whole.
    requests::Dict{ValueId,Vector{Tuple{Int,Symbol}}}
    refused::Dict{ValueId,Symbol}
    dropped::Dict{ValueId,Int}
end

# Whether a producer reads an input in `role` at its own rows.
_cell_row_role(role) =
    role in (:leaf, :gather, :linear_gather, :column, :factor_leaf, :factor, :part,
             :range_source, :reshape_source)

# `outlined` are the ids of the recipes a cell lowering outlined (`_cell_outline`);
# `blocked`, when given, collects the ids of the producers whose values a reader
# asks for at rows but that have no row form; `known` keeps the forms of recipes
# already analyzed, by recipe.
function _cell_pushdowns(p::Plan; outlined::Set{Int} = Set{Int}(),
                         blocked::Union{Nothing,Set{Int}} = nothing,
                         known::IdDict{Recipe,Any} = IdDict{Recipe,Any}())
    any(r -> r.op isa _AuthoredPlateCellOp, p.recipes) || return nothing
    g = p.graph
    cidof(v) = canon_id(g, v.id)
    producers = Dict{ValueId,Vector{Recipe}}()
    readers = Dict{ValueId,Vector{Recipe}}()
    for r in p.recipes
        foreach(o -> push!(get!(producers, cidof(o), Recipe[]), r), r.outputs)
        foreach(c -> push!(get!(readers, c, Recipe[]), r), unique(map(cidof, r.inputs)))
    end
    static_type(v) = let candidates = get(producers, cidof(v), Recipe[])
        length(candidates) == 1 && only(candidates).op isa _BoundConstant ?
            typeof(only(candidates).op.value) : valtype(v)
    end
    fixed = Set(cidof(v) for v in Iterators.flatten((p.have, p.want)))
    forms = Dict{Int,_CellForm}()
    function form(cid)
        candidates = get(producers, cid, Recipe[])
        (cid in fixed || length(candidates) != 1) && return nothing
        r = only(candidates)
        haskey(forms, r.id) && return forms[r.id]
        f = get!(() -> _cell_form(r; outlined = r.id in outlined), known, r)
        f === nothing || (forms[r.id] = f)
        f
    end
    # A read needs the value's full shape unless it reads rows `v[I, j...]`,
    # takes rows of a product's left factor, or reads positions of a range
    # view's or a reshape's source.
    needs_shape(role) = !(role in (:gather, :factor, :range_source, :reshape_source))
    # Full axes are known for pointwise values, identities, column reads,
    # `hcat`s, `vcat`s, range views and reshapes, and for a product of two
    # operands whose right operand is an array or a number
    # (`_cell_product_shape`) and whose left factor's axes are known.
    function shaped(cid)
        f = forms[only(producers[cid]).id]
        f.kind in (:pointwise, :identity, :columns, :hcat, :vcat, :range_view, :reshape) &&
            return true
        f.operand == 0 && return false
        inputs = only(producers[cid]).inputs
        factor = cidof(inputs[findfirst(==(:factor), f.roles)])
        static_type(inputs[f.operand]) <: Union{AbstractVecOrMat,Number} &&
            (form(factor) === nothing || shaped(factor))
    end
    fits(cid, role) = form(cid) !== nothing && (!needs_shape(role) || shaped(cid))
    # Static operand requirements: integer vector indices, array factors (a
    # column read and an `hcat` are arrays) and the right operand of a dotted
    # operand's product (`_cell_product_shape`), arrays of numbers read at
    # literal positions.
    array_form(v) = let f = form(cidof(v))
        f !== nothing && f.kind in (:columns, :hcat)
    end
    function operands_fit(cid)
        r = only(producers[cid])
        f = forms[r.id]
        all(zip(f.roles, r.inputs, f.gather_index)) do (role, input, paired)
            role === :index ? static_type(input) <: AbstractVector{<:Integer} :
            role === :factor ? static_type(input) <: AbstractArray || array_form(input) :
            role === :factor_leaf ? (static_type(input) <: AbstractArray || array_form(input)) &&
                static_type(r.inputs[paired]) <: Union{AbstractVecOrMat,Number} :
            role === :scalar ? static_type(input) <: AbstractArray{<:Number} :
            role === :via ? _cell_positions_type(static_type(input)) :
            role in (:range_source, :reshape_source) ?
                static_type(input) <: AbstractArray || form(cidof(input)) !== nothing :
            true
        end && all(f.literal === nothing ? () : f.literal) do part
            part isa Int || static_type(r.inputs[part.input]) <: Integer
        end
    end
    captures = Dict{Int,Dict{Int,Tuple{Int,Int}}}()
    batched = Dict{Int,Set{Int}}()
    pushed = Set{ValueId}()
    requests = Dict{ValueId,Vector{Tuple{Int,Symbol}}}()
    refused = Dict{ValueId,Symbol}()
    dropped = Dict{ValueId,Int}()
    work = ValueId[]
    function reach!(cid, role, reader)
        push!(get!(() -> Tuple{Int,Symbol}[], requests, cid), (reader, role))
        cid in pushed && return
        if form(cid) === nothing
            candidates = get(producers, cid, Recipe[])
            (cid in fixed || length(candidates) != 1) && return
            blocked === nothing || push!(blocked, only(candidates).id)
            refused[cid] = :no_form
            return
        end
        if !fits(cid, role)
            refused[cid] = :unshaped
            return
        elseif !operands_fit(cid)
            refused[cid] = :operands
            return
        end
        push!(pushed, cid)
        push!(work, cid)
    end
    for r in p.recipes
        r.op isa _AuthoredPlateCellOp || continue
        positions = _cell_capture_positions(r, static_type)
        isempty(positions) || (captures[r.id] = positions)
        foreach(k -> reach!(cidof(r.inputs[1 + k]), :linear_gather, r.id), keys(positions))
        atomic = typeof(r.op.plate).parameters[2]
        lanes = Set(k for k in 1:length(r.inputs)-1 if !(k in atomic))
        batched[r.id] = lanes
        foreach(k -> reach!(cidof(r.inputs[1 + k]), :leaf, r.id), lanes)
    end
    while !isempty(work)
        cid = pop!(work)
        r = only(producers[cid])
        f = forms[r.id]
        for (j, input) in enumerate(r.inputs)
            _cell_row_role(f.roles[j]) && reach!(cidof(input), f.roles[j], r.id)
        end
    end
    # Keep a value at rows only while every reader reads it at rows.
    changed = true
    while changed
        changed = false
        for cid in collect(pushed)
            failing = nothing
            for reader in get(readers, cid, Recipe[])
                slots = findall(v -> cidof(v) == cid, collect(reader.inputs))
                ok = if reader.op isa _AuthoredPlateCellOp
                    positions = get(captures, reader.id, Dict{Int,Tuple{Int,Int}}())
                    lanes = get(batched, reader.id, Set{Int}())
                    all(slots) do s
                        s > 1 && ((haskey(positions, s - 1) &&
                            fits(cid, :linear_gather)) ||
                            (s - 1 in lanes && fits(cid, :leaf)))
                    end
                elseif length(reader.outputs) == 1 && cidof(only(reader.outputs)) in pushed
                    f = forms[reader.id]
                    all(s -> _cell_row_role(f.roles[s]) && fits(cid, f.roles[s]), slots)
                else
                    false
                end
                ok || (failing = reader.id; break)
            end
            failing === nothing && continue
            delete!(pushed, cid)
            dropped[cid] = failing
            changed = true
        end
    end
    recipes = Set(only(producers[cid]).id for cid in pushed)
    _CellPushdowns(g, pushed, recipes,
        Dict(cid => only(producers[cid]) for cid in pushed), forms, captures, batched,
        Dict{Int,Int}(), Dict{Int,Any}(), Dict(r.id => k for (k, r) in enumerate(p.recipes)),
        requests, refused, dropped)
end

# ---------------------------------------------------------------- outlining

# A recipe without a row form can hold a subexpression with one where a form
# reads a named value: `vcat(observed, drawn)[order]` gathers from an inline
# `vcat`, so a cell computed the whole `vcat` and gathered from it, while
# `stacked = vcat(observed, drawn); stacked[order]` computes both at the
# gathered rows (snag `inline-vcat-gath-bb73ae09`). A cell lowering outlines
# such subexpressions along the recipe's split nodes (`_KernelSplit`, the
# decomposition `bound=` partial evaluation splits data-only parts along,
# `_partial_emit_split!`): each becomes a recipe of its own, read by name.
# Split nodes are strict subexpressions only, so no part runs where the
# authored expression would not, and the parts apply the authored operations
# to the same operands in the same order: values are unchanged. An outline is
# kept only while its recipe is then computed at the cell's rows; any other
# recipe stays as authored.

# Whether `ex` is a dotted call `f.(x...)`, or one of a dotted operator `x .+ y`.
_cell_dotted_call(ex) = ex isa Expr && ex.head === :. && length(ex.args) == 2 &&
    Meta.isexpr(ex.args[2], :tuple)
function _cell_dotted(ex)
    _cell_dotted_call(ex) && return true
    ex isa Expr && ex.head === :call && !isempty(ex.args) || return false
    name = _kernel_split_callee_name(ex.args[1])
    name isa Symbol && length(String(name)) > 1 && startswith(String(name), ".")
end

# Whether a row form reads operand `child` of `parent` as a named value: a
# gather's source or index, an operand of `vcat`, `hcat` or `*`, or a
# broadcast operand that is not itself fused into the broadcast.
function _cell_named_operand(parent::Expr, child)
    parent.head === :ref && return true
    _cell_dotted(parent) && return !_cell_dotted(child)
    parent.head === :call && !isempty(parent.args) || return false
    _kernel_split_callee_name(parent.args[1]) in (:vcat, :hcat, :*)
end

# The split parts of recipe `r` (`_partial_split_parts`) with the nodes to
# outline whole marked, or `nothing`: each node in a named-operand position
# that holds no such node itself. A node holding a marked one is split around
# it (`_partial_emit_split!`).
function _cell_outline_nodes(r::Recipe)
    length(r.outputs) == 1 && !r.effectful || return nothing
    parts = _partial_split_parts(r)
    parts === nothing && return nothing
    nodes = parts.split.nodes
    index = IdDict{Any,Int}(node.expr => i for (i, node) in enumerate(nodes))
    named = falses(length(nodes))
    function walk(ex)
        ex isa Expr && !(ex.head in (:quote, :inert)) || return
        for child in (_cell_dotted_call(ex) ? ex.args[2].args : ex.args)
            i = get(index, child, 0)
            i == 0 || (named[i] = _cell_named_operand(ex, child))
            walk(child)
        end
    end
    walk(_kernel_branch_unwrap(r.source))
    marked = falses(length(nodes))
    below = falses(length(nodes))
    for (i, node) in enumerate(nodes)          # post-order: kids come first
        below[i] = any(k -> named[k] || below[k], node.kids)
        marked[i] = named[i] && !below[i]
    end
    any(marked) ? (; parts, marked = collect(marked)) : nothing
end

# `p` with each recipe in `outlines` (id => `_cell_outline_nodes`) outlined,
# and the ids of the recipes the outlines made. New values live in a copy of
# the graph; `p.graph` is unchanged.
function _cell_outline(p::Plan, outlines::Dict{Int,Any})
    isempty(outlines) && return p, Set{Int}()
    g = p.graph
    copied = Graph(copy(g.values), copy(g.recipes), copy(g.producers), copy(g.aliases),
                   g.version)
    next_id = Ref(minimum((r.id for r in p.recipes); init = 0) - 1)
    fresh() = (id = next_id[]; next_id[] -= 1; id)
    recipes = Recipe[]
    outlined = Set{Int}()
    readable = Dict{Int,Any}()
    for r in p.recipes
        outline = get(outlines, r.id, nothing)
        if outline === nothing
            push!(recipes, r)
            continue
        end
        start = length(recipes)
        _partial_emit_split!(recipes, readable, copied, r, outline.parts, outline.marked,
                             fresh)
        foreach(q -> push!(outlined, q.id), @view recipes[start+1:end])
    end
    template = Plan(copied, p.have, p.want, p.recipes, p.producer, p.cost, p.candidates)
    _partial_subplan(template, p.have, p.want, recipes), outlined
end

"""
    _cell_outlined_pushdowns(p::Plan) -> (plan, pushdowns)

`p` with the recipes outlined that the cells then compute at their rows (see
above), and that plan's pushdowns (`_cell_pushdowns`; `nothing` when the plan
has no cell). A recipe is outlined when a reader asks for its value at
rows and it has no row form; the outline stays while the recipe is pushed, and
is dropped for good otherwise. `p` itself when nothing is outlined.
"""
function _cell_outlined_pushdowns(p::Plan)
    any(r -> r.op isa _AuthoredPlateCellOp, p.recipes) || return p, nothing
    known = IdDict{Recipe,Any}()
    byid = Dict(r.id => r for r in p.recipes)
    candidates = Dict{Int,Any}()
    candidate(id) = haskey(byid, id) ?
        get!(() -> _cell_outline_nodes(byid[id]), candidates, id) : nothing
    kept = Dict{Int,Any}()
    refused = Set{Int}()
    while true
        q, outlined = _cell_outline(p, kept)
        blocked = Set{Int}()
        cells = _cell_pushdowns(q; outlined, blocked, known)
        lost = Int[id for id in keys(kept) if cells === nothing || !(id in cells.recipes)]
        if !isempty(lost)
            union!(refused, lost)
            foreach(id -> delete!(kept, id), lost)
            continue
        end
        new = Int[id for id in blocked if !haskey(kept, id) && !(id in refused) &&
                  candidate(id) !== nothing]
        isempty(new) && return q, cells
        foreach(id -> kept[id] = candidates[id], new)
    end
end

# ---------------------------------------------------------------- emission

# Emits, into `body`, the rows of pushed values one cell reads. Each pushed
# value is computed ONCE per cell, at the union of the rows its readers ask for
# (the cell's own reads, and each pushed reader's): a value two readers read at
# different rows, such as a predictor gathered by two index vectors in two
# definitions, or one whose readers read it at different rows of a shared
# source, would otherwise be computed, with everything upstream of it, once
# per read. The requests are collected first, from the readers down, in
# reverse plan order (`_cell_emit_windows!`): every reader of a value comes
# after it in the plan, so a value's rows are complete when it is reached.
# The windows are then computed in plan order, each reader taking its rows from
# its operand's window (`_cell_take!`, or a `_CellWindow` for a gather).
# Shapes are memoized per value, operand row slices per (value, row binding).
# `runtime_ops` / `runtime_recipes` are the operation table being built; a
# pushed plate's code reads the slots reserved at its own position.
struct _CellEmitter
    body::Expr
    pushdowns::_CellPushdowns
    names::Dict{ValueId,Symbol}
    runtime_ops::Vector{Any}
    runtime_recipes::Vector{Recipe}
    shapes::Dict{ValueId,Symbol}
    rows::Dict{Tuple{ValueId,Symbol},Symbol}
    requests::Dict{ValueId,Vector{Symbol}}
    unions::Dict{ValueId,Symbol}
    reads::Dict{ValueId,Vector{Any}}
    data::Dict{ValueId,Symbol}
end
_CellEmitter(body, pushdowns, names, runtime_ops, runtime_recipes) =
    _CellEmitter(body, pushdowns, names, runtime_ops, runtime_recipes,
        Dict{ValueId,Symbol}(), Dict{Tuple{ValueId,Symbol},Symbol}(),
        Dict{ValueId,Vector{Symbol}}(), Dict{ValueId,Symbol}(),
        Dict{ValueId,Vector{Any}}(), Dict{ValueId,Symbol}())

_cell_ref(name) = GlobalRef(@__MODULE__, name)

function _cell_bind!(e::_CellEmitter, prefix::Symbol, value)
    name = gensym(prefix)
    push!(e.body.args, Expr(:(=), name, value))
    name
end

function _cell_shape!(e::_CellEmitter, cid::ValueId)
    haskey(e.shapes, cid) && return e.shapes[cid]
    d = e.pushdowns
    r = d.producer[cid]
    f = d.forms[r.id]
    input_shape(input) = input in d.pushed ? _cell_shape!(e, input) :
        Expr(:call, _cell_ref(:_cell_leaf_shape), e.names[input])
    if f.kind === :product
        factor = canon_id(d.graph, r.inputs[findfirst(==(:factor), f.roles)].id)
        operand = canon_id(d.graph, r.inputs[f.operand].id)
        return e.shapes[cid] = _cell_bind!(e, :cell_shape,
            Expr(:call, _cell_ref(:_cell_product_shape), input_shape(factor),
                e.names[operand]))
    end
    f.kind === :vcat && return e.shapes[cid] = _cell_bind!(e, :cell_shape,
        Expr(:call, _cell_ref(:_cell_vcat_shape), _cell_vcat_part_shapes!(e, r, f)))
    f.kind === :range_view && return e.shapes[cid] = _cell_bind!(e, :cell_shape,
        Expr(:call, _cell_ref(:_cell_range_shape), _cell_literal!(e, r, f)))
    f.kind === :reshape && return e.shapes[cid] = _cell_bind!(e, :cell_shape,
        Expr(:call, _cell_ref(:_cell_dims_shape), _cell_literal!(e, r, f)))
    parts = Any[_cell_part_shape!(e, r, f, j) for j in f.leaves]
    shape = f.kind === :columns ?
        Expr(:call, _cell_ref(:_cell_column_shape), only(parts)) :
        f.kind === :hcat ? Expr(:call, _cell_ref(:_cell_hcat_shape), Expr(:tuple, parts...)) :
        Expr(:call, _cell_ref(:_plate_broadcast_shape), Expr(:tuple, parts...))
    e.shapes[cid] = _cell_bind!(e, :cell_shape, shape)
end

# The full axes of a `vcat`'s operands, in order: a tuple expression.
function _cell_vcat_part_shapes!(e::_CellEmitter, r::Recipe, f::_CellForm)
    d = e.pushdowns
    shapes = map(f.parts) do part
        part isa Int || return Expr(:tuple)
        input = canon_id(d.graph, r.inputs[part].id)
        input in d.pushed ? _cell_shape!(e, input) :
            Expr(:call, _cell_ref(:_cell_leaf_shape), e.names[input])
    end
    Expr(:tuple, shapes...)
end

# The full axes of the broadcast operand that input `j` of recipe `r` (form `f`)
# makes: the input itself, its column `A[:, k...]`, or its product `A * w`.
function _cell_part_shape!(e::_CellEmitter, r::Recipe, f::_CellForm, j::Int)
    d = e.pushdowns
    input = canon_id(d.graph, r.inputs[j].id)
    shape = input in d.pushed ? _cell_shape!(e, input) :
        Expr(:call, _cell_ref(:_cell_leaf_shape), e.names[input])
    f.roles[j] === :column && return Expr(:call, _cell_ref(:_cell_column_shape), shape)
    f.roles[j] === :factor_leaf || return shape
    w = e.names[canon_id(d.graph, r.inputs[f.gather_index[j]].id)]
    _cell_bind!(e, :cell_shape, Expr(:call, _cell_ref(:_cell_product_shape), shape, w))
end

# A form's literal (a range view's bounds, a reshape's dimensions) as a tuple
# expression: each part its integer, or the name of the input it reads.
function _cell_literal!(e::_CellEmitter, r::Recipe, f::_CellForm)
    parts = map(f.literal) do part
        part isa Int ? part : e.names[canon_id(e.pushdowns.graph, r.inputs[part.input].id)]
    end
    Expr(:tuple, parts...)
end

function _cell_operand_row!(e::_CellEmitter, cid::ValueId, row::Symbol)
    _footprint_record!(cid, :rows, row)
    get!(e.rows, (cid, row)) do
        _cell_bind!(e, :cell_operand, Expr(:call, _cell_ref(:_cell_row_slice),
            Expr(:call, GlobalRef(Base, :broadcastable), e.names[cid]), row))
    end
end

# Asks for pushed value `cid` at rows `row` (one row or an integer vector).
function _cell_request!(e::_CellEmitter, cid::ValueId, row::Symbol)
    rows = get!(() -> Symbol[], e.requests, cid)
    row in rows || push!(rows, row)
    row
end

# The rows every request of `cid` reads: the one request, or their union.
function _cell_union!(e::_CellEmitter, cid::ValueId)
    rows = e.requests[cid]
    e.unions[cid] = length(rows) == 1 ? only(rows) :
        _cell_bind!(e, :cell_rows_union, Expr(:call, _cell_ref(:_cell_union_rows), rows...))
end

# The data of pushed value `cid` at rows `row`, one of its requests: its
# window when that request is the window's rows, else those rows of it.
function _cell_take!(e::_CellEmitter, cid::ValueId, row::Symbol)
    e.unions[cid] === row && return e.data[cid]
    _cell_bind!(e, :cell_taken,
        Expr(:call, _cell_ref(:_cell_take_rows), e.data[cid], e.unions[cid], row))
end

# Computes every requested pushed value's window, once each: requests from the
# readers down (reverse plan order), then the windows in plan order.
function _cell_emit_windows!(e::_CellEmitter)
    d = e.pushdowns
    done = Set{ValueId}(keys(e.unions))
    processed = ValueId[]
    while true
        pending = [cid for cid in keys(e.requests) if !(cid in done)]
        isempty(pending) && break
        cid = pending[argmax([d.order[d.producer[c].id] for c in pending])]
        push!(done, cid)
        push!(processed, cid)
        _cell_union!(e, cid)
        _cell_request_inputs!(e, cid)
    end
    for cid in sort!(processed; by = c -> d.order[d.producer[c].id])
        e.data[cid] = _cell_window!(e, cid)
    end
    e
end

# The rows value `cid`'s window reads of each of its inputs, at its union rows
# `U`: requested of a pushed input, and recorded (`reads[cid][j]`, or per
# `vcat` operand) for the window to read.
function _cell_request_inputs!(e::_CellEmitter, cid::ValueId)
    d = e.pushdowns
    r = d.producer[cid]
    f = d.forms[r.id]
    U = e.unions[cid]
    if f.kind === :vcat
        starts = _cell_bind!(e, :cell_starts,
            Expr(:call, _cell_ref(:_cell_vcat_starts), _cell_vcat_part_shapes!(e, r, f)))
        e.reads[cid] = Any[starts; map(enumerate(f.parts)) do (k, part)
            part isa Int || return nothing
            local_rows = _cell_bind!(e, :cell_row,
                Expr(:call, _cell_ref(:_cell_vcat_part_rows), starts, k, U))
            input = canon_id(d.graph, r.inputs[part].id)
            input in d.pushed && _cell_request!(e, input, local_rows)
            local_rows
        end]
        return
    end
    reads = Any[nothing for _ in r.inputs]
    for (j, input) in enumerate(r.inputs)
        icid = canon_id(d.graph, input.id)
        role = f.roles[j]
        if role in (:leaf, :index, :column) && icid in d.pushed
            reads[j] = _cell_request!(e, icid, _cell_bind!(e, :cell_row,
                Expr(:call, _cell_ref(:_cell_leaf_row), _cell_shape!(e, icid), U)))
        elseif role in (:gather, :linear_gather) && icid in d.pushed
            # The rows every gather index reads (`_cell_gather_rows` joins
            # those of several indices).
            ats = map(f.indices[j]) do k
                index = canon_id(d.graph, r.inputs[k].id)
                at = _cell_operand_row!(e, index, U)
                f.vias[j] == 0 ? Expr(:call, _cell_ref(:_cell_gather_row), at, U) :
                    Expr(:call, _cell_ref(:_cell_via_rows),
                        e.names[canon_id(d.graph, r.inputs[f.vias[j]].id)], at)
            end
            at = length(ats) == 1 ? only(ats) :
                Expr(:call, _cell_ref(:_cell_gather_rows), ats...)
            role === :gather ||
                (at = Expr(:call, _cell_ref(:_cell_linear_row), _cell_shape!(e, icid), at))
            reads[j] = _cell_request!(e, icid, _cell_bind!(e, :cell_row, at))
        elseif role === :factor_leaf
            # The rows of the product `A * w` this operand broadcasts: rows of `A`.
            reads[j] = _cell_bind!(e, :cell_row, Expr(:call, _cell_ref(:_cell_leaf_row),
                _cell_part_shape!(e, r, f, j), U))
            icid in d.pushed && _cell_request!(e, icid, reads[j])
        elseif role === :factor
            reads[j] = U
            icid in d.pushed && _cell_request!(e, icid, U)
        elseif role === :range_source
            reads[j] = _cell_bind!(e, :cell_row,
                Expr(:call, _cell_ref(:_cell_range_rows), _cell_literal!(e, r, f), U))
            icid in d.pushed && _cell_request!(e, icid, reads[j])
        elseif role === :reshape_source
            reads[j] = _cell_bind!(e, :cell_row,
                Expr(:call, _cell_ref(:_cell_reshape_rows), _cell_literal!(e, r, f), U))
            icid in d.pushed && _cell_request!(e, icid, reads[j])
        end
    end
    e.reads[cid] = reads
end

# The window of pushed value `cid` at its union rows, by its own operation.
function _cell_window!(e::_CellEmitter, cid::ValueId)
    d = e.pushdowns
    r = d.producer[cid]
    f = d.forms[r.id]
    U = e.unions[cid]
    f.kind === :vcat && return _cell_vcat_window!(e, cid, r, f, U)
    if f.kind in (:pointwise, :identity, :columns, :hcat, :range_view, :reshape)
        push!(e.body.args, Expr(:call, _cell_ref(:_cell_check_row),
            _cell_shape!(e, cid), U))
    end
    reads = e.reads[cid]
    if f.kind in (:range_view, :reshape)
        # The source's positions, by the source's own window when it is pushed
        # and as a bounds-checked view of its whole value otherwise.
        j = only(f.leaves)
        icid = canon_id(d.graph, r.inputs[j].id)
        icid in d.pushed || _footprint_record!(icid, :rows, reads[j])
        positions = icid in d.pushed ? _cell_take!(e, icid, reads[j]) :
            Expr(:call, _cell_ref(:_cell_factor_rows), e.names[icid], reads[j])
        f.kind === :range_view && return _cell_bind!(e, :cell_rows, positions)
        return _cell_bind!(e, :cell_rows, Expr(:call, _cell_ref(:_cell_reshape_window),
            positions, U, _cell_literal!(e, r, f)))
    end
    args = Any[]
    for (j, input) in enumerate(r.inputs)
        icid = canon_id(d.graph, input.id)
        role = f.roles[j]
        if role in (:leaf, :index, :column)
            push!(args, icid in d.pushed ? _cell_take!(e, icid, reads[j]) :
                _cell_operand_row!(e, icid, U))
        elseif role in (:gather, :linear_gather) && icid in d.pushed
            shape = role === :gather ? nothing : _cell_shape!(e, icid)
            push!(args, _cell_bind!(e, :cell_window, Expr(:call,
                _cell_ref(:_CellWindow), e.data[icid], e.unions[icid], shape)))
        elseif role in (:gather, :linear_gather)
            # The whole source, read by the gather at its index's rows.
            for k in f.indices[j]
                at = _cell_operand_row!(e, canon_id(d.graph, r.inputs[k].id), U)
                f.vias[j] == 0 || (at = Expr(:call, _cell_ref(:_cell_via_rows),
                    e.names[canon_id(d.graph, r.inputs[f.vias[j]].id)], at))
                _footprint_record!(icid, role === :gather ? :gathered_rows : :gathered, at)
            end
            push!(args, e.names[icid])
        elseif role in (:factor_leaf, :factor)
            icid in d.pushed || _footprint_record!(icid, :rows, reads[j])
            push!(args, icid in d.pushed ? _cell_take!(e, icid, reads[j]) :
                Expr(:call, _cell_ref(:_cell_factor_rows), e.names[icid], reads[j]))
        else
            _footprint_record!(icid, :whole, nothing)
            push!(args, e.names[icid])
        end
    end
    if r.op isa _AuthoredPlateOp
        # The plate over these one-row arguments, with the table slots reserved
        # at its own position (`_lower_with_ops`).
        offset = d.slots[r.id]
        window = gensym(:cell_rows)
        _lower_authored_plate_native!(e.body, e.runtime_ops[1:offset],
            e.runtime_recipes[1:offset], r.op, args, d.plates[r.id], window, nothing)
        return window
    end
    call = Expr(:call, Expr(:ref, _OPS_ARG, d.slots[r.id]), args...)
    f.kind === :identity &&
        (call = Expr(:call, GlobalRef(Base, :convert), valtype(only(r.outputs)), call))
    _cell_bind!(e, :cell_rows, call)
end

# The rows `U` of a pushed `vcat`: each operand at the rows of it they hold
# (possibly none), by its own producer when it is pushed and as a view of its
# whole value otherwise, then those rows in the order read.
function _cell_vcat_window!(e::_CellEmitter, cid::ValueId, r::Recipe, f::_CellForm,
                            U::Symbol)
    d = e.pushdowns
    shape = _cell_shape!(e, cid)
    push!(e.body.args, Expr(:call, _cell_ref(:_cell_check_row), shape, U))
    starts, locals = e.reads[cid][1], e.reads[cid][2:end]
    parts = map(zip(f.parts, locals)) do (part, local_rows)
        part isa Int || return part
        input = canon_id(d.graph, r.inputs[part].id)
        input in d.pushed ? _cell_take!(e, input, local_rows) :
            _cell_operand_row!(e, input, local_rows)
    end
    _cell_bind!(e, :cell_rows, Expr(:call, _cell_ref(:_cell_vcat_rows), U, starts,
        Expr(:tuple, parts...), shape))
end

# The row a cell reads of pushed capture `cid`, at linear position
# `only(slice)` (the cell's batched element), or at `via[only(slice)]`; either
# is one position or an integer vector of them. Requested; read with
# `_cell_capture!` once the windows are emitted. A partition (`rows`) reads
# the positions every element of its slice holds, joined
# (`_cell_partition_positions`), or those `via` holds at them.
function _cell_capture_request!(e::_CellEmitter, cid::ValueId, slice, via = nothing;
                                rows::Bool = false)
    position = _cell_bind!(e, :cell_read, _cell_capture_element(slice, via; rows))
    _cell_request!(e, cid, _cell_bind!(e, :cell_row,
        Expr(:call, _cell_ref(:_cell_linear_row), _cell_shape!(e, cid), position)))
end

# The positions a cell reads of a captured value: its batched element (one
# position or an integer vector of them), or `via` at it; a partition's are
# those of every element of its slice, joined.
function _cell_capture_element(slice, via; rows::Bool)
    rows && return via === nothing ?
        Expr(:call, _cell_ref(:_cell_partition_positions), slice) :
        Expr(:call, _cell_ref(:_cell_via_rows), via, slice)
    only = Expr(:call, GlobalRef(Base, :only), slice)
    via === nothing ? only : Expr(:call, GlobalRef(Base, :getindex), via, only)
end

# The positions a partition's slice of a batched index argument holds: each
# element one position or an integer vector of them, joined in order.
@inline _cell_partition_positions(slice) =
    Int[row for q in slice for row in (q isa Integer ? (q,) : q)]

# The window a cell reads in place of pushed capture `cid`.
_cell_capture!(e::_CellEmitter, cid::ValueId) = _cell_bind!(e, :cell_capture,
    Expr(:call, _cell_ref(:_CellWindow), e.data[cid], e.unions[cid], _cell_shape!(e, cid)))
