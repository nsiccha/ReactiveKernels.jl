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
# row or several. A matrix read as `M[i, :]` or `M[i, 2]` is read at those rows
# too. A cell may read it through several batched elements (the
# outputs of one `@plate for` loop compose into the cell, each over a domain of
# its own): it reads the rows of each. When that array's producer computes it
# row by row, the native lowering runs the producer's own operation on those
# rows instead: the producer is given views of its row-aligned operands at
# those rows, and the cell is given a `_CellWindow` holding the result, which
# answers reads of those rows and throws `BoundsError` for any other. The
# rewrite applies to these source shapes, read off the closure's native body
# (`_kernel_native_body`):
#
# - a dotted expression (one fused broadcast) whose operands are recipe
#   inputs, numeric literals, numbers computed from them alone (`sqrt(2pi)`,
#   `1 / 2`), gathers `A[I]` / `A[I, 1, ...]` of an input by
#   an integer vector input `I` (integer literals after it), column reads
#   `A[:, 1, ...]` (also as `view(A, :, 1, ...)`, the view RKPPL reads a
#   per-observation value's column through), products `A * w` of two inputs,
#   and literal-position reads `a[1]` of an array of numbers (read whole);
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
#   array or a number (`_cell_product_shape`), or from its inferred vector
#   result and `A`'s row axis, including `A * (weights .* z)` and `A * W * z`;
#   a product without either shape proof is read only by gathers;
# - an authored plate: row `r` of its value is the plate over row `r` of each
#   batched argument, lowered natively over those one-row views at the reading
#   cell. A shared argument its body reads only as `v[q]`, `q` the element of
#   a batched integer argument (one or several), is a linear gather by those
#   arguments, as in a cell (`_cell_capture_reader`); its other shared
#   arguments are read whole.
#   Its full axes are its batched arguments' broadcast axes, as its loop
#   combines them;
# - a reduction over the axes other than the first, `sum(X; dims = 2)` (also
#   `prod`, `maximum`, `minimum`; `dims` a literal axis or tuple of axes, all
#   >= 2), alone, under `vec` (the reduced value then keeps one entry per row,
#   which the window checks) or under `dropdims(…; dims)` of reduced axes. Its
#   operand is an input or any broadcast operand above (a dotted expression, a
#   gather): row `r` of it is the same reduction over row `r` of the operand,
#   the shape BRM emits for a grouped HSGP field, `vec(sum(PHI .* rows;
#   dims = 2))` (snag `run-brm-s-full-g-2dc0667a`). A gather `A[I, :]` (every
#   later index a position or the whole axis `:`) reads whole rows of its
#   source, and `transpose(w)`, `adjoint(w)` or `w'` of a vector input, alone or
#   as a broadcast operand, is one row, read whole: broadcasting extrudes it
#   along the rows;
# - a call of a function whose owning package gives it a row method
#   (`_cell_row_method`): RKPPL's `_ppl_rows(x, n)`, whose row `r` is row `r` of
#   `x`, and its LKJ factors, whose row `r` reads their partials.
#
# Row `r` of the dotted expression is the same expression over row `r` of each
# array operand (the operand itself when broadcasting extrudes it), and row
# `r` of a gather by `I` reads row `I[r]` of its source; rows `R` read rows
# `R`, and `I[R]`. Those rows are again computed at those rows when their
# producers have one of these shapes, and are otherwise computed whole and
# indexed. A value is computed only at rows when every reader of it reads rows
# this way, so the program computes no value twice in different ways. The
# cell's batched element must be provably an integer (a linear position) or an
# integer vector, each ordinary gather index provably an integer vector, and each array
# read at literal positions provably an array of numbers: by its declared type,
# or by its producer's operands' types (`static_type` in `_cell_pushdowns`). A
# recipe that holds one of these shapes inside one expression, such as a
# gather from an inline `vcat`, is first split into recipes of its own along
# its authored structure (`_cell_outlined_pushdowns`, below).
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
    _CellWindow(data, rows, shape, sorted)

Rows of a value read by one cell. `rows` is one row (an `Int`), or the rows
its readers read (a `Vector{Int}`): a cell's in the order read, repeats
allowed; a partition's sorted and unique (`sorted`, `_cell_canonical_rows`),
found by bisection. `data` holds the value at those rows, in that order
along its first axis, every later dimension whole; `shape` is the value's
full axes, or `nothing` when only row reads `v[I, j...]` are made of it.
Reading any other row throws `BoundsError`.
"""
struct _CellWindow{D,R<:Union{Int,Vector{Int}},S,X}
    data::D
    rows::R
    shape::S
    sorted::Bool
    # The gather index this window was prepared for and its entries' positions
    # in `rows` (`_cell_window_positions`; a partition's plan computes them
    # once per label); `nothing` and empty when none is prepared.
    index::X
    positions::Vector{Int}
end
_CellWindow(data, rows, shape, sorted::Bool) = _CellWindow(data, rows, shape, sorted, nothing, Int[])

Base.axes(w::_CellWindow{<:Any,<:Any,<:Tuple}) = w.shape
Base.size(w::_CellWindow{<:Any,<:Any,<:Tuple}) = map(length, w.shape)
Base.length(w::_CellWindow{<:Any,<:Any,<:Tuple}) = prod(length, w.shape; init = 1)
Base.ndims(w::_CellWindow{<:Any,<:Any,<:Tuple}) = length(w.shape)
Base.axes(w::_CellWindow{<:Any,<:Any,<:Tuple}, d::Integer) = axes(LinearIndices(w.shape), d)
Base.size(w::_CellWindow{<:Any,<:Any,<:Tuple}, d::Integer) = size(LinearIndices(w.shape), d)
Base.firstindex(w::_CellWindow{<:Any,<:Any,<:Tuple}, args...) =
    firstindex(LinearIndices(w.shape), args...)
Base.lastindex(w::_CellWindow{<:Any,<:Any,<:Tuple}, args...) =
    lastindex(LinearIndices(w.shape), args...)

# The position along `data`'s first axis of row `row`; `at` is the read. The
# lookup reads the window's rows only, so native reverse differentiation sees
# index arithmetic apart from the data (below).
@inline function _cell_local_row(w::_CellWindow{<:Any,Int}, row, at)
    row == w.rows || throw(BoundsError(w.rows, at))
    firstindex(w.data, 1)
end
@inline function _cell_local_row(w::_CellWindow{<:Any,Vector{Int}}, row, at)
    firstindex(w.data, 1) + _cell_row_position(w.rows, row, at, w.sorted) - 1
end
# The position of `row` among `rows`: by bisection when they are sorted and
# unique, or directly when those rows are contiguous. Otherwise use the
# first match (a cell's few rows). A partition's lookup is at most O(log rows)
# per read, never a scan, so r row reads cost at most O(r log r), not O(r²).
@inline function _cell_row_position(rows::Vector{Int}, row, at, sorted::Bool)
    if sorted
        # Sorted partition rows are unique, so their span proves contiguity.
        # Ragged gathers of a contiguous window need only an offset, even
        # when each gather has its own index vector and no cached positions.
        if row isa Integer && !isempty(rows)
            lo, hi = first(rows), last(rows)
            if hi - lo == length(rows) - 1
                lo <= row <= hi || throw(BoundsError(rows, at))
                return Int(row - lo) + 1
            end
        end
        k = searchsortedfirst(rows, row)
        (k > length(rows) || @inbounds(rows[k]) != row) && throw(BoundsError(rows, at))
        return k
    end
    k = findfirst(==(row), rows)
    k === nothing && throw(BoundsError(rows, at))
    k
end
# The positions of rows `I`, computed before the data is read: a vector read
# is then one gather of the data at those positions. Reading the data row by
# row inside the position lookup instead (`[w[i, j] for i in I]`) made native
# Enzyme Reverse refuse the read with `EnzymeRuntimeActivityError`, the
# constant positions and the active data meeting in one loop.
@inline function _cell_local_rows(w::_CellWindow{<:Any,<:Any,<:Any,Nothing},
                                  I::AbstractVector{<:Integer}, at)
    positions = Vector{Int}(undef, length(I))
    # A partition often holds one contiguous block. Decide that once for the
    # gather: repeating the span proof inside every row lookup also repeats
    # its control flow in the ordinary reverse pass.
    if w.rows isa Vector{Int} && w.sorted && !isempty(w.rows)
        lo, hi = first(w.rows), last(w.rows)
        if hi - lo == length(w.rows) - 1
            offset = firstindex(w.data, 1) - lo
            for (k, i) in enumerate(I)
                lo <= i <= hi || throw(BoundsError(w.rows, at))
                positions[k] = Int(i) + offset
            end
            return positions
        end
    end
    for (k, i) in enumerate(I)
        positions[k] = _cell_local_row(w, i, at)
    end
    positions
end
# A window prepared for its gather index (`_cell_window!`: one direct index
# lane) holds the entries' positions, computed apart from the gather and once
# per label with a plan, and the read is by that index. The two cases are two
# methods: a fallback computing fresh positions here would merge a constant
# vector (the plan's) with a fresh one in one value, which native Reverse
# refuses (`EnzymeRuntimeActivityError`); a read by another index is an
# internal error instead.
@inline function _cell_local_rows(w::_CellWindow{<:Any,<:Any,<:Any,<:AbstractVector},
                                  I::AbstractVector{<:Integer}, at)
    I === w.index && length(w.positions) == length(I) || _cell_window_unprepared(at)
    w.positions
end
@noinline _cell_window_unprepared(at) = throw(ArgumentError(
    "internal: a cell window prepared for one gather index was read with another " *
    "($(summary(at)))"))

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

# Row reads `v[i, j...]` and gathers `v[I, j...]`, each later index a position
# or the whole axis `:` (`v[i, :]`, a row of a matrix).
@inline Base.getindex(w::_CellWindow, i::Integer, j::Union{Integer,Colon},
                      rest::Union{Integer,Colon}...) =
    w.data[_cell_local_row(w, i, (i, j, rest...)), j, rest...]
# A dotted call consumes this row before the slice can escape. Delegate to
# the same checked read-in-place path as a dense operand. Ordinary indexing
# above keeps its owning copy and concrete dispatch semantics.
@inline _lane_gathered(w::_CellWindow, i::Integer, j::Union{Integer,Colon},
                       rest::Union{Integer,Colon}...) =
    _lane_gathered(w.data, _cell_local_row(w, i, (i, j, rest...)), j, rest...)
@inline function Base.getindex(w::_CellWindow, I::AbstractVector{<:Integer},
                               j::Union{Integer,Colon}, rest::Union{Integer,Colon}...)
    _cell_reads_rows(w, I) && return w.data[:, j, rest...]
    w.data[_cell_local_rows(w, I, (I, j, rest...)), j, rest...]
end

# A read-only gather of a vector (`_readonly_gather`, core.jl, the view RKPPL
# reads a bare gathered definition through) reads the window's rows as the
# linear gather `w[I]` does; any other shape keeps `_readonly_gather`'s
# `MethodError`.
@inline _readonly_gather(w::_CellWindow{<:Any,<:Any,<:Tuple{Any}},
                         indices::AbstractVector{<:Integer}) = w[indices]
# `view(v, I)` of a vector has the linear gather's form (`_cell_form`): at a
# cell's rows it reads the window's rows `I` as that gather does. Values are
# read-only in a cell, so these rows serve as the view.
@inline Base.view(w::_CellWindow{<:Any,<:Any,<:Tuple{Any}},
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
# The same read in two steps, as the emitter makes it (`_cell_operand_row!`):
# the positions read of an operand of shape `shape` at `rows` (its own rows,
# or its one row repeated when broadcasting extrudes its first axis; `nothing`
# for a number), integer work on the shape alone, which a partition's plan
# computes once per label; then the view of the operand at those positions.
@inline _cell_slice_positions(::Tuple{}, ::Integer) = nothing
@inline _cell_slice_positions(::Tuple{}, ::Vector{Int}) = nothing
@inline _cell_slice_positions(shape::Tuple, row::Integer) =
    length(first(shape)) == 1 ? Int(first(first(shape))) : Int(row)
@inline _cell_slice_positions(shape::Tuple, rows::Vector{Int}) =
    length(first(shape)) == 1 ? fill(Int(first(first(shape))), length(rows)) : rows
# A view of `arg` at `positions` along its first axis, every later dimension
# whole. Not Base's `view`: that copies an index vector that may share memory
# with the array (`Base.unalias`), so a plan's constant row vector and a fresh
# copy would meet in one value, which native Reverse refuses
# (`EnzymeRuntimeActivityError`); a row vector of the lowering never aliases a
# value. Bounds are checked as `view` checks them.
@inline _cell_rows_view(arg, ::Nothing) = arg
@inline _cell_rows_view(arg::Tuple, positions::Vector{Int}) = [arg[p] for p in positions]
@inline _cell_rows_view(arg::Tuple, position::Integer) = (arg[position],)
@inline function _cell_rows_view(arg::AbstractArray{<:Any,N}, positions::Vector{Int}) where {N}
    rest = ntuple(_ -> Colon(), Val(N - 1))
    @boundscheck checkbounds(arg, positions, rest...)
    Base.unsafe_view(arg, Base.to_indices(arg, (positions, rest...))...)
end
@inline _cell_rows_view(arg::AbstractArray{<:Any,N}, position::Integer) where {N} =
    view(arg, position:position, ntuple(_ -> Colon(), Val(N - 1))...)

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

# The full axes of a gather `A[I, k...]` keeping the `Colon` axes of its tail
# (`A[I, :]`): the index's axis, then each kept axis of the source (an axis past
# the source's rank is the trailing singleton Julia's indexing allows).
@generated function _cell_gather_shape(index::Tuple, source::Tuple, ::Val{tail}) where {tail}
    n = fieldcount(source)
    axes = Any[:(first(index))]
    for (d, t) in enumerate(tail)
        t isa Colon || continue
        push!(axes, d + 1 <= n ? :(source[$(d + 1)]) : :(Base.OneTo(1)))
    end
    Expr(:tuple, axes...)
end

# The full axes of `transpose(w)` or `adjoint(w)` of a vector, one row; a
# number's are its own.
@inline _cell_transposed_shape(::Tuple{}) = ()
@inline _cell_transposed_shape(shape::Tuple{Any}) = (Base.OneTo(1), first(shape))

# The full axes of a reduction over the axes `dims` of a value with full axes
# `shape`: each reduced axis a singleton, the axes `drop` dropped (`dropdims`);
# under `vec`, the first axis alone, once every other axis is a singleton —
# otherwise `vec` interleaves the rows and the value has no row form.
@generated function _cell_reduce_shape(shape::Tuple, ::Val{dims}, ::Val{vec},
                                       ::Val{drop}) where {dims,vec,drop}
    n = fieldcount(shape)
    axes = Any[d in dims ? :(Base.OneTo(1)) : :(shape[$d]) for d in 1:n]
    if vec
        checks = Any[:(length(shape[$d]) == 1 || _cell_reduce_vec_error($d, length(shape[$d])))
                     for d in 2:n if !(d in dims)]
        return Expr(:block, checks..., Expr(:tuple, n == 0 ? :(Base.OneTo(1)) : axes[1]))
    end
    Expr(:tuple, Any[axes[d] for d in 1:n if !(d in drop)]...)
end
@noinline _cell_reduce_vec_error(d, len) = throw(ArgumentError(
    "`vec` of a reduction over the axes other than the first keeps one entry per row " *
    "only when every other axis of the reduced value has length 1; axis $d has length " *
    "$len, so the value has no row form (reduce over that axis too, or read the " *
    "reduced array without `vec`)"))

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
@inline function _cell_via_rows(w, qs)
    v = vec(w)
    _cell_joined_rows(qs) do q
        v[q]
    end
end

# The rows several gather indices read, joined in order (a tuple of single rows
# and vectors: the comprehension, which native Enzyme Reverse types; a sized
# loop over the mixed tuple does not pass its type analysis).
@inline _cell_gather_rows(ats...) =
    Int[row for at in ats for row in (at isa Integer ? (at,) : at)]

# `f` of each entry of a vector `entries` (every entry one row or several, of
# one type), joined in order into one vector sized first (a nested
# comprehension grows it row by row).
@inline function _cell_joined_rows(f::F, entries::AbstractVector) where {F}
    n = 0
    for q in entries
        rows = f(q)
        n += rows isa Integer ? 1 : length(rows)
    end
    out = Vector{Int}(undef, n)
    k = 0
    for q in entries
        rows = f(q)
        if rows isa Integer
            k += 1
            @inbounds out[k] = rows
        else
            for row in rows
                k += 1
                @inbounds out[k] = row
            end
        end
    end
    out
end

# The rows several readers ask a value for, each once: in first-asked order for
# a cell, sorted for a partition (its windows are read by bisection).
@inline _cell_union_rows(ats...) = unique!(_cell_gather_rows(ats...))
@inline _cell_union_rows_sorted(ats...) = unique!(sort!(_cell_gather_rows(ats...)))
# Rows as a partition's requests hold them: sorted and unique (the rows
# themselves when they already are), so one reader's request is its value's
# window as is and a value asked for at the same row by many cells is computed
# once (`_CellEmitter.sorted` wraps every derived request; the partition's own
# rows come sorted from its table). One row is itself.
@inline _cell_canonical_rows(row::Integer) = row
@inline function _cell_canonical_rows(rows::Vector{Int})
    for k in 2:length(rows)
        @inbounds rows[k - 1] < rows[k] || return unique!(sort(rows))
    end
    rows
end

# Rows `rows` of a value's window `data` computed at rows `union` (which hold
# them), in the order asked, every later dimension whole. A value without rows
# (a dotted expression of numbers is a number) reads the same at every row.
# The positions in `union` of rows `rows` (one or several), and of a gather
# index `I` of a window over a vector (its linear positions are its rows; a
# shaped source's are not, and get none). Integer work apart from the data,
# which a partition's plan computes once per label.
@inline _cell_take_positions(union::Vector{Int}, rows, sorted::Bool) =
    Int[_cell_row_position(union, row, row, sorted)
        for row in (rows isa Integer ? (rows,) : rows)]
@inline _cell_window_positions(union::Vector{Int}, I::AbstractVector{<:Integer}, sorted::Bool,
                               ::Union{Nothing,Tuple{Base.OneTo}}) =
    Int[_cell_row_position(union, i, i, sorted) for i in I]
@inline _cell_window_positions(::Vector{Int}, ::AbstractVector{<:Integer}, ::Bool, ::Tuple) = Int[]
# A single cell's window holds one row: every entry of the index is that row,
# at the window's first position.
@inline _cell_window_positions(union::Integer, I::AbstractVector{<:Integer}, ::Bool,
                               ::Union{Nothing,Tuple{Base.OneTo}}) =
    Int[(i == union || throw(BoundsError(union, i)); 1) for i in I]
@inline _cell_window_positions(::Integer, ::AbstractVector{<:Integer}, ::Bool, ::Tuple) = Int[]
@inline _cell_take_rows(data::Number, ::Vector{Int}) = data
@inline _cell_take_rows(data::AbstractArray{<:Any,0}, ::Vector{Int}) = data
@inline _cell_take_rows(data::AbstractArray{<:Any,N}, positions::Vector{Int}) where {N} =
    _cell_rows_view(data, positions)

# The source of a range view or a reshape at rows: a vector, its positions.
@inline _cell_source_rows(source::AbstractVector, rows) = _cell_factor_rows(source, rows)
@inline _cell_source_rows(source::AbstractArray, rows) = throw(ArgumentError(
    "a literal range or a reshape reads an array of $(ndims(source)) dimensions at a cell's " *
    "rows; only a vector is read this way"))
@inline _cell_vector_rows(rows::AbstractVector) = rows
@inline _cell_vector_rows(rows::AbstractArray) = throw(ArgumentError(
    "a literal range or a reshape reads an array of $(ndims(rows)) dimensions at a cell's " *
    "rows; only a vector is read this way"))

# Rows of a product's left factor; bounds-checked, never extruded.
@inline _cell_factor_rows(arg::AbstractArray{<:Any,N}, row::Integer) where {N} =
    view(arg, row:row, ntuple(_ -> Colon(), Val(N - 1))...)
@inline _cell_factor_rows(arg::AbstractArray{<:Any,N}, rows::Vector{Int}) where {N} =
    _cell_rows_view(arg, rows)

# A dotted expression's rows must lie in its full first axis.
@inline function _cell_check_row(shape::Tuple, row::Integer)
    valid = isempty(shape) ? row == 1 : checkindex(Bool, first(shape), row)
    valid || throw(BoundsError(CartesianIndices(shape), row))
    row
end
@inline _cell_check_row(shape::Tuple, row::Integer, sorted::Bool) = _cell_check_row(shape, row)
@inline function _cell_check_row(shape::Tuple, rows::Vector{Int}, sorted::Bool = false)
    if sorted && !isempty(rows)
        # Sorted rows: the extremes bound them all.
        _cell_check_row(shape, @inbounds(rows[1]))
        _cell_check_row(shape, @inbounds(rows[end]))
    else
        foreach(row -> _cell_check_row(shape, row), rows)
    end
    rows
end

# The rows of a computed operand that rows `row` of its reader need.
@inline _cell_leaf_row(shape::Tuple{}, row::Integer) = 1
@inline _cell_leaf_row(shape::Tuple, row::Integer) =
    length(first(shape)) == 1 ? Int(first(first(shape))) : row
@inline _cell_leaf_row(shape::Tuple, rows::Vector{Int}) =
    isempty(shape) || length(first(shape)) == 1 ?
        Int[_cell_leaf_row(shape, row) for row in rows] : rows      # the rows themselves

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

# A captured array's indices are still read by the authored cell body,
# possibly inside a lazy branch. Window construction requests only existing
# rows; it must not turn an inactive out-of-bounds read into an eager error.
# The window retains the full axes and the original indexing expression, so
# an active invalid read still throws at that read.
@inline function _cell_capture_rows(shape::Tuple, positions)
    domain = CartesianIndices(shape)
    Int[_cell_linear_row(shape, q) for q in
        (positions isa Integer ? (positions,) : positions) if checkbounds(Bool, domain, q)]
end

@inline function _cell_capture_via_rows(source, positions)
    _cell_joined_rows(positions) do q
        checkbounds(Bool, source, q) ? source[q] : ()
    end
end

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

# `vec(x)` of an array `x` with axes `shape`: position `u` is row
# `mod1(u, n)` of `x` at trailing position `cld(u, n)` (column-major), `n`
# its row count. The rows of `x` positions `U` read, sorted and unique; the
# window then reads each position from `x`'s rows (`_cell_vec_window`).
@inline _cell_vec_shape(shape::Tuple) = (Base.OneTo(prod(length, shape; init = 1)),)
@inline _cell_vec_rows(shape::Tuple, u::Integer) =
    isempty(shape) ? 1 : mod1(u, length(first(shape)))
@inline _cell_vec_rows(shape::Tuple, U::Vector{Int}) =
    isempty(shape) ? (isempty(U) ? Int[] : [1]) :
    sort!(unique!(Int[mod1(u, length(first(shape))) for u in U]))
@inline _cell_vec_window(data, R::Integer, u::Integer, shape::Tuple) =
    [_cell_vec_entry(data, 1, isempty(shape) ? 1 : cld(u, length(first(shape))))]
function _cell_vec_window(data, R::Vector{Int}, U::Vector{Int}, shape::Tuple)
    isempty(shape) && return [_cell_vec_entry(data, 1, 1) for _ in U]
    n = length(first(shape))
    [_cell_vec_entry(data, searchsortedfirst(R, mod1(u, n)), cld(u, n)) for u in U]
end
# Entry `t` (linear over the trailing axes) of window row `p`.
@inline _cell_vec_entry(data::Number, p::Int, t::Int) = data
@inline _cell_vec_entry(data::AbstractVector, p::Int, t::Int) =
    (t == 1 || throw(BoundsError(data, (p, t))); data[p])
@inline _cell_vec_entry(data::AbstractArray, p::Int, t::Int) =
    data[p, CartesianIndices(Base.tail(axes(data)))[t]]

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
# role `:part`. `vias[j][m]` is the shared index array an authored plate reads
# source `j` through at its `m`th index (`v[w[q]]`, `q` the element of index
# input `indices[j][m]`), 0 for a direct read; that index array has role `:via`.
# A `:program_gather` uses the scalar body's integer index program, held in
# `literal.programs`; its data dependencies are `literal.dependencies`.
# A batched `:program_lane` supplies index metadata before windows are built,
# so its producer remains whole while its selected rows feed the scalar body.
struct _CellForm
    kind::Symbol
    roles::Vector{Symbol}
    gather_index::Vector{Int}
    leaves::Vector{Int}
    operand::Int
    indices::Vector{Vector{Int}}
    parts::Vector{Any}
    vias::Vector{Vector{Int}}
    # A range/reshape's literal dimensions, or a plate's capture index programs.
    literal::Any
    # Per input, each gather's index-input position and trailing-axis pattern,
    # with integer indices represented by 1: `(1, Colon())` for `A[I, k, :]`.
    # The authored operation keeps the actual indices. A `Colon` keeps that
    # source axis in the gather's full shape (`_cell_gather_shape`).
    # `nothing` for an input that is not gathered.
    tails::Vector{Any}
    # A `:reduce`'s reduction `(; dims, vec, drop)`: the literal axes reduced,
    # whether `vec` follows, and the axes `dropdims` drops; else `nothing`.
    reduction::Any
    # An ordinary helper's recovered element program, emitted at these rows.
    rowmap::Any
end
_CellForm(kind, roles, gather_index, leaves, operand, indices, parts, vias,
          literal, tails, reduction) =
    _CellForm(kind, roles, gather_index, leaves, operand, indices, parts, vias,
              literal, tails, reduction, nothing)
_CellForm(kind, roles, gather_index, leaves, operand, indices = Vector{Int}[
        role in (:gather, :linear_gather) ? [gather_index[j]] : Int[]
        for (j, role) in enumerate(roles)], parts = Any[]) =
    _CellForm(kind, roles, gather_index, leaves, operand, indices, parts,
        _cell_direct_reads(indices), nothing)
_CellForm(kind, roles, gather_index, leaves, operand, indices, parts, vias) =
    _CellForm(kind, roles, gather_index, leaves, operand, indices, parts, vias, nothing)
_CellForm(kind, roles, gather_index, leaves, operand, indices, parts, vias, literal) =
    _CellForm(kind, roles, gather_index, leaves, operand, indices, parts, vias, literal,
        Any[nothing for _ in roles], nothing)

# Every index of every source read directly (no shared index array).
_cell_direct_reads(indices) = Vector{Int}[zeros(Int, length(ix)) for ix in indices]

# A package's own function with a row form its source does not show: `nothing`,
# or `(; at, rows, dims)` for the calls `f(args...)` of `f`.
#
# `at(U, args...)` returns the value at `U` (one row, or a vector of rows), with
# those rows along its first axis and every later axis whole. The arguments at
# the positions `rows` are given at `U`, as a broadcast reads them (a number stays
# a number, and a one-row operand repeats its row); every other argument is whole,
# so `at` may read all of it. Only `at` is called at rows: it needs none of `f`'s
# run-time tests of the whole value, and every call keeps one type.
#
# The value's full axes are `1:args[d]` for the positions `d` in `dims`, broadcast
# with the axes of the `rows` arguments. Each call argument is a recipe input or
# a literal; an argument nested in the source is outlined when the recipe can be
# split, and otherwise leaves the call without a form.
_cell_row_method(f) = nothing

# The row method of a recipe (`_CellForm.rowmap`): `at`, and the arguments of the
# authored call, each a literal or a `_CellInputLiteral`.
struct _CellRowMethod
    at::GlobalRef
    arguments::Vector{Any}
end

# The form of a call of a function with a row method, its arguments as
# `_cell_row_method` describes them; `nothing` when they do not fit.
function _cell_row_method_form(r::Recipe, f, arguments)
    method = _cell_row_method(f)
    method === nothing && return nothing
    at = GlobalRef(parentmodule(method.at), nameof(method.at))
    isdefined(at.mod, at.name) && getglobal(at.mod, at.name) === method.at || return nothing
    all(a -> a isa Union{_CellInputLiteral,Number}, arguments) || return nothing
    n = length(r.inputs)
    roles = fill(:whole, n)
    leaves = Int[]
    for k in method.rows
        arguments[k] isa _CellInputLiteral || return nothing
        input = arguments[k].input
        roles[input] = :leaf
        push!(leaves, input)
    end
    # An input read at rows cannot also be read whole by the same call.
    allunique(leaves) && all(enumerate(arguments)) do (k, a)
        k in method.rows || !(a isa _CellInputLiteral) || roles[a.input] === :whole
    end || return nothing
    literal = Tuple(arguments[d] for d in method.dims)
    all(part -> part isa Union{Int,_CellInputLiteral}, literal) || return nothing
    indices = Vector{Int}[Int[] for _ in 1:n]
    _CellForm(:rows, roles, zeros(Int, n), leaves, 0, indices, Any[],
        _cell_direct_reads(indices), literal, Any[nothing for _ in 1:n], nothing,
        _CellRowMethod(at, arguments))
end

# The type of a number computed from literals and named constants alone
# (`sqrt(2pi)`, `1 / 2`), inferred without running it, or `nothing`. Broadcasting
# extrudes such an operand, so it reads no row; its source is evaluated as
# authored. Any other parameter-free call may return an array (`ones(3)`) that a
# window could not broadcast against, and a call of no argument may depend on
# state (`rand()`): both stay without a row form.
function _cell_constant_number(ex, params, mod)
    ex isa Number && return typeof(ex)
    if ex isa Symbol || ex isa GlobalRef
        ex isa Symbol && ex in params && return nothing
        constant = ex isa GlobalRef ? isconst(ex.mod, ex.name) :
            mod isa Module && isdefined(mod, ex) && isconst(mod, ex)
        constant || return nothing
        value = _cell_binding(ex, mod)
        return value isa Number ? typeof(value) : nothing
    end
    Meta.isexpr(ex, :call) && length(ex.args) >= 2 &&
        !_native_callee_shadowed(ex.args[1], params) || return nothing
    f = _cell_binding(ex.args[1], mod)
    f isa Function || return nothing
    types = Any[]
    for arg in ex.args[2:end]
        T = _cell_constant_number(arg, params, mod)
        T === nothing && return nothing
        push!(types, T)
    end
    T = Base.promote_op(f, types...)
    isconcretetype(T) && T <: Number ? T : nothing
end

# Expression heads that bind no names; any other head declines the rewrite.
const _CELL_PLAIN_HEADS = (:call, :ref, :tuple, :., Symbol("'"), :vect, :hcat,
    :vcat, :row, :nrow, :ncat, :parameters, :kw, :comparison, :&&, :||, :if,
    :curly, :..., :string, :(::))

function _cell_source(r::Recipe; outlined::Bool = false, branches::Bool = false)
    closure = _lane_source_closure(r; outlined, branches)
    closure === nothing && return nothing
    _, params, mod = closure
    source = _lane_expand_dot(_lane_strip_lines(r.source), mod)
    source isa Expr && !_lane_has_macro(source) || return nothing
    (source, params, mod)
end

function _cell_native(r::Recipe; outlined::Bool = false, branches::Bool = false)
    analyzed = _cell_source(r; outlined, branches)
    analyzed === nothing && return nothing
    source, params, mod = analyzed
    native = _lane_strip_lines(_kernel_native_body(source, mod, Set{Symbol}(params)))
    (native, params, mod)
end

_cell_materialize(ex) = ex isa Expr && ex.head === :call && length(ex.args) == 2 &&
    ex.args[1] == GlobalRef(@__MODULE__, :_native_broadcast_materialize)
_cell_broadcasted(ex) = ex isa Expr && ex.head === :call && length(ex.args) >= 2 &&
    ex.args[1] in (GlobalRef(Base, :broadcasted),
                  GlobalRef(@__MODULE__, :_native_broadcasted))

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
    # A product over a bound sparse factor's rows (cell_sparse_products.jl): its
    # operand is read at the columns those rows store.
    r.op isa _CellSparseProduct &&
        return _CellForm(:sparse_product, [:sparse_rows, :sparse_operand], [0, 0], Int[], 2)
    r.source isa Symbol && return _cell_identity_form(r)
    if r.op === Base.:* && length(r.inputs) >= 2
        # A bare exact call `A * B...` keeps Base's `*` itself as its
        # operation (`_kernel_operation_body`); it is called positionally.
        n = length(r.inputs)
        roles = fill(:whole, n)
        roles[1] = :factor
        return _CellForm(:product, roles, zeros(Int, n), Int[], n == 2 ? 2 : 0)
    end
    # The column of a scalar read: row `r` of `A[:, j...]` is `A[r, j...]`.
    r.op isa _CellColumnRead && return _CellForm(:columns, [:leaf], [0], [1], 0)
    if (r.op === Base.transpose || r.op === Base.adjoint) && length(r.inputs) == 1
        # A bare exact call `transpose(w)` likewise keeps Base's function as its
        # operation: one row of a vector, read whole (a matrix is refused by
        # `operands_fit`).
        return _CellForm(:transposed, [:transposed], [0], Int[], 0)
    end
    if r.op === Base.reshape && length(r.inputs) >= 3
        # A bare exact call `reshape(v, n, k...)`, every dimension a distinct
        # input, keeps Base's `reshape` as its operation: the same form as
        # `reshape(v, n, 2)`, which `_cell_reshape` reads from its source.
        n = length(r.inputs)
        roles = fill(:whole, n)
        roles[1] = :reshape_source
        indices = Vector{Int}[Int[] for _ in 1:n]
        return _CellForm(:reshape, roles, zeros(Int, n), [1], 0, indices, Any[],
            _cell_direct_reads(indices), Tuple(_CellInputLiteral(j) for j in 2:n),
            Any[nothing for _ in 1:n], nothing)
    end
    # So does a bare exact call of a function with a row method.
    r.op isa _KernelSourceOp || return _cell_row_method_form(r, r.op,
        Any[_CellInputLiteral(j) for j in eachindex(r.inputs)])
    analyzed = _cell_native(r; outlined)
    analyzed === nothing && return nothing
    native, params, mod = analyzed
    checked = _cell_checked_form(r, native, params, mod)
    checked === nothing || return checked
    if Meta.isexpr(native, :call) && !_native_callee_shadowed(native.args[1], params)
        callee = _cell_binding(native.args[1], mod)
        if _cell_row_method(callee) !== nothing
            slots = Dict{Symbol,Int}(name => j for (j, name) in enumerate(params))
            return _cell_row_method_form(r, callee, Any[
                a isa Symbol && haskey(slots, a) ? _CellInputLiteral(slots[a]) : a
                for a in native.args[2:end]])
        end
    end
    position = Dict{Symbol,Int}(name => j for (j, name) in enumerate(params))
    uses = [Set{Symbol}() for _ in params]
    # A source's gather indices, in first-read order, and whether its gathers
    # are linear (`A[I]`) or row reads (`A[I, k...]`), the same for every read.
    gathers = Dict{Int,Tuple{Vector{Int},Bool}}()
    tails = Dict{Int,Vector{Tuple{Int,Tuple}}}()
    products = Dict{Int,Int}()
    leaves = Int[]
    param(x) = x isa Symbol && haskey(position, x)
    # `_readonly_gather(A, I)` and `view(A, I)` have the linear gather's shape.
    # Binding a range can turn `view(A, lo:hi)` into `view(A, bound_range)`.
    # `operands_fit` still requires integer-vector indices.
    readonly(ex) = ex isa Expr && ex.head === :call && length(ex.args) == 3 &&
        param(ex.args[3]) && !_native_callee_shadowed(ex.args[1], params) &&
        _cell_binding(ex.args[1], mod) in (_readonly_gather, Base.view)
    # `view(A, :, k...)`, RKPPL's in-place column read of a per-observation
    # value (`_observation_slice_views`): the column read `A[:, k...]`, as an
    # `Expr(:ref)`, or `nothing`.
    column_view(ex) = ex isa Expr && ex.head === :call && length(ex.args) >= 4 &&
        !_native_callee_shadowed(ex.args[1], params) &&
        _cell_binding(ex.args[1], mod) === Base.view && _cell_colon(ex.args[3], mod) ?
        Expr(:ref, ex.args[2], :(:), ex.args[4:end]...) : nothing
    function gather!(ex)
        length(ex.args) >= 2 || return false
        source, index = ex.args[1], ex.args[2]
        haskey(position, source) && haskey(position, index) || return false
        tail = _cell_gather_tail(ex.args[3:end], mod)
        tail === nothing && return false
        linear = length(ex.args) == 2
        found, linear_reads = get!(() -> (Int[], linear), gathers, position[source])
        linear_reads == linear || return false
        # Every gather contributes its own kept axes. Different integer
        # columns have identical shapes; their values stay in the source.
        pattern = map(t -> t isa Colon ? t : 1, tail)
        patterns = get!(Vector{Tuple{Int,Tuple}}, tails, position[source])
        part = (position[index], pattern)
        part in patterns || push!(patterns, part)
        position[index] in found || push!(found, position[index])
        push!(uses[position[source]], linear ? :linear_gather : :gather)
        push!(uses[position[index]], :index)
        push!(leaves, position[index])
        true
    end
    function operand!(ex)
        _cell_broadcasted(ex) && return lazy!(ex)
        # A broadcast a non-dotted parent materializes (`sum(a .* b; dims = 2)`).
        _cell_materialize(ex) && return operand!(ex.args[2])
        if ex isa Symbol
            haskey(position, ex) || return false
            push!(uses[position[ex]], :leaf)
            push!(leaves, position[ex])
            return true
        end
        ex isa Union{Int,Float64,Bool} && return true
        _cell_constant_number(ex, params, mod) === nothing || return true
        ex isa Expr && ex.head === :ref && return gather!(ex) || column!(ex) || scalar!(ex)
        readonly(ex) && return gather!(Expr(:ref, ex.args[2], ex.args[3]))
        column = column_view(ex)
        column === nothing || return column!(column)
        transposed = _cell_transposed(ex, position, mod)
        transposed === nothing || return transposed!(transposed)
        ex isa Expr && ex.head === :call && return product!(ex)
        false
    end
    # `transpose(w)`, `adjoint(w)` or `w'` of a vector: one row that broadcasting
    # extrudes along the rows, so `w` is read whole.
    function transposed!(j)
        push!(uses[j], :transposed)
        true
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
    reduction = nothing
    native = something(column_view(native), native)
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
    elseif (reduced = _cell_reduction(native, position, mod)) !== nothing
        # `sum(X; dims = 2)` (also `prod`, `maximum`, `minimum`), alone, under
        # `vec` or under `dropdims`: row `r` is the same reduction over row `r`
        # of `X`, an input or a broadcast operand (`operand!`).
        kind = :reduce
        reduction = reduced.spec
        operand!(reduced.operand) || return nothing
    elseif (transposed = _cell_transposed(native, position, mod)) !== nothing
        # `transpose(w)` of a vector input alone: one row, read whole.
        kind = :transposed
        push!(uses[transposed], :transposed)
    else
        return nothing
    end
    kind in (:pointwise, :reduce) && isempty(leaves) && return nothing
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
        operand, indices, parts, _cell_direct_reads(indices), literal,
        Any[get(tails, j, nothing) for j in eachindex(params)], reduction)
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

# Whether a `:range_view` recipe is the `view` call, not the copying `A[lo:hi]`.
function _cell_is_view_call(r::Recipe)
    analyzed = _cell_native(r; outlined = true)
    analyzed === nothing && return false
    native = first(analyzed)
    native isa Expr && native.head === :call
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

# The trailing indices of a gather `A[I, k...]`, each an integer literal or the
# whole axis `:` (`A[I, :]`, whole rows of a matrix), as a tuple of `Int`s and
# `Colon()`s; `nothing` for any other index.
function _cell_gather_tail(exs, mod)
    tail = Any[]
    for ex in exs
        if ex isa Int
            push!(tail, ex)
        elseif ex === :(:) || _cell_colon(ex, mod)
            push!(tail, Colon())
        else
            return nothing
        end
    end
    Tuple(tail)
end

# `transpose(w)`, `adjoint(w)` or `w'` of a parameter: its position, else `nothing`.
function _cell_transposed(ex, position, mod)
    ex isa Expr || return nothing
    if ex.head === Symbol("'") && length(ex.args) == 1
        arg = ex.args[1]
    elseif ex.head === :call && length(ex.args) == 2 &&
           (_cell_resolves_to(ex.args[1], mod, position, Base.transpose) ||
            _cell_resolves_to(ex.args[1], mod, position, Base.adjoint))
        arg = ex.args[2]
    else
        return nothing
    end
    arg isa Symbol && haskey(position, arg) ? position[arg] : nothing
end

# The reductions that keep an array's rows apart when they reduce its other
# axes: row `r` of `f(X; dims = d)`, `d` axes other than the first, is
# `f(X[r:r, …]; dims = d)`.
const _CELL_ROW_REDUCTIONS = (Base.sum, Base.prod, Base.maximum, Base.minimum)

# `f(X; dims = d)` with `f` one of `_CELL_ROW_REDUCTIONS` and `d` a literal
# axis or tuple of axes, all >= 2 — alone, under `vec(…)` (the reduced value
# keeps one entry per row when `d` covers every other axis, which the window
# checks, `_cell_reduce_shape`) or under `dropdims(…; dims = d′)` with
# `d′ ⊆ d`: `(; operand = X, spec = (; dims, vec, drop))`, else `nothing`.
function _cell_reduction(native, position, mod)
    native isa Expr && native.head === :call && !isempty(native.args) || return nothing
    vectorized = false
    drop = ()
    inner = native
    if _cell_resolves_to(native.args[1], mod, position, Base.vec)
        length(native.args) == 2 || return nothing
        vectorized = true
        inner = native.args[2]
    elseif _cell_resolves_to(native.args[1], mod, position, Base.dropdims)
        parts = _cell_call_parts(native)
        parts === nothing && return nothing
        positional, keywords = parts
        length(positional) == 1 && collect(keys(keywords)) == [:dims] || return nothing
        dropped = _cell_literal_dims(keywords[:dims])
        dropped === nothing && return nothing
        drop = dropped
        inner = only(positional)
    end
    inner isa Expr && inner.head === :call && !isempty(inner.args) || return nothing
    any(f -> _cell_resolves_to(inner.args[1], mod, position, f), _CELL_ROW_REDUCTIONS) ||
        return nothing
    parts = _cell_call_parts(inner)
    parts === nothing && return nothing
    positional, keywords = parts
    length(positional) == 1 && collect(keys(keywords)) == [:dims] || return nothing
    dims = _cell_literal_dims(keywords[:dims])
    dims === nothing && return nothing
    issubset(drop, dims) || return nothing
    (; operand = only(positional), spec = (; dims, vec = vectorized, drop))
end

# A call's positional arguments and its keyword arguments by name, written
# `f(x; k = v)` or `f(x, k = v)`; `nothing` for a splat or a repeated name.
function _cell_call_parts(ex::Expr)
    positional = Any[]
    keywords = Dict{Symbol,Any}()
    function keyword!(kw)
        Meta.isexpr(kw, :kw, 2) && kw.args[1] isa Symbol && !haskey(keywords, kw.args[1]) ||
            return false
        keywords[kw.args[1]] = kw.args[2]
        true
    end
    for arg in ex.args[2:end]
        if Meta.isexpr(arg, :parameters)
            all(keyword!, arg.args) || return nothing
        elseif Meta.isexpr(arg, :kw)
            keyword!(arg) || return nothing
        else
            push!(positional, arg)
        end
    end
    (positional, keywords)
end

# A literal reduction axis `d`, or tuple of axes `(d, …)`, every axis an integer
# literal >= 2 (the first axis holds the rows), as a tuple; else `nothing`.
function _cell_literal_dims(ex)
    ex isa Int && return ex >= 2 ? (ex,) : nothing
    Meta.isexpr(ex, :tuple) && !isempty(ex.args) &&
        all(d -> d isa Int && d >= 2, ex.args) || return nothing
    Tuple(Int.(ex.args))
end

# The operands of a form beyond its leaves whose axes shape its value: each a
# transposed vector input (`kind = :transposed`), or a gather of `input` by
# `index` keeping the `Colon` axes of its `tail` (`kind = :gather`). A gather
# dropping every trailing axis adds nothing: its index is a leaf.
function _cell_extra_parts(f::_CellForm)
    extra = NamedTuple[]
    for (j, role) in enumerate(f.roles)
        if role === :transposed
            push!(extra, (; kind = :transposed, input = j, index = 0, tail = ()))
        elseif role === :gather && f.tails[j] !== nothing
            for (k, tail) in f.tails[j]
                any(t -> t isa Colon, tail) || continue
                push!(extra, (; kind = :gather, input = j, index = k, tail))
            end
        end
    end
    extra
end

# The rank of an array type, `0` for a number; `nothing` when not known.
_cell_static_ndims(T) = T isa Type && T !== Union{} && T <: Number ? 0 :
    T isa Type && T <: AbstractArray && hasmethod(ndims, Tuple{Type{T}}) ? ndims(T) : nothing

# The rank of a form's broadcast operand from its inputs' static types: the
# largest rank among its parts; `nothing` when any part's rank is unknown.
function _cell_operand_rank(r::Recipe, f::_CellForm, static_type)
    rank = 0
    for j in f.leaves
        role = f.roles[j]
        n = if role === :column
            1
        elseif role === :factor_leaf
            W = static_type(r.inputs[f.gather_index[j]])
            W isa Type && W <: AbstractVector ? 1 : W isa Type && W <: AbstractMatrix ? 2 :
                W isa Type && W <: Number ? _cell_static_ndims(static_type(r.inputs[j])) : nothing
        elseif role === :index
            1
        else
            _cell_static_ndims(static_type(r.inputs[j]))
        end
        n === nothing && return nothing
        rank = max(rank, n)
    end
    for part in _cell_extra_parts(f)
        n = part.kind === :transposed ?
            let T = static_type(r.inputs[part.input])
                T isa Type && T <: Number ? 0 : T isa Type && T <: AbstractVector ? 2 : nothing
            end : 1 + count(t -> t isa Colon, part.tail)
        n === nothing && return nothing
        rank = max(rank, n)
    end
    rank
end

# Whether a `:reduce` form under `vec` keeps one entry per row as far as its
# operand's rank is known statically: every axis past the first is reduced.
# An unknown rank passes here; the window checks the shape it computes.
function _cell_reduce_fits(r::Recipe, f::_CellForm, static_type)
    f.kind === :reduce && f.reduction.vec || return true
    rank = _cell_operand_rank(r, f, static_type)
    rank === nothing || all(d -> d in f.reduction.dims, 2:rank)
end

# A checked alias `f(value, others...)` (`_footprint_checked_alias`) whose
# owner supplies a validator (`_checked_alias_validator`): its rows are its
# value argument's rows, and the validator checks the whole argument from its
# type and axes once per evaluation, so the check still runs and no other row
# is read. Every argument must be one input, the value only once.
function _cell_checked_form(r::Recipe, native, params, mod)
    Meta.isexpr(native, :call) && !_native_callee_shadowed(native.args[1], params) ||
        return nothing
    f = _cell_binding(native.args[1], mod)
    k = _footprint_checked_alias(f)
    k > 0 || return nothing
    validator = _checked_alias_validator(f)
    validator === nothing && return nothing
    reference = GlobalRef(parentmodule(validator), nameof(validator))
    isdefined(reference.mod, reference.name) &&
        getglobal(reference.mod, reference.name) === validator || return nothing
    arguments = native.args[2:end]
    length(arguments) >= k && all(a -> a isa Symbol && a in params, arguments) ||
        return nothing
    slots = Int[findfirst(==(a), params) for a in arguments]
    alias = slots[k]
    count(==(alias), slots) == 1 || return nothing
    n = length(r.inputs)
    roles = fill(:whole, n)
    roles[alias] = :leaf
    indices = Vector{Int}[Int[] for _ in 1:n]
    _CellForm(:checked, roles, zeros(Int, n), [alias], 0, indices, Any[],
        _cell_direct_reads(indices),
        (; validator = reference, others = Int[slots[i] for i in eachindex(slots) if i != k]))
end

# An identity `b::T = a` (a typed alias RK keeps as a recipe): its rows are
# its input's rows, converted to the declared type as the typed local the
# whole program assigns converts them (`_declare_typed_output!`). Only a
# declaration that converts no elements (`AbstractVector`, `Any`) qualifies:
# it holds for the rows exactly when it holds for the whole value.
function _cell_identity_form(r::Recipe)
    op = r.op
    op isa _KernelSourceOp && kernel_sourceop_form(op) === :fused &&
        getfield(op, :ignored_throws) !== nothing && length(r.inputs) == 1 || return nothing
    params = _lane_source_params(_lane_source_callable(getfield(op, :f)), r)
    params === nothing && return nothing
    name = only(params)
    # A gensym parameter is stored without its `##` prefix (`_lane_source_arguments`).
    r.source in (name, Symbol("##", name)) || return nothing
    T = valtype(only(r.outputs))
    T === Any || T isa Type && T <: AbstractArray && eltype(T) === Any || return nothing
    _CellForm(:identity, [:leaf], [0], [1], 0)
end

# An authored plate's value, row by row: its batched arguments are broadcast
# operands, a shared argument read only at batched integer arguments'
# elements, or through a shared index array at them (`v[w[q]]`, as a total
# over a group of rows reads them), is gathered by them, and any other shared
# argument is read whole.
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
    indices = Vector{Int}[Int[] for _ in 1:n]
    vias = Vector{Int}[Int[] for _ in 1:n]
    wants = Set(canon_id(ig, w.id) for w in inner.want)
    reads = Dict{Int,Vector{Tuple{Int,Int}}}()
    for k in atomic
        roots[k] in wants && continue
        found = _cell_capture_reader(inner, roots, atomic, roots[k])
        found === nothing || (reads[k] = found)
    end
    # An index array is read whole; it is not also gathered.
    through = Set(w for found in values(reads) for (_, w) in found if w != 0)
    for (k, found) in reads
        k in through && continue
        roles[k] = :linear_gather
        gather_index[k] = first(first(found))
        for (b, w) in found
            roles[b] = :index
            push!(indices[k], b)
            push!(vias[k], w)
        end
    end
    foreach(w -> roles[w] = :via, through)
    programs = Dict{Int,Any}()
    dependencies = Dict{Int,Set{Int}}()
    for k in atomic
        roles[k] === :whole || continue
        program = _cell_capture_index_program(inner, k)
        program === nothing && continue
        programs[k] = program
        dependencies[k] = _cell_index_dependencies(program)
    end
    # Index metadata must remain available before producer windows are built.
    # A capture used to find another capture's indices is therefore not pushed.
    metadata = Set(j for (k, js) in dependencies for j in js if j != k)
    for k in collect(keys(programs))
        k in metadata && (delete!(programs, k); delete!(dependencies, k))
    end
    if !isempty(programs)
        metadata = Set(j for (k, js) in dependencies for j in js if j != k)
        for j in metadata
            roles[j] = j in atomic ? :whole : :program_lane
        end
        foreach(k -> roles[k] = :program_gather, keys(programs))
    end
    _CellForm(:pointwise, roles, gather_index, leaves, 0, indices, Any[], vias,
        (; programs, dependencies))
end

# The type of an authored plate's value from its input types: an array of its
# cell's result type, typed from a walk of its body as the generated body
# types the buffer it allocates (`_checked_plate_element!`, `_plate_similar_output`).
# Preserve the buffer's rank when its batched inputs prove it; a product of
# the plate's weights then keeps its vector shape through an inline broadcast.
# An unknown buffer keeps only the known element type, or `Any` if neither is known.
function _cell_plate_output_type(op::_AuthoredPlateOp, input_types)
    all(T -> T isa Type, input_types) || return Any
    element = _checked_plate_element!(_UninferredValue[], op, input_types, Symbol[])
    element isa Type && element !== Union{} || return Any
    atomic = Val(typeof(op).parameters[2])
    if element === Any
        # A standard array/tuple axis still proves the buffer's rank when an
        # abstract live operand hides the cell's element type. Narrowing an
        # Array's elements after the loop preserves that rank. A custom
        # similar method may depend on the element type, so needs inference.
        A = typeof(op).parameters[2]
        batched = Tuple(input_types[i] for i in eachindex(input_types) if !(i in A))
        _plate_marker_type(batched) <: Union{Array,Tuple} || return Any
        ranks = map(T -> T <: Tuple ? 1 : _cell_static_ndims(T), batched)
        any(isnothing, ranks) && return Any
        return AbstractArray{<:Any,maximum(ranks)}
    end
    array = _plate_array_type(element, atomic, input_types...)
    # This is the actual allocation type, before any abstract-array stand-in.
    # Retain it so type-only branches can be proved without assuming that an
    # arbitrary AbstractArray uses Base's eltype or other methods.
    isconcretetype(array) && return array
    # Abstract array containers may hide `similar`'s result, as they hide a
    # broadcast's. The same element/rank stand-ins used below recover rank.
    array === Any && (array = _plate_array_type(element, atomic,
        Any[_cell_standin_type(T) for T in input_types]...))
    shaped = _cell_element_type(array)
    shaped !== Any && return shaped
    isconcretetype(element) ? AbstractArray{element} : AbstractArray{<:element}
end

# A concrete array of an abstract array type's element type and rank, which
# Base computes the same elements from; any other type unchanged.
_cell_standin_type(T) = T isa DataType && T <: AbstractArray && !isconcretetype(T) &&
    isconcretetype(eltype(T)) ? Array{eltype(T),ndims(T)} : T

# What a value computed from such stand-ins is known to be: its element type
# and rank (its container follows from its operands'), a number, or `Any`.
_cell_element_type(S) =
    S isa DataType && S <: AbstractArray && isconcretetype(eltype(S)) ?
        AbstractArray{eltype(S),ndims(S)} :
    S isa Type && S !== Union{} && S <: Number ? S : Any

# A matrix transpose is not a row form: its rows read columns of its source.
# It still preserves numeric element/rank information. A transpose of an
# expression with a known row form may therefore use the same stand-ins for
# type inference, without acquiring permission to compute at rows. This
# matters for a product's shared right factor, e.g. transpose(scale .* L).
function _cell_transpose_standin(r::Recipe; outlined::Bool = false)
    analyzed = _cell_native(r; outlined)
    analyzed === nothing && return false
    native, params, mod = analyzed
    Meta.isexpr(native, :call, 2) &&
        !_native_callee_shadowed(native.args[1], params) &&
        _cell_binding(native.args[1], mod) in (Base.transpose, Base.adjoint) || return false
    Meta.isexpr(r.source, :call, 2) || return false
    operand = Recipe(r.id, r.inputs, r.outputs, r.op, r.cost, r.cse_key,
                     r.effectful, r.source.args[2])
    _cell_form(operand; outlined) !== nothing
end

# One type analysis for row demand, scalar reads and locality reporting. Keep
# array rank: forgetting it here also loses the scalar type of `exp(a[1])`.
# Stand-ins are used for row forms and their transposes, whose element
# operations and axes are known, never to infer an arbitrary function's
# behavior on an abstract array. Exact exemplar types, when supplied by
# preparation, survive abstract port declarations that already contain them.
function _cell_value_types(p::Plan; argument_types = nothing,
                           outlined = Set{Int}(), known = IdDict{Recipe,Any}())
    argument_types === nothing && (argument_types = Any[valtype(v) for v in p.have])
    bound = Set(canon_id(p.graph, only(r.outputs).id) for r in p.recipes
                if r.op isa _BoundConstant)
    _plan_value_types(p, argument_types) do r, input_types
        r.op isa _BoundConstant && return (typeof(r.op.value),)
        inputs = Any[valtype(v) !== Any && !(T <: valtype(v)) &&
                     !(canon_id(p.graph, v.id) in bound) ?
                     valtype(v) : T for (v, T) in zip(r.inputs, input_types)]
        form = get!(() -> _cell_form(r; outlined = r.id in outlined), known, r)
        if form === nothing
            form = _cell_arithmetic_form(r, inputs; outlined = r.id in outlined)
            form === nothing &&
                (form = _cell_helper_form(r, inputs; outlined = r.id in outlined))
            form === nothing &&
                (form = _cell_position_form(r, inputs; outlined = r.id in outlined))
            form === nothing && (form = _cell_vec_form(r, inputs; outlined = r.id in outlined))
            known[r] = form
        end
        inferred = if length(r.outputs) == 1 && r.op isa _AuthoredPlateOp
            (_cell_plate_output_type(r.op, inputs),)
        elseif length(r.outputs) == 1 && r.op isa _AuthoredPlateCellOp && !r.op.rows
            (_checked_plate_element!(_UninferredValue[], r.op.plate,
                                     inputs[2:end], Symbol[]),)
        else
            _recipe_inferred_output_types(r, inputs)
        end
        # A checked literal array read has the array's element type even when
        # inference through a partial-evaluation closure loses that fact.
        literal = length(r.outputs) == 1 ? _footprint_literal_read(r) : nothing
        fallback = nothing
        map(eachindex(r.outputs)) do k
            declared = valtype(r.outputs[k])
            T = inferred[k]
            declared !== Any && !(T <: declared) && return declared
            if T === Any && literal !== nothing
                S = inputs[first(literal)]
                S isa Type && S <: AbstractArray && eltype(S) <: Number && return eltype(S)
            end
            if T === Any && !(r.op isa _AuthoredPlateOp)
                f = get!(() -> _cell_form(r; outlined = r.id in outlined), known, r)
                if f !== nothing || _cell_transpose_standin(r; outlined = r.id in outlined)
                    fallback === nothing && (fallback = _recipe_inferred_output_types(r,
                        Any[_cell_standin_type(S) for S in inputs]))
                    T = _cell_element_type(fallback[k])
                end
            end
            T === Union{} ? Any : T
        end
    end
end

# `*` as the closure resolves it is Base's, or the native body's companion
# for it (`_native_mul`, core.jl).
function _cell_base_product(callee, mod, params)
    callee isa GlobalRef && return isdefined(callee.mod, callee.name) &&
        getglobal(callee.mod, callee.name) in (Base.:*, _native_mul)
    callee === :* && !(callee in params) && mod isa Module || return false
    isdefined(mod, :*) && getglobal(mod, :*) === Base.:*
end

# The value a callee or argument names in the closure's module: a `GlobalRef`,
# or a bare or qualified name (`_kernel_resolve_binding`); `nothing` when it
# names no binding.
_cell_binding(ex::GlobalRef, mod) =
    isdefined(ex.mod, ex.name) ? getglobal(ex.mod, ex.name) : nothing
_cell_binding(ex, mod) = mod isa Module ? _kernel_resolve_binding(mod, ex) : nothing

# Whether a call argument is the whole-axis index `:` (`Colon()`): the bare or
# resolved `:`, or `Colon()` called with no arguments.
_cell_colon(ex, mod) = _cell_binding(ex, mod) === Colon() ||
    Meta.isexpr(ex, :call, 1) && _cell_binding(ex.args[1], mod) === Colon

# The positions `k` of the cell's shared arguments that its body reads only as
# `v[q]` or `v[w[q]]` (or `v[q, :]`, a row), each `q` the element of a batched
# argument `b` and `w` another shared argument, as `k => [(b, w), ...]`
# (`w = 0` for `v[q]`), one pair per distinct read (`_cell_capture_reader`).
# Each element `q`, or `w[q]`, is a linear position or an integer vector of
# them. The second value lists the positions read that way whose index is not
# provably one: those are read whole, and the locality report names them.
function _cell_capture_positions(r::Recipe, static_type; branch_read = nothing)
    plate = (r.op::_AuthoredPlateCellOp).plate
    atomic = typeof(plate).parameters[2]
    inner = plate.kernel.plan
    ig = inner.graph
    roots = ValueId[canon_id(ig, v.id) for v in inner.have]
    captures = Dict{Int,Vector{Tuple{Int,Int}}}()
    untyped = Int[]
    length(roots) == length(r.inputs) - 1 && allunique(roots) || return captures, untyped
    wants = Set(canon_id(ig, w.id) for w in inner.want)
    for k in atomic
        roots[k] in wants && continue
        reads = _cell_capture_reader(inner, roots, atomic, roots[k];
            branch_read = branch_read === nothing ? nothing :
                (b, via, tail) -> branch_read(k, b, via, tail))
        reads === nothing && continue
        typed = all(reads) do (b, via)
            lane = static_type(r.inputs[1 + b])
            via == 0 ? _cell_positions_type(lane) :
                lane <: AbstractArray{<:Integer} &&
                    _cell_positions_type(static_type(r.inputs[1 + via]))
        end
        typed || (push!(untyped, k); continue)
        captures[k] = reads
    end
    captures, untyped
end

# An array whose elements are linear positions, or integer vectors of them.
_cell_positions_type(T) =
    T <: AbstractArray{<:Integer} || T <: AbstractArray{<:AbstractVector{<:Integer}}

# An ordinary gather takes one integer per output row. A plate's element
# can itself hold a vector of positions, including a bound per-cell cache.
# A Boolean vector is a logical mask, not a vector of linear positions.
_cell_index_type(op, T) = T <: AbstractVector{<:Integer}
_cell_index_type(::_AuthoredPlateOp, T) = T <: AbstractVector{<:Integer} ||
    T <: AbstractVector{<:AbstractVector{<:Integer}} &&
    typeintersect(eltype(eltype(T)), Bool) === Union{}

# Resolve an unchanged outer-cell argument, including a linear index alias.
function _cell_capture_root(inner::Plan, roots, input)
    ig = inner.graph
    cid = canon_id(ig, input.id)
    k = findfirst(==(cid), roots)
    k === nothing || return k
    # `vec` preserves linear positions, including when the alias crosses a
    # scan's shared-argument boundary.
    for producer in inner.recipes
        length(producer.outputs) == 1 &&
            canon_id(ig, only(producer.outputs).id) == cid || continue
        if producer.op === Base.vec && length(producer.inputs) == 1
            return findfirst(==(canon_id(ig, only(producer.inputs).id)), roots)
        end
        analyzed = _cell_native(producer)
        analyzed === nothing && return nothing
        native, params, mod = analyzed
        length(params) == 1 && Meta.isexpr(native, :call, 2) &&
            native.args[2] === only(params) &&
            !_native_callee_shadowed(native.args[1], params) &&
            _cell_binding(native.args[1], mod) === Base.vec || return nothing
        return findfirst(==(canon_id(ig, only(producer.inputs).id)), roots)
    end
    nothing
end

# Child formals that receive their parent's input unchanged. A scan's carry,
# iterated elements and history are deliberately absent: they are not the
# outer cell's indices or captures. A step may never run, so its capture
# reads need the same bound-index proof as a lazy branch.
_cell_capture_body(op, n) = nothing
function _cell_capture_body(op::_AuthoredScanOp, n)
    atomic = typeof(op).parameters[2]
    last = n - _scan_has_history(op)
    iterated = count(i -> !(i in atomic), 2:last)
    shared = [i for i in 2:last if i in atomic]
    pairs = [1 + iterated + j => i for (j, i) in enumerate(shared)]
    (op.kernel.plan, pairs, true)
end

# The compiler's gather-map interchange preserves its authored arm's read set
# by construction (_rk_gather_cover). Keep that read proof when analysing a
# capture: the generated _rk_gather_source call otherwise looks like a whole
# argument use, forcing even an absent observation's producer to execute.
# Numerical lowering still runs the full guarded source. No authored branch
# acquires this equivalence merely by having the same expression shape.
function _cell_capture_native(r::Recipe; branches::Bool = false)
    analyzed = _cell_native(r; branches)
    analyzed === nothing && return nothing
    r.op.f isa _KernelGatherMap || return analyzed
    _, params, mod = analyzed
    source = _lane_expand_dot(_lane_strip_lines(r.source), mod)
    Meta.isexpr(source, :if, 3) || return nothing
    native = _lane_strip_lines(_kernel_native_body(source.args[3], mod, Set{Symbol}(params)))
    (native, params, mod)
end

# How the plate body reads shared argument `cid`: only as `v[q]` or `v[w[q]]`,
# `q` the element of a batched argument `b`, `w` one other shared argument.
# Each distinct read `(b, w)`, `w = 0` for `v[q]`, in first-read order, or
# `nothing`. Nested scans retain the outer root numbering for unchanged
# shared arguments; their carried and iterated values acquire no root.
# A value read through several batched arguments is read at the rows of each.
function _cell_capture_reader(inner::Plan, roots, atomic, cid;
        branch_read = nothing, guarded = false)
    ig = inner.graph
    any(v -> canon_id(ig, v.id) == cid, inner.want) && return nothing
    found = Tuple{Int,Int}[]
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
        _cell_capture_root(inner, roots, recipe.inputs[j])
    end
    for recipe in inner.recipes
        slots = findall(_input_list(v -> canon_id(ig, v.id) == cid, recipe.inputs))
        isempty(slots) && continue
        nested = _cell_capture_body(recipe.op, length(recipe.inputs))
        if nested !== nothing
            child, pairs, conditional = nested
            all(s -> any(p -> last(p) == s, pairs), slots) || return nothing
            child_roots = Union{Nothing,ValueId}[nothing for _ in roots]
            for (j, s) in pairs
                k = _cell_capture_root(inner, roots, recipe.inputs[s])
                k === nothing && continue
                id = canon_id(child.graph, child.have[j].id)
                child_roots[k] in (nothing, id) || return nothing
                child_roots[k] = id
            end
            k = findfirst(==(cid), roots)
            k === nothing && return nothing
            reads = _cell_capture_reader(child, child_roots, atomic, child_roots[k];
                branch_read, guarded = guarded || conditional)
            reads === nothing && return nothing
            for read in reads
                read in found || push!(found, read)
            end
            continue
        end
        recipe.op isa _KernelSourceOp || return nothing
        branch = recipe.op.f isa _KernelBranch
        analyzed = _cell_capture_native(recipe; branches = branch_read !== nothing)
        analyzed === nothing && return nothing
        native, params, mod = analyzed
        reads = Any[]
        tails = Any[]
        _cell_capture_reads!(reads, native, Set(params[slots]), mod;
            tails = branch || guarded ? tails : nothing) || return nothing
        for (j, read) in enumerate(reads)
            q, via = read isa Symbol ? (read, nothing) : (read.args[2], read.args[1])
            b = root(params, recipe, q)
            (b === nothing || b in atomic) && return nothing
            w = via === nothing ? 0 : via_root(params, recipe, via)
            (w === nothing || (w != 0 && !(w in atomic)) || w == b) && return nothing
            # A branch may not force an inactive index lookup: both the
            # optional shared lookup and its result must be bound and valid.
            if branch || guarded
                branch_read !== nothing && branch_read(b, w, tails[j]) || return nothing
            end
            (b, w) in found || push!(found, (b, w))
        end
    end
    isempty(found) ? nothing : found
end

# Whether `ex` reads the names `captured` only as `v[q]` or `v[w[q]]`, with `q`
# and `w` names, or as `v[q, j...]` / `v[w[q], j...]` with every later index the
# whole axis `:` (a row of a matrix) or a literal position (an element of that
# row); each such `q`, or `w[q]`, is recorded.
function _cell_capture_reads!(reads, ex, captured, mod; tails = nothing)
    ex isa Symbol && return !(ex in captured)
    ex isa Expr || return true
    ex.head in _CELL_PLAIN_HEADS || return false
    if ex.head === :ref && length(ex.args) >= 2 && ex.args[1] isa Symbol &&
       ex.args[1] in captured
        index = ex.args[2]
        name(x) = x isa Symbol && !(x in captured)
        name(index) || Meta.isexpr(index, :ref, 2) && all(name, index.args) ||
            return false
        all(arg -> _cell_colon(arg, mod) || arg isa Int, ex.args[3:end]) || return false
        push!(reads, index)
        tails === nothing || push!(tails, Any[_cell_colon(arg, mod) ? Colon() : arg
                                             for arg in ex.args[3:end]])
        return true
    end
    all(arg -> _cell_capture_reads!(reads, arg, captured, mod; tails), ex.args)
end

# Retained branches may request both arms' capture rows only when those
# requests cannot introduce an inactive invalid access. Prove their indices
# from bound data and their full shapes from the existing row forms. No arm,
# condition, active parameter or captured producer is evaluated here. The
# producers of a capture are strict, outside the retained branch already;
# computing their in-range rows adds no branch-arm work. Unknown shapes or
# indices keep the ordinary lazy whole-capture path.
function _cell_bound_capture_check(p::Plan, producers, static_type, known, outlined)
    g = p.graph
    cidof(v) = canon_id(g, v.id)
    producer(cid) = let rs = get(producers, cid, Recipe[])
        length(rs) == 1 ? only(rs) : nothing
    end
    bound(v) = let r = producer(cidof(v))
        r !== nothing && r.op isa _BoundConstant ? Some(r.op.value) : nothing
    end
    shapes = Dict{ValueId,Any}()
    function shape(v)
        cid = cidof(v)
        haskey(shapes, cid) && return shapes[cid]
        shapes[cid] = nothing
        value = bound(v)
        if value !== nothing
            return shapes[cid] = axes(Base.broadcastable(something(value)))
        end
        static_type(v) <: Number && return shapes[cid] = ()
        r = producer(cid)
        r === nothing && return nothing
        f = get!(() -> _cell_form(r; outlined = r.id in outlined), known, r)
        f === nothing && return nothing
        if f.kind in (:range_view, :reshape)
            parts = Any[]
            for part in f.literal
                x = part isa Int ? Some(part) : bound(r.inputs[part.input])
                x !== nothing && something(x) isa Integer || return nothing
                push!(parts, something(x))
            end
            literal = Tuple(parts)
            return shapes[cid] = f.kind === :range_view ?
                _cell_range_shape(literal) : _cell_dims_shape(literal)
        end
        if f.kind === :vec
            s = shape(r.inputs[only(f.leaves)])
            return shapes[cid] = s === nothing ? nothing : _cell_vec_shape(s)
        end
        if f.kind === :rows
            parts = Any[]
            for part in f.literal
                x = part isa Int ? Some(part) : bound(r.inputs[part.input])
                x !== nothing && something(x) isa Integer || return nothing
                push!(parts, something(x))
            end
            full = Any[_cell_dims_shape(Tuple(parts))]
            for j in f.leaves
                s = shape(r.inputs[j])
                s === nothing && return nothing
                push!(full, s)
            end
            return shapes[cid] = _plate_broadcast_shape(Tuple(full))
        end
        f.kind in (:identity, :checked, :pointwise, :columns, :hcat, :reduce, :transposed,
                   :array_arithmetic, :position_map) ||
            return nothing
        parts = Any[]
        for j in f.leaves
            s = shape(r.inputs[j])
            s === nothing && return nothing
            f.roles[j] === :factor_leaf && return nothing
            push!(parts, f.roles[j] === :column ? _cell_column_shape(s) : s)
        end
        for part in _cell_extra_parts(f)
            s = shape(r.inputs[part.input])
            s === nothing && return nothing
            if part.kind === :transposed
                length(s) <= 1 || return nothing
                push!(parts, _cell_transposed_shape(s))
            else
                i = shape(r.inputs[part.index])
                i === nothing && return nothing
                push!(parts, _cell_gather_shape(i, s, Val(part.tail)))
            end
        end
        f.kind === :columns && return shapes[cid] = _cell_column_shape(only(parts))
        if f.kind === :position_map
            # An array result's trailing extents: literals, or sizes of
            # inputs of known shape.
            extents = Int[]
            for x in f.rowmap.extents
                if x[1] === :const
                    push!(extents, x[2])
                else
                    s = shape(r.inputs[x[2]])
                    (s === nothing || x[3] > length(s)) && return nothing
                    push!(extents, length(s[x[3]]))
                end
            end
            rows = _cell_position_shape(parts...)
            return shapes[cid] = isempty(extents) ? rows :
                _cell_position_extents(rows, Tuple(extents))
        end
        f.kind === :hcat && return shapes[cid] = _cell_hcat_shape(Tuple(parts))
        f.kind === :array_arithmetic && return shapes[cid] = Base.promote_shape(parts...)
        full = _plate_broadcast_shape(Tuple(parts))
        f.kind === :reduce || return shapes[cid] = full
        (; dims, drop) = f.reduction
        # A `vec` keeping another axis has no row form: no shape is proven.
        f.reduction.vec && !all(d -> d in dims || length(full[d]) == 1, 2:length(full)) &&
            return nothing
        shapes[cid] = _cell_reduce_shape(full, Val(dims), Val(f.reduction.vec), Val(drop))
    end
    function valid(r, capture, lane, via, tail)
        indices = bound(r.inputs[1 + lane])
        indices === nothing && return false
        indices = something(indices)
        _cell_positions_type(typeof(indices)) || return false
        s = shape(r.inputs[1 + capture])
        s === nothing && return false
        linear = LinearIndices(s)
        shared = via == 0 ? nothing : bound(r.inputs[1 + via])
        via == 0 || shared !== nothing &&
            _cell_positions_type(typeof(something(shared))) || return false
        all(indices) do qs
            if via != 0
                source = something(shared)
                checkbounds(Bool, source, qs) || return false
                qs = source[qs]
            end
            qs isa Integer && (qs = (qs,))
            all(qs) do q
                checkbounds(Bool, linear, q, tail...)
            end
        end
    end
    valid
end

"""
    _CellPushdowns

The plan-wide pushdown decision for `_lower_with_ops`: `recipes` are the
producers computed only at their readers' rows (their table slots are kept and
recorded in `slots` as lowering reaches them: an operation's slot, or, for an
authored plate, the table length before its reserved slots, with the
argument values it is lowered with in `plates`). `captures[cell recipe id]`
maps each shared argument position read at the cell's own rows to each pair of
a batched argument holding rows it reads and the shared argument they are read
through (`_cell_capture_positions`); `batched[cell recipe id]` names the cell's
ordinary batched argument positions; `order[recipe id]` is its position in the
plan, producers first.
"""
struct _CellPushdowns
    graph::Graph
    types::Dict{ValueId,Any}
    views::Set{ValueId}
    pushed::Set{ValueId}
    recipes::Set{Int}
    producer::Dict{ValueId,Recipe}
    forms::Dict{Int,_CellForm}
    captures::Dict{Int,Dict{Int,Vector{Tuple{Int,Int}}}}
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
    # Per cell recipe, the shared argument positions its body reads as `v[q]`
    # with `q` of no known type: read whole, named `:untyped_index`.
    untyped::Dict{Int,Vector{Int}}
end

# Whether an operation's output is a lazy view of its inputs, built at no cost
# per element and read at rows as a view of them (ReactiveKernelsPPL's compact
# overlay of active coordinates on a packed base). Such a value is never
# computed at rows or refused for being computed whole: reading it at rows
# reads its inputs at those rows. Extend for an operation of that kind.
_cell_lazy_view(@nospecialize(op)) = false

# A cell returning one of its inputs without evaluating a body selects
# existing storage. In particular, selecting one bound ragged array does not
# compute its elements. Only input-backed projections qualify: selecting a
# computed array of arrays must not hide the work that produced its elements.
# A property read `x.name` that loads a field of `x`: `x` is a named tuple, or
# a concrete struct whose `getproperty` is Base's `getfield` fallback. Any
# other `getproperty` may compute its value.
function _cell_field_projection(T)
    T isa Type || return false
    T <: NamedTuple && return true
    isconcretetype(T) && isstructtype(T) || return false
    m = which(Base.getproperty, Tuple{T,Symbol})
    m.module === Base && m.sig == Tuple{typeof(Base.getproperty),Any,Symbol}
end

# Whether the native form `native` of a source over `params` is the property
# read `x.name` of its one parameter.
function _native_field_read(native, params, mod)
    length(params) == 1 || return false
    field = Meta.isexpr(native, :., 2) ? native.args :
        Meta.isexpr(native, :call, 3) &&
        !_native_callee_shadowed(native.args[1], params) &&
        _cell_binding(native.args[1], mod) === Base.getproperty ?
        native.args[2:3] : nothing
    field !== nothing && first(field) == only(params) &&
        last(field) isa QuoteNode && last(field).value isa Symbol
end

function _cell_input_views(p::Plan; types = nothing)
    cid(v) = canon_id(p.graph, v.id)
    type_of(v) = types === nothing ? valtype(v) : get(types, cid(v), valtype(v))
    storage = Set(cid(v) for v in p.have)
    views = Set{ValueId}()
    for r in p.recipes
        if r.op isa _BoundConstant
            union!(storage, (cid(v) for v in r.outputs))
            continue
        end
        projection = if r.op isa _AuthoredPlateCellOp && !r.op.rows
            body = r.op.plate.kernel.plan
            if isempty(body.recipes) && length(body.want) == 1
                input = findfirst(v -> canon_id(body.graph, v.id) ==
                    canon_id(body.graph, only(body.want).id), body.have)
                input !== nothing && cid(r.inputs[input + 1]) in storage
            else
                false
            end
        else
            false
        end
        # A genuine view of existing storage also stays lazy when its bounds
        # are shape expressions rather than a literal range row form. Typed
        # conversions that copy the view are excluded by _footprint_access.
        if !projection
            access = _footprint_access(r)
            projection = access !== nothing && access.kind === :view &&
                _footprint_indices_free(access) && cid(r.inputs[access.slot]) in storage
        end
        # Flattening existing storage changes only its indexing. This also
        # preserves provenance when a selected cell's body contains vec(data).
        if !projection && !r.effectful && length(r.inputs) == 1 &&
                length(r.outputs) == 1 && valtype(only(r.outputs)) === Any &&
                cid(only(r.inputs)) in storage
            analyzed = _cell_native(r; outlined = true)
            if r.op === Base.vec
                projection = true
            elseif analyzed !== nothing
                native, params, mod = analyzed
                projection = Meta.isexpr(native, :call, 2) &&
                    !_native_callee_shadowed(native.args[1], params) &&
                    _cell_binding(native.args[1], mod) === Base.vec &&
                    native.args[2] == only(params)
                # A field of existing storage is existing storage: a cell's
                # NamedTuple of row vectors (`rows[i].weight1`) is read at rows.
                projection |= _native_field_read(native, params, mod) &&
                    _cell_field_projection(type_of(only(r.inputs)))
            end
        end
        if _cell_lazy_view(r.op) || projection
            union!(views, (cid(v) for v in r.outputs))
            union!(storage, (cid(v) for v in r.outputs))
        end
    end
    views
end

# Whether a producer reads an input in `role` at its own rows.
_cell_row_role(role) =
    role in (:leaf, :gather, :linear_gather, :column, :factor_leaf, :factor, :part,
             :range_source, :reshape_source, :row, :vec_source, :sparse_operand,
             :program_gather)

# `outlined` are the ids of the recipes a cell lowering outlined (`_cell_outline`);
# `blocked`, when given, collects the ids of the producers whose values a reader
# asks for at rows but that have no row form; `known` keeps the forms of recipes
# already analyzed, by recipe.
function _cell_pushdowns(p::Plan; outlined::Set{Int} = Set{Int}(),
                         blocked::Union{Nothing,Set{Int}} = nothing,
                         known::IdDict{Recipe,Any} = IdDict{Recipe,Any}(),
                         argument_types = nothing)
    any(r -> r.op isa _AuthoredPlateCellOp, p.recipes) || return nothing
    g = p.graph
    cidof(v) = canon_id(g, v.id)
    producers = Dict{ValueId,Vector{Recipe}}()
    readers = Dict{ValueId,Vector{Recipe}}()
    for r in p.recipes
        foreach(o -> push!(get!(producers, cidof(o), Recipe[]), r), r.outputs)
        foreach(c -> push!(get!(readers, c, Recipe[]), r), unique(map(cidof, r.inputs)))
    end
    types = _cell_value_types(p; argument_types, outlined, known)
    static_type(v) = let candidates = get(producers, cidof(v), Recipe[])
        length(candidates) == 1 && only(candidates).op isa _BoundConstant ?
            typeof(only(candidates).op.value) :
            get(types, cidof(v), valtype(v))
    end
    branch_read = _cell_bound_capture_check(p, producers, static_type, known, outlined)
    # An input-backed view's output is read like a HAVE port: at rows as a view.
    views = _cell_input_views(p; types)
    fixed = Set(cidof(v) for v in Iterators.flatten((p.have, p.want)))
    union!(fixed, views)
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
    # (`_cell_product_shape`) and whose left factor's axes are known. A
    # product inferred to return a vector needs only its left factor's row
    # axis, including when its right operand is an expression rather than
    # one input (`A * (weights .* z)`).
    function shaped(cid)
        r = only(producers[cid])
        f = forms[r.id]
        f.kind in (:pointwise, :identity, :checked, :columns, :hcat, :vcat, :range_view, :reshape,
                   :reduce, :transposed, :array_arithmetic, :position_map, :vec, :rows,
                   :sparse_product) && return true
        inputs = r.inputs
        factor = cidof(inputs[findfirst(==(:factor), f.roles)])
        (f.operand == 0 ? static_type(only(r.outputs)) <: AbstractVector :
            static_type(inputs[f.operand]) <: Union{AbstractVecOrMat,Number}) &&
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
            role === :index ? _cell_index_type(r.op, static_type(input)) :
            role === :factor ? static_type(input) <: AbstractArray || array_form(input) :
            role === :factor_leaf ? (static_type(input) <: AbstractArray || array_form(input)) &&
                static_type(r.inputs[paired]) <: Union{AbstractVecOrMat,Number} :
            role === :scalar ? static_type(input) <: AbstractArray{<:Number} :
            role === :via ? _cell_positions_type(static_type(input)) :
            role in (:range_source, :reshape_source) ?
                static_type(input) <: AbstractVector ||
                    (!(static_type(input) <: AbstractArray) && form(cidof(input)) !== nothing) :
            # A transposed operand is one row only when it is a vector (a
            # number transposes to itself); a matrix's transpose reads columns.
            role === :transposed ? static_type(input) <: Union{AbstractVector,Number} :
            role === :vec_source ? static_type(input) <: AbstractArray :
            true
        end && all(f.kind in (:range_view, :reshape, :rows) ? f.literal : ()) do part
            part isa Int || static_type(r.inputs[part.input]) <: Integer
        end && _cell_reduce_fits(r, f, static_type)
    end
    captures = Dict{Int,Dict{Int,Vector{Tuple{Int,Int}}}}()
    untyped = Dict{Int,Vector{Int}}()
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
        positions, unknown = _cell_capture_positions(r, static_type;
            branch_read = (capture, lane, via, tail) ->
                branch_read(r, capture, lane, via, tail))
        isempty(positions) || (captures[r.id] = positions)
        isempty(unknown) || (untyped[r.id] = unknown)
        foreach(k -> get!(refused, cidof(r.inputs[1 + k]), :untyped_index), unknown)
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
            # A sparse product's operand is computed at the rows it reads when it
            # can be. Otherwise it is computed whole, as the right operand of any
            # product is, and that is no violation.
            if f.roles[j] === :sparse_operand
                operand = cidof(input)
                operand in pushed || (form(operand) !== nothing &&
                    fits(operand, :sparse_operand) && operands_fit(operand)) || continue
            end
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
                    positions = get(captures, reader.id, Dict{Int,Vector{Tuple{Int,Int}}}())
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
    _CellPushdowns(g, types, views, pushed, recipes,
        Dict(cid => only(producers[cid]) for cid in pushed), forms, captures, batched,
        Dict{Int,Int}(), Dict{Int,Any}(), Dict(r.id => k for (k, r) in enumerate(p.recipes)),
        requests, refused, dropped, untyped)
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
    parent.head === Symbol("'") && return true
    parent.head === :call && !isempty(parent.args) || return false
    # A row method's source names its function: its arguments are read by name.
    parent.args[1] isa GlobalRef &&
        _cell_row_method(_cell_binding(parent.args[1], nothing)) !== nothing && return true
    name = _kernel_split_callee_name(parent.args[1])
    name in (:vcat, :hcat, :*, :vec, :dropdims, :transpose, :adjoint) &&
        return !(child isa Expr && child.head in (:parameters, :kw))
    # The operand of a reduction over axes (`sum(x; dims = 2)`): the row forms
    # read it by name; a keyword is not an operand.
    name in (:sum, :prod, :maximum, :minimum) || return false
    parts = _cell_call_parts(parent)
    parts !== nothing && haskey(parts[2], :dims) && any(===(child), parts[1])
end

# The split parts of recipe `r` (`_partial_split_parts`) with the nodes to
# outline whole marked, or `nothing`: each node in a named-operand position
# that holds no such node itself. A node holding a marked one is split around
# it (`_partial_emit_split!`).
function _cell_outline_nodes(r::Recipe)
    length(r.outputs) == 1 && !r.effectful || return nothing
    # The strict boundaries also separate an undotted scalar argument of a
    # broadcast (`sd .* (z .- total / n)`), which runs once before it either way.
    for factors in (false, true)
        parts = _partial_split_parts(r; factors)
        found = parts === nothing ? nothing : _cell_marked_nodes(r, parts)
        found === nothing || return found
    end
    nothing
end
function _cell_marked_nodes(r::Recipe, parts)
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

# A selected producer cell may return an array whose reader asks for inner
# rows. Expose that cell's pure body so the ordinary row forms can carry the
# inner demand. Its input projections keep the outer index and the complete
# broadcast domain (including absorbed checks); only projections of existing
# storage are input-backed views. No data-sized domain is expanded here.
function _cell_expand_selected(p::Plan, blocked, types)
    selected = Set(r.id for r in p.recipes if r.id in blocked &&
        r.op isa _AuthoredPlateCellOp && !r.op.rows && !r.effectful &&
        !(get(types, canon_id(p.graph, only(r.outputs).id), Any) <: Number) &&
        length(_plate_atomic(r.op.plate)) < length(r.op.plate.kernel.plan.have) &&
        !isempty(r.op.plate.kernel.plan.recipes) &&
        all(q -> !q.effectful, r.op.plate.kernel.plan.recipes))
    isempty(selected) && return p
    old = p.graph
    g = Graph(copy(old.values), copy(old.recipes), copy(old.producers),
              copy(old.aliases), old.version)
    next_id = Ref(minimum((r.id for r in p.recipes); init = 0) - 1)
    freshid() = (id = next_id[]; next_id[] -= 1; id)
    recipes = Recipe[]
    for r in p.recipes
        if !(r.id in selected)
            push!(recipes, r)
            continue
        end
        body = r.op.plate.kernel.plan
        atomic = _plate_atomic(r.op.plate)
        batched = [i for i in eachindex(body.have) if !(i in atomic)]
        mapping = Dict{ValueId,Value}()
        for (i, v) in enumerate(body.have)
            mapping[canon_id(body.graph, v.id)] = if i in atomic
                r.inputs[i + 1]
            else
                small = Graph()
                roots = [value!(small, Symbol(:arg_, j), valtype(body.have[j])) for j in batched]
                scalar = prepare(small; have = roots, want = roots[findfirst(==(i), batched)])
                checks = Tuple(Tuple(findfirst(==(j), batched) for j in check)
                               for check in r.op.plate.axis_checks)
                plate = _AuthoredPlateOp{typeof(scalar),()}(scalar, checks)
                output = value!(g, Symbol(only(r.outputs).name, :_input_, i), valtype(v))
                push!(recipes, Recipe(freshid(),
                    (first(r.inputs), Tuple(r.inputs[j + 1] for j in batched)...),
                    (output,), _AuthoredPlateCellOp(plate, r.op.locality),
                    r.cost, nothing, false, nothing))
                output
            end
        end
        for q in body.recipes, v in q.outputs
            mapping[canon_id(body.graph, v.id)] =
                value!(g, Symbol(only(r.outputs).name, :_, v.name), valtype(v))
        end
        mapped(v) = mapping[canon_id(body.graph, v.id)]
        for q in body.recipes
            push!(recipes, Recipe(freshid(), map(mapped, q.inputs), map(mapped, q.outputs),
                                 q.op, q.cost, q.cse_key, q.effectful, q.source))
        end
        output = mapped(only(body.want))
        target = only(r.outputs)
        if valtype(target) === Any || valtype(target) === valtype(output)
            g.aliases[canon_id(g, target.id)] = canon_id(g, output.id)
            g.version += 1
        else
            push!(recipes, Recipe(r.id, (output,), r.outputs, identity,
                                  0.0, nothing, false, output.name))
        end
    end
    template = Plan(g, p.have, p.want, p.recipes, p.producer, p.cost, p.candidates)
    _partial_subplan(template, p.have, p.want, recipes)
end

# A scalar index, proven from operand types: `v[i]` of a vector, or `A[i, j...]`
# with every later index a literal position and one index per axis of `A`.
# Inspect the resolved getindex binding as for row forms; a shadowed function
# is not indexing.
function _cell_scalar_index(native, params, mod, inputs, type_of)
    if Meta.isexpr(native, :call) && length(native.args) >= 3 &&
       !_native_callee_shadowed(native.args[1], params) &&
       _cell_binding(native.args[1], mod) === Base.getindex
        native = Expr(:ref, native.args[2:end]...)
    end
    Meta.isexpr(native, :ref) && length(native.args) >= 2 && native.args[1] in params ||
        return nothing
    tail = native.args[3:end]
    all(j -> j isa Int, tail) || return nothing
    a = inputs[findfirst(==(native.args[1]), params)]
    index = native.args[2]
    position = findfirst(==(index), params)
    i = position === nothing ? nothing : inputs[position]
    A = type_of(a)
    I = i === nothing ? typeof(index) : type_of(i)
    A isa Type && A <: AbstractArray{<:Any,1 + length(tail)} &&
        I isa Type && I <: Integer && !(I <: Bool) || return nothing
    (; array = a, index = i === nothing ? index : i, type = I, tail = Tuple{Vararg{Int}}(tail))
end

"""
    _CellColumnRead(tail)

The operation `A -> A[:, tail...]` of a scalar read `A[i, tail...]`: row `i` of
this column is that element, so the read's identity cell (`_cell_index_op`)
computes `A` at row `i` when `A` has a row form. It is an operation rather than
a source so that cell lowering introduces the column without an authored one.
"""
struct _CellColumnRead{J<:Tuple{Vararg{Int}}}
    tail::J
end
@inline (c::_CellColumnRead)(A) = A[:, c.tail...]

# An ordinary identity cell evaluates exactly one vector element. The same
# operation serves scalar reads and individual terms of a cell factor.
function _cell_index_op(locality)
    inner = Graph()
    x = value!(inner, :term, Any)
    scalar = prepare(inner; have = (x,), want = (x,))
    plate = _AuthoredPlateOp{typeof(scalar),()}(scalar)
    _AuthoredPlateCellOp(plate, locality)
end

# The scalar reads at strict positions of a recipe with no split boundaries, with
# that recipe recompiled over them. Binding data emits a maximal live node of a
# cell's body as one closure (`_partial_emit_split!`): `exp(a[p])` beside a hoisted
# `paths[p]`. Inference through it on an abstract `a` finds no number, and it would
# read `a` whole. `read_of(ex)` proves one read, or returns `nothing`. Returns
# `(; reads, inputs, op, source)`, with `reads` as `expression => read` in the
# order `op` takes them after `inputs`, or `nothing`.
function _cell_part_reads(r::Recipe, params, mod, read_of)
    op = r.op
    op isa _KernelSourceOp && kernel_sourceop_form(op) === :fused && mod isa Module ||
        return nothing
    record = getfield(op, :f)
    record isa _KernelSource && _source_native(record) === nothing &&
        isempty(_source_captures(record)) || return nothing
    r.source isa Expr || return nothing
    body = _lane_strip_lines(r.source)
    body isa Expr || return nothing
    reads = Pair{Any,Any}[]
    function visit(ex)
        ex isa Expr || return
        read = read_of(ex)
        if read === nothing
            foreach(visit, _kernel_split_children(ex))
        else
            push!(reads, ex => read)
        end
    end
    visit(body)
    isempty(reads) && return nothing
    spelled = _kernel_collect_symbols!(Set{Symbol}(params), body)
    # One argument per distinct read: `a[p]` read twice is one element.
    distinct = unique(first.(reads))
    names = Symbol[Symbol("__rk_read_", k) for k in eachindex(distinct)]
    any(n -> n in spelled, names) && return nothing
    mapping = IdDict{Any,Any}(ex => names[findfirst(isequal(ex), distinct)] for (ex, _) in reads)
    source = _kernel_split_replace(body, mapping)
    used = _kernel_collect_symbols!(Set{Symbol}(), source)
    kept = Int[j for j in eachindex(params) if params[j] in used]
    operation = _partial_source_op(kernel_sourceop_token(op), :_reads,
        vcat(params[kept], names), source, mod)
    operation === nothing && return nothing
    (; reads = Pair{Any,Any}[ex => last(reads[findfirst(q -> isequal(first(q), ex), reads)])
                             for ex in distinct],
       inputs = r.inputs[kept], op = operation, source)
end

# Expose indexed scalar subexpressions along the strict split boundaries
# retained at authoring, then lower them as identity cells. A named scalar
# such as `w = exp.(a[1])` has no row form itself, but its read must still
# demand only row 1 of `a`. Splitting respects lazy branches and broadcast
# fusion. Unknown index/container types keep their authored recipe.
function _cell_scalar_reads(p::Plan; argument_types = nothing)
    types = _cell_value_types(p; argument_types)
    type_of(v) = get(types, canon_id(p.graph, v.id), Any)
    # Numeric input elements need no implementation cell. Array-valued
    # elements do: the projection records their input-backed provenance when
    # a downstream cell asks for rows inside that selected array.
    fixed = Set(canon_id(p.graph, v.id) for v in p.have)
    union!(fixed, _cell_input_views(p; types))
    for r in p.recipes
        r.op isa _BoundConstant || continue
        union!(fixed, (canon_id(p.graph, v.id) for v in r.outputs))
    end
    # Only introduce row demand where a producer can carry it. Other reads
    # retain their authored role: e.g. a broadcast's scalar coefficient may
    # come from `vcat([sigma])`, whose whole construction is already legitimate.
    row_sources = Set{ValueId}()
    for r in p.recipes
        _cell_form(r) === nothing && _cell_outline_nodes(r) === nothing && continue
        union!(row_sources, (canon_id(p.graph, v.id) for v in r.outputs))
    end
    function scalar_read(args...)
        read = _cell_scalar_index(args...)
        read === nothing && return nothing
        cid = canon_id(p.graph, read.array.id)
        (cid in row_sources && !(cid in fixed) ||
         cid in fixed && type_of(read.array) <: AbstractVector{<:AbstractArray}) ? read : nothing
    end
    outlines = Dict{Int,Any}()
    for r in p.recipes
        length(r.outputs) == 1 && !r.effectful || continue
        parts = _partial_split_parts(r; factors = true)
        parts === nothing && continue
        params = collect(first(_kernel_split_spec(parts.native)))
        mod = _kernel_endpoint_source_module(r.op)
        marked = Bool[scalar_read(node.expr, params, mod, r.inputs, type_of) !==
                      nothing for node in parts.split.nodes]
        any(marked) && (outlines[r.id] = (; parts, marked))
    end
    q, outlined = _cell_outline(p, outlines)
    # Copy even without an outline: a literal index needs a new graph value.
    old = q.graph
    g = Graph(copy(old.values), copy(old.recipes), copy(old.producers),
              copy(old.aliases), old.version)
    types = _cell_value_types(q; argument_types, outlined)
    next_id = minimum((r.id for r in q.recipes); init = 0) - 1
    recipes = Recipe[]
    changed = !isempty(outlined)
    # The identity cell of one read, its output `output`. The original cells
    # still enforce their locality over the complete cone; implementation cells
    # add no independent warning policy.
    function read_cell!(read, id, output, cost, source)
        stem = only(output).name
        i = read.index
        if !(i isa Value)
            i = value!(g, Symbol(stem, :_index), read.type)
            push!(recipes, Recipe(next_id, (), (i,), _BoundConstant(read.index),
                                  0.0, nothing, false, read.index))
            next_id -= 1
        end
        array = read.array
        if !isempty(read.tail)
            array = value!(g, Symbol(stem, :_column), Any)
            push!(recipes, Recipe(next_id, (read.array,), (array,),
                                  _CellColumnRead(read.tail), cost, nothing, false, nothing))
            next_id -= 1
        end
        push!(recipes, Recipe(id, (i, array), output, _cell_index_op(:ignore),
                              cost, nothing, false, source))
    end
    for r in q.recipes
        analyzed = length(r.outputs) == 1 && !r.effectful ?
            _cell_native(r; outlined = r.id in outlined) : nothing
        read = analyzed === nothing ? nothing :
            scalar_read(analyzed..., r.inputs, type_of)
        if read !== nothing
            read_cell!(read, r.id, r.outputs, r.cost, r.source)
            changed = true
            continue
        end
        parts = analyzed === nothing ? nothing : _cell_part_reads(r, analyzed[2], analyzed[3],
            ex -> scalar_read(ex, analyzed[2], analyzed[3], r.inputs, type_of))
        if parts === nothing
            push!(recipes, r)
            continue
        end
        values = Value[]
        for (k, (ex, part)) in enumerate(parts.reads)
            value = value!(g, Symbol(only(r.outputs).name, :_read, k), Any)
            read_cell!(part, next_id, (value,), r.cost, ex)
            next_id -= 1
            push!(values, value)
        end
        push!(recipes, Recipe(r.id, (parts.inputs..., values...), r.outputs, parts.op,
                              r.cost, nothing, false, parts.source))
        changed = true
    end
    changed || return p, outlined
    template = Plan(g, q.have, q.want, q.recipes, q.producer, q.cost, q.candidates)
    _partial_subplan(template, q.have, q.want, recipes), outlined
end

# A numeric scalar needed only by partition rows has a singleton demand
# domain. Represent that domain as an implementation plate: a nonempty
# consumer requests its one cell, while an empty consumer requests none.
# Existing plate fusion and shared row windows then carry that demand back
# through scalar arithmetic and indexed reads, including their footprints.
# Scalars with whole-value readers or boundary outputs keep their original
# representation. Row-method consumers must already be pushed, so their
# broadcast operands are read through the row method, never its whole call.
function _cell_scalar_plates(p::Plan; argument_types = nothing,
                             outlined::Set{Int} = Set{Int}())
    partitions = Set(r.id for r in p.recipes if r.op isa _AuthoredPlateCellOp && r.op.rows)
    isempty(partitions) && return p
    cid(v) = canon_id(p.graph, v.id)
    # A partial WANT or bound label can retain fewer plates than the original
    # label check. Only use that check's nonempty guarantee when the retained
    # plates still cover every table it accepts.
    tables = Set(cid(v) for r in p.recipes if r.op isa _PartitionLabel for v in r.inputs[2:end])
    covered = Set{ValueId}()
    for r in p.recipes
        r.id in partitions || continue
        lanes = get(p.producer, cid(first(r.inputs)), nothing)
        lanes !== nothing && lanes.op isa _PartitionLanes && push!(covered, cid(first(lanes.inputs)))
    end
    complete = !isempty(tables) && issubset(tables, covered)
    complete && length(partitions) == 1 && return p
    cells = _cell_pushdowns(p; argument_types, outlined)
    types = cells.types
    boundary = Set(cid(v) for v in (p.have..., p.want...))
    readers = Dict{ValueId,Vector{Recipe}}()
    for r in p.recipes, v in r.inputs
        push!(get!(readers, cid(v), Recipe[]), r)
    end
    # Every valid label occurs in at least one queried plate. A scalar needed
    # by all of them is therefore always needed, and keeps its scalar form.
    # Boundary outputs are needed independently of the partition as well.
    needed = Dict(id => copy(partitions) for id in boundary)
    for r in reverse(p.recipes)
        uses = r.id in partitions ? Set((r.id,)) : Set{Int}()
        if !(r.id in partitions)
            for v in r.outputs
                union!(uses, get(needed, cid(v), Set{Int}()))
            end
        end
        for v in r.inputs
            union!(get!(needed, cid(v), Set{Int}()), uses)
        end
    end
    lifted = Set{ValueId}()
    function accepts(r, id)
        length(r.outputs) == 1 && cid(only(r.outputs)) in lifted && return true
        slots = findall(v -> cid(v) == id, r.inputs)
        if r.op isa _AuthoredPlateCellOp && r.op.rows
            return all(>(1), slots)
        end
        length(r.outputs) == 1 && cid(only(r.outputs)) in cells.pushed || return false
        f = get(cells.forms, r.id, nothing)
        f !== nothing && f.kind in (:pointwise, :rows) &&
            all(j -> f.roles[j] === :leaf, slots)
    end
    for r in reverse(p.recipes)
        length(r.outputs) == 1 && !r.effectful || continue
        id = cid(only(r.outputs))
        T = get(types, id, Any)
        id in boundary && continue
        complete && get(needed, id, partitions) == partitions && continue
        T <: Number && isconcretetype(T) || continue
        r.op isa _BoundConstant && continue
        if r.op isa _AuthoredPlateCellOp
            inner = r.op.plate.kernel.plan
            !r.op.rows && length(r.inputs) == 2 && isempty(inner.recipes) &&
                get(types, cid(first(r.inputs)), Any) <: Integer || continue
        end
        rs = get(readers, id, Recipe[])
        !isempty(rs) && all(q -> accepts(q, id), rs) && push!(lifted, id)
    end
    isempty(lifted) && return p
    g = Graph(copy(p.graph.values), copy(p.graph.recipes), copy(p.graph.producers),
        copy(p.graph.aliases), p.graph.version)
    replacements = Dict(id => Value{AbstractVector{types[id]}}(id, g.values[id].name)
        for id in lifted)
    for (id, v) in replacements
        g.values[id] = v
    end
    mapped(v) = get(replacements, cid(v), v)
    recipes = Recipe[]
    next_id = minimum((r.id for r in p.recipes); init = 0) - 1
    singleton = value!(g, :_cell_scalar_axis, UnitRange{Int})
    push!(recipes, Recipe(next_id, (), (singleton,), _BoundConstant(1:1), 0.0, nothing, false, nothing))
    next_id -= 1
    for r in p.recipes
        inputs = Tuple(mapped(v) for v in r.inputs)
        outputs = Tuple(mapped(v) for v in r.outputs)
        if length(outputs) == 1 && cid(only(r.outputs)) in lifted
            inner = Graph()
            if r.op isa _AuthoredPlateCellOp
                # A scalar indexed read becomes a one-row gather, so an empty
                # downstream partition requests no source position at all.
                index_type = get(types, cid(first(r.inputs)), Int)
                index = value!(g, Symbol(only(outputs).name, :_indices), Vector{index_type})
                index_op = _partial_source_op(:cell_scalar, :indices, [:i], :(vcat(i)), @__MODULE__)
                push!(recipes, Recipe(next_id, (inputs[1],), (index,),
                    index_op, 0.0, nothing, false, :(vcat(i))))
                next_id -= 1
                i = value!(inner, :i, index_type)
                x = value!(inner, :x, get(types, cid(r.inputs[2]), Any))
                result = value!(inner, :result, get(types, cid(only(r.outputs)), Any))
                op = _partial_source_op(:cell_scalar, :index, [:i, :x], :(x[i]), @__MODULE__)
                add!(inner, (i, x) => result, op; source = :(x[i]))
                kernel = prepare(inner; have = (i, x), want = result)
                plate = _AuthoredPlateOp{typeof(kernel),(2,)}(kernel)
                push!(recipes, Recipe(r.id, (index, inputs[2]), outputs, plate, r.cost, nothing, false, nothing))
            else
                axis = value!(inner, :_scalar_axis, Int)
                roots = [value!(inner, Symbol(:input_, j), get(types, cid(v), valtype(v)))
                    for (j, v) in enumerate(r.inputs)]
                result = value!(inner, :result, get(types, cid(only(r.outputs)), Any))
                add!(inner, Tuple(roots) => result, r.op; source = r.source)
                kernel = prepare(inner; have = (axis, roots...), want = result)
                atomic = Tuple(j + 1 for j in eachindex(r.inputs) if !(cid(r.inputs[j]) in lifted))
                plate = _AuthoredPlateOp{typeof(kernel),atomic}(kernel)
                push!(recipes, Recipe(r.id, (singleton, inputs...), outputs, plate, r.cost, nothing, false, nothing))
            end
        else
            op = r.op
            if op isa _AuthoredPlateCellOp
                atomic = Tuple(j for j in _plate_atomic(op.plate) if !(cid(r.inputs[j + 1]) in lifted))
                plate = _AuthoredPlateOp{typeof(op.plate.kernel),atomic}(op.plate.kernel, op.plate.axis_checks)
                op = _AuthoredPlateCellOp(plate, op.locality, op.rows, op.partitioned, op.plan, op.plan_constant)
            end
            push!(recipes, Recipe(r.id, inputs, outputs, op, r.cost, r.cse_key, r.effectful, r.source))
        end
    end
    template = Plan(g, p.have, p.want, recipes, p.producer, p.cost, p.candidates)
    _partial_subplan(template, p.have, p.want, recipes)
end

# The plates several cells read that fusion left for the cell lowering
# (`_shared_cell_producer`, codegen.jl) but that the pushdowns `cells` do not
# compute at rows: each is composed into its cells after all, as before, and
# `again` analyzes the composed plan. `(p, cells)` when there is none.
function _cell_compose_unpushed(again, p::Plan, cells)
    cells === nothing && return p, cells
    unpushed = _cell_shared_producers(p, cells.pushed)
    isempty(unpushed) && return p, cells
    composed = _fuse_authored_plate_chains(p; cells_only = true, compose_shared = unpushed)
    composed === p ? (p, cells) : again(composed)
end

# `p` with every plate read only by several cells composed into those cells.
function _cell_compose_shared(p::Plan)
    shared = _cell_shared_producers(p, ())
    isempty(shared) ? p :
        _fuse_authored_plate_chains(p; cells_only = true, compose_shared = shared)
end

# The outputs of the plates read only by several cells, other than `pushed`.
function _cell_shared_producers(p::Plan, pushed)
    g = p.graph
    boundary = Set(canon_id(g, v.id) for v in vcat(p.have, p.want))
    readers = Dict{ValueId,Vector{Recipe}}()
    for r in p.recipes, cid in unique(canon_id(g, v.id) for v in r.inputs)
        push!(get!(readers, cid, Recipe[]), r)
    end
    shared = Set{ValueId}()
    for r in p.recipes
        r.op isa _AuthoredPlateOp && length(r.outputs) == 1 || continue
        cid = canon_id(g, only(r.outputs).id)
        (cid in boundary || cid in pushed) && continue
        rs = get(readers, cid, Recipe[])
        length(rs) >= 2 && all(q -> q.op isa _AuthoredPlateCellOp, rs) && push!(shared, cid)
    end
    shared
end

"""
    _cell_outlined_pushdowns(p::Plan) -> (plan, pushdowns)

`p` with the recipes outlined that the cells then compute at their rows (see
above), and that plan's pushdowns (`_cell_pushdowns`; `nothing` when the plan
has no cell). A recipe is outlined when a reader asks for its value at
rows and it has no row form; the outline stays while the recipe is pushed, and
is dropped for good otherwise. `p` itself when nothing is outlined.
"""
function _cell_outlined_pushdowns(p::Plan; argument_types = nothing)
    any(r -> r.op isa _AuthoredPlateCellOp, p.recipes) || return p, nothing
    if any(r -> r.op isa _KernelSourceOp && r.op.f isa _KernelBranch, p.recipes)
        types = _cell_value_types(p; argument_types)
        recipes = map(p.recipes) do r
            selected = _type_selected_branch(r,
                Any[get(types, canon_id(p.graph, v.id), Any) for v in r.inputs])
            selected === nothing ? r : selected.recipe
        end
        if any(pair -> pair[1] !== pair[2], zip(recipes, p.recipes))
            p = _partial_subplan(p, p.have, p.want, recipes)
        end
    end
    p, scalar_outlined = _cell_scalar_reads(p; argument_types)
    p = _cell_scalar_plates(p; argument_types, outlined = scalar_outlined)
    p = _cell_sparse_products(p; argument_types, outlined = scalar_outlined)
    p = _fuse_authored_plate_chains(p; cells_only = true)
    known = IdDict{Recipe,Any}()
    byid = Dict(r.id => r for r in p.recipes)
    candidates = Dict{Int,Any}()
    candidate(id) = haskey(byid, id) ?
        get!(() -> _cell_outline_nodes(byid[id]), candidates, id) : nothing
    kept = Dict{Int,Any}()
    refused = Set{Int}()
    while true
        q, outlined = _cell_outline(p, kept)
        union!(outlined, scalar_outlined)
        blocked = Set{Int}()
        cells = _cell_pushdowns(q; argument_types, outlined, blocked, known)
        cells === nothing && return q, cells
        expanded = _cell_expand_selected(q, blocked, cells.types)
        expanded === q || return _cell_outlined_pushdowns(expanded; argument_types)
        lost = Int[id for id in keys(kept) if cells === nothing || !(id in cells.recipes)]
        if !isempty(lost)
            union!(refused, lost)
            foreach(id -> delete!(kept, id), lost)
            continue
        end
        new = Int[id for id in blocked if !haskey(kept, id) && !(id in refused) &&
                  candidate(id) !== nothing]
        isempty(new) && return _cell_compose_unpushed(q, cells) do composed
            _cell_outlined_pushdowns(composed; argument_types)
        end
        foreach(id -> kept[id] = candidates[id], new)
    end
end

# Whole work used exclusively by one partition sum can execute in that sum's
# nonempty-row scope. Row pushdowns already avoid unused rows, but an atomic
# operand (for example the right factor of a matrix product) has no row domain
# of its own. Its entire pure dependency chain belongs to its sole consumer.
# Keep shared producers, query outputs, row-index dependencies and effects in
# the enclosing plan. A sum lowered with others by one window emitter
# (`_cell_lowering_groups`) keeps its region: the group's statements are then
# placed under each member's guard by what they read (`_cell_guard_group`).
function _cell_partition_sum_type(p, cells, r)
    inputs = Any[get(cells.types, canon_id(p.graph, v.id), valtype(v))
                 for v in r.inputs[2:end]]
    T = _checked_plate_element!(_UninferredValue[], r.op.plate, inputs, Symbol[])
    if T === Any
        # A pushed capture is read only by indexed element/row accesses, as
        # proved by the capture analysis. Its generated window has the known
        # numeric element type and rank even when the original container is
        # abstract. Use those facts when inferring scalar helpers around the
        # accesses; never substitute an arbitrary whole-array argument.
        captured = get(cells.captures, r.id, Dict{Int,Vector{Tuple{Int,Int}}}())
        window_inputs = Any[k in keys(captured) &&
            canon_id(p.graph, r.inputs[k + 1].id) in cells.pushed ?
            _cell_standin_type(S) : S for (k, S) in enumerate(inputs)]
        window_inputs == inputs || (T = _checked_plate_element!(
            _UninferredValue[], r.op.plate, window_inputs, Symbol[]))
    end
    T isa Type && T <: Number && isconcretetype(T) &&
        Base.promote_op(+, T, T) === T ? T : Any
end

function _cell_partition_regions(p::Plan, cells, groups)
    cells === nothing && return Dict{Int,Int}()
    cid(v) = canon_id(p.graph, v.id)
    boundary = Set(cid(v) for v in (p.have..., p.want...))
    readers = Dict{ValueId,Vector{Int}}()
    for r in p.recipes, v in r.inputs
        push!(get!(readers, cid(v), Int[]), r.id)
    end
    # A region's guard must be available before any of its work is evaluated.
    indices = Set(cid(first(r.inputs)) for r in p.recipes
                  if r.op isa _AuthoredPlateCellOp)
    for r in reverse(p.recipes)
        any(v -> cid(v) in indices, r.outputs) || continue
        union!(indices, (cid(v) for v in r.inputs))
    end
    regions = Dict{Int,Int}()
    for r in reverse(p.recipes)
        if r.op isa _AuthoredPlateCellOp
            r.op.rows || continue
            T = length(r.outputs) == 1 ? _cell_partition_sum_type(p, cells, r) : Any
            T <: Number && isconcretetype(T) && (regions[r.id] = r.id)
            continue
        end
        (r.effectful || r.op isa _BoundConstant ||
         any(v -> cid(v) in boundary || cid(v) in indices, r.outputs)) && continue
        uses = unique(get(regions, reader, 0) for v in r.outputs
                      for reader in get(readers, cid(v), Int[]))
        length(uses) == 1 && only(uses) != 0 && (regions[r.id] = only(uses))
    end
    # Do not add guards where row demand already accounts for all numerical
    # work. In particular a source view is only metadata, not a whole read.
    # A region that reads a port of its sum's separability rule directly (a
    # scalar coordinate `u[7]`) is kept even without array work: unguarded, an
    # empty partition would read another partition's coordinates.
    views = _cell_input_views(p; types = cells.types)
    ruled = Dict{Int,Set{ValueId}}()
    for r in p.recipes
        r.op isa _AuthoredPlateCellOp && get(regions, r.id, 0) == r.id &&
            !isempty(r.op.partitioned) || continue
        ports = Set{Symbol}(first(rule) for rule in r.op.partitioned)
        ruled[r.id] = Set{ValueId}(cid(v) for v in p.have if v.name in ports)
    end
    needed = Set{Int}()
    for r in p.recipes
        owner = get(regions, r.id, 0)
        owner == 0 && continue
        r.id in cells.recipes && continue
        r.op isa _AuthoredPlateCellOp && continue
        (any(v -> !(cid(v) in views) && get(cells.types, cid(v), Any) <: AbstractArray,
             r.outputs) ||
         any(v -> cid(v) in get(ruled, owner, ()), r.inputs)) && push!(needed, owner)
    end
    filter!(pair -> last(pair) in needed, regions)
    # Deferring a potentially throwing pure calculation past an effect would
    # change their observable order on a populated partition. A grouped sum's
    # region runs where its group lowers, at the group's last member.
    emitted_at = Dict{Int,Int}(owner => (group = get(groups, owner, nothing);
                                         group === nothing ? owner : last(group))
                               for owner in unique(values(regions)))
    starts = Dict{Int,Int}()
    effects = 0
    for r in p.recipes
        effects += r.effectful
        owner = get(regions, r.id, 0)
        owner == 0 || get!(starts, owner, effects)
        for (o, at) in emitted_at
            at == r.id && effects != get(starts, o, effects) && delete!(needed, o)
        end
    end
    filter!(pair -> last(pair) in needed, regions)
    regions
end

# The members of one emitter group with an exclusive region, while the group
# is emitted (`_lower_authored_plate_cells!`): which member owns each value
# (an exclusive pushed value, or whole work its region computes), each member's
# nonempty-row guard, the names its region binds, and, per emitted statement,
# the member it is forced to (`_CELL_UNGUARDED`, or `_CELL_BY_READS` to follow
# what it reads). `requested` holds every row binding asked of a pushed value.
mutable struct _CellRegions
    const owner::Dict{ValueId,Int}
    const guards::Dict{Int,Any}
    const symbols::Dict{Symbol,Int}
    const tags::Vector{Int}
    const requested::Set{Symbol}
    forced::Int
end
const _CELL_UNGUARDED = 0
const _CELL_BY_READS = -1
_CellRegions(owner, guards, symbols) = _CellRegions(owner, guards, symbols, Int[],
    Set{Symbol}(), _CELL_UNGUARDED)

# The rows of a request exported from a member's guard to a shared value, as
# one vector type whether the member is empty or not.
@inline _cell_rows_vector(row::Integer) = Int[row]
@inline _cell_rows_vector(rows::Vector{Int}) = rows
@inline _cell_rows_vector(rows::AbstractVector{<:Integer}) = Vector{Int}(rows)
@inline _cell_no_rows() = Int[]

# Places the statements of an emitter group with guarded members (`_CellRegions`,
# one tag per statement) under those members' guards, after each member's
# region (`preludes`). A statement forced to a member goes under its guard; one
# that follows what it reads goes under the guard of the member whose values it
# reads, or stays unguarded when it reads none. An unguarded statement may read
# a guarded member's request rows: they are exported as no rows when the member
# is empty, since an empty member asks for none. Each member's last guarded
# block assigns its sum zero otherwise (`defaults`). Returns the statements and
# the member owning each name bound under a guard, or `nothing` when a
# statement reads two members' values, or an unguarded statement reads a
# member's value other than request rows.
function _cell_guard_group(statements::Vector{Any}, t::_CellRegions,
                           preludes::Dict{Int,Vector{Any}}, defaults::Dict{Int,Any},
                           order::Vector{Int})
    length(t.tags) == length(statements) ||
        error("internal: $(length(t.tags)) tags for $(length(statements)) group statements")
    owned = copy(t.symbols)
    out = Any[]
    pending = Dict{Int,Vector{Any}}(m => copy(get(preludes, m, Any[])) for m in order)
    exported = Dict{Symbol,Symbol}()
    function flush!(member)
        isempty(pending[member]) && return
        push!(out, Expr(:if, copy(t.guards[member]), Expr(:block, pending[member]...)))
        pending[member] = Any[]
    end
    for (k, st) in enumerate(statements)
        read = Dict{Int,Vector{Symbol}}()
        for name in _symbol_refs!(Set{Symbol}(), st)
            m = get(owned, name, _CELL_UNGUARDED)
            m == _CELL_UNGUARDED || push!(get!(() -> Symbol[], read, m), name)
        end
        member = t.tags[k]
        if member == _CELL_BY_READS
            length(read) > 1 && return nothing
            member = isempty(read) ? _CELL_UNGUARDED : only(keys(read))
        end
        for (m, names) in read
            m == member && continue
            member == _CELL_UNGUARDED && all(in(t.requested), names) || return nothing
            mapping = Dict{Symbol,Any}()
            for name in names
                x = get(exported, name, nothing)
                if x === nothing
                    flush!(m)
                    x = gensym(:cell_rows_exported)
                    push!(out, Expr(:(=), x, Expr(:if, copy(t.guards[m]),
                        Expr(:call, _cell_ref(:_cell_rows_vector), name),
                        Expr(:call, _cell_ref(:_cell_no_rows)))))
                    exported[name] = x
                end
                mapping[name] = x
            end
            st = _footprint_replace(st, mapping)
        end
        if member == _CELL_UNGUARDED
            push!(out, st)
        else
            haskey(pending, member) ||
                error("internal: a group statement is forced to unguarded member $member")
            push!(pending[member], st)
            foreach(name -> owned[name] = member, _assigned_names!(Set{Symbol}(), st))
        end
    end
    for member in order
        push!(out, Expr(:if, copy(t.guards[member]), Expr(:block, pending[member]...),
                        Expr(:block, defaults[member])))
    end
    out, owned
end

# The read expression of a footprint entry under its guards, and those guards.
function _footprint_entry_parts(kind, expr)
    guards = Any[]
    while kind === :conditional
        push!(guards, expr[1])
        kind, expr = expr[2], expr[3]
    end
    kind, expr, guards
end

# Footprint entries recorded while a group with guarded members lowered
# (`snapshot`: each value's entry count before it): an entry reading a name
# bound under a member's guard (`owned`) is recorded under that guard too.
function _footprint_guard_reads!(recording, snapshot, owned::Dict{Symbol,Int}, guards)
    for (cid, entries) in recording.records
        for k in get(snapshot, cid, 0) + 1:length(entries)
            kind, expr = entries[k]
            _, inner, present = _footprint_entry_parts(kind, expr)
            for name in _symbol_refs!(Set{Symbol}(), inner)
                m = get(owned, name, _CELL_UNGUARDED)
                m == _CELL_UNGUARDED && continue
                guard = guards[m]
                any(x -> x == guard, present) && continue
                push!(present, guard)
                kind, expr = :conditional, (guard, kind, expr)
            end
            entries[k] = (kind, expr)
        end
    end
    recording
end

# The same entries when the group lowers unguarded after all: without the
# members' guards.
function _footprint_unguard_reads!(recording, snapshot, guards)
    for (cid, entries) in recording.records
        for k in get(snapshot, cid, 0) + 1:length(entries)
            inner_kind, inner, present = _footprint_entry_parts(entries[k]...)
            entry = (inner_kind, inner)
            for guard in Iterators.reverse(present)
                any(x -> x == guard, guards) && continue
                entry = (:conditional, (guard, entry...))
            end
            entries[k] = entry
        end
    end
    recording
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
    original_types::Dict{ValueId,Symbol}
    rows::Dict{Tuple{ValueId,Symbol},Symbol}
    requests::Dict{ValueId,Vector{Symbol}}
    unions::Dict{ValueId,Symbol}
    reads::Dict{ValueId,Vector{Any}}
    data::Dict{ValueId,Symbol}
    # Whether every request's rows are sorted and unique (a partition's: its
    # windows are then read by bisection, and a value asked for at the same
    # row by many cells is computed once), or a cell's few rows as read.
    sorted::Bool
    # The names of the HAVE values constant per label (a partition's
    # `plan_constant`): a lazy view may serve its rows from such a part.
    constant::Set{Symbol}
    # A group with guarded members (`_CellRegions`), or `nothing`.
    regions::Union{Nothing,_CellRegions}
end
_CellEmitter(body, pushdowns, names, runtime_ops, runtime_recipes; sorted::Bool = false,
             constant::Set{Symbol} = Set{Symbol}(),
             regions::Union{Nothing,_CellRegions} = nothing) =
    _CellEmitter(body, pushdowns, names, runtime_ops, runtime_recipes,
        Dict{ValueId,Symbol}(), Dict{ValueId,Symbol}(), Dict{Tuple{ValueId,Symbol},Symbol}(),
        Dict{ValueId,Vector{Symbol}}(), Dict{ValueId,Symbol}(),
        Dict{ValueId,Vector{Any}}(), Dict{ValueId,Symbol}(), sorted, constant, regions)

# The member that owns value `cid` in a group with guarded members, else
# `_CELL_UNGUARDED`.
_cell_owner(e::_CellEmitter, cid::ValueId) =
    e.regions === nothing ? _CELL_UNGUARDED : get(e.regions.owner, cid, _CELL_UNGUARDED)
# Bookkeeping of value `cid` (its shape, rows and requests): a shared value's
# stays outside every guard; an exclusive value's follows what it reads, so the
# label-only part of it remains plannable once per label.
_cell_bookkeeping(e::_CellEmitter, cid::ValueId) =
    _cell_owner(e, cid) == _CELL_UNGUARDED ? _CELL_UNGUARDED : _CELL_BY_READS

# Forces the statements emitted from here on to `member` (tagging those emitted
# so far with the previous force), and records footprint reads under that
# member's guard. Returns the previous force.
function _cell_force!(e::_CellEmitter, member::Int)
    t = e.regions
    t === nothing && return member
    n = length(e.body.args)
    length(t.tags) < n && append!(t.tags, fill(t.forced, n - length(t.tags)))
    previous = t.forced
    t.forced = member
    recording = get(task_local_storage(), _FOOTPRINT, nothing)
    recording === nothing ||
        (recording.guard = member > 0 ? t.guards[member] : nothing)
    previous
end
function _cell_forced(f, e::_CellEmitter, member::Int)
    e.regions === nothing && return f()
    previous = _cell_force!(e, member)
    try
        f()
    finally
        _cell_force!(e, previous)
    end
end

# A request's rows as the emitter holds them: canonical for a partition.
_cell_canon(e::_CellEmitter, rows) =
    e.sorted ? Expr(:call, _cell_ref(:_cell_canonical_rows), rows) : rows

_cell_ref(name) = GlobalRef(@__MODULE__, name)

function _cell_bind!(e::_CellEmitter, prefix::Symbol, value)
    name = gensym(prefix)
    push!(e.body.args, Expr(:(=), name, value))
    name
end

# The full axes of an operand read whole: `()` for a value known to be a number,
# so a partition's plan holds the shape without reading the value (a scalar
# computed from the active vector would otherwise tie every row derived from
# it to the data), else the axes of its broadcastable form.
function _cell_leaf_shape_expr(e::_CellEmitter, input::ValueId)
    get(e.pushdowns.types, input, Any) <: Number && return Expr(:tuple)
    Expr(:call, _cell_ref(:_cell_leaf_shape), e.names[input])
end

# Infer the value the original recipe would produce, without evaluating its
# full array. Window operands have different types and cannot justify dispatch
# of the original expression. `_promote_op` folds from the actual HAVE types.
function _cell_original_type_expr(e::_CellEmitter, input::ValueId)
    haskey(e.original_types, input) && return e.original_types[input]
    d = e.pushdowns
    input in d.pushed || return Expr(:call, GlobalRef(Core, :typeof), e.names[input])
    _cell_forced(e, _cell_bookkeeping(e, input)) do
        r = d.producer[input]
        types = Any[_cell_original_type_expr(e, canon_id(d.graph, v.id)) for v in r.inputs]
        e.original_types[input] = _cell_bind!(e, :cell_type,
            _slot_type_expr(r, d.slots[r.id], types))
    end
end

# A shape is bookkeeping of its value (`_cell_bookkeeping`), memoized once.
function _cell_shape!(e::_CellEmitter, cid::ValueId)
    haskey(e.shapes, cid) && return e.shapes[cid]
    _cell_forced(() -> _cell_compute_shape!(e, cid), e, _cell_bookkeeping(e, cid))
end
function _cell_compute_shape!(e::_CellEmitter, cid::ValueId)
    haskey(e.shapes, cid) && return e.shapes[cid]
    d = e.pushdowns
    r = d.producer[cid]
    f = d.forms[r.id]
    if f.kind === :array_arithmetic
        for (j, T) in zip(f.literal.positions, f.literal.types)
            input = canon_id(d.graph, r.inputs[j].id)
            push!(e.body.args, Expr(:call, _cell_ref(:_cell_check_arithmetic_type),
                _cell_original_type_expr(e, input), QuoteNode(T)))
        end
    end
    input_shape(input) = input in d.pushed ? _cell_shape!(e, input) :
        _cell_leaf_shape_expr(e, input)
    f.kind === :sparse_product && return e.shapes[cid] = _cell_bind!(e, :cell_shape,
        Expr(:call, _cell_ref(:_cell_sparse_shape), e.names[canon_id(d.graph, r.inputs[1].id)],
            input_shape(canon_id(d.graph, r.inputs[2].id))))
    if f.kind === :product
        factor = canon_id(d.graph, r.inputs[findfirst(==(:factor), f.roles)].id)
        # With no single right-operand port, `shaped` proved that the whole
        # product is a vector. Its only axis is the left factor's row axis;
        # do not evaluate the right expression to discover its shape.
        f.operand == 0 && return e.shapes[cid] = _cell_bind!(e, :cell_shape,
            Expr(:call, _cell_ref(:_cell_column_shape), input_shape(factor)))
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
    f.kind === :vec && return e.shapes[cid] = _cell_bind!(e, :cell_shape,
        Expr(:call, _cell_ref(:_cell_vec_shape),
             input_shape(canon_id(d.graph, r.inputs[only(f.leaves)].id))))
    f.kind === :rows && return e.shapes[cid] = _cell_bind!(e, :cell_shape,
        Expr(:call, _cell_ref(:_plate_broadcast_shape), Expr(:tuple,
            Expr(:call, _cell_ref(:_cell_dims_shape), _cell_literal!(e, r, f)),
            Any[_cell_part_shape!(e, r, f, j) for j in f.leaves]...)))
    parts = Any[_cell_part_shape!(e, r, f, j) for j in f.leaves]
    # A transposed vector is one row; a gather keeping axes of its source spans
    # them (`_cell_extra_parts`).
    for part in _cell_extra_parts(f)
        source = input_shape(canon_id(d.graph, r.inputs[part.input].id))
        push!(parts, part.kind === :transposed ?
            Expr(:call, _cell_ref(:_cell_transposed_shape), source) :
            Expr(:call, _cell_ref(:_cell_gather_shape),
                 input_shape(canon_id(d.graph, r.inputs[part.index].id)), source,
                 Val(part.tail)))
    end
    shape = f.kind === :columns ?
        Expr(:call, _cell_ref(:_cell_column_shape), only(parts)) :
        f.kind === :position_map ? _cell_position_shape_expr!(e, r, f, parts, input_shape) :
        f.kind === :hcat ? Expr(:call, _cell_ref(:_cell_hcat_shape), Expr(:tuple, parts...)) :
        f.kind === :array_arithmetic ? Expr(:call, GlobalRef(Base, :promote_shape), parts...) :
        Expr(:call, _cell_ref(:_plate_broadcast_shape), Expr(:tuple, parts...))
    f.kind === :reduce && (shape = Expr(:call, _cell_ref(:_cell_reduce_shape), shape,
        Val(f.reduction.dims), Val(f.reduction.vec), Val(f.reduction.drop)))
    e.shapes[cid] = _cell_bind!(e, :cell_shape, shape)
end

# The full axes of a `vcat`'s operands, in order: a tuple expression.
function _cell_vcat_part_shapes!(e::_CellEmitter, r::Recipe, f::_CellForm)
    d = e.pushdowns
    shapes = map(f.parts) do part
        part isa Int || return Expr(:tuple)
        input = canon_id(d.graph, r.inputs[part].id)
        input in d.pushed ? _cell_shape!(e, input) :
            _cell_leaf_shape_expr(e, input)
    end
    Expr(:tuple, shapes...)
end

# The full axes of the broadcast operand that input `j` of recipe `r` (form `f`)
# makes: the input itself, its column `A[:, k...]`, or its product `A * w`.
function _cell_part_shape!(e::_CellEmitter, r::Recipe, f::_CellForm, j::Int)
    d = e.pushdowns
    input = canon_id(d.graph, r.inputs[j].id)
    shape = input in d.pushed ? _cell_shape!(e, input) :
        _cell_leaf_shape_expr(e, input)
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
    haskey(e.rows, (cid, row)) && return e.rows[(cid, row)]
    # The slice follows what it reads: the operand, and the rows.
    _cell_forced(e, _CELL_BY_READS) do
        _cell_compute_operand_row!(e, cid, row)
    end
end
function _cell_compute_operand_row!(e::_CellEmitter, cid::ValueId, row::Symbol)
    get!(e.rows, (cid, row)) do
        # The positions read, decided on the operand's shape apart from the
        # view (`_cell_slice_positions`): a partition's plan decides them once
        # per label, and the view's index is never a value merging the plan's
        # rows with a fresh vector.
        shape = _cell_bind!(e, :cell_shape, _cell_leaf_shape_expr(e, cid))
        positions = _cell_bind!(e, :cell_positions,
            Expr(:call, _cell_ref(:_cell_slice_positions), shape, row))
        _cell_bind!(e, :cell_operand, Expr(:call, _cell_ref(:_cell_rows_view),
            Expr(:call, GlobalRef(Base, :broadcastable), e.names[cid]), positions))
    end
end

# Asks for pushed value `cid` at rows `row` (one row or an integer vector).
function _cell_request!(e::_CellEmitter, cid::ValueId, row::Symbol)
    rows = get!(() -> Symbol[], e.requests, cid)
    row in rows || push!(rows, row)
    e.regions === nothing || push!(e.regions.requested, row)
    row
end

# The rows every request of `cid` reads: the one request, or their union.
function _cell_union!(e::_CellEmitter, cid::ValueId)
    rows = e.requests[cid]
    e.unions[cid] = length(rows) == 1 ? only(rows) :
        _cell_bind!(e, :cell_rows_union, Expr(:call,
            _cell_ref(e.sorted ? :_cell_union_rows_sorted : :_cell_union_rows), rows...))
end

# The data of pushed value `cid` at rows `row`, one of its requests: its
# window when that request is the window's rows, else those rows of it.
function _cell_take!(e::_CellEmitter, cid::ValueId, row::Symbol)
    e.unions[cid] === row && return e.data[cid]
    positions = _cell_bind!(e, :cell_positions,
        Expr(:call, _cell_ref(:_cell_take_positions), e.unions[cid], row, e.sorted))
    _cell_bind!(e, :cell_taken, Expr(:call, _cell_ref(:_cell_take_rows), e.data[cid], positions))
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
        _cell_forced(e, _cell_bookkeeping(e, cid)) do
            _cell_union!(e, cid)
            _cell_request_inputs!(e, cid)
        end
    end
    # A value one guarded member owns is computed only under its guard.
    for cid in sort!(processed; by = c -> d.order[d.producer[c].id])
        e.data[cid] = _cell_forced(() -> _cell_window!(e, cid), e, _cell_owner(e, cid))
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
    reads = Vector{Any}(nothing, length(r.inputs))
    for (j, input) in enumerate(r.inputs)
        icid = canon_id(d.graph, input.id)
        role = f.roles[j]
        if role === :program_gather
            program = f.literal.programs[j]
            dependencies = f.literal.dependencies[j]
            atomic = typeof(r.op).parameters[2]
            args = Any[if k == j
                    shape = icid in d.pushed ? _cell_shape!(e, icid) :
                        _cell_leaf_shape_expr(e, icid)
                    Expr(:call, GlobalRef(Base, :LinearIndices), shape)
                elseif !(k in atomic) && k in dependencies
                    _cell_operand_row!(e, canon_id(d.graph, v.id), U)
                elseif !(k in atomic)
                    # An unused lane contributes axes only. Its numerical
                    # producer may itself be awaiting these window requests.
                    lane = canon_id(d.graph, v.id)
                    shape = lane in d.pushed ? _cell_shape!(e, lane) :
                        _cell_leaf_shape_expr(e, lane)
                    positions = Expr(:call, _cell_ref(:_cell_slice_positions), shape, U)
                    Expr(:call, _cell_ref(:_cell_rows_view),
                        Expr(:call, GlobalRef(Base, :LinearIndices), shape), positions)
                elseif k in dependencies
                    _footprint_record!(canon_id(d.graph, v.id), :whole, nothing)
                    e.names[canon_id(d.graph, v.id)]
                else
                    nothing
                end for (k, v) in enumerate(r.inputs)]
            gathered = _footprint_plate_reads!(e.body, e.runtime_ops, e.runtime_recipes,
                program, atomic, args, d.plates[r.id])
            at = Expr(:call, _cell_ref(:_cell_partition_positions), gathered)
            icid in d.pushed && (at = Expr(:call, _cell_ref(:_cell_capture_rows),
                _cell_shape!(e, icid), at))
            reads[j] = _cell_bind!(e, :cell_row, _cell_canon(e, at))
            icid in d.pushed && _cell_request!(e, icid, reads[j])
        elseif role in (:leaf, :index, :column) && icid in d.pushed
            reads[j] = _cell_request!(e, icid, _cell_bind!(e, :cell_row, _cell_canon(e,
                Expr(:call, _cell_ref(:_cell_leaf_row), _cell_shape!(e, icid), U))))
        elseif role in (:gather, :linear_gather) && icid in d.pushed
            # The rows every gather index reads (`_cell_gather_rows` joins
            # those of several indices).
            ats = map(f.indices[j], f.vias[j]) do k, via
                index = canon_id(d.graph, r.inputs[k].id)
                at = _cell_operand_row!(e, index, U)
                if via == 0 && r.op isa _AuthoredPlateOp
                    return Expr(:call, _cell_ref(:_cell_partition_positions), at)
                end
                via == 0 ? Expr(:call, _cell_ref(:_cell_gather_row), at, U) :
                    Expr(:call, _cell_ref(r.op isa _AuthoredPlateOp ?
                        :_cell_capture_via_rows : :_cell_via_rows),
                        e.names[canon_id(d.graph, r.inputs[via].id)], at)
            end
            at = length(ats) == 1 ? only(ats) :
                Expr(:call, _cell_ref(:_cell_gather_rows), ats...)
            role === :gather ||
                (at = Expr(:call, _cell_ref(r.op isa _AuthoredPlateOp ?
                    :_cell_capture_rows : :_cell_linear_row), _cell_shape!(e, icid), at))
            reads[j] = _cell_request!(e, icid, _cell_bind!(e, :cell_row, _cell_canon(e, at)))
        elseif role === :factor_leaf
            # The rows of the product `A * w` this operand broadcasts: rows of `A`.
            reads[j] = _cell_bind!(e, :cell_row, _cell_canon(e, Expr(:call,
                _cell_ref(:_cell_leaf_row), _cell_part_shape!(e, r, f, j), U)))
            icid in d.pushed && _cell_request!(e, icid, reads[j])
        elseif role === :factor
            reads[j] = U
            icid in d.pushed && _cell_request!(e, icid, U)
        elseif role === :sparse_operand
            # The columns the product's rows store are the operand's rows.
            reads[j] = _cell_bind!(e, :cell_row, _cell_canon(e, Expr(:call,
                _cell_ref(:_cell_sparse_columns), e.names[canon_id(d.graph, r.inputs[1].id)], U)))
            icid in d.pushed && _cell_request!(e, icid, reads[j])
        elseif role === :row
            # A position map's argument at exactly its rows, never extruded.
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
        elseif role === :vec_source
            # The rows of the source that the requested positions lie in.
            shape = icid in d.pushed ? _cell_shape!(e, icid) : _cell_leaf_shape_expr(e, icid)
            reads[j] = _cell_bind!(e, :cell_row,
                Expr(:call, _cell_ref(:_cell_vec_rows), shape, U))
            icid in d.pushed && _cell_request!(e, icid, reads[j])
        end
    end
    e.reads[cid] = reads
end

# The rows of an unpushed source: a lazy view of inputs (`_cell_lazy_view`)
# may serve them from its parts (`_cell_lazy_rows`; a compact overlay reads
# its active vector at the slots of the rows, bookkeeping a partition's plan
# computes once per label), else the source itself is read at the rows.
function _cell_source_rows_expr(e::_CellEmitter, icid::ValueId, rows)
    d = e.pushdowns
    # An unpushed value's producer is found in the graph (`d.producer` holds
    # the pushed values' only).
    ids = producers_of(d.graph, icid)
    r = length(ids) == 1 ? d.graph.recipes[only(ids)] : nothing
    if r !== nothing && _cell_lazy_view(r.op)
        inputs = _input_list(v -> e.names[canon_id(d.graph, v.id)], r.inputs)
        served = _cell_lazy_rows(r.op, inputs, rows,
                                 (prefix, value) -> _cell_bind!(e, prefix, value),
                                 name -> name in e.constant)
        served === nothing || return served
    end
    Expr(:call, _cell_ref(:_cell_source_rows), e.names[icid], rows)
end
# The expression reading rows `rows` of a lazy view's value from its inputs'
# names, or `nothing` to read the value itself; a lazy view op's own method.
# `bind(prefix, expr)` emits a statement of its own and returns its name:
# bookkeeping bound apart from the read (the slots of the rows) is a
# statement a partition's plan computes once per label. `constant(name)`
# tells whether an input is constant per label (`plan_constant`): a view
# whose rows depend on a part that varies per call reads itself.
_cell_lazy_rows(@nospecialize(op), inputs, rows, bind, constant) = nothing

# The window of pushed value `cid` at its union rows, by its own operation.
function _cell_window!(e::_CellEmitter, cid::ValueId)
    d = e.pushdowns
    r = d.producer[cid]
    f = d.forms[r.id]
    U = e.unions[cid]
    f.kind === :vcat && return _cell_vcat_window!(e, cid, r, f, U)
    if f.kind in (:pointwise, :identity, :checked, :columns, :hcat, :range_view, :reshape, :reduce,
                  :transposed, :array_arithmetic, :position_map, :vec, :rows, :sparse_product)
        push!(e.body.args, Expr(:call, _cell_ref(:_cell_check_row),
            _cell_shape!(e, cid), U, e.sorted))
    end
    reads = e.reads[cid]
    if f.kind === :vec
        # The source's rows, by its own window when it is pushed and as a
        # bounds-checked view of its whole value otherwise.
        j = only(f.leaves)
        icid = canon_id(d.graph, r.inputs[j].id)
        icid in d.pushed || _footprint_record!(icid, :rows, reads[j])
        rows = icid in d.pushed ? _cell_take!(e, icid, reads[j]) :
            Expr(:call, _cell_ref(:_cell_factor_rows), e.names[icid], reads[j])
        shape = icid in d.pushed ? _cell_shape!(e, icid) : _cell_leaf_shape_expr(e, icid)
        return _cell_bind!(e, :cell_rows,
            Expr(:call, _cell_ref(:_cell_vec_window), rows, reads[j], U, shape))
    end
    if f.kind in (:range_view, :reshape)
        # The source's positions, by the source's own window when it is pushed
        # and as a bounds-checked view of its whole value otherwise.
        j = only(f.leaves)
        icid = canon_id(d.graph, r.inputs[j].id)
        icid in d.pushed || _footprint_record!(icid, :rows, reads[j])
        positions = icid in d.pushed ?
            Expr(:call, _cell_ref(:_cell_vector_rows), _cell_take!(e, icid, reads[j])) :
            _cell_source_rows_expr(e, icid, reads[j])
        f.kind === :range_view && return _cell_bind!(e, :cell_rows, positions)
        return _cell_bind!(e, :cell_rows, Expr(:call, _cell_ref(:_cell_reshape_window),
            positions, U, _cell_literal!(e, r, f)))
    end
    args = Any[]
    footprint_programs = Pair{Int,Any}[]
    for (j, input) in enumerate(r.inputs)
        icid = canon_id(d.graph, input.id)
        role = f.roles[j]
        if role in (:leaf, :index, :column, :program_lane)
            push!(args, icid in d.pushed ? _cell_take!(e, icid, reads[j]) :
                _cell_operand_row!(e, icid, U))
        elseif role in (:gather, :linear_gather, :program_gather) && icid in d.pushed
            shape = role === :gather ? nothing : _cell_shape!(e, icid)
            window = Any[e.data[icid], e.unions[icid], shape, e.sorted]
            # One index lane read directly: its entries' positions in the
            # window's rows are computed apart from the gather (once per label
            # with a plan), and the gather `v[I]` of that slice reads them.
            if !(r.op isa _AuthoredPlateOp) &&
                    length(f.indices[j]) == 1 && only(f.vias[j]) == 0
                index = canon_id(d.graph, r.inputs[only(f.indices[j])].id)
                if !(index in d.pushed) && haskey(e.rows, (index, U))
                    slice = e.rows[(index, U)]
                    push!(window, slice, _cell_bind!(e, :cell_positions, Expr(:call,
                        _cell_ref(:_cell_window_positions), e.unions[icid], slice, e.sorted,
                        shape === nothing ? nothing : shape)))
                end
            end
            push!(args, _cell_bind!(e, :cell_window, Expr(:call, _cell_ref(:_CellWindow),
                                                          window...)))
        elseif role === :program_gather
            _footprint_record!(icid, :gathered, reads[j])
            push!(args, e.names[icid])
        elseif role in (:gather, :linear_gather)
            # The whole source, read by the gather at its index's rows.
            for (k, via) in zip(f.indices[j], f.vias[j])
                at = _cell_operand_row!(e, canon_id(d.graph, r.inputs[k].id), U)
                via == 0 || (at = Expr(:call, _cell_ref(:_cell_via_rows),
                    e.names[canon_id(d.graph, r.inputs[via].id)], at))
                if role === :gather
                    _footprint_record!(icid, :gathered_rows, at)
                elseif r.op isa _AuthoredPlateOp
                    _footprint_record_capture!(d, icid, at)
                else
                    _footprint_record!(icid, :gathered, at)
                end
            end
            push!(args, e.names[icid])
        elseif role === :row
            # Fresh dense rows: the helper is proven for vectors (matrices) of
            # the argument's element type (`_cell_position_proof`).
            icid in d.pushed || _footprint_record!(icid, :rows, U)
            rows = icid in d.pushed ? _cell_take!(e, icid, reads[j]) :
                j in f.rowmap.matrices ?
                    Expr(:call, _cell_ref(:_cell_factor_rows), e.names[icid], U) :
                    _cell_source_rows_expr(e, icid, U)
            push!(args, _cell_bind!(e, :cell_rows, Expr(:call, _cell_ref(:_cell_dense_rows), rows)))
        elseif role in (:factor_leaf, :factor)
            icid in d.pushed || _footprint_record!(icid, :rows, reads[j])
            push!(args, icid in d.pushed ? _cell_take!(e, icid, reads[j]) :
                Expr(:call, _cell_ref(:_cell_factor_rows), e.names[icid], reads[j]))
        elseif role === :sparse_operand
            icid in d.pushed || _footprint_record!(icid, :gathered_rows, reads[j])
            push!(args, icid in d.pushed ? _cell_take!(e, icid, reads[j]) :
                Expr(:call, _cell_ref(:_cell_factor_rows), e.names[icid], reads[j]))
        else
            program = _footprint_recording() && role === :whole &&
                r.op isa _AuthoredPlateOp ? _footprint_capture_program(r.op.kernel.plan, j) : nothing
            if program === nothing
                _footprint_record!(icid, :whole, nothing)
            else
                push!(footprint_programs, j => program)
            end
            push!(args, e.names[icid])
        end
    end
    f.rowmap isa _CellRowMethod && return _cell_bind!(e, :cell_rows, Expr(:call, f.rowmap.at, U,
        Any[a isa _CellInputLiteral ? args[a.input] : a for a in f.rowmap.arguments]...))
    f.kind === :sparse_product && return _cell_bind!(e, :cell_rows, Expr(:call,
        _cell_ref(:_cell_sparse_product), args[1], U, reads[2], args[2]))
    f.rowmap isa _CellPositionMap && return _cell_position_window!(e, cid, r, f, args, U)
    f.rowmap === nothing || return _cell_helper_window!(e, f.rowmap, args)
    if r.op isa _AuthoredPlateOp
        for (j, program) in footprint_programs
            positions = _footprint_plate_reads!(e.body, e.runtime_ops, e.runtime_recipes,
                program, typeof(r.op).parameters[2], args, d.plates[r.id])
            _footprint_record!(canon_id(d.graph, r.inputs[j].id), :gathered, positions)
        end
        # The plate over these one-row arguments, with the table slots reserved
        # at its own position (`_lower_with_ops`).
        offset = d.slots[r.id]
        window = gensym(:cell_rows)
        _lower_authored_plate_native!(e.body, e.runtime_ops[1:offset],
            e.runtime_recipes[1:offset], r.op, args, d.plates[r.id], window, nothing)
        return window
    end
    if f.kind === :checked
        # Validate the whole argument once, from its type and full axes, then
        # read the argument's rows: the alias returns its argument unchanged.
        k = only(f.leaves)
        input = canon_id(d.graph, r.inputs[k].id)
        shape = input in d.pushed ? _cell_shape!(e, input) : _cell_leaf_shape_expr(e, input)
        push!(e.body.args, Expr(:call, f.literal.validator,
            _cell_original_type_expr(e, input), shape, args[f.literal.others]...))
        return _cell_bind!(e, :cell_rows,
            Expr(:call, GlobalRef(Base, :convert), valtype(only(r.outputs)), args[k]))
    end
    call = f.kind === :array_arithmetic ?
        Expr(:call, GlobalRef(Base, nameof(f.literal.f)), args[f.literal.positions]...) :
        _slot_call_expr(r, d.slots[r.id], args)
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
    push!(e.body.args, Expr(:call, _cell_ref(:_cell_check_row), shape, U, e.sorted))
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
    position = _cell_bind!(e, :cell_read, via === nothing ?
        _cell_capture_element(slice, nothing; rows) :
        Expr(:call, _cell_ref(:_cell_capture_via_rows), via, slice))
    _cell_request!(e, cid, _cell_bind!(e, :cell_row, _cell_canon(e,
        Expr(:call, _cell_ref(:_cell_capture_rows), _cell_shape!(e, cid), position))))
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
@inline _cell_partition_positions(slice) = _cell_joined_rows(identity, slice)

# The window a cell reads in place of pushed capture `cid`.
_cell_capture!(e::_CellEmitter, cid::ValueId) = _cell_bind!(e, :cell_capture,
    Expr(:call, _cell_ref(:_CellWindow), e.data[cid], e.unions[cid], _cell_shape!(e, cid),
         e.sorted))
