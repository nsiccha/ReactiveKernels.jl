# Values a plate cell reads at its own position, computed at that position
# only (`plate_cell`, cells.jl; snag `rkppl-per-subjec-a3d6e6d8`, increment 1b).
#
# A cell that captures a whole array and reads it at its batched element,
# `log_k[i]` in `plate(i, ...) do i ... exp(log_k[i]) end`, needs one row of
# it. The same is true when `log_k` is itself a batched input to the cell's
# plate. When that array's producer computes it row by row, the native lowering
# runs the producer's own operation on the one row instead: the producer is
# given views of its row-aligned operands at that row, and the cell is given a
# `_CellWindow` holding the result, which answers reads of that row and throws
# `BoundsError` for any other. The rewrite applies to these source shapes,
# read off the closure's native body (`_kernel_native_body`):
#
# - a dotted expression (one fused broadcast) whose operands are recipe
#   inputs, numeric literals, or gathers `A[I]` / `A[I, 1, ...]` of an input by
#   an integer vector input `I` (integer literals after it);
# - such a gather on its own;
# - a product `A * W...` (a source expression, or a bare call that keeps
#   Base's `*` as its operation) whose left factor `A` is an input array:
#   row `r` of the product is `A[r:r, ...] * W...`. Its full axes, which a
#   dotted reader and the cell's linear reads need, are known for a
#   matrix-vector product `A * v` (`(axes(A, 1),)`); any other product is read
#   only by gathers;
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
# `r` of a gather by `I` reads row `I[r]` of its source. Those rows are again
# computed at that row when their producers have one of these shapes, and are
# otherwise computed whole and indexed. A value is computed only at rows when
# every reader of it reads rows this way, so the program computes no value
# twice in different ways. The cell's batched element must be provably an
# integer (a linear position), and each gather index provably an integer
# vector (`_cell_static_type`).
#
# Shapes and bounds keep the whole program's checks for this cell: each
# dotted expression's full broadcast shape is computed from its operands' axes
# (so mismatched operands still throw `DimensionMismatch`), the row is checked
# against it, factor rows are bounds-checked views, and window reads check
# their row. Values at other rows are not computed, so an error that only
# another row raises (an out-of-range gather index elsewhere, a domain error
# in another subject's predictor) is not raised, as the cell itself does not
# run other cells. Everything else, including the tensorized lowering, the
# operation table, and `on_error = :ignore` twins, is unchanged.

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

# The position along `data`'s first axis of row `row`; `at` is the read.
@inline function _cell_local_row(w::_CellWindow{<:Any,Int}, row, at)
    row == w.rows || throw(BoundsError(w, at))
    firstindex(w.data, 1)
end
@inline function _cell_local_row(w::_CellWindow{<:Any,Vector{Int}}, row, at)
    k = findfirst(==(row), w.rows)
    k === nothing && throw(BoundsError(w, at))
    firstindex(w.data, 1) + k - 1
end

# Whether gather index `I` reads exactly a vector value's window rows, in
# order: then the gather is the window's data.
@inline _cell_reads_rows(w::_CellWindow{<:Any,Int}, I) = false
@inline _cell_reads_rows(w::_CellWindow{<:Any,Vector{Int}}, I) =
    isone(firstindex(w.data, 1)) && I == w.rows

# Linear reads `v[q]`, as the cell and single-index gathers make them.
@inline function Base.getindex(w::_CellWindow{<:Any,<:Any,Tuple{}}, q::Integer)
    q == 1 || throw(BoundsError(w, q))
    w.data[]
end
@inline function Base.getindex(w::_CellWindow{<:Any,<:Any,<:Tuple{Any,Vararg{Any}}},
                               q::Integer)
    position = Tuple(CartesianIndices(w.shape)[q])
    w.data[_cell_local_row(w, first(position), q), Base.tail(position)...]
end
@inline Base.getindex(w::_CellWindow{<:Any,<:Any,<:Tuple}, I::AbstractVector{<:Integer}) =
    [w[q] for q in I]
# A vector value with one-based axes: linear position `q` is row `q`.
@inline function Base.getindex(w::_CellWindow{<:Any,<:Any,<:Tuple{Base.OneTo}},
                               I::AbstractVector{<:Integer})
    _cell_reads_rows(w, I) && return w.data[:]
    [w[q] for q in I]
end

# Row reads `v[i, j...]` and gathers `v[I, j...]`.
@inline Base.getindex(w::_CellWindow, i::Integer, j::Integer, rest::Integer...) =
    w.data[_cell_local_row(w, i, (i, j, rest...)), j, rest...]
@inline function Base.getindex(w::_CellWindow, I::AbstractVector{<:Integer}, j::Integer,
                               rest::Integer...)
    _cell_reads_rows(w, I) && return w.data[:, j, rest...]
    [w[i, j, rest...] for i in I]
end

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

# ---------------------------------------------------------------- analysis

# How a source recipe computes its value row by row. `roles[j]` is how input
# `j` is read: `:leaf` (a broadcast operand), `:index` (a gather index, and
# possibly a broadcast operand), `:gather` / `:linear_gather` (the source of
# `A[I, j...]` / `A[I]`, by input `gather_index[j]`), `:column` (a broadcast
# operand read as a column `A[:, k...]`), `:factor_leaf` (the left factor of a
# broadcast operand `A * w`, `w` input `gather_index[j]`), `:factor` (a
# product's left factor), `:scalar` (an array of numbers read at literal
# positions, `a[1]`, so whole), or `:whole`. `leaves` are the inputs whose
# axes make up a pointwise value's full shape. `operand` is the input `v` of a
# product `A * v` of exactly two inputs (0 otherwise): with a matrix `A` and a
# vector `v`, its full axes are `(axes(A, 1),)`.
struct _CellForm
    kind::Symbol
    roles::Vector{Symbol}
    gather_index::Vector{Int}
    leaves::Vector{Int}
    operand::Int
end

# Expression heads that bind no names; any other head declines the rewrite.
const _CELL_PLAIN_HEADS = (:call, :ref, :tuple, :., Symbol("'"), :vect, :hcat,
    :vcat, :row, :nrow, :ncat, :parameters, :kw, :comparison, :&&, :||, :if,
    :curly, :..., :string, :(::))

function _cell_native(r::Recipe)
    closure = _lane_source_closure(r)
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

function _cell_form(r::Recipe)
    length(r.outputs) == 1 && !r.effectful || return nothing
    r.op isa _AuthoredPlateOp && return _cell_plate_form(r)
    if r.op === Base.:* && length(r.inputs) >= 2
        # A bare exact call `A * B...` keeps Base's `*` itself as its
        # operation (`_kernel_operation_body`); it is called positionally.
        n = length(r.inputs)
        roles = fill(:whole, n)
        roles[1] = :factor
        return _CellForm(:product, roles, zeros(Int, n), Int[], n == 2 ? 2 : 0)
    end
    r.op isa _KernelSourceOp || return nothing
    analyzed = _cell_native(r)
    analyzed === nothing && return nothing
    native, params, mod = analyzed
    position = Dict{Symbol,Int}(name => j for (j, name) in enumerate(params))
    uses = [Set{Symbol}() for _ in params]
    gathers = Dict{Int,Tuple{Int,Bool}}()
    products = Dict{Int,Int}()
    leaves = Int[]
    param(x) = x isa Symbol && haskey(position, x)
    function gather!(ex)
        length(ex.args) >= 2 || return false
        source, index = ex.args[1], ex.args[2]
        haskey(position, source) && haskey(position, index) || return false
        all(i -> i isa Int, ex.args[3:end]) || return false
        linear = length(ex.args) == 2
        key = (position[index], linear)
        get!(gathers, position[source], key) == key || return false
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
    elseif native isa Expr && native.head === :ref
        gather!(native) || return nothing
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
    elseif native isa Expr && native.head === :call && length(native.args) >= 3 &&
           _cell_base_product(native.args[1], mod, params) &&
           native.args[2] isa Symbol && haskey(position, native.args[2])
        kind = :product
        push!(uses[position[native.args[2]]], :factor)
        length(native.args) == 3 && native.args[3] isa Symbol &&
            haskey(position, native.args[3]) && (operand = position[native.args[3]])
        all(arg -> _cell_whole_uses!(uses, arg, position, :whole), native.args[3:end]) ||
            return nothing
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
    gather_index = Int[first(get(gathers, j, (get(products, j, 0), false)))
                       for j in eachindex(params)]
    _CellForm(kind, roles, gather_index, kind === :hcat ? leaves : unique!(leaves), operand)
end

# An authored plate's value, row by row: its batched arguments are broadcast
# operands, a shared argument read only at one batched integer argument's
# element is gathered by it, and any other shared argument is read whole.
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
    wants = Set(canon_id(ig, w.id) for w in inner.want)
    for k in atomic
        roots[k] in wants && continue
        read = _cell_capture_reader(inner, roots, atomic, roots[k])
        read === nothing && continue
        b, via = read
        via == 0 || continue
        roles[k] = :linear_gather
        gather_index[k] = b
        roles[b] = :index
    end
    _CellForm(:pointwise, roles, gather_index, leaves, 0)
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
ordinary batched argument positions.
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
end

# Whether a producer reads an input in `role` at its own rows.
_cell_row_role(role) = role in (:leaf, :gather, :linear_gather, :column, :factor_leaf, :factor)

function _cell_pushdowns(p::Plan)
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
        f = _cell_form(r)
        f === nothing || (forms[r.id] = f)
        f
    end
    # A read needs the value's full shape unless it reads rows `v[I, j...]`
    # or takes rows of a product's left factor.
    needs_shape(role) = !(role in (:gather, :factor))
    # Full axes are known for pointwise values, column reads and `hcat`s, and
    # for a product of two operands whose right operand is an array or a number
    # (`_cell_product_shape`) and whose left factor's axes are known.
    function shaped(cid)
        f = forms[only(producers[cid]).id]
        f.kind in (:pointwise, :columns, :hcat) && return true
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
            true
        end
    end
    captures = Dict{Int,Dict{Int,Tuple{Int,Int}}}()
    batched = Dict{Int,Set{Int}}()
    pushed = Set{ValueId}()
    work = ValueId[]
    function reach!(cid, role)
        cid in pushed && return
        fits(cid, role) && operands_fit(cid) || return
        push!(pushed, cid)
        push!(work, cid)
    end
    for r in p.recipes
        r.op isa _AuthoredPlateCellOp || continue
        positions = _cell_capture_positions(r, static_type)
        isempty(positions) || (captures[r.id] = positions)
        foreach(k -> reach!(cidof(r.inputs[1 + k]), :linear_gather), keys(positions))
        atomic = typeof(r.op.plate).parameters[2]
        lanes = Set(k for k in 1:length(r.inputs)-1 if !(k in atomic))
        batched[r.id] = lanes
        foreach(k -> reach!(cidof(r.inputs[1 + k]), :leaf), lanes)
    end
    while !isempty(work)
        cid = pop!(work)
        r = only(producers[cid])
        f = forms[r.id]
        for (j, input) in enumerate(r.inputs)
            _cell_row_role(f.roles[j]) && reach!(cidof(input), f.roles[j])
        end
    end
    # Keep a value at rows only while every reader reads it at rows.
    changed = true
    while changed
        changed = false
        for cid in collect(pushed)
            ok = all(get(readers, cid, Recipe[])) do reader
                slots = findall(v -> cidof(v) == cid, collect(reader.inputs))
                if reader.op isa _AuthoredPlateCellOp
                    positions = get(captures, reader.id, Dict{Int,Tuple{Int,Int}}())
                    lanes = get(batched, reader.id, Set{Int}())
                    return all(slots) do s
                        s > 1 && ((haskey(positions, s - 1) &&
                            fits(cid, :linear_gather)) ||
                            (s - 1 in lanes && fits(cid, :leaf)))
                    end
                end
                length(reader.outputs) == 1 && cidof(only(reader.outputs)) in pushed ||
                    return false
                f = forms[reader.id]
                all(s -> _cell_row_role(f.roles[s]) && fits(cid, f.roles[s]), slots)
            end
            ok && continue
            delete!(pushed, cid)
            changed = true
        end
    end
    isempty(pushed) && return nothing
    recipes = Set(only(producers[cid]).id for cid in pushed)
    _CellPushdowns(g, pushed, recipes,
        Dict(cid => only(producers[cid]) for cid in pushed), forms, captures, batched,
        Dict{Int,Int}(), Dict{Int,Any}())
end

# ---------------------------------------------------------------- emission

# Emits, into `body`, the rows of pushed values one cell reads. Shapes are
# memoized per value, operand rows and windows per (value, row binding).
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
    windows::Dict{Tuple{ValueId,Symbol},Symbol}
end
_CellEmitter(body, pushdowns, names, runtime_ops, runtime_recipes) =
    _CellEmitter(body, pushdowns, names, runtime_ops, runtime_recipes,
        Dict{ValueId,Symbol}(), Dict{Tuple{ValueId,Symbol},Symbol}(),
        Dict{Tuple{ValueId,Symbol},Symbol}())

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
    parts = Any[_cell_part_shape!(e, r, f, j) for j in f.leaves]
    shape = f.kind === :columns ?
        Expr(:call, _cell_ref(:_cell_column_shape), only(parts)) :
        f.kind === :hcat ? Expr(:call, _cell_ref(:_cell_hcat_shape), Expr(:tuple, parts...)) :
        Expr(:call, _cell_ref(:_plate_broadcast_shape), Expr(:tuple, parts...))
    e.shapes[cid] = _cell_bind!(e, :cell_shape, shape)
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

function _cell_operand_row!(e::_CellEmitter, cid::ValueId, row::Symbol)
    get!(e.rows, (cid, row)) do
        _cell_bind!(e, :cell_operand, Expr(:call, _cell_ref(:_cell_row_slice),
            Expr(:call, GlobalRef(Base, :broadcastable), e.names[cid]), row))
    end
end

# The rows `row:row` of pushed value `cid`, by its own operation.
function _cell_window!(e::_CellEmitter, cid::ValueId, row::Symbol)
    haskey(e.windows, (cid, row)) && return e.windows[(cid, row)]
    d = e.pushdowns
    r = d.producer[cid]
    f = d.forms[r.id]
    if f.kind in (:pointwise, :columns, :hcat)
        push!(e.body.args, Expr(:call, _cell_ref(:_cell_check_row),
            _cell_shape!(e, cid), row))
    end
    args = Any[]
    for (j, input) in enumerate(r.inputs)
        icid = canon_id(d.graph, input.id)
        role = f.roles[j]
        if role in (:leaf, :index, :column)
            if icid in d.pushed
                inner = _cell_bind!(e, :cell_row, Expr(:call, _cell_ref(:_cell_leaf_row),
                    _cell_shape!(e, icid), row))
                push!(args, _cell_window!(e, icid, inner))
            else
                push!(args, _cell_operand_row!(e, icid, row))
            end
        elseif role in (:gather, :linear_gather) && icid in d.pushed
            index = canon_id(d.graph, r.inputs[f.gather_index[j]].id)
            at = Expr(:call, _cell_ref(:_cell_gather_row), _cell_operand_row!(e, index, row),
                row)
            shape = role === :gather ? nothing : _cell_shape!(e, icid)
            role === :gather ||
                (at = Expr(:call, _cell_ref(:_cell_linear_row), shape, at))
            source_row = _cell_bind!(e, :cell_row, at)
            window = _cell_window!(e, icid, source_row)
            push!(args, _cell_bind!(e, :cell_window, Expr(:call,
                _cell_ref(:_CellWindow), window, source_row, shape)))
        elseif role === :factor_leaf
            # The rows of the product `A * w` this operand broadcasts: rows of `A`.
            inner = _cell_bind!(e, :cell_row, Expr(:call, _cell_ref(:_cell_leaf_row),
                _cell_part_shape!(e, r, f, j), row))
            push!(args, icid in d.pushed ? _cell_window!(e, icid, inner) :
                Expr(:call, _cell_ref(:_cell_factor_rows), e.names[icid], inner))
        elseif role === :factor && icid in d.pushed
            push!(args, _cell_window!(e, icid, row))
        elseif role === :factor
            push!(args, Expr(:call, _cell_ref(:_cell_factor_rows), e.names[icid], row))
        else
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
        return e.windows[(cid, row)] = window
    end
    call = Expr(:call, Expr(:ref, _OPS_ARG, d.slots[r.id]), args...)
    e.windows[(cid, row)] = _cell_bind!(e, :cell_rows, call)
end

# The window a cell reads in place of pushed capture `cid`, at linear
# position `only(slice)` (the cell's batched element), or at `via[only(slice)]`;
# either is one position or an integer vector of them.
function _cell_capture!(e::_CellEmitter, cid::ValueId, slice, via = nothing)
    element = Expr(:call, GlobalRef(Base, :only), slice)
    via === nothing || (element = Expr(:call, GlobalRef(Base, :getindex), via, element))
    position = _cell_bind!(e, :cell_read, element)
    shape = _cell_shape!(e, cid)
    row = _cell_bind!(e, :cell_row, Expr(:call, _cell_ref(:_cell_linear_row), shape, position))
    window = _cell_window!(e, cid, row)
    _cell_bind!(e, :cell_capture, Expr(:call, _cell_ref(:_CellWindow), window, row, shape))
end
