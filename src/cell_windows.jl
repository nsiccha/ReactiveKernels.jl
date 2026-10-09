# Values a plate cell reads at its own position, computed at that position
# only (`plate_cell`, cells.jl; snag `rkppl-per-subjec-a3d6e6d8`, increment 1b).
#
# A cell that captures a whole array and reads it at its batched element,
# `log_k[i]` in `plate(i, ...) do i ... exp(log_k[i]) end`, needs one row of
# it. When that array's producer computes it row by row, the native lowering
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
#   only by gathers.
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
    _CellWindow(data, row, shape)

Rows of a value read by one cell. `data` holds the value's rows `row:row`,
every later dimension whole; `shape` is the value's full axes, or `nothing`
when only row reads `v[I, j...]` are made of it. Reading any other row throws
`BoundsError`.
"""
struct _CellWindow{D,S}
    data::D
    row::Int
    shape::S
end

@inline _cell_local_row(data) = firstindex(data, 1)

# Linear reads `v[q]`, as the cell and single-index gathers make them.
@inline function Base.getindex(w::_CellWindow{<:Any,Tuple{}}, q::Integer)
    q == 1 || throw(BoundsError(w, q))
    w.data[]
end
@inline function Base.getindex(w::_CellWindow{<:Any,<:Tuple{Any,Vararg{Any}}}, q::Integer)
    position = Tuple(CartesianIndices(w.shape)[q])
    first(position) == w.row || throw(BoundsError(w, q))
    w.data[_cell_local_row(w.data), Base.tail(position)...]
end
@inline Base.getindex(w::_CellWindow{<:Any,<:Tuple}, I::AbstractVector{<:Integer}) =
    [w[q] for q in I]

# Row reads `v[i, j...]` and gathers `v[I, j...]`.
@inline function Base.getindex(w::_CellWindow, i::Integer, j::Integer, rest::Integer...)
    i == w.row || throw(BoundsError(w, (i, j, rest...)))
    w.data[_cell_local_row(w.data), j, rest...]
end
@inline Base.getindex(w::_CellWindow, I::AbstractVector{<:Integer}, j::Integer,
                      rest::Integer...) = [w[i, j, rest...] for i in I]

Base.show(io::IO, w::_CellWindow) =
    print(io, "cell window at row ", w.row, " of ", summary(w.data))

# The full broadcast axes of an operand (`Base.broadcastable` as `broadcasted`
# applies it); folded across operands by `_plate_broadcast_shape`.
@inline _cell_leaf_shape(arg) = axes(Base.broadcastable(arg))

# Row `row` of a broadcast operand: a one-row view, or the operand itself when
# broadcasting extrudes its first dimension. Always a `UnitRange` view (a
# union of range types defeats native Reverse type analysis).
@inline _cell_row_slice(arg, row) = arg
@inline _cell_row_slice(arg::Tuple, row) = length(arg) == 1 ? arg : (arg[row],)
@inline function _cell_row_slice(arg::AbstractArray{<:Any,N}, row) where {N}
    N == 0 && return arg
    axis = axes(arg, 1)
    position = length(axis) == 1 ? first(axis) : row
    view(arg, position:position, ntuple(_ -> Colon(), Val(N - 1))...)
end

# The full axes of a matrix-vector product `A * v`.
@inline _cell_factor_shape(A::AbstractMatrix) = (axes(A, 1),)

# Row `row` of a product's left factor; bounds-checked, never extruded.
@inline _cell_factor_rows(arg::AbstractArray{<:Any,N}, row) where {N} =
    view(arg, row:row, ntuple(_ -> Colon(), Val(N - 1))...)

# A dotted expression's row must lie in its full first axis.
@inline function _cell_check_row(shape::Tuple, row)
    valid = isempty(shape) ? row == 1 : checkindex(Bool, first(shape), row)
    valid || throw(BoundsError(CartesianIndices(shape), row))
    row
end

# The row of a computed operand that row `row` of its reader needs.
@inline _cell_leaf_row(shape::Tuple{}, row) = 1
@inline _cell_leaf_row(shape::Tuple, row) =
    length(first(shape)) == 1 ? Int(first(first(shape))) : row

# The row holding linear position `q` of a value with full axes `shape`.
@inline function _cell_linear_row(shape::Tuple{}, q::Integer)
    q == 1 || throw(BoundsError(CartesianIndices(shape), q))
    1
end
@inline _cell_linear_row(shape::Tuple, q::Integer) =
    Int(first(Tuple(CartesianIndices(shape)[q])))

# The one index a row of a gather index holds.
@inline _cell_gather_row(index) = Int(only(index))

# ---------------------------------------------------------------- analysis

# How a source recipe computes its value row by row. `roles[j]` is how input
# `j` is read: `:leaf` (a broadcast operand), `:index` (a gather index, and
# possibly a broadcast operand), `:gather` / `:linear_gather` (the source of
# `A[I, j...]` / `A[I]`, by input `gather_index[j]`), `:factor` (a product's
# left factor), or `:whole`. `leaves` are the inputs whose axes make up a
# pointwise value's full shape. `operand` is the input `v` of a product
# `A * v` of exactly two inputs (0 otherwise): with a matrix `A` and a vector
# `v`, its full axes are `(axes(A, 1),)`.
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
    leaves = Int[]
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
        ex isa Expr && ex.head === :ref && return gather!(ex)
        false
    end
    lazy!(ex) = _cell_free_of(ex.args[2], params) && all(operand!, ex.args[3:end])
    kind = :pointwise
    operand = 0
    if _cell_materialize(native)
        lazy = native.args[2]
        _cell_broadcasted(lazy) && lazy!(lazy) || return nothing
    elseif native isa Expr && native.head === :ref
        gather!(native) || return nothing
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
    gather_index = Int[first(get(gathers, j, (0, false))) for j in eachindex(params)]
    _CellForm(kind, roles, gather_index, unique!(leaves), operand)
end

# `*` as the closure resolves it is Base's.
function _cell_base_product(callee, mod, params)
    callee isa GlobalRef && return isdefined(callee.mod, callee.name) &&
        getglobal(callee.mod, callee.name) === Base.:*
    callee === :* && !(callee in params) && mod isa Module || return false
    isdefined(mod, :*) && getglobal(mod, :*) === Base.:*
end

# The positions `k` of the cell's shared arguments that its body reads only as
# `v[q]`, `q` the element of one batched argument `b`, as `k => b`.
function _cell_capture_positions(r::Recipe, static_type)
    plate = (r.op::_AuthoredPlateCellOp).plate
    atomic = typeof(plate).parameters[2]
    inner = plate.kernel.plan
    ig = inner.graph
    roots = Int[canon_id(ig, v.id) for v in inner.have]
    length(roots) == length(r.inputs) - 1 && allunique(roots) || return Dict{Int,Int}()
    wants = Set(canon_id(ig, w.id) for w in inner.want)
    captures = Dict{Int,Int}()
    for k in atomic
        roots[k] in wants && continue
        b = _cell_capture_reader(inner, roots, atomic, roots[k])
        b === nothing && continue
        static_type(r.inputs[1 + b]) <: AbstractArray{<:Integer} || continue
        captures[k] = b
    end
    captures
end

function _cell_capture_reader(inner::Plan, roots, atomic, cid)
    ig = inner.graph
    batched = nothing
    for recipe in inner.recipes
        slots = findall(v -> canon_id(ig, v.id) == cid, collect(recipe.inputs))
        isempty(slots) && continue
        recipe.op isa _KernelSourceOp || return nothing
        analyzed = _cell_native(recipe)
        analyzed === nothing && return nothing
        native, params, _ = analyzed
        reads = Symbol[]
        _cell_capture_reads!(reads, native, Set(params[slots])) || return nothing
        for q in reads
            j = findfirst(==(q), params)
            j === nothing && return nothing
            b = findfirst(==(canon_id(ig, recipe.inputs[j].id)), roots)
            (b === nothing || b in atomic) && return nothing
            batched === nothing && (batched = b)
            batched == b || return nothing
        end
    end
    batched
end

# Whether `ex` reads the names `captured` only as `v[q]` with `q` a name;
# each such `q` is recorded.
function _cell_capture_reads!(reads, ex, captured)
    ex isa Symbol && return !(ex in captured)
    ex isa Expr || return true
    ex.head in _CELL_PLAIN_HEADS || return false
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] isa Symbol &&
       ex.args[1] in captured
        ex.args[2] isa Symbol && !(ex.args[2] in captured) || return false
        push!(reads, ex.args[2])
        return true
    end
    all(arg -> _cell_capture_reads!(reads, arg, captured), ex.args)
end

"""
    _CellPushdowns

The plan-wide pushdown decision for `_lower_with_ops`: `recipes` are the
producers computed only at their readers' rows (their table slots are kept and
recorded in `slots` as lowering reaches them), and `captures[cell recipe id]`
maps each shared argument position read at the cell's own row to the batched
argument holding that row.
"""
struct _CellPushdowns
    graph::Graph
    pushed::Set{Int}
    recipes::Set{Int}
    producer::Dict{Int,Recipe}
    forms::Dict{Int,_CellForm}
    captures::Dict{Int,Dict{Int,Int}}
    slots::Dict{Int,Int}
end

function _cell_pushdowns(p::Plan)
    any(r -> r.op isa _AuthoredPlateCellOp, p.recipes) || return nothing
    g = p.graph
    cidof(v) = canon_id(g, v.id)
    producers = Dict{Int,Vector{Recipe}}()
    readers = Dict{Int,Vector{Recipe}}()
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
    # A read needs the value's full shape unless it reads rows `v[I, j...]`.
    needs_shape(role) = role !== :gather
    # Full axes are known for pointwise values and matrix-vector products.
    function shaped(cid)
        f = forms[only(producers[cid]).id]
        f.kind === :pointwise && return true
        f.operand == 0 && return false
        inputs = only(producers[cid]).inputs
        static_type(inputs[findfirst(==(:factor), f.roles)]) <: AbstractMatrix &&
            static_type(inputs[f.operand]) <: AbstractVector
    end
    fits(cid, role) = form(cid) !== nothing && (!needs_shape(role) || shaped(cid))
    # Static operand requirements: integer vector indices, array factors.
    function operands_fit(cid)
        r = only(producers[cid])
        all(zip(forms[r.id].roles, r.inputs)) do (role, input)
            role === :index ? static_type(input) <: AbstractVector{<:Integer} :
            role === :factor ? static_type(input) <: AbstractArray : true
        end
    end
    captures = Dict{Int,Dict{Int,Int}}()
    pushed = Set{Int}()
    work = Int[]
    function reach!(cid, role)
        cid in pushed && return
        fits(cid, role) && operands_fit(cid) || return
        push!(pushed, cid)
        push!(work, cid)
    end
    for r in p.recipes
        r.op isa _AuthoredPlateCellOp || continue
        positions = _cell_capture_positions(r, static_type)
        isempty(positions) && continue
        captures[r.id] = positions
        foreach(k -> reach!(cidof(r.inputs[1 + k]), :linear_gather), keys(positions))
    end
    while !isempty(work)
        cid = pop!(work)
        r = only(producers[cid])
        f = forms[r.id]
        for (j, input) in enumerate(r.inputs)
            f.roles[j] in (:leaf, :gather, :linear_gather) && reach!(cidof(input), f.roles[j])
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
                    positions = get(captures, reader.id, Dict{Int,Int}())
                    return fits(cid, :linear_gather) &&
                        all(s -> s > 1 && haskey(positions, s - 1), slots)
                end
                length(reader.outputs) == 1 && cidof(only(reader.outputs)) in pushed ||
                    return false
                f = forms[reader.id]
                all(s -> f.roles[s] in (:leaf, :gather, :linear_gather) &&
                         fits(cid, f.roles[s]), slots)
            end
            ok && continue
            delete!(pushed, cid)
            changed = true
        end
    end
    isempty(pushed) && return nothing
    recipes = Set(only(producers[cid]).id for cid in pushed)
    _CellPushdowns(g, pushed, recipes,
        Dict(cid => only(producers[cid]) for cid in pushed), forms, captures,
        Dict{Int,Int}())
end

# ---------------------------------------------------------------- emission

# Emits, into `body`, the rows of pushed values one cell reads. Shapes are
# memoized per value, operand rows and windows per (value, row binding).
struct _CellEmitter
    body::Expr
    pushdowns::_CellPushdowns
    names::Dict{Int,Symbol}
    shapes::Dict{Int,Symbol}
    rows::Dict{Tuple{Int,Symbol},Symbol}
    windows::Dict{Tuple{Int,Symbol},Symbol}
end
_CellEmitter(body, pushdowns, names) = _CellEmitter(body, pushdowns, names,
    Dict{Int,Symbol}(), Dict{Tuple{Int,Symbol},Symbol}(), Dict{Tuple{Int,Symbol},Symbol}())

_cell_ref(name) = GlobalRef(@__MODULE__, name)

function _cell_bind!(e::_CellEmitter, prefix::Symbol, value)
    name = gensym(prefix)
    push!(e.body.args, Expr(:(=), name, value))
    name
end

function _cell_shape!(e::_CellEmitter, cid::Int)
    haskey(e.shapes, cid) && return e.shapes[cid]
    d = e.pushdowns
    r = d.producer[cid]
    f = d.forms[r.id]
    if f.kind === :product
        factor = canon_id(d.graph, r.inputs[findfirst(==(:factor), f.roles)].id)
        return e.shapes[cid] = _cell_bind!(e, :cell_shape,
            Expr(:call, _cell_ref(:_cell_factor_shape), e.names[factor]))
    end
    parts = Any[]
    for j in f.leaves
        input = canon_id(d.graph, r.inputs[j].id)
        push!(parts, input in d.pushed ? _cell_shape!(e, input) :
            Expr(:call, _cell_ref(:_cell_leaf_shape), e.names[input]))
    end
    e.shapes[cid] = _cell_bind!(e, :cell_shape,
        Expr(:call, _cell_ref(:_plate_broadcast_shape), Expr(:tuple, parts...)))
end

function _cell_operand_row!(e::_CellEmitter, cid::Int, row::Symbol)
    get!(e.rows, (cid, row)) do
        _cell_bind!(e, :cell_operand, Expr(:call, _cell_ref(:_cell_row_slice),
            Expr(:call, GlobalRef(Base, :broadcastable), e.names[cid]), row))
    end
end

# The rows `row:row` of pushed value `cid`, by its own operation.
function _cell_window!(e::_CellEmitter, cid::Int, row::Symbol)
    haskey(e.windows, (cid, row)) && return e.windows[(cid, row)]
    d = e.pushdowns
    r = d.producer[cid]
    f = d.forms[r.id]
    if f.kind === :pointwise
        push!(e.body.args, Expr(:call, _cell_ref(:_cell_check_row),
            _cell_shape!(e, cid), row))
    end
    args = Any[]
    for (j, input) in enumerate(r.inputs)
        icid = canon_id(d.graph, input.id)
        role = f.roles[j]
        if role in (:leaf, :index)
            if icid in d.pushed
                inner = _cell_bind!(e, :cell_row, Expr(:call, _cell_ref(:_cell_leaf_row),
                    _cell_shape!(e, icid), row))
                push!(args, _cell_window!(e, icid, inner))
            else
                push!(args, _cell_operand_row!(e, icid, row))
            end
        elseif role in (:gather, :linear_gather) && icid in d.pushed
            index = canon_id(d.graph, r.inputs[f.gather_index[j]].id)
            at = Expr(:call, _cell_ref(:_cell_gather_row), _cell_operand_row!(e, index, row))
            shape = role === :gather ? nothing : _cell_shape!(e, icid)
            role === :gather ||
                (at = Expr(:call, _cell_ref(:_cell_linear_row), shape, at))
            source_row = _cell_bind!(e, :cell_row, at)
            window = _cell_window!(e, icid, source_row)
            push!(args, _cell_bind!(e, :cell_window, Expr(:call,
                _cell_ref(:_CellWindow), window, source_row, shape)))
        elseif role === :factor
            push!(args, Expr(:call, _cell_ref(:_cell_factor_rows), e.names[icid], row))
        else
            push!(args, e.names[icid])
        end
    end
    call = Expr(:call, Expr(:ref, _OPS_ARG, d.slots[r.id]), args...)
    e.windows[(cid, row)] = _cell_bind!(e, :cell_rows, call)
end

# The window a cell reads in place of pushed capture `cid`, at linear
# position `only(slice)` (the cell's batched element).
function _cell_capture!(e::_CellEmitter, cid::Int, slice)
    position = _cell_bind!(e, :cell_read, Expr(:call, GlobalRef(Base, :only), slice))
    shape = _cell_shape!(e, cid)
    row = _cell_bind!(e, :cell_row, Expr(:call, _cell_ref(:_cell_linear_row), shape, position))
    window = _cell_window!(e, cid, row)
    _cell_bind!(e, :cell_capture, Expr(:call, _cell_ref(:_CellWindow), window, row, shape))
end
