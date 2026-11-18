# Bind-time shapes of live arrays in plate cells (snag `butterfly-positi-b8e56042`).
#
# A cell often builds an array from a live scalar and bound dimensions, and
# then asks for its size: a whole-event law receives `hcat(fill(theta, n),
# never)` for each event, checks its rows against the event's observations and
# selects a row per position with `ifelse(size(logits, 1) == 1, 1, i)`. The
# array's values are live, but its shape is fixed by `n` and `never`, which are
# bound. While the size query reads the live array it is live too, and so is
# everything computed from it: the row index of every position, and each read
# through that index. Plate partial evaluation then cannot cache the index,
# group equal positions (`nested_duplicate_cells.jl`), or tell that a position
# reads the same row as the next.
#
# Here, per binding, before a plate cell's recipes are split into bind-time and
# per-evaluation parts (`_partial_plate_analysis`, `_partial_nested_shared_recipe`),
# a size query whose array has a shape computable from bind-time values alone
# is replaced by a query of that shape (`_BoundShapeQuery`). The query reads
# only bind-time values, so it is evaluated once per lane at binding like any
# other data-only recipe. Shapes are followed through
#
# - `fill(x, d...)` with integer dimensions `d` that are literals, bind-time
#   values, or `length`/`size(_, k)` of bind-time values (the filled value may
#   be live);
# - `hcat(a...)` and `[a b ...]` of numbers (live, bound or written in the
#   source), vectors and matrices whose shapes are known;
# - a check that returns one argument or throws (`cond ? x : throw(...)`,
#   with `if`/`else` too), a recipe that returns its argument, and `identity`;
# - any bind-time value, whose shape is its own;
#
# within one recipe's expression as across recipes. The queries are
# `size(x, d)` with a literal `d`, and `length(x)`.
#
# Values are unchanged. Every recognized producer has exactly the shape the
# query computes whenever it returns: a `fill` has its dimensions, an `hcat`
# the rows of its operands (it throws when they differ, so a size read after it
# is never reached with other operands) and its summed columns, and a check
# returns its argument. Where the live producer throws, the bind-time query
# does not: it computes the shape the producer would have had, and the
# producer still throws in every evaluation that reaches it, as before.

# A shape whose parts are bind-time values, over the query's argument
# positions:
#   (:size, i)          the shape of argument `i` (a number's shape is `()`)
#   (:dims, items)      one dimension per item: an `Int` literal, `(:arg, i)`
#                       the integer argument `i`, `(:length, i)` its length or
#                       `(:dim, i, k)` its size along `k`
#   (:hcat, parts)      the rows of the first part, the columns of all parts
#   (:scalar,)          a number (a 1x1 block of an `hcat`)
# `D` is the queried dimension, `0` for `length`. Both are type parameters, so
# a query's result type is inferred as `Int`.
struct _BoundShapeQuery{T,D} end
@inline (::_BoundShapeQuery{T,D})(args::Vararg{Any,N}) where {T,D,N} =
    _bound_shape_dim(_bound_shape(T, args), D)::Int
_readable_callee(::_BoundShapeQuery) = :bound_shape
_opname(::_BoundShapeQuery{T,D}) where {T,D} = D == 0 ? "bound_length" : "bound_size_$D"

function _bound_shape(tree::Tuple, args::Tuple)
    kind = first(tree)
    if kind === :size
        x = args[tree[2]]
        return x isa AbstractArray ? size(x) : ()
    elseif kind === :dims
        return map(item -> _bound_shape_item(item, args), tree[2])
    elseif kind === :scalar
        return ()
    elseif kind === :hcat
        parts = map(part -> _bound_shape(part, args), tree[2])
        rows = isempty(parts) ? 0 : _bound_shape_rows(first(parts))
        columns = sum(_bound_shape_columns, parts; init = 0)
        return (rows, columns)
    end
    throw(ArgumentError("internal: unknown bound shape $(repr(kind))"))
end
_bound_shape_item(item::Int, args) = item
function _bound_shape_item(item::Tuple, args)
    x = args[item[2]]
    first(item) === :arg && return Int(x)::Int
    first(item) === :length && return length(x)::Int
    first(item) === :dim && return size(x, item[3])::Int
    throw(ArgumentError("internal: unknown bound dimension $(repr(item))"))
end
_bound_shape_rows(shape::Tuple) = isempty(shape) ? 1 : first(shape)
_bound_shape_columns(shape::Tuple) = length(shape) <= 1 ? 1 : shape[2]
_bound_shape_dim(shape::Tuple, d::Int) =
    d == 0 ? prod(shape; init = 1) : (d <= length(shape) ? shape[d] : 1)

# The callee a source expression names, resolved in the source's module
# without evaluating anything, or `nothing`.
function _bound_shape_callee(x, mod::Module)
    x isa GlobalRef && return isdefined(x.mod, x.name) ? getglobal(x.mod, x.name) : nothing
    x isa Symbol && return isdefined(mod, x) ? getglobal(mod, x) : nothing
    if x isa Expr && x.head === :. && length(x.args) == 2 && x.args[2] isa QuoteNode
        parent = _bound_shape_callee(x.args[1], mod)
        name = x.args[2].value
        parent isa Module && name isa Symbol && isdefined(parent, name) &&
            return getglobal(parent, name)
    end
    nothing
end

# Recipe `r`'s source over its input values: the source expression and a map
# from each name it reads its inputs by to that input, or `nothing`.
function _bound_shape_source(r::Recipe)
    r.effectful && return nothing
    op = r.op
    op isa _KernelSourceOp || return nothing
    f = getfield(op, :f)
    f isa _KernelBranch && (f = getfield(f, :call))
    params = _lane_source_params(f, r)
    params === nothing && return nothing
    source = _bound_shape_unblock(_lane_strip_lines(r.source))
    source isa Union{Expr,Symbol} || return nothing
    names = Dict{Symbol,Value}()
    for (param, input) in zip(params, r.inputs)
        names[param] = input
        name = string(param)
        # A gensym parameter may be spelled with its `##` prefix.
        occursin('#', name) && !startswith(name, "##") &&
            (names[Symbol("##", name)] = input)
    end
    (; source, names, mod = _kernel_endpoint_source_module(op))
end

# The expression of a block that holds one expression.
function _bound_shape_unblock(x)
    while x isa Expr && x.head === :block && length(x.args) == 1
        x = only(x.args)
    end
    x
end

# A number written in the source: a literal, a named constant such as `Inf`,
# or its negation.
function _bound_shape_literal(x, names, mod)
    x isa Number && return true
    x isa Symbol && return !haskey(names, x) && _bound_shape_callee(x, mod) isa Number
    x isa Expr && x.head === :call && length(x.args) == 2 &&
        _bound_shape_callee(first(x.args), mod) in (-, +) &&
        return _bound_shape_literal(x.args[2], names, mod)
    false
end

# The operands of an `hcat(...)` call or an `[a b ...]` row, or `nothing`.
function _bound_shape_hcat_args(x, mod)
    x isa Expr || return nothing
    x.head === :hcat && return x.args
    x.head === :call || return nothing
    _bound_shape_callee(first(x.args), mod) === hcat || return nothing
    args = x.args[2:end]
    any(a -> a isa Expr && a.head in (:parameters, :kw, :...), args) && return nothing
    args
end

function _bound_shape_throws(x, mod)
    x = _bound_shape_unblock(x)
    x isa Expr && x.head === :block && !isempty(x.args) &&
        return _bound_shape_throws(last(x.args), mod)
    x isa Expr && x.head === :call &&
        _bound_shape_callee(first(x.args), mod) in (throw, error)
end

# The argument a check returns (`cond ? x : throw(...)`), or `nothing`.
function _bound_shape_checked(x, mod)
    x isa Expr && x.head === :if && length(x.args) == 3 || return nothing
    for (kept, other) in ((x.args[2], x.args[3]), (x.args[3], x.args[2]))
        _bound_shape_throws(other, mod) || continue
        return _bound_shape_unblock(kept)
    end
    nothing
end

# Known types of a plan's values, for telling a live number from a live array:
# `types` holds the types of its HAVE ports (an element type for a port read
# per lane) and is extended by inferring recipes on demand.
_bound_shape_context(p::Plan, types) =
    (; g = p.graph, types = Dict{ValueId,Any}(types),
       producers = Dict{ValueId,Recipe}(canon_id(p.graph, o.id) => r
                                        for r in p.recipes for o in r.outputs))

# The type of value `v`, as declared, recorded or inferred from its producer
# over its inputs' types (`Any` when unknown).
function _bound_shape_type(ctx, v::Value, depth = 0)
    T = valtype(v)
    T === Any || return T
    ctx === nothing && return Any
    cid = canon_id(ctx.g, v.id)
    haskey(ctx.types, cid) && return ctx.types[cid]
    depth > 16 && return Any
    r = get(ctx.producers, cid, nothing)
    (r === nothing || length(r.outputs) != 1) && return Any
    argtypes = Any[_bound_shape_type(ctx, w, depth + 1) for w in r.inputs]
    T = any(U -> U === Any, argtypes) ? Any :
        try
            _partial_prefix_type(r.op, argtypes)
        catch err
            err isa InterruptException && rethrow()
            Any
        end
    ctx.types[cid] = T
    T
end
_bound_shape_number(ctx, v::Value) = (T = _bound_shape_type(ctx, v); T isa Type && T <: Number)

# The shape tree (over values) of value `v`, or `nothing`. A bind-time value
# has its own shape; a live one the shape recorded for it.
function _bound_shape_tree(g::Graph, v::Value, shapes, dataonly)
    cid = canon_id(g, v.id)
    cid in dataonly && return (:size, v)
    get(shapes, cid, nothing)
end

# A dimension of a `fill` written in the source, over values, or `nothing`.
function _bound_shape_dimension(g::Graph, x, names, mod, dataonly)
    x isa Int && return x
    bound(a) = a isa Symbol && haskey(names, a) &&
        canon_id(g, names[a].id) in dataonly ? names[a] : nothing
    v = bound(x)
    v === nothing || return v
    x isa Expr && x.head === :call || return nothing
    f = _bound_shape_callee(first(x.args), mod)
    if f === length && length(x.args) == 2
        v = bound(x.args[2])
        return v === nothing ? nothing : (:length, v)
    elseif f === size && length(x.args) == 3 && x.args[3] isa Int && x.args[3] >= 1
        v = bound(x.args[2])
        return v === nothing ? nothing : (:dim, v, x.args[3])
    end
    nothing
end

# The shape tree of expression `x` of a recipe's source, or `nothing`.
function _bound_shape_expr(g::Graph, x, names, mod, shapes, dataonly, ctx, depth = 0)
    depth > 32 && return nothing
    x = _bound_shape_unblock(x)
    if x isa Symbol && haskey(names, x)
        v = names[x]
        tree = _bound_shape_tree(g, v, shapes, dataonly)
        tree === nothing && _bound_shape_number(ctx, v) && return (:scalar,)
        return tree
    end
    _bound_shape_literal(x, names, mod) && return (:scalar,)
    x isa Expr || return nothing
    checked = _bound_shape_checked(x, mod)
    checked === nothing ||
        return _bound_shape_expr(g, checked, names, mod, shapes, dataonly, ctx, depth + 1)
    parts = _bound_shape_hcat_args(x, mod)
    if parts !== nothing
        isempty(parts) && return nothing
        trees = Any[]
        for a in parts
            tree = _bound_shape_expr(g, a, names, mod, shapes, dataonly, ctx, depth + 1)
            tree === nothing && return nothing
            push!(trees, tree)
        end
        return (:hcat, Tuple(trees))
    end
    x.head === :call || return nothing
    f = _bound_shape_callee(first(x.args), mod)
    args = x.args[2:end]
    any(a -> a isa Expr && a.head in (:parameters, :kw, :...), args) && return nothing
    if f === identity && length(args) == 1
        return _bound_shape_expr(g, only(args), names, mod, shapes, dataonly, ctx, depth + 1)
    elseif f === fill && length(args) >= 2
        dims = Any[]
        for d in args[2:end]
            item = _bound_shape_dimension(g, d, names, mod, dataonly)
            item === nothing && return nothing
            push!(dims, item)
        end
        return (:dims, Tuple(dims))
    end
    nothing
end

# The shape tree of the one output of recipe `r` when it is computable from
# bind-time values, or `nothing`.
function _bound_shape_produced(g::Graph, r::Recipe, shapes, dataonly, ctx = nothing)
    length(r.outputs) == 1 || return nothing
    if r.op === identity && length(r.inputs) == 1
        return _bound_shape_tree(g, only(r.inputs), shapes, dataonly)
    end
    parsed = _bound_shape_source(r)
    parsed === nothing && return nothing
    (; source, names, mod) = parsed
    _bound_shape_expr(g, source, names, mod, shapes, dataonly, ctx)
end

# The array value and queried dimension (`0` for `length`) of a size query
# recipe over a live array with a known shape, or `nothing`.
function _bound_shape_consumer(g::Graph, r::Recipe, shapes, dataonly)
    length(r.outputs) == 1 || return nothing
    parsed = _bound_shape_source(r)
    parsed === nothing && return nothing
    (; source, names, mod) = parsed
    source isa Expr && source.head === :call || return nothing
    f = _bound_shape_callee(first(source.args), mod)
    args = source.args[2:end]
    (f === size && length(args) == 2 && args[2] isa Int && args[2] >= 1) ||
        (f === length && length(args) == 1) || return nothing
    v = args[1] isa Symbol ? get(names, args[1], nothing) : nothing
    v === nothing && return nothing
    cid = canon_id(g, v.id)
    cid in dataonly && return nothing
    tree = get(shapes, cid, nothing)
    tree === nothing && return nothing
    (; tree, dim = f === size ? args[2]::Int : 0)
end

# The tree with each value replaced by its position among `inputs`, which it
# extends.
function _bound_shape_positions(tree, inputs::Vector{Value}, g::Graph)
    function position(v::Value)
        cid = canon_id(g, v.id)
        at = findfirst(w -> canon_id(g, w.id) == cid, inputs)
        at === nothing && (push!(inputs, v); at = length(inputs))
        at
    end
    item(d::Int) = d
    item(d::Value) = (:arg, position(d))
    item(d::Tuple) = first(d) === :length ? (:length, position(d[2])) :
        (:dim, position(d[2]), d[3])
    kind = first(tree)
    kind === :size && return (:size, position(tree[2]))
    kind === :scalar && return tree
    kind === :dims && return (:dims, map(item, tree[2]))
    kind === :hcat && return (:hcat, map(p -> _bound_shape_positions(p, inputs, g), tree[2]))
    throw(ArgumentError("internal: unknown bound shape $(repr(kind))"))
end

# The shape trees of the live values of plan `p` computable from bind-time
# values (`dataonly`, which the walk extends by recipes reading only such
# values), keyed by value id; `types` as for `_bound_shape_context`.
function _bound_shapes(p::Plan, dataonly::Set{ValueId}, types = Dict{ValueId,Any}())
    g = p.graph
    ctx = _bound_shape_context(p, types)
    shapes = Dict{ValueId,Any}()
    for r in p.recipes
        if all(v -> canon_id(g, v.id) in dataonly, r.inputs)
            foreach(v -> push!(dataonly, canon_id(g, v.id)), r.outputs)
            continue
        end
        tree = _bound_shape_produced(g, r, shapes, dataonly, ctx)
        tree === nothing || (shapes[canon_id(g, only(r.outputs).id)] = tree)
    end
    shapes
end

# The known types of cell plan `inner`'s HAVE ports, given its plate recipe
# `r`'s operation `op` and the types of its bound ports (`bound_types`, keyed
# by the cell's value ids) and of the enclosing plan's values (`outer_types`):
# a live operand read per cell has its declared array's element type.
function _partial_shape_types(inner::Plan, r::Recipe, @nospecialize(op), bound_types,
                              outer_types = Dict{ValueId,Any}(), outer_graph = nothing)
    types = Dict{ValueId,Any}(bound_types)
    atomic = typeof(op).parameters[2]
    for (i, (input, h)) in enumerate(zip(r.inputs, inner.have))
        cid = canon_id(inner.graph, h.id)
        haskey(types, cid) && continue
        T = outer_graph === nothing ? valtype(input) :
            get(outer_types, canon_id(outer_graph, input.id), valtype(input))
        types[cid] = i in atomic ? T :
            T isa Type && T <: AbstractArray ? eltype(T) : Any
    end
    types
end

"""
    _partial_shape_recipes(inner::Plan, bound, excluded, readable, types) -> Plan

`inner` with each size query of a live array whose shape is computable from
bind-time values (`bound`, or values computed from them alone) replaced by a
`_BoundShapeQuery` of that shape, which reads bind-time values only. `types`
are the known types of `inner`'s HAVE ports, an element type for a port read
per cell. Readable recipes of the replacements are added to `readable`.
Returns `inner` itself when nothing is replaced.
"""
function _partial_shape_recipes(inner::Plan, bound::Set{ValueId}, excluded::Set{Int},
                                readable, types)
    g = inner.graph
    ctx = _bound_shape_context(inner, types)
    dataonly = copy(bound)
    shapes = Dict{ValueId,Any}()
    recipes = Recipe[]
    changed = false
    for r in inner.recipes
        if !(r.id in excluded) && all(v -> canon_id(g, v.id) in dataonly, r.inputs)
            foreach(v -> push!(dataonly, canon_id(g, v.id)), r.outputs)
            push!(recipes, r)
            continue
        end
        query = r.id in excluded ? nothing : _bound_shape_consumer(g, r, shapes, dataonly)
        if query === nothing
            tree = _bound_shape_produced(g, r, shapes, dataonly, ctx)
            tree === nothing || (shapes[canon_id(g, only(r.outputs).id)] = tree)
            push!(recipes, r)
            continue
        end
        inputs = Value[]
        positions = _bound_shape_positions(query.tree, inputs, g)
        op = _BoundShapeQuery{positions,query.dim}()
        replacement = Recipe(r.id, Tuple(inputs), r.outputs, op, r.cost, nothing,
                             false)
        push!(recipes, replacement)
        readable[r.id] = replacement
        foreach(v -> push!(dataonly, canon_id(g, v.id)), r.outputs)
        changed = true
    end
    changed || return inner
    _partial_subplan(inner, inner.have, inner.want, recipes)
end
