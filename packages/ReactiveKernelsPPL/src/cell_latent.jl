# A dotted cell latent, `v[i] .~ D.(args…)` in a `@plate` cell with `v` not
# data, states one iteration of its Julia loop: `v[i]` holds the broadcast of
# `D` over the arguments' values at index `i`. That is a number when they are
# all numbers, and otherwise an array of their broadcast shape, whose size
# may vary by index. The coordinates of every index are one ordinary array
# parameter, in plate order, under the declaration's prior with each
# argument's entries flattened in that order. The prior, transform and
# Jacobian are therefore the array prior's, and each cell reads its own block.
# The shapes must follow from data at binding. They are read from data per
# index, from scalar parameters and from the dotted arithmetic between them.

struct _CellLatentScope
    data::Set{Symbol}               # data and data-only definitions
    definitions::Dict{Symbol,Any}   # model-level definitions
    scalars::Set{Symbol}            # scalar parameters declared at model level
end

# Model-level `x ~ D(…)` with a scalar prior: a number wherever it is read.
function _scalar_parameter_names(args)
    out = Set{Symbol}()
    for st in args
        Meta.isexpr(st, :call, 3) && st.args[1] === :~ && st.args[2] isa Symbol ||
            continue
        _scalar_prior_object(st.args[3]) && push!(out, st.args[2])
    end
    return out
end
_scalar_prior_object(ex) = Meta.isexpr(ex, :call) && !isempty(ex.args) &&
    (ex.args[1] in _SCALAR_PRIOR_HEADS || ex.args[1] in (:truncated, :restricted) &&
        length(ex.args) >= 2 && _scalar_prior_object(ex.args[2]))

_is_cell_array_latent(c, ivar::Symbol, data::Set{Symbol}) =
    Meta.isexpr(c, :call, 3) && c.args[1] === :.~ &&
    Meta.isexpr(c.args[2], :ref, 2) && c.args[2].args[1] isa Symbol &&
    c.args[2].args[2] === ivar && !(c.args[2].args[1] in data) &&
    _is_cell_latent_object(c.args[3])

_is_cell_latent_object(ex) = _is_dotted_call(ex) && ex.args[1] isa Symbol &&
    _cell_object_call(ex.args[1])

# Rebuild a dotted constructor tree (`truncated.(Normal.(m, s), lo, hi)`),
# replacing each argument that is not itself a constructor by `f(argument)`.
_cell_latent_object(f, ex) = _is_cell_latent_object(ex) ?
    Expr(:., ex.args[1], Expr(:tuple, (_cell_latent_object(f, a) for a in ex.args[2].args)...)) :
    f(ex)

function _cell_latent_leaves(ex)
    out = Any[]
    _cell_latent_object(ex) do leaf
        push!(out, leaf)
        leaf
    end
    return out
end

# What one plate's cells say about the shapes of the values they read.
struct _CellShapes
    ivar::Symbol
    scope::_CellLatentScope
    locals::Vector{Pair{Symbol,Any}}
    dataonly::Set{Symbol}                       # cell locals reading only data
    scalars::Set{Symbol}                        # scalar latents of the cell
    arrays::Dict{Symbol,Tuple{Symbol,Symbol}}   # array latent => (dims, position)
end

_cell_reads_only_data(ex, cx::_CellShapes) = all(s -> s in cx.scope.data ||
    s === cx.ivar || s in cx.dataonly, _value_symbols(ex))

function _push_cell_local!(cx::_CellShapes, name::Symbol, rhs)
    push!(cx.locals, name => rhs)
    _cell_reads_only_data(rhs, cx) && push!(cx.dataonly, name)
    return cx
end

# The shape of a cell expression: `:scalar` when it is a number before any
# data is bound, an expression computing its axes from data at each index,
# or `(nothing, why)` when the shape is not known before sampling.
const _CELL_SCALAR_CALLS = (:+, :-, :*, :/, :^, :exp, :log, :log1p, :expm1,
    :sqrt, :abs, :abs2, :tanh, :logistic, :logit, :log10, :sin, :cos,
    :min, :max, :inv)

function _cell_latent_shape(ex, cx::_CellShapes, active::Set{Symbol} = Set{Symbol}())
    ex isa Number && return :scalar
    ex isa Union{Symbol,Expr} || return (nothing, "it is not a value")
    _cell_reads_only_data(ex, cx) &&
        return Expr(:call, GlobalRef(Base, :axes), ex)
    if ex isa Symbol
        k = findlast(l -> first(l) === ex, cx.locals)
        k === nothing || return _cell_latent_shape(last(cx.locals[k]), cx, active)
        ex in cx.scope.scalars && return :scalar
        # A free name reads its module constant, as `Inf` does.
        !haskey(cx.scope.definitions, ex) && isdefined(Base, ex) &&
            getfield(Base, ex) isa Number && return :scalar
        if haskey(cx.scope.definitions, ex) && !(ex in active)
            push!(active, ex)
            shape = _cell_latent_shape(cx.scope.definitions[ex], cx, active)
            delete!(active, ex)
            return shape
        end
        return (nothing, "`$ex` is not data, a scalar parameter or a cell value")
    end
    if Meta.isexpr(ex, :ref, 2) && ex.args[1] isa Symbol && ex.args[2] === cx.ivar
        ex.args[1] in cx.scalars && return :scalar
        if haskey(cx.arrays, ex.args[1])
            dims, pos = cx.arrays[ex.args[1]]
            return Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_latent_axes), dims, pos)
        end
        return (nothing, "`$(ex.args[1])` is not data or a latent of this cell")
    end
    if ex.head === :ref && all(i -> i isa Integer, ex.args[2:end])
        return :scalar
    end
    dotted = _is_dotted_call(ex) ? ex.args[2].args :
        Meta.isexpr(ex, :call) && ex.args[1] isa Symbol &&
            startswith(string(ex.args[1]), ".") ? ex.args[2:end] : nothing
    if dotted !== nothing
        return _cell_broadcast_shape(Any[Meta.isexpr(a, :call, 2) && a.args[1] === :Ref ?
            :scalar : _cell_latent_shape(a, cx, active) for a in dotted])
    end
    if Meta.isexpr(ex, :call) && ex.args[1] isa Symbol
        f = ex.args[1]
        f in REDUCTION_FNS && length(ex.args) == 2 && return :scalar
        if f in _CELL_SCALAR_CALLS
            shapes = Any[_cell_latent_shape(a, cx, active) for a in ex.args[2:end]]
            for s in shapes
                s isa Tuple && return s
            end
            all(s -> s === :scalar, shapes) && return :scalar
            f in (:+, :-, :*, :/) || return (nothing, "`$f` of an array is not elementwise")
            return Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_latent_call_axes),
                QuoteNode(f), (s === :scalar ? :(()) : s for s in shapes)...)
        end
    end
    return (nothing, "`$ex` computes from sampled values, and its shape is " *
        "not known before sampling")
end

function _cell_broadcast_shape(shapes)
    for s in shapes
        s isa Tuple && return s
    end
    all(s -> s === :scalar, shapes) && return :scalar
    return Expr(:call, GlobalRef(Base.Broadcast, :broadcast_shape),
        (s === :scalar ? :(()) : s for s in shapes)...)
end

# Rewrite one plate's dotted array latents. Each becomes a model-level packed
# array with an ordinary array prior, read per cell as its block. Returns
# the model-level statements, the plate, and each packed array's authored
# name and dimensions.
function _expand_cell_array_latents(st, taken, scope::_CellLatentScope)
    sites = Dict{Symbol,Tuple{Symbol,Symbol}}()
    loop = st.args[3]
    (Meta.isexpr(loop, :for) && Meta.isexpr(loop.args[1], :(=), 2) &&
        Meta.isexpr(loop.args[2], :block)) || return Any[], st, sites
    ivar, domain = loop.args[1].args
    ivar isa Symbol || return Any[], st, sites
    any(c -> _is_cell_array_latent(c, ivar, scope.data), loop.args[2].args) ||
        return Any[], st, sites
    Meta.isexpr(domain, :call) && domain.args[1] === :levels && _sfail("cell " *
        "latents `v[k] .~ D.(…)` holding arrays over `levels(…)` are not built " *
        "yet; iterate `eachindex`, `axes` or a range")
    fresh(base) = _cell_scan_fresh!(taken, base)
    iterator = deepcopy(domain)
    if Meta.isexpr(iterator, :call) && iterator.args[1] in (:eachindex, :axes)
        iterator.args[1] = GlobalRef(Base, iterator.args[1])
    end
    indices = Expr(:call, GlobalRef(Base, :collect), iterator)
    position = Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_scan_position), domain, ivar)
    cx = _CellShapes(ivar, scope, Pair{Symbol,Any}[], Set{Symbol}(), Set{Symbol}(),
        Dict{Symbol,Tuple{Symbol,Symbol}}())
    prelude, cells = Any[], Any[]
    column(rhs) = _plate_column_call(ivar, _needed_cell_locals(cx.locals, rhs), rhs,
        nothing; indices)
    for c in loop.args[2].args
        if Meta.isexpr(c, :(=), 2) && c.args[1] isa Symbol
            _push_cell_local!(cx, c.args[1], c.args[2])
        elseif Meta.isexpr(c, :call, 3) && c.args[1] === :~ &&
                Meta.isexpr(c.args[2], :ref, 2) && c.args[2].args[2] === ivar &&
                c.args[2].args[1] isa Symbol && !(c.args[2].args[1] in scope.data)
            push!(cx.scalars, c.args[2].args[1])
        end
        _is_cell_array_latent(c, ivar, scope.data) || (push!(cells, c); continue)
        name = c.args[2].args[1]
        leaves = _cell_latent_leaves(c.args[3])
        shapes = Any[]
        for leaf in leaves
            shape = _cell_latent_shape(leaf, cx)
            shape isa Tuple && _sfail("cell latent `$name[$ivar] .~ …`: at each " *
                "index `$name[$ivar]` holds the broadcast of its arguments, so " *
                "their shapes must follow from data; the shape of " *
                "`$(repr(leaf))` does not: $(last(shape))")
            push!(shapes, shape)
        end
        if all(s -> s === :scalar, shapes)
            # Numbers at every index: one scalar draw per index.
            push!(cx.scalars, name)
            push!(cells, Expr(:call, :~, c.args[2], c.args[3]))
            continue
        end
        prefix = Symbol(:_rkppl_latent_, name)
        site, dims = fresh(prefix), fresh(Symbol(prefix, :_dims))
        offsets, count = fresh(Symbol(prefix, :_offsets)), fresh(Symbol(prefix, :_size))
        shape = _cell_broadcast_shape(shapes)
        dimscells = fresh(Symbol(prefix, :_shapes))
        push!(prelude, Expr(:(=), dimscells, column(Expr(:call,
            GlobalRef(@__MODULE__, :_ppl_cell_latent_dims), shape))))
        push!(prelude, Expr(:(=), dims, Expr(:call,
            GlobalRef(@__MODULE__, :_ppl_cell_latent_shapes), QuoteNode(name), dimscells)))
        push!(prelude, Expr(:(=), offsets,
            Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_latent_offsets), dims)))
        push!(prelude, Expr(:(=), count,
            Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_latent_size), offsets)))
        # A shared number broadcasts against the packed array as written;
        # every other argument contributes its entries at each index.
        k = 0
        object = _cell_latent_object(c.args[3]) do leaf
            k += 1
            shapes[k] === :scalar && !_expr_has_sym(leaf, ivar) &&
                !any(l -> _expr_has_sym(leaf, first(l)), cx.locals) && return leaf
            entries = fresh(Symbol(prefix, :_arg, k, :_cells))
            flat = fresh(Symbol(prefix, :_arg, k))
            push!(prelude, Expr(:(=), entries, column(Expr(:call,
                GlobalRef(@__MODULE__, :_ppl_cell_latent_entries), leaf,
                Expr(:ref, dims, position)))))
            push!(prelude, Expr(:(=), flat, Expr(:call,
                GlobalRef(@__MODULE__, :_ppl_cell_latent_flatten), entries)))
            return flat
        end
        push!(prelude, Expr(:call, :.~, Expr(:ref, site, Expr(:call, :(:), 1, count)), object))
        pos = fresh(Symbol(prefix, :_cell))
        push!(cells, Expr(:(=), pos, position))
        push!(cells, Expr(:(=), c.args[2], Expr(:call,
            GlobalRef(@__MODULE__, :_ppl_cell_latent_block), site, offsets, dims, pos)))
        _push_cell_local!(cx, pos, position)
        cx.arrays[name] = (dims, pos)
        sites[site] = (name, dims)
    end
    return prelude, Expr(st.head, st.args[1:2]...,
        Expr(:for, loop.args[1], Expr(:block, cells...))), sites
end

# A cell reading its array latent's block. Such a plate is retained, one
# iteration per index, as a cell `@scan` is.
_is_cell_latent_block(c) = Meta.isexpr(c, :(=), 2) && Meta.isexpr(c.args[2], :call) &&
    c.args[2].args[1] == GlobalRef(@__MODULE__, :_ppl_cell_latent_block)

# --- Values the rewritten program computes -------------------------------

# One index's dimensions: the broadcast axes of the arguments there.
_ppl_cell_latent_dims(axes::Tuple) = map(length, axes)

# Every index's dimensions. With one rank at every index each cell's block
# has one concrete type. Different ranks are valid Julia but make the cells
# type unstable, so they are warned about rather than refused.
function _ppl_cell_latent_shapes(name::Symbol, cells::AbstractVector)
    isempty(cells) && return Tuple{}[]
    N = length(first(cells))
    all(d -> d isa NTuple{N,Int}, cells) && return NTuple{N,Int}[d for d in cells]
    @warn "cell latent `$name`: its arguments broadcast to values of " *
        "different dimensions at different indices " *
        "($(join(sort!(unique(length.(cells))), ", ")) dimensions), so its cells " *
        "read values of different types and the prepared program is not type " *
        "stable; give every index's `$name[i]` the same number of dimensions"
    return Tuple{Vararg{Int}}[d for d in cells]
end

function _ppl_cell_latent_offsets(dims::AbstractVector{<:Tuple})
    offsets = Vector{Int}(undef, length(dims) + 1)
    offsets[1] = 0
    for (k, d) in enumerate(dims)
        offsets[k + 1] = offsets[k] + prod(d)
    end
    return offsets
end
_ppl_cell_latent_size(offsets::AbstractVector{Int}) = last(offsets)

# The axes of an earlier array latent of the cell at its position.
_ppl_cell_latent_axes(dims, pos) = map(Base.OneTo, dims[pos])

# Julia's result shape of an undotted `+`, `-`, `*` or `/` with an array
# operand: equal shapes for a sum, one array beside numbers for a product.
function _ppl_cell_latent_call_axes(f::Symbol, axes::Tuple...)
    arrays = filter(!isempty, collect(axes))
    isempty(arrays) && return ()
    f in (:+, :-) && return Base.Broadcast.broadcast_shape(axes...)
    length(arrays) == 1 && return only(arrays)
    throw(ArgumentError("a cell latent argument applies `$f` to several arrays; " *
        "write its elementwise form (`.$f`)"))
end

# One index's entries of an argument, broadcast to that index's dimensions
# (column-major), in one fresh vector.
_ppl_cell_latent_entries(x::Number, dims::Tuple) = fill(x, prod(dims))
function _ppl_cell_latent_entries(x::AbstractArray, dims::Tuple)
    out = Vector{eltype(x)}(undef, prod(dims))
    for (k, I) in enumerate(CartesianIndices(dims))
        # A singleton dimension of `x` stretches, as broadcasting does.
        out[k] = x[ntuple(d -> size(x, d) == 1 ? 1 : I[d], Val(ndims(x)))...]
    end
    return out
end

# Every index's entries, in plate order, in one fresh vector.
function _ppl_cell_latent_flatten(cells::AbstractVector)
    n = 0
    for c in cells
        n += length(c)
    end
    T = isempty(cells) ? Float64 : mapreduce(eltype, promote_type, cells)
    out = Vector{T}(undef, n)
    k = 0
    for c in cells, x in c
        out[k += 1] = x
    end
    return out
end

# A cell's block of the packed array: a number, a vector, or an array of
# its dimensions. The packed values are read in place.
_ppl_cell_latent_block(site, offsets, dims::AbstractVector{<:Tuple}, pos::Int) =
    _ppl_cell_latent_value(site, offsets, dims[pos], pos)
_ppl_cell_latent_value(site, offsets, ::Tuple{}, pos::Int) = site[offsets[pos] + 1]
_ppl_cell_latent_value(site, offsets, ::Tuple{Int}, pos::Int) =
    view(site, (offsets[pos] + 1):offsets[pos + 1])
_ppl_cell_latent_value(site, offsets, dims::Tuple, pos::Int) =
    reshape(view(site, (offsets[pos] + 1):offsets[pos + 1]), dims)

# --- Draws: one array per index under the authored name ------------------

function _cell_latent_dims(layout, site::Symbol)
    _, dims = layout.cell_latents[site]
    haskey(layout.bound_values, dims) || throw(ContractValidationError(
        "[layout] cell latent $(first(layout.cell_latents[site])) needs bound data"))
    return layout.bound_values[dims]
end

function _cell_latent_nested(layout, site::Symbol, v::AbstractVector)
    dims = _cell_latent_dims(layout, site)
    offsets = _ppl_cell_latent_offsets(dims)
    return [isempty(d) ? v[offsets[k] + 1] :
        reshape(v[(offsets[k] + 1):offsets[k + 1]], d) for (k, d) in enumerate(dims)]
end

function _cell_latent_flat(layout, site::Symbol, value)
    name = first(layout.cell_latents[site])
    dims = _cell_latent_dims(layout, site)
    value isa AbstractVector && length(value) == length(dims) || throw(
        ContractValidationError("[layout] cell latent $name needs one value per " *
            "index ($(length(dims)))"))
    for (k, (x, d)) in enumerate(zip(value, dims))
        size(x) == d || throw(ContractValidationError("[layout] cell latent " *
            "$name[$k] has size $(size(x)), want $d"))
    end
    return reduce(vcat, (vec(collect(x)) for x in value); init = Float64[])
end

function _cell_latent_coordinates(layout, site::Symbol)
    shown = _draw_name(layout, site)
    names = Symbol[]
    for (k, d) in enumerate(_cell_latent_dims(layout, site))
        if isempty(d)
            push!(names, Symbol(shown, ".", k))
        else
            for j in CartesianIndices(d)
                push!(names, Symbol(shown, ".", k, ".", join(Tuple(j), ".")))
            end
        end
    end
    return names
end
