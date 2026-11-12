# Native composition of nonescaping array expressions across nested plate
# boundaries. The graph and operation table retain the authored values; only
# their native representation changes, after nested regions have been spliced.
abstract type _CellArray{T,N} <: AbstractArray{T,N} end

# Index types can remain unknown until specialization. Scalar readers below
# select their entries directly. A nonscalar reader (an integer vector, a
# range, a colon or a mask) selects its entries the same way, into a fresh
# array of Julia's result shape, so a group cell's `x[rows[g]]` computes those
# rows only instead of the whole recipe at every read. Like the scalar reads,
# the selection stays inlined, keeping the recipe visible to scalar
# replacement (below).
@inline function Base.getindex(a::_CellArray, I...)
    J = to_indices(a, I)
    @boundscheck checkbounds(a, J...)
    _cell_array_select(a, J)
end
Base.getindex(a::_CellArray) = getindex(_cell_array_materialize(a))
@inline _cell_array_select(a::_CellArray, J::Tuple{Integer,Vararg{Integer}}) =
    @inbounds a[Base._to_subscript_indices(a, map(Int, J)...)...]
@inline function _cell_array_select(a::_CellArray{T}, J::Tuple) where {T}
    selected = similar(Array{T}, Base.index_shape(J...))
    @inbounds for (k, index) in zip(eachindex(selected), Iterators.product(J...))
        selected[k] = a[index...]
    end
    selected
end
@inline Base.getindex(a::_CellArray, I::CartesianIndex) = getindex(a, Tuple(I)...)
@inline _cell_array_materialize(a) = a
# Base's `map` over a tuple of 32 or more operands goes through a heap vector
# and loses the tuple's concrete type; the structural map keeps it.
@inline _cell_array_materialize(bc::Base.Broadcast.Broadcasted) =
    Base.Broadcast.Broadcasted(bc.style, bc.f,
        _native_broadcast_map(_cell_array_materialize, bc.args), bc.axes)

# A nonescaping fill snapshots one primitive scalar, with the same runtime
# dimensions as its dense array. Scalar readers expose that shared value to
# native optimization instead of loading it from a freshly filled buffer.
struct _CellFill{T,N} <: _CellArray{T,N}
    value::T
    dimensions::NTuple{N,Int}
end
Base.size(a::_CellFill) = a.dimensions
@inline _cell_array_materialize(a::_CellFill) = fill(a.value, a.dimensions)
@inline function Base.getindex(a::_CellFill, i::Int, I::Int...)
    @boundscheck checkbounds(a, i, I...)
    a.value
end
@inline function Base.getindex(a::_CellFill)
    @boundscheck checkbounds(a)
    a.value
end
@inline _cell_array_fill(value, dims...) = fill(value, dims...)
@inline function _cell_array_fill(value::T, dims::Vararg{Int,N}) where {T<:Union{Float32,Float64},N}
    # Invalid or overflowing dimensions keep Base's allocation path and its
    # original exception. The conservative byte limit also leaves huge empty
    # layouts to Base, without depending on version-specific Array internals.
    limit = div(typemax(Int), sizeof(T))
    count = 1
    for dim in dims
        0 <= dim <= limit || return fill(value, dims...)
        if dim == 0
            count = 0
        else
            count <= div(limit, dim) || return fill(value, dims...)
            count *= dim
        end
    end
    _CellFill(value, dims)
end

struct _CellHcat{T,A} <: _CellArray{T,2}
    arguments::A
    rows::Int
    columns::Int
end
Base.size(a::_CellHcat) = (a.rows, a.columns)
@inline _cell_array_materialize(a::_CellHcat) =
    _native_hcat(map(_cell_array_materialize, a.arguments)...)
# Keep the selected operand's load inside its call. Otherwise LLVM can merge
# same-typed active and constant array pointers before the load, losing the
# separate operand activity that ordinary reverse differentiation needs.
@noinline _cell_array_entry(a, i, j) = a[i, j]
@noinline _cell_array_entry(a, I::CartesianIndex) = a[I]
# Recipes stay visible to scalar replacement. Passing an entire recipe to a
# non-inlined call would instead package active scalars with borrowed constant
# arrays. Only the dense operand load needs a call boundary.
@inline _cell_array_entry(a::_CellArray, i, j) = a[i, j]
@inline _cell_array_entry(a::_CellArray, I::CartesianIndex) = a[I]
@inline _cell_hcat_entry(args::Tuple, i, j) =
    j <= size(first(args), 2) ? _cell_array_entry(first(args), i, j) :
    _cell_hcat_entry(Base.tail(args), i, j - size(first(args), 2))
@inline _cell_hcat_entry(::Tuple{}, i, j) = throw(BoundsError())
@inline function Base.getindex(a::_CellHcat{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(a, i, j)
    convert(T, _cell_hcat_entry(a.arguments, i, j))
end
@inline function Base.getindex(a::_CellHcat, i::Int)
    @boundscheck checkbounds(a, i)
    a[mod1(i, a.rows), div(i - 1, a.rows) + 1]
end

struct _CellBroadcast{T,N,B} <: _CellArray{T,N}
    broadcast::B
end
Base.axes(a::_CellBroadcast) = axes(a.broadcast)
Base.size(a::_CellBroadcast) = map(length, axes(a))
@inline function _cell_array_materialize(a::_CellBroadcast{T}) where {T}
    bc = _cell_array_materialize(a.broadcast)
    _native_broadcast_wide(bc) || return _native_broadcast_materialize(bc)
    # `bc` carries the axes `_cell_array_broadcast` computed from its operands,
    # and the fresh output aliases none of them.
    _native_broadcast_wide_copyto!(similar(bc, T), bc)
end
@inline function Base.getindex(a::_CellBroadcast{T,N}, I::Vararg{Int,M}) where {T,N,M}
    @boundscheck checkbounds(a, I...)
    index = M == N ? CartesianIndex(I) : M > N ?
        CartesianIndex(ntuple(i -> I[i], Val(N))) :
        CartesianIndices(axes(a))[LinearIndices(a)[I...]]
    convert(T, _cell_broadcast_entry(a.broadcast, index))
end
# Base's entry read splats the operand tuple, which Julia stops specializing
# past 32 operands; a wide broadcast reads through the structural expansion.
# `index` is in bounds of the operands' combined axes (checked above, as
# Base's own read checks it), so the operand reads are too.
@inline _cell_broadcast_entry(bc, index) = _native_broadcast_wide(bc) ?
    (@inbounds _native_broadcast_getindex(bc, index)) : bc[index]

# Delaying a user function or custom indexing could suppress an exception or
# change observable effects at an unread element. Compose only total arithmetic
# over ordinary floating storage. Every other runtime type keeps materialization.
_cell_array_operand(x) = false
_cell_array_operand(x::Union{Float32,Float64,Array{Float32},Array{Float64},_CellArray}) = true
_cell_array_operand(x::SubArray{<:Union{Float32,Float64},<:Any,<:Array}) =
    all(i -> i isa Union{Int,Base.Slice{Base.OneTo{Int}},UnitRange{Int},Vector{Int}}, x.indices)
_cell_array_operand(x::Union{Int,Int32,Bool}) = true
_cell_array_operand(x::Base.Broadcast.Broadcasted) =
    _cell_total_function(x.f) && _cell_array_operands(x.args)
# `all` over a tuple splats it, which Julia stops specializing past 32 operands.
@inline @generated function _cell_array_operands(args::Tuple)
    foldr((i, rest) -> :(_cell_array_operand(getfield(args, $i)) && $rest),
          1:fieldcount(args); init=true)
end
_cell_total_function(f) = f === (+) || f === (-) || f === (*) || f === (/) ||
    f === exp || f === abs || f === abs2 || f === identity ||
    f === LogExpFunctions.logistic || f === LogExpFunctions.log1pexp

@inline function _cell_array_hcat(args...)
    if !isempty(args) && all(_cell_array_operand, args) &&
            all(a -> ndims(a) <= 2, args)
        rows = size(first(args), 1)
        all(a -> size(a, 1) == rows, args) ||
            throw(DimensionMismatch("hcat operands must have equal numbers of rows"))
        T = Base.promote_eltypeof(args...)
        return _CellHcat{T,typeof(args)}(args, rows, sum(a -> size(a, 2), args))
    end
    _native_hcat(map(_cell_array_materialize, args)...)
end
# Only one-dimensional broadcasts compose; a higher-rank result keeps its
# materialization. On Julia 1.10, ordinary Enzyme Reverse of composed reads
# from a matrix broadcast (`x .+ y` with a column and a row) fails with an
# internal "Illegal cached pointer" error in Enzyme 0.13.216
# (EnzymeAD/Enzyme.jl#3822). Julia 1.13, and Enzyme with that issue's repair,
# differentiate it.
@inline function _cell_array_broadcast(bc::Base.Broadcast.Broadcasted)
    if bc.style isa Base.Broadcast.DefaultArrayStyle &&
            _cell_array_operand(bc)
        ready = _cell_broadcast_instantiate(bc)
        T = _cell_broadcast_eltype(ready)
        if (T === Float32 || T === Float64) && length(axes(ready)) == 1
            return _CellBroadcast{T,1,typeof(ready)}(ready)
        end
    end
    _native_broadcast_materialize(_cell_array_materialize(bc))
end
@inline _cell_array_broadcast(x) = _native_broadcast_materialize(x)
# Base's `instantiate` and `combine_eltypes` splat the operand tuple, which
# Julia stops specializing past 32 operands (an n-ary `.+` of many terms). A
# wide broadcast computes the same axes, with the same pairwise checks, and
# the same element type through the structural expansions of
# `_native_broadcast_materialize`. Here `bc` is the lazy descriptor, which
# carries no axes yet.
@inline _cell_broadcast_instantiate(bc) = _native_broadcast_wide(bc) ?
    Base.Broadcast.Broadcasted(bc.style, bc.f, bc.args, _native_broadcast_axes(bc)) :
    Base.Broadcast.instantiate(bc)
@inline _cell_broadcast_eltype(bc) = _native_broadcast_wide(bc) ?
    _native_broadcast_eltype(bc) : Base.Broadcast.combine_eltypes(bc.f, bc.args)

# A gather whose readers only take its entries or its shape. A dense floating
# vector read at an integer vector is borrowed: `_readonly_gather` checks every
# index before returning, as `x[I]` does, and nothing writes either side while
# the gather is read (effectful regions keep their copies). Every other source
# or index is Julia's indexing, unchanged.
@inline _cell_array_gather(x, I) = x[I]
@inline _cell_array_gather(x::Union{Vector{Float32},Vector{Float64}}, I::Vector{Int}) =
    _readonly_gather(x, I)

# A declaration still converts values as authored. A same-element, same-rank
# dense Array declaration can keep the internal recipe representation; all
# other declarations materialize and convert normally.
@inline _cell_array_as(::Type{T}, x) where {T} = convert(T, x)
@inline _cell_array_as(::Type{Array{T,N}}, x::_CellArray{T,N}) where {T,N} = x

_cell_array_ref(name) = GlobalRef(@__MODULE__, name)
_cell_array_has(x, names) = x isa Symbol ? x in names :
    x isa Expr && !(x.head in (:quote, :inert)) && any(a -> _cell_array_has(a, names), x.args)
function _cell_array_function(x)
    x isa GlobalRef && isdefined(x.mod, x.name) && return getglobal(x.mod, x.name)
    nothing
end

# Read only source bodies whose positional parameters and captures agree with
# the executable closure. Split/ignore bodies differ from the authored source
# and deliberately keep their ordinary execution.
function _cell_array_source(r::Recipe, args)
    r.effectful && return nothing
    op = r.op
    op isa _KernelSourceOp || return nothing
    kernel_sourceop_form(op) === :fused && getfield(op, :ignored_throws) !== nothing ||
        return nothing
    f = getfield(op, :f)
    f isa _KernelBranch && (f = getfield(f, :call))
    length(args) == length(r.inputs) || return nothing
    params = _lane_source_params(f, r)
    params === nothing && return nothing
    mod = _kernel_endpoint_source_module(op)
    source = _lane_expand_dot(_lane_strip_lines(r.source), mod)
    source === nothing && return nothing
    native = _kernel_native_body(source, mod, Set{Symbol}(params))
    ctx = _LaneSourceContext(mod, Dict{Symbol,Any}(zip(params, args)))
    _cell_array_source_node(ctx, native)
end
function _cell_array_source_node(ctx, x)
    if x isa Expr && x.head in (:if, :&&, :||, :block)
        args = Any[_cell_array_source_node(ctx, a) for a in x.args if !(a isa LineNumberNode)]
        any(isnothing, args) && return nothing
        return Expr(x.head, args...)
    end
    _lane_source_node(ctx, x, nothing)
end

function _cell_array_expand_call(rhs, recipes)
    rhs isa Expr || return rhs
    # An operation call, or the recipe's source inlined in its place
    # (`inline_sources.jl`): both name the recipe whose source is read.
    if rhs.head === :call
        slot = _callable_slot(rhs.args[1])
        if slot === nothing
            # A call by name (`_const_call_ref`): an identity alias is its argument.
            rhs.args[1] == GlobalRef(Base, :identity) && length(rhs.args) == 2 &&
                return rhs.args[2]
            return rhs
        end
        args = rhs.args[2:end]
    else
        inlined = _inline_statement_call(rhs)
        inlined === nothing && return rhs
        slot, args = inlined
    end
    recipe = recipes[slot]
    recipe.op === identity && length(args) == 1 && return only(args)
    source = _cell_array_source(recipe, args)
    source === nothing ? rhs : source
end

# Array-valued leaves of a conditional/copy. Shape-checking aliases retain
# their condition and exception arm at the original program point.
function _cell_array_aliases(x, names)
    x isa Symbol && return x in names
    x isa Expr || return false
    x.head === :call && length(x.args) == 2 &&
        _cell_array_function(first(x.args)) === identity &&
        return _cell_array_aliases(x.args[2], names)
    _cell_array_lane(x, names) && return true
    if x.head === :if
        arms = x.args[2:end]
        return any(a -> _cell_array_aliases(a, names), arms) &&
            all(a -> _cell_array_aliases(a, names) || _cell_array_throws(a), arms)
    end
    x.head === :block && return !isempty(x.args) && _cell_array_aliases(last(x.args), names)
    false
end
# A native plate reads each lane argument through broadcast preparation: the
# argument itself (`Base.broadcastable` of an array is the array), its
# `Extruded` wrapper (`Broadcast.preprocess(nothing, ·)`), and the plate's
# axis marker, which is the first lane that is an axis. Each of these results
# holds the array whole, so it is an alias; its readers are then checked like
# any other. The plate itself reads it only through shape queries and one
# `_broadcast_getindex` per cell coordinate (`_cell_array_lane_reads`).
function _cell_array_lane(x, names)
    x isa Expr && x.head === :call || return false
    fn = _cell_array_function(first(x.args))
    named(a) = a isa Symbol && a in names
    fn === Base.broadcastable && length(x.args) == 2 && return named(x.args[2])
    fn === Base.Broadcast.preprocess && length(x.args) == 3 &&
        x.args[2] === nothing && return named(x.args[3])
    if fn === _authored_plate_marker
        atomic = _cell_array_marker_atomic(x)
        atomic === nothing && return false
        return any(((i, a),) -> !(i in atomic) && named(a), enumerate(x.args[3:end]))
    end
    false
end
function _cell_array_marker_atomic(x)
    length(x.args) >= 2 || return nothing
    marker = x.args[2]
    marker isa Expr && marker.head === :call &&
        _cell_array_function(marker.args[1]) === Val &&
        length(marker.args) == 2 && marker.args[2] isa QuoteNode || return nothing
    atomic = marker.args[2].value
    atomic isa Tuple ? atomic : nothing
end
# The plate's own reads of a lane: shape and type queries over the whole
# value, and the entry at a cell coordinate that names no candidate.
function _cell_array_lane_reads(fn, x, names)
    named(a) = a isa Symbol && a in names
    free(args) = all(a -> !_cell_array_has(a, names), args)
    if fn === _authored_plate_is_axis || fn === _plate_unconditional_root
        return length(x.args) == 2 && named(x.args[2])
    elseif fn === _plate_dependency_changed || fn === _plate_total_seed ||
            fn === _plate_spans_axis
        return length(x.args) >= 3 && named(last(x.args)) && free(x.args[2:end-1])
    elseif fn === Base.Broadcast._broadcast_getindex || fn === _plate_read ||
            fn === _cell_array_gather
        # A gather of a composed value is its indexing (`_cell_array_select`).
        return length(x.args) >= 3 && named(x.args[2]) && free(x.args[3:end])
    end
    nothing
end
function _cell_array_throws(x)
    x isa Expr || return false
    x.head === :block && return !isempty(x.args) && _cell_array_throws(last(x.args))
    x.head === :call && _cell_array_function(first(x.args)) in (throw, error)
end

function _cell_array_reads(x, names; alias=false, internal=false)
    _cell_array_has(x, names) || return true
    x isa Symbol && return alias
    x isa Expr || return true
    if x.head === :ref && first(x.args) isa Symbol && first(x.args) in names
        return all(a -> !_cell_array_has(a, names), x.args[2:end])
    elseif x.head === :call
        fn = _cell_array_function(first(x.args))
        if fn in (size, axes, length, ndims, eltype, firstindex, lastindex) ||
                (internal && fn === typeof)
            return length(x.args) >= 2 && x.args[2] isa Symbol &&
                x.args[2] in names && all(a -> !_cell_array_has(a, names), x.args[3:end])
        elseif fn === _cell_array_hcat
            return all(a -> a isa Symbol && a in names ||
                _cell_array_reads(a, names; internal), x.args[2:end])
        elseif fn === _cell_array_as
            return _cell_array_reads(x.args[3], names; alias, internal)
        elseif fn === identity && length(x.args) == 2
            return _cell_array_reads(x.args[2], names; alias, internal)
        elseif fn === _cell_array_broadcast
            return _cell_array_broadcast_reads(x.args[2], names)
        elseif internal && fn === _authored_plate_marker
            # An atomic argument is read inside the cell, where those reads
            # are checked. A lane that is a candidate is a bare name: the
            # marker's result may be that array, so it is an alias.
            atomic = _cell_array_marker_atomic(x)
            atomic === nothing || return all(enumerate(x.args[3:end])) do (i, a)
                i in atomic || !_cell_array_has(a, names) || a isa Symbol
            end
        elseif internal && (fn === Base.broadcastable || fn === Base.Broadcast.preprocess)
            return alias && _cell_array_lane(x, names)
        else
            # An entry or a shape query cannot observe the representation, so
            # it is a reader also inside an inlined recipe's source, whose
            # arguments carry the plate's lane reads.
            lane = _cell_array_lane_reads(fn, x, names)
            lane === nothing || return lane
        end
        return all(a -> _cell_array_reads(a, names; internal), x.args)
    elseif x.head === :if
        return _cell_array_reads(first(x.args), names; internal) &&
            all(a -> _cell_array_reads(a, names; alias, internal), x.args[2:end])
    elseif x.head === :block
        return all(enumerate(x.args)) do (i, a)
            _cell_array_reads(a, names; alias=alias && i == length(x.args), internal)
        end
    end
    all(a -> _cell_array_reads(a, names; internal), x.args)
end

function _cell_array_broadcast_reads(x, names)
    !_cell_array_has(x, names) && return true
    x isa Symbol && return x in names
    x isa Expr && x.head === :call &&
        _cell_array_function(x.args[1]) in (Base.Broadcast.broadcasted,
                                          _native_broadcasted) || return false
    !_cell_array_has(x.args[2], names) &&
        all(a -> _cell_array_broadcast_reads(a, names), x.args[3:end])
end

# A statement whose value is one gather `x[I]`, read only entry by entry,
# borrows `x` (`_cell_array_gather`). Only a whole statement: a gather inside
# another expression keeps Julia's indexing. An index naming `end` or `begin`
# keeps it too, since that syntax means `x`'s own extent.
function _cell_array_statement_producer(x)
    producer = _cell_array_producer(x)
    producer === nothing || return producer
    x isa Expr && x.head === :ref && length(x.args) == 2 &&
        !Meta.isexpr(x.args[2], :...) &&
        !_cell_array_has(x.args[2], Set((:end, :begin))) || return nothing
    # A literal position is one entry, and the program's tables are read whole.
    x.args[2] isa Integer && return nothing
    x.args[1] in (:__constants__, :__callables__) && return nothing
    Expr(:call, _cell_array_ref(:_cell_array_gather), x.args[1], x.args[2])
end
function _cell_array_producer(x)
    x isa Expr && x.head === :call || return nothing
    fn = _cell_array_function(first(x.args))
    fn === fill && return Expr(:call, _cell_array_ref(:_cell_array_fill), x.args[2:end]...)
    fn === _native_hcat && return Expr(:call, _cell_array_ref(:_cell_array_hcat),
        map(_cell_array_child, x.args[2:end])...)
    fn === _native_broadcast_materialize && return Expr(:call,
        _cell_array_ref(:_cell_array_broadcast), map(_cell_array_child, x.args[2:end])...)
    nothing
end
function _cell_array_child(x)
    producer = _cell_array_producer(x)
    producer === nothing || return producer
    if x isa Expr && x.head === :call &&
            _cell_array_function(first(x.args)) in (Base.Broadcast.broadcasted,
                                                  _native_broadcasted)
        return Expr(:call, x.args[1], x.args[2], map(_cell_array_child, x.args[3:end])...)
    end
    x
end

function _cell_array_elementwise_operands!(out, ex::Expr)
    fn = _cell_array_function(first(ex.args))
    if fn === _native_broadcasted || fn === Base.Broadcast.broadcasted
        for a in ex.args[3:end]
            a isa Symbol && push!(out, a)
        end
    elseif fn === _authored_plate_marker
        for a in ex.args[3:end]
            a isa Symbol && push!(out, a)
        end
    end
    out
end

# Analyze flattened native def/use, including calls whose source belongs to a
# different embedded kernel. Keep names and executable tables unchanged. A
# returned value or unknown reader makes the candidate ineligible.
function _lower_cell_arrays_native(body::Expr, recipes)
    # Effectful work can mutate an operand after its authored snapshot. Keep
    # storage in such regions instead of borrowing the operand through it.
    any(r -> r.effectful, recipes) && return body
    statements = Dict{Symbol,Expr}()
    sources = Dict{Symbol,Any}()
    declarations = Dict{Symbol,Any}()
    repeated = Set{Symbol}()
    ordered = Symbol[]
    # The names some dotted call or plate reads entry by entry: only those
    # gathers are candidates, so the per-candidate check stays off the many
    # scalar reads `x[i]` of a cell body.
    elementwise = Set{Symbol}()
    _index_run_visit(body) do ex
        if ex.head === :(=) && length(ex.args) == 2 && ex.args[1] isa Symbol
            name = ex.args[1]
            push!(ordered, name)
            haskey(statements, name) && push!(repeated, name)
            statements[name] = ex
            sources[name] = _cell_array_expand_call(ex.args[2], recipes)
        elseif ex.head === :call
            _cell_array_elementwise_operands!(elementwise, ex)
        elseif ex.head === :local && length(ex.args) == 1
            typed = only(ex.args)
            typed isa Expr && typed.head === :(::) && typed.args[1] isa Symbol &&
                (declarations[typed.args[1]] = typed.args[2])
        end
    end
    # An inlined recipe's raw statement names its `let` locals; its expanded
    # source names the values it reads.
    for source in values(sources)
        source isa Expr || continue
        _index_run_visit(ex -> ex.head === :call &&
            _cell_array_elementwise_operands!(elementwise, ex), source)
    end
    replacements = Dict{Symbol,Any}()
    changed = Set{Symbol}()
    # Consumers first: composing a concat makes its broadcast operands eligible.
    candidates = reverse(unique(ordered))
    progress = true
    while progress
        progress = false
        for name in candidates
            name in changed && continue
            producer = _cell_array_statement_producer(sources[name])
            producer === nothing && continue
            _cell_array_function(first(producer.args)) === _cell_array_gather &&
                !(name in elementwise) && continue
            aliases = Set([name])
            grew = true
            while grew
                grew = false
                for (lhs, rhs) in sources
                    lhs in aliases && continue
                    if _cell_array_aliases(rhs, aliases)
                        push!(aliases, lhs); grew = true
                    end
                end
            end
            isempty(intersect(aliases, repeated)) || continue
            valid = Ref(true)
            function check(ex)
                valid[] || return
                ex isa Expr || return
                ex.head in (:quote, :inert) && return
                if ex.head === :(=) && length(ex.args) == 2 && ex.args[1] isa Symbol
                    lhs = ex.args[1]
                    # `sources` holds a name's last assignment; a name assigned
                    # more than once (a plate's running total) is read here.
                    rhs = lhs in repeated ? ex.args[2] : get(sources, lhs, ex.args[2])
                    valid[] = _cell_array_reads(rhs, aliases; alias=lhs in aliases,
                        internal=rhs === ex.args[2])
                elseif ex.head === :local && length(ex.args) == 1 &&
                        ex.args[1] isa Expr && ex.args[1].head === :(::)
                    return
                elseif ex.head in (:block, :for, :while, :let, :if)
                    foreach(check, ex.args)
                else
                    valid[] = _cell_array_reads(ex, aliases; internal=true)
                end
            end
            check(body)
            valid[] || continue
            replacements[name] = producer
            sources[name] = producer
            union!(changed, aliases)
            progress = true
        end
    end
    isempty(replacements) && return body
    _index_run_replace(body) do ex
        if ex.head === :local && length(ex.args) == 1
            typed = only(ex.args)
            typed isa Expr && typed.head === :(::) && typed.args[1] in changed &&
                return Expr(:local, typed.args[1])
        elseif ex.head === :(=) && length(ex.args) == 2 && ex.args[1] isa Symbol
            lhs = ex.args[1]
            rhs = get(replacements, lhs, ex.args[2])
            # Inline the verified source of descriptor consumers. Their original
            # source closures can carry concrete array argument annotations.
            # (Only a name assigned once: `sources` holds its statement.)
            if !(lhs in repeated) && _cell_array_has(get(sources, lhs, rhs), changed)
                rhs = sources[lhs]
            end
            if lhs in changed && haskey(declarations, lhs)
                rhs = Expr(:call, _cell_array_ref(:_cell_array_as), declarations[lhs], rhs)
            end
            rhs === ex.args[2] || return Expr(:(=), lhs, rhs)
        end
        ex
    end
end
