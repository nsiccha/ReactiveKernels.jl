# Native composition of nonescaping array expressions across nested plate
# boundaries. The graph and operation table retain the authored values; only
# their native representation changes, after nested regions have been spliced.
abstract type _CellArray{T,N} <: AbstractArray{T,N} end

# Index types can remain unknown until specialization. Slices and other
# nonscalar readers keep the authored materialization; scalar readers below
# select their entries directly.
Base.getindex(a::_CellArray, I...) = getindex(_cell_array_materialize(a), I...)
@inline Base.getindex(a::_CellArray, I::CartesianIndex) = getindex(a, Tuple(I)...)
@inline _cell_array_materialize(a) = a
@inline _cell_array_materialize(bc::Base.Broadcast.Broadcasted) =
    Base.Broadcast.Broadcasted(bc.style, bc.f,
        map(_cell_array_materialize, bc.args), bc.axes)

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
@inline _cell_array_materialize(a::_CellBroadcast) =
    _native_broadcast_materialize(_cell_array_materialize(a.broadcast))
@inline function Base.getindex(a::_CellBroadcast{T,N}, I::Vararg{Int,M}) where {T,N,M}
    @boundscheck checkbounds(a, I...)
    index = M == N ? CartesianIndex(I) : M > N ?
        CartesianIndex(ntuple(i -> I[i], Val(N))) :
        CartesianIndices(axes(a))[LinearIndices(a)[I...]]
    convert(T, a.broadcast[index])
end

# Delaying a user function or custom indexing could suppress an exception or
# change observable effects at an unread element. Compose only total arithmetic
# over ordinary floating storage. Every other runtime type keeps materialization.
_cell_array_operand(x) = false
_cell_array_operand(x::Union{Float32,Float64,Array{Float32},Array{Float64},_CellArray}) = true
_cell_array_operand(x::Union{Int,Int32,Bool}) = true
_cell_array_operand(x::Base.Broadcast.Broadcasted) =
    _cell_total_function(x.f) && all(_cell_array_operand, x.args)
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
@inline function _cell_array_broadcast(bc::Base.Broadcast.Broadcasted)
    if bc.style isa Base.Broadcast.DefaultArrayStyle &&
            _cell_array_operand(bc)
        ready = Base.Broadcast.instantiate(bc)
        T = Base.Broadcast.combine_eltypes(ready.f, ready.args)
        if (T === Float32 || T === Float64) && !isempty(axes(ready))
            return _CellBroadcast{T,length(axes(ready)),typeof(ready)}(ready)
        end
    end
    _native_broadcast_materialize(_cell_array_materialize(bc))
end
@inline _cell_array_broadcast(x) = _native_broadcast_materialize(x)

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
    kernel_sourceop_form(op) === :fused && op.ignored_throws !== nothing || return nothing
    f = op.f
    f isa _KernelBranch && (f = f.call)
    f = _kernel_native_source(f)
    f isa Function && fieldcount(typeof(f)) == 0 || return nothing
    params = _closure_parameter_names(f, length(args))
    params === nothing && return nothing
    source = _lane_expand_dot(_lane_strip_lines(r.source), parentmodule(typeof(f)))
    source === nothing && return nothing
    native = _kernel_native_body(source, parentmodule(typeof(f)), Set{Symbol}(params))
    ctx = _LaneSourceContext(parentmodule(typeof(f)), Dict{Symbol,Any}(zip(params, args)))
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
    rhs isa Expr && rhs.head === :call || return rhs
    fn = rhs.args[1]
    fn isa Expr && fn.head === :ref && length(fn.args) == 2 &&
        fn.args[1] === _OPS_ARG && fn.args[2] isa Int || return rhs
    recipe = recipes[fn.args[2]]
    recipe.op === identity && length(rhs.args) == 2 && return rhs.args[2]
    source = _cell_array_source(recipe, rhs.args[2:end])
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
    if x.head === :if
        arms = x.args[2:end]
        return any(a -> _cell_array_aliases(a, names), arms) &&
            all(a -> _cell_array_aliases(a, names) || _cell_array_throws(a), arms)
    end
    x.head === :block && return !isempty(x.args) && _cell_array_aliases(last(x.args), names)
    false
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
            marker = x.args[2]
            if marker isa Expr && marker.head === :call &&
                    _cell_array_function(marker.args[1]) === Val &&
                    length(marker.args) == 2 && marker.args[2] isa QuoteNode
                atomic = marker.args[2].value
                return atomic isa Tuple && all(enumerate(x.args[3:end])) do (i, a)
                    i in atomic || !_cell_array_has(a, names)
                end
            end
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
        _cell_array_function(x.args[1]) === Base.Broadcast.broadcasted || return false
    !_cell_array_has(x.args[2], names) &&
        all(a -> _cell_array_broadcast_reads(a, names), x.args[3:end])
end

function _cell_array_producer(x)
    x isa Expr && x.head === :call || return nothing
    fn = _cell_array_function(first(x.args))
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
            _cell_array_function(first(x.args)) === Base.Broadcast.broadcasted
        return Expr(:call, x.args[1], x.args[2], map(_cell_array_child, x.args[3:end])...)
    end
    x
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
    _index_run_visit(body) do ex
        if ex.head === :(=) && length(ex.args) == 2 && ex.args[1] isa Symbol
            name = ex.args[1]
            push!(ordered, name)
            haskey(statements, name) && push!(repeated, name)
            statements[name] = ex
            sources[name] = _cell_array_expand_call(ex.args[2], recipes)
        elseif ex.head === :local && length(ex.args) == 1
            typed = only(ex.args)
            typed isa Expr && typed.head === :(::) && typed.args[1] isa Symbol &&
                (declarations[typed.args[1]] = typed.args[2])
        end
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
            producer = _cell_array_producer(sources[name])
            producer === nothing && continue
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
                    rhs = get(sources, lhs, ex.args[2])
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
            if _cell_array_has(get(sources, lhs, rhs), changed)
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
