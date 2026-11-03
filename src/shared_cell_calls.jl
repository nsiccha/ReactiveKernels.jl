# Equal plate cells share one data-only value.
#
# A plate's cells often derive the same structure from equal data: every
# subject of a simulation scenario shares one regimen and one observation
# grid, so each subject's design (index tables, time grids) is the same value.
# RK evaluates cell values per cell, so a position-batched read computed that
# design once per subject and call and kept one copy per subject, which the
# per-draw work then read (snag data-only-node-r-85fe8c3f: 256 copies of one
# 53 KB index table per read of a 256-subject population).
#
# Where RK evaluates a plate's data-only cell values over the whole domain at
# once — the per-call prefix plate of a position-batched reader
# (`_replicated_split_plate`, position_cells.jl) and bind-time partial
# evaluation (`_partial_plate_recipe`, partial_evaluation.jl) — an authored
# call whose arguments compared by value include an array or a record runs
# once per distinct argument tuple, and every cell with that tuple receives the
# same result object. An argument that varies across cells is compared by
# value (`hash`, then `isequal`); one that every cell reads as the same object
# (a closure or `Ref` operand, a value of the closures alone, or the result of
# an earlier shared call, which equal cells already share) is compared by
# identity. A call of numbers alone (an index, a gathered scalar) costs less
# than its key and runs in every cell, as before.
#
# Values are unchanged: a pure call of equal arguments returns an equal value,
# and graph values are read-only, so sharing the object is visible only to
# `===`. The table of one shared call is a local of the native plate loop
# (`_lower_authored_plate_loop_native!`): `nothing` before the first cell, a
# `Dict` typed by that call's key and result from then on, gone when the plate
# returns. It is never kept on a kernel, shared between reader copies or
# threads, or reused by a later call, so a caller may change its data between
# calls. Anywhere else (a cell kernel called on its own, the plate operation's
# per-cell fallback) the call runs in every cell.
#
# Under automatic differentiation each cell evaluates its call, as before
# (`EnzymeCore.within_autodiff`): sharing saves primal work, and a table of
# results that may be active would need runtime activity.

# An argument compared by identity. `objectid` agrees with `===` for mutable
# and immutable values alike.
struct _SameObject{T}
    x::T
end
Base.hash(s::_SameObject, h::UInt) = hash(objectid(s.x), h)
Base.isequal(a::_SameObject, b::_SameObject) = a.x === b.x

# The calls equal cells share: an authored expression, or a call of a function
# of a package or of the authoring module. Base's and Core's generic functions
# (indexing, arithmetic, concatenation) cost about what their key costs, and
# ReactiveKernels' own operations (plates, scans, derivative rules) are
# structure the lowering reads, so neither is shared. One method for every
# operation type: this runs for every recipe of every rebinding.
Base.@nospecializeinfer function _shared_call_op(@nospecialize(op))
    op isa _KernelSourceOp && return true
    # The `on_error = :ignore` twin of a call (error_policy.jl).
    op isa _IgnoredThrowFunction && return _shared_call_op(getfield(op, :f))
    op isa Function || return false
    root = Base.moduleroot(parentmodule(typeof(op)))
    !(root === Base || root === Core || root === Base.moduleroot(@__MODULE__))
end

# A value that is a few numbers: cheaper to compute than to key.
_shared_scalar(::Type) = false
_shared_scalar(::Type{<:Union{Number,Symbol,Char,Nothing,Missing}}) = true

# Worth a key only when an argument compared by value is more than a number.
@generated function _shared_worthwhile(::Val{V}, args::Tuple) where {V}
    any(i -> V[i] && !_shared_scalar(fieldtype(args, i)), 1:fieldcount(args))
end

@generated function _shared_key(::Val{V}, args::Tuple) where {V}
    Expr(:tuple, (V[i] ? :(getfield(args, $i)) :
                  :(_SameObject(getfield(args, $i))) for i in 1:fieldcount(args))...)
end

"""
    _SharedCall{Value}(f)

A plate cell's call of `f` that cells with equal arguments share
(`Value[i]`: argument `i` is compared by value, otherwise by identity). Called
on its own it calls `f`; the native plate loop shares it through
`_shared_step` and a table of its own.
"""
struct _SharedCall{Value,F}
    f::F
end
_SharedCall{Value}(f::F) where {Value,F} = _SharedCall{Value,F}(f)
@inline (c::_SharedCall)(args::Vararg{Any,N}) where {N} = c.f(args...)
_opname(c::_SharedCall) = "shared(" * _opname(c.f) * ")"
# The `on_error = :ignore` body (error_policy.jl) shares the throw-stripped call.
_kernel_ignore_throw_op(c::_SharedCall{V}) where {V} = _SharedCall{V}(_kernel_ignore_throw_op(c.f))

# One cell's evaluation of shared call `c`, with the plate loop's table for
# it: `nothing` before the first cell that keys it. Returns the table (created
# at that first cell, typed by the call's key and result) and the cell's
# value, which is the first equal cell's result object.
@inline function _shared_step(table, c::_SharedCall{V}, args::Vararg{Any,N}) where {V,N}
    (EnzymeCore.within_autodiff() || !_shared_worthwhile(Val(V), args)) &&
        return (table, c.f(args...))
    key = _shared_key(Val(V), args)
    R = _promote_op(c.f, map(typeof, args)...)
    T = isconcretetype(R) ? R : Any
    dict = _shared_table(table, typeof(key), T)
    dict === nothing && return (table, c.f(args...))
    (dict, get!(() -> c.f(args...), dict, key)::T)
end
# Any other operation in a shared call's slot is called in every cell.
@inline _shared_step(table, f, args::Vararg{Any,N}) where {N} = (table, f(args...))
@inline _shared_table(::Nothing, ::Type{K}, ::Type{T}) where {K,T} = Dict{K,T}()
@inline _shared_table(table::Dict{K,T}, ::Type{K}, ::Type{T}) where {K,T} = table
# A cell whose arguments have other types than the first keyed cell's keeps
# its own evaluation.
@inline _shared_table(table, ::Type, ::Type) = nothing

# The parameter of a cell lowering that holds recipe `id`'s table, or `nothing`
# when the lowering is not embedded in a plate loop that carries one.
function _shared_table_param(shares::Vector{Pair{Int,Symbol}}, id::Int)
    index = findfirst(pair -> first(pair) == id, shares)
    index === nothing ? nothing : last(shares[index])
end

# The shared calls of a cell's recipes `recipes` (in order), in graph `g`:
# each call (`_shared_call_op`) that reads a cell-varying value (`varying`)
# becomes a `_SharedCall`. An argument is compared by value when it varies
# across cells and was not returned by an earlier shared call, by identity
# otherwise. A call none of whose arguments is compared by value is left
# alone. `readable` maps each rewritten recipe's id to the rewritten recipe.
function _shared_cell_calls(g::Graph, recipes::Vector{Recipe}, varying::Set{ValueId},
                            readable)
    shared_outputs = Set{ValueId}()
    rewritten = Recipe[]
    for q in recipes
        if _shared_call_op(q.op) && !q.effectful && length(q.outputs) == 1
            by_value = Tuple(canon_id(g, v.id) in varying &&
                             !(canon_id(g, v.id) in shared_outputs) for v in q.inputs)
            if any(by_value)
                shared = Recipe(q.id, q.inputs, q.outputs, _SharedCall{by_value}(q.op),
                                q.cost, nothing, false)
                readable[q.id] = shared
                push!(shared_outputs, canon_id(g, only(q.outputs).id))
                push!(rewritten, shared)
                continue
            end
        end
        push!(rewritten, q)
    end
    rewritten
end

# Bind time (`_partial_plate_recipe`): which of recipe `r`'s broadcast
# arguments `args` are compared by value, or `nothing` when no shared
# evaluation is worth a key and every cell keeps its own. A `Ref` operand (an
# atomic argument or a value of the closures alone) and the cell values of an
# earlier shared call (`shared`) are compared by identity; a per-cell array
# of numbers is not worth a key.
Base.@nospecializeinfer function _partial_shared_by_value(
        r::Recipe, @nospecialize(args::Tuple), g::Graph, shared::Set{ValueId})
    _shared_call_op(r.op) && !r.effectful || return nothing
    by_value = Vector{Bool}(undef, length(args))
    worthwhile = false
    for i in eachindex(by_value)
        arg = args[i]
        by_value[i] = !(arg isa Ref) && !(canon_id(g, r.inputs[i].id) in shared)
        worthwhile |= by_value[i] && arg isa AbstractArray && !_shared_scalar(eltype(arg))
    end
    worthwhile ? Tuple(by_value) : nothing
end

# Every cell's value of one data-only recipe, broadcast over the plate's bound
# domain in `batch`, into `result`, cells with equal arguments (`Value`, as
# `_partial_shared_by_value` gives it) receiving the first such cell's result
# object.
function _shared_cell_values!(result, batch::Base.Broadcast.Broadcasted, ::Val{V}) where {V}
    isempty(CartesianIndices(axes(batch))) && return result
    index = first(CartesianIndices(axes(batch)))
    key = _shared_key(Val(V), Base.Broadcast._getindex(batch.args, index))
    _shared_cell_values!(result, batch, Val(V), Dict{typeof(key),eltype(result)}())
end
function _shared_cell_values!(result, batch::Base.Broadcast.Broadcasted, ::Val{V},
                              dict::Dict{K}) where {V,K}
    f = batch.f
    for index in CartesianIndices(axes(batch))
        args = Base.Broadcast._getindex(batch.args, index)
        key = _shared_key(Val(V), args)
        result[index] = key isa K ? get!(() -> f(args...), dict, key) : f(args...)
    end
    result
end
