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
# once, an authored call (or a call of a function of a package or of the
# authoring module) whose arguments compared by value include an array or a
# record runs once per distinct argument tuple, and every cell with that tuple
# receives the same result object. Those evaluations are the data-only part of a
# plate that also reads a position (the per-call prefix plate of a
# position-batched reader, `_replicated_split_plate`, position_cells.jl; bind-time
# partial evaluation, `_partial_plate_recipe`, partial_evaluation.jl) and a plate
# that reads data alone, in either prefix (`_shared_data_plate`). The last keeps
# its own loop unless a cell calls a package's function or reads a value equal
# cells already share: a cell of Base arithmetic costs about what its key costs,
# and its loop may have a vectorized form.
#
# Every key compares identities first, which costs the same whatever an argument's
# size: arguments that are the same objects have equal values, so the second cell
# holding them hashes nothing. An argument that varies across cells is compared by
# value when its object is new, at most once per distinct object: `isequal` alone
# while the table holds one value, otherwise `hash` then `isequal`. The same
# object says the values are equal, but different objects do not
# say they differ. Each subject's rows of a data column are different indices, so
# the column's values gathered for equal subjects are equal values in fresh objects,
# and a call reading them shares only by value.
#
# Which arguments can skip the comparison by value: a closure or `Ref` operand and
# a value of the closures alone are one object in every cell. A value equal cells
# usually hold as one object (the result of an earlier shared call, a field read of
# such a value, or the cell of an earlier data-only plate whose cells are such
# values: `one_object`) skips it in an expression of Base's functions alone, whose
# cost is about its key's, and not in a package's call, which is what sharing pays
# for. A call of numbers alone (an index, a gathered scalar) costs less than its key
# and runs in every cell, as before; so does a field read `x.name`, whose key would
# hash the whole record to save a load.
#
# Values are unchanged: a pure call of equal arguments returns an equal value,
# and graph values are read-only, so sharing the object is visible only to
# `===`. The table of one shared call is a local of the native plate loop
# (`_lower_authored_plate_loop_native!`): `nothing` before the first cell, a
# `_SharedTable` typed by that call's keys and result from then on, gone when the
# plate returns. A scheduled prefix's chunks share a synchronized table. It is
# never kept on a kernel, shared between reader copies, or reused by a later call,
# so a caller may change its data between
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

# Worth a key when an argument compared by value is more than a number, or when
# no argument is compared by value: the key is then a few identities.
@generated function _shared_worthwhile(::Val{V}, args::Tuple) where {V}
    !any(V) || any(i -> V[i] && !_shared_scalar(fieldtype(args, i)), 1:fieldcount(args))
end

@generated function _shared_key(::Val{V}, args::Tuple) where {V}
    Expr(:tuple, (V[i] ? :(getfield(args, $i)) :
                  :(_SameObject(getfield(args, $i))) for i in 1:fieldcount(args))...)
end

# The key of the arguments' identities alone.
@generated function _shared_objects(args::Tuple)
    Expr(:tuple, (:(_SameObject(getfield(args, $i))) for i in 1:fieldcount(args))...)
end

# One shared call's results: by the identities of the arguments (`objects`), and by
# the key that compares them by value (`values`), reached only for new objects.
struct _SharedTable{O,K,T}
    objects::Dict{O,T}
    values::Dict{K,T}
end

# The result for `args` (`objects` and `key` being their two keys), computed by the
# first cell with equal arguments.
@inline function _shared_result(f::F, table::_SharedTable, objects, key, args::Tuple) where {F}
    get!(table.objects, objects) do
        # A common plate has only one distinct argument tuple, but gathers
        # produce a fresh copy in every cell. Compare that sole entry directly:
        # hashing its arrays first costs more than the equality check itself.
        # Once a second distinct tuple appears, keep the dictionary lookup so
        # diverse cells never search a growing list of earlier arguments.
        if length(table.values) == 1
            entry = first(table.values)
            isequal(first(entry), key) && return last(entry)
        end
        get!(() -> f(args...), table.values, key)
    end
end

"""
    _SharedCall{Value}(f)

A plate cell's call of `f` that cells with equal arguments share. Cells compare
their arguments by identity, then, when those objects are new, compare each
argument `i` with `Value[i]` by value. Called on its own it calls `f`; the native
plate loop shares it through `_shared_step` and a table of its own.
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
# at that first cell, typed by the call's keys and result) and the cell's
# value, which is the first equal cell's result object.
@inline function _shared_step(table, c::_SharedCall{V}, args::Vararg{Any,N}) where {V,N}
    (EnzymeCore.within_autodiff() || !_shared_worthwhile(Val(V), args)) &&
        return (table, c.f(args...))
    objects = _shared_objects(args)
    key = _shared_key(Val(V), args)
    R = _promote_op(c.f, map(typeof, args)...)
    T = isconcretetype(R) ? R : Any
    shared = _shared_table(table, typeof(objects), typeof(key), T)
    shared === nothing && return (table, c.f(args...))
    (shared, _shared_result(c.f, shared, objects, key, args)::T)
end
# Any other operation in a shared call's slot is called in every cell.
@inline _shared_step(table, f, args::Vararg{Any,N}) where {N} = (table, f(args...))
@inline _shared_table(::Nothing, ::Type{O}, ::Type{K}, ::Type{T}) where {O,K,T} =
    _SharedTable(Dict{O,T}(), Dict{K,T}())
@inline _shared_table(table::_SharedTable{O,K,T}, ::Type{O}, ::Type{K}, ::Type{T}) where {O,K,T} = table
# A cell whose arguments have other types than the first keyed cell's keeps
# its own evaluation.
@inline _shared_table(table, ::Type, ::Type, ::Type) = nothing

# A scheduled prefix shares one table across its chunks, for this call only.
# Reserve a key under the lock, then evaluate outside it: distinct expensive
# calls run concurrently, while equal cells wait for the same result. Failed
# evaluations also wake every waiter before the scheduler joins the tasks.
mutable struct _ConcurrentSharedTable
    lock::ReentrantLock
    table::Any
end
_ConcurrentSharedTable() = _ConcurrentSharedTable(ReentrantLock(), nothing)

mutable struct _SharedPending{T}
    ready::Base.Event
    value::Union{Nothing,Some{T}}
    error::Any
end
_SharedPending{T}() where {T} = _SharedPending{T}(Base.Event(), nothing, nothing)

@inline function _shared_step(slot::_ConcurrentSharedTable, c::_SharedCall{V},
                              args::Vararg{Any,N}) where {V,N}
    (EnzymeCore.within_autodiff() || !_shared_worthwhile(Val(V), args)) &&
        return (slot, c.f(args...))
    objects, key = _shared_objects(args), _shared_key(Val(V), args)
    R = _promote_op(c.f, map(typeof, args)...)
    T = isconcretetype(R) ? R : Any
    pending, owner = _shared_reserve!(slot, objects, key, T)
    pending === nothing && return (slot, c.f(args...))
    if owner
        try
            pending.value = Some{T}(c.f(args...))
        catch error
            pending.error = Some(error)
        finally
            notify(pending.ready)
        end
    else
        wait(pending.ready)
    end
    pending.error === nothing || throw(something(pending.error))
    (slot, something(pending.value)::T)
end

function _shared_reserve!(slot::_ConcurrentSharedTable, objects::O, key::K,
                          ::Type{T}) where {O,K,T}
    lock(slot.lock)
    try
        table = _shared_table(slot.table, O, K, _SharedPending{T})
        table === nothing && return (nothing, false)
        slot.table = table
        owner = false
        pending = _shared_result(table, objects, key, ()) do
            owner = true
            _SharedPending{T}()
        end
        (pending, owner)
    finally
        unlock(slot.lock)
    end
end

# The parameter of a cell lowering that holds recipe `id`'s table, or `nothing`
# when the lowering is not embedded in a plate loop that carries one.
function _shared_table_param(shares::Vector{Pair{Int,Symbol}}, id::Int)
    index = findfirst(pair -> first(pair) == id, shares)
    index === nothing ? nothing : last(shares[index])
end

# Whether recipe `q` is the property read `x.name` of one value.
function _shared_field_read(q::Recipe)
    q.op isa _KernelSourceOp && length(q.inputs) == 1 || return false
    analyzed = _cell_native(q; outlined = true)
    analyzed === nothing ? false : _native_field_read(analyzed...)
end

# Whether expression `ex`, over `params`, names a function of a package or of the
# authoring module.
function _names_package_function(ex, params, mod)
    if ex isa Symbol || ex isa GlobalRef
        ex in params && return false
        f = _cell_binding(ex, mod)
        return f isa Function && _shared_call_op(f)
    end
    ex isa Expr || return false
    any(arg -> _names_package_function(arg, params, mod), ex.args)
end

# Whether recipe `q` calls a function of a package or of the authoring module,
# as far as its source shows: an expression of Base's functions does not.
function _shared_package_call(q::Recipe)
    q.op isa _KernelSourceOp || return true
    analyzed = _cell_native(q; outlined = true, branches = true)
    analyzed === nothing ? false : _names_package_function(analyzed...)
end

# The shared calls of a cell's recipes `recipes` (in order), in graph `g`: each
# call (`_shared_call_op`) that reads a cell-varying value (`varying`) becomes a
# `_SharedCall`, except a field read, which stays as it is. `one_object` holds the
# values equal cells usually hold as one object: it starts with the cell's arguments
# of that kind and gains each shared call's result and each field read of a value
# already in it. A varying argument is compared by value when its object is new,
# unless it is in `one_object` and the call is an expression of Base's functions
# alone (`_shared_package_call`). With `package_calls_only`, such an expression is
# shared only if all its varying arguments are in `one_object`. `readable` maps each
# rewritten recipe's id to the rewritten recipe.
function _shared_cell_calls(g::Graph, recipes::Vector{Recipe}, varying::Set{ValueId},
                            readable, one_object::Set{ValueId};
                            package_calls_only::Bool = false)
    rewritten = Recipe[]
    for q in recipes
        if _shared_call_op(q.op) && !q.effectful && length(q.outputs) == 1
            ids = ValueId[canon_id(g, v.id) for v in q.inputs]
            output = canon_id(g, only(q.outputs).id)
            if any(id -> id in varying, ids)
                if _shared_field_read(q)
                    only(ids) in one_object && push!(one_object, output)
                else
                    package = _shared_package_call(q)
                    held = all(id -> !(id in varying) || id in one_object, ids)
                    if !package_calls_only || held || package
                        by_value = Tuple(id in varying && (package || !(id in one_object)) for id in ids)
                        shared = Recipe(q.id, q.inputs, q.outputs, _SharedCall{by_value}(q.op),
                                        q.cost, nothing, false)
                        readable[q.id] = shared
                        push!(one_object, output)
                        push!(rewritten, shared)
                        continue
                    end
                end
            end
        end
        push!(rewritten, q)
    end
    rewritten
end

# Whether the cell of plate `recipe` has one output per recipe and one
# distinguished result, as the native plate lowering and these rewrites expect.
Base.@nospecializeinfer function _plate_cell_rewritable(
        recipe::Recipe, @nospecialize(op::_AuthoredPlateOp))
    kernel = op.kernel
    inner = kernel.plan
    length(recipe.outputs) == 1 || return false
    length(inner.have) == length(recipe.inputs) || return false
    length(inner.want) == 1 || return false
    length(kernel.lowered_recipes) == length(inner.recipes) || return false
    assigned = Set(canon_id(inner.graph, v.id) for v in inner.have)
    for q in inner.recipes
        q.effectful && return false
        length(q.outputs) == 1 || return false
        cid = canon_id(inner.graph, only(q.outputs).id)
        cid in assigned && return false
        push!(assigned, cid)
    end
    true
end

# A plate that reads data alone, with its cells' calls shared. `one_object` holds
# the outer values whose cells equal cells usually hold as one object: the plate
# reads those as such and joins them when its own result is such a value. A plate
# whose output is in `kept` stays as it is, as does one with nothing to share.
_shared_data_plate(g, recipe, op, kept, one_object) = recipe
Base.@nospecializeinfer function _shared_data_plate(
        g, recipe::Recipe, @nospecialize(op::_AuthoredPlateOp), kept::Set{ValueId},
        one_object::Set{ValueId})
    _plate_cell_rewritable(recipe, op) || return recipe
    any(v -> canon_id(g, v.id) in kept, recipe.outputs) && return recipe
    atomic = _plate_atomic(op)
    kernel = op.kernel
    inner = kernel.plan
    ig = inner.graph
    varying = Set{ValueId}()
    same = Set{ValueId}()
    for j in eachindex(inner.have)
        j in atomic && continue
        cid = canon_id(ig, inner.have[j].id)
        push!(varying, cid)
        canon_id(g, recipe.inputs[j].id) in one_object && push!(same, cid)
    end
    isempty(varying) && return recipe
    for q in inner.recipes
        any(v -> canon_id(ig, v.id) in varying, q.inputs) &&
            push!(varying, canon_id(ig, only(q.outputs).id))
    end
    readable = Dict{Int,Any}(q.id => source for (q, source) in
                             zip(inner.recipes, kernel.lowered_recipes))
    rewritten = _shared_cell_calls(ig, inner.recipes, varying, readable, same;
                                   package_calls_only = true)
    canon_id(ig, only(inner.want).id) in same &&
        push!(one_object, canon_id(g, only(recipe.outputs).id))
    any(q -> q.op isa _SharedCall, rewritten) || return recipe
    template = Plan(ig, inner.have, inner.want, inner.recipes, inner.producer,
                    inner.cost, inner.candidates)
    shared = _position_cell_kernel(_partial_subplan(template, inner.have, inner.want, rewritten),
                                   readable)
    Recipe(recipe.id, recipe.inputs, recipe.outputs,
           _AuthoredPlateOp{typeof(shared),atomic}(shared, op.axis_checks),
           recipe.cost, nothing, recipe.effectful, recipe.source)
end

# The values of plan `p` a data-only plate must not share cells for: what the
# caller receives, whose cells it may write, and what a plate-cell reader selects
# from, whose lowering reads the plate itself.
Base.@nospecializeinfer function _shared_kept_values(p::Plan)
    kept = Set{ValueId}(canon_id(p.graph, v.id) for v in p.want)
    for r in p.recipes
        r.op isa _AuthoredPlateCellLike || continue
        foreach(v -> push!(kept, canon_id(p.graph, v.id)), r.inputs)
    end
    kept
end

# Bind time (`_partial_plate_recipe`): which of recipe `r`'s broadcast arguments
# `args` are compared by value when their objects are new, or `nothing` when no
# shared evaluation is worth a key and every cell keeps its own. A `Ref` operand (an
# atomic argument or a value of the closures alone) is not compared by value, nor
# are the cell values of a value equal cells usually hold as one object (`shared`)
# unless the recipe calls a package's function (`package`); a per-cell array of
# numbers is not worth a key. A recipe with no argument compared by value keys
# identities alone.
Base.@nospecializeinfer function _partial_shared_by_value(
        r::Recipe, @nospecialize(args::Tuple), g::Graph, shared::Set{ValueId}, package::Bool)
    _shared_call_op(r.op) && !r.effectful || return nothing
    by_value = Vector{Bool}(undef, length(args))
    worthwhile = false
    per_cell = false
    for i in eachindex(by_value)
        arg = args[i]
        by_value[i] = !(arg isa Ref) && (package || !(canon_id(g, r.inputs[i].id) in shared))
        per_cell |= arg isa AbstractArray
        worthwhile |= by_value[i] && arg isa AbstractArray && !_shared_scalar(eltype(arg))
    end
    worthwhile || (per_cell && !any(by_value)) ? Tuple(by_value) : nothing
end

# Every cell's value of one data-only recipe, broadcast over the plate's bound
# domain in `batch`, into `result`, cells with equal arguments (`Value`, as
# `_partial_shared_by_value` gives it) receiving the first such cell's result
# object.
function _shared_cell_values!(result, batch::Base.Broadcast.Broadcasted, ::Val{V}) where {V}
    isempty(CartesianIndices(axes(batch))) && return result
    args = Base.Broadcast._getindex(batch.args, first(CartesianIndices(axes(batch))))
    table = _shared_table(nothing, typeof(_shared_objects(args)),
                          typeof(_shared_key(Val(V), args)), eltype(result))
    _shared_cell_values!(result, batch, Val(V), table)
end
function _shared_cell_values!(result, batch::Base.Broadcast.Broadcasted, ::Val{V},
                              table::_SharedTable{O,K}) where {V,O,K}
    f = batch.f
    for index in CartesianIndices(axes(batch))
        args = Base.Broadcast._getindex(batch.args, index)
        objects = _shared_objects(args)
        key = _shared_key(Val(V), args)
        result[index] = objects isa O && key isa K ?
            _shared_result(f, table, objects, key, args) : f(args...)
    end
    result
end
