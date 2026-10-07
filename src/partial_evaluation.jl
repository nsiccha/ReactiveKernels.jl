# Partial evaluation of planned kernels.
#
# One general, reusable Plan-level pre-pass (user-resolved shape, decisions
# 2026-09-01T09-38-08-191-028fb60 + follow-up comment): given the subset of
# HAVE ports whose runtime values are fixed per binding ("bound" data ports)
# and those values, transform the Plan itself —
#
#   1. partition the selected recipes into a data-only prefix (every input
#      reachable exclusively from bound ports) and the residual body;
#   2. prepare and run the prefix exactly once, here;
#   3. return an ordinary residual Plan over the remaining HAVE ports, in
#      which each hoisted value re-enters as a zero-input constant recipe.
#
# The output is a plain `Plan`, so every downstream surface — `prepare`,
# `prepare_nonallocating`, AD preparation, plates, embedded composition,
# compile backends — consumes it unchanged: the pass only transforms the
# graph/AST layer and adds no new runtime object or calling convention. The
# constants ride the existing `__ops__` tuple as nullary operations, so the
# generated body keeps the established `(__ops__, args...)` ABI.
#
# Purity of the prefix is the ordinary recipe contract: effectful recipes
# never enter a plan (planner.jl), so running the prefix at bind time cannot
# observe or produce side effects beyond what every prepared call already did.
# The shared `Graph` is never mutated: synthetic constant recipes exist only
# in the returned residual Plan, under negative ids that cannot collide with
# graph recipe ids, so later plans over the same graph are unaffected.

"""
    _BoundConstant(value)

A nullary recipe operation carrying one bind-time-evaluated value into a
residual plan. Calling it returns the stored value; it allocates nothing.
"""
struct _BoundConstant{T}
    value::T
end
@inline (c::_BoundConstant)() = c.value
Base.show(io::IO, c::_BoundConstant) = print(io, "bound_constant(",
                                             summary(c.value), ")")
_opname(::_BoundConstant) = "bound_constant"

"""
    _partial_split(p::Plan, bound::Set{Int}, excluded = Set{Int}()) -> (prefix, residual, prefix_owned)

Partition `p.recipes` (kept in execution order) into the data-only `prefix` —
recipes whose every input is a bound HAVE port or a value owned by an earlier
prefix recipe — and the `residual` rest. Ownership follows the lowering's
first-producer-wins rule: a value emitted collaterally by a later recipe is a
discarded duplicate, so it neither makes that recipe's consumers hoistable nor
un-hoists the authoritative producer. `prefix_owned` is the canonical id set
available at bind time: the bound ports plus every prefix-owned output.

Zero-input recipes are data-only by definition and hoist under any bound set,
including an empty one. Recipes whose ids are in `excluded` stay in the
residual whatever their inputs, and so do their data-only consumers.
"""
function _partial_split(p::Plan, bound::Set{Int}, excluded::Set{Int} = Set{Int}())
    g = p.graph
    prefix_owned = copy(bound)
    assigned = Set(canon_id(g, v.id) for v in p.have)
    prefix = Recipe[]
    residual = Recipe[]
    for r in p.recipes
        if !(r.id in excluded) &&
           all(inp -> canon_id(g, inp.id) in prefix_owned, r.inputs)
            push!(prefix, r)
            for o in r.outputs
                cid = canon_id(g, o.id)
                cid in assigned && continue
                push!(assigned, cid)
                push!(prefix_owned, cid)
            end
        else
            push!(residual, r)
            for o in r.outputs
                cid = canon_id(g, o.id)
                cid in assigned || push!(assigned, cid)
            end
        end
    end
    prefix, residual, prefix_owned
end

# The hoisted-constant boundary: every value the residual body (or the WANT
# list) consumes whose authoritative binding is bind-time — a bound HAVE port
# or a prefix-owned output. Ordered deterministically by first residual use,
# then first WANT use.
function _partial_constants(p::Plan, prefix_owned::Set{Int},
                            residual::Vector{Recipe})
    g = p.graph
    constants = Value[]
    seen = Set{Int}()
    consider(v::Value) = begin
        cid = canon_id(g, v.id)
        cid in prefix_owned || return
        cid in seen && return
        push!(seen, cid)
        push!(constants, g.values[cid])
    end
    for r in residual, inp in r.inputs
        consider(inp)
    end
    for w in p.want
        consider(w)
    end
    constants
end

# Build a Plan over a subset (or synthetic extension) of an existing plan's
# recipes. The relative execution order of `recipes` must already be a valid
# topological order for the `have` boundary; both callers inherit it from
# `p.recipes`, prepending only zero-input constant recipes.
function _partial_subplan(p::Plan, have::Vector{Value}, want::Vector{Value},
                          recipes::Vector{Recipe})
    g = p.graph
    have_ids = Set(canon_id(g, v.id) for v in have)
    producer = Dict{Int,Recipe}()
    for r in recipes, o in r.outputs
        cid = canon_id(g, o.id)
        cid in have_ids && continue
        haskey(producer, cid) || (producer[cid] = r)
    end
    cost = sum(r.cost for r in recipes; init = 0.0)
    Plan(g, have, want, recipes, producer, cost, recipes)
end

_partial_prefix_values(want::Vector{Value}, result) =
    length(want) == 1 ? (result,) : result

# Inner plates retain their original arguments, even when the scalar residual
# no longer reads them: unused arguments still constrain the broadcast domain.
# Cached frontier values are additional ordinary plate inputs. This keeps shape
# validation, marker discovery, and subsequent plate-chain composition on their
# existing paths, at the cost of retaining the original bound storage.
_partial_plate_input(value) = false
_partial_plate_input(::Number) = true
_partial_plate_input(::AbstractArray) = true

# Keep the cache boundary on ordinary primitive numeric values and dense arrays
# of them. A per-cell array (a ragged index list, say) is cached as an array of
# arrays, the same operand representation as bound ragged data the cell could
# already receive. An isbits tuple/struct is safe to hold natively, but an array
# of those values would introduce a new operand representation at the tensor
# backend boundary.
const _PartialPlateScalar = Union{Bool,Int8,Int16,Int32,Int64,
    UInt8,UInt16,UInt32,UInt64,Float16,Float32,Float64}
_partial_plate_cacheable(::Type) = false
_partial_plate_cacheable(::Type{<:_PartialPlateScalar}) = true
_partial_plate_cacheable(::Type{<:Array{<:_PartialPlateScalar}}) = true

# A live array can make a singleton or previously absent dimension empty.
# Such a domain must keep the original lazy per-cell execution: eager prefix
# evaluation could otherwise throw for a cell that is never visited. A known
# rank is safe only when bound non-singleton axes already fix each dimension
# it can contribute. Atomic arguments never contribute axes.
_partial_plate_live_domain(::Type, bound_axes) = false
_partial_plate_live_domain(::Type{<:Number}, bound_axes) = true
function _partial_plate_live_domain(::Type{<:AbstractArray{T,N}},
                                    bound_axes) where {T,N}
    N isa Int && N <= length(bound_axes) &&
        all(d -> length(bound_axes[d]) > 1, 1:N)
end

_partial_plate_recipe(g, recipe, known, op) = nothing
# Generated inner-kernel types are data to this structural pass. Specializing
# on them recompiles the entire partial evaluator once per authored plate.
Base.@nospecializeinfer function _partial_plate_recipe(
        g, recipe, known, @nospecialize(op::_AuthoredPlateOp))
    atomic_inputs = typeof(op).parameters[2]
    inner = op.kernel.plan
    length(inner.have) == length(recipe.inputs) || return nothing
    length(inner.want) == 1 || return nothing
    length(op.kernel.ops) == length(inner.recipes) || return nothing

    # Native scalar plate lowering expects one output per recipe. Do not turn
    # overlapping producer bindings or effects into a new specialization
    # boundary; the ordinary lowering remains authoritative. A nested plate,
    # scan or embedded kernel is never evaluated here: it stays in the residual
    # cell, lowered exactly as before, and only the cell's other bound-only
    # recipes are cached.
    assigned = Set(canon_id(inner.graph, v.id) for v in inner.have)
    excluded = Set{Int}()
    for r in inner.recipes
        r.effectful && return nothing
        length(r.outputs) == 1 || return nothing
        if r.op isa Union{_AuthoredPlateOp,_AuthoredScanOp} ||
           _embedded_kernel(r.op) !== nothing
            push!(excluded, r.id)
        end
        cid = canon_id(inner.graph, only(r.outputs).id)
        cid in assigned && return nothing
        push!(assigned, cid)
    end

    bound = Set{Int}()
    arguments = Dict{Int,Any}()
    types = Dict{Int,Any}()
    for (index, (outer, input)) in enumerate(zip(recipe.inputs, inner.have))
        cid = canon_id(g, outer.id)
        haskey(known, cid) || continue
        data = known[cid]
        index in atomic_inputs || _partial_plate_input(data) || return nothing
        cid = canon_id(inner.graph, input.id)
        push!(bound, cid)
        arguments[cid] = index in atomic_inputs ? Ref(data) : data
        types[cid] = index in atomic_inputs || data isa Number ?
            typeof(data) : eltype(data)
    end
    # Nullary inner recipes can form a prefix even when this plate has no
    # bound inputs. Without a bound domain, retain its per-cell execution.
    isempty(arguments) && return nothing

    # A known empty domain has no data-dependent cells to evaluate. Keeping the
    # original plate also preserves its empty-result typing and error behavior.
    bound_axes = Base.Broadcast.combine_axes(values(arguments)...)
    any(isempty, bound_axes) && return nothing
    fixed_domain = all(enumerate(recipe.inputs)) do (index, input)
        index in atomic_inputs || haskey(known, canon_id(g, input.id)) ||
            _partial_plate_live_domain(valtype(input), bound_axes)
    end
    fixed_domain || return nothing
    # No possible batched axis: preserve the ordinary rejection.
    isempty(bound_axes) && return nothing

    have_ids = Set(canon_id(inner.graph, v.id) for v in inner.have)
    bound_types = types
    local residual, frontier, selected
    while true
        prefix, residual, owned = _partial_split(inner, bound, excluded)
        isempty(prefix) && return nothing
        frontier = filter(_partial_constants(inner, owned, residual)) do v
            !(canon_id(inner.graph, v.id) in have_ids)
        end
        isempty(frontier) && return nothing

        # Keep only the selected prefix producers needed by the cut. Never
        # re-plan: doing so could select a different producer from the public
        # graph.
        needed = Set(canon_id(inner.graph, v.id) for v in frontier)
        selected = Recipe[]
        for r in Iterators.reverse(prefix)
            canon_id(inner.graph, only(r.outputs).id) in needed || continue
            push!(selected, r)
            union!(needed, (canon_id(inner.graph, v.id) for v in r.inputs))
        end
        reverse!(selected)

        # Establish the complete eligible prefix before executing any of it.
        # Intermediates only need a concrete type: they are bind-time
        # temporaries, as they were per-evaluation ones. Only frontier values
        # cross the cache boundary. A cached array is a bind-time constant
        # shared by every later evaluation, read-only to the pure residual like
        # any top-level hoisted value or bound array.
        types = copy(bound_types)
        for r in selected
            T = Base.promote_op(r.op,
                (types[canon_id(inner.graph, v.id)] for v in r.inputs)...)
            isconcretetype(T) || return nothing
            types[canon_id(inner.graph, only(r.outputs).id)] = T
        end
        # A frontier value the cache cannot hold (a tuple, struct, view or
        # range, such as a static-vector scan seed) keeps its producer in the
        # residual cell, evaluated per cell as before; the values that producer
        # reads become the frontier instead.
        frontier_ids = Set(canon_id(inner.graph, v.id) for v in frontier)
        uncached = Int[]
        for r in selected
            cid = canon_id(inner.graph, only(r.outputs).id)
            cid in frontier_ids && !_partial_plate_cacheable(types[cid]) &&
                push!(uncached, r.id)
        end
        isempty(uncached) && break
        union!(excluded, uncached)
    end
    for r in selected
        args = Tuple(arguments[canon_id(inner.graph, v.id)] for v in r.inputs)
        batch = Base.Broadcast.instantiate(Base.broadcasted(r.op, args...))
        cid = canon_id(inner.graph, only(r.outputs).id)
        if isempty(axes(batch))
            arguments[cid] = Ref(batch[CartesianIndex()])
        else
            # Preserve the broadcast container and concrete element type. Bool
            # results may use packed BitArrays; bound-array externalization also
            # supports these at the tensor boundary.
            result = similar(batch, types[cid])
            for index in CartesianIndices(axes(batch))
                result[index] = batch[index]
            end
            arguments[cid] = result
        end
    end

    cache_values = Value[]
    cache_data = Any[]
    atomic = Int[atomic_inputs...]
    for (index, v) in enumerate(frontier)
        data = arguments[canon_id(inner.graph, v.id)]
        if data isa Ref
            data = data[]
            push!(atomic, length(recipe.inputs) + index)
        end
        push!(cache_values, value!(g, Symbol(:bound_plate_, v.name), typeof(data)))
        push!(cache_data, data)
    end
    scalar_plan = _partial_subplan(inner, vcat(inner.have, frontier),
                                  inner.want, residual)
    kernel = _prepare(scalar_plan,
        _lower_with_ops(scalar_plan; inline_embedded = false)...)
    readable = Dict(r.id => source for (r, source) in
                    zip(inner.recipes, op.kernel.lowered_recipes))
    kernel = _prepared_kernel(kernel.f, kernel.ops, kernel.inputs, kernel.outputs,
        kernel.plan, kernel.ast, Tuple(readable[r.id] for r in residual))
    replacement = _AuthoredPlateOp{typeof(kernel),Tuple(atomic)}(
        kernel, op.axis_checks)
    updated = Recipe(recipe.id, (recipe.inputs..., cache_values...),
        recipe.outputs, replacement, recipe.cost, nothing,
        recipe.effectful, recipe.source)
    (; recipe = updated, cache_values, cache_data)
end

# ---------------------------------------------------------------------------
# Data-bound branch partitioning of authored plates.
#
# A plate cell recipe whose right-hand side is a top-level lazy branch keeps
# its parts (`_KernelBranch`). When every port of the condition is bound data
# (a bound plate argument, or a bound-only cell value the cache pass above
# turned into one), the condition is evaluated per lane here, the lanes are
# split by outcome, and the plate becomes one plate per taken arm over its own
# lanes — so no backend ever receives that branch, no lane evaluates an arm it
# does not take, and every gather inside an arm is in bounds by construction
# (docs/src/constraints.md). The number of arm plates is bounded by the cell's
# branch structure, never by the data; lane-subset sizes only change shapes.

"""
    _LaneGather(n)

Gathers a live plate argument onto one arm's lanes: a lane-length vector is
indexed by the arm's (bound) lane indices, a singleton array is repeated over
them, and a scalar broadcasts unchanged. An array result is always a fresh
gather, never the argument itself, and the call inlines with its shape
check thrown out of line: native Enzyme reverse cannot statically resolve the
activity of a result that is sometimes an alias of its input, nor of an array
returned across a non-inlined call. Any other shape cannot belong to the
partitioned one-dimensional domain and is rejected.
"""
struct _LaneGather
    n::Int
end
@inline (::_LaneGather)(x::Number, lanes) = x
@inline function (gather::_LaneGather)(x::AbstractArray, lanes)
    length(x) == 1 && return x[ones(Int, length(lanes))]
    ndims(x) == 1 && length(x) == gather.n || _lane_gather_mismatch(x, gather.n)
    x[lanes]
end
@noinline _lane_gather_mismatch(x, n) = throw(DimensionMismatch(
    "a partitioned plate argument of size $(size(x)) does not match its $n lanes"))
(::_LaneGather)(x, lanes) = throw(ArgumentError("a partitioned plate " *
    "argument must be a number or a lane vector, got $(typeof(x))"))
# Callable structs have no `nameof`, so without these the readable
# Generated-kernel pane would render the partition's sourceless lane recipes
# as an opaque `operation(...)` (refused by docs/kernel_examples.jl).
_readable_callee(::_LaneGather) = :lane_gather
_opname(::_LaneGather) = "lane_gather"

"""
    _LaneAssemble(order)

Reassembles arm pointwise vectors into lane order: `vcat(parts...)[order]`
(one gather, `order = invperm(vcat(arm_lanes...))`).
"""
struct _LaneAssemble
    order::Vector{Int}
end
(assemble::_LaneAssemble)(parts...) = vcat(parts...)[assemble.order]
_readable_callee(::_LaneAssemble) = :lane_assemble
_opname(::_LaneAssemble) = "lane_assemble"

"""
    _LaneAnchored(f)

An arm body that reads no lane argument, given one it ignores so that plate
lowering evaluates it once per lane.
"""
struct _LaneAnchored{F}
    f::F
end
@inline (arm::_LaneAnchored)(anchor, args...) = arm.f(args...)

# An ignored domain anchor does not make a shared branch condition vary by
# lane. Retain its metadata with shifted argument positions, including when
# several unused lane inputs anchor the terminal recipe.
_plate_branch_anchor(operation) = nothing
function _plate_branch_anchor(operation::_KernelSourceOp{D,F,N,T}) where
        {D,F,N<:_KernelBranch,T<:_KernelBranch}
    _KernelSourceOp(Val(D), Val(F), _plate_branch_anchor_metadata(operation.f),
        _plate_branch_anchor_metadata(operation.tensor_f), operation.ignored_throws)
end
function _plate_branch_anchor_metadata(branch::_KernelBranch{CI,TI,EI}) where {CI,TI,EI}
    shift(indices) = map(index -> index + 1, indices)
    _KernelBranch(Val(shift(CI)), Val(shift(TI)), Val(shift(EI)),
        _LaneAnchored(branch.call), branch.condition, branch.then_arm, branch.else_arm)
end
function _plate_branch_anchor(operation::_LaneAnchored)
    inner = _plate_branch_anchor(operation.f)
    inner === nothing ? nothing : _plate_branch_anchor(inner)
end
@inline function _tensorized_plate_dispatch(operation::_LaneAnchored, args::Tuple)
    branch = _plate_branch_anchor(operation.f)
    branch === nothing ? _tensorized_plate_default_call(operation, args) :
        _tensorized_plate_dispatch(branch, args)
end

_partition_plate_recipe(g, recipe, known, op, pending, want, fresh) = nothing
# Branch partitioning likewise consumes the operation metadata dynamically;
# its residual operation still retains the precise generated kernel type.
Base.@nospecializeinfer function _partition_plate_recipe(
        g, recipe, known, @nospecialize(op::_AuthoredPlateOp),
        pending::Vector{Recipe}, want, fresh)
    atomic_inputs = typeof(op).parameters[2]
    inner = op.kernel.plan
    length(inner.have) == length(recipe.inputs) || return nothing
    length(inner.want) == 1 || return nothing
    length(op.kernel.ops) == length(inner.recipes) || return nothing
    length(recipe.outputs) == 1 || return nothing
    for r in inner.recipes
        r.effectful && return nothing
        length(r.outputs) == 1 || return nothing
        r.op isa Union{_AuthoredPlateOp,_AuthoredScanOp} && return nothing
        _embedded_kernel(r.op) === nothing || return nothing
    end

    # Bound plate arguments by inner port, and the one-dimensional lane count
    # they fix. Live arguments are gathered at run time (`_LaneGather`).
    bound = Dict{Int,Any}()
    n = nothing
    singleton_axis = false
    unknown_axis = false
    for (index, (outer, input)) in enumerate(zip(recipe.inputs, inner.have))
        cid = canon_id(g, outer.id)
        if !haskey(known, cid)
            index in atomic_inputs || valtype(outer) <: Number ||
                (unknown_axis = true)
            continue
        end
        data = known[cid]
        if !(index in atomic_inputs) && data isa AbstractArray
            ndims(data) == 1 || return nothing
            singleton_axis |= length(data) == 1
            if length(data) != 1
                n === nothing && (n = length(data))
                length(data) == n || return nothing
            end
        elseif !(index in atomic_inputs) && !(data isa Number)
            return nothing
        end
        bound[canon_id(inner.graph, input.id)] =
            index in atomic_inputs ? Ref(data) : data
    end
    # A bound singleton is the complete domain when every other live
    # operand is shared/scalar. Keep broadcast expansion conservative when
    # an unbound lane array could determine a larger domain.
    n === nothing && singleton_axis && !unknown_axis && (n = 1)
    (n === nothing || n == 0) && return nothing

    # The first branch recipe whose condition reads bound data only.
    target = nothing
    for r in inner.recipes
        r.op isa _KernelSourceOp && r.op.f isa _KernelBranch &&
            r.op.tensor_f isa _KernelBranch || continue
        CI = typeof(r.op.f).parameters[1]
        ports = [canon_id(inner.graph, r.inputs[i].id) for i in CI]
        all(cid -> haskey(bound, cid), ports) || continue
        target = (r, ports)
        break
    end
    target === nothing && return nothing
    branch_recipe, cond_ports = target
    native, tensor = branch_recipe.op.f, branch_recipe.op.tensor_f
    pred = broadcast(native.condition, (bound[cid] for cid in cond_ports)...)
    arms = if pred isa Bool
        # A lane-invariant condition selects its arm statically.
        pred ? [(:then, collect(1:n))] : [(:else, collect(1:n))]
    else
        pred isa AbstractVector{Bool} && length(pred) == n || return nothing
        filter(arm -> !isempty(last(arm)),
            [(:then, findall(pred)), (:else, findall(!, pred))])
    end

    # An arm computed only from shared values (a constant fallback such as
    # `-Inf`) does not vary per lane; as its own plate its cell would be one
    # scalar rather than one value per lane. Such an arm is anchored to a
    # bound lane argument (`_LaneAnchored`, which ignores it) so both the
    # native loop and the tensorized batch evaluate it per lane. Known lane
    # values: bound lane-length vectors and live arguments declared as arrays,
    # plus every cell value computed from one.
    lane_values = Set{Int}()
    anchor = nothing
    for (index, (outer, input)) in enumerate(zip(recipe.inputs, inner.have))
        index in atomic_inputs && continue
        cid = canon_id(inner.graph, input.id)
        if haskey(bound, cid)
            bound[cid] isa AbstractArray && length(bound[cid]) == n || continue
            anchor === nothing && (anchor = input)
        else
            valtype(outer) <: AbstractArray || continue
        end
        push!(lane_values, cid)
    end
    for r in inner.recipes
        any(v -> canon_id(inner.graph, v.id) in lane_values, r.inputs) &&
            push!(lane_values, canon_id(inner.graph, only(r.outputs).id))
    end

    token = kernel_sourceop_token(branch_recipe.op)
    form = kernel_sourceop_form(branch_recipe.op)
    pointwise = only(recipe.outputs)
    single = length(arms) == 1
    lanes_value = Dict{Symbol,Value}()
    expansion = Recipe[]
    arm_outputs = Value[]
    for (side, lanes) in arms
        # The arm's scalar body: the branch recipe runs only this arm (a
        # nested branch arm stays a `_KernelBranch` and partitions again).
        positions = typeof(native).parameters[side === :then ? 2 : 3]
        arm_f = getfield(native, side === :then ? :then_arm : :else_arm)
        arm_tf = getfield(tensor, side === :then ? :then_arm : :else_arm)
        arm_inputs = Value[branch_recipe.inputs[i] for i in positions]
        if !any(v -> canon_id(inner.graph, v.id) in lane_values, arm_inputs)
            arm_f, arm_tf = _LaneAnchored(arm_f), _LaneAnchored(arm_tf)
            pushfirst!(arm_inputs, anchor)
        end
        arm_op = _KernelSourceOp(Val(Symbol(token, :_, side)), Val(form),
                                 arm_f, arm_tf)
        replaced = Recipe(branch_recipe.id, Tuple(arm_inputs),
            branch_recipe.outputs, arm_op, branch_recipe.cost, nothing,
            branch_recipe.effectful, branch_recipe.source)
        arm_recipes = Recipe[r === branch_recipe ? replaced : r
                             for r in inner.recipes]
        arm_plan = _partial_subplan(inner, inner.have, inner.want, arm_recipes)
        kernel = _prepare(arm_plan,
            _lower_with_ops(arm_plan; inline_embedded = false)...)
        readable = Dict(r.id => source for (r, source) in
                        zip(inner.recipes, op.kernel.lowered_recipes))
        kernel = _prepared_kernel(kernel.f, kernel.ops, kernel.inputs,
            kernel.outputs, kernel.plan, kernel.ast,
            Tuple(readable[r.id] for r in arm_recipes))
        arm_plate = _AuthoredPlateOp{typeof(kernel),atomic_inputs}(
            kernel, op.axis_checks)

        # The arm's plate arguments: shared/scalar arguments pass unchanged,
        # bound lane vectors are gathered now, live ones at run time.
        arguments = Value[]
        for (index, outer) in enumerate(recipe.inputs)
            cid = canon_id(g, outer.id)
            if single || index in atomic_inputs
                push!(arguments, outer)
            elseif haskey(known, cid)
                data = known[cid]
                if data isa AbstractArray && length(data) == n
                    gathered = data[lanes]
                    v = value!(g, Symbol(:lane_, side, :_, outer.name),
                               typeof(gathered))
                    push!(expansion, Recipe(fresh(), (), (v,),
                        _BoundConstant(gathered), 0.0, nothing, false))
                    known[canon_id(g, v.id)] = gathered
                    push!(arguments, v)
                else
                    push!(arguments, outer)
                end
            else
                idx = get!(lanes_value, side) do
                    v = value!(g, Symbol(:plate_lanes_, side), Vector{Int})
                    push!(expansion, Recipe(fresh(), (), (v,),
                        _BoundConstant(lanes), 0.0, nothing, false))
                    known[canon_id(g, v.id)] = lanes
                    v
                end
                v = value!(g, Symbol(:lane_, side, :_, outer.name), valtype(outer))
                push!(expansion, Recipe(fresh(), (outer, idx), (v,),
                    _LaneGather(n), 0.0, nothing, false))
                push!(arguments, v)
            end
        end
        output = single ? pointwise :
            value!(g, Symbol(pointwise.name, :_, side), valtype(pointwise))
        push!(expansion, Recipe(fresh(), Tuple(arguments), (output,), arm_plate,
            recipe.cost, nothing, false, recipe.source))
        push!(arm_outputs, output)
    end
    single && return (; expansion, rewrite = nothing)

    # Combine the arms. A sole `sum(pointwise)` consumer becomes a sum of
    # per-arm sums (each fuses into its arm's loop); otherwise the pointwise
    # vector is reassembled in lane order.
    pid = canon_id(g, pointwise.id)
    consumers = [r for r in pending
                 if any(inp -> canon_id(g, inp.id) == pid, r.inputs)]
    total = _partition_sum_consumer(consumers, pointwise)
    wanted = any(w -> canon_id(g, w.id) == pid, want)
    if total !== nothing && length(consumers) == 1 && !wanted
        totals = Value[]
        for output in arm_outputs
            t = value!(g, Symbol(output.name, :_total), valtype(only(total.outputs)))
            push!(expansion, Recipe(fresh(), (output,), (t,), sum, 0.0, nothing,
                false, Expr(:call, :sum, output.name)))
            push!(totals, t)
        end
        combined = Recipe(total.id, Tuple(totals), total.outputs, +, 0.0,
            nothing, false, Expr(:call, :+, (t.name for t in totals)...))
        return (; expansion, rewrite = total.id => combined)
    end
    order = invperm(reduce(vcat, (last(arm) for arm in arms)))
    push!(expansion, Recipe(fresh(), Tuple(arm_outputs), (pointwise,),
        _LaneAssemble(order), 0.0, nothing, false))
    (; expansion, rewrite = nothing)
end

function _partition_sum_consumer(consumers, pointwise)
    length(consumers) == 1 || return nothing
    r = only(consumers)
    r.effectful && return nothing
    length(r.inputs) == 1 && length(r.outputs) == 1 || return nothing
    source = r.source
    source isa Expr && source.head === :call && length(source.args) == 2 ||
        return nothing
    (source.args[1] === :sum || source.args[1] === GlobalRef(Base, :sum)) ||
        return nothing
    source.args[2] === pointwise.name || return nothing
    (r.op === sum || r.op isa _KernelSourceOp) || return nothing
    r
end

function _partial_inner_plates(p::Plan, known)
    any(r -> r.op isa _AuthoredPlateOp, p.recipes) || return p
    # A shallow structural copy preserves public Value identities and recipe
    # operations. Only its value registry is extended; no shared graph changes.
    g = p.graph
    copied = Graph(copy(g.values), copy(g.recipes), copy(g.producers),
                   copy(g.aliases), g.version)
    known = Dict{Int,Any}(known)
    recipes = Recipe[]
    next_id = Ref(minimum((r.id for r in p.recipes); init = 0) - 1)
    fresh() = (id = next_id[]; next_id[] -= 1; id)
    changed = false
    # A worklist: an arm plate emitted by the branch partition is examined
    # again (a nested arm or a further branch in its cell), and a rewritten
    # `sum` consumer replaces the original later in the queue.
    pending = copy(p.recipes)
    while !isempty(pending)
        r = popfirst!(pending)
        specialized = _partial_plate_recipe(copied, r, known, r.op)
        if specialized !== nothing
            changed = true
            for (value, data) in zip(specialized.cache_values, specialized.cache_data)
                push!(recipes, Recipe(fresh(), (), (value,), _BoundConstant(data),
                                      0.0, nothing, false))
                known[canon_id(copied, value.id)] = data
            end
            r = specialized.recipe
        end
        partitioned = _partition_plate_recipe(copied, r, known, r.op, pending,
                                              p.want, fresh)
        if partitioned === nothing
            push!(recipes, r)
            continue
        end
        changed = true
        if partitioned.rewrite !== nothing
            id, combined = partitioned.rewrite
            pending[findfirst(q -> q.id == id, pending)] = combined
        end
        prepend!(pending, partitioned.expansion)
    end
    changed || return p
    template = Plan(copied, p.have, p.want, p.recipes, p.producer,
                    p.cost, p.candidates)
    _partial_subplan(template, p.have, p.want, recipes)
end

"""
    partial_evaluation(p::Plan, bound, values) -> Plan

The general partial-evaluation pre-pass. `bound` names the HAVE ports (as
`Value`s) whose runtime `values` are fixed for this binding; they must form a
subset of `p.have`. The data-only prefix — every recipe whose transitive
inputs reach only bound ports — is prepared and executed exactly once, inside
this call. The result is an ordinary residual `Plan` whose HAVE boundary is
the remaining ports, with each hoisted value re-entering as a zero-input
[`_BoundConstant`](@ref) recipe, so every preparation surface consumes it
unchanged and per-call work contains no data-only recomputation.

For a residual authored `plate(...) do` with a transparent scalar plan, named
bound-only inner recipes are also evaluated at preparation. Each recipe uses
only its bound inputs' broadcast coordinates, so a prefix with singleton axes
is not expanded to a larger live domain. Boolean, standard 8–64-bit integer,
and Float16/32/64 results needed by the residual, and dense `Array`s of those
element types (a per-cell index list, for example), enter as additional
ordinary plate inputs; per-cell arrays are cached as an array of arrays and,
like every bound or hoisted value, are shared read-only by later calls.
Intermediate prefix values need only a concrete type. The original arguments
remain available for shape and marker validation: their storage is retained
even if the scalar residual no longer reads their elements. This trades setup
and cache storage for repeated computation; inexpensive recipes need not run
faster. Rebinding rebuilds these caches.

The inner pass conservatively retains the original plate for known empty
domains or live inputs that could introduce an empty broadcast dimension.
A live array is eligible only when its declared rank is known and each of its
dimensions is fixed by a bound axis of length greater than one; live numeric
scalars and explicit atomic inputs do not contribute dimensions.
Other exclusions are non-array/non-numeric batched bound inputs, non-concrete
prefix results and overlapping inner producers. A nested plate, scan or
embedded prepared kernel is never evaluated here: it stays in the residual
cell, as does the producer of any data-only value that is neither such a
number nor a dense `Array` of them (a tuple, struct, view or range); the
cell's other eligible values are still cached.
Ordinary recipe purity remains required. Like top-level partial evaluation,
eligible computations execute eagerly at preparation over their bound domain.
Mixed-input recipes remain indivisible: inline data-only subexpressions inside
one such recipe are not extracted or algebraically simplified.

The pass only transforms the plan: it adds no runtime wrapper and never
mutates the original graph. New cached value identities use a private graph
copy. Rebinding new data means running the pass
again from the same original plan.
"""
function partial_evaluation(p::Plan, bound, values)
    bound_values = collect(Value, _astuple(bound))
    value_tuple = Tuple(_astuple(values))
    length(bound_values) == length(value_tuple) || throw(ArgumentError(
        "partial evaluation received $(length(bound_values)) bound ports " *
        "but $(length(value_tuple)) bound values"))
    boundary = _partial_boundary(p, bound_values)
    hoisted = _partial_hoisted(p, boundary, value_tuple)
    last(_partial_residual(p, boundary, hoisted))
end

"""
    _PartialBoundary

The value-independent half of a partial evaluation: the bound HAVE ports (in
the caller's order), the remaining HAVE ports (in plan order), the data-only
`prefix` and the `residual` recipe partition, and the hoisted `constants`
boundary between them (in constant-slot order). Everything here is a function
of the plan and the bound PORT set alone, so a cached bound preparation
(`prepare!(cache, …; bound = …)`) computes it once per boundary.
"""
struct _PartialBoundary
    bound::Vector{Value}
    remaining::Vector{Value}
    prefix::Vector{Recipe}
    residual::Vector{Recipe}
    constants::Vector{Value}
end

function _partial_boundary(p::Plan, bound_values::Vector{Value})
    g = p.graph
    have_ids = Set(canon_id(g, v.id) for v in p.have)
    bound_ids = Set{Int}()
    for v in bound_values
        cid = canon_id(g, v.id)
        cid in have_ids || throw(ArgumentError(
            "bound port $(v.name) is not in the plan's HAVE boundary"))
        cid in bound_ids && throw(ArgumentError(
            "bound port $(v.name) is designated twice"))
        push!(bound_ids, cid)
    end
    remaining = Value[v for v in p.have if !(canon_id(g, v.id) in bound_ids)]
    prefix, residual, prefix_owned = _partial_split(p, bound_ids)
    constants = _partial_constants(p, prefix_owned, residual)
    _PartialBoundary(bound_values, remaining, prefix, residual, constants)
end

# The prefix plan: bound ports in, hoisted constants out.
_partial_prefix_plan(p::Plan, b::_PartialBoundary) =
    _partial_subplan(p, b.bound, b.constants, b.prefix)

# Prepare and run the prefix once for one binding's values; `()` when nothing
# the residual reads is bind-time.
_partial_hoisted(p::Plan, b::_PartialBoundary, values::Tuple) =
    isempty(b.constants) ? () :
    _partial_prefix_values(b.constants, prepare(_partial_prefix_plan(p, b))(values...))

# The zero-input constant recipes carrying one binding's hoisted values into
# the residual plan, under negative ids that cannot collide with graph recipes.
function _partial_constant_recipes(constants::Vector{Value}, @nospecialize(hoisted))
    # A loop, not a comprehension: its closure would be typed by `hoisted` and
    # compiled again for every binding's value types.
    recipes = Vector{Recipe}(undef, length(constants))
    for (index, value) in enumerate(constants)
        recipes[index] = Recipe(-index, (), (value,), _BoundConstant(hoisted[index]),
                                0.0, nothing, false)
    end
    recipes
end

# The residual plan for one binding: constant slots first, then the residual
# recipes.
function _partial_residual_plan(p::Plan, b::_PartialBoundary, @nospecialize(hoisted))
    recipes = isempty(b.constants) ? b.residual :
        vcat(_partial_constant_recipes(b.constants, hoisted), b.residual)
    _partial_subplan(p, b.remaining, p.want, recipes)
end

# That plan and its inner-plate specialization over the hoisted values. The
# second is the plan `prepare` compiles; it is the first object exactly when
# the inner-plate pass had nothing to do.
function _partial_residual(p::Plan, b::_PartialBoundary, @nospecialize(hoisted))
    g = p.graph
    known = Dict{Int,Any}()
    for (value, data) in zip(b.constants, hoisted)
        known[canon_id(g, value.id)] = data
    end
    residual = _partial_residual_plan(p, b, hoisted)
    residual, _partial_inner_plates(residual, known)
end

# ---------------------------------------------------------------------------
# Cached bound preparation (`prepare!(cache, …; bound = …)`).
#
# `prepare(…; bound = …)` repeats, per binding, work that does not depend on
# the bound VALUES at all: exact planning, the prefix plan's lowering and
# compilation, the residual plan's lowering(s) and compilation(s), and the
# embedded-marker analysis. For a graph prepared over and over with fresh data
# — a request-time `bound=` per user interaction — that fixed cost dominates
# once the compiled bodies themselves are content-cached (snag
# prepare-with-bou-c237dc00: ~1.3 ms of the ~2.6 ms `prepare` of the ShinyRK
# simulation graph on strato2, against ~1.25 ms of genuine bound-data
# mathematics in its prefix). A `_BoundEntry` keeps the planned boundary and
# the compiled prefix kernel per (graph, boundary, bound-port set, passes),
# and a `_BoundTemplate` keeps the compiled residual, so rebinding runs the
# prefix on the new values and rebuilds the residual `PreparedKernel` around
# the same compiled callable with a fresh constant table.
#
# The residual body is value-independent exactly when the inner-plate pass
# (`_partial_inner_plates`) leaves the residual plan alone: its plate
# specializations and data-bound branch partitions bake bound values in. That
# pass is the only authority on whether it changes a plan, so a residual
# containing authored plates runs it on every binding: a binding it rewrites
# keeps the ordinary per-binding specialization on the cached plan and prefix,
# and every binding it leaves alone shares the one template, whatever its
# values' types or array extents. The pass declines cheaply, before any
# evaluation, when a plate has nothing bound to specialize. A residual without
# authored plates skips the pass and shares the template outright. Nothing is
# kept per binding value, type or extent (snag inline-plate-reb-7387f072:
# keying templates on bound array sizes re-lowered the ShinyRK simulation
# graph at every new schedule length and kept one template per length).
#
# Nothing here is on a kernel's call path, and the kernel types involved are
# as varied as the graphs prepared, so an entry and a template are untyped
# containers and every function over them takes its arguments unspecialized:
# each is compiled once, for every graph and every bound type, instead of once
# per prefix and residual kernel type at the first rebinding in a process
# (snag plain-prepare-wi-4cd01ccf: 290 ms for the ShinyRK simulation graph).
# A template also keeps no binding's values: only the compiled callable and
# the value-independent tails of the operation table and readable recipes.

struct _BoundTemplate
    f::Any           # the compiled residual callable, shared by every binding
    ops::Tuple       # the operation table after the constant slots
    inputs::Tuple
    outputs::Tuple
    ast::Expr
    lowered::Tuple   # the readable recipes after the constant slots
end

# The template slot before the first binding the inner-plate pass left alone.
struct _UnbuiltTemplate end

mutable struct _BoundEntry
    const plan::Plan
    const boundary::_PartialBoundary
    const prefix::Any          # prepared prefix kernel, or `nothing` without constants
    const plates::Bool         # residual authored plates: the inner-plate pass runs per binding
    # A `_BoundTemplate`; the kernel itself when it has no constant slots;
    # `nothing` (a residual whose leading slots are not exactly its
    # constants: the per-binding path); or `_UnbuiltTemplate()`.
    template::Any
end

function _bound_entry(p::Plan, @nospecialize(ports))
    boundary = _partial_boundary(p, collect(Value, _astuple(ports)))
    prefix = isempty(boundary.constants) ? nothing :
             prepare(_partial_prefix_plan(p, boundary))
    plates = any(r -> r.op isa _AuthoredPlateOp, boundary.residual)
    _BoundEntry(p, boundary, prefix, plates, _UnbuiltTemplate())
end

_bound_hoisted(entry::_BoundEntry, @nospecialize(data::Tuple)) =
    entry.prefix === nothing ? () :
    _partial_prefix_values(entry.boundary.constants, entry.prefix(data...))

# One binding's residual plan, and the plan the inner-plate pass specialized
# from it (the same object when the pass left it alone, or did not run).
function _bound_residual(entry::_BoundEntry, @nospecialize(hoisted))
    entry.plates || return nothing, nothing
    _partial_residual(entry.plan, entry.boundary, hoisted)
end

# The residual plan `_bound_residual` built, or, where the inner-plate pass did
# not run, the plain one. Its callers take `residual` unspecialized, and this
# dispatch keeps the plan a concrete `Plan` after the call: reassigning the
# `Any` argument instead made every later call in `_bound_rebind` a dynamic
# dispatch (snag first-rebinding-64d82bfc).
_bound_residual_plan(::_BoundEntry, residual::Plan, @nospecialize(hoisted)) = residual
_bound_residual_plan(entry::_BoundEntry, ::Nothing, @nospecialize(hoisted)) =
    _partial_residual_plan(entry.plan, entry.boundary, hoisted)

# The first value-independent binding: the ordinary preparation, capturing
# the template when the leading operation-table slots are exactly the
# constants. Returns the template (or `nothing`) and this binding's kernel.
function _bound_build(entry::_BoundEntry, @nospecialize(residual),
                      @nospecialize(hoisted), @nospecialize(passes))
    kernel = prepare(_bound_residual_plan(entry, residual, hoisted); passes = passes)
    n = length(entry.boundary.constants)
    # Rebinding swaps the leading operation-table slots, so they must be
    # exactly this binding's constants in slot order.
    templated = all(i -> kernel.ops[i] isa _BoundConstant &&
                         kernel.ops[i].value === hoisted[i], 1:n)
    templated || return nothing, kernel
    n == 0 && return kernel, kernel
    _BoundTemplate(kernel.f, kernel.ops[(n + 1):end], kernel.inputs, kernel.outputs,
                   kernel.ast, kernel.lowered_recipes[(n + 1):end]), kernel
end

# A later binding of a residual whose leading slots are not its constants:
# compilation on the cached plan and prefix.
function _bound_rebind(entry::_BoundEntry, ::Nothing, @nospecialize(residual),
                       @nospecialize(hoisted), @nospecialize(passes))
    prepare(_bound_residual_plan(entry, residual, hoisted); passes = passes)
end

# A residual that reads no hoisted value is one kernel for every binding.
_bound_rebind(::_BoundEntry, @nospecialize(kernel::PreparedKernel), @nospecialize(residual),
              @nospecialize(hoisted), @nospecialize(passes)) = kernel

# A later value-independent binding: the same compiled callable over a fresh
# constant table.
function _bound_rebind(entry::_BoundEntry, template::_BoundTemplate,
                       @nospecialize(residual), @nospecialize(hoisted),
                       @nospecialize(passes))
    plan = _bound_residual_plan(entry, residual, hoisted)
    recipes = plan.recipes[1:length(entry.boundary.constants)]
    ops = (Any[recipe.op for recipe in recipes]..., template.ops...)
    _prepared_kernel(template.f, ops, template.inputs, template.outputs, plan,
                     template.ast, (recipes..., template.lowered...))
end

# Normalize the public `bound` kwarg — one `Value => data` pair or an
# iterable of them — into parallel port/value tuples.
_partial_bound_pairs(bound::Pair{<:Value}) = ((first(bound),), (last(bound),))
function _partial_bound_pairs(bound)
    ports = Value[]
    data = Any[]
    for entry in bound
        entry isa Pair{<:Value} || throw(ArgumentError(
            "bound entries must be `Value => data` pairs, got $(repr(entry))"))
        push!(ports, first(entry))
        push!(data, last(entry))
    end
    Tuple(ports), Tuple(data)
end

"""
    _partial_apply(p::Plan, bound) -> Plan

Apply the partial-evaluation pre-pass when `bound` is non-empty; otherwise
return `p` unchanged (the no-flag path stays byte-identical). `bound` is one
`Value => data` pair or an iterable of them.
"""
function _partial_apply(p::Plan, bound)
    bound === () && return p
    ports, data = _partial_bound_pairs(bound)
    isempty(ports) && return p
    partial_evaluation(p, ports, data)
end
