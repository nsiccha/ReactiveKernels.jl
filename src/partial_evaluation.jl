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
    _partial_split(p::Plan, bound::Set{Int}) -> (prefix, residual, prefix_owned)

Partition `p.recipes` (kept in execution order) into the data-only `prefix` —
recipes whose every input is a bound HAVE port or a value owned by an earlier
prefix recipe — and the `residual` rest. Ownership follows the lowering's
first-producer-wins rule: a value emitted collaterally by a later recipe is a
discarded duplicate, so it neither makes that recipe's consumers hoistable nor
un-hoists the authoritative producer. `prefix_owned` is the canonical id set
available at bind time: the bound ports plus every prefix-owned output.

Zero-input recipes are data-only by definition and hoist under any bound set,
including an empty one.
"""
function _partial_split(p::Plan, bound::Set{Int})
    g = p.graph
    prefix_owned = copy(bound)
    assigned = Set(canon_id(g, v.id) for v in p.have)
    prefix = Recipe[]
    residual = Recipe[]
    for r in p.recipes
        if all(inp -> canon_id(g, inp.id) in prefix_owned, r.inputs)
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

# Keep the first cache boundary on ordinary primitive numeric array elements.
# An isbits tuple/struct is safe to hold natively, but an array of those values
# would introduce a new operand representation at the tensor backend boundary.
const _PartialPlateScalar = Union{Bool,Int8,Int16,Int32,Int64,
    UInt8,UInt16,UInt32,UInt64,Float16,Float32,Float64}

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
function _partial_plate_recipe(g, recipe, known, op::_AuthoredPlateOp{K,A}) where {K,A}
    inner = op.kernel.plan
    length(inner.have) == length(recipe.inputs) || return nothing
    length(inner.want) == 1 || return nothing
    length(op.kernel.ops) == length(inner.recipes) || return nothing

    # Native scalar plate lowering expects one output per recipe. Do not turn
    # overlapping producer bindings, effects, or nested batching into a new
    # specialization boundary; the ordinary lowering remains authoritative.
    assigned = Set(canon_id(inner.graph, v.id) for v in inner.have)
    for r in inner.recipes
        r.effectful && return nothing
        length(r.outputs) == 1 || return nothing
        r.op isa Union{_AuthoredPlateOp,_AuthoredScanOp} && return nothing
        _embedded_kernel(r.op) === nothing || return nothing
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
        index in A || _partial_plate_input(data) || return nothing
        cid = canon_id(inner.graph, input.id)
        push!(bound, cid)
        arguments[cid] = index in A ? Ref(data) : data
        types[cid] = index in A || data isa Number ? typeof(data) : eltype(data)
    end
    prefix, residual, owned = _partial_split(inner, bound)
    isempty(prefix) && return nothing
    have_ids = Set(canon_id(inner.graph, v.id) for v in inner.have)
    frontier = filter(_partial_constants(inner, owned, residual)) do v
        !(canon_id(inner.graph, v.id) in have_ids)
    end
    isempty(frontier) && return nothing

    # A known empty domain has no data-dependent cells to evaluate. Keeping the
    # original plate also preserves its empty-result typing and error behavior.
    bound_axes = Base.Broadcast.combine_axes(values(arguments)...)
    any(isempty, bound_axes) && return nothing
    fixed_domain = all(enumerate(recipe.inputs)) do (index, input)
        index in A || haskey(known, canon_id(g, input.id)) ||
            _partial_plate_live_domain(valtype(input), bound_axes)
    end
    fixed_domain || return nothing
    # No possible batched axis: preserve the ordinary rejection.
    isempty(bound_axes) && return nothing

    # Keep only the selected prefix producers needed by the cut. Never re-plan:
    # doing so could select a different producer from the public graph.
    needed = Set(canon_id(inner.graph, v.id) for v in frontier)
    selected = Recipe[]
    for r in Iterators.reverse(prefix)
        canon_id(inner.graph, only(r.outputs).id) in needed || continue
        push!(selected, r)
        union!(needed, (canon_id(inner.graph, v.id) for v in r.inputs))
    end
    reverse!(selected)

    # Establish the complete eligible prefix before executing any of it. Only
    # concrete scalar isbits values cross this new cache boundary. In particular
    # no mutable per-cell object gains shared identity through specialization.
    for r in selected
        T = Base.promote_op(r.op,
            (types[canon_id(inner.graph, v.id)] for v in r.inputs)...)
        isconcretetype(T) && isbitstype(T) || return nothing
        types[canon_id(inner.graph, only(r.outputs).id)] = T
    end
    all(v -> types[canon_id(inner.graph, v.id)] <: _PartialPlateScalar,
        frontier) || return nothing
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
    atomic = Int[A...]
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
    kernel = PreparedKernel(kernel.f, kernel.ops, kernel.inputs, kernel.outputs,
        kernel.plan, kernel.ast, Tuple(readable[r.id] for r in residual))
    replacement = _AuthoredPlateOp{typeof(kernel),Tuple(atomic)}(
        kernel, op.axis_checks)
    updated = Recipe(recipe.id, (recipe.inputs..., cache_values...),
        recipe.outputs, replacement, recipe.cost, nothing,
        recipe.effectful, recipe.source)
    (; recipe = updated, cache_values, cache_data)
end

function _partial_inner_plates(p::Plan, known)
    any(r -> r.op isa _AuthoredPlateOp, p.recipes) || return p
    # A shallow structural copy preserves public Value identities and recipe
    # operations. Only its value registry is extended; no shared graph changes.
    g = p.graph
    copied = Graph(copy(g.values), copy(g.recipes), copy(g.producers),
                   copy(g.aliases), g.version)
    recipes = Recipe[]
    next_id = minimum((r.id for r in p.recipes); init = 0) - 1
    changed = false
    for r in p.recipes
        specialized = _partial_plate_recipe(copied, r, known, r.op)
        if specialized === nothing
            push!(recipes, r)
            continue
        end
        changed = true
        for (value, data) in zip(specialized.cache_values, specialized.cache_data)
            push!(recipes, Recipe(next_id, (), (value,), _BoundConstant(data),
                                  0.0, nothing, false))
            next_id -= 1
        end
        push!(recipes, specialized.recipe)
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
and Float16/32/64 results needed by the residual enter as additional ordinary
plate inputs. The original arguments
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
or mutable prefix results, nonnumeric cache frontiers, nested plate/scan
bodies, and overlapping inner producers.
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
    g = p.graph
    bound_values = collect(Value, _astuple(bound))
    value_tuple = Tuple(_astuple(values))
    length(bound_values) == length(value_tuple) || throw(ArgumentError(
        "partial evaluation received $(length(bound_values)) bound ports " *
        "but $(length(value_tuple)) bound values"))
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
    constant_values = _partial_constants(p, prefix_owned, residual)

    recipes = residual
    known = Dict{Int,Any}()
    if !isempty(constant_values)
        prefix_plan = _partial_subplan(p, bound_values, constant_values, prefix)
        hoisted = _partial_prefix_values(
            constant_values, prepare(prefix_plan)(value_tuple...))
        for (value, data) in zip(constant_values, hoisted)
            known[canon_id(g, value.id)] = data
        end
        recipes = vcat(
            [Recipe(-index, (), (value,), _BoundConstant(hoisted[index]),
                    0.0, nothing, false)
             for (index, value) in enumerate(constant_values)],
            residual)
    end
    _partial_inner_plates(_partial_subplan(p, remaining, p.want, recipes), known)
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
