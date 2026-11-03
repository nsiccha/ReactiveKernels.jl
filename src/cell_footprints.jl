# Structural reads through views. These programs manipulate integer index
# maps, never proxy numerical values or infer dependencies from their values.
# An opaque consumer reads its entire input view; an ordinary gather reads
# the mapped indices. Shape-only checked aliases may preserve that view.

# The argument a call returns unchanged after validating it (a shape check),
# or 0: reads of its result are reads of that argument. A package extends it
# for its own checking helpers (ReactiveKernelsPPL's joint cell latents).
_footprint_checked_alias(f) = 0

# A recipe that indexes one of its inputs without reading other entries:
# `kind` is `:view` (`view(a, I...)`), `:read` (`a[I...]`) or `:alias` (a call
# that returns its argument, `_footprint_checked_alias`), `slot` the indexed
# input. `typed`: whether the lowering may hold the output in a typed local
# (`_declare_typed_output!`). Its conversion copies a view whose type is not
# the declared one (`x::Vector{Float64} = view(...)`), so a typed view or
# alias keeps its parent's identity only when inference proves the operation
# returns the declared type. A read's conversion reads only the read entries.
function _footprint_access(r::Recipe; typed::Bool = true)
    length(r.outputs) == 1 && !r.effectful || return nothing
    analyzed = _cell_native(r)
    if analyzed === nothing
        r.op in (Base.view, Base.getindex) || return nothing
        params = [gensym(:index_arg) for _ in r.inputs]
        native = Expr(:call, QuoteNode(r.op), params...)
        f = r.op
    else
        native, params, mod = analyzed
        if Meta.isexpr(native, :ref)
            f = Base.getindex
            native = Expr(:call, GlobalRef(Base, :getindex), native.args...)
        elseif Meta.isexpr(native, :call) &&
                !_native_callee_shadowed(native.args[1], params)
            f = _cell_binding(native.args[1], mod)
        else
            return nothing
        end
    end
    alias = _footprint_checked_alias(f)
    kind = f === Base.view ? :view : f === Base.getindex ? :read :
           alias > 0 ? :alias : nothing
    kind === nothing && return nothing
    declared = valtype(only(r.outputs))
    if typed && kind !== :read && declared !== Any
        inferred = only(_recipe_inferred_output_types(r, Any[valtype(v) for v in r.inputs]))
        inferred <: declared || return nothing
    end
    arg = kind === :alias ? alias + 1 : 2
    length(native.args) >= arg || return nothing
    slot = findfirst(==(native.args[arg]), params)
    slot === nothing && return nothing
    (; kind, slot, native, params, mod = analyzed === nothing ? nothing : last(analyzed))
end

# Calls that read only an array's shape. An index map (`LinearIndices`, or a
# view of one) has its array's axes, so an index expression may read the
# viewed array through them: `view(u, 1:length(u))`.
const _FOOTPRINT_SHAPE_CALLS = (Base.length, Base.size, Base.axes, Base.firstindex,
    Base.lastindex, Base.ndims)

# Whether the index arguments of an access read its source only through shape
# calls, so evaluating them with the source's index map gives the same indices.
function _footprint_indices_free(access)
    source = Set([access.params[access.slot]])
    all(ex -> _footprint_index_free(ex, source, access.params, access.mod),
        access.native.args[3:end])
end
function _footprint_index_free(ex, source, params, mod)
    ex isa Symbol && return !(ex in source)
    ex isa Expr || return true
    ex.head in _CELL_PLAIN_HEADS || return false
    if ex.head === :call && !_native_callee_shadowed(ex.args[1], params) &&
            any(f -> f === _cell_binding(ex.args[1], mod), _FOOTPRINT_SHAPE_CALLS)
        return all(a -> a isa Symbol || _footprint_index_free(a, source, params, mod),
                   ex.args[2:end])
    end
    all(a -> _footprint_index_free(a, source, params, mod), ex.args)
end

# The names a generated body binds at its top level, and its HAVE arguments.
function _footprint_bound_names(body::Expr, p::Plan, names)
    out = Set{Symbol}(names[canon_id(p.graph, v.id)] for v in p.have)
    for st in body.args
        Meta.isexpr(st, :(=)) && st.args[1] isa Symbol && push!(out, st.args[1])
    end
    out
end

_footprint_replace(ex::Symbol, mapping) = get(mapping, ex, ex)
_footprint_replace(ex::Expr, mapping) =
    Expr(ex.head, (_footprint_replace(a, mapping) for a in ex.args)...)
_footprint_replace(ex, mapping) = ex

function _footprint_join(reads...)
    out = Int[]
    for read in reads
        read isa Integer ? push!(out, read) : append!(out, vec(read))
    end
    unique!(sort!(out))
end

# A selected plate's footprint for one shared array. Independent recipes
# supply the indices; dependent numerical computations are absent. Keeping
# these as ordinary graph recipes retains data-size-independent loop code.
function _footprint_capture_program(inner::Plan, root::Int)
    old = inner.graph
    cid(v) = canon_id(old, v.id)
    g = Graph()
    mapped = Dict{ValueId,Value}()
    fresh() = value!(g, gensym(:read), Any)
    have = Value[]
    for v in inner.have
        w = fresh()
        mapped[cid(v)] = w
        push!(have, w)
    end
    dependent = Set([cid(inner.have[root])])
    views = Dict{ValueId,Value}()
    indices = fresh()
    add!(g; inputs=(have[root],), outputs=(indices,), op=LinearIndices)
    views[cid(inner.have[root])] = indices
    reads = Value[]
    seen_view = false
    for r in inner.recipes
        slots = findall(v -> cid(v) in dependent, r.inputs)
        if isempty(slots)
            outs = Tuple(value!(g, gensym(:data), valtype(v)) for v in r.outputs)
            add!(g; inputs=Tuple(mapped[cid(v)] for v in r.inputs), outputs=outs,
                 op=r.op, cost=r.cost, effectful=r.effectful, source=r.source)
            for (v, w) in zip(r.outputs, outs)
                mapped[cid(v)] = w
            end
            continue
        end
        union!(dependent, (cid(v) for v in r.outputs))
        access = _footprint_access(r)
        if access !== nothing && slots == [access.slot] &&
                haskey(views, cid(r.inputs[access.slot]))
            source = views[cid(r.inputs[access.slot])]
            if access.kind === :alias
                views[cid(only(r.outputs))] = source
                continue
            end
            # No dependent value may also occur as an index (a[a[1]], etc.).
            if _footprint_indices_free(access)
                args = Tuple(j == access.slot ? source : mapped[cid(v)]
                             for (j, v) in enumerate(r.inputs))
                w = fresh()
                f = compile(Expr(:->, Expr(:tuple, access.params...), access.native))
                add!(g; inputs=args, outputs=(w,), op=f)
                if access.kind === :view
                    views[cid(only(r.outputs))] = w
                    seen_view = true
                else
                    push!(reads, w)
                end
                continue
            end
        elseif _cell_identity_form(r) !== nothing && length(slots) == 1 &&
                haskey(views, cid(only(r.inputs)))
            views[cid(only(r.outputs))] = views[cid(only(r.inputs))]
            continue
        end
        for j in slots
            v = r.inputs[j]
            haskey(views, cid(v)) || continue
            cid(v) == cid(inner.have[root]) && return nothing
            push!(reads, views[cid(v)])
        end
    end
    seen_view || return nothing
    for v in inner.want
        haskey(views, cid(v)) || continue
        cid(v) == cid(inner.have[root]) && return nothing
        push!(reads, views[cid(v)])
    end
    result = fresh()
    add!(g; inputs=Tuple(reads), outputs=(result,), op=_footprint_join)
    p = plan(g; have, want=result)
    _prepare(p, _lower_with_ops(p; inline_embedded=false)...)
end

# Transfer reads of a view to its parent without materializing the parent.
# LinearIndices supplies storage positions while retaining the view's axes,
# including multidimensional and logical indexing semantics.
function _footprint_view_positions(indexmap, view, records)
    positions = _footprint_positions(view, records)
    positions === :whole ? collect(vec(indexmap)) : vec(indexmap)[positions]
end
