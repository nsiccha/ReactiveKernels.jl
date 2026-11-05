# Structural reads through views. These programs manipulate integer index
# maps, never proxy numerical values or infer dependencies from their values.
# An opaque consumer reads its entire input view; an ordinary gather reads
# the mapped indices. Shape-only checked aliases may preserve that view.

# The argument a call returns unchanged after validating it (a shape check),
# or 0: reads of its result are reads of that argument. A package extends it
# for its own checking helpers (ReactiveKernelsPPL's joint cell latents).
_footprint_checked_alias(f) = 0

# The validator of a checked alias's row form, or `nothing` (no row form).
# Called as `validator(T, axes, others...)` with the type and full axes of the
# alias's argument and `f`'s other arguments in order, it makes the checks
# `f` makes and throws as `f` would. A cell then reads the argument at its rows
# (`_cell_checked_form`), and the check still covers the whole argument.
_checked_alias_validator(f) = nothing

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
        _footprint_append!(out, read)
    end
    unique!(sort!(out))
end
_footprint_append!(out, read) =
    read isa Integer ? push!(out, read) : append!(out, vec(read))

# Keep generator binders and guards in runtime loops over integer maps. No
# closure or data-length argument tuple is introduced in the footprint kernel.
function _footprint_loop(gather, clauses)
    isempty(clauses) && return gather
    out = gensym(:positions)
    body = Expr(:call, GlobalRef(@__MODULE__, :_footprint_append!), out, gather)
    for clause in Iterators.reverse(clauses)
        if Meta.isexpr(clause, :filter)
            body = Expr(:if, clause.args[1], body)
            clause = clause.args[2]
        end
        body = Expr(:for, clause, body)
    end
    Expr(:let, Expr(:(=), out, :(Int[])), Expr(:block, body, out))
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
    seen_access = false
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
                else
                    push!(reads, w)
                end
                seen_access = true
                continue
            end
        elseif _cell_identity_form(r) !== nothing && length(slots) == 1 &&
                haskey(views, cid(only(r.inputs)))
            views[cid(only(r.outputs))] = views[cid(only(r.inputs))]
            continue
        end
        # A compound recipe that reads index-mapped values only through
        # gathers (`w .* v[I]`) reads the gathered positions.
        mapped_slots = Int[j for j in slots if haskey(views, cid(r.inputs[j]))]
        gathers = isempty(mapped_slots) ? nothing :
            _footprint_inner_gathers(r, mapped_slots, slots)
        if gathers !== nothing && all(gathers) do (slot, refs, _)
                    all(j -> j == slot || haskey(mapped, cid(r.inputs[j])), refs)
                end
            for (slot, refs, ex) in gathers
                w = fresh()
                f = compile(Expr(:->, Expr(:tuple, (r_params(r)[j] for j in refs)...), ex))
                add!(g; inputs=Tuple(j == slot ? views[cid(r.inputs[j])] : mapped[cid(r.inputs[j])]
                                     for j in refs), outputs=(w,), op=f)
                push!(reads, w)
            end
            seen_access = true
            continue
        end
        for j in slots
            v = r.inputs[j]
            haskey(views, cid(v)) || continue
            cid(v) == cid(inner.have[root]) && return nothing
            push!(reads, views[cid(v)])
        end
    end
    seen_access || return nothing
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

r_params(r::Recipe) = _cell_native(r)[2]

# A direct scalar expression's index program can also request a producer's
# numerical rows. Unlike footprint-only view traversal, this needs the body
# to accept a sparse window: constructing an inner view may read more of that
# window than the final consumer does. Keep such bodies on the ordinary path.
function _cell_capture_index_program(inner::Plan, root::Int)
    length(inner.recipes) == 1 || return nothing
    r = only(inner.recipes)
    cid(v) = canon_id(inner.graph, v.id)
    slot = findfirst(v -> cid(v) == cid(inner.have[root]), r.inputs)
    slot === nothing && return nothing
    _footprint_inner_gathers(r, [slot], [slot]) === nothing && return nothing
    _footprint_capture_program(inner, root)
end

function _cell_index_dependencies(program)
    p = program.plan
    cid(v) = canon_id(p.graph, v.id)
    used = Set(cid(v) for r in p.recipes for v in r.inputs)
    Set(k for (k, v) in enumerate(p.have) if cid(v) in used)
end

# Evaluate the index program over exactly the plate lanes the value reads,
# both for selected top-level cells and for a producer plate's row window.
function _footprint_plate_reads!(body, runtime_ops, runtime_recipes,
                                 program, atomic, args, values)
    readplate = _AuthoredPlateOp{typeof(program),atomic}(program)
    reads = gensym(:cell_reads)
    _lower_authored_plate_native!(body, runtime_ops, runtime_recipes,
        readplate, args, values, reads, nothing; element_type = Vector{Int})
    reads
end

# The gathers `v[I...]` of index-mapped inputs (`mapped_slots`) inside a
# compound recipe, as `(slot, referenced input slots, expression)`, when the
# recipe reads those inputs in no other way and no index reads a dependent
# value (`dependent_slots`) except a mapped source's shape. `nothing`
# otherwise, or when the recipe cannot be analysed.
function _footprint_inner_gathers(r::Recipe, mapped_slots, dependent_slots)
    length(r.outputs) == 1 && !r.effectful || return nothing
    # Read the authored generator before native fold lowering introduces its
    # accumulator and loop. The numerical recipe itself is left unchanged.
    analyzed = _cell_source(r)
    analyzed === nothing && return nothing
    native, params, mod = analyzed
    mod isa Module || return nothing
    sources = Dict{Symbol,Int}(params[j] => j for j in mapped_slots)
    dependent = Set{Symbol}(params[j] for j in dependent_slots)
    found = Tuple{Int,Vector{Int},Any}[]
    function walk(ex, clauses = Any[], bound = Set{Symbol}(params))
        ex isa Symbol && return !haskey(sources, ex)
        ex isa Expr || return true
        if Meta.isexpr(ex, :call) && !isempty(ex.args) &&
                !_native_callee_shadowed(ex.args[1], bound) &&
                _cell_binding(ex.args[1], mod) === Base.sum
            parts = _kernel_generator_sum_parts(
                Expr(:call, GlobalRef(Base, :sum), ex.args[2:end]...))
            if parts !== nothing
                # Only eager sums with data-only domains and guards. An
                # arbitrary generator consumer may stop before a later read.
                parts.variable in bound && return false
                _cell_free_of(parts.iterator, dependent) || return false
                _cell_free_of(parts.condition, dependent) || return false
                walk(parts.init, clauses, bound) || return false
                clause = Expr(:(=), parts.variable, parts.iterator)
                parts.condition === nothing ||
                    (clause = Expr(:filter, parts.condition, clause))
                return walk(parts.term, [clauses; clause], union(bound, [parts.variable]))
            end
        end
        ex.head in _CELL_PLAIN_HEADS || return false
        # Do not hoist a possibly invalid read out of a lazy term branch.
        ex.head in (:if, :&&, :||) &&
            !_cell_free_of(ex, Set(keys(sources))) && return false
        source, indices = if Meta.isexpr(ex, :ref) && length(ex.args) >= 2
            ex.args[1], ex.args[2:end]
        elseif Meta.isexpr(ex, :call) && length(ex.args) >= 3 &&
                !_native_callee_shadowed(ex.args[1], bound) &&
                _cell_binding(ex.args[1], mod) === Base.getindex
            ex.args[2], ex.args[3:end]
        else
            nothing, nothing
        end
        if source isa Symbol && haskey(sources, source)
            others = setdiff(dependent, (source,))
            all(i -> _footprint_index_free(i, Set([source]), params, mod) &&
                     _cell_free_of(i, others), indices) || return false
            gather = Expr(:call, GlobalRef(Base, :getindex), source, indices...)
            names = Dict(q => q for q in bound)
            gather = _inline_qualify(gather, mod, names)
            qualified_clauses = Any[_inline_qualify(c, mod, names) for c in clauses]
            (gather === nothing || any(isnothing, qualified_clauses)) && return false
            gather = _footprint_loop(gather, qualified_clauses)
            refs = Int[j for (j, q) in enumerate(params) if _footprint_mentions(gather, q)]
            push!(found, (sources[source], refs, gather))
            return true
        end
        all(a -> walk(a, clauses, bound), ex.args)
    end
    walk(native) && !isempty(found) ? found : nothing
end
_footprint_mentions(ex::Symbol, q) = ex === q
_footprint_mentions(ex::Expr, q) = any(a -> _footprint_mentions(a, q), ex.args)
_footprint_mentions(ex, q) = false

# Transfer reads of a view to its parent without materializing the parent.
# LinearIndices supplies storage positions while retaining the view's axes,
# including multidimensional and logical indexing semantics.
function _footprint_view_positions(indexmap, view, records)
    positions = _footprint_positions(view, records)
    positions === :whole ? collect(vec(indexmap)) : vec(indexmap)[positions]
end
