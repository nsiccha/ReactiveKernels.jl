# A compact vector presented in the authored packed frame. Unlike the cell
# overlay there is no inactive base value: discovering an omitted dependency
# is an error, never a silently fixed parameter or a full-width AD shadow.
"""
    smc_window_coordinates(target_or_state)

Return an owned coordinate descriptor for a window prepared with
`coordinates=:auto`, or an initialized/completed state from that target.
`compact` contains authored coordinate identities and geometry in particle
order; `current_indices` maps them to the current model's packed frame.
`active_old` gives their old-carry indices, ordered by `active_slots` in the
compact vector. `fresh_slots` are the remaining compact positions.
`joint` describes the full recovery/carry order (current layout first, then
retained old coordinates absent from it); `joint_indices` locates the compact
coordinates in that full order. Indices are one based.
"""
function smc_window_coordinates end

"""
    smc_recover(target_or_state, compact_u; rng::Xoshiro)

Draw one full joint unconstrained vector for a selective window, retaining the
supplied active and fresh coordinates exactly. Inactive old coordinates are
drawn from `q_old(inactive | active)`, including its correlations. The returned
order is `smc_window_coordinates(target_or_state).joint`. The supplied RNG
advances; no production RNG is consumed implicitly and the compact point is read only.
Full-sized recovery storage is allocated only when requested.
"""
function smc_recover end

# Fresh priors use the shared authored-factor inventory. Observations use
# the existing partition lowering, with all rows in one prepared partition.
_smc_window_terms(built, binding, fresh, u0) = isempty(fresh) ?
    _smc_likelihood_terms(built, binding, u0) :
    _smc_fresh_window_terms(built, binding, fresh, u0)

function _smc_fresh_window_terms(built, binding, fresh, u0)
    built = _binding_build(built, binding)
    catalogue = density_factors(built, binding)
    isfresh = falses(built.layout.total)
    isfresh[fresh] .= true
    events, ownerless = DensityFactor[], DensityFactor[]
    for event in catalogue
        owned = factor_coordinates(catalogue, event)
        if isempty(owned)
            push!(ownerless, event)
            continue
        end
        any(i -> isfresh[i], owned) || continue
        all(i -> isfresh[i], owned) || throw(ArgumentError(
            "selective SMC cannot split partly fresh atomic factor $event"))
        push!(events, event)
    end
    if !isempty(ownerless)
        # Scalar index reads need the same native locality lowering as the
        # likelihood. The catalogue's conservative support can include a
        # whole array even when an extra density reads only retained z[1].
        selected = select_factors(catalogue, ownerless)
        derived = _smc_scalar_reads(_density_selection_spec(built, binding, selected;
            locality=:error, on_error=nothing, u0))
        any(i -> isfresh[i], _smc_read_support(derived, u0)) &&
            throw(ArgumentError("selective SMC cannot assign an ownerless " *
                "density whose structural read support includes fresh coordinates: $ownerless"))
    end
    selected = select_factors(catalogue, events)
    factors = _smc_scalar_reads(_density_selection_spec(built, binding, selected;
        locality=:error, on_error=nothing, u0))
    likelihood = _smc_likelihood_terms(built, binding, u0)
    # Share only the inputs. Each cut keeps its own internal values so a
    # whole-model producer cannot replace the selected producer of the same
    # generated name when the two graphs are composed.
    function boundary(terms)
        names = unique([terms.have..., terms.want])
        ReactiveKernels.KernelSpec(terms.spec.graph,
            Dict(name => terms.spec.ports[name] for name in names), names,
            collect(terms.have), [terms.want], nothing)
    end
    spec = merge(boundary(likelihood), boundary(factors))
    want = :_ppl_smc_window
    output = value!(spec.graph, want, Float64)
    spec.ports[want] = output
    push!(spec.port_order, want)
    add!(spec.graph, (spec[likelihood.want], spec[factors.want]) => output, +)
    (; spec, have=Tuple(unique([likelihood.have..., factors.have...])), want,
        bound=merge(likelihood.bound, factors.bound), binding)
end

function _smc_likelihood_terms(built, binding, u0)
    _check_built_program(built, binding)
    pointwise_kernel = _prepare_wants(built, binding, :pointwise; exemplars=(u0,))
    pointwise = Base.invokelatest(pointwise_kernel, u0)
    # Pointwise values supply shapes only. Read support comes from the
    # compiler below, never from numerical derivatives or density values.
    body = built.program.args[2].args
    statement = only(st for st in body if Meta.isexpr(st, :(=), 2) && st.args[1] === :pointwise)
    nodes = Dict(field.args[1] => field.args[2] for field in statement.args[2].args)
    spec = deepcopy(built.spec)
    names, terms = Symbol[], Symbol[]
    keys_by_name = Dict{Symbol,Symbol}()
    bindings = Pair{Symbol,Any}[pairs(_query_bound(binding))...]
    for (name, values) in pairs(pointwise)
        if values isa Real
            push!(terms, nodes[name])
        elseif !isempty(values)
            key = Symbol(:_ppl_smc_rows_, name)
            haskey(spec.ports, key) && throw(ArgumentError("SMC port name collision: $key"))
            spec.ports[key] = value!(spec.graph, key, Vector{Bool})
            push!(spec.port_order, key)
            push!(names, name)
            keys_by_name[name] = key
            push!(bindings, key => fill(true, length(values)))
        end
    end
    have = (_query_have(binding)..., (keys_by_name[name] for name in names)...)
    if !isempty(names)
        local_build = merge(built, (; spec))
        have = (have..., _PARTITION_PORT, _PLAN_PORT)
        partition_spec, partition_want = _partition_query_spec(local_build, binding,
            names, keys_by_name; locality=:error)
        planner = Base.invokelatest(_prepare_cached_partition, binding,
            partition_spec; have, want=partition_want, bound=(; bindings...))
        rowplan = Base.invokelatest(planner, u0, true, nothing)
        spec, partition_want = _partition_query_spec(local_build, binding,
            names, keys_by_name; locality=:error, plan_type=typeof(rowplan))
        push!(bindings, _PARTITION_PORT => true, _PLAN_PORT => rowplan)
        push!(terms, partition_want)
    end
    want = :_ppl_smc_likelihood
    output = value!(spec.graph, want, Float64)
    spec.ports[want] = output
    push!(spec.port_order, want)
    if isempty(terms)
        add!(spec.graph, () => output, () -> 0.0)
    else
        add!(spec.graph, Tuple(spec[t] for t in terms) => output, +)
    end
    terms = (; spec, have, want, bound=(; bindings...), binding)
    if any(value -> value isa Real, values(pointwise))
        # A scalar observation has no authored plate. Give its indexed
        # inputs the same local read semantics as observation cells, using
        # the existing factor-query scalar-read lowering. In particular,
        # z[2] must not count the whole packed view producing z as active.
        return _smc_scalar_reads(terms)
    end
    terms
end

function _smc_scalar_reads(terms)
    spec = terms.spec
    selected = ReactiveKernels.KernelSpec(spec.graph, copy(spec.ports),
        copy(spec.port_order), collect(terms.have), [terms.want], nothing)
    selected_plan = ReactiveKernels.plan(selected)
    outputs = Set(v.id for recipe in selected_plan.recipes for v in recipe.outputs)
    spec = ReactiveKernels._cell_factor_scalar_reads(selected, outputs;
        bound=terms.bound, locality=:error)
    (; terms..., spec)
end

function _smc_read_support(terms, u0)
    kernel = Base.invokelatest(_prepare_cached, terms.binding, terms.spec; have=terms.have,
        want=terms.want, bound=terms.bound, exemplars=(u0,))
    read = Base.invokelatest(partition_footprint, kernel, u0).unconstrained
    read === :whole ? collect(1:length(u0)) : Vector{Int}(read)
end

struct _SMCSelectedPoint{A<:AbstractVector{Float64}} <: AbstractVector{Float64}
    active::A
    slots::Vector{Int32}
end
Base.size(point::_SMCSelectedPoint) = size(point.slots)
Base.IndexStyle(::Type{<:_SMCSelectedPoint}) = IndexLinear()
@inline function Base.getindex(point::_SMCSelectedPoint, i::Int)
    slot = point.slots[i]
    iszero(slot) && throw(ArgumentError("SMC target reads an omitted coordinate"))
    point.active[slot]
end
@inline _smc_selected_point(active, slots) = _SMCSelectedPoint(active, slots)
ReactiveKernels._cell_lazy_view(::typeof(_smc_selected_point)) = true
ReactiveKernels._native_mul_operand(::_SMCSelectedPoint) = true
function ReactiveKernels._cell_lazy_rows(::typeof(_smc_selected_point), inputs, rows, bind, constant)
    constant(inputs[2]) || return nothing
    slots = bind(:smc_positions, Expr(:call, GlobalRef(@__MODULE__, :_overlay_slots), inputs[2], rows))
    Expr(:call, GlobalRef(ReactiveKernels, :_cell_source_rows), inputs[1], slots)
end

function _smc_selected_kernel(terms, indices, u0)
    spec = deepcopy(terms.spec)
    for (name, type) in ((:_ppl_smc_active, AbstractVector{Float64}),
                         (:_ppl_smc_slots, Vector{Int32}))
        haskey(spec.ports, name) && throw(ArgumentError("SMC port name collision: $name"))
        spec.ports[name] = value!(spec.graph, name, type)
        push!(spec.port_order, name)
    end
    add!(spec.graph, (spec[:_ppl_smc_active], spec[:_ppl_smc_slots]) =>
        spec[:unconstrained], _smc_selected_point)
    have = (:_ppl_smc_active, :_ppl_smc_slots,
        (name for name in terms.have if name !== :unconstrained)...)
    slots = _compact_cell_slots(length(u0), indices)
    bound = merge(terms.bound, (; _ppl_smc_slots=slots))
    kernel = Base.invokelatest(_prepare_cached, terms.binding, spec; have, want=terms.want, bound,
        exemplars=(Vector{Float64}(u0[indices]),))
    ReactiveKernels._kernel_latest_prepared(kernel)
end

# Reuse the public authored coordinate inventory, including Cartesian/cell
# indices, scoped names and coefficient levels. Offsets are never identities.
function _smc_bound_identity(layout)
    sampled = Dict(_smc_output_name(e) => _smc_output_path(layout, e) for e in layout.entries)
    visiting = Set{Symbol}()
    dynamic_cache = Dict{Symbol,Bool}()
    function dynamic(name)
        haskey(sampled, name) && return true
        haskey(dynamic_cache, name) && return dynamic_cache[name]
        haskey(layout.bound_values, name) && return false
        haskey(layout.bound_exprs, name) || return false
        name in visiting && throw(ArgumentError("cyclic SMC transform dependency $name"))
        push!(visiting, name)
        result = any(dynamic, _value_symbols(layout.bound_exprs[name]))
        delete!(visiting, name)
        dynamic_cache[name] = result
        result
    end
    lookup = _layout_lookup(layout, Dict{Symbol,Any}())
    frozen = Dict{Symbol,Any}()
    function freeze(ex)
        ex isa Real && return ex
        # One bound per element, already this binding's numbers.
        ex isa AbstractArray{<:Real} && return copy(ex)
        if !any(dynamic, _value_symbols(ex))
            # Fold data-only subexpressions, so a bound depending on a
            # scalar summary retains that summary instead of old data.
            return deepcopy(_eval_value_expr(ex, lookup, :smc_transform_bound))
        elseif ex isa Symbol
            haskey(sampled, ex) && return (; parameter=sampled[ex])
            return get!(frozen, ex) do
                freeze(layout.bound_exprs[ex])
            end
        elseif ex isa Expr
            ex.head === :call && return Expr(:call, ex.args[1], map(freeze, ex.args[2:end])...)
            return Expr(ex.head, map(freeze, ex.args)...)
        end
        deepcopy(ex)
    end
    freeze
end

# Sampling expressions are identity metadata, never executable checkpoint
# state. Use the same namespace normalization as the built program: a fresh
# lowering module must not survive inside a serialized coordinate frame.
_smc_sampling_id(::Nothing) = nothing
_smc_sampling_id(s) = (; rhs=_program_identity(s.rhs), s.broadcast, s.constructor_broadcast)

function _smc_coordinate_records(layout::LayoutTable)
    records = NamedTuple[]
    labels = coordinate_names(layout)
    bound_identity = _smc_bound_identity(layout)
    for entry in layout.entries
        path = _smc_output_path(layout, entry)
        sampling = _smc_sampling_id(entry.sampling)
        coupled = entry.kind !== :external_plate &&
            (!(entry.transform in (:identity, :exp, :logistic, :interval, :floored, :upper, :bounded)) ||
             entry.sampling !== nothing || haskey(layout.cell_geometries, entry.name))
        event_shape = coupled ? (entry.size, Tuple(entry.dims)) : nothing
        if entry.kind === :array && _is_slice_transform(entry.transform)
            axis = _slice_transform_orientation(entry.transform) isa _SliceRows ? 2 : 1
            event_shape = (_slice_packed_dims(entry.transform, entry.dims)[axis],
                (entry.dims[axis],))
        elseif entry.kind === :array && _is_lkj_stack(entry.transform)
            width = entry.dims[1]
            event_shape = (width * (width-1) ÷ 2, (width, width))
        end
        event_shapes = fill(event_shape, entry.size)
        if haskey(layout.cell_geometries, entry.name)
            # A joint event is one cell, not the complete array of cells.
            # Store only shape/identity metadata, never callable geometry.
            cell = layout.cell_geometries[entry.name]
            lookup = _layout_lookup(layout, Dict{Symbol,Any}())
            offsets, shapes = lookup(cell.offsets), lookup(cell.shapes)
            event_shapes = Any[nothing for _ in 1:entry.size]
            for k in eachindex(shapes), j in offsets[k]+1:offsets[k+1]
                event_shapes[j] = (offsets[k+1]-offsets[k], Tuple(shapes[k]))
            end
            sampling = (; arguments=cell.arguments, keywords=cell.keywords,
                coordinates=cell.geometry.coordinate_names)
        end
        lo, hi = bound_identity(entry.lo), bound_identity(entry.hi)
        for j in 1:entry.size
            label = string(labels[entry.offset+j-1])
            geometry = (; kind=entry.kind, transform=entry.transform,
                lo, hi, event_shape=event_shapes[j], sampling)
            push!(records, (; identity=(path, label), geometry))
        end
    end
    length(records) == layout.total || throw(DimensionMismatch("SMC coordinate inventory"))
    allunique(r.identity for r in records) ||
        throw(ArgumentError("authored SMC coordinate identities are ambiguous"))
    deepcopy(records)
end

function _smc_match_coordinates(previous, current)
    index = Dict(r.identity => i for (i,r) in enumerate(previous))
    old = zeros(Int, length(current))
    fresh = Int[]
    for (i, coordinate) in enumerate(current)
        j = get(index, coordinate.identity, 0)
        if iszero(j)
            push!(fresh, i)
        else
            _same_program(previous[j].geometry, coordinate.geometry) ||
                throw(ArgumentError("SMC coordinate $(coordinate.identity) changed its transform or event geometry"))
            old[i] = j
        end
    end
    (; old, fresh)
end
