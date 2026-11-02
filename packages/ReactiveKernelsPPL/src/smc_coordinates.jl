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
Recovery allocates one full vector only when requested.
"""
function smc_recover end

# Fresh priors use the shared authored-factor inventory. Observations use
# the existing partition lowering, with all rows in one prepared partition.
_smc_window_terms(built, binding, fresh, u0) = isempty(fresh) ?
    _smc_likelihood_terms(built, binding, u0) :
    _smc_fresh_window_terms(built, binding, fresh, u0)

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
        planner = Base.invokelatest(ReactiveKernels.partition_plan_kernel,
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
    (; spec, have, want, bound=(; bindings...))
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
    kernel = Base.invokelatest(prepare, spec; have, want=terms.want, bound,
        exemplars=(Vector{Float64}(u0[indices]),))
    ReactiveKernels._kernel_latest_prepared(kernel)
end

# Reuse the public authored coordinate inventory, including Cartesian/cell
# indices, scoped names and coefficient levels. Offsets are never identities.
function _smc_coordinate_records(layout::LayoutTable)
    records = NamedTuple[]
    labels = coordinate_names(layout)
    for entry in layout.entries
        path = _smc_output_path(layout, entry)
        sampling = entry.sampling === nothing ? nothing :
            (; entry.sampling.rhs, entry.sampling.broadcast, entry.sampling.constructor_broadcast)
        coupled = !(entry.transform in (:identity, :exp, :logistic, :interval, :floored, :upper)) ||
            entry.sampling !== nothing || haskey(layout.cell_geometries, entry.name)
        event_shapes = fill(coupled ? (entry.size, Tuple(entry.dims)) : nothing, entry.size)
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
        for j in 1:entry.size
            label = string(labels[entry.offset+j-1])
            geometry = (; kind=entry.kind, transform=entry.transform,
                lo=entry.lo, hi=entry.hi, event_shape=event_shapes[j], sampling)
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
