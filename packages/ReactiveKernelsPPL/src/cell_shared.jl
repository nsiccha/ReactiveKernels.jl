# A cell family owns one value cache. Its compact derivatives condition on
# the shared coordinates; a separate accumulation pass retains their mixed
# derivatives and performs the shared pullback once after all local terms.
mutable struct _CellShared{C,P,F}
    const cache::C
    const prefix_plan::P
    const frontier::F
    const local_positions::BitVector
    const shared_positions::Vector{Int}
    const shared_point::Vector{Float64}
    generation::UInt
end

function _make_cell_shared(parts, layout, labels, exemplar, on_error)
    cache = parts === nothing ? nothing :
        _prepare_value_prefix(parts, layout, labels, exemplar, on_error)
    positions = findall(ismissing, labels)
    point = cache === nothing ? exemplar[positions] : cache.point
    _CellShared(cache, parts === nothing ? nothing : parts.prefix,
        parts === nothing ? () : Tuple(v.name for v in parts.frontier),
        BitVector(.!ismissing.(labels)), positions, point, UInt(0))
end

_cell_has_prefix(::Nothing) = false
_cell_has_prefix(s::_CellShared) = s.cache !== nothing
_cell_frontier_names(::Nothing) = ()
_cell_frontier_names(s::_CellShared) = s.frontier
_cell_shared_values(s::_CellShared) = s.cache === nothing ? () : s.cache.values
_cell_factor_bound(f::CellFactors) = _query_bound(f.plan)

_check_cell_local_selection(::Nothing, indices) = nothing
function _check_cell_local_selection(s::_CellShared, indices)
    all(i -> s.local_positions[i], indices) || throw(ContractValidationError(
        "[query] a prefix-cached compact derivative is conditional on shared coordinates; " *
        "select only coordinates declared partitioned. Use prepare_cell_accumulator and " *
        "cell_pullback! for the mixed shared derivatives."))
    nothing
end

_refresh_cell_shared!(::Nothing, u) = ()
function _refresh_cell_shared!(s::_CellShared, u)
    length(u) == length(s.local_positions) || throw(ContractValidationError(
        "[query] packed point has the wrong length for the shared cell cache"))
    changed = s.cache !== nothing && !s.cache.valid || any(eachindex(s.shared_positions)) do j
        !isequal(s.shared_point[j], u[s.shared_positions[j]])
    end
    if changed
        # Increment before evaluation: a failed refresh also invalidates a
        # live accumulation pass. It cannot finish with stale shared values.
        s.generation += UInt(1)
        if s.cache === nothing
            for (j, i) in enumerate(s.shared_positions)
                s.shared_point[j] = u[i]
            end
        else
            _refresh_partition_prefix!(s.cache, u)
        end
    end
    _cell_shared_values(s)
end

# The shared splitter proves that the frontier does not read partitioned
# coordinates, and the caller checks that the compact selection contains
# only those coordinates. Represent that conditional function as its own
# graph. Keeping the original producers would incorrectly describe the
# frontier as downstream of the compact overlay's whole packed vector.
# Mixed derivatives still activate this frontier and use the original prefix
# in cell_pullback!; ordinary RK HAVE cuts keep their provenance checks.
function _conditional_cell_plan(p::ReactiveKernels.Plan)
    RK = ReactiveKernels
    old = p.graph
    recipes = RK.Recipe[RK.Recipe(i, r.inputs, r.outputs, r.op, r.cost,
        r.cse_key, r.effectful, r.source) for (i, r) in enumerate(p.recipes)]
    graph = Graph(copy(old.values), recipes, Dict{RK.ValueId,Vector{Int}}(),
                  copy(old.aliases), old.version, nothing)
    RK._reindex_producers!(graph)
    template = RK.Plan(graph, p.have, p.want, recipes, p.producer, p.cost, recipes)
    RK._partial_subplan(template, p.have, p.want, recipes)
end

function _prepare_shared_compact_ad(prepare_operator, built, plan, observations,
        base0, indices0, backend, exemplars...; on_error, locality, partitioned)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_compact_cell_ad requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    _check_compact_cell_indices(base0, indices0, indices0, built.layout.total)
    names = observations isa Symbol ? (observations,) : Tuple(Symbol.(observations))
    isempty(names) && throw(ContractValidationError("[query] at least one observation is required"))
    _refuse_absent_cells(plan, names)
    spec, want = _cell_query_spec(built, plan, names; locality)
    rule = _query_partition_rule(plan, built.layout, partitioned, nothing)
    parts = _query_shared_parts(spec, plan, (_query_have(plan)..., _CELL_PORT),
        want, rule, (_CELL_PORT,), on_error)
    shared = _make_cell_shared(parts, built.layout, last(only(rule)), base0, on_error)
    _check_cell_local_selection(shared, indices0)
    selected = Vector{Int}(indices0)
    slots = _compact_cell_slots(built.layout.total, selected)
    active0 = base0[selected]
    point = (base0, active0, slots, 1, _cell_shared_values(shared)...)
    kernel = if parts === nothing
        have = _compact_overlay!(spec, plan)
        Base.invokelatest(prepare, spec; have, want, bound = _query_bound(plan), on_error)
    else
        residual = _conditional_cell_plan(_compact_partition_plan(parts.residual))
        Base.invokelatest(prepare, residual;
                         exemplars = point, on_error)
    end
    prepared = Base.invokelatest(prepare_operator, kernel, backend, exemplars...,
        point...; active = :_ppl_cell_active)
    CompactCellAD(prepared, built.layout, names, selected, slots, shared)
end

mutable struct CellPullback{S,P,V}
    const shared::S
    const prepared::P
    const base::Vector{Float64}
    const gradient::Vector{Float64}
    shadows::V
    generation::UInt
    active::Bool
end

"""
    prepare_cell_pullback(owner, backend, u0)

Prepare one shared pullback for a `CellFactors` or `CompactCellAD` constructed
with `partitioned`. `backend` is a reverse backend. Use [`cell_pullback!`](@ref)
to accumulate contributions from operators sharing this owner's cache. Storage
is linear in the packed point and shared frontier, independent of the number
of cells; no cell-by-population Jacobian is formed. One object per concurrent
caller, including its owner and associated cell operators.
"""
function prepare_cell_pullback(owner::Union{CellFactors,CompactCellAD,CellFactorAD}, backend,
        u0::AbstractVector{Float64})
    shared = owner.shared
    shared === nothing && throw(ContractValidationError(
        "[query] prepare cell pullbacks from an owner with partitioned coordinates"))
    base = Vector{Float64}(u0)
    values = Base.invokelatest(_refresh_cell_shared!, shared, base)
    shadows = map(v -> ReactiveKernels._ad_zero_cotangent(
        ReactiveKernels._ad_active_point_component(v)), values)
    prepared = if _cell_has_prefix(shared)
        seed = _partition_frontier_seed(map(ReactiveKernels._ad_active_value, shadows))
        c = shared.cache
        Base.invokelatest(prepare_ad_pullback, c.kernel, backend, seed,
            base, c.point, c.slots; active = :_ppl_cell_active)
    else
        nothing
    end
    CellPullback(shared, prepared, base, zeros(length(base)), shadows, shared.generation, false)
end

_zero_cell_shadow!(x::AbstractArray) = (fill!(x, zero(eltype(x))); x)
_zero_cell_shadow!(x::Base.RefValue) = (x[] = zero(x[]); x)
_zero_cell_shadow!(x::Union{Tuple,NamedTuple}) = (foreach(_zero_cell_shadow!, x); x)

"""
    cell_pullback!(visit, gradient, pullback, u)

Run `visit(pullback)` at a copied packed point `u`, then write the accumulated
gradient to `gradient`. In the callback, call [`accumulate_cell!`](@ref) for
each weighted value, first directional derivative, or second directional
derivative. After the callback, one shared-prefix pullback propagates their
combined cotangent. Local and direct packed derivatives are included too.

The shared cache refreshes automatically when its population coordinates change.
Do not change those coordinates through another operator during the callback:
that invalidates the pass and throws. A failed callback produces no output;
the next call resets all accumulation storage. Calls cannot be nested or run
concurrently on the same pullback.

```julia
cell_pullback!(gradient, pullback, u) do batch
    for i in eachindex(cells)
        accumulate_cell!(batch, third, directions[i], weights[i],
                         active[i], indices[i], cells[i])
    end
end
```
"""
function cell_pullback!(visit::F, gradient::AbstractVector{Float64}, p::CellPullback,
        u::AbstractVector{Float64}) where {F}
    p.active && throw(ContractValidationError("[query] cell pullbacks cannot be nested"))
    length(gradient) == length(u) == length(p.base) || throw(DimensionMismatch(
        "cell pullback point and gradient must match the packed layout"))
    copyto!(p.base, u)
    values = Base.invokelatest(_refresh_cell_shared!, p.shared, p.base)
    point = map(ReactiveKernels._ad_active_point_component, values)
    if !ReactiveKernels._ad_tangent_matches(point, p.shadows)
        p.shadows = map(ReactiveKernels._ad_zero_cotangent, point)
    end
    _zero_cell_shadow!(p.shadows)
    fill!(p.gradient, 0.0)
    p.generation = p.shared.generation
    p.active = true
    try
        visit(p)
        _check_cell_pass(p, p.shared)
        if p.prepared !== nothing
            c = p.shared.cache
            seed = _partition_frontier_seed(map(ReactiveKernels._ad_active_value, p.shadows))
            propagated = Base.invokelatest(ad_pullback, p.prepared, seed,
                p.base, c.point, c.slots)
            for (j, i) in enumerate(c.positions)
                p.gradient[i] += propagated[j]
            end
        end
        copyto!(gradient, p.gradient)
    finally
        p.active = false
    end
    gradient
end

function _check_cell_pass(p::CellPullback, shared)
    p.active && p.shared === shared && p.generation == shared.generation ||
        throw(ContractValidationError(
            "[query] accumulator must use this active cell pullback's unchanged shared cache"))
    nothing
end

struct CellAccumulator{P,Q,S}
    prepared::P
    query::Q
    local_shadow::S
end

"""
    prepare_cell_accumulator(query, backend, u0; order=0, cell0=1, term0=1, sums0=nothing)

Prepare additive mixed derivatives from a prefix-cached `CompactCellAD` or
`CellFactorAD`. `order` and `backend` follow RK's `prepare_ad_accumulator`:
0 accumulates gradients, 1 local-direction HVPs, and 2 third contractions
with two local directions. The resulting gradient includes shared coordinates.
For an outer factor, the direction also includes its independent sum inputs,
as `(sum_direction, coordinate_direction)`. Use the same cached factor owner
for all operators participating in a [`cell_pullback!`](@ref) pass.
"""
function prepare_cell_accumulator(q::Union{CompactCellAD,CellFactorAD}, backend,
        u0::AbstractVector{Float64}; order::Integer = 0, cell0::Int = 1,
        term0::Int = 1, sums0 = nothing)
    q.shared === nothing && throw(ContractValidationError(
        "[query] a cell accumulator requires a query prepared with partitioned coordinates"))
    _check_cell_local_selection(q.shared, q.selected)
    values = Base.invokelatest(_refresh_cell_shared!, q.shared, u0)
    active0 = u0[q.selected]
    outer = q isa CellFactorAD && q.outer
    sums = outer ? (sums0 === nothing ? copy(q.exemplar_sums) : Vector{Float64}(sums0)) : Float64[]
    tail = q isa CompactCellAD ? (active0, q.slots, cell0) : outer ?
        (sums, active0, q.slots, cell0) : (active0, q.slots, cell0, term0)
    local_ports = outer ? (:_ppl_cell_sums, :_ppl_cell_active) : (:_ppl_cell_active,)
    directional = outer ? local_ports : only(local_ports)
    prepared = Base.invokelatest(ReactiveKernels.prepare_ad_accumulator,
        q.prepared.kernel, backend, u0, tail..., values...;
        active = (:_ppl_cell_base, local_ports..., _cell_frontier_names(q.shared)...),
        directional, order)
    shadow = outer ? (zero(sums), zero(active0)) : zero(active0)
    CellAccumulator(prepared, q, shadow)
end

"""
    accumulate_cell!(batch, operator, direction, weight, active, indices, cell)

Accumulate one compact cell's weighted derivative contraction in an active
[`cell_pullback!`](@ref) callback. An outer factor takes
`sums, active, indices, cell` last; a summand takes `active, indices, cell, term`.
There is no base argument: the pass owns its copied packed point.
Pass `nothing` as direction for order zero.

Returns `(weighted_contraction, local_derivative)`. The local derivative
follows the ordinary compact operator's shape (including independent sum
derivatives for an outer factor), and is reused on the next call. Consume or
copy it before reusing the operator. Its packed-coordinate entries are also
accumulated into the pass's gradient. Sum entries are returned for the caller's
chain rule; sum inputs remain independent, exactly as for factor AD.
"""
accumulate_cell!(p::CellPullback, q::CellAccumulator, direction, weight, args...) =
    Base.invokelatest(_accumulate_cell!, p, q, direction, weight, args...)

function _accumulate_cell!(p, a::CellAccumulator{P,Q}, direction, weight,
        active, indices, cell) where {P,Q<:CompactCellAD}
    q = a.query
    _prepare_cell_contribution!(p, a, active, indices)
    value = ReactiveKernels.ad_accumulate!(a.prepared,
        (p.gradient, a.local_shadow, p.shadows...), direction, weight,
        p.base, active, q.slots, cell, _cell_shared_values(q.shared)...)
    _scatter_cell_contribution!(p.gradient, a.local_shadow, indices)
    value, a.local_shadow
end

function _accumulate_cell!(p, a::CellAccumulator{P,Q}, direction, weight,
        first, second, third, fourth) where {P,Q<:CellFactorAD}
    q = a.query
    if q.outer
        sums, active, indices, cell = first, second, third, fourth
        length(sums) == q.nsums || throw(DimensionMismatch("wrong number of factor sums"))
        _prepare_cell_contribution!(p, a, active, indices)
        value = ReactiveKernels.ad_accumulate!(a.prepared,
            (p.gradient, a.local_shadow..., p.shadows...), direction, weight,
            p.base, sums, active, q.slots, cell, _cell_shared_values(q.shared)...)
        _scatter_cell_contribution!(p.gradient, last(a.local_shadow), indices)
    else
        active, indices, cell, term = first, second, third, fourth
        _prepare_cell_contribution!(p, a, active, indices)
        value = ReactiveKernels.ad_accumulate!(a.prepared,
            (p.gradient, a.local_shadow, p.shadows...), direction, weight,
            p.base, active, q.slots, cell, term, _cell_shared_values(q.shared)...)
        _scatter_cell_contribution!(p.gradient, a.local_shadow, indices)
    end
    value, a.local_shadow
end

function _prepare_cell_contribution!(p, a, active, indices)
    q = a.query
    _check_cell_pass(p, q.shared)
    _select_compact_cells!(q, p.base, active, indices)
    _check_cell_local_selection(q.shared, indices)
    _zero_cell_shadow!(a.local_shadow)
end
function _scatter_cell_contribution!(gradient, local_gradient, indices)
    for (j, i) in enumerate(indices)
        gradient[i] += local_gradient[j]
    end
    nothing
end
