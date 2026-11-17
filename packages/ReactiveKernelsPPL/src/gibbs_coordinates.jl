# Conditionals of one packed coordinate. Changing one coefficient `w[m]` moves
# a product `A * w` by `(w[m] - previous) .* A[:, m]`, so the conditional reads
# each such product from a cached value instead of recomputing it. The planner
# applies no algebraic identities; this file states that one for the emitted
# program. Everything downstream of the product is the emitted program.

"Prepared coordinate conditionals returned by [`prepare_coordinate_conditional`](@ref)."
struct CoordinateConditional{L,J,C,U,R}
    layout::L
    joint::J
    conditional::C
    update::U
    refresh::R
    products::Vector{Symbol}
end

_is_packed_view(f) = f === Base.view || f == :(Base.view) ||
    (f isa GlobalRef && f.mod === Base && f.name === :view)

# Names bound to `Base.view(unconstrained, first:last)`, with both bounds as
# emitted (runtime values of the binding, never spliced numbers).
function _packed_views(statements)
    views = Dict{Symbol,Tuple{Any,Any}}()
    for st in statements
        name = _assigned_name(st)
        name === nothing && continue
        rhs = st.args[2]
        Meta.isexpr(rhs, :call, 3) && _is_packed_view(rhs.args[1]) &&
            rhs.args[2] === :unconstrained || continue
        range = rhs.args[3]
        Meta.isexpr(range, :call, 3) && range.args[1] === :(:) || continue
        views[name] = (range.args[2], range.args[3])
    end
    return views
end

# Statements `name = A * w` where `w` is a packed view and `A` is constant in
# the coordinates. Any other product stays as emitted and is recomputed.
function _coordinate_products(statements)
    views = _packed_views(statements)
    dependent = _gibbs_dependents(statements, (:unconstrained,))
    nodes = NamedTuple[]
    for (position, st) in enumerate(statements)
        name = _assigned_name(st)
        name === nothing && continue
        rhs = st.args[2]
        Meta.isexpr(rhs, :call, 3) && rhs.args[1] === :* || continue
        A, w = rhs.args[2], rhs.args[3]
        A isa Symbol && w isa Symbol && haskey(views, w) && A ∉ dependent || continue
        lo, hi = views[w]
        push!(nodes, (; position, name, matrix=A, vector=w, first=lo, last=hi))
    end
    return nodes
end

"""
    prepare_coordinate_conditional(built, plan, u0)

Prepare the conditional log density of one packed coordinate given all others,
for coordinate-wise transitions such as slice sampling within Gibbs.

The conditional keeps the emitted density terms that read the packed
coordinates and omits the rest, so it equals the joint log density up to a
constant. Each emitted matrix–vector product `A * w` whose vector is a
contiguous run of packed coordinates and whose matrix does not depend on them
is read from a cached value: a trial at coordinate `k` costs one column of `A`
instead of the whole product. Products over a transformed or reshaped vector
are recomputed as emitted. Every kept term is evaluated per trial, including
priors over the other coordinates.

Use [`coordinate_state`](@ref) to hold a point with its cached products,
[`coordinate_logdensity`](@ref) for a trial value and [`set_coordinate!`](@ref)
to move. Prepare once per binding; states are independent of each other.
Models with free integer parameters are not supported yet.
"""
function prepare_coordinate_conditional(built, plan::StructuralPlan, u0::AbstractVector;
        on_error=nothing)
    built = _binding_build(built, plan)
    _check_built_program(built, plan)
    _has_integer_parameters(plan) && throw(ArgumentError(
        "prepare_coordinate_conditional does not support free integer parameters yet"))
    layout = built.layout
    u = Vector{Float64}(u0)
    length(u) == layout.total || throw(DimensionMismatch(
        "coordinate conditional position has the wrong dimension"))

    sites = _DensitySite[]
    def = Base.invokelatest(_kernel_expr_latest, plan, layout; density_sites=sites)
    def = _elide_named_location_aliases(plan, def)
    def = _pointwise_arithmetic_plates(plan, def)
    def = _name_plate_cells(def, _location_value_aliases(plan, def))
    statements = Expr[s for s in def.args[2].args if s isa Expr && s.head !== :return]
    terms = Any[]
    for name in (:prior, :likelihood, :log_jacobian)
        st = only(s for s in statements if _assigned_name(s) === name)
        append!(terms, _gibbs_sum_terms(st.args[2]))
    end
    dependent = _gibbs_dependents(statements, (:unconstrained,))
    selected = filter(t -> _gibbs_reads(t, dependent), terms)

    nodes = _coordinate_products(statements)
    ports = Symbol[Symbol(:_ppl_cg_product_, i) for i in eachindex(nodes)]
    derived = copy(statements)
    for (node, port) in zip(nodes, ports)
        m = :(_ppl_cg_index - ($(node.first) - 1))
        # A coordinate outside this product's vector leaves the product as cached.
        derived[node.position] = :($(node.name)::Vector{Float64} =
            ($(node.first) <= _ppl_cg_index) & (_ppl_cg_index <= $(node.last)) ?
            $port .+ ($(node.vector)[$m] - _ppl_cg_previous) .*
                Base.view($(node.matrix), Base.Colon(), $m) :
            $port)
    end
    products = Expr(:tuple, (node.name for node in nodes)...)
    signature = copy(def.args[1])
    append!(signature.args, [:($port::Vector{Float64}) for port in ports])
    push!(signature.args, :(_ppl_cg_index::Int), :(_ppl_cg_previous::Float64))
    cached = isempty(nodes) ? Expr[] : Expr[:(_ppl_cg_products = $products)]
    body = Expr(:block, derived...,
        :(_ppl_cg_density::Float64 = $(_sum_terms(selected))), cached...,
        :(return _ppl_cg_density))
    spec = _eval_kernel_def(Expr(:(=), signature, body))
    _register_packed_check!(spec, layout)
    have = (_query_have(plan)..., ports..., :_ppl_cg_index, :_ppl_cg_previous)
    exemplars = (u, (Float64[] for _ in nodes)..., 1, 0.0)
    cut(want) = Base.invokelatest(_prepare_cached, plan, spec;
        have, want, bound=_query_bound(plan), exemplars, on_error)
    conditional = cut(:_ppl_cg_density)
    # A program without such a product has nothing to cache or to bring along.
    update, refresh = if isempty(nodes)
        Returns(()), Returns(())
    else
        refresh_spec = _eval_kernel_def(Expr(:(=), def.args[1], Expr(:block,
            statements..., :(_ppl_cg_products = $products),
            :(return _ppl_cg_products))))
        _register_packed_check!(refresh_spec, layout)
        cut(:_ppl_cg_products), Base.invokelatest(_prepare_cached, plan, refresh_spec;
            have=_query_have(plan), want=:_ppl_cg_products, bound=_query_bound(plan),
            exemplars=(u,), on_error)
    end
    joint = prepare_query(built, plan, :sampler; exemplars=(u,), on_error)
    return CoordinateConditional(layout, joint, conditional, update, refresh,
        Symbol[node.name for node in nodes])
end

"One point with its cached products, returned by [`coordinate_state`](@ref)."
mutable struct CoordinateState{P,T<:Tuple}
    const problem::P
    const position::Vector{Float64}
    products::T
end

"""
    coordinate_state(problem, u0)

Start a state at the packed point `u0`. The state owns a copy of the point and
its cached products; the joint log density must be finite there.
"""
function coordinate_state(problem::CoordinateConditional, u0::AbstractVector)
    position = Vector{Float64}(u0)
    length(position) == problem.layout.total || throw(DimensionMismatch(
        "coordinate state position has the wrong dimension"))
    isfinite(problem.joint(position)) || throw(ArgumentError(
        "coordinate state needs a finite joint density at its start"))
    return CoordinateState(problem, position, map(copy, problem.refresh(position)))
end

"""
    coordinate_logdensity(state, k, z)

Conditional log density of coordinate `k` at the value `z`, with every other
coordinate at the state's point. The state is left unchanged.
"""
function coordinate_logdensity(state::CoordinateState, k::Integer, z::Real)
    u = state.position
    checkbounds(u, k)
    previous = u[k]
    u[k] = z
    try
        return state.problem.conditional(u, state.products..., Int(k), previous)
    finally
        u[k] = previous
    end
end

"""
    set_coordinate!(state, k, z)

Move coordinate `k` to `z` and update each cached product by one column. If
the update throws, the state keeps its point.
"""
function set_coordinate!(state::CoordinateState, k::Integer, z::Real)
    u = state.position
    checkbounds(u, k)
    previous = u[k]
    u[k] = z
    state.products = try
        state.problem.update(u, state.products..., Int(k), previous)
    catch
        u[k] = previous
        rethrow()
    end
    return state
end

"""
    refresh_products!(state)

Recompute every cached product from the state's point and return the largest
absolute change, the rounding drift accumulated by the column updates.
"""
function refresh_products!(state::CoordinateState)
    fresh = map(copy, state.problem.refresh(state.position))
    drift = 0.0
    for (old, new) in zip(state.products, fresh)
        isempty(new) || (drift = max(drift, maximum(abs, old .- new)))
    end
    state.products = fresh
    return drift
end
