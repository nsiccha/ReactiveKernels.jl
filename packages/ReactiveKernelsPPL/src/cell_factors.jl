"""
    prepare_cell_factors(built, plan, observation; on_error = nothing, locality = nothing)

Discover sums read by one observation's cell, including sums in earlier
plates composed into it. The model may name a sum or write it inline. Factored
sums have elementwise operands with data-defined gathered rows; other sums
remain in the outer computation.
This includes weighted operands `sum(exp.(eta[rows[k]]) .* w[rows[k]])`
and a cell-local `mu = exp.(eta[rows[k]])` used by several sums, such as
`NegativeBinomial2(sum(mu), phi * sum(mu)^2 / sum(mu .^ 2))`.
Ordinary array sums such as `sum(z[rows[k]] + w[rows[k]])` also expose
local summands, including through transparent forwarding helpers. Base array
dispatch and the original full-array shape checks must be proved; addition
is not treated as a broadcast with singleton expansion.
Typed identity assignments between plates do not hide these sums when the
input already satisfies the declaration; real conversions remain in the graph.
`cell_reductions(f)` gives the reductions in graph order;
`reduction_rows(f, j, cell)` gives reduction `j`'s row labels in term order.
When several operands gather different arrays, labels come from the first
gather; each term still reads all operands at its corresponding position.

Use [`prepare_cell_factor_query`](@ref) for values and packed-coordinate
footprints before selecting coordinates for [`prepare_cell_factor_ad`](@ref).
Both prepare the outer density (`:outer`) or one
summand (reduction index `j`). With no sums the outer is the ordinary cell.
The outer's sum inputs are independent values: their dependence on the packed
point is supplied by the summand operators and the chain rule.

A sum whose summands read another sum (a density total over `dose .* f.(t)`, with
`dose` a sum) is no reduction: it stays in the outer factor, which reads the inner
sum from its sum vector and computes the total at its own rows.

Pass `partitioned = :z, u0 = u` to cache shared-only whole-array producers
before factoring. Values refresh when the shared coordinates change. Compact
factor AD then selects only partitioned coordinates. Mixed shared derivatives
use [`prepare_cell_accumulator`](@ref) and [`cell_pullback!`](@ref); operators
prepared from this factor family share one prefix cache and pullback pass.
Inside the callback, factor derivative verbs accept the active pullback in
place of `base`, reusing its copied point without rescanning shared coordinates.
"""
function prepare_cell_factors(built, plan::StructuralPlan, observation::Symbol;
        on_error = nothing, locality = nothing, partitioned = (), u0 = nothing)
    built = _binding_build(built, plan)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_cell_factors requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    _refuse_absent_cells(plan, (observation,))
    spec, want = _cell_query_spec(built, plan, (observation,); locality)
    have = (_query_have(plan)..., _CELL_PORT)
    rule = _query_partition_rule(plan, built.layout, partitioned, nothing)
    parts = _query_shared_parts(spec, plan, have, want, rule, (_CELL_PORT,), on_error)
    exemplar = _query_exemplar(built.layout, u0)
    shared = isempty(rule) ? nothing : _make_cell_shared(parts, built.layout,
        last(only(rule)), exemplar, on_error)
    factor_bound = _query_bound(plan)
    factors = if parts === nothing
        _with_query_data(plan, spec, on_error) do cache
            ReactiveKernels._cell_factor_graph(spec, want;
                have, bound = factor_bound, on_error, locality, _data_cache = cache)
        end
    else
        Base.invokelatest(ReactiveKernels._cell_factor_graph, parts.residual,
            spec.ports, spec.port_order; on_error, locality)
    end
    # Factoring exposes new expressions inside cells. Reintroduce the original
    # data ports as bound HAVE values so ordinary partial evaluation can hoist
    # those expressions too. The shared frontier stays a live HAVE boundary.
    if parts !== nothing
        for name in keys(factor_bound)
            name in factors.spec.have_names || push!(factors.spec.have_names, name)
        end
    end
    gathered = NamedTuple[]
    for s in factors.sums
        rows = ReactiveKernels._cell_factor_rows(factors.spec, s.argument; bound = factor_bound)
        rows === nothing || push!(gathered, (; s..., positions = rows))
    end
    innermost, enclosing = ReactiveKernels._cell_factor_nesting(factors.spec, gathered)
    # Removing the packed point proves the row query is data-defined.
    rowhave = Tuple(n for n in factors.spec.have_names if n !== :unconstrained &&
                    n ∉ _cell_frontier_names(shared))
    reductions = map(innermost) do s
        kernel = Base.invokelatest(_prepare_cached, plan, factors.spec; have = rowhave,
            want = s.positions, bound = factor_bound, on_error)
        (; s..., rows = kernel)
    end
    CellFactors(factors.spec, plan, built.layout, observation, want, reductions, enclosing,
                on_error, locality, shared, exemplar)
end

struct CellFactors{S,P,L,R,N,E,C,H}
    spec::S
    plan::P
    layout::L
    observation::Symbol
    want::Symbol
    reductions::R
    # Sums that read a reduction: part of the outer factor (`_cell_factor_nesting`).
    enclosing::N
    on_error::E
    locality::C
    shared::H
    # The packed point that types strict preparation (`u0`, or zeros).
    exemplar::Vector{Float64}
end

"Reduction names in the order used by the outer factor's sum vector."
cell_reductions(f::CellFactors) = Tuple(r.name for r in f.reductions)

"The row labels of reduction `j` in `cell`, in summand-position order."
reduction_rows(f::CellFactors, j::Integer, cell::Integer) =
    Base.invokelatest(f.reductions[j].rows, cell)

function _cell_factor_spec(f::CellFactors, factor)
    RK = ReactiveKernels
    outer = factor === :outer
    outer || factor isa Integer && 1 <= factor <= length(f.reductions) ||
        throw(ContractValidationError("[query] factor must be :outer or a reduction index"))
    omit = outer ? Set(r.value.id for r in Iterators.flatten((f.reductions, f.enclosing))) :
        Set{RK.ValueId}()
    spec = RK._copy_cell_factor_spec(f.spec; omit)
    if outer
        name = :_ppl_cell_sums
        haskey(spec.ports, name) && throw(ContractValidationError(
            "[query] factor port `$name` collides with a model port"))
        sums = value!(spec.graph, name, AbstractVector{Float64})
        spec.ports[name] = sums
        push!(spec.port_order, name)
        push!(spec.have_names, name)
        for (j, r) in enumerate(f.reductions)
            add!(spec.graph, (sums,) => r.value, RK._CellFactorSumRead(j))
        end
        foreach(s -> RK._cell_factor_sum_cell!(spec, s, f.locality), f.enclosing)
        want = f.want
    else
        name = :_ppl_cell_term
        haskey(spec.ports, name) && throw(ContractValidationError(
            "[query] factor port `$name` collides with a model port"))
        term = value!(spec.graph, name, Int)
        spec.ports[name] = term
        push!(spec.port_order, name)
        push!(spec.have_names, name)
        want = RK._cell_factor_term!(spec, f.reductions[factor].argument, term, f.locality)
    end
    spec, want
end

"""
    prepare_cell_factor_query(f::CellFactors, factor) -> CellFactorQuery

Prepare a scalar value query for the outer density (`factor = :outer`) or
one summand (`factor = j`), without AD or a compact coordinate selection.
Call the outer as `q(u, sums, cell)` and a summand as `q(u, cell, term)`.
`sums` follows `cell_reductions(f)`; `term` is a position in
`reduction_rows(f, j, cell)`, not a row label. Empty reductions have no
summand calls. With no reductions, pass an empty sum vector to the outer.

An `AbstractVector{Float64}` packed point is passed through without copying,
so a recording vector can observe the reads made by that evaluation.
Recording observes only the executed path. For a structural footprint,
including both arms of retained branches, prepare
`fp = ReactiveKernels.partition_footprint_kernel(q.kernel)` once and call
`Base.invokelatest(fp, args...).unconstrained` with the same arguments as `q`.
The result is sorted unique packed indices, or `:whole`; the latter means
`collect(eachindex(u))`. Structural footprints may be conservative, as for
`prepare_cell_query`. The outer footprint treats the supplied sums as
independent and therefore contains only its direct packed-coordinate reads.
Scalar vector reads such as `eta[anchor[cell]]` retain row demand through
elementwise predictors, including when the same predictor feeds a sum.

Use these footprints to select compact coordinates for
[`prepare_cell_factor_ad`](@ref). The public `kernel` property also supports
RK's `code_expr` and `partition_locality` inspection. Other real-valued packed
vectors are converted to `Vector{Float64}`, as for `CellQuery`.

Native preparation uses a packed `Vector{Float64}`, `Int` positions and a
`Vector{Float64}` of sums, so that a strict cell knows which of its values are
numbers. Query calls keep other argument types by preparing and caching a
specialization for them, as `CellQuery` does. Without a shared prefix, direct
calls to `q.kernel` and its full footprint kernel require those exact types.
"""
function prepare_cell_factor_query(f::CellFactors, factor)
    spec, want = _cell_factor_spec(f, factor)
    outer = factor === :outer
    frontiers = _cell_frontier_names(f.shared)
    base_have = _query_have(f.plan)
    tail = outer ? (:_ppl_cell_sums, _CELL_PORT) : (_CELL_PORT, :_ppl_cell_term)
    have = (base_have..., tail..., frontiers...)
    bound = _cell_factor_bound(f)
    args = (f.exemplar, (outer ? (zeros(length(f.reductions)), 1) : (1, 1))...,
            _cell_shared_values(f.shared)...)
    residual = Base.invokelatest(_prepare_cached, f.plan, spec; have, want, bound, on_error = f.on_error,
                                 exemplars = args)
    kernel = if _cell_has_prefix(f.shared)
        full = ReactiveKernels._cell_factor_full_spec(spec, f.shared.prefix_plan)
        Base.invokelatest(_prepare_cached, f.plan, full; have = (base_have..., tail...), want,
                         bound, on_error = f.on_error)
    else
        residual
    end
    CellFactorQuery(kernel, residual, f.shared, outer, length(f.reductions), args,
                    Dict{DataType,Any}(), f.on_error)
end

"Scalar outer or summand query; construct with [`prepare_cell_factor_query`](@ref)."
struct CellFactorQuery{K,R,S,A,E}
    kernel::K
    residual::R
    shared::S
    outer::Bool
    nsums::Int
    # The arguments `residual` was prepared for, and its other specializations.
    args::A
    specializations::Dict{DataType,Any}
    on_error::E
end

# The residual for these arguments: the prepared one is known from the types
# alone, so a call with its exact types never reaches the cache or boxes it.
@inline _cell_factor_residual(q::CellFactorQuery{<:Any,<:Any,<:Any,A}, args::A) where {A} =
    q.residual
_cell_factor_residual(q::CellFactorQuery, args) = get!(q.specializations, typeof(args)) do
    Base.invokelatest(prepare, q.residual.plan; exemplars = args, on_error = q.on_error)
end

function _check_cell_factor_args(q::CellFactorQuery, a)
    if q.outer
        a isa AbstractVector{Float64} || throw(ContractValidationError(
            "[query] an outer factor expects a Float64 sum vector and a cell"))
        length(a) == q.nsums || throw(ContractValidationError(
            "[query] expected $(q.nsums) sum values, got $(length(a))"))
    end
    nothing
end
function (q::CellFactorQuery)(u::AbstractVector{Float64}, a, b)
    _check_cell_factor_args(q, a)
    Base.invokelatest(_refresh_cell_shared!, q.shared, u)
    # Keep the latest-world refresh, but read its stored values through the
    # concrete cache type. Splatting invokelatest's Any result boxes the
    # prepared callable repeatedly, even when the shared tuple is empty.
    args = (u, a, b, _cell_shared_values(q.shared)...)
    Base.invokelatest(_cell_factor_residual(q, args), args...)::Float64
end
(q::CellFactorQuery)(u::AbstractVector{<:Real}, a, b) = q(Vector{Float64}(u), a, b)

"""
    prepare_cell_factor_ad(prepare_operator, f, factor, base0, indices0, backend, exemplars...;
                          cell0 = 1, term0 = 1)

Prepare the outer density (`factor = :outer`) or one summand (`factor = j`)
with any RK derivative preparation accepted by `prepare_compact_cell_ad`.
The outer uses two active ports `(sums, active)`, so its directions and
derivatives are tuples of those vectors. Call RK derivative verbs with
`base, sums, active, indices, cell` last. A summand has only `active`; call
with `base, active, indices, cell, term` last, where `term` is a position in
`reduction_rows(f, j, cell)`, not a row label. Select the packed coordinates
read directly by the outer, or by that summand, respectively. Selections may
change between calls as for `prepare_compact_cell_ad`.

Each prepared operator owns its compact selection map; use one per concurrent
caller. `base` and `indices` are read only. Empty reductions have no summand
calls and contribute a zero sum to the outer. `cell0` and `term0` select the
preparation exemplar; use a nonempty cell for a backend that evaluates its
summand exemplar. The outer's exemplar sums are evaluated from `base0` once
at preparation and always form a vector, including a single reduction.
"""
function prepare_cell_factor_ad(prepare_operator, f::CellFactors, factor,
        base0::AbstractVector{Float64}, indices0::AbstractVector{Int}, backend, exemplars...;
        cell0::Int = 1, term0::Int = 1)
    _check_compact_cell_indices(base0, indices0, indices0, f.layout.total)
    _check_cell_local_selection(f.shared, indices0)
    shared_values = Base.invokelatest(_refresh_cell_shared!, f.shared, base0)
    spec, want = _cell_factor_spec(f, factor)
    _compact_overlay!(spec, f.plan)
    outer = factor === :outer
    selected = Vector{Int}(indices0)
    slots = _compact_cell_slots(f.layout.total, selected)
    active0 = base0[indices0]
    sums0 = if outer && !isempty(f.reductions)
        query_have = _query_have(f.plan)
        sumkernel = Base.invokelatest(_prepare_cached, f.plan, f.spec;
            have = (query_have..., _CELL_PORT, _cell_frontier_names(f.shared)...),
            want = Tuple(r.value for r in f.reductions),
            bound = _cell_factor_bound(f), on_error = f.on_error,
            exemplars = (base0, cell0, shared_values...))
        values = Base.invokelatest(sumkernel, base0, cell0, shared_values...)
        # A prepared kernel unwraps one WANT to a scalar. Collecting that
        # scalar would produce a zero-dimensional array, not the promised
        # one-element active sum vector.
        length(f.reductions) == 1 ? Float64[values] : collect(Float64, values)
    else
        Float64[]
    end
    point = outer ? (base0, sums0, active0, slots, cell0) :
                    (base0, active0, slots, cell0, term0)
    have = outer ?
        (:_ppl_cell_base, :_ppl_cell_sums, :_ppl_cell_active, :_ppl_cell_slots,
         _query_columns(f.plan)..., _frozen_graph_names(f.plan)..., _CELL_PORT,
         _cell_frontier_names(f.shared)...) :
        (:_ppl_cell_base, :_ppl_cell_active, :_ppl_cell_slots,
         _query_columns(f.plan)..., _frozen_graph_names(f.plan)..., _CELL_PORT, :_ppl_cell_term,
         _cell_frontier_names(f.shared)...)
    kernel = Base.invokelatest(_prepare_cached, f.plan, spec; have, want, bound = _cell_factor_bound(f),
                              on_error = f.on_error, exemplars = (point..., shared_values...))
    active = outer ? (:_ppl_cell_sums, :_ppl_cell_active) : :_ppl_cell_active
    # An outer factor with no reductions differentiates its one active vector.
    # Its sum operand is empty and unread; as a member of a tuple point it needs
    # a tuple derivative, which forward over reverse cannot take and which static
    # activity analysis refuses once a large cell is not inlined.
    single = outer && isempty(f.reductions)
    tangents = single ? map(_cell_no_sums, exemplars) : exemplars
    prepared = Base.invokelatest(prepare_operator, kernel, backend, tangents...,
        point..., shared_values...; active = single ? :_ppl_cell_active : active)
    CellFactorAD(prepared, f.layout, selected, slots, outer, length(f.reductions), f.shared, sums0)
end

# The `(sum_part, coordinate_part)` shape a caller uses for an outer factor, to
# and from the one active vector of an outer factor with no reductions.
_cell_no_sums(x) = x
_cell_no_sums(x::Tuple) = map(_cell_no_sums, x)
function _cell_no_sums(x::Tuple{AbstractVector{<:Real},AbstractVector{<:Real}})
    isempty(first(x)) || throw(ContractValidationError(
        "[query] this outer factor has no sums, so a direction's sum part must be empty"))
    last(x)
end
_cell_with_sums(x, empty) = x
_cell_with_sums(x::Tuple, empty) = map(v -> _cell_with_sums(v, empty), x)
_cell_with_sums(x::AbstractVector{<:Real}, empty) = (empty, x)
_cell_single_active(q) = q.outer && q.nsums == 0

struct CellFactorAD{P,L,S}
    prepared::P
    layout::L
    selected::Vector{Int}
    slots::Vector{Int32}
    outer::Bool
    nsums::Int
    shared::S
    exemplar_sums::Vector{Float64}
end

for verb in (:ad_gradient, :ad_value_and_gradient, :ad_value_and_gradient!,
        :ad_pullback, :ad_value_and_pullback, :ad_value_and_pullback!,
        :ad_pushforward, :ad_value_and_pushforward,
        :ad_hvp, :ad_hvp!, :ad_gradient_and_hvp, :ad_gradient_and_hvp!,
        :ad_third, :ad_third!, :ad_gradient_hvp_and_third,
        :ad_gradient_hvp_and_third!)
    @eval ReactiveKernels.$verb(q::CellFactorAD, args...) =
        Base.invokelatest(_cell_factor_ad_call, ReactiveKernels.$verb, q, args...)
end

function _cell_factor_ad_call(verb, q::CellFactorAD, args::Vararg{Any,N}) where {N}
    N >= 5 || throw(ContractValidationError("[query] a cell factor derivative needs five point arguments"))
    leading = ntuple(k -> args[k], Val(N - 5))
    if q.outer
        base_arg, sums, active, indices, cell = args[N-4:N]
        base = _cell_ad_base(q.shared, base_arg)
        length(sums) == q.nsums || throw(ContractValidationError(
            "[query] expected $(q.nsums) sum values, got $(length(sums))"))
        _select_compact_cells!(q, base, active, indices)
        _check_cell_local_selection(q.shared, indices)
        shared_values = _cell_ad_values(q.shared, base_arg)
        _cell_single_active(q) || return verb(q.prepared, leading..., base, sums, active,
            q.slots, cell, shared_values...)
        _cell_with_sums(verb(q.prepared, map(_cell_no_sums, leading)..., base, sums, active,
            q.slots, cell, shared_values...), q.exemplar_sums)
    else
        base_arg, active, indices, cell, term = args[N-4:N]
        base = _cell_ad_base(q.shared, base_arg)
        _select_compact_cells!(q, base, active, indices)
        _check_cell_local_selection(q.shared, indices)
        shared_values = _cell_ad_values(q.shared, base_arg)
        verb(q.prepared, leading..., base, active, q.slots, cell, term, shared_values...)
    end
end
