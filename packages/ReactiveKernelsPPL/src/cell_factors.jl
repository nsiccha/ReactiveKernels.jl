"""
    prepare_cell_factors(built, plan, observation; on_error = nothing, locality = nothing)

Discover sums read by one observation's cell, including sums in earlier
plates composed into it. The model may name a sum or write it inline. Factored
sums have elementwise operands with data-defined gathered rows; other sums
remain in the outer computation.
`cell_reductions(f)` gives the reductions in graph order;
`reduction_rows(f, j, cell)` gives reduction `j`'s row labels in term order.
When several operands gather different arrays, labels come from the first
gather; each term still reads all operands at its corresponding position.

Use [`prepare_cell_factor_ad`](@ref) for the outer density (`:outer`) or one
summand (reduction index `j`). With no sums the outer is the ordinary cell.
The outer's sum inputs are independent values: their dependence on the packed
point is supplied by the summand operators and the chain rule.
"""
function prepare_cell_factors(built, plan::StructuralPlan, observation::Symbol;
        on_error = nothing, locality = nothing)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_cell_factors requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    _refuse_absent_cells(plan, (observation,))
    spec, want = _cell_query_spec(built, plan, (observation,); locality)
    have = (_query_have(plan)..., _CELL_PORT)
    factors = ReactiveKernels._cell_factor_graph(spec, want;
        have, bound = _query_bound(plan), on_error)
    reductions = map(factors.sums) do s
        rows = ReactiveKernels._cell_factor_rows(factors.spec, s.argument)
        rows === nothing && return nothing
        # Removing the packed point proves the row query is data-defined.
        rowhave = Tuple(n for n in have if n !== :unconstrained)
        kernel = Base.invokelatest(prepare, factors.spec; have = rowhave,
            want = rows, bound = _query_bound(plan), on_error)
        (; s..., rows = kernel)
    end
    filter!(!isnothing, reductions)
    CellFactors(factors.spec, plan, built.layout, observation, want, reductions,
                on_error, locality)
end

struct CellFactors{S,P,L,R,E,C}
    spec::S
    plan::P
    layout::L
    observation::Symbol
    want::Symbol
    reductions::R
    on_error::E
    locality::C
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
    omit = outer ? Set(r.value.id for r in f.reductions) : Set{RK.ValueId}()
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
            add!(spec.graph, (sums,) => r.value, Base.Fix2(getindex, j))
        end
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
at preparation.
"""
function prepare_cell_factor_ad(prepare_operator, f::CellFactors, factor,
        base0::AbstractVector{Float64}, indices0::AbstractVector{Int}, backend, exemplars...;
        cell0::Int = 1, term0::Int = 1)
    _check_compact_cell_indices(base0, indices0, indices0, f.layout.total)
    spec, want = _cell_factor_spec(f, factor)
    _compact_overlay!(spec, f.plan)
    outer = factor === :outer
    have = outer ?
        (:_ppl_cell_base, :_ppl_cell_sums, :_ppl_cell_active, :_ppl_cell_slots,
         _query_columns(f.plan)..., _frozen_graph_names(f.plan)..., _CELL_PORT) :
        (:_ppl_cell_base, :_ppl_cell_active, :_ppl_cell_slots,
         _query_columns(f.plan)..., _frozen_graph_names(f.plan)..., _CELL_PORT, :_ppl_cell_term)
    kernel = Base.invokelatest(prepare, spec; have, want, bound = _query_bound(f.plan),
                              on_error = f.on_error)
    selected = Vector{Int}(indices0)
    slots = _compact_cell_slots(f.layout.total, selected)
    active0 = base0[indices0]
    sums0 = if outer && !isempty(f.reductions)
        sumkernel = Base.invokelatest(prepare, f.spec;
            have = (_query_have(f.plan)..., _CELL_PORT),
            want = Tuple(r.value for r in f.reductions),
            bound = _query_bound(f.plan), on_error = f.on_error)
        collect(Float64, Base.invokelatest(sumkernel, base0, cell0))
    else
        Float64[]
    end
    point = outer ? (base0, sums0, active0, slots, cell0) :
                    (base0, active0, slots, cell0, term0)
    active = outer ? (:_ppl_cell_sums, :_ppl_cell_active) : :_ppl_cell_active
    prepared = Base.invokelatest(prepare_operator, kernel, backend, exemplars..., point...; active)
    CellFactorAD(prepared, f.layout, selected, slots, outer, length(f.reductions))
end

struct CellFactorAD{P,L}
    prepared::P
    layout::L
    selected::Vector{Int}
    slots::Vector{Int32}
    outer::Bool
    nsums::Int
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
        base, sums, active, indices, cell = args[N-4:N]
        length(sums) == q.nsums || throw(ContractValidationError(
            "[query] expected $(q.nsums) sum values, got $(length(sums))"))
        _select_compact_cells!(q, base, active, indices)
        verb(q.prepared, leading..., base, sums, active, q.slots, cell)
    else
        base, active, indices, cell, term = args[N-4:N]
        _select_compact_cells!(q, base, active, indices)
        verb(q.prepared, leading..., base, active, q.slots, cell, term)
    end
end
