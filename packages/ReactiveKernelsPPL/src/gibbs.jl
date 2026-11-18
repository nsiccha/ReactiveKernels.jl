# PPL conditional construction. WarmupHMC owns the continuous transition;
# this layer only derives model blocks and updates their named integer state.
"Prepared model conditionals returned by [`prepare_gibbs`](@ref)."
struct GibbsProblem{L,C,I,F}
    layout::L
    continuous::C
    integers::I
    joint::F
end

"""
    prepare_gibbs(built, plan, u0; discrete, backend=nothing, blocks=nothing)

Prepare exact finite integer conditionals and one continuous conditional over
all packed real coordinates. `blocks` is a tuple of tuples of integer names;
by default each scalar integer declaration is its own block. Group dependent
integers together when single-site updates cannot move between valid states.
Every free integer must occur exactly once. Names use the author namespaces,
for example `Symbol("child.k")` for a nested declaration.

Each conditional keeps the emitted density terms that transitively read its
block. Terms constant in that block are omitted; the resulting conditional
log density need not equal the full joint log density. Finite supports remain
runtime plate axes. A joint block enumerates their Cartesian product.

Integer arrays, declared (`z[axes(y, 1)] .~ Categorical(w)`) or written as
`@plate` loop cells, are updated exactly as their own blocks after the scalar
ones (`GibbsArrayBlock`). An array's elements may each have their own support,
when a support bound is read per element (`Binomial.(n, p)` with a vector
`n`). In a model whose other integers are marginalized
(`marginalize(model, :a)`), the explicit integers are updated from their
collapsed conditionals, the marginalized ones summed out exactly.

Continuous derivatives use the supplied ordinary AD backend. No backend is
needed for a model with zero continuous coordinates. Prepare one problem per
concurrent chain: its continuous query owns mutable derivative buffers.
"""
function prepare_gibbs(built, plan::StructuralPlan, u0::AbstractVector;
        discrete, backend=nothing, blocks=nothing, on_error=nothing)
    built = _binding_build(built, plan)
    _check_built_program(built, plan)
    haskey(plan.frozen.values, :_ppl_discrete) && throw(ArgumentError(
        "prepare_gibbs needs mutable integer inputs, not a frozen integer query"))
    layout = built.layout
    u = Vector{Float64}(u0)
    length(u) == layout.total || throw(DimensionMismatch("Gibbs position has the wrong dimension"))
    _check_integer_parameters(layout, discrete)
    # Marginalized integers are summed out inside the program's density terms
    # (`marginalize`); every conditional reads those sums, so the remaining
    # integers are updated from their collapsed conditionals.
    all_ps = [p for p in _integer_parameters(plan) if p.name ∉ plan.marginalized_integers]
    isempty(all_ps) && throw(ArgumentError("prepare_gibbs needs at least one " *
        (isempty(plan.marginalized_integers) ? "integer parameter" :
            "integer parameter that is not marginalized; sample a fully " *
            "marginalized model with NUTS on its real coordinates")))
    ps = [p for p in all_ps if p isa SampledParameter]
    # A declared integer array and an integer `@plate` cell (`z[i] ~
    # Categorical(w)` in a loop) are both one array of integers.
    array_ps = [p for p in all_ps if p isa ArrayParameter || p isa PlateParameter]
    length(ps) + length(array_ps) == length(all_ps) || throw(ArgumentError(
        "prepare_gibbs updates scalar integers and integer arrays"))
    layout.total == 0 || backend !== nothing || throw(ArgumentError(
        "prepare_gibbs requires an AD backend for its continuous block"))
    paths = [get(layout.name_paths, p.name, (p.name,)) for p in ps]
    authored = Symbol.(_scope_path_string.(paths))
    groups = blocks === nothing ? Tuple((name,) for name in authored) : Tuple(blocks)
    all(g -> g isa Tuple && !isempty(g) && all(n -> n isa Symbol, g), groups) ||
        throw(ArgumentError("blocks must contain nonempty tuples of integer parameter names"))
    selected = [name for group in groups for name in group]
    length(selected) == length(authored) && Set(selected) == Set(authored) ||
        throw(ArgumentError("blocks must include each free scalar integer exactly once: " *
            "$(Tuple(authored)); integer arrays are updated as their own blocks"))

    # Record individual prior/Jacobian events, so independent scalar priors
    # are not grouped into a single aggregate before selecting conditionals.
    sites = _DensitySite[]
    def = Base.invokelatest(_kernel_expr_latest, plan, layout; density_sites=sites)
    def = _elide_named_location_aliases(plan, def)
    def = _pointwise_arithmetic_plates(plan, def)
    def = _name_plate_cells(def, _location_value_aliases(plan, def))
    # As `kernel_expr` does: marginalized integers are summed out of `prior`
    # and `likelihood` by one runtime plate over their joint states.
    isempty(plan.marginalized_integers) || (def = _integer_marginal_program(plan, def))
    statements = Expr[s for s in def.args[2].args if s isa Expr && s.head !== :return]
    terms = Any[]
    for name in (:prior, :likelihood, :log_jacobian)
        st = only(s for s in statements if _assigned_name(s) === name)
        append!(terms, _gibbs_sum_terms(st.args[2]))
    end
    discrete = _gibbs_state(discrete)
    exemplar = (u, discrete)
    continuous = if layout.total == 0
        nothing
    else
        dependent = _gibbs_dependents(statements, (:unconstrained,))
        selected_terms = filter(t -> _gibbs_reads(t, dependent), terms)
        body = Expr(:block, statements...,
            :(_ppl_gibbs_density::Float64 = $(_sum_terms(selected_terms))),
            :(return _ppl_gibbs_density))
        spec = _eval_kernel_def(Expr(:(=), def.args[1], body))
        _register_packed_check!(spec, layout)
        kern = Base.invokelatest(_prepare_cached, plan, spec;
            have=_query_have(plan), want=:_ppl_gibbs_density,
            bound=_query_bound(plan), exemplars=exemplar, on_error)
        prep = Base.invokelatest(prepare_ad, kern, backend, exemplar...; active=:unconstrained)
        SamplerQuery(kern, prep, layout, deepcopy(discrete))
    end
    integers = map(groups) do group
        indices = [only(findall(==(name), authored)) for name in group]
        parameters = ps[indices]
        names = [p.name for p in parameters]
        dependent = _gibbs_dependents(statements, names)
        selected_terms = filter(t -> _gibbs_reads(t, dependent), terms)
        supports = [_integer_support_expr(p) for p in parameters]
        needed = Set{Symbol}()
        for ex in (selected_terms..., supports...)
            _statement_value_reads!(needed, ex)
        end
        _gibbs_ancestors!(needed, statements; stop=Set(names))
        outer, cell = Expr[], Expr[]
        for st in statements
            name = _assigned_name(st)
            name in names && continue # supplied by the support's current state
            name in needed || continue
            push!(name in dependent ? cell : outer, st)
        end
        prefix = Expr[:($(name) = _ppl_gibbs_state[$i]) for (i, name) in enumerate(names)]
        push!(outer, :(_ppl_gibbs_states = collect(Iterators.product($(supports...)))))
        body = Expr(:block, LineNumberNode(0, :generator), prefix..., cell...,
            _sum_terms(selected_terms))
        loop = Expr(:do, Expr(:call, :plate, :_ppl_gibbs_states),
            Expr(:->, Expr(:tuple, :_ppl_gibbs_state), body))
        push!(outer, :(_ppl_gibbs_logweights = $loop))
        push!(outer, :(_ppl_gibbs_conditional = (;
            states=_ppl_gibbs_states, logweights=_ppl_gibbs_logweights)))
        push!(outer, :(return _ppl_gibbs_conditional))
        spec = _eval_kernel_def(Expr(:(=), def.args[1], Expr(:block, outer...)))
        _register_packed_check!(spec, layout)
        kernel = Base.invokelatest(_prepare_cached, plan, spec;
            have=_query_have(plan), want=:_ppl_gibbs_conditional,
            bound=_query_bound(plan), exemplars=exemplar, on_error)
        (; names=group, paths=Tuple(paths[indices]), kernel)
    end
    arrays = map(p -> _gibbs_array_block(plan, p, def, statements, terms, layout,
        exemplar, on_error), array_ps)
    joint = prepare_query(built, plan, :sampler; exemplars=exemplar, on_error)
    return GibbsProblem(layout, continuous, (integers..., arrays...), joint)
end

"""
One declared integer array updated as a block of a Gibbs sweep.

When every density term that reads the array is a sum over plate cells, each
reading the array only at its own element, the elements are conditionally
independent given everything else. All of them are then drawn together and
exactly: for each support value `s`, one evaluation of those cells with every
element set to `s` gives each element's log weight for `s`. Otherwise the
elements are drawn one after another from their exact conditionals, each
evaluating the terms that read the array.

The elements share one support, or each has its own when a support bound is
read per element (`Binomial.(n, p)` with a vector `n`). Then the elementwise
update evaluates each value of the union of the supports once: an element
outside its own support at that value keeps an in-support value in the
evaluation, and its weight for the value is zero.
"""
struct GibbsArrayBlock{C,S}
    name::Symbol
    path::Tuple
    elementwise::Bool
    cells::C       # (u, discrete) -> per-element log weights at that integer
                   # state (elementwise), or the summed terms (one site at a time)
    support::S     # (u, discrete) -> one shared range, or an array of ranges,
                   # one per element
end

# Names whose value reads the integer array `root` elementwise: entry `i` reads
# `root` only at `i`. They are `root`, a gather `A[x]` from a value `A` that
# does not read `root`, dotted broadcasts of such values with values that do
# not read `root`, and plates over such operands whose cells capture nothing
# else that reads it.
function _gibbs_elementwise(statements, root::Symbol, dependent)
    elementwise = Set{Symbol}([root])
    positions = _gibbs_position_names(statements)
    for st in statements
        name = _assigned_name(st)
        (name === nothing || name === root || name ∉ dependent) && continue
        _gibbs_elementwise_expr(st.args[2], elementwise, dependent, positions) &&
            push!(elementwise, name)
    end
    return elementwise
end
# Names the program binds to every position of a vector, in order.
_gibbs_position_names(statements) = Set{Symbol}(_assigned_name(st) for st in statements
    if _assigned_name(st) !== nothing && _gibbs_all_positions(st.args[2]))
function _gibbs_elementwise_expr(ex, elementwise, dependent, positions=Set{Symbol}())
    ex isa Symbol && return ex in elementwise
    ex isa Expr || return false
    free(a) = !_gibbs_reads(a, dependent)
    item(a) = free(a) || _gibbs_elementwise_expr(a, elementwise, dependent, positions)
    all_positions(a) = a isa Symbol ? a in positions : _gibbs_all_positions(a)
    if ex.head === :ref && length(ex.args) == 2
        return free(ex.args[1]) && ex.args[2] isa Symbol && ex.args[2] in elementwise
    elseif ex.head === :. && length(ex.args) == 2 && Meta.isexpr(ex.args[2], :tuple)
        return free(ex.args[1]) && all(item, ex.args[2].args)
    elseif ex.head === :call && !isempty(ex.args) && ex.args[1] isa Symbol &&
            startswith(string(ex.args[1]), ".")
        return all(item, ex.args[2:end])
    elseif _gibbs_plate_parts(ex) !== nothing
        operands, captured = _gibbs_plate_parts(ex)
        isempty(intersect(captured, dependent)) && all(item, operands) &&
            any(a -> !free(a), operands) && return true
        # A loop's cells over every position (`@plate for i in eachindex(y)`)
        # reading the array only at their own position (`mu[z[i]]`): cell `i`
        # holds element `i`.
        params = ex.args[2].args[1]
        param = params isa Symbol ? params :
            (Meta.isexpr(params, :tuple, 1) ? params.args[1] : nothing)
        return length(operands) == 1 && param isa Symbol &&
            all_positions(operands[1]) &&
            _gibbs_reads(ex.args[2].args[2], dependent) &&
            _gibbs_reads_own(ex.args[2].args[2], param, elementwise, dependent)
    elseif ex.head === :call && length(ex.args) == 3 &&
            _gibbs_callee(ex.args[1], :_ppl_range_values)
        # A response's selection of a location at every position, in order.
        return item(ex.args[2]) && free(ex.args[3]) && all_positions(ex.args[3])
    end
    return false
end

_gibbs_callee(f, name::Symbol) = f === name || (f isa GlobalRef && f.name === name) ||
    (Meta.isexpr(f, :., 2) && f.args[2] == QuoteNode(name))
# Every position of a vector in order: `collect(eachindex(x))`,
# `eachindex(x)` or `collect(axes(x, 1))`, as the plate and range desugars
# spell a loop over all of `x`.
function _gibbs_all_positions(ex)
    Meta.isexpr(ex, :call) || return false
    _gibbs_callee(ex.args[1], :collect) && length(ex.args) == 2 &&
        return _gibbs_all_positions(ex.args[2])
    _gibbs_callee(ex.args[1], :eachindex) && length(ex.args) == 2 && return true
    return _gibbs_callee(ex.args[1], :axes) && length(ex.args) == 3 && ex.args[3] == 1
end
# The cell body reads names that depend on the array only as `a[param]`,
# with `a` an elementwise value.
function _gibbs_reads_own(ex, param, elementwise, dependent)
    ex isa Symbol && return ex ∉ dependent
    ex isa Expr || return true
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] isa Symbol &&
            ex.args[1] in dependent
        return ex.args[1] in elementwise && ex.args[2] === param
    end
    return all(a -> _gibbs_reads_own(a, param, elementwise, dependent), ex.args)
end

# `(operands, captured)` of `plate(operands...) do cells...; body end`, or
# `nothing` for any other expression (including keyword operands).
function _gibbs_plate_parts(ex)
    Meta.isexpr(ex, :do) && Meta.isexpr(ex.args[1], :call) &&
        ex.args[1].args[1] === :plate && Meta.isexpr(ex.args[2], :->) || return nothing
    operands = ex.args[1].args[2:end]
    any(a -> a isa Expr && a.head in (:parameters, :kw), operands) && return nothing
    params = ex.args[2].args[1]
    params = params isa Symbol ? Symbol[params] : Symbol[a for a in params.args if a isa Symbol]
    captured = setdiff(_statement_value_reads!(Set{Symbol}(), ex.args[2].args[2]), params)
    return operands, captured
end

# The plate a density term sums when its cells read the array elementwise: the
# term is `sum(P)`, `P` is a plate whose operands read the array only through
# elementwise names, and whose cell body captures nothing that reads it.
function _gibbs_elementwise_plate(statements, term, elementwise, dependent)
    term isa Symbol || return nothing
    st = _named_statement(statements, term)
    st === nothing && return nothing
    rhs = st.args[2]
    Meta.isexpr(rhs, :call, 2) && rhs.args[1] === :sum && rhs.args[2] isa Symbol ||
        return nothing
    pst = _named_statement(statements, rhs.args[2])
    pst === nothing && return nothing
    _gibbs_plate_parts(pst.args[2]) === nothing && return nothing
    _gibbs_elementwise_expr(pst.args[2], elementwise, dependent,
        _gibbs_position_names(statements)) || return nothing
    return rhs.args[2]
end

function _gibbs_array_block(plan, p, def, statements, terms, layout, exemplar, on_error)
    path = get(layout.name_paths, p.name, (p.name,))
    at = findfirst(st -> _assigned_name(st) === p.name, statements)
    at === nothing && throw(ArgumentError(
        "prepare_gibbs found no program value for the integer array $(p.name)"))
    dependent = _gibbs_dependents(statements, (p.name,))
    selected = filter(t -> _gibbs_reads(t, dependent), terms)
    elementwise = _gibbs_elementwise(statements, p.name, dependent)
    plates = [_gibbs_elementwise_plate(statements, t, elementwise, dependent) for t in selected]
    separable = !isempty(plates) && all(!isnothing, plates)
    support = _integer_support_expr(p)
    bound, have = _query_bound(plan), _query_have(plan)
    if separable
        # Each cell holds its own element's log weight at the integer state it
        # is given; the draw gives every element one support value at a time.
        body = Expr(:block, statements...,
            :(_ppl_gibbs_cells = $(GlobalRef(@__MODULE__, :_gibbs_cell_sum))($(plates...))),
            :(_ppl_gibbs_support = $support),
            :(return _ppl_gibbs_cells))
        spec = _eval_kernel_def(Expr(:(=), def.args[1], body))
        _register_packed_check!(spec, layout)
        vectorized(want) = Base.invokelatest(_prepare_cached, plan, spec; have, want,
            bound, exemplars=exemplar, on_error)
        return GibbsArrayBlock(p.name, path, true, vectorized(:_ppl_gibbs_cells),
            vectorized(:_ppl_gibbs_support))
    end
    body = Expr(:block, statements...,
        :(_ppl_gibbs_density::Float64 = $(_sum_terms(selected))),
        :(_ppl_gibbs_support = $support),
        :(return _ppl_gibbs_density))
    spec = _eval_kernel_def(Expr(:(=), def.args[1], body))
    _register_packed_check!(spec, layout)
    site(want) = Base.invokelatest(_prepare_cached, plan, spec; have, want, bound,
        exemplars=exemplar, on_error)
    return GibbsArrayBlock(p.name, path, false, site(:_ppl_gibbs_density),
        site(:_ppl_gibbs_support))
end

@traceable function _gibbs_cell_sum(first, rest...)
    total = collect(Float64, vec(first))
    for cells in rest
        length(cells) == length(total) || throw(ArgumentError(
            "Gibbs integer array: density cells of different lengths"))
        total .+= vec(cells)
    end
    return total
end

_gibbs_lookup(state::NamedTuple, path::Tuple) = isempty(Base.tail(path)) ?
    state[first(path)] : _gibbs_lookup(state[first(path)], Base.tail(path))

# Each element's support: one shared range, or the element's own range when a
# support bound is read per element.
function _gibbs_element_supports(block::GibbsArrayBlock, support, n)
    if support isa AbstractArray{<:AbstractRange}
        length(support) == n || throw(DimensionMismatch("Gibbs integer array " *
            "$(block.name): $(length(support)) supports for $n elements"))
        supports = vec(support)
        i = findfirst(isempty, supports)
        i === nothing || throw(ArgumentError(
            "Gibbs block $(block.name) has empty support at element $i"))
        return supports
    end
    isempty(support) && throw(ArgumentError("Gibbs block $(block.name) has empty support"))
    return support isa AbstractRange{Int} ? support : collect(Int, support)
end
_gibbs_element_support(support::AbstractVector{Int}, i) = support
_gibbs_element_support(supports::AbstractVector{<:AbstractRange}, i) = supports[i]
# The values the elementwise update evaluates: the shared support, or every
# value some element's own support holds.
_gibbs_values(support::AbstractVector{Int}) = support
_gibbs_values(supports::AbstractVector{<:AbstractRange}) = sort!(unique!(
    collect(Int, Iterators.flatten(supports))))

# The block's per-element supports at the current state.
function _gibbs_block_supports(block::GibbsArrayBlock, position, values)
    n = length(_gibbs_lookup(values, block.path))
    return _gibbs_element_supports(block,
        Base.invokelatest(block.support, position, values), n)
end

# Every element's exact conditional log weights when the elements are
# conditionally independent: `(candidates, L)` with `L[i, j]` element `i`'s log
# weight for `candidates[j]`, −Inf outside the element's own support. Each
# candidate costs one evaluation of the block's cells.
_gibbs_elementwise_logweights(block::GibbsArrayBlock, position, values) =
    _gibbs_elementwise_logweights(block, position, values,
        _gibbs_block_supports(block, position, values))
function _gibbs_elementwise_logweights(block::GibbsArrayBlock, position, values, support)
    current = _gibbs_lookup(values, block.path)
    n = length(current)
    candidates = _gibbs_values(support)
    logweights = Matrix{Float64}(undef, n, length(candidates))
    state = similar(current, Int)
    for (j, s) in enumerate(candidates)
        _gibbs_candidate_state!(state, support, s)
        cells = Base.invokelatest(block.cells, position,
            _gibbs_set(values, block.path, state))
        length(cells) == n || throw(ArgumentError("Gibbs integer array " *
            "$(block.name): $(length(cells)) density cells for $n elements"))
        _gibbs_store_cells!(logweights, j, cells, support, s)
    end
    return candidates, logweights
end
# Every element at `s`; with own supports, an element outside its own keeps an
# in-support value, so no read sees an excluded state.
_gibbs_candidate_state!(state, ::AbstractVector{Int}, s) = fill!(state, s)
function _gibbs_candidate_state!(state, supports::AbstractVector{<:AbstractRange}, s)
    for i in eachindex(state, supports)
        state[i] = s in supports[i] ? s : first(supports[i])
    end
    return state
end
# Column `j` of the log weights from one evaluation's cells; an element whose
# own support excludes `s` has weight zero there.
function _gibbs_store_cells!(logweights, j, cells, ::AbstractVector{Int}, s)
    logweights[:, j] .= vec(cells)
    return logweights
end
function _gibbs_store_cells!(logweights, j, cells::AbstractArray,
        supports::AbstractVector{<:AbstractRange}, s)
    for (i, c) in zip(eachindex(supports), vec(cells))
        logweights[i, j] = s in supports[i] ? c : -Inf
    end
    return logweights
end

# Element `i`'s exact conditional log weights over its own support `own`,
# the other elements fixed at their values in `draw` (which ends at `own`'s
# last value).
function _gibbs_site_logweights!(logweights, block::GibbsArrayBlock, position,
        values, draw, i, own)
    for (j, s) in enumerate(own)
        draw[i] = s
        logweights[j] = Base.invokelatest(block.cells, position,
            _gibbs_set(values, block.path, draw))
    end
    return logweights
end

# One exact draw of every element of an integer array block.
_draw_array_block(block::GibbsArrayBlock, rng, position, values) =
    _draw_array_block(block, rng, position, values,
        _gibbs_block_supports(block, position, values))
function _draw_array_block(block::GibbsArrayBlock, rng, position, values, support)
    current = _gibbs_lookup(values, block.path)
    n = length(current)
    if block.elementwise
        candidates, logweights = _gibbs_elementwise_logweights(block, position,
            values, support)
        # One uniform per element, in element order, as a draw per element.
        indices = _membership_draws(rng, _gibbs_probabilities!(logweights))
        draw = similar(current, Int)
        for i in 1:n
            draw[i] = candidates[indices[i]]
        end
        return _gibbs_set(values, block.path, draw)
    end
    draw = reshape(collect(Int, current), size(current))
    buffer = Vector{Float64}(undef,
        maximum(i -> length(_gibbs_element_support(support, i)), 1:n))
    for i in 1:n
        own = _gibbs_element_support(support, i)
        logweights = _gibbs_site_logweights!(view(buffer, 1:length(own)), block,
            position, values, draw, i, own)
        draw[i] = own[_integer_draw_index(rng, _gibbs_weights(logweights))]
    end
    return _gibbs_set(values, block.path, draw)
end

# Only split the addition tree the PPL emitter uses for density aggregates.
# A distribution's own internals remain an indivisible density term.
_gibbs_sum_terms(x) = Meta.isexpr(x, :call) && x.args[1] === :+ ?
    reduce(vcat, _gibbs_sum_terms.(x.args[2:end]); init=Any[]) : Any[x]
_gibbs_reads(x, names) = !isempty(intersect(_statement_value_reads!(Set{Symbol}(), x), names))
function _gibbs_dependents(statements, roots)
    dependent = Set{Symbol}(roots)
    while true
        before = length(dependent)
        for st in statements
            name = _assigned_name(st)
            name === nothing || !_gibbs_reads(st.args[2], dependent) || push!(dependent, name)
        end
        length(dependent) == before && return dependent
    end
end
function _gibbs_ancestors!(needed, statements; stop=Set{Symbol}())
    while true
        before = length(needed)
        for st in statements
            name = _assigned_name(st)
            name in needed && name ∉ stop && _statement_value_reads!(needed, st.args[2])
        end
        length(needed) == before && return needed
    end
end

function _gibbs_set(state::NamedTuple, path::Tuple, value)
    key, rest = first(path), Base.tail(path)
    updated = isempty(rest) ? convert(typeof(state[key]), value) :
        _gibbs_set(state[key], rest, value)
    return merge(state, NamedTuple{(key,)}((updated,)))
end
_gibbs_state(state::NamedTuple) = map(_gibbs_state, state)
_gibbs_state(value::Integer) = Int(value)
_gibbs_state(value) = deepcopy(value)
# Row `i` of `logweights` (elements × values) becomes element `i`'s
# probabilities, in place: the values `_gibbs_weights` gives that row, without
# allocating per element.
function _gibbs_probabilities!(logweights::AbstractMatrix{Float64})
    for i in axes(logweights, 1)
        top = -Inf
        for j in axes(logweights, 2)
            w = logweights[i, j]
            isfinite(w) || w == -Inf || throw(ArgumentError(
                "Gibbs conditional contains NaN or positive infinity"))
            top = max(top, w)
        end
        isfinite(top) || throw(ArgumentError(
            "Gibbs conditional has zero probability at every state"))
        total = 0.0
        for j in axes(logweights, 2)
            w = exp(logweights[i, j] - top)
            logweights[i, j] = w
            total += w
        end
        for j in axes(logweights, 2)
            logweights[i, j] /= total
        end
    end
    return logweights
end

function _gibbs_weights(logweights)
    isempty(logweights) && throw(ArgumentError("Gibbs block has empty support"))
    all(w -> isfinite(w) || w == -Inf, logweights) || throw(ArgumentError(
        "Gibbs conditional contains NaN or positive infinity"))
    top = maximum(logweights)
    isfinite(top) || throw(ArgumentError("Gibbs conditional has zero probability at every state"))
    weights = exp.(vec(logweights) .- top)
    return weights ./ sum(weights)
end

"""
    initialize_gibbs(rng, problem, u0; discrete, stepsize=nothing, metric=nothing,
        max_tree_depth=10, warmup=1000)

Initialize a chain for `prepare_gibbs`. Load `WarmupHMC`, `LogDensityProblems`
and `Random`.

Without `stepsize`, the continuous NUTS kernel is adapted during `warmup`
sweeps. Each sweep draws the integers exactly and then takes one transition of
WarmupHMC's continuously adapting kernel (`ContinuousNUTSState`, started by
Pathfinder from `u0`), which follows the continuous conditional as the
integers change. The chain starts at the last warm-up state, and its kernel
is then fixed: the adapted diagonal scale and step size. `state.warmup`
reports the warm-up sweeps, their divergences and gradient evaluations.

With `stepsize`, that fixed kernel is used from the start and no warm-up runs;
`metric` is then a scale L (covariance L*L'), defaulting to identity. A
`metric` without a `stepsize` is refused. Models without continuous
coordinates run no warm-up.

The state owns copies of the initial position, integer state and RNG, storing
integer parameters as `Int` so a narrow initial integer type does not truncate
the enumerated support. Integer
draws and NUTS use separate reproducible RNG streams. The full joint density
must be finite at the supplied start.
"""
function initialize_gibbs end

"""
    gibbs_step!(state)

Draw each finite integer block exactly in declaration/block order, then take
one WarmupHMC fixed-kernel NUTS transition on all continuous coordinates.
Return `(; state, divergent, stats)`. The state advances in place only after
the complete sweep succeeds. Each continuous transition reevaluates the
target under the new integers. No adaptation occurs in this operation.
"""
function gibbs_step! end

"""
    gibbs_draws(state, n; discard=0)

Advance a Gibbs state through `discard` sweeps, then save `n` sweeps. Returns
owned `positions` (coordinates × draws), `discrete` states, constrained
`parameters`, and per-draw `divergent` flags. Discarding sweeps does not adapt
the fixed NUTS kernel; adaptation happens only in `initialize_gibbs`.
Further steps cannot mutate saved draws.
"""
function gibbs_draws end
