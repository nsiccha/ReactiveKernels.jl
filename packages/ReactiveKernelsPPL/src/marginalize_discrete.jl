# Exact finite-support elimination. The support is an RK plate axis, so data
# determine the iteration count at execution, never the emitted body count.
#
# The integer declarations `marginalize` may name: a scalar `k ~ law`, a
# declared array `z[ax] .~ law` (or `law.(…)`), and the integers of a `@plate`
# loop, `z[i] ~ law`.
function _integer_marginal_names(ast, requested)
    declarations = Set{Symbol}()
    _integer_declarations!(declarations, ast)
    return Symbol[name for name in requested if name in declarations]
end
function _integer_declarations!(names, ex)
    ex isa Expr || return names
    if Meta.isexpr(ex, :call, 3) && ex.args[1] in (:~, :.~)
        lhs, rhs = ex.args[2:3]
        name = lhs isa Symbol && ex.args[1] === :~ ? lhs :
            Meta.isexpr(lhs, :ref) && lhs.args[1] isa Symbol ? lhs.args[1] : nothing
        name === nothing || !_integer_law(rhs) || push!(names, name)
        return names
    end
    foreach(a -> _integer_declarations!(names, a), ex.args)
    return names
end
# A law call `f(…)` or its broadcast `f.(…)`, with the head it calls.
_law_call(ex) = Meta.isexpr(ex, :call) && !isempty(ex.args) ? (first(ex.args), ex.args[2:end]) :
    Meta.isexpr(ex, :., 2) && Meta.isexpr(ex.args[2], :tuple) ? (ex.args[1], ex.args[2].args) :
    nothing
function _integer_law(rhs)
    call = _law_call(rhs)
    call === nothing && return false
    head, args = call
    if head in (:restricted, :truncated) && !isempty(args)
        call = _law_call(first(args))
        call === nothing && return false
        head = first(call)
    end
    family = get(_PARAM_FAMILIES, head, nothing)
    return get(SAMPLED_SUPPORT, family, nothing) === :integer
end

function _integer_declaration(plan, name)
    for ps in (plan.parameters, plan.plate_parameters, plan.array_parameters), p in ps
        p.name === name && return p
    end
    throw(ArgumentError("marginalize(`$name`): `$name` is observed, so it has no " *
        "integer state to sum out"))
end
_integer_array_parameter(p) = p isa Union{ArrayParameter,PlateParameter}

function _integer_marginal_program(plan, def)
    names = plan.marginalized_integers
    any(n -> n in plan.conditioned, names) && throw(ArgumentError(
        "a conditioned integer parameter cannot also be marginalized"))
    declared = [_integer_declaration(plan, name) for name in names]
    # Each array sums out element by element first; scalars then sum the
    # resulting program over their joint states.
    arrays = filter(_integer_array_parameter, declared)
    for p in arrays
        def = _integer_array_marginal_program(plan, p, def)
    end
    isempty(arrays) || (def = _integer_elements_record(plan, arrays, def))
    ps = filter(!_integer_array_parameter, declared)
    isempty(ps) && return _integer_array_latents(def)
    names = Symbol[p.name for p in ps]
    statements = Expr[s for s in def.args[2].args if s isa Expr && s.head !== :return]
    # Graph dependence, not source order: an assignment may precede a value
    # the planner supplies it. Keep every independent value as an outer node.
    dependent = Set(names)
    while true
        before = length(dependent)
        for st in statements
            name = _assigned_name(st)
            name === nothing && continue
            reads = _statement_value_reads!(Set{Symbol}(), st.args[2])
            isempty(intersect(reads, dependent)) || push!(dependent, name)
        end
        length(dependent) == before && break
    end
    :log_jacobian in dependent && throw(ArgumentError(
        "marginalize cannot yet eliminate an integer that changes a continuous " *
        "parameter's transform: the same real coordinates would represent " *
        "different constrained points in different integer states"))
    outer, cell = Expr[], Expr[]
    for st in statements
        name = _assigned_name(st)
        name === :posterior && continue
        # Recover these outputs separately, conditional on one joint state.
        # In particular an existing mixture/HMM posterior must not be lost or
        # leave a second definition of the public latents port behind.
        name in (:_ppl_predictive, :_ppl_latents) && continue
        # Element posteriors of summed-out arrays that read these integers
        # are recovered the same way, one record per joint state.
        name === :_ppl_integer_elements && name in dependent && continue
        if name in dependent || name in (:prior, :likelihood, :pointwise)
            push!(cell, st)
        else
            push!(outer, st)
        end
    end
    supports = Symbol[Symbol(:_ppl_support_, n) for n in names]
    for (p, support) in zip(ps, supports)
        rhs = _integer_support_expr(p)
        isempty(intersect(_statement_value_reads!(Set{Symbol}(), rhs), dependent)) ||
            throw(ArgumentError("marginalize needs data-bounded support for `$(p.name)`"))
        push!(outer, :($support = $rhs))
    end
    # The product's rank is the number of declarations; its extents are data.
    push!(outer, :(_ppl_integer_states = collect(Iterators.product($(supports...)))))
    prefix = Expr[:($(n) = _ppl_integer_state[$i]) for (i, n) in enumerate(names)]
    push!(cell, :((; prior, likelihood, pointwise)))
    body = Expr(:block, LineNumberNode(0, :generator), prefix..., cell...)
    loop = Expr(:do, Expr(:call, :plate, :_ppl_integer_states),
        Expr(:->, Expr(:tuple, :_ppl_integer_state), body))
    push!(outer, :(_ppl_integer_scores = $loop))
    push!(outer, :(_ppl_integer_priors = plate(_ppl_integer_scores) do score
        score.prior
    end))
    push!(outer, :(_ppl_integer_weights = plate(_ppl_integer_scores) do score
        score.prior + score.likelihood
    end))
    lse = GlobalRef(LogExpFunctions, :logsumexp)
    push!(outer, :(prior = $lse(_ppl_integer_priors)))
    push!(outer, :(_ppl_integer_joint = $lse(_ppl_integer_weights)))
    push!(outer, :(likelihood = if prior == -Inf
        -Inf
    else
        _ppl_integer_joint - prior
    end))
    push!(outer, :(pointwise = $(GlobalRef(@__MODULE__, :_integer_pointwise))(
        _ppl_integer_scores, _ppl_integer_priors)))
    # The joint state table is necessary when several eliminated parameters
    # remain dependent: drawing their separate marginals would be incorrect.
    push!(outer, :(_ppl_integer_probabilities = exp.(_ppl_integer_weights .- _ppl_integer_joint)))
    # Names belong in the NamedTuple type, not symbol-valued graph leaves.
    # Numeric state columns can then be stacked by the ordinary vectorizer.
    columns = Pair{Symbol,Symbol}[]
    for (i, name) in enumerate(names)
        column = Symbol(:_ppl_integer_values_, name)
        push!(outer, :($column = plate(_ppl_integer_states) do state
            state[$i]
        end))
        push!(columns, name => column)
    end
    paths = _scope_name_paths(plan.submodel_scopes)
    values = _integer_columns_expr(_scope_namedtuple(Any[
        (get(paths, n, (n,)), v) for (n, v) in columns]))
    elements = any(st -> _assigned_name(st) === :_ppl_integer_elements, statements)
    record = elements ?
        :((; values = $values, probabilities = _ppl_integer_probabilities,
            elements = _ppl_integer_elements)) :
        :((; values = $values, probabilities = _ppl_integer_probabilities))
    push!(outer, :(_ppl_latents = (; parameters = $record)))
    for target in (:_ppl_predictive, :_ppl_latents, :_ppl_integer_elements)
        any(st -> _assigned_name(st) === target, statements) || continue
        target === :_ppl_integer_elements && target ∉ dependent && continue
        conditional = Symbol(target, :_conditional)
        needed = Set([target])
        while true
            before = length(needed)
            for st in statements
                _assigned_name(st) in needed || continue
                _statement_value_reads!(needed, st.args[2])
            end
            length(needed) == before && break
        end
        selected = filter(st -> _assigned_name(st) in needed &&
            (_assigned_name(st) in dependent || _assigned_name(st) === target), statements)
        conditional_body = Expr(:block, LineNumberNode(0, :generator),
            prefix..., selected..., target)
        conditional_loop = Expr(:do, Expr(:call, :plate, :(vec(_ppl_integer_states))),
            Expr(:->, Expr(:tuple, :_ppl_integer_state), conditional_body))
        push!(outer, :($conditional = $conditional_loop))
        columns = Expr(:call, GlobalRef(@__MODULE__, :_integer_conditional_columns), conditional)
        if target === :_ppl_predictive
            push!(outer, :(_ppl_predictive = (;
                _integer_mixture = (; probabilities = vec(_ppl_integer_probabilities),
                    conditional = $columns))))
        elseif target === :_ppl_integer_elements
            # A trailing joint-state axis on every element record.
            push!(outer, :(_ppl_integer_elements = $columns))
        else
            # Replace the integer-only record, rather than defining its port twice.
            popat!(outer, findfirst(st -> _assigned_name(st) === :_ppl_latents, outer))
            push!(outer, :(_ppl_latents = (; parameters = $record, _conditional = $columns)))
        end
    end
    push!(outer, :(posterior = _ppl_integer_joint + log_jacobian))
    push!(outer, :(return posterior))
    return Expr(:(=), def.args[1], Expr(:block, outer...))
end

# ── Integer arrays: one sum per element ──────────────────────────────────
#
# An integer array `z` (a declaration `z[ax] .~ law`, or the integers of a
# `@plate` loop, `z[i] ~ law`) whose density terms each sum plate cells that
# read `z` only at their own element has conditionally independent elements:
# the structure `prepare_gibbs` draws every element from at once
# (`_gibbs_elementwise_plate`). It then sums out element by element,
# `Σ_i logsumexp_s (prior_i(s) + likelihood_i(s))`, in time linear in the
# number of elements and in the support size. Each element's plate cell
# reduces its support scores locally. Both axes have runtime lengths.
# Elements that couple (a term reading `sum(z)`, a recurrence) would need the
# joint support, exponential in the number of elements, and are refused.

_integer_array_fail(x, why) = throw(ArgumentError("marginalize(`$x`) sums an integer " *
    "array out element by element, which needs every density term that reads `$x` " *
    "to sum cells that each read `$x` only at their own element; " * why))

# The authored statement behind a generated density term, where its name says.
function _integer_array_term_label(role, term)
    name = string(term)
    m = match(r"^_ppl_lik_(.+)_resp$", name)
    m === nothing || return "the likelihood of `$(m.captures[1])`"
    m = match(r"^_ppl_prior_(.+)$", name)
    m === nothing || return "the prior of `$(m.captures[1])`"
    return "the $role term `$term`"
end

# A marginalized coordinate packs no entries in the layout. Its axes still
# come from its declaration, including an observation used only for its size.
_integer_array_axes(plan, p::ArrayParameter) = map(_array_axis_read, p.dims)
function _integer_array_axes(plan, p::PlateParameter)
    range = p.range
    range isa UnitRange && return (length(range),)
    range isa Symbol && return (:($(GlobalRef(Base, :length))($range)),)
    range isa Expr && return (_read_length(range),)
    return (_rows_source(_value_rows(plan, p.name), _value_rows_anchors(plan, p.name)),)
end

# Inline a density plate at a single element. Independent broadcast operands
# become lanes of the outer plate. A nested gather keeps its table shared and
# uses the candidate as the index. A loop cell's explicit z[i] uses the same
# candidate. _gibbs_elementwise_plate has already proved these accesses local.
function _integer_cell_expr(ex, dependent, elementwise, localvalue, operand)
    scalar(a) = _integer_cell_expr(a, dependent, elementwise, localvalue, operand)
    if !_gibbs_reads(ex, dependent)
        return operand(ex)
    elseif ex isa Symbol
        return localvalue(ex)
    elseif _gibbs_plate_parts(ex) !== nothing
        args, lambda = ex.args[1].args[2:end], ex.args[2]
        params = lambda.args[1] isa Symbol ? (lambda.args[1],) : lambda.args[1].args
        body = lambda.args[2]
        if length(params) == 1
            for name in elementwise
                body = _substitute_expr(body, :($name[$(first(params))]), localvalue(name))
            end
        end
        for (param, arg) in zip(params, args)
            body = _substitute_name(body, param, scalar(arg))
        end
        return body
    elseif ex.head === :ref
        return Expr(:ref, ex.args[1], scalar(ex.args[2]))
    elseif ex.head === :.
        return Expr(:call, ex.args[1], map(scalar, ex.args[2].args)...)
    elseif _gibbs_callee(ex.args[1], :_ppl_range_values)
        return scalar(ex.args[2])
    end
    return Expr(:call, Symbol(chopprefix(string(ex.args[1]), ".")),
        map(scalar, ex.args[2:end])...)
end

# Build only the selected cut. A density needs a scalar log-sum-exp per
# element. Recovery also needs the vector of scores on that element.
function _integer_score_plate(x, roots, statements, prior, likelihood, dependent,
        elementwise, dims, support, values, value; probabilities = false)
    terms = [prior; likelihood]
    needed = _gibbs_ancestors!(Set(terms), statements; stop = Set(roots))
    index = Symbol(:_ppl_element_index_, x)
    lanes = Any[:(reshape(1:$(foldl(_read_mul, dims.args)), $(dims.args...)))]
    params = Symbol[index]
    function operand(a)
        a isa Number && return a
        Meta.isexpr(a, :call, 2) && _gibbs_callee(a.args[1], :Ref) && return a.args[2]
        i = findfirst(==(a), lanes)
        if i === nothing
            push!(lanes, a)
            push!(params, Symbol(:_ppl_element_operand_, x, :_, length(lanes)))
            i = length(lanes)
        end
        return params[i]
    end
    localvalue(n) = n in roots ? value : Symbol(:_ppl_element_, x, :_, n)
    scores = Expr[]
    for st in statements
        n = _assigned_name(st)
        n in needed && n in dependent && n ∉ roots || continue
        flat = length(roots) > 1 && _gibbs_reads(st.args[2], Set(roots[2:end]))
        lane(a) = operand(flat && !(a isa Number) ?
            :($(GlobalRef(@__MODULE__, :_integer_shaped_operand))($a, $dims)) : a)
        rhs = _integer_cell_expr(st.args[2], dependent, elementwise, localvalue, lane)
        if Meta.isexpr(rhs, :block)
            # Keep a guard as the RHS of its own recipe. Statements in two
            # inlined plate bodies may originally have had the same local name.
            parts = Any[a for a in rhs.args if !(a isa LineNumberNode)]
            for part in parts[1:end-1]
                localname = _assigned_name(part)
                localname === nothing && continue
                fresh = Symbol(localvalue(n), :_, localname)
                parts = Any[_substitute_name(a, localname, fresh) for a in parts]
            end
            append!(scores, parts[1:end-1])
            rhs = last(parts)
        end
        push!(scores, Expr(:(=), localvalue(n), rhs))
    end
    body = Expr(:block, scores..., _sum_terms(map(localvalue, terms)))
    allowed = GlobalRef(@__MODULE__, :_integer_array_allowed)
    weights = Symbol(:_ppl_element_scores_, x)
    lse = GlobalRef(LogExpFunctions, :logsumexp)
    # For the mass, iterate the element's actual support. Only a recovery
    # record needs the union axis, with zeros where this element lacks a value.
    own = GlobalRef(@__MODULE__, :_integer_element_support)
    domain = probabilities ? values : :($own($support, $index))
    score = probabilities ? :(if $allowed($support, $index, $value)
        $body
    else
        -Inf
    end) : body
    inner_body = probabilities ? Expr(:block, LineNumberNode(0, :generator), score) : score
    inner = Expr(:do, Expr(:call, :plate, domain), Expr(:->, Expr(:tuple, value), inner_body))
    mass = Symbol(:_ppl_element_logmass_, x)
    result = probabilities ? :(plate($weights, $values) do logweight, candidate
        $allowed($support, $index, candidate) ? exp(logweight - $mass) : 0.0
    end) : mass
    outer = Expr(:block, LineNumberNode(0, :generator), :($weights = $inner),
        :($mass = $lse($weights)), result)
    return Expr(:do, Expr(:call, :plate, lanes...), Expr(:->, Expr(:tuple, params...), outer))
end

_integer_shaped_operand(v, dims) = v
_integer_shaped_operand(v::AbstractVector, dims) = length(v) == 1 ? v : reshape(v, dims)

function _integer_array_marginal_program(plan, p, def)
    x = p.name
    statements = Expr[s for s in def.args[2].args if s isa Expr && s.head !== :return]
    extents = _integer_array_axes(plan, p)
    # A multidimensional array is also read through its column-major `vec`,
    # which the layout's transform statements would have defined.
    flat = length(extents) > 1 ? (_array_flat_name(x, length(extents)),) : ()
    roots = (x, flat...)
    dependent = _gibbs_dependents(statements, roots)
    :log_jacobian in dependent && throw(ArgumentError(
        "marginalize cannot yet eliminate an integer that changes a continuous " *
        "parameter's transform: the same real coordinates would represent " *
        "different constrained points in different integer states"))
    :_ppl_latents in dependent && _integer_array_fail(x, "the observation latents " *
        "(a mixture or hidden Markov response) read `$x`, which is not supported yet")
    elementwise = _gibbs_elementwise(statements, roots, dependent)
    plates = Dict(:prior => Symbol[], :likelihood => Symbol[])
    kept = Dict(:prior => Any[], :likelihood => Any[])
    for role in (:prior, :likelihood)
        st = _named_statement(statements, role)
        for term in _gibbs_sum_terms(st.args[2])
            if !_gibbs_reads(term, dependent)
                push!(kept[role], term)
                continue
            end
            plate = _gibbs_elementwise_plate(statements, term, elementwise, dependent)
            plate === nothing && _integer_array_fail(x, _integer_array_term_label(role, term) *
                " reads several of its elements together (as `sum($x)` or a recurrence " *
                "over `$x` does), and summing those out jointly costs exponentially many " *
                "states; keep `$x` explicit (`prepare_gibbs` updates it)")
            push!(plates[role], plate)
        end
    end
    # Pointwise fields holding a likelihood's cells take its sequential
    # factors, element by element, in field order.
    order = Symbol[]
    pointwise_name = Symbol(:_ppl_array_pointwise_, x)
    pointwise = _named_statement(statements, :pointwise)
    pointwise_rhs = pointwise === nothing ? nothing : _integer_array_pointwise_expr!(order,
        pointwise.args[2], plates[:likelihood], dependent, x, pointwise_name)
    likelihoods = Symbol[order..., setdiff(plates[:likelihood], order)...]
    support, values, value = Symbol(:_ppl_support_, x), Symbol(:_ppl_support_values_, x),
        Symbol(:_ppl_support_value_, x)
    rhs = _integer_support_expr(p)
    isempty(intersect(_statement_value_reads!(Set{Symbol}(), rhs), dependent)) ||
        throw(ArgumentError("marginalize needs data-bounded support for `$x`"))
    dims = Expr(:tuple, extents...)
    here = @__MODULE__
    candidate = Expr[:($x = $(GlobalRef(here, :_integer_array_candidate))($support, $value, $dims))]
    foreach(f -> push!(candidate, :($f = vec($x))), flat)
    prior_term, likelihood_term = Symbol(:_ppl_array_prior_, x), Symbol(:_ppl_array_likelihood_, x)
    joint_term = Symbol(:_ppl_array_joint_, x)
    probabilities = Symbol(:_ppl_array_probabilities_, x)
    masses = [Symbol(:_ppl_array_mass_, x, :_, k) for k in 0:length(likelihoods)]
    score(k; kwargs...) = _integer_score_plate(x, roots, statements, plates[:prior],
        likelihoods[1:k], dependent, elementwise, dims, support, values, value; kwargs...)
    added = Expr[:($support = $rhs),
        :($values = $(GlobalRef(here, :_integer_array_values))($support))]
    for k in unique([0:length(order); length(likelihoods)])
        push!(added, :($(masses[k+1]) = $(score(k))))
    end
    push!(added,
        :($prior_term::Float64 = sum($(first(masses)))),
        :($joint_term::Float64 = sum($(last(masses)))),
        :($likelihood_term::Float64 = $prior_term == -Inf ? -Inf : $joint_term - $prior_term))
    rows = Symbol(:_ppl_probability_rows_, x)
    push!(added, :($rows = $(score(length(likelihoods); probabilities = true))),
        :($probabilities = $(GlobalRef(here, :_integer_probability_columns))($rows, $values)),
        :($(Symbol(:_ppl_integer_element_, x)) = (; values = plate($values) do value
            value
        end, probabilities = $probabilities)))
    points = [:(plate($(masses[k+1]), $(masses[k])) do current, previous
        previous == -Inf ? -Inf : current - previous
    end) for k in eachindex(order)]
    isempty(points) || push!(added, :($pointwise_name = ($(points...),)))
    # Values reading the array have no single value once it is summed out;
    # queries of them are refused by name. The density aggregates follow the
    # values they read.
    aggregates = (:prior, :likelihood, :pointwise, :posterior)
    out = Expr[st for st in statements if _assigned_name(st) ∉ dependent &&
        _assigned_name(st) ∉ aggregates]
    append!(out, added)
    if :_ppl_predictive in dependent
        _integer_array_predictive!(out, statements, dependent, roots, candidate, value,
            values, probabilities)
    end
    for st in statements
        name = _assigned_name(st)
        if name === :prior || name === :likelihood
            terms = Any[kept[name]..., name === :prior ? prior_term : likelihood_term]
            push!(out, Expr(:(=), st.args[1], _sum_terms(terms)))
        elseif name === :pointwise
            push!(out, Expr(:(=), st.args[1], pointwise_rhs))
        end
    end
    # The posterior reads the joint mass alone: the prior mass it would add
    # and the likelihood subtract cancel, so a sampler never computes it.
    posterior = _named_statement(statements, :posterior)
    push!(out, Expr(:(=), posterior.args[1],
        _sum_terms(Any[kept[:prior]..., kept[:likelihood]..., joint_term, :log_jacobian])),
        :(return posterior))
    return Expr(:(=), def.args[1], Expr(:block, out...))
end

# Only the responses reading this array need conditional law columns. Keep
# the array's name on those records so a draw can reuse one set of component
# indices across all responses that read it, including after vectorization.
function _integer_array_predictive!(out, statements, dependent, roots, candidate, value,
        values, probabilities)
    x = first(roots)
    fields = _named_statement(statements, :_ppl_predictive).args[2].args
    selected = filter(field -> _gibbs_reads(field.args[2], dependent), fields)
    needed = Set{Symbol}()
    foreach(f -> _statement_value_reads!(needed, f.args[2]), selected)
    _gibbs_ancestors!(needed, statements; stop = Set(roots))
    cells = [st for st in statements if _assigned_name(st) in needed &&
        _assigned_name(st) in dependent && _assigned_name(st) ∉ roots]
    conditional = Symbol(:_ppl_array_laws_, x)
    body = Expr(:block, LineNumberNode(0, :generator), candidate..., cells...,
        Expr(:tuple, selected...))
    loop = Expr(:do, Expr(:call, :plate, values), Expr(:->, Expr(:tuple, value), body))
    columns = GlobalRef(@__MODULE__, :_integer_conditional_columns)
    push!(out, :($conditional = $columns($loop)))
    entries = map(fields) do f
        f in selected || return f
        name = f.args[1]
        record = :((; probabilities = $probabilities, conditional = $conditional.$name))
        Expr(:(=), name, :((; _integer_element = (; $x = $record))))
    end
    push!(out, Expr(:(=), :_ppl_predictive, Expr(:tuple, entries...)))
end

# The pointwise NamedTuple with each field that holds a likelihood plate of
# the array replaced by its element-wise sequential factor; `order` collects
# those plates in field order.
function _integer_array_pointwise_expr!(order, ex, plates, dependent, x, name)
    if Meta.isexpr(ex, :tuple) || Meta.isexpr(ex, :parameters)
        return Expr(ex.head, Any[_integer_array_pointwise_expr!(order, a, plates,
            dependent, x, name) for a in ex.args]...)
    elseif (Meta.isexpr(ex, :(=), 2) || Meta.isexpr(ex, :kw, 2)) && ex.args[1] isa Symbol
        return Expr(ex.head, ex.args[1], _integer_array_pointwise_expr!(order, ex.args[2],
            plates, dependent, x, name))
    elseif ex isa Symbol && ex in plates
        push!(order, ex)
        return :($name[$(length(order))])
    elseif _gibbs_reads(ex, dependent)
        _integer_array_fail(x, "the pointwise value `$ex` reads it otherwise")
    end
    return ex
end

# Every summed-out array's element posteriors, keyed by author path.
function _integer_elements_record(plan, arrays, def)
    paths = _scope_name_paths(plan.submodel_scopes)
    record = _integer_columns_expr(_scope_namedtuple(Any[
        (get(paths, p.name, (p.name,)), Symbol(:_ppl_integer_element_, p.name)) for p in arrays]))
    statements = def.args[2].args
    at = findlast(st -> Meta.isexpr(st, :return), statements)
    insert!(statements, at, :(_ppl_integer_elements = $record))
    return def
end

# With no summed-out scalar, the latent record holds the arrays' element
# posteriors beside any observation latents the program already recovers.
function _integer_array_latents(def)
    statements = def.args[2].args
    at = findfirst(st -> _assigned_name(st) === :_ppl_latents, statements)
    latents = :((; parameters = (; elements = _ppl_integer_elements)))
    if at !== nothing
        st = statements[at]
        statements[at] = Expr(:(=), :_ppl_latents_observed, st.args[2])
        latents = :((; _ppl_latents_observed..., parameters = (; elements = _ppl_integer_elements)))
    end
    insert!(statements, findlast(st -> Meta.isexpr(st, :return), statements),
        :(_ppl_latents = $latents))
    return def
end

# The values an array's cells are evaluated at: its one support, or every
# value some element's own support holds.
_integer_array_values(support::AbstractRange{<:Integer}) = support
_integer_array_values(supports::AbstractArray{<:AbstractRange}) = _gibbs_values(vec(supports))

# Every element at `value`; with own supports, an element outside its own
# keeps an in-support value, so no read sees an excluded state.
_integer_array_candidate(::AbstractRange{<:Integer}, value, dims) = fill(value, dims)
function _integer_array_candidate(supports::AbstractArray{<:AbstractRange}, value, dims)
    length(supports) == prod(dims) || throw(DimensionMismatch("[query] an integer array " *
        "of size $dims has $(length(supports)) element supports"))
    state = Array{Int}(undef, dims)
    for (i, own) in enumerate(supports)
        isempty(own) && throw(ArgumentError(
            "[query] element $i of a marginalized integer array has an empty support"))
        state[i] = value in own ? value : first(own)
    end
    return state
end

_integer_element_support(support::AbstractRange{<:Integer}, i) = support
_integer_element_support(supports::AbstractArray{<:AbstractRange}, i) = supports[i]

_integer_array_allowed(::AbstractRange{<:Integer}, i, value) = true
_integer_array_allowed(supports::AbstractArray{<:AbstractRange}, i, value) = value in supports[i]

# The element axes precede the support axis in a public recovery record.
@traceable function _integer_probability_columns(rows, values)
    dims = (size(rows)..., length(values))
    if isempty(rows)
        zeros(Float64, dims)
    else
        reshape(stack(vec(rows); dims = 1), dims)
    end
end

# RK's existing numeric-tree batching machinery owns the array storage. The
# final axis here selects a joint integer state; vectorization adds a draw axis.
@traceable function _integer_conditional_columns(cells)
    isempty(cells) && throw(ArgumentError(
        "[query] the integer support is empty: conditional outputs are undefined"))
    result = ReactiveKernels._replicated_output!(Ref{Any}(nothing),
        first(cells), length(cells))
    for (index, cell) in enumerate(cells)
        ReactiveKernels._replicated_store!(result, index, cell)
    end
    return result
end

_integer_columns_expr(x) = x
_integer_columns_expr(x::NamedTuple) = Expr(:tuple,
    (Expr(:(=), n, _integer_columns_expr(v)) for (n, v) in pairs(x))...)

# Marginal observations are dependent. Their pointwise terms are sequential
# predictive densities in the existing NamedTuple/column-major order. Updating
# the finite state weights yields that factorization in O(states × observations).
@traceable function _integer_pointwise(scores, prior_weights)
    isempty(scores) && throw(ArgumentError(
        "[query] the integer support is empty: pointwise conditionals are undefined"))
    weights = vec(copy(prior_weights))
    pointwise = map(vec(scores)) do score
        score.pointwise
    end
    return _integer_pointwise_value(pointwise, weights)
end

@traceable _integer_pointwise_fields(values, weights, ::Tuple{}) = ()
@traceable function _integer_pointwise_fields(values, weights, names::Tuple)
    name = first(names)
    column = map(values) do value
        getproperty(value, name)
    end
    field = _integer_pointwise_value(column, weights)
    return (field, _integer_pointwise_fields(values, weights, Base.tail(names))...)
end
@traceable function _integer_pointwise_value(
        values::AbstractArray{<:NamedTuple{names}}, weights) where {names}
    return NamedTuple{names}(_integer_pointwise_fields(values, weights, names))
end
# Pointwise leaves are Float64, as in the numeric-array method below. Their
# container type depends on the declared nesting, not on any data extent.
_integer_pointwise_type(::Type{<:Number}) = Float64
_integer_pointwise_type(::Type{<:AbstractArray{T,N}}) where {T,N} =
    Array{_integer_pointwise_type(T),N}
_integer_pointwise_type(::Type{NamedTuple{names,T}}) where {names,T} =
    NamedTuple{names,Tuple{map(_integer_pointwise_type, fieldtypes(T))...}}
@traceable function _integer_pointwise_value(values::AbstractArray{<:AbstractArray}, weights)
    template = first(values)
    result = similar(template, _integer_pointwise_type(eltype(template)))
    for i in eachindex(template)
        cells = map(values) do value
            value[i]
        end
        result[i] = _integer_pointwise_value(cells, weights)
    end
    return result
end
@traceable function _integer_pointwise_value(
        values::AbstractArray{<:AbstractArray{<:Number}}, weights)
    result = similar(first(values), Float64)
    previous = logsumexp(weights)
    for i in eachindex(result)
        for (s, value) in enumerate(values)
            weights[s] += value[i]
        end
        mass = logsumexp(weights)
        result[i] = previous == -Inf ? -Inf : mass - previous
        previous = mass
    end
    return result
end
@traceable function _integer_pointwise_value(values::AbstractArray{<:Number}, weights)
    previous = logsumexp(weights)
    for (s, value) in enumerate(values)
        weights[s] += value
    end
    mass = logsumexp(weights)
    return previous == -Inf ? -Inf : mass - previous
end
