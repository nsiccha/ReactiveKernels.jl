# Exact finite-support elimination. The support is an RK plate axis, so data
# determine the iteration count at execution, never the emitted body count.
function _integer_marginal_names(ast, requested)
    declarations = Dict{Symbol,Any}()
    for st in ast.args
        Meta.isexpr(st, :call, 3) && st.args[1] === :~ || continue
        lhs, rhs = st.args[2:3]
        lhs isa Symbol && Meta.isexpr(rhs, :call) || continue
        law = rhs
        if first(law.args) in (:restricted, :truncated) && length(law.args) >= 2
            law = law.args[2]
        end
        Meta.isexpr(law, :call) || continue
        family = get(_PARAM_FAMILIES, first(law.args), nothing)
        get(SAMPLED_SUPPORT, family, nothing) === :integer || continue
        declarations[lhs] = rhs
    end
    return Symbol[name for name in requested if haskey(declarations, name)]
end

function _integer_marginal_program(plan, def)
    names = plan.marginalized_integers
    any(n -> n in plan.conditioned, names) && throw(ArgumentError(
        "a conditioned integer parameter cannot also be marginalized"))
    ps = [only(p for p in plan.parameters if p.name === name) for name in names]
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
    push!(outer, :(_ppl_latents = (; parameters = (;
        values = $values, probabilities = _ppl_integer_probabilities))))
    for target in (:_ppl_predictive, :_ppl_latents)
        any(st -> _assigned_name(st) === target, statements) || continue
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
        else
            # Replace the integer-only record, rather than defining its port twice.
            popat!(outer, findfirst(st -> _assigned_name(st) === :_ppl_latents, outer))
            push!(outer, :(_ppl_latents = (; parameters = (;
                values = $values, probabilities = _ppl_integer_probabilities),
                _conditional = $columns)))
        end
    end
    push!(outer, :(posterior = _ppl_integer_joint + log_jacobian))
    push!(outer, :(return posterior))
    return Expr(:(=), def.args[1], Expr(:block, outer...))
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
