# Public numeric views of an observation's distribution object. The field
# names are semantic API; the values are ports of the very object the density
# calls. In particular, link inverses are never reimplemented by this query.
const _OBSERVATION_PARAMETERS = Dict(
    :bernoulli => ((:p,), (:probability => :p, :logit => :logit)),
    :exponential => ((:scale,), (:scale => :scale,)),
    :poisson => ((:rate,), (:rate => :rate,)),
    :binomial => ((:n, :p), (:trials => :n, :probability => :p, :logit => :logit)),
    :negative_binomial => ((:r, :p), (:shape => :r, :probability => :p)),
    :negative_binomial2 => ((:mu, :phi), (:mean => :mu, :dispersion => :phi)),
    :gamma => ((:shape, :rate), (:shape => :shape, :scale => :scale)),
    :weibull => ((:k, :theta), (:shape => :k, :scale => :theta)),
    :inverse_gaussian => ((:mu, :lambda), (:mean => :mu, :shape => :lambda)),
    :beta => ((:a, :b), (:alpha => :a, :beta => :b)),
    :beta_binomial => ((:n, :alpha, :beta),
        (:trials => :n, :alpha => :alpha, :beta => :beta)),
    :zero_inflated_poisson => ((:rate, :zi), (:rate => :rate, :zero_probability => :zi)),
    :zero_inflated_binomial => ((:n, :p, :zi),
        (:trials => :n, :probability => :p, :zero_probability => :zi)),
)

_observation_argument_node(r) = Symbol(:_ppl_obsargs_, r.label)
_observation_tuple(fields) = Expr(:tuple,
    (Expr(:(=), field, value) for (field, value) in fields)...)

# An authored plate needs an array axis. A shared number supplies a singleton
# axis while deriving a port, then recovers its scalar shape. Actual array
# axes, including empty arrays, retain their shape and runtime trip count.
@traceable _ppl_parameter_axis(x::Number) = fill(x, 1)
@traceable _ppl_parameter_axis(x::AbstractArray{T,0}) where {T} = fill(x[], 1)
_ppl_parameter_axis(x) = x
const _ScalarObservationParameter = Union{Number,AbstractArray{<:Any,0}}
@traceable _ppl_parameter_value(x, args::Vararg{_ScalarObservationParameter}) = x[1]
_ppl_parameter_value(x, args...) = x

# A printable structural constructor lets the ordinary graph splicer insert
# the selected port's recipes. Owner ports are not object endpoint methods.
_observation_parameter_graph(name::Symbol, have::Tuple, port::Symbol) =
    ReactiveKernels.extract(getfield(PPLGeneratedModels, name); have, want = port)

function ReactiveKernels._kernel_resolve_graph_constructor(
        ::typeof(_observation_parameter_graph), arguments, mod::Module)
    length(arguments) == 3 || return nothing
    name, have, port = arguments
    name isa QuoteNode && port isa QuoteNode && Meta.isexpr(have, :tuple) || return nothing
    all(a -> a isa QuoteNode && a.value isa Symbol, have.args) || return nothing
    return _observation_parameter_graph(name.value, Tuple(a.value for a in have.args), port.value)
end

function _observation_port(name::Symbol, bindings, port::Symbol)
    haskey(bindings, port) && return bindings[port]
    # Ask the owned graph which bindings this port reads. For example,
    # gamma's scale reads rate alone: a varying shape must not turn that
    # shared scale into one value per observation merely because it occurs
    # in the same constructor. No family-specific inverse formula is needed.
    spec = _observation_parameter_graph(name, Tuple(sort!(collect(keys(bindings)))), port)
    selected = ReactiveKernels.plan(spec)
    graph = selected.graph
    cid(v) = ReactiveKernels.canon_id(graph, v.id)
    needed = ReactiveKernels._needed_have(graph, selected, Set(cid.(selected.have)))
    ids = Set(cid.(needed))
    names = [v.name for v in selected.have if cid(v) in ids]
    callee = Expr(:call, GlobalRef(@__MODULE__, :_observation_parameter_graph),
        QuoteNode(name), Expr(:tuple, QuoteNode.(names)...), QuoteNode(port))
    return Expr(:call, callee, (deepcopy(bindings[n]) for n in names)...)
end

function _observation_parameter_stmts(r::LikelihoodSpec, stmts::Vector{Expr})
    r.family in _OBSERVATION_LOCATION_SCALE_FAMILIES && return Expr[]
    i = findlast(st -> _assigned_name(st) === _pw_name(r.label), stmts)
    i === nothing && return Expr[]
    rhs = stmts[i].args[2]
    _is_plate_do(rhs) || return Expr[]
    lanes = rhs.args[1].args[2:end]
    lambda = rhs.args[2]
    params = Meta.isexpr(lambda.args[1], :tuple) ? lambda.args[1].args : Any[lambda.args[1]]
    length(params) == length(lanes) || return Expr[]
    isempty(params) && return Expr[]
    first(params) isa Symbol || return Expr[]
    body = Expr[a for a in lambda.args[2].args if !(a isa LineNumberNode)]
    isempty(body) && return Expr[]
    locals, final = body[1:end-1], body[end]
    # Weights change the density contribution, not its distribution's
    # parameters. Undo only the wrapper that `_weighted_cell` itself adds.
    if r.weights !== nothing && Meta.isexpr(final, :call, 3) && final.args[1] === :*
        final = final.args[3]
        if final === :_ppl_arm && !isempty(locals)
            _assigned_name(locals[end]) === :_ppl_arm || return Expr[]
            final = pop!(locals).args[2]
        end
    end
    endpoint = _endpoint_density(final, first(params), r.evidence.kind)
    endpoint === nothing && return Expr[]
    fields = _law_fields(endpoint)
    fields === nothing && return Expr[]
    ctor = endpoint.args[1]
    name = ctor isa GlobalRef ? ctor.name : ctor
    schema = get(_OBSERVATION_PARAMETERS, name, nothing)
    schema === nothing && return Expr[]
    positional, public = schema
    bindings = Dict{Symbol,Any}()
    for field in fields[2:end]
        key, value = field.args
        index = findfirst(j -> key === Symbol(:_, j), eachindex(positional))
        bindings[index === nothing ? key : positional[index]] = value
    end
    result, values = Expr[], Pair{Symbol,Any}[]
    for (field, port) in public
        node = Symbol(:_ppl_obsarg_, r.label, :_, field)
        value = _observation_port(name, bindings, port)
        statements, reads = _needed_statements(Expr[locals..., :($node = $value)])
        selected = findall(p -> p in reads, params)
        # A directly supplied argument keeps its original shape and does
        # not need a one-field plate. Constants and shared expressions also
        # stay outside the observation loop.
        direct = findfirst(==(value), params)
        if length(statements) == 1 && direct !== nothing
            push!(result, :($node = $(deepcopy(lanes[direct]))))
        elseif isempty(selected)
            aliases = Dict{Symbol,Symbol}(n => Symbol(node, :_, n)
                for st in statements[1:end-1] for n in _assigned_names(st))
            append!(result, [_hsubst(st, aliases) for st in statements])
        else
            raw = Symbol(node, :_plate)
            operands = deepcopy(lanes[selected])
            axes = [Symbol(node, :_axis_, j) for j in eachindex(selected)]
            for (axis, operand) in zip(axes, operands)
                push!(result, :($axis = $(GlobalRef(@__MODULE__, :_ppl_parameter_axis))($operand)))
            end
            cell = Expr(:block, LineNumberNode(0, :generator), statements..., node)
            plate = Expr(:do, Expr(:call, :plate, axes...),
                Expr(:(->), Expr(:tuple, params[selected]...), cell))
            push!(result, :($raw = $plate),
                :($node = $(GlobalRef(@__MODULE__, :_ppl_parameter_value))($raw, $(operands...))))
        end
        push!(values, field => node)
    end
    push!(result, :($(_observation_argument_node(r)) = $(_observation_tuple(values))))
    return result
end
