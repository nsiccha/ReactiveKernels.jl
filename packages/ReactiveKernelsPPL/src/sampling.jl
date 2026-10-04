"""
    sampling_logdensity(rhs, value)

The constrained-value log density of a sampling RHS. Extend this function on
caller-owned types. A density includes its normalization and support checks;
it must be pure and support ordinary backend AD. Model-sized iteration belongs
in a visible submodel fragment or an authored plate/scan, not an opaque density.
No random generation is performed by this interface.
"""
function sampling_logdensity(rhs, value)
    throw(ArgumentError("sampling RHS $(typeof(rhs)) has no sampling_logdensity method"))
end

"""
    LogDensity(f, args...)

Convenience sampling RHS whose density is `f(value, args...)`. Observed use
requires no geometry or random generator. Parameter use also needs a
[`sampling_geometry`](@ref) method, as for any other external RHS.
"""
struct LogDensity{F,A,K}
    f::F
    args::A
    kwargs::K
    LogDensity(f::F, args::A, kwargs::K, ::Val{:packed}) where {F,A,K} = new{F,A,K}(f, args, kwargs)
end
LogDensity(f, args...; kwargs...) = LogDensity(f, args, (;kwargs...), Val(:packed))
sampling_logdensity(d::LogDensity, value) = d.f(value, d.args...; d.kwargs...)

"""
    ParameterGeometry(shape, unconstrained; support, constrain, unconstrain,
                      logjac, coordinate_names=nothing)

Caller-owned parameter geometry. `shape` is the constrained shape (empty for a
scalar); `unconstrained` is the packed dimension, independently of that shape.
They may contain data-only Julia expressions, evaluated when binding data,
never by evaluating sampled values. `support` describes the support; the density
and inverse must enforce it. Endpoints are ordinary pure Julia callables:
`constrain(u, shape, args...)`, `unconstrain(value, shape, args...)` and
`logjac(u, shape, args...)`. The first returns the constrained value, the second
a packed vector, and the third the log absolute Jacobian. Host and kernel
execution call these same endpoints. `coordinate_names`, when supplied, has
one Symbol per packed coordinate. Draw generation is outside this contract.
"""
struct ParameterGeometry{S,N,P,C,U,J,L}
    shape::S
    unconstrained::N
    support::P
    constrain::C
    unconstrain::U
    logjac::J
    coordinate_names::L
end
function ParameterGeometry(shape, unconstrained; support, constrain, unconstrain,
        logjac, coordinate_names=nothing)
    return ParameterGeometry(Tuple(shape), unconstrained, support, constrain,
        unconstrain, logjac, coordinate_names)
end

"""
    sampling_geometry(constructor, argument_expressions::Tuple, declared_shape::Tuple)

Return `ParameterGeometry` for parameter use of an external RHS. This is a
structural method: its arguments are the authored Julia expressions, not active
parameter values. It must not evaluate the constructor or statistical arguments.
Observed use does not call this method. A future generated-only role can use the
same RHS without acquiring a sampler layout or invoking these transforms.
"""
function sampling_geometry(constructor, args, shape)
    throw(ArgumentError("sampling RHS $constructor needs sampling_geometry for parameter use"))
end

"""
    sampling_fragment(binding) -> RKPPLSubmodel or nothing

Public structural expansion adapter for an external sampling RHS binding.
Return an `RKPPLSubmodel(name, argument_names, body_ast, defining_module)` whose
ordinary sampling/definition/plate/scan statements express the external model.
The body need not originate in `@rkppl`. Return a stable fragment, without
executing the model or mutating a StructuralPlan. Scope, nesting, conditioning
and loop handling are shared with authored RKPPL submodels.
"""
sampling_fragment(binding) = nothing

# Builtin spellings select existing optimized lowerings. Everything else
# follows ordinary module resolution and the open protocol, not an allow-list.
function _external_rhs(rhs)
    rhs isa Union{Symbol,GlobalRef} && return true
    rhs isa Expr || return false
    rhs.head === :. && length(rhs.args) == 2 && rhs.args[2] isa QuoteNode && return true
    head = rhs.head === :call ? first(rhs.args) :
        _is_dotted_call(rhs) ? first(rhs.args) : nothing
    head === nothing && return false
    head isa Symbol || return true
    return !(head in union(keys(_PARAM_FAMILIES), _PANEL_OBS_HEADS, _GLM_HEADS,
        (:BernoulliLogit, :PoissonLog, :BinomialLogit,
         :NegativeBinomialLog, :NegativeBinomial2Log,
        (:HalfNormal, :HalfCauchy, :Flat, :flat, :positive, :truncated, :restricted, :Horseshoe,
         :weighted, :censored, :interval_censored, :Ordered, :Dirichlet,
         :LKJCholesky, :LKJCovarianceFactor, :MixtureModel,
         :Gaussian, :Bernoulli, :Poisson, :Binomial, :NegativeBinomial,
         :NegativeBinomial2, :Gamma, :Beta, :BetaBinomial2, :BetaKappa,
         :CircularVonMises, :Categorical, :CategoricalLogit, :OrderedLogistic, :Ordinal,
         :Multinomial, :MvNormal, :MvNormalCholesky, :VonMises,
         :InverseGaussian, :HurdlePoisson, :ZeroInflatedPoisson,
         :ZeroInflatedBinomial, :normal_id_glm, :bernoulli_logit_glm,
         :poisson_log_glm)))
end

function _external_parameter(name, rhs, shape, mod, names; observed=false,
        broadcast=false)
    bare = rhs isa Union{Symbol,GlobalRef} ||
        (Meta.isexpr(rhs, :., 2) && rhs.args[2] isa QuoteNode)
    resolved = bare ? (rhs isa Symbol && rhs in names ? rhs :
        _resolve_call_head(rhs, mod, names, "sampling RHS $name")) :
        _resolve_module_calls(rhs, mod, names, "sampling RHS $name")
    call = _is_dotted_call(resolved) ?
        Expr(:call, resolved.args[1], resolved.args[2].args...) : resolved
    (bare || call isa Expr && call.head === :call) || _sfail("sampling RHS $name needs a callable or RHS value")
    arguments = bare ? Any[] : call.args[2:end]
    args = Tuple(arguments)
    geometry = if observed
        nothing
    else
        head = bare ? call : call.args[1]
        head isa GlobalRef || _sfail("external RHS $name needs a module-visible constructor")
        g = sampling_geometry(getglobal(head.mod, head.name), deepcopy(args), Tuple(shape))
        g isa ParameterGeometry || _sfail("sampling_geometry for $name must return ParameterGeometry")
        Tuple(shape) == g.shape || _sfail("sampling geometry for $name has shape $(g.shape), " *
            "but its declaration has shape $(Tuple(shape)); declare its constrained dimensions")
        g
    end
    return (rhs=call, geometry=geometry, broadcast=broadcast || _is_dotted_call(rhs))
end

function _validate_external_parameter(plan, p; observation=false)
    p.support_override === nothing || _fail(p.label, "external RHS supplies its own support")
    g = p.args.geometry
    (observation || p.name in plan.conditioned || g isa ParameterGeometry) ||
        _fail(p.label, "external parameter needs ParameterGeometry")
    known = union(_all_names(plan), Set(keys(plan.columns)))
    for ref in _value_symbols(p.args.rhs)
        (ref in known || !isbound(plan)) || _fail(p.label, "external RHS references unknown name $ref")
    end
    return nothing
end

function _sampling_extent(plan, ex, name)
    ex isa Int && ex >= 0 && return ex
    (_is_axis_dim(ex) || _is_levels_dim(ex) || _levels_count(ex) !== nothing) &&
        return _array_dim_size(plan, name, name, ex)
    expressions = Dict(a.name=>a.expr for a in (plan.assignments..., plan.derived...))
    active = Set{Symbol}()
    function lookup(n)
        haskey(plan.columns, n) && return plan.columns[n]
        (haskey(expressions, n) && n ∉ active) || _fail(name,
            "sampling geometry dimensions must depend only on bound data; unavailable $n")
        push!(active, n)
        v = _eval_value_expr(expressions[n], lookup, name)
        delete!(active, n)
        return v
    end
    value = _eval_value_expr(ex, lookup, name)
    value isa Integer && value >= 0 || _fail(name,
        "sampling geometry dimension must be a nonnegative integer, got $(repr(value))")
    return Int(value)
end

function _external_array_lhs(lhs, rhs, data, names)
    _external_rhs(rhs) && Meta.isexpr(lhs, :ref) && lhs.args[1] isa Symbol &&
        lhs.args[1] ∉ data || return nothing
    target = lhs.args[1]
    dims = Any[]
    for axis in lhs.args[2:end]
        if Meta.isexpr(axis, :call, 3) && first(axis.args) === :(:) && axis.args[2] === 1
            push!(dims, axis.args[3])
        else
            push!(dims, _array_axis(target, axis, data, names))
        end
    end
    return target, dims
end

function _external_layout_entry(plan, p, offset, dims)
    g = p.args.geometry
    n = _sampling_extent(plan, g.unconstrained, p.name)
    labels = g.coordinate_names === nothing ?
        (isempty(dims) && n == 1 ? [p.name] : [Symbol(p.name, ".", i) for i in 1:n]) :
        Symbol[Symbol(p.name, ".", label) for label in g.coordinate_names]
    length(labels) == n || _fail(p.label, "sampling coordinate names must match packed dimension $n")
    return LayoutEntry(:external, nothing, p.name, labels, offset, n, :external,
        NaN, NaN, dims, LayoutEntry[], p.args)
end

_external_call(args) = Meta.isexpr(args.rhs, :call)
_external_authored_arguments(args) = _external_call(args) ? args.rhs.args[2:end] : Any[]
_external_arguments(args) = filter(a->!Meta.isexpr(a, :parameters), _external_authored_arguments(args))
_external_keywords(args) = Any[k for a in _external_authored_arguments(args)
    if Meta.isexpr(a, :parameters) for k in a.args]
_external_constructor(args) = _external_call(args) ? args.rhs.args[1] : args.rhs
_sampling_rhs_broadcast(rhs::Union{AbstractArray,Tuple,Ref}) = rhs
_sampling_rhs_broadcast(rhs) = Ref(rhs)
_external_u(e) = :(unconstrained[$(e.offset):$(e.offset + e.size - 1)])
_external_geometry_name(e) = Symbol(:_ppl_geometry_, e.name)
_sampling_ast_literal(x) = x
_sampling_ast_literal(x::Symbol) = QuoteNode(x)
_sampling_ast_literal(x::Expr) = Expr(:call, GlobalRef(Core, :Expr), QuoteNode(x.head),
    (_sampling_ast_literal(a) for a in x.args)...)
_sampling_ast_literal(x::QuoteNode) = Expr(:call, GlobalRef(Core, :QuoteNode), _sampling_ast_literal(x.value))
_sampling_ast_literal(x::GlobalRef) = Expr(:call, GlobalRef(Core, :GlobalRef), x.mod, QuoteNode(x.name))
function _external_geometry_expr(e)
    # Reconstruct the same structural descriptor through the public method.
    # Quoted authored expressions are constants, never active model values.
    # This also keeps kernel_expr printable/replayable with closure endpoints.
    return Expr(:call, GlobalRef(@__MODULE__, :sampling_geometry),
        _external_constructor(e.sampling), Expr(:tuple, (_sampling_ast_literal(a) for a in _external_authored_arguments(e.sampling))...),
        Expr(:tuple, (_sampling_ast_literal(d) for d in e.sampling.geometry.shape)...))
end
function _external_edge(e, edge)
    return _external_endpoint_call(GlobalRef(@__MODULE__, edge),
        Any[_external_geometry_name(e), _external_u(e), Expr(:tuple, e.dims...),
            _external_arguments(e.sampling)...], _external_keywords(e.sampling))
end
function _external_endpoint_call(head, args, keywords)
    return isempty(keywords) ? Expr(:call, head, args...) :
        Expr(:call, head, Expr(:parameters, keywords...), args...)
end
_sampling_constrain(g, u, shape, args...; kwargs...) = g.constrain(u, shape, args...; kwargs...)
_sampling_logjac(g, u, shape, args...; kwargs...) = g.logjac(u, shape, args...; kwargs...)

function _sampling_scalar_constrain(g, u, args...; kwargs...)
    value = g.constrain([u], (), args...; kwargs...)
    value isa Number || throw(ArgumentError("scalar sampling geometry must constrain to a number"))
    return value
end
function _sampling_scalar_unconstrain(g, value, args...; kwargs...)
    u = g.unconstrain(value, (), args...; kwargs...)
    u isa AbstractVector && length(u) == 1 ||
        throw(ArgumentError("scalar sampling geometry must invert to one packed coordinate"))
    return only(u)
end
_sampling_scalar_logjac(g, u, args...; kwargs...) = g.logjac([u], (), args...; kwargs...)

function _external_host_cells(e, edge, values, args; kwargs...)
    endpoint = getglobal(@__MODULE__, Symbol(:_sampling_scalar_, edge))
    cells = broadcast((x, a...)->endpoint(e.sampling.geometry, x, a...; kwargs...), values, args...)
    cells isa AbstractVector && length(cells) == e.size ||
        throw(ContractValidationError("[layout] external plate $(e.name) arguments have incompatible axes"))
    return Float64.(cells)
end
function _external_cells_expr(e, edge)
    inputs = Any[_external_u(e)]
    arguments = Any[]
    for a in _external_arguments(e.sampling)
        if a isa Number || a isa QuoteNode || a isa GlobalRef
            push!(arguments, a)
        else
            push!(inputs, a)
            push!(arguments, _dovar(length(inputs)))
        end
    end
    endpoint = GlobalRef(@__MODULE__, Symbol(:_sampling_scalar_, edge))
    keywords = _external_cell_keywords!(inputs, e.sampling)
    cell = _external_endpoint_call(endpoint,
        Any[_external_geometry_expr(e), _dovar(1), arguments...], keywords)
    # Same retained plate body for transforms and log-Jacobians; only the
    # endpoint changes. Host broadcasting calls these identical endpoints.
    return first(_plate_sum_stmts(:unused, :unused_sum, inputs, cell)).args[2]
end

function _external_host_args(layout, e, values)
    lookup = _layout_lookup(layout, values)
    return map(x->_eval_value_expr(x, lookup, e.name), Tuple(_external_arguments(e.sampling)))
end
function _external_host_kwargs(layout, e, values)
    lookup = _layout_lookup(layout, values)
    pairs = Pair{Symbol,Any}[]
    for keyword in _external_keywords(e.sampling)
        if Meta.isexpr(keyword, :kw, 2)
            push!(pairs, keyword.args[1] => _eval_value_expr(keyword.args[2], lookup, e.name))
        elseif Meta.isexpr(keyword, :..., 1)
            append!(pairs, collect(Base.pairs(_eval_value_expr(keyword.args[1], lookup, e.name))))
        else
            _fail(e.name, "invalid Julia keyword argument $(repr(keyword))")
        end
    end
    return (; pairs...)
end
function _external_cell_keywords!(inputs, args)
    keywords = Any[]
    for keyword in _external_keywords(args)
        value = keyword.args[end]
        push!(inputs, Expr(:call, GlobalRef(Base, :Ref), value))
        push!(keywords, Expr(keyword.head, keyword.args[1:end-1]..., _dovar(length(inputs))))
    end
    return keywords
end

function _external_density_statements(p)
    node = Symbol(:_ppl_prior_, p.name)
    density = GlobalRef(@__MODULE__, :sampling_logdensity)
    if !p.args.broadcast
        return Expr[:($node::Float64 = $density($(p.args.rhs), $(p.name)))]
    end
    inputs = Any[p.name]
    args = Any[]
    for arg in _external_arguments(p.args)
        # Function values and literal scalars are invariant; array/value
        # operands participate in Julia broadcasting through the RK plate.
        if arg isa Number || arg isa QuoteNode || arg isa GlobalRef
            push!(args, arg)
        else
            push!(inputs, arg)
            push!(args, _dovar(length(inputs)))
        end
    end
    rhs = if _external_call(p.args)
        _external_endpoint_call(_external_constructor(p.args), args,
            _external_cell_keywords!(inputs, p.args))
    elseif p.args.rhs isa GlobalRef
        p.args.rhs
    else
        push!(inputs, Expr(:call, GlobalRef(@__MODULE__, :_sampling_rhs_broadcast), p.args.rhs))
        _dovar(length(inputs))
    end
    cell = Expr(:call, density, rhs, _dovar(1))
    return _plate_sum_stmts(Symbol(:_ppl_pw_prior_, p.name), node, inputs, cell)
end
