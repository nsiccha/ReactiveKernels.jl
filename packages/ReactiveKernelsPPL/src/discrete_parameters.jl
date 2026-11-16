# Integer parameters have no unconstrained real coordinates. Their named
# values remain runtime inputs, including when a query changes only one block.
_integer_parameter(p) = get(SAMPLED_SUPPORT, p.family, nothing) === :integer
_integer_parameters(plan) = [p for p in (plan.parameters..., plan.plate_parameters..., plan.array_parameters...)
    if _integer_parameter(p) && p.name ∉ plan.conditioned]
_has_integer_parameters(plan) = any(p -> p.name ∉ plan.marginalized_integers,
    _integer_parameters(plan))

# The first integer-parameter increment is explicitly data-bounded:
# ReactiveKernels:rkppl decision 2026-11-16T15-57-32-777-1551do1, option 1.
# This applies to latent declarations, not integer-valued observation trials.
function _validate_integer_supports(plan)
    names = setdiff(_all_name_set(plan), plan.conditioned)
    for p in _integer_parameters(plan)
        bounds = p.family === :binomial ? (p.args.arg1,) :
            p.family === :discrete_uniform ? (p.args.arg1, p.args.arg2) : ()
        p.support_override === nothing || (bounds = (bounds..., p.support_override[2:end]...))
        for bound in bounds, name in _expr_value_symbols(bound)
            _reads_data_only(plan, name, names, Set{Symbol}()) || _fail(p.label,
                "integer support bounds must read data: `$name` reads a free parameter")
        end
    end
    return nothing
end

# Only the support's endpoints determine enumeration. In particular the
# Binomial probability is not a support dependency.
function _integer_support_expr(p)
    base = _integer_support_expr(Val(p.family), p.args)
    p.support_override === nothing && return base
    # Contract validation admits only unnormalized interval restrictions for
    # integer laws. Enumerate their intersection, so expressions that index by
    # the latent never see an excluded state, even before a density guard.
    _, lo, hi = p.support_override
    return Expr(:call, GlobalRef(@__MODULE__, :_restricted_integer_support),
        base, lo, hi)
end
_integer_support_expr(::Val{:binomial}, a) = Expr(:call,
    GlobalRef(@__MODULE__, :_binomial_support), a.arg1)
_integer_support_expr(::Val{:bernoulli}, a) = :(0:1)
_integer_support_expr(::Val{:categorical}, a) = :(1:length($(a.arg1)))
_integer_support_expr(::Val{:discrete_uniform}, a) = Expr(:call,
    GlobalRef(@__MODULE__, :_discrete_uniform_support), a.arg1, a.arg2)

@traceable function _binomial_support(n)
    n isa Real && isfinite(n) && isinteger(n) && n >= 0 ||
        throw(ArgumentError("Binomial trials must be a nonnegative integer, got $n"))
    return 0:Int(n)
end

@traceable function _restricted_integer_support(support, lo, hi)
    lo isa Real && hi isa Real && !isnan(lo) && !isnan(hi) && lo <= hi ||
        throw(ArgumentError("integer restriction needs ordered real bounds"))
    lower, upper = max(first(support), lo), min(last(support), hi)
    return lower > upper ? (1:0) : (ceil(Int, lower):floor(Int, upper))
end

# Cell/partition queries cache row ownership and shared prefixes. Their integer
# state is fixed at preparation, just like index data: changing it can change
# both footprints and row plans. Freeze a private copy at the existing graph
# input, so the ordinary partial-application path owns those caches.
function _fixed_integer_query(plan, layout, discrete = nothing)
    existing = get(plan.frozen.values, :_ppl_discrete, nothing)
    state = discrete === nothing ? existing : discrete
    _check_integer_parameters(layout, state)
    state === nothing && return plan
    state === existing && return plan
    frozen = FrozenValues(merge(plan.frozen.values, (; _ppl_discrete = deepcopy(state))),
        plan.frozen.inputs)
    return _with(plan; frozen)
end
@traceable function _discrete_uniform_support(lo, hi)
    lo isa Integer && hi isa Integer && lo <= hi ||
        throw(ArgumentError("DiscreteUniform requires integer bounds with lower ≤ upper"))
    return lo:hi
end

_integer_value(values::NamedTuple, name::Symbol, dims::Tuple) =
    _integer_value(values, (name,), dims)
function _integer_value(values::NamedTuple, name::Tuple, dims::Tuple)
    value = _scope_lookup(values, name)
    value === _MISSING_SCOPED_VALUE && throw(ArgumentError("missing integer parameter `$name`"))
    valid = isempty(dims) ? value isa Integer :
        value isa AbstractArray{<:Integer} && size(value) == dims
    valid || throw(ArgumentError("integer parameter `$name` requires " *
        (isempty(dims) ? "an integer" : "an integer array of size $dims")))
    return value
end

_discrete_args(::Nothing) = ()
_discrete_args(d::NamedTuple) = (d,)
function _query_integer_state(plan, layout, discrete)
    fixed = get(plan.frozen.values, :_ppl_discrete, nothing)
    _check_integer_parameters(layout, discrete === nothing ? fixed : discrete)
    return fixed === nothing ? discrete : nothing
end
function _check_integer_parameters(layout, discrete)
    entries = filter(e -> e.kind === :integer, layout.entries)
    if isempty(entries)
        discrete === nothing || throw(ArgumentError("this model has no integer parameters"))
    else
        discrete isa NamedTuple || throw(ArgumentError(
            "supply integer parameters as `discrete = (; " *
            join((string(e.name) * " = value" for e in entries), ", ") * ")`"))
        for e in entries
            _integer_value(discrete, get(layout.name_paths, e.name, (e.name,)), Tuple(e.dims))
        end
    end
    return nothing
end
