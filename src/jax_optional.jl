# An optional numerical scalar has one JAX pytree structure in both arms:
# (present, value). The absent scalar is never read as a Julia value. Host
# preparation keeps ordinary Python None and numbers instead.
function _jax_optional_payload(@nospecialize(T))
    T isa Union || return nothing
    parts = Base.uniontypes(T)
    length(parts) == 2 && Nothing in parts || return nothing
    E = only(filter(!=(Nothing), parts))
    isconcretetype(E) && E <: Number ? E : nothing
end
_jax_has_optional(T) = _jax_optional_payload(T) !== nothing ||
    _jax_is_tuple(T) && any(_jax_has_optional, fieldtypes(T))

function _jax_optional_branch_type(A, B)
    a, b = _jax_optional_payload(A), _jax_optional_payload(B)
    a === nothing && b === nothing && A !== Nothing && B !== Nothing && return nothing
    A === Nothing && B === Nothing && return Nothing
    a = A === Nothing ? nothing : a === nothing ? A : a
    b = B === Nothing ? nothing : b === nothing ? B : b
    all(T -> T === nothing || T isa Type && isconcretetype(T) && T <: Number, (a, b)) ||
        return nothing
    E = a === nothing ? b : b === nothing ? a : promote_type(a, b)
    Union{Nothing,E}
end

function _jax_optional_value(ctx, v, T)
    E = _jax_optional_payload(T)
    E === nothing && return nothing
    v.T === T && return v
    present, payload = if v.T === Nothing
        "False", _jax_literal(zero(E))
    elseif _jax_optional_payload(v.T) !== nothing
        v = _jax_bind!(ctx, :optional, v)
        _py_atom(v.code) * "[0]", _PyVal(_py_atom(v.code) * "[1]", _jax_optional_payload(v.T))
    else
        "True", v
    end
    payload = _jax_branch_value(ctx, payload, E)
    _PyVal("(" * present * ", " * payload.code * ")", T, v.static)
end

function _jax_optional_arm(ctx, arm, T)
    saved = ctx.scope
    ctx.scope = arm.scope
    try
        _JaxArm(arm.hint, arm.scope, _jax_branch_value(ctx, arm.value, T))
    finally
        ctx.scope = saved
    end
end

function _jax_call_positional(ctx::_JaxContext, ::typeof(isnothing), args::Vector{_PyVal})
    x = only(args)
    x.T === Nothing && return _jax_literal(true)
    if _jax_optional_payload(x.T) === nothing
        x.T isa Type && typeintersect(x.T, Nothing) === Union{} && return _jax_literal(false)
        return _jax_traceable_or_lowered(ctx, isnothing, args)
    end
    ctx.host === nothing ? _PyVal("jnp.logical_not(" * _py_atom(x.code) * "[0])", Bool) :
        _PyVal("(" * x.code * " is None)", Bool, true)
end

function _jax_call_positional(ctx::_JaxContext, ::typeof(something), args::Vector{_PyVal})
    isempty(args) && return _jax_call(ctx, throw,
        [_PyVal(repr("No value arguments present"), ArgumentError)])
    x = first(args)
    x.T === Nothing && return _jax_call_positional(ctx, something, args[2:end])
    E = _jax_optional_payload(x.T)
    if E === nothing
        x.T isa Type && typeintersect(x.T, Union{Nothing,Some}) === Union{} && return x
        return _jax_traceable_or_lowered(ctx, something, args)
    end
    x = _jax_bind!(ctx, :optional, x)
    present = ctx.host === nothing ? _PyVal(_py_atom(x.code) * "[0]", Bool) :
        _PyVal("(" * x.code * " is not None)", Bool, true)
    payload = _PyVal(ctx.host === nothing ? _py_atom(x.code) * "[1]" : x.code, E)
    _jax_conditional(ctx, present, () -> payload,
        () -> _jax_call_positional(ctx, something, args[2:end]))
end
