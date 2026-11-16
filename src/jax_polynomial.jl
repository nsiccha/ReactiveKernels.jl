# A runtime prefix retains the source array's capacity and carries its
# logical length. Its unused storage follows the ordinary padded-array
# convention; no coefficient count is copied into the emitted program.
function _jax_prefix(ctx, x, indices)
    x.T <: AbstractVector{<:Number} && length(indices) == 1 || return nothing
    index = only(indices)
    Meta.isexpr(index, :call, 3) && index.args[1] === :(:) && index.args[2] == 1 || return nothing
    hi = _jax_bind!(ctx, :prefix_stop, _jax_bound_expr(ctx, x, 1, index.args[3]))
    hi.T <: Integer || return nothing
    n = _jax_call(ctx, length, [x])
    _jax_call(ctx, _runtime_check,
        [_PyVal("(" * hi.code * " <= " * n.code * ")", Bool, _jax_known(ctx, hi) && _jax_known(ctx, n)),
         _PyVal(repr("BoundsError: prefix exceeds array length"), BoundsError)])
    if _jax_known(ctx, hi)
        return _PyVal(_py_atom(x.code) * "[:max(" * hi.code * ", 0)]", x.T, x.static)
    end
    count = _jax_bind!(ctx, :prefix_length, _PyVal("jnp.maximum(" * hi.code * ", 0)", Int))
    _jax_repeat_last_valid(ctx, _PyVal(x.code, x.T, false, count.code, "", 1))
end

# Horner's recurrence over the logical coefficient count. Starting at the
# last coefficient preserves the constant-polynomial case even at Inf/NaN.
function _jax_call_positional(ctx::_JaxContext, ::typeof(evalpoly), args::Vector{_PyVal})
    length(args) == 2 || return _jax_traceable_or_lowered(ctx, evalpoly, args)
    x, coefficients = args
    x.T <: Real && coefficients.T <: AbstractVector{<:Real} ||
        return _jax_traceable_or_lowered(ctx, evalpoly, args)
    T = promote_type(x.T, eltype(coefficients.T))
    capacity = _PyVal(_jax_dim_code(ctx, coefficients, 1), Int, true)
    source = quote
        if capacity == 0
            throw(BoundsError("evalpoly requires at least one coefficient"))
        else
            n = length(coefficients)
            n > 0 || throw(BoundsError("evalpoly requires at least one coefficient"))
            result = $T(coefficients[n])
            for k in 1:(n - 1)
                result = muladd(x, result, coefficients[n - k])
            end
            result
        end
    end
    _jax_source_expr(ctx, source, (:x, :coefficients, :capacity), [x, coefficients, capacity],
                     @__MODULE__, nothing)
end
