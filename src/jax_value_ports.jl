# An array port carries its physical array and logical extent together. Tuple
# layouts describe authored structure, never a data length or observed shape.
function _jax_port_pack(ctx, v::_PyVal)
    T = _jax_argument_type(v)
    if _jax_is_tuple(T)
        parts = [_jax_port_pack(ctx, _jax_tuple_element(ctx, v, _jax_literal(k)))
                 for k in 1:fieldcount(T)]
        return _jax_tuple_code([_PyVal(first(p), Any) for p in parts]), Tuple(last.(parts))
    end
    isempty(v.extent) && return v.code, nothing
    axes = _jax_padded_axes(v)
    axes == 0 && _jax_error("a value port with an unrepresented padded array layout",
                            _jax_where(ctx))
    "(" * v.code * ", " * v.extent * ")", axes
end

function _jax_port_unpack(ctx, code, T, layout)
    if layout isa Tuple
        values = [_jax_port_unpack(ctx, _py_atom(code) * "[" * string(k - 1) * "]",
                                  fieldtype(T, k), layout[k]) for k in eachindex(layout)]
        value = _jax_call_positional(ctx, tuple, values)
        return T <: NamedTuple ? _PyVal(value.code, T, value.static, value.extent,
            value.limit, value.padded, value.origin, value.field_extents) : value
    end
    layout === nothing && return _PyVal(code, T)
    _PyVal(_py_atom(code) * "[0]", T, false, _py_atom(code) * "[1]", "", layout)
end

_jax_port_has_layout(::Nothing) = false
_jax_port_has_layout(::Integer) = true
_jax_port_has_layout(layout::Tuple) = any(_jax_port_has_layout, layout)

function _jax_common_port_layout(ctx, a, b)
    a == b && return a
    a === nothing && return b
    b === nothing && return a
    a isa Tuple && b isa Tuple && length(a) == length(b) &&
        return map((x, y) -> _jax_common_port_layout(ctx, x, y), a, b)
    _jax_error("conditional array arms with different padded axes", _jax_where(ctx))
end

# A dense arm uses its physical length when the other arm carries a logical
# one. Both return the same pytree; the selected length remains runtime data.
function _jax_port_pack_as(ctx, v, layout)
    if layout isa Tuple
        parts = [_jax_port_pack_as(ctx, _jax_tuple_element(ctx, v, _jax_literal(k)), layout[k])
                 for k in eachindex(layout)]
        return _jax_tuple_code([_PyVal(p, Any) for p in parts])
    end
    layout === nothing && return v.code
    extent = v.extent
    if isempty(extent)
        count_ones(layout) == 1 ||
            _jax_error("a dense conditional arm matching multiple padded axes", _jax_where(ctx))
        v = _jax_bind!(ctx, :arm_array, v)
        extent = _py_atom(v.code) * ".shape[" * string(trailing_zeros(layout)) * "]"
    end
    "(" * v.code * ", jnp.asarray(" * extent * ", dtype=jnp.int64))"
end

function _jax_conditional_ports(ctx, condition, a, b, T)
    any(arm -> !isempty(arm.value.extent) || any(!isempty, arm.value.field_extents), (a, b)) ||
        return nothing
    converted = map((a, b)) do arm
        outer = ctx.scope
        ctx.scope = arm.scope
        try
            v = _jax_branch_value(ctx, arm.value, T)
            layout = last(_jax_port_pack(ctx, v))
            (_JaxArm(arm.hint, arm.scope, v), layout)
        finally
            ctx.scope = outer
        end
    end
    layout = _jax_common_port_layout(ctx, last(converted[1]), last(converted[2]))
    _jax_port_has_layout(layout) || return nothing
    arms = map(converted) do (arm, _)
        outer = ctx.scope
        ctx.scope = arm.scope
        try
            code = _jax_port_pack_as(ctx, arm.value, layout)
            union!(outer.used, arm.scope.used)
            _JaxArm(arm.hint, arm.scope, _PyVal(code, Any))
        finally
            ctx.scope = outer
        end
    end
    value = if condition.static
        _jax_static_branch!(ctx, condition, arms..., Any)
    else
        yes, no = map(arm -> _jax_emit_arm!(ctx, arm), arms)
        op = layout == 1 && T isa Type && T <: AbstractVector{<:Number} ?
             _jax_helper!(ctx, "_rk_vector_cond", _JAX_VECTOR_COND) : "jax.lax.cond"
        _PyVal(op * "(" * condition.code * ", " * yes.callable * ", " * no.callable * ")", Any)
    end
    result = _jax_bind!(ctx, :branch, value)
    _jax_port_unpack(ctx, result.code, T, layout)
end

# The branches can have different storage widths, for example an empty
# vector and the compacted rows of a group. Shape evaluation determines a
# common capacity without running either numerical branch.
const _JAX_VECTOR_COND = """
def _rk_vector_cond(predicate, yes, no):
    yes_width = jax.eval_shape(yes)[0].shape[0]
    no_width = jax.eval_shape(no)[0].shape[0]
    if yes_width == no_width:
        return jax.lax.cond(predicate, yes, no)
    width = max(yes_width, no_width)

    def at_width(arm):
        values, length = arm()
        mode = "edge" if values.shape[0] else "constant"
        return jnp.pad(values, (0, width - values.shape[0]), mode=mode), length

    return jax.lax.cond(predicate, lambda: at_width(yes), lambda: at_width(no))
"""
