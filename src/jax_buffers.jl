# Reference-element vectors need an assignment bitmap as well as numerical
# storage. Both are leaves of the loop/branch carry, so `isassigned` observes
# the writes that actually ran. No Julia values are inspected to size a row.
mutable struct _JaxRowShape
    dims::Tuple{Vararg{Int}}
    stored::Bool
end
struct _JaxRowView end
mutable struct _JaxSlotLayout
    scope::_JaxScope
    line::Int
    name::String
    dims::Union{Nothing,Tuple{Vararg{Int}}}
end
struct _JaxDenseRows end

_jax_slots(v::_PyVal) = v.origin isa _JaxSlotLayout
_jax_carried_origin(v::_PyVal) = v.origin isa Union{_JaxSlotLayout,_JaxRowShape,_JaxRowView} ? v.origin : nothing
_jax_row_metadata(v::_PyVal) = v.origin isa Union{_JaxRowShape,_JaxRowView}
function _jax_row_alias(f, args, result)
    f in (view, Base.maybeview, reshape, vec, dropdims, transpose, adjoint) &&
        !isempty(args) && _jax_row_metadata(first(args)) && _jax_is_array(result.T) ?
        _jax_with_origin(result, _JaxRowView()) : result
end
_jax_outer_rows(v::_PyVal) = _jax_slots(v) || v.origin isa _JaxDenseRows
_jax_slot_data(v::_PyVal) = _py_atom(v.code) * "[0]"
_jax_slot_mask(v::_PyVal) = _py_atom(v.code) * "[1]"
_jax_row_shape(v::_PyVal, dims) = _jax_with_origin(v, _JaxRowShape(Tuple(dims), false))

function _jax_slots_allocate(ctx, T, dims)
    T <: AbstractVector{<:AbstractVector{<:Number}} && length(dims) == 1 ||
        _jax_error("a reference-element allocation needs a vector of numerical vectors", _jax_where(ctx))
    n, extent = _jax_allocation_size(ctx, only(dims))
    _jax_call(ctx, _runtime_check, [_PyVal("(" * only(dims).code * " >= 0)", Bool, only(dims).static),
        _PyVal(repr("invalid Array dimensions"), ArgumentError)])
    name = _jax_fresh(ctx, :row_shape)
    _jax_emit!(ctx, name * " = None")
    layout = _JaxSlotLayout(ctx.scope, length(ctx.scope.lines), name, nothing)
    code = "(jnp.zeros((" * n * ",) + " * name * ", dtype=" * _jax_dtype(eltype(eltype(T))) *
           "), jnp.zeros((" * n * ",), dtype=bool))"
    _PyVal(code, T, false, extent, "", isempty(extent) ? 0 : 1, layout)
end

function _jax_slot_shape!(ctx, target, row)
    layout = target.origin
    shape = row.origin
    shape isa _JaxRowShape ||
        _jax_error("a nested-vector write needs a fresh row with an authored fixed shape", _jax_where(ctx))
    isempty(row.extent) || _jax_error("a nested-vector write with a data-dependent row length", _jax_where(ctx))
    if layout.dims === nothing
        layout.dims = shape.dims
        layout.scope.lines[layout.line] = "    "^layout.scope.indent * layout.name * " = " *
            "(" * join(string.(shape.dims), ", ") * (length(shape.dims) == 1 ? ",)" : ")")
    elseif layout.dims != shape.dims
        _jax_error("nested-vector writes with different row shapes need ragged storage", _jax_where(ctx))
    end
    shape.stored = true
end

function _jax_slot_position(ctx, x, indices; checked=true)
    length(indices) == 1 || _jax_error("a nested vector needs one integer index", _jax_where(ctx))
    proxy = _PyVal(_jax_slot_mask(x), Vector{Bool}, false, x.extent)
    index = only(indices)
    if index isa _PyVal
        index.T isa Type && index.T <: Integer && index.T !== Bool ||
            _jax_error("a nested vector needs one integer index", _jax_where(ctx))
        part, keeps, i = _jax_minus_one(index), 0, index
    else
        part, keeps, i = _jax_index_part(ctx, proxy, 1, index)
    end
    keeps == 0 && i !== nothing || _jax_error("slicing a reference-element buffer", _jax_where(ctx))
    n = isempty(x.extent) ? _jax_slot_mask(x) * ".shape[0]" : x.extent
    valid = "((" * i.code * " >= 1) & (" * i.code * " <= " * n * "))"
    checked && _jax_call(ctx, _runtime_check, [_PyVal(valid, Bool),
        _PyVal(repr("BoundsError: nested-vector index out of bounds"), BoundsError)])
    part, valid
end

function _jax_slots_index(ctx, x, indices)
    part, _ = _jax_slot_position(ctx, x, indices)
    _jax_call(ctx, _runtime_check, [_PyVal(_jax_slot_mask(x) * "[" * part * "]", Bool),
        _PyVal(repr("UndefRefError: access to undefined reference"), UndefRefError)])
    shape = x.origin.dims
    shape === nothing && _jax_error("a nested-vector allocation without a known row shape", _jax_where(ctx))
    _jax_with_origin(_PyVal(_jax_slot_data(x) * "[" * part * "]", eltype(x.T)),
                    _JaxRowShape(shape, true))
end

function _jax_borrowed_buffer(ctx, target)
    frames = get(ctx.mod.helper_names, :borrowed_slot_frames, nothing)
    borrowed = frames === nothing ? () : get(frames, ctx.locals, ())
    any(layout -> layout === target.origin, borrowed)
end
function _jax_slots_write!(ctx, lhs, target, row)
    _jax_borrowed_buffer(ctx, target) &&
        _jax_error("a device helper mutating a reference-element buffer argument needs caller writeback",
                   _jax_where(ctx))
    _jax_slot_shape!(ctx, target, row)
    part, _ = _jax_slot_position(ctx, target, lhs.args[2:end])
    name = _jax_fresh(ctx, lhs.args[1])
    _jax_emit!(ctx, name * " = (" * _jax_slot_data(target) * ".at[" * part * "].set(" * row.code *
                    "), " * _jax_slot_mask(target) * ".at[" * part * "].set(True))")
    for (key, v) in ctx.locals
        v.origin === target.origin || continue
        ctx.locals[key] = _PyVal(name, v.T, false, v.extent, v.limit, v.padded, v.origin)
    end
end

function _jax_slots_inputs!(ctx, inputs)
    layouts = [v.origin for v in inputs if _jax_slots(v) || _jax_row_metadata(v)]
    isempty(layouts) && return
    frames = get!(IdDict{Any,Any}, ctx.mod.helper_names, :borrowed_slot_frames)
    frames[ctx.locals] = layouts
end

# Rebinding one alias writes the same Julia vector. Branches and loops carry
# every live alias of that storage, not just the spelling on the left side.
function _jax_slot_aliases!(ctx, assigned)
    layouts = [ctx.locals[n].origin for n in assigned if haskey(ctx.locals, n) && _jax_slots(ctx.locals[n])]
    for (n, v) in ctx.locals
        any(o -> o === v.origin, layouts) && push!(assigned, n)
    end
    assigned
end

function _jax_function_call(ctx::_JaxContext, ::typeof(isassigned), args::Vector{_PyVal})
    x, indices = first(args), args[2:end]
    _jax_slots(x) || _jax_error("`isassigned` of this array representation", _jax_where(ctx))
    part, valid = _jax_slot_position(ctx, x, indices; checked=false)
    # An empty physical buffer cannot be gathered, even in an untaken cond
    # arm. `take` with an explicit false fill has the right out-of-bounds value.
    mask = _jax_slot_mask(x)
    code = "(" * valid * " & jnp.take(jnp.concatenate((" * mask *
           ", jnp.zeros((1,), dtype=bool))), " * part * ", mode='fill', fill_value=False))"
    _PyVal(code, Bool)
end

function _jax_slots_rows(ctx, v)
    _jax_slots(v) || return v
    mask = _jax_slot_mask(v)
    valid = isempty(v.extent) ? mask : "(" * mask * " | (jnp.arange(" * mask *
        ".shape[0]) >= " * v.extent * "))"
    _jax_call(ctx, _runtime_check, [_PyVal("jnp.all(" * valid * ")", Bool),
        _PyVal(repr("UndefRefError: access to undefined reference"), UndefRefError)])
    _PyVal(_jax_slot_data(v), v.T, false, v.extent, "", v.padded, _JaxDenseRows())
end

function _jax_slots_repeat(ctx, v)
    isempty(v.extent) && return v
    v = _jax_bind!(ctx, :slots, v)
    data, mask = _jax_slot_data(v), _jax_slot_mask(v)
    edge = "jax.lax.dynamic_index_in_dim(" * data * ", jnp.maximum(" * v.extent * " - 1, 0), 0, keepdims=True)"
    filled = "(jnp.where((jnp.arange(" * data * ".shape[0]) < " * v.extent *
             ")[:, None], " * data * ", " * edge * ") if " * data * ".shape[0] else " * data * ")"
    _jax_bind!(ctx, :slots, _PyVal("(" * filled * ", " * mask * ")", v.T, false,
        v.extent, v.limit, v.padded, v.origin))
end

function _jax_check_slot_shapes(ctx)
    for v in values(ctx.locals)
        _jax_slots(v) && v.origin.dims === nothing &&
            _jax_error("a nested-vector allocation without an authored row shape", _jax_where(ctx))
    end
end

# A logical empty buffer has no cells, even if other groups need physical
# storage. Masking its output after evaluating a cell can poison a shared
# argument's gradient (for example, a * log(0)). Guard the cell itself.
# Abstract shape evaluation performs no numerical cell work, and closing
# over its arguments keeps prepared StaticInts static during that trace.
function _jax_slots_active_cell(ctx, cell, params)
    name = _jax_fresh(ctx, :active_rows_cell)
    shape = _jax_fresh(ctx, :cell_shape)
    call = cell * "(" * join(params, ", ") * ")"
    _jax_emit!(ctx, "def " * name * "(" * join(params, ", ") * "):")
    _jax_emit!(ctx, "    " * shape * " = jax.eval_shape(lambda: " * call * ")")
    _jax_emit!(ctx, "    return jax.lax.cond(" * last(params) * ", lambda: " * call *
        ", lambda: jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), " * shape * "))")
    name
end
