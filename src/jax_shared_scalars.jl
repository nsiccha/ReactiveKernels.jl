# A scalar call inside a mapped conditional would otherwise receive its
# operands broadcast over the condition's axes by lax.cond's batching rule.
# Reduce demand only along axes on which every operand is shared. The
# original conditional still selects which cells consume the result.
# The barrier keeps the any-row reduction separate from elementwise
# derivatives that consume the same mask.
const _JAX_SCALAR_DEMAND = raw"""
@jax.custom_batching.custom_vmap
def _rk_scalar_demand(active, *parameters):
    return active

@_rk_scalar_demand.def_vmap
def _rk_scalar_demand_batch(size, batched, active, *parameters):
    if batched[0] and not any(batched[1:]):
        return _rk_scalar_demand(jnp.any(jax.lax.optimization_barrier(active), axis=0), *parameters), False
    return active, batched[0]
"""

# A guarded scalar's tangent is shared by the cells while its coefficients are
# selected per cell, so reverse mode sums the cells' products. Broadcasting
# the tangent to the cells behind a barrier makes that sum read one stored
# product; otherwise the GPU compiler can put the cells' other elementwise
# derivatives inside the summing kernel, which runs them slower.
const _JAX_CELL_TANGENT = raw"""
def _rk_cell_tangent(selected, tangent):
    return jax.lax.optimization_barrier(jnp.where(selected, tangent, jnp.zeros_like(tangent)))
"""

const _JAX_SELECTED_CUT = raw"""
# A cut returns a tuple of scalar outputs. Select each coefficient before
# its cell multiplies a tangent or a cotangent, including at a singularity.
def _rk_selected_cut(cut):
    @jax.custom_jvp
    def selected(active, demand, *args):
        values = _rk_if_nonempty(demand, lambda: cut(*args))
        return tuple(jnp.where(active, v, jnp.zeros_like(v)) for v in values)

    def selected_jvp(primals, tangents):
        active, demand, *args = primals
        values = selected(*primals)
        # Differentiate the actual cut locally for higher derivatives. A
        # shared pullback would fold a zero from an independent inactive call
        # into an active call with a nonfinite derivative.
        columns = []
        for i, tangent in enumerate(tangents[2:]):
            if (isinstance(tangent, jax.custom_derivatives.SymbolicZero) or
                    not jnp.issubdtype(jnp.result_type(args[i]), jnp.inexact)):
                continue
            column = _rk_if_nonempty(active, lambda: jax.jacfwd(cut, argnums=i)(*args))
            columns.append((column, tangent))

        def product(k, value):
            if not jnp.issubdtype(value.dtype, jnp.inexact):
                return jnp.zeros(value.shape, dtype=jax.dtypes.float0)
            terms = [column[k] * tangent for column, tangent in columns]
            if not terms:
                return jnp.zeros_like(value)
            return sum(terms[1:], terms[0]).astype(value.dtype)

        dots = tuple(product(k, value) for k, value in enumerate(values))
        return values, dots

    selected.defjvp(selected_jvp, symbolic_zeros=True)
    return selected
"""

# Recognize only the common Julia/Python scalar expression grammar emitted
# for arithmetic and comparisons. An index, helper call or arm-local binding
# must stay at its original program point. Guard projections additionally
# exclude arithmetic: testing an inner predicate before entering an outer
# arm must be total on the original cell inputs.
function _jax_scalar_parameters!(reads, ex, parameters, guard, dtype)
    ex isa Real && return true
    if ex isa Symbol
        ex in (:True, :False) && return true
        ex in parameters || return false
        push!(reads, ex)
        return true
    end
    ex isa Expr || return false
    ex in (:(jnp.inf), :(jnp.nan), :(np.inf), :(np.nan)) && return true
    ex.head === :call || return false
    op = first(ex.args)
    allowed = op in (:<, :<=, :>, :>=, :(==), :(!=), :&, :|,
                     :(jnp.logical_and), :(jnp.logical_or), :(jnp.logical_not),
                     :(jnp.isfinite), :(jnp.isinf), :(jnp.isnan))
    if !allowed
        guard && return false
        terms = ex.args[2:end]
        # These operands are read before the original guard. Admit affine
        # expressions with finite slopes, never a potentially singular
        # division or a product whose Jacobian depends on another operand.
        literal(x) = x isa Real && isfinite(dtype(x))
        (op in (:+, :-) || op === :* && length(terms) == 2 && any(literal, terms) ||
         op === :/ && length(terms) == 2 && literal(last(terms)) &&
         !iszero(dtype(last(terms))) && isfinite(inv(dtype(last(terms))))) || return false
    end
    all(arg -> _jax_scalar_parameters!(reads, arg, parameters, guard, dtype), ex.args[2:end])
end

function _jax_selected_cut!(ctx, cut)
    get!(ctx.mod.helper_names, (:selected_cut, cut)) do
        name = _jax_private_identifier(ctx.mod.used, Symbol(cut, "_selected"))
        helper = _jax_helper!(ctx, "_rk_selected_cut", _JAX_SELECTED_CUT)
        push!(ctx.mod.helpers, name * " = " * helper * "(" * cut * ")")
        name
    end
end

function _jax_guarded_scalar(ctx, owner, args, value, N, cuts)
    ctx.scope.conditional_arm && value.T isa Type && value.T <: Real || return nothing
    cell = ctx.scope
    while cell.conditional_arm
        isempty(cell.carried) || return nothing
        cell = cell.parent
    end
    parameters = cell.scalar_parameters
    parameters === nothing && return nothing

    operands = Set{Symbol}()
    for arg in args
        dtype = arg.T === Float32 ? Float32 : _jax_float_type()
        _jax_scalar_parameters!(operands, Meta.parse(arg.code; raise=false), parameters, false, dtype) ||
            return nothing
    end
    isempty(operands) && return nothing
    guards = String[]
    controls = Set{Symbol}()
    for (scope, guard) in _JAX_GUARDS
        _jax_scope_within(scope, cell) || continue
        _jax_scalar_parameters!(controls, Meta.parse(guard; raise=false), parameters, true, _jax_float_type()) ||
            return nothing
        push!(guards, guard)
    end
    isempty(guards) && return nothing
    issubset(controls, operands) && return nothing
    demand = _jax_helper!(ctx, "_rk_scalar_demand", _JAX_SCALAR_DEMAND)
    active = foldl((a, b) -> "jnp.logical_and(" * a * ", " * b * ")", guards)
    refs = join(("jax.lax.stop_gradient(" * string(n) * ")" for n in sort!(collect(operands))), ", ")
    guarded = get!(ctx.mod.helper_names, (:guarded_scalar, owner)) do
        f = _jax_private_identifier(ctx.mod.used, Symbol(owner, "_guarded"))
        primal = _jax_if_nonempty(ctx, "demand", "lambda: " * owner * "(*args)")
        push!(ctx.mod.helpers, "@jax.custom_jvp\ndef " * f *
              "(selected, demand, *args):\n    value = " * primal *
              "\n    return jnp.where(selected, value, jnp.zeros_like(value))")
        _jax_scalar_jvp!(ctx, f, N, cuts, ctx.mod.helper_names[(:scalar_cuts, owner)]; guarded = true)
        f
    end
    name = _jax_fresh(ctx, :shared_value)
    selected = _jax_fresh(ctx, :selected_cell)
    prefix = "    "^cell.indent
    push!(cell.lines, prefix * selected * " = " * active,
          prefix * name * " = " * guarded * "(" * selected * ", " *
          demand * "(" * selected * ", " * refs * "), " * _jax_args(args) * ")")
    scope = ctx.scope
    while scope !== cell
        scope = scope.parent
        push!(scope.used, name, selected)
    end
    _PyVal(name, value.T)
end
