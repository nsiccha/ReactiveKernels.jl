# Adapter mechanics only. Every primal, forward branch and staged reverse
# branch below is translated from the owned rule's mathematical graph.
function _jax_call_positional(ctx::_JaxContext, region::_RuleRegion, args::Vector{_PyVal})
    p = region.plan
    length(args) == length(p.have) ||
        _jax_error("a derivative region's argument count differs from its source plan", _jax_where(ctx))
    _jax_plan_call(ctx, region, p, args)
end

const _JAX_DERIVATIVE_RULE = raw"""
from jax.experimental import hijax as _rk_hijax


def _rk_rule_metadata(value, layout):
    if isinstance(layout, tuple):
        return tuple(_rk_rule_metadata(v, l) for v, l in zip(value, layout))
    return value[1] if layout is not None else None


def _rk_rule_repack(value, source, target, metadata, tangent=False):
    # Logical lengths are execution data, with zero tangent. Cotangents of
    # physical padding must not flow into repeated padding values upstream.
    if isinstance(target, tuple):
        return tuple(_rk_rule_repack(v, s, t, m, tangent)
                     for v, s, t, m in zip(value, source, target, metadata))
    if source is not None:
        value = value[0]
    if target is None:
        return value
    mask = True
    for axis in range(value.ndim):
        if target & (1 << axis):
            shape = [1] * value.ndim
            shape[axis] = value.shape[axis]
            mask = mask & (jnp.arange(value.shape[axis]).reshape(shape) < metadata)
    value = jnp.where(mask, value, jnp.zeros_like(value))
    length = (_rk_hijax.Zero(jax.typeof(metadata).to_tangent_aval())
              if tangent else metadata)
    return value, length


class _RKDerivativeRule(_rk_hijax.HiPrim):
    skip_linearization_on_zero_tangents = True
    lin, linearized = _rk_hijax.linearize_from_jvp

    def __init__(self, primal, forward, stages, differentiable, layouts, out_layout,
                 back_layouts, args):
        self.primal = primal
        self.forward = forward
        self.stages = stages
        self.differentiable = differentiable
        self.layouts = layouts
        self.out_layout = out_layout
        self.back_layouts = back_layouts
        self.in_avals = jax.tree.map(jax.typeof, args)
        graph, shape = jax.make_jaxpr(primal, return_shape=True)(*args)
        self.out_aval = jax.tree.unflatten(jax.tree.structure(shape), graph.out_avals)
        self.effects = frozenset(graph.effects)
        ins, intree = jax.tree.flatten(self.in_avals)
        outs, outtree = jax.tree.flatten(self.out_aval)
        self.params = dict(_key=(primal, forward, stages, differentiable, layouts,
                                 out_layout, back_layouts,
                                 tuple(ins), intree, tuple(outs), outtree))
        super().__init__()

    def expand(self, *args):
        return self.primal(*args)

    def _check_activity(self, mask):
        if mask & ~self.differentiable:
            raise ValueError("derivative_rule cannot differentiate data-only inputs")

    def jvp(self, primals, tangents):
        zero = lambda t: isinstance(t, _rk_hijax.Zero)
        active = [any(not zero(t) for t in jax.tree.leaves(ts, is_leaf=zero))
                  for ts in tangents]
        mask = sum(1 << i for i, a in enumerate(active) if a)
        self._check_activity(mask)
        if self.forward is None:
            raise ValueError("derivative_rule has no forward branch")
        directions = [_rk_rule_repack(
            jax.tree.map(_rk_hijax.instantiate_zeros, ts, is_leaf=zero), layout, layout,
            _rk_rule_metadata(primal, layout))
            for i, (primal, ts, layout) in enumerate(zip(primals, tangents, self.layouts))
            if self.differentiable & (1 << i)]
        value, direction = self.forward(*primals[:-1], *directions, primals[-1])
        return value, _rk_rule_repack(direction, self.out_layout, self.out_layout,
                                      _rk_rule_metadata(value, self.out_layout), True)

    def vjp_fwd(self, nzs_in, *args):
        mask = sum(1 << i for i, nz in enumerate(nzs_in) if any(jax.tree.leaves(nz)))
        self._check_activity(mask)
        if mask == 0:
            return self.primal(*args), (StaticInt(0), (), (), None, args[-1])
        if self.stages is None:
            raise ValueError("derivative_rule has no reverse branch")
        values = self.stages[mask][0](*args)
        metadata = tuple(_rk_rule_metadata(a, l) for a, l in zip(args, self.layouts))
        out_metadata = _rk_rule_metadata(values[0], self.out_layout)
        return values[0], (StaticInt(mask), values[1:], metadata, out_metadata, args[-1])

    def vjp_bwd_retval(self, residuals, covector):
        mask, values, metadata, out_metadata, active = residuals
        zero = lambda t: isinstance(t, _rk_hijax.Zero)
        covector = jax.tree.map(_rk_hijax.instantiate_zeros, covector, is_leaf=zero)
        cotangents = iter(())
        if mask:
            covector = _rk_rule_repack(covector, self.out_layout, self.out_layout, out_metadata)
            cotangents = iter(zip(self.stages[mask][1](*values, covector, active), self.back_layouts[mask]))
        result = []
        for i, (aval, layout) in enumerate(zip(self.in_avals, self.layouts)):
            if mask & (1 << i):
                cotangent, source = next(cotangents)
                result.append(_rk_rule_repack(cotangent, source, layout, metadata[i], True))
            else:
                result.append(jax.tree.map(lambda a: _rk_hijax.Zero(a.to_tangent_aval()), aval))
        return tuple(result)

    def batch(self, axis_data, args, dims):
        # Map whole calls so required lazy branches stay local to each lane.
        mapped = _rk_hijax.bdims_at_front(axis_data, args, dims)
        result = jax.lax.map(lambda xs: self(*xs), mapped)
        return result, jax.tree.map(lambda _: 0, self.out_aval)
"""

_jax_rule_layout_code(::Nothing) = "None"
_jax_rule_layout_code(layout::Integer) = string(layout)
_jax_rule_layout_code(layout::Tuple) = "(" * join(_jax_rule_layout_code.(layout), ", ") *
    (length(layout) == 1 ? ",)" : ")")

function _jax_derivative_rule(ctx::_JaxContext,
        rule::DerivativeRule{Name,N,Inputs,Primal,Forward,Reverse,Staged},
        args::Vector{_PyVal}) where {Name,N,Inputs,Primal,Forward,Reverse,Staged}
    (ctx.host !== nothing || all(a -> a.static, args)) &&
        return _jax_rule_primal(ctx, rule, Primal, args)
    types = Tuple(a.T for a in args)
    packed = [_jax_port_pack(ctx, a) for a in args]
    layouts = Tuple(last.(packed))
    key = (:derivative_rule, objectid(rule), types, layouts, ctx.traced_branches > 0)
    cached = get(ctx.mod.helper_names, key, nothing)
    if cached === nothing
        effects_before = _JAX_EFFECTS[]
        name = _jax_private_identifier(ctx.mod.used, Symbol("_rk_", Name))
        _jax_helper!(ctx, "StaticInt", _JAX_STATIC_INT)
        _jax_helper!(ctx, "_RKDerivativeRule", _JAX_DERIVATIVE_RULE)
        primal = _jax_private_identifier(ctx.mod.used, Symbol(name, "_primal"))
        primal_text, primal_value = _jax_rule_function(ctx, primal, Primal, rule.primal_ops, types;
            owner=name, layouts, pack=true, guarded=true)
        push!(ctx.mod.helpers, primal_text)
        out_layout = last(_jax_port_pack(ctx, primal_value))
        D = typeof(_rule_activity(rule)).parameters[1]
        forward = "None"
        if Forward !== nothing
            forward = _jax_private_identifier(ctx.mod.used, Symbol(name, "_forward"))
            directions = Tuple(types[i] for i in 1:N if D & (1 << (i - 1)) != 0)
            direction_layouts = Tuple(layouts[i] for i in 1:N if D & (1 << (i - 1)) != 0)
            forward_text, forward_value = _jax_rule_function(ctx, forward, Forward, rule.forward_ops,
                (types..., directions...); owner=name, layouts=(layouts..., direction_layouts...),
                pack=true, guarded=true)
            forward_layout = last(_jax_port_pack(ctx, forward_value))
            forward_layout == (out_layout, out_layout) ||
                _jax_error("a derivative-rule forward branch with incompatible output layouts", _jax_where(ctx))
            push!(ctx.mod.helpers, forward_text)
        end
        stages = "None"
        back_layouts = Any[nothing]
        if Staged !== nothing
            pairs = String["None"]
            for mask in eachindex(Staged)
                if Staged[mask] === nothing
                    push!(pairs, "None")
                    push!(back_layouts, nothing)
                    continue
                end
                stage, back, frontier = Staged[mask]
                stage = (stage[1], stage[2], (Primal[3], frontier...))
                stage_name = _jax_private_identifier(ctx.mod.used, Symbol(name, "_stage_", mask))
                back_name = _jax_private_identifier(ctx.mod.used, Symbol(name, "_back_", mask))
                stage_text, result = _jax_rule_function(ctx, stage_name, stage,
                    rule.staged_ops[mask][1], types; owner=name, layouts, pack=true, guarded=true)
                push!(ctx.mod.helpers, stage_text)
                stage_layout = last(_jax_port_pack(ctx, result))
                first(stage_layout) == out_layout ||
                    _jax_error("a derivative-rule reverse stage with an incompatible primal layout", _jax_where(ctx))
                back_types = (fieldtypes(result.T)[2:end]..., fieldtype(result.T, 1))
                back_ret = back[3] isa Symbol ? (back[3],) : back[3]
                back_text, back_value = _jax_rule_function(ctx, back_name,
                    (back[1], back[2], back_ret), rule.staged_ops[mask][2], back_types;
                    owner=name, layouts=(stage_layout[2:end]..., out_layout), pack=true, guarded=true)
                push!(ctx.mod.helpers, back_text)
                push!(back_layouts, last(_jax_port_pack(ctx, back_value)))
                push!(pairs, "(" * stage_name * ", " * back_name * ")")
            end
            stages = "(" * join(pairs, ", ") * ",)"
        end
        params = ["x" * string(i) for i in 1:N]
        push!(params, "active")
        arguments = "(" * join(params, ", ") * ")"
        push!(ctx.mod.helpers, "def " * name * "(" * join(params, ", ") * "):\n" *
            "    rule = _RKDerivativeRule(" * primal * ", " * forward * ", " * stages * ", " * string(D) *
            ", " * _jax_rule_layout_code((layouts..., nothing)) * ", " * _jax_rule_layout_code(out_layout) *
            ", " * _jax_rule_layout_code(Tuple(back_layouts)) *
            ", " * arguments * ")\n    return rule(*" * arguments * ")")
        cached = (name, _jax_argument_type(primal_value), out_layout, _JAX_EFFECTS[] - effects_before)
        ctx.mod.helper_names[key] = cached
    else
        _JAX_EFFECTS[] += cached[4]
    end
    name, result_type, out_layout, _ = cached
    guard = something(_jax_active_condition(ctx), "True")
    result = _jax_bind!(ctx, :value,
        _PyVal(name * "(" * join([first.(packed); guard], ", ") * ")", Any))
    _jax_port_unpack(ctx, result.code, result_type, out_layout)
end
