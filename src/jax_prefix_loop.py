# The body includes the emitter's lazy guard. Both schedules evaluate that
# same body, including captured differentiable inputs and runtime checks.
from numbers import Integral as _rk_loop_integer

try:
    from jax.experimental import hijax as _rk_loop_ad
except ImportError:
    _rk_loop_ad = None


if _rk_loop_ad is not None:
    class _RKPrefixLoop(_rk_loop_ad.HiPrim):
        skip_linearization_on_zero_tangents = True
        lin, linearized = _rk_loop_ad.linearize_from_jvp

        def __init__(self, run, bounded, args, out_aval, effects):
            self.expand = run
            self.bounded = bounded
            self.in_avals = jax.tree.map(jax.typeof, args)
            self.out_aval = out_aval
            self.effects = effects
            self.params = dict(loop=run)
            super().__init__()

        @staticmethod
        def _dense(tangents):
            return jax.tree.map(_rk_loop_ad.instantiate_zeros, tangents,
                                is_leaf=lambda t: isinstance(t, _rk_loop_ad.Zero))

        def jvp(self, primals, tangents):
            return jax.jvp(self.bounded, primals, self._dense(tangents))

        def vjp_fwd(self, nonzeros, *primals):
            return jax.vjp(self.bounded, *primals)

        def vjp_bwd_retval(self, pullback, cotangents):
            return pullback(self._dense(cotangents))

        def batch(self, axis, args, dims):
            out_dims = jax.tree.map(lambda _: 0, self.out_aval)
            run = _rk_loop_ad.vmap_rule(axis, self.expand, dims, out_dims)
            bounded = _rk_loop_ad.vmap_rule(axis, self.bounded, dims, out_dims)
            out = jax.tree.map(lambda a: _rk_loop_ad.unmap_avals(axis, a, 0), self.out_aval)
            return _RKPrefixLoop(run, bounded, args, out, self.effects)(*args), out_dims


# Triton can keep a pure scalar recurrence and its loop condition in one CUDA
# kernel. Check the traced body, since a scalar interface can hide calls,
# array accesses or effects. All other bodies run as ordinary JAX loops.
def _rk_scalar_loop(graph):
    def supported(var):
        aval = var.aval
        return (getattr(aval, "shape", None) == () and
                getattr(getattr(aval, "dtype", None), "name", None) in
                ("bool", "int32", "int64", "float64"))

    if graph.effects or not all(map(supported, (*graph.constvars, *graph.invars, *graph.outvars))):
        return False
    for eq in graph.eqns:
        name = eq.primitive.name
        if name == "cond":
            if not all(_rk_scalar_loop(branch.jaxpr) for branch in eq.params["branches"]):
                return False
        elif name not in ("abs", "add", "and", "convert_element_type", "div", "eq",
                          "ge", "gt", "integer_pow", "le", "lt", "mul", "ne", "neg",
                          "not", "or", "sub"):
            return False
        if name == "div" and eq.outvars[0].aval.dtype.kind != "f":
            return False
        if not all(map(supported, (*eq.invars, *eq.outvars))):
            return False
    return True


def _rk_scalar_step(graph, constants, inputs, predicate):
    from jax.extend.core import Literal
    env = dict(zip((*graph.constvars, *graph.invars), (*constants, *inputs)))

    def read(var):
        return var.val if isinstance(var, Literal) else env[var]

    for eq in graph.eqns:
        args = [read(var) for var in eq.invars]
        if eq.primitive.name == "cond":
            index, *operands = args
            def arm(branch):
                return lambda *xs: _rk_scalar_step(branch.jaxpr, branch.consts, xs, index != 0)
            outs = jax.lax.switch(index, tuple(map(arm, eq.params["branches"])), *operands)
        else:
            outs = [eq.primitive.bind(*args, **eq.params)]
        env.update(zip(eq.outvars, outs))

    # Triton's scalar branch terminator needs SSA values, including for a
    # Julia branch that returns a literal (e.g. the false flag on break).
    return [jax.lax.select(predicate, read(var), read(var)) if isinstance(var, Literal)
            else read(var) for var in graph.outvars]


def _rk_cuda_prefix(run, args, initial):
    try:
        from jax.experimental import pallas
        from jax.experimental.pallas import triton
    except ImportError:
        return run(*args)
    leaves, arg_tree = jax.tree.flatten(args)

    def kernel(*refs):
        values = [ref[()] for ref in refs[:len(leaves)]]
        result = run(*jax.tree.unflatten(arg_tree, values), device=True)
        for dest, value in zip(refs[len(leaves):], jax.tree.leaves(result)):
            dest[()] = value

    outputs = [jax.ShapeDtypeStruct((), jax.typeof(x).dtype) for x in jax.tree.leaves(initial)]
    call = pallas.pallas_call(kernel, grid=(1,), out_shape=outputs,
                              compiler_params=triton.CompilerParams(num_warps=1, num_stages=1))
    return jax.tree.unflatten(jax.tree.structure(initial), call(*leaves))


def _rk_prefix_fori(start, stop, body, initial, running_slot, last_step):
    if _rk_loop_ad is None:
        return jax.lax.fori_loop(start, stop, body, initial)

    # Stage once and pass every closed-over array explicitly through AD.
    closed = jax.make_jaxpr(body)(jnp.asarray(start), initial)
    # Replacing constvars detaches the captured values on JAX versions where
    # ClosedJaxpr and Jaxpr share a representation. They enter through args.
    graph = closed.jaxpr.replace(constvars=closed.jaxpr.constvars)
    tree = jax.tree.structure(initial)

    def step(i, carry, consts):
        leaves = jax.core.eval_jaxpr(graph, consts, i, *jax.tree.leaves(carry))
        return jax.tree.unflatten(tree, leaves)

    def run(lo, hi, last, carry, consts, *, device=False):
        def keep(state):
            i, values = state
            within = i < hi
            if last is not None:
                within = within & (i <= last)
            if running_slot is not None:
                within = within & values[running_slot]
            return within

        def advance(state):
            i, values = state
            if device:
                leaves = _rk_scalar_step(graph, consts, (i, *jax.tree.leaves(values)), i < hi)
                return i + 1, jax.tree.unflatten(tree, leaves)
            return i + 1, step(i, values, consts)

        return jax.lax.while_loop(keep, advance, (lo, carry))[1]

    # Preserve concrete endpoints so JAX transposes a bounded scan. A
    # genuinely dynamic range retains its ordinary JAX differentiation rules.
    static_lo = int(start) if isinstance(start, _rk_loop_integer) else None
    static_hi = int(stop) if isinstance(stop, _rk_loop_integer) else None

    def bounded(lo, hi, last, carry, consts):
        begin = lo if static_lo is None else static_lo
        end = hi if static_hi is None else static_hi
        return jax.lax.fori_loop(begin, end, lambda i, c: step(i, c, consts), carry)

    args = (start, stop, last_step, initial, tuple(closed.consts))
    execute = run
    if running_slot is not None and _rk_scalar_loop(graph):
        def execute(*xs):
            return jax.lax.platform_dependent(
                cuda=lambda: _rk_cuda_prefix(run, xs, xs[3]), default=lambda: run(*xs))

    out = jax.tree.map(jax.typeof, initial)
    return _RKPrefixLoop(execute, bounded, args, out, graph.effects)(*args)
