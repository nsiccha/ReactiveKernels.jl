# Included in modules that export indexed density events. Pull one or more
# requested output positions through the traced array operations. JAX still
# executes and differentiates the numerical primitives themselves.
from contextlib import contextmanager
import numpy as np
from jax.extend import core as _rk_core


def _rk_coordinates(index, shape):
    return tuple(0 if n == 1 else k for k, n in zip(index, shape))


def _rk_child_index(index, dims):
    return tuple(index[i] for i in dims)


def _rk_lane_shape(index):
    return jnp.broadcast_shapes(*(jnp.shape(i) for i in index))


def _rk_unravel(flat, shape):
    parts = []
    for size in reversed(shape):
        parts.append(flat % size if size > 1 else 0)
        flat = flat // size if size > 1 else flat
    return tuple(reversed(parts))


def _rk_static_index(x):
    return isinstance(x, (int, np.integer))


def _rk_spread(k, n):
    return k if _rk_static_index(k) else jnp.reshape(k, k.shape + (1,) * n)


class _RKIndexedGraph:
    """Propagate requested output coordinates to input leaves while tracing.

    read() handles arbitrary gathered coordinates. region() retains regular
    slices, including the entire coupled input of a joint prior or transform.
    Neither path substitutes a dense evaluation for an unknown operation.
    """
    def __init__(self, closed, inputs, regions=None, parent=None):
        graph = closed.jaxpr
        if graph.effects:
            raise NotImplementedError("indexed JAX evaluation needs an effect-aware row rule for " +
                                      str(graph.effects))
        self.parent = parent
        self.inputs = dict(zip(graph.invars, inputs))
        self.regions = dict(zip(graph.invars, inputs if regions is None else regions))
        for var, value in zip(graph.constvars, closed.consts):
            self.inputs[var] = self.regions[var] = self.leaf(value)
        self.producers = {var: (eq, j) for eq in graph.eqns for j, var in enumerate(eq.outvars)}
        self.outputs = graph.outvars
        self.cache = {}

    @staticmethod
    def leaf(value):
        def read(index):
            return jnp.asarray(value)[index] if index else jnp.asarray(value)
        return read

    def read(self, var, index):
        if isinstance(var, _rk_core.Literal):
            return self.leaf(var.val)(index)
        # Keep references to index tracers in the entry as well as their ids.
        key = (var, tuple(int(k) if _rk_static_index(k) else ('index', id(k)) for k in index))
        if key not in self.cache:
            if var in self.inputs:
                value = self.inputs[var](index)
            else:
                eq, j = self.producers[var]
                value = self.evaluate(eq, j, index)
            self.cache[key] = (index, value)
        return self.cache[key][1]

    def nested(self, eq):
        return self.child(eq.params['jaxpr'], eq.invars)

    def child(self, closed, args):
        return _RKIndexedGraph(closed,
            [lambda index, v=v: self.read(v, index) for v in args],
            [lambda slices, v=v: self.region(v, tuple(
                range(n)[s] for n, s in zip(v.aval.shape, slices))) for v in args], self)

    @contextmanager
    def branch_scope(self):
        # Values first traced in an arm belong to that arm's jaxpr.
        graph, saved = self, []
        while graph is not None:
            saved.append((graph, graph.cache))
            graph.cache = {}
            graph = graph.parent
        try:
            yield
        finally:
            for graph, cache in saved:
                graph.cache = cache

    def conditional(self, eq, output, read):
        def arm(closed):
            def run():
                with self.branch_scope():
                    graph = self.child(closed, eq.invars[1:])
                    return read(graph, graph.outputs[output])
            return run
        return jax.lax.switch(self.read(eq.invars[0], ()),
                              [arm(branch) for branch in eq.params['branches']])

    def region(self, var, axes):
        """Read a rectangular part of a joint event, retaining its coupled operations."""
        slices = tuple(slice(a.start, a.stop, a.step) for a in axes)
        if isinstance(var, _rk_core.Literal):
            return self.leaf(var.val)(slices)
        if var in self.regions:
            return self.regions[var](slices)
        key = (var, axes)
        if key not in self.cache:
            eq, output = self.producers[var]
            self.cache[key] = (axes, self.rectangular(eq, output, axes))
        return self.cache[key][1]

    def rectangular(self, eq, output, axes):
        name, p, args = eq.primitive.name, eq.params, eq.invars
        whole = lambda v: tuple(range(n) for n in v.aval.shape)
        shape = tuple(len(a) for a in axes)
        if name == 'cond':
            return self.conditional(eq, output, lambda g, v: g.region(v, axes))
        if name in ('jit', 'pjit'):
            nested = self.nested(eq)
            return nested.region(nested.outputs[output], axes)
        if name == 'slice':
            strides = p['strides'] or (1,) * len(axes)
            parent = tuple(range(start + a.start * step, start + a.stop * step, a.step * step)
                           for a, start, step in zip(axes, p['start_indices'], strides))
            return self.region(args[0], parent)
        if name in _RK_POINTWISE:
            def part(v):
                dims = axes[len(axes)-len(v.aval.shape):]
                return self.region(v, tuple(range(1) if n == 1 else a
                                            for n, a in zip(v.aval.shape, dims)))
            return eq.primitive.bind(*jnp.broadcast_arrays(*(part(v) for v in args)), **p)
        if name in _RK_REDUCTIONS:
            kept = iter(axes)
            parent = tuple(range(n) if i in p['axes'] else next(kept)
                           for i, n in enumerate(args[0].aval.shape))
            return eq.primitive.bind(self.region(args[0], parent), **p)
        if name == 'concatenate':
            d = p['dimension']
            a = axes[d]
            parts, offset = [], 0
            for v in args:
                width = v.aval.shape[d]
                start = max(0, (offset - a.start + a.step - 1) // a.step)
                end = max(0, (offset + width - a.start + a.step - 1) // a.step)
                chosen = a[start:end]
                if chosen:
                    local = range(chosen.start-offset, chosen.stop-offset, chosen.step)
                    parts.append(self.region(v, axes[:d] + (local,) + axes[d+1:]))
                offset += width
            return jnp.concatenate(parts, axis=d) if parts else jnp.empty(shape, eq.outvars[output].aval.dtype)
        if name in ('cumsum', 'cumprod'):
            d, a = p['axis'], axes[p['axis']]
            if not a:
                return jnp.empty(shape, eq.outvars[output].aval.dtype)
            source = (range(a[0], args[0].aval.shape[d]) if p['reverse']
                      else range(0, a[-1] + 1))
            values = self.region(args[0], axes[:d] + (source,) + axes[d+1:])
            scan = eq.primitive.bind(values, **p)
            return jax.lax.slice_in_dim(scan, a[0]-source.start, a[-1]-source.start+1, a.step, axis=d)
        if name in ('dot_general', 'triangular_solve', 'cholesky') and axes == whole(eq.outvars[output]):
            # All entries of this coupled result are needed by the atomic event.
            return eq.primitive.bind(*(self.region(v, whole(v)) for v in args), **p)
        grid = jnp.ix_(*(jnp.arange(a.start, a.stop, a.step) for a in axes)) if axes else ()
        return jnp.broadcast_to(self.evaluate(eq, output, grid), shape)

    def evaluate(self, eq, output, index):
        name = eq.primitive.name
        p = eq.params
        args = eq.invars
        shape = eq.outvars[output].aval.shape

        def arg(i, coords):
            return self.read(args[i], coords)

        if name == 'cond':
            # A broadcast arm can supply the requested coordinates with a
            # scalar or a smaller grid. The switch must join equal shapes.
            lane_shape = _rk_lane_shape(index)
            return self.conditional(eq, output,
                                    lambda g, v: jnp.broadcast_to(g.read(v, index), lane_shape))
        if name in ('jit', 'pjit'):
            nested = self.nested(eq)
            return nested.read(nested.outputs[output], index)
        if name == 'broadcast_in_dim':
            return arg(0, _rk_coordinates(_rk_child_index(index, p['broadcast_dimensions']),
                                          args[0].aval.shape))
        if name == 'slice':
            strides = p['strides'] or (1,) * len(index)
            return arg(0, tuple(i * step + start for i, step, start in
                                zip(index, strides, p['start_indices'])))
        if name == 'dynamic_slice':
            starts = [jnp.clip(arg(j+1, ()), 0, n - width)
                      for j, (n, width) in enumerate(zip(args[0].aval.shape, p['slice_sizes']))]
            return arg(0, tuple(start + i for start, i in zip(starts, index)))
        if name == 'squeeze':
            values = iter(index)
            return arg(0, tuple(0 if d in p['dimensions'] else next(values)
                                for d in range(len(args[0].aval.shape))))
        if name == 'transpose':
            return arg(0, _rk_child_index(index, np.argsort(p['permutation'])))
        if name == 'reshape':
            flat = 0
            for i, n in zip(index, shape):
                flat = flat * n + i
            order = p['dimensions'] or tuple(range(len(args[0].aval.shape)))
            orig = _rk_unravel(flat, tuple(args[0].aval.shape[d] for d in order))
            return arg(0, _rk_child_index(orig, np.argsort(order)))
        if name == 'iota':
            return jnp.asarray(index[p['dimension']], dtype=p['dtype'])
        if name == 'concatenate':
            axis = p['dimension']
            at = index[axis]
            if _rk_static_index(at):
                for j, v in enumerate(args):
                    width = v.aval.shape[axis]
                    if at < width:
                        return arg(j, index[:axis] + (at,) + index[axis+1:])
                    at -= width
            raise NotImplementedError("indexed JAX concatenation needs a statically selected field")
        if name == 'gather':
            dims = p['dimension_numbers']
            indices = tuple(i for j, i in enumerate(index) if j not in dims.offset_dims)
            starts = [0] * len(args[0].aval.shape)
            for j, d in enumerate(dims.start_index_map):
                start = arg(1, indices + (j,))
                if p['mode'] not in (jax.lax.GatherScatterMode.CLIP,
                                     jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS):
                    raise NotImplementedError("indexed JAX gather mode " + str(p['mode']))
                starts[d] = jnp.clip(start, 0, args[0].aval.shape[d] - p['slice_sizes'][d])
            for d, e in zip(dims.operand_batching_dims, dims.start_indices_batching_dims):
                starts[d] = indices[e]
            offsets = iter(_rk_child_index(index, dims.offset_dims))
            return arg(0, tuple(s if d in dims.collapsed_slice_dims or d in dims.operand_batching_dims
                                else s + next(offsets) for d, s in enumerate(starts)))
        if name == 'dot_general':
            contractions, batches = p['dimension_numbers']
            lane_shape = _rk_lane_shape(index)
            widths = tuple(args[0].aval.shape[d] for d in contractions[0])
            grids = jnp.ix_(*(jnp.arange(n) for n in widths)) if widths else ()
            field = len(batches[0])
            operands = []
            for side, var in enumerate(args):
                coords = []
                for d in range(len(var.aval.shape)):
                    if d in contractions[side]:
                        coords.append(grids[contractions[side].index(d)])
                    elif d in batches[side]:
                        coords.append(_rk_spread(index[batches[side].index(d)], len(widths)))
                    else:
                        coords.append(_rk_spread(index[field], len(widths)))
                        field += 1
                values = arg(side, tuple(coords))
                operands.append(jnp.broadcast_to(values, lane_shape + widths))
            batch = tuple(range(len(lane_shape)))
            contract = tuple(range(len(batch), len(batch) + len(widths)))
            dims = ((contract, contract), (batch, batch))
            return eq.primitive.bind(*operands, **dict(p, dimension_numbers=dims))
        if name in _RK_REDUCTIONS:
            axes = tuple(sorted(p['axes']))
            widths = tuple(args[0].aval.shape[d] for d in axes)
            m = len(axes)
            lane_shape = _rk_lane_shape(index)
            coordinates = []
            kept = iter(index)
            for d in range(len(args[0].aval.shape)):
                if d in axes:
                    k = axes.index(d)
                    coordinates.append(jnp.arange(widths[k]).reshape((1,)*k + (widths[k],) + (1,)*(m-k-1)))
                else:
                    coordinates.append(_rk_spread(next(kept), m))
            values = jnp.broadcast_to(arg(0, tuple(coordinates)), lane_shape + widths)
            return eq.primitive.bind(values, **dict(p, axes=tuple(range(len(lane_shape), values.ndim))))
        if name in _RK_POINTWISE:
            operands = []
            for j, v in enumerate(args):
                rank = len(v.aval.shape)
                coords = _rk_coordinates(index[len(index)-rank:], v.aval.shape)
                operands.append(arg(j, coords))
            operands = jnp.broadcast_arrays(*operands)
            return eq.primitive.bind(*operands, **p)
        raise NotImplementedError("indexed JAX evaluation needs a row rule for " + name)


_RK_REDUCTIONS = {'reduce_sum', 'reduce_max', 'reduce_min', 'reduce_prod', 'reduce_or', 'reduce_and'}


_RK_POINTWISE = {
    'abs', 'acos', 'acosh', 'add', 'and', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
    'cbrt', 'ceil', 'clamp', 'conj', 'convert_element_type',
    'cos', 'cosh', 'digamma', 'div', 'eq', 'erf', 'erfc', 'erf_inv', 'exp', 'expm1',
    'floor', 'ge', 'gt', 'imag', 'integer_pow', 'is_finite', 'le', 'lgamma', 'log',
    'log1p', 'logistic', 'lt', 'max', 'min', 'mul', 'ne', 'neg', 'not', 'or',
    'pow', 'real', 'rem', 'round', 'rsqrt', 'select_n', 'sign', 'sin', 'sinh',
    'sqrt', 'square', 'stop_gradient', 'sub', 'tan', 'tanh', 'xor',
}


def _rk_strict_batch(fn):
    # A vmapped cond predicates both arms. Preserve a ragged row guard across
    # vmap in both passes, using the pullback JAX generates from the same body.
    sequential = jax.custom_batching.sequential_vmap
    result = jax.custom_vjp(sequential(fn))
    def forward(*args):
        return fn(*args), args
    def backward(args, seed):
        _, pullback = jax.vjp(fn, *args)
        return pullback(seed)
    result.defvjp(sequential(forward), sequential(backward))
    return result


def _rk_has_cond(closed):
    for eq in closed.jaxpr.eqns:
        if eq.primitive.name == 'cond':
            return True
        if eq.primitive.name in ('jit', 'pjit') and _rk_has_cond(eq.params['jaxpr']):
            return True
    return False


def _rk_read_event(fn, u, row, prepared):
    closed = jax.make_jaxpr(lambda point, data: fn(point, **data))(u, prepared)
    def selected(point, index, data):
        inputs = [_RKIndexedGraph.leaf(v) for v in jax.tree.leaves((point, data))]
        return _rk_selected(_RKIndexedGraph(closed, inputs), index)
    if _rk_has_cond(closed):
        return _rk_strict_batch(selected)(u, row, prepared)
    return selected(u, row, prepared)


def _rk_selected(graph, row):
    output, = graph.outputs
    shape = output.aval.shape
    if not shape:
        return graph.region(output, ())
    coords = tuple(reversed(_rk_unravel(row, tuple(reversed(shape)))))
    return jnp.broadcast_to(graph.read(output, coords), jnp.shape(row))
