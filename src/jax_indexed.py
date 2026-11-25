# Included in modules that export indexed density events. Pull one or more
# requested output positions through the traced array operations. JAX still
# executes and differentiates the numerical primitives themselves.
from contextlib import contextmanager
import math
import numpy as np
from jax.extend import core as _rk_core


# Keep these generated unary elementwise helpers visible as calls in the
# event trace. Reapplying the callable to selected inputs retains JAX's
# registered derivatives, including the custom rule inside softplus.
_RK_UNARY_CALLS = {name: globals()[name] for name in ('_rk_softplus', '_rk_log1mlogistic')
                   if name in globals()}
for _name, _fn in _RK_UNARY_CALLS.items():
    globals()[_name] = jax.jit(_fn)


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


def _rk_requested_slots(index, shape):
    """Index a table by the requested flat coordinates, without sorting them.

    Used only when the table is smaller than the request. Inactive slots
    read the first requested entry so an invalid, unselected input cannot
    enter the value or its pullback.
    """
    flat = 0
    for k, n in zip(index, shape):
        flat = flat * n + jnp.asarray(k)
    size = math.prod(shape)
    flat = jnp.broadcast_to(flat, _rk_lane_shape(index))
    keys = flat.ravel()
    # Repeated keys write the same coordinate to the same slot.
    slots = jnp.full(size, keys[0], dtype=keys.dtype).at[keys].set(keys)
    return slots, flat


def _rk_independent_axis(shape, index):
    """Find a repeated row coordinate broadcast against an independent grid."""
    lanes = _rk_lane_shape(index)
    axes = [tuple(d for d, n in enumerate(jnp.shape(k), len(lanes) - jnp.ndim(k)) if n > 1)
            for k in index]
    for axis, varying in enumerate(axes):
        if len(varying) == 1:
            lane, = varying
            others = [dims for i, dims in enumerate(axes) if i != axis]
            if lanes[lane] > shape[axis] and any(others) and all(lane not in dims for dims in others):
                return axis, lane
    return None


def _rk_static_index(x):
    return isinstance(x, (int, np.integer))


def _rk_spread(k, n):
    return k if _rk_static_index(k) else jnp.reshape(k, k.shape + (1,) * n)


def _rk_point_dependence(jaxpr, reads):
    """Whether each variable of `jaxpr` depends on the packed point, given
    whether each of its inputs does."""
    dependent = dict(zip(jaxpr.invars, reads))
    dependent.update((v, False) for v in jaxpr.constvars)
    for eq in jaxpr.eqns:
        flag = any(dependent[v] for v in eq.invars if not isinstance(v, _rk_core.Literal))
        dependent.update((v, flag) for v in eq.outvars)
    return dependent


def _rk_input_dependence(eq, dependent):
    return [not isinstance(v, _rk_core.Literal) and dependent[v] for v in eq.invars]


def _rk_runs_effect(eq, dependent, mode):
    """Whether `eq` holds an effect that runs in this mode: 'data' runs the
    effects that read no parameter, 'point' those that do, 'all' every one.
    An effect under a condition on the point depends on the point."""
    if not eq.effects:
        return False
    reads = _rk_input_dependence(eq, dependent)
    if eq.primitive.name == 'cond':
        if reads[0]:
            return mode != 'data'
        return any(_rk_has_effect(b, reads[1:], mode) for b in eq.params['branches'])
    if eq.primitive.name in ('jit', 'pjit'):
        return _rk_has_effect(eq.params['jaxpr'], reads, mode)
    return mode == 'all' or any(reads) == (mode == 'point')


def _rk_has_effect(closed, reads, mode):
    dependent = _rk_point_dependence(closed.jaxpr, reads)
    return any(_rk_runs_effect(eq, dependent, mode) for eq in closed.jaxpr.eqns)


def _rk_run_effects(closed, args, reads, mode, wanted=None):
    """Run the effects of a traced event, such as its runtime checks, over the
    whole event, evaluating only what they (and the `wanted` outputs) need.

    A selected read pulls requested rows of the event's value, so it never
    reaches an equation without outputs. A check of the data has the same
    outcome at every point: prepare_partition runs those once (mode 'data',
    the point's entry of `args` is None). Each selected read runs the checks
    that read the point (mode 'point'), as the full density runs every check
    at each evaluation. `reads` says which inputs depend on the point.
    Returns the outputs, None where not wanted.
    """
    jaxpr = closed.jaxpr
    dependent = _rk_point_dependence(jaxpr, reads)
    wanted = [False] * len(jaxpr.outvars) if wanted is None else wanted
    needed = {v for v, w in zip(jaxpr.outvars, wanted) if w and not isinstance(v, _rk_core.Literal)}
    equations = []
    for eq in reversed(jaxpr.eqns):
        runs = _rk_runs_effect(eq, dependent, mode)
        if runs or any(v in needed for v in eq.outvars):
            equations.append((eq, runs))
            needed.update(v for v in eq.invars if not isinstance(v, _rk_core.Literal))
    env = dict(zip(jaxpr.constvars, closed.consts))
    env.update(zip(jaxpr.invars, args))
    read = lambda v: v.val if isinstance(v, _rk_core.Literal) else env.get(v)
    for eq, runs in reversed(equations):
        name, values = eq.primitive.name, [read(v) for v in eq.invars]
        inner = _rk_input_dependence(eq, dependent)
        outputs = [v in needed for v in eq.outvars]
        if runs and name == 'cond':
            branches = eq.params['branches']
            arm_mode = 'all' if inner[0] else mode
            def arm(branch):
                return lambda: _rk_run_effects(branch, values[1:], inner[1:], arm_mode, outputs)
            if mode == 'data':
                # The data are concrete here: run the arm the condition takes.
                results = arm(branches[int(np.clip(values[0], 0, len(branches) - 1))])()
            else:
                results = jax.lax.switch(values[0], [arm(b) for b in branches])
        elif runs and name in ('jit', 'pjit'):
            results = _rk_run_effects(eq.params['jaxpr'], values, inner, mode, outputs)
        elif mode == 'data' and any(inner):
            # Needs the point, which preparation does not have.
            results = [None] * len(eq.outvars)
        else:
            results = eq.primitive.bind(*values, **eq.primitive.get_bind_params(eq.params))
            results = results if eq.primitive.multiple_results else [results]
        env.update(zip(eq.outvars, results))
    return [read(v) if w else None for v, w in zip(jaxpr.outvars, wanted)]


def _rk_event_trace(fn, point, prepared):
    """The event's trace with the data as inputs, the data's leaves, and
    which inputs depend on the point."""
    closed = jax.make_jaxpr(lambda u, data: fn(u, **data))(point, prepared)
    data = jax.tree.leaves(prepared)
    return closed, data, [True] + [False] * len(data)


def _rk_check_event_data(fn, point, prepared):
    """Run the event's checks of the data once, at their values in `prepared`."""
    closed, data, reads = _rk_event_trace(fn, point, prepared)
    _rk_run_effects(closed, [None] + [jnp.asarray(v) for v in data], reads, 'data')
    return closed


class _RKIndexedGraph:
    """Propagate requested output coordinates to input leaves while tracing.

    read() handles arbitrary gathered coordinates. region() retains regular
    slices, including the entire coupled input of a joint prior or transform.
    Neither path substitutes a dense evaluation for an unknown operation.
    An equation that has no output, such as a runtime check, is never pulled:
    _rk_run_effects runs those.
    """
    def __init__(self, closed, inputs, regions=None, parent=None, first_axis=None):
        graph = closed.jaxpr
        self.parent = parent
        # Preparation and evaluation visit the same graph in the same order.
        # Its children share the one planned read, including inside jit calls.
        self.first_axis = parent.first_axis if parent is not None else ([first_axis] if first_axis else [])
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
            elif index and math.prod(_rk_lane_shape(index)) > math.prod(var.aval.shape) > 0:
                # More requested entries than the value has, as when many
                # rows read one shared factor: compute each distinct one once.
                value = self.distinct(var, index)
            else:
                eq, j = self.producers[var]
                value = self.evaluate(eq, j, index)
            self.cache[key] = (index, value)
        return self.cache[key][1]

    def distinct(self, var, index):
        """Read the distinct requested entries of a computed value once each.

        Only requested entries are computed: an unrequested one may be
        invalid, such as a padded row, and its derivative must not enter.
        """
        shape = var.aval.shape
        if self.first_axis:
            axes = _rk_independent_axis(shape, index)
            if axes is not None:
                axis, lane = axes
                plan = self.first_axis.pop()(index[axis])
                if plan is not None:
                    keys, inverse = plan
                    lanes = _rk_lane_shape(index)
                    grid = [1] * len(lanes)
                    grid[lane] = keys.size
                    compact = index[:axis] + (keys.reshape(grid),) + index[axis + 1:]
                    eq, output = self.producers[var]
                    values = jnp.broadcast_to(self.evaluate(eq, output, compact), _rk_lane_shape(compact))
                    if keys.size == 1:
                        return jnp.broadcast_to(values, lanes)
                    return jnp.take(values, inverse, axis=lane, mode='clip')
        slots, requested = _rk_requested_slots(index, shape)
        eq, j = self.producers[var]
        values = jnp.broadcast_to(self.evaluate(eq, j, _rk_unravel(slots, shape)), slots.shape)
        return values[requested]

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

    def whole(self, eq):
        """All outputs of an operation that couples all of its inputs, computed once."""
        key = ('whole', id(eq))
        if key not in self.cache:
            values = eq.primitive.bind(*(self.region(v, tuple(range(n) for n in v.aval.shape))
                                         for v in eq.invars), **eq.primitive.get_bind_params(eq.params))
            self.cache[key] = (eq, values if eq.primitive.multiple_results else [values])
        return self.cache[key][1]

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
            fn = _RK_UNARY_CALLS.get(p['name'])
            if fn is not None:
                return fn(self.region(args[0], axes))
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

        if name == 'platform_index':
            # The device selector is a scalar with no inputs.
            return eq.primitive.bind(**p)
        if name == 'cond':
            # A broadcast arm can supply the requested coordinates with a
            # scalar or a smaller grid. The switch must join equal shapes.
            lane_shape = _rk_lane_shape(index)
            return self.conditional(eq, output,
                                    lambda g, v: jnp.broadcast_to(g.read(v, index), lane_shape))
        if name in ('jit', 'pjit'):
            fn = _RK_UNARY_CALLS.get(p['name'])
            if fn is not None:
                return fn(arg(0, index))
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
            # Read every part at the requested positions it could hold, and
            # keep the one that holds each.
            lane_shape, result, offset = _rk_lane_shape(index), None, 0
            for j, v in enumerate(args):
                width = v.aval.shape[axis]
                if width:
                    local = jnp.clip(at - offset, 0, width - 1)
                    part = jnp.broadcast_to(arg(j, index[:axis] + (local,) + index[axis+1:]), lane_shape)
                    result = part if result is None else jnp.where(at >= offset, part, result)
                offset += width
            return result
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
        if name in _RK_CUMULATIVE:
            # Scan the whole axis at each requested position's other
            # coordinates, then pick the position.
            axis, n = p['axis'], shape[p['axis']]
            lane_shape = _rk_lane_shape(index)
            coords = tuple(jnp.arange(n) if d == axis else _rk_spread(k, 1) for d, k in enumerate(index))
            values = jnp.broadcast_to(arg(0, coords), lane_shape + (n,))
            scanned = eq.primitive.bind(values, **dict(p, axis=len(lane_shape)))
            at = jnp.broadcast_to(index[axis], lane_shape)
            return jnp.take_along_axis(scanned, at[..., None], axis=-1)[..., 0]
        if name == 'cholesky':
            # A factor couples its two trailing axes. Factor each requested
            # matrix of the batch whole, once, then pick the requested entries.
            lane_shape = _rk_lane_shape(index)
            rank, n, dims = len(lane_shape), shape[-1], shape[:-2]
            batch = [jnp.reshape(b, (1,) * (rank - jnp.ndim(b)) + jnp.shape(b))
                     for b in index[:-2]]
            square = (jnp.arange(n).reshape(n, 1), jnp.arange(n).reshape(1, n))
            requested = jnp.broadcast_shapes(*(jnp.shape(b) for b in batch)) if batch else ()
            if math.prod(requested) > math.prod(dims):
                # More requested matrices than the batch has: factor each
                # distinct one once.
                slots, entries = _rk_requested_slots(batch, dims)
                matrices = jnp.broadcast_to(
                    arg(0, tuple(_rk_spread(c, 2) for c in _rk_unravel(slots, dims)) + square),
                    slots.shape + (n, n))
                factors = eq.primitive.bind(matrices, **p)[entries]
            else:
                factors = eq.primitive.bind(arg(0, tuple(_rk_spread(b, 2) for b in batch) + square), **p)
            selected = factors.shape[:-2]
            lead = tuple(0 if m == 1 else jnp.arange(m).reshape((1,) * d + (m,) + (1,) * (rank - d - 1))
                         for d, m in enumerate(selected))
            return jnp.broadcast_to(factors[lead + tuple(index[-2:])], lane_shape)
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
        if name in _RK_WHOLE:
            # A loop's carry or a scatter's writes couple every entry.
            value = self.whole(eq)[output]
            return value[index] if index else value
        raise NotImplementedError("indexed JAX evaluation needs a row rule for " + name)


_RK_CUMULATIVE = {'cumsum', 'cumprod', 'cummax', 'cummin', 'cumlogsumexp'}


# Evaluated whole when any entry is read: a loop inside a plate's cell is
# evaluated for every cell of the event.
_RK_WHOLE = {'scan', 'while', 'scatter', 'scatter-add', 'scatter-mul', 'scatter-min', 'scatter-max'}


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
    # Only the first argument, the packed point, is differentiated. The
    # residuals are that pullback's, the selected rows' values: the data never
    # pass through as residuals or dense zero cotangents.
    sequential = jax.custom_batching.sequential_vmap
    def batched(*args):
        others = len(args) - 1
        def forward(point, *rest):
            return jax.vjp(lambda p: fn(p, *rest), point)
        def backward(pullback, seed):
            return (*pullback(seed), *([None] * others))
        result = jax.custom_vjp(sequential(fn))
        result.defvjp(sequential(forward), sequential(backward))
        return result(*args)
    return batched


def _rk_scan_control(eq, reads):
    # JAX's partial evaluator propagates input dependence across iterations.
    # A carry can first depend on a lane after the initial iteration.
    from jax.interpreters.partial_eval import partial_eval_jaxpr_nounits
    params = eq.params
    def scan(*args):
        return eq.primitive.bind(*args, **eq.primitive.get_bind_params(params))
    inputs = [jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in eq.invars]
    try:
        _, _, outputs, _ = partial_eval_jaxpr_nounits(jax.make_jaxpr(scan)(*inputs),
                                                      reads, instantiate=False)
    except NotImplementedError:
        return True, ()
    if 'ft_in' in params:
        consts, carry, _ = params['ft_in'].update(reads).unpack()
        begin, end = len(consts), len(consts) + len(carry)
    else:
        begin = params['num_consts']
        end = begin + params['num_carry']
    carry = [initial or final for initial, final in zip(reads[begin:end], outputs)]
    body_inputs = reads[:begin] + carry + reads[end:]
    guard, _ = _rk_batch_control(params['jaxpr'], body_inputs)
    return guard, carry + outputs[len(carry):]


def _rk_batch_control(closed, varying):
    """Can a selected read take different branches in different lanes?

    A platform switch has a shared selector even when its operands vary.
    Inspect its arms and nested loops before allowing JAX to batch the read.
    The second result tracks lane-dependent outputs for the enclosing call.
    """
    graph = getattr(closed, 'jaxpr', closed)
    flags = dict(zip(graph.invars, varying))
    flags.update((v, True) for v in graph.constvars)
    for eq in graph.eqns:
        name, p = eq.primitive.name, eq.params
        reads = _rk_input_dependence(eq, flags)
        out = [any(reads)] * len(eq.outvars)
        if name == 'cond':
            if 'branches_platforms' not in p and reads[0]:
                return True, ()
            arms = [_rk_batch_control(branch, reads[1:]) for branch in p['branches']]
            if any(guard for guard, _ in arms):
                return True, ()
            out = list(map(any, zip(*(values for _, values in arms))))
        elif name in ('jit', 'pjit'):
            guard, out = _rk_batch_control(p['jaxpr'], reads)
            if guard:
                return True, ()
        elif name == 'scan':
            guard, out = _rk_scan_control(eq, reads)
            if guard:
                return True, ()
        elif name == 'while' or (not reads and name not in ('platform_index', 'iota')):
            return True, ()
        elif _rk_params_have_cond(p) and not getattr(p.get('rule'), '_rk_checks_all_lanes', False):
            # _rk_check reduces across the mapped lanes before branching.
            # An unfamiliar higher-order primitive owns its batching rule.
            return True, ()
        flags.update(zip(eq.outvars, out))
    return False, [not isinstance(v, _rk_core.Literal) and flags[v] for v in graph.outvars]


def _rk_params_have_cond(params):
    for param in params.values():
        for sub in param if isinstance(param, (tuple, list)) else (param,):
            graph = getattr(sub, 'jaxpr', sub)
            if hasattr(graph, 'eqns') and _rk_has_cond(graph):
                return True
    return False


def _rk_has_cond(closed):
    graph = getattr(closed, 'jaxpr', closed)
    return any(eq.primitive.name == 'cond' or _rk_params_have_cond(eq.params) for eq in graph.eqns)


def _rk_prepare_axis(closed, point, prepared, rows, offsets, common):
    """Plan the first oversized Cartesian row gather in the common rows.

    Trace its row coordinates with the point and data as separate inputs.
    Only a point-independent index can be shared by all draws at a unit.
    Conditional reads retain their ordinary branch-local indexing.
    """
    if not common or closed.effects or _rk_has_cond(closed):
        return ()

    def coordinates(u, row, data):
        found = []
        inputs = [_RKIndexedGraph.leaf(v) for v in jax.tree.leaves((u, data))]
        _rk_selected(_RKIndexedGraph(closed, inputs, first_axis=found.append), row)
        return tuple(map(jnp.ravel, found))

    prototype = jax.ShapeDtypeStruct((common,), rows.dtype)
    try:
        trace = jax.make_jaxpr(coordinates)(point, prototype, prepared)
    except NotImplementedError:
        # The ordinary reader reports unsupported operations when called.
        return ()
    if not trace.jaxpr.outvars:
        return ()
    source, = trace.jaxpr.outvars
    reads = [True, False] + [False] * len(jax.tree.leaves(prepared))
    if not isinstance(source, _rk_core.Literal) and _rk_point_dependence(trace.jaxpr, reads)[source]:
        return ()
    def unit_indices(selected, data):
        return _rk_run_effects(trace, [None, selected, *jax.tree.leaves(data)], reads, 'data', [True])[0]
    selected = rows[offsets[:-1, None] + np.arange(common)]
    indices = np.asarray(jax.vmap(unit_indices, in_axes=(0, None))(selected, prepared))
    distinct = [np.unique(row, return_inverse=True) for row in indices]
    width = max(keys.size for keys, _ in distinct)
    keys = np.stack([np.pad(k, (0, width - k.size), mode='edge') for k, _ in distinct])
    # A singleton row broadcasts directly. No per-observation inverse is needed.
    inverse = np.stack([inv for _, inv in distinct]) if width > 1 else np.empty((len(keys), 0), dtype=int)
    return keys, inverse


def _rk_read_event(fn, u, row, prepared, axis_plan=()):
    # The event's checks of the data ran once in prepare_partition. Those
    # that read the point run here, over the whole event.
    closed, _, reads = _rk_event_trace(fn, u, prepared)
    checks = _rk_has_effect(closed, reads, 'point')
    def selected(point, index, data):
        leaves = jax.tree.leaves((point, data))
        if checks:
            _rk_run_effects(closed, leaves, reads, 'point')
        inputs = [_RKIndexedGraph.leaf(v) for v in leaves]
        graph = _RKIndexedGraph(closed, inputs, first_axis=(lambda _: axis_plan) if axis_plan else None)
        return _rk_selected(graph, index)
    if _rk_has_cond(closed) and _rk_batch_control(closed, [True] * len(closed.jaxpr.invars))[0]:
        return _rk_strict_batch(selected)(u, row, prepared)
    return selected(u, row, prepared)


def _rk_selected(graph, row):
    output, = graph.outputs
    shape = output.aval.shape
    if not shape:
        return graph.region(output, ())
    coords = tuple(reversed(_rk_unravel(row, tuple(reversed(shape)))))
    return jnp.broadcast_to(graph.read(output, coords), jnp.shape(row))
