# Keep the costly matrix operation and its JAX-generated derivatives under
# the same lane schedule, including when grad is applied before an outer vmap.
from jax.extend import core as _rk_expm_core
from jax.interpreters import ad as _rk_expm_ad
from jax.interpreters import batching as _rk_expm_batching
from jax.interpreters import mlir as _rk_expm_mlir

_RK_EXPM_TILE_SIZE = 32
_rk_expm_map = _rk_expm_core.Primitive("rk_active_matrix_map")
_rk_expm_map.multiple_results = True


def _rk_expm_shapes(args, ndim):
    return tuple(jax.ShapeDtypeStruct(x.shape[ndim:], x.dtype) for x in args)


def _rk_expm_map_impl(active, *args, body):
    # The enclosing cond has already checked scalar activity.
    if not active.ndim:
        return body(*args)

    size = active.size
    outputs = jax.eval_shape(body, *_rk_expm_shapes(args, active.ndim))
    result = tuple(jnp.zeros((size,) + y.shape, dtype=y.dtype) for y in outputs)
    if size:
        # Compact the selected lanes on device. Padding the last tile with its
        # first live input keeps even a nonfinite inactive operand out of expm.
        width = min(_RK_EXPM_TILE_SIZE, size)
        capacity = ((size + width - 1) // width) * width
        indices = jnp.nonzero(active.reshape(-1), size=capacity, fill_value=size)[0]
        count = jnp.sum(active, dtype=indices.dtype)
        flat = tuple(x.reshape((size,) + x.shape[active.ndim:]) for x in args)

        def tile(i, buffers):
            lanes = jax.lax.dynamic_slice_in_dim(indices, i * width, width)
            valid = jnp.where(lanes < size, lanes, lanes[0])
            values = jax.vmap(body)(*(x[valid] for x in flat))
            return tuple(y.at[lanes].set(v, mode="drop") for y, v in zip(buffers, values))

        result = jax.lax.fori_loop(0, (count + width - 1) // width, tile, result)
    return tuple(y.reshape(active.shape + y.shape[1:]) for y in result)


def _rk_expm_map_abstract(active, *args, body):
    graph = jax.make_jaxpr(body)(*_rk_expm_shapes(args, active.ndim))
    return tuple(y.update(shape=active.shape + y.shape) for y in graph.out_avals), graph.effects


def _rk_expm_map_batch(args, axes, *, body):
    size = next(x.shape[d] for x, d in zip(args, axes) if d is not None)
    rows = [jnp.broadcast_to(x, (size,) + x.shape) if d is None else jnp.moveaxis(x, d, 0)
            for x, d in zip(args, axes)]
    values = _rk_expm_map.bind(*rows, body=body)
    return values, (0,) * len(values)


def _rk_expm_map_jvp(primals, tangents, *, body):
    active, *xs = primals
    # Differentiate the scalar JAX body, then schedule that program as well.
    count = len(xs)

    def push(*args):
        return jax.jvp(body, args[:count], args[count:])[1]

    values = _rk_expm_map.bind(active, *xs, body=body)
    directions = map(_rk_expm_ad.instantiate_zeros, tangents[1:])
    return values, _rk_expm_map.bind(active, *xs, *directions, body=push)


def _rk_expm_map_transpose(seeds, active, *args, body):
    # Transpose the linear call JAX made above, lane by lane. Only primal
    # inputs cross the checkpoint. Squaring histories stay inside a tile.
    unknown = tuple(_rk_expm_ad.is_undefined_primal(x) for x in args)
    saved = [x for x, u in zip(args, unknown) if not u]
    variables = _rk_expm_shapes([x.aval for x, u in zip(args, unknown) if u], active.ndim)
    count = len(saved)

    def pull(*row):
        def linear(*directions):
            known, variable = iter(row[:count]), iter(directions)
            return body(*(next(variable) if u else next(known) for u in unknown))
        return jax.linear_transpose(linear, *variables)(tuple(row[count:]))

    cotangents = map(_rk_expm_ad.instantiate_zeros, seeds)
    grads = iter(_rk_expm_map.bind(active, *saved, *cotangents, body=pull))
    return [None, *(next(grads) if u else None for u in unknown)]


_rk_expm_map.def_impl(_rk_expm_map_impl)
_rk_expm_map.def_effectful_abstract_eval(_rk_expm_map_abstract)
_rk_expm_mlir.register_lowering(_rk_expm_map, _rk_expm_mlir.lower_fun(_rk_expm_map_impl, multiple_results=True))
_rk_expm_batching.primitive_batchers[_rk_expm_map] = _rk_expm_map_batch
_rk_expm_ad.primitive_jvps[_rk_expm_map] = _rk_expm_map_jvp
_rk_expm_ad.primitive_transposes[_rk_expm_map] = _rk_expm_map_transpose


def _rk_expm_value(x):
    return (jax.scipy.linalg.expm(x),)


@jax.checkpoint
def _rk_guarded_expm(matrix, active):
    dtype = jax.eval_shape(jax.scipy.linalg.expm, matrix).dtype
    identity = jnp.broadcast_to(jnp.eye(matrix.shape[-1], dtype=dtype), matrix.shape)

    def evaluate(x):
        flags = jnp.broadcast_to(active, x.shape[:-2])
        value, = _rk_expm_map.bind(flags, x, body=_rk_expm_value)
        return jnp.where(active, value, identity)

    return jax.lax.cond(_rk_any_active(active), evaluate, lambda _: identity, matrix)
