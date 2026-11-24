# JAX translation of the native ODE bridges `rk_ode_rk45_tol` and
# `rk_ode_bdf_tol` (`native_ode.jl`): adaptive diffrax integration with the
# caller's tolerances and per-output step budget, outputs from the solver's
# dense interpolant, and reverse-mode derivatives of the discrete solve
# (diffrax's `RecursiveCheckpointAdjoint`), so a gradient's accuracy follows
# the caller's tolerances as the native forward sensitivities' does. A module
# imports diffrax only when it solves an ODE.

# `_RKOutputBudget` counts accepted steps toward the first requested output
# after the time a step starts at, and fails a solve whose count for one
# output exceeds `limit`: the native bridges' `OutputBudget`, which lets
# `while t < output` take at most `maxsteps` accepted steps per output.
# diffrax's own `max_steps` is a static bound on attempted steps that sizes
# no buffer here (`checkpoints` does, so a gradient's memory does not grow
# with it): it is the largest int32, as the native solve sets no such bound.
# A non-finite starting state or unusable step size fails the solve. RK45
# requires representable forward progress at the current time. The stiff
# controller has an explicit minimum step.
# `_rk_ode_solve` returns the times-by-states outputs and a status: 0 solved,
# 1 invalid inputs, 2 a per-output budget exceeded, 3 the solver failed. Only
# the first `n` output times are requested; later entries (a padded row's)
# repeat the last requested time, so their rows repeat its state.
#
# On CUDA (`jax.lax.platform_dependent`) the same solves run per lane (snag
# jax-ode-solve-gp-d427eb5a). An XLA while loop on a GPU pays a round of kernel
# launches and a device-to-host check per iteration, diffrax runs a step's
# Runge-Kutta stages as a nested loop, and `vmap` keeps every solve of a batch
# in lockstep, so a batched solve cost ~0.4 ms per step for its slowest lane.
# Instead `_rk_lane_kernel`, a Pallas (Triton) kernel, integrates each lane (a
# plate cell of a chain: `custom_vmap` joins every vmapped call into one
# kernel call) with its own step sizes, 32 lanes per program; only those 32
# wait for each other. Both implementations use the same Dopri5 tableau,
# first step, sensitivity error norm, PID control, dense output, output budget
# and failed-trial guard. The right-hand
# side runs from its traced jaxpr one state element at a time
# (`_rk_soa_eval`), as Triton's power-of-two blocks require. Gradients are
# forward-mode sensitivities: the kernel carries the tangents of the perturbed
# inputs through the steps (diffrax stops gradients through step sizes, so this
# is the derivative its reverse pass computes), and a custom VJP contracts them.
# `rk_ode_bdf_tol` (Kvaerno5) and a right-hand side using a primitive the
# element-wise evaluation lacks keep diffrax there, with forward-mode
# sensitivities (`_rk_ode_forward`). `RK_ODE_KERNEL`
# (environment `RK_JAX_ODE_KERNEL`) is "auto"; "off" keeps diffrax on CUDA
# too, "interpret" runs the CUDA path anywhere with Pallas' interpreter (tests).
const _JAX_ODE_SOLVE = """
import diffrax
import equinox as eqx


class _RKFiniteTrial:
    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        step = super().step
        inputs = (terms, t0, t1, y0, args, solver_state, made_jump)
        candidate, error, *_ = step(*inputs)
        finite = jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(error))

        # The accepted solution has no dependence on a discarded trial. Mask
        # its inputs in AD: a zero cotangent at its output can still multiply
        # an infinite intermediate in the Runge-Kutta stages. For values alone
        # stop_gradient is the identity and both calls have the same inputs.
        def step_input(x):
            return jnp.where(finite, x, jax.lax.stop_gradient(x)) if eqx.is_inexact_array(x) else x

        candidate, error, dense, state, result = step(*jax.tree.map(step_input, inputs))
        # PID scales error by the size of the candidate. Inf / Inf gives NaN
        # and hence a NaN step size. A finite scale and infinite error ask it
        # to retry at a smaller step. Diffrax discards the solver result on a
        # rejection; a controller that accepts this trial must report failure.
        return (jnp.where(finite, candidate, y0), jnp.where(finite, error, jnp.inf),
                dense, state, diffrax.RESULTS.where(finite, result, diffrax.RESULTS.nonfinite))


class _RKDopri5(_RKFiniteTrial, diffrax.Dopri5):
    unguarded = diffrax.Dopri5


class _RKKvaerno5(_RKFiniteTrial, diffrax.Kvaerno5):
    unguarded = diffrax.Kvaerno5


def _rk_next_trial_time(keep, t0, t1, proposed):
    # A rejection must shorten the interval even when rounding would repeat it.
    smaller = jnp.nextafter(jax.lax.stop_gradient(t1), jax.lax.stop_gradient(t0))
    return jnp.where(~keep & (proposed >= t1), smaller, proposed)


class _RKOutputBudget(diffrax.AbstractAdaptiveStepSizeController):
    # ReactiveKernels' per-output step budget: at most `limit` accepted steps
    # between consecutive requested outputs.
    inner: diffrax.PIDController
    ts: Any
    limit: Any
    check_trials: bool = eqx.field(static=True)

    @property
    def rtol(self):
        return self.inner.rtol

    @property
    def atol(self):
        return self.inner.atol

    @property
    def norm(self):
        return self.inner.norm

    def wrap(self, direction):
        return _RKOutputBudget(self.inner.wrap(direction), self.ts, self.limit, self.check_trials)

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        t, state = self.inner.init(terms, t0, t1, y0, dt0, args, func, error_order)
        target = jnp.searchsorted(self.ts, t0, side="right")
        previous_end = t0 if self.inner.dtmin is None else None
        return t, (state, jnp.zeros((), dtype=target.dtype), target, previous_end)

    def adapt_step_size(self, t0, t1, y0, y1_candidate, args, y_error, error_order,
                        controller_state):
        state, count, target, previous_end = controller_state
        keep, next_t0, next_t1, jump, state, result = self.inner.adapt_step_size(
            t0, t1, y0, y1_candidate, args, y_error, error_order, state)
        if self.inner.dtmin is None:
            next_t1 = _rk_next_trial_time(keep, t0, t1, next_t1)
        now = jnp.searchsorted(self.ts, t0, side="right")
        count = jnp.where(now == target, count, 0)
        count = jnp.where(keep, count + 1, count)
        result = diffrax.RESULTS.where(count > self.limit, diffrax.RESULTS.max_steps_reached,
                                       result)
        # A non-finite starting state or unusable next step cannot advance.
        unusable = ~jnp.isfinite(next_t1 - next_t0)
        if previous_end is not None:
            # Diffrax clips near the final time after this controller returns.
            # If that repeats a rejected interval, another retry cannot change it.
            unusable = (next_t0 < self.ts[-1]) & (unusable | (next_t1 <= next_t0) |
                                                  (~keep & (t1 == previous_end)))
            previous_end = t1
        unstable = ~jnp.all(jnp.isfinite(y0)) | unusable
        if self.check_trials:
            # Even a rejected infinite error can leave the solve successful
            # but poison its reverse pass. Inspect attempted steps here.
            unstable |= ~jnp.all(jnp.isfinite(y1_candidate)) | ~jnp.all(jnp.isfinite(y_error))
        result = diffrax.RESULTS.where(unstable, diffrax.RESULTS.nonfinite, result)
        return keep, next_t0, next_t1, jump, (state, count, now, previous_end), result


# ==== per-lane ODE solves on CUDA (snag jax-ode-solve-gp-d427eb5a) ====
import functools as _functools
import os as _os
import warnings as _warnings

try:
    from jax.experimental import pallas as _pl
    from jax.experimental.pallas import triton as _plt
except Exception:  # an old JAX: diffrax everywhere
    _pl = _plt = None

try:
    from jax.extend.core import Literal as _RKLiteral
except ImportError:
    from jax.core import Literal as _RKLiteral

# "auto": per-lane solves on CUDA, diffrax elsewhere; "off": diffrax everywhere;
# "interpret": the CUDA path on any backend, its kernel in Pallas' interpreter (tests).
RK_ODE_KERNEL = _os.environ.get("RK_JAX_ODE_KERNEL", "auto")
_RK_LANES = 32
_RK_LANE_WARPS = 1
_RK_LANE_DEBUG = None  # tests: a list receives each lane kernel call's solver steps per lane
_RK_OK, _RK_BUDGET, _RK_FAILED = 0, 2, 3


# -- one array element at a time --------------------------------------------------
# A value is a numpy object array of its own shape; its elements are `_RKLane`s (one
# element across all lanes) or constants. A function of one small state vector so
# runs on many lanes at once with no array of the state's size, as a Triton kernel's
# power-of-two blocks require.

# Elementwise primitives that lower in a float64 Triton kernel (checked on JAX
# 0.12.0.1; digamma, erf, erfc, lgamma and round do not).
_RK_SOA_ELEMENTWISE = frozenset('''
abs acos acosh add and asin asinh atan atan2 atanh cbrt ceil clamp convert_element_type cos cosh
div eq erf_inv exp exp2 expm1 floor ge gt integer_pow is_finite le log log1p
logistic lt max min mul ne neg nextafter not or pow rem rsqrt select_n
sign sin sinh sqrt square sub tan tanh xor
'''.split())
_RK_SOA_CALLS = frozenset("pjit jit closed_call core_call remat checkpoint".split())
_RK_REVERSE_CALLS = frozenset(("custom_vjp_call", "custom_vjp_call_jaxpr"))
_RK_SOA_STRUCTURAL = frozenset('''
copy copy_p broadcast_in_dim reshape squeeze expand_dims concatenate stack slice transpose rev dynamic_slice
dynamic_update_slice iota reduce_sum reduce_max reduce_min reduce_prod reduce_and reduce_or dot_general cond
'''.split())


class _RKSoAUnsupported(Exception):
    pass


class _RKLane:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v


def _rk_soa_lift(x, lane_shape):
    return x.v if isinstance(x, _RKLane) else jnp.broadcast_to(jnp.asarray(x), lane_shape)


def _rk_soa_elem(prim, params, lane_shape, *xs):
    if not any(isinstance(x, _RKLane) for x in xs):
        return prim.bind(*[x if isinstance(x, jax.core.Tracer) else np.asarray(x) for x in xs], **params)
    return _RKLane(prim.bind(*[_rk_soa_lift(x, lane_shape) for x in xs], **params))


def _rk_soa_select(lane_shape, which, *cases):
    # Pallas' Triton lowering takes `select_n` only with two cases and a bool
    # predicate: an int32 one, as a branch's index, fails to compile on the card,
    # and Pallas' interpreter does not check it (JAX 0.12.0.1). So an int32 `which`
    # selects as XLA lowers `select_n`: a tree of `which < mid` tests, which takes
    # the first case below range and the last above it.
    if len(cases) == 1:
        return cases[0]
    w = which.v if isinstance(which, _RKLane) else which
    dtype = np.dtype(w.dtype) if hasattr(w, "dtype") else np.asarray(w).dtype
    if dtype == np.bool_:
        return _rk_soa_elem(jax.lax.select_n_p, {}, lane_shape, which, *cases)

    def tree(offset, cs):
        if len(cs) == 1:
            return cs[0]
        mid = len(cs) // 2
        low = _rk_soa_elem(jax.lax.lt_p, {}, lane_shape, which, np.asarray(offset + mid, dtype))
        return _rk_soa_elem(jax.lax.select_n_p, {}, lane_shape, low,
                            tree(offset + mid, cs[mid:]), tree(offset, cs[:mid]))
    return tree(0, list(cases))


def _rk_soa_obj(shape):
    return np.empty(shape, dtype=object)


def _rk_soa_const(value, dtype=None):
    a = np.asarray(value, dtype=dtype)
    out = _rk_soa_obj(a.shape)
    for i in np.ndindex(*a.shape):
        out[i] = a[i]
    return out


def _rk_soa_lanes(vs, shape):
    out = _rk_soa_obj(shape)
    for i, v in zip(np.ndindex(*shape), vs):
        out[i] = _RKLane(v)
    return out


def _rk_soa_apply(f, ins, shape):
    out = _rk_soa_obj(shape)
    if not shape:
        out[()] = f(*[i[()] for i in ins])
        return out
    bs = np.broadcast_arrays(*ins)
    for i in np.ndindex(*shape):
        out[i] = f(*[b[i] for b in bs])
    return out


@_functools.cache
def _rk_soa_binop_params(name):
    sample = {"add": jnp.add, "mul": jnp.multiply, "max": jnp.maximum, "min": jnp.minimum,
              "and": jnp.logical_and, "or": jnp.logical_or}[name]
    args = (True, True) if name in ("and", "or") else (1.0, 1.0)
    eqns = jax.make_jaxpr(sample)(*args).eqns
    return eqns[-1].primitive, dict(eqns[-1].params)


def _rk_soa_binop(name, lane_shape):
    prim, params = _rk_soa_binop_params(name)
    return lambda a, b: _rk_soa_elem(prim, params, lane_shape, a, b)


def _rk_soa_primitives(jaxpr, names=None):
    names = set() if names is None else names
    for e in jaxpr.eqns:
        names.add(e.primitive.name)
        for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
            if key in e.params:
                sub = e.params[key]
                _rk_soa_primitives(getattr(sub, "jaxpr", sub), names)
        for b in e.params.get("branches", ()):
            _rk_soa_primitives(b.jaxpr, names)
    return names


def _rk_soa_check(closed_jaxpr):
    for name in sorted(_rk_soa_primitives(closed_jaxpr.jaxpr)):
        if name not in _RK_SOA_ELEMENTWISE | _RK_SOA_CALLS | _RK_SOA_STRUCTURAL:
            raise _RKSoAUnsupported(f"primitive `{name}`")


def _rk_soa_eval(jaxpr, consts, args, lane_shape):
    env = {}

    def read(v):
        return _rk_soa_const(v.val, v.aval.dtype) if isinstance(v, _RKLiteral) else env[v]

    for v, x in zip(jaxpr.constvars, consts):
        env[v] = x
    for v, x in zip(jaxpr.invars, args):
        env[v] = x
    for eqn in jaxpr.eqns:
        for v, o in zip(eqn.outvars, _rk_soa_rule(eqn, [read(v) for v in eqn.invars], lane_shape)):
            env[v] = o
    return [read(v) for v in jaxpr.outvars]


def _rk_soa_rule(eqn, ins, lane_shape):
    name, p = eqn.primitive.name, eqn.params
    if name == "select_n":
        return [_rk_soa_apply(lambda w, *cs: _rk_soa_select(lane_shape, w, *cs), ins,
                              eqn.outvars[0].aval.shape)]
    if name in _RK_SOA_ELEMENTWISE:
        return [_rk_soa_apply(lambda *xs: _rk_soa_elem(eqn.primitive, p, lane_shape, *xs), ins,
                              eqn.outvars[0].aval.shape)]
    x = ins[0] if ins else None
    if name in ("copy", "copy_p"):
        return [x]
    if name == "broadcast_in_dim":
        shape, expanded = p["shape"], [1] * len(p["shape"])
        for i, d in enumerate(p["broadcast_dimensions"]):
            expanded[d] = x.shape[i]
        return [np.broadcast_to(x.reshape(expanded), shape).copy()]
    if name == "reshape":
        return [x.reshape(p["new_sizes"]).copy()]
    if name == "squeeze":
        return [np.squeeze(x, axis=tuple(p["dimensions"])).copy()]
    if name == "expand_dims":
        return [np.expand_dims(x, tuple(p["dimensions"])).copy()]
    if name == "concatenate":
        return [np.concatenate(ins, axis=p["dimension"])]
    if name == "stack":
        return [np.stack(ins, axis=p["axis"])]
    if name == "slice":
        strides = p["strides"] or (1,) * len(p["start_indices"])
        return [x[tuple(slice(s, l, st) for s, l, st in zip(p["start_indices"], p["limit_indices"], strides))].copy()]
    if name == "transpose":
        return [np.transpose(x, p["permutation"]).copy()]
    if name == "rev":
        return [np.flip(x, axis=tuple(p["dimensions"])).copy()]
    if name in ("dynamic_slice", "dynamic_update_slice"):
        starts = ins[1:] if name == "dynamic_slice" else ins[2:]
        if not all(s.ndim == 0 and not isinstance(s[()], (_RKLane, jax.core.Tracer)) for s in starts):
            raise _RKSoAUnsupported(f"`{name}` at a traced position")
        sizes = p["slice_sizes"] if name == "dynamic_slice" else ins[1].shape
        where = tuple(slice(c, c + k) for c, k in
                      ((int(np.clip(int(s[()]), 0, d - k)), k) for s, d, k in zip(starts, x.shape, sizes)))
        if name == "dynamic_slice":
            return [x[where].copy()]
        out = x.copy()
        out[where] = ins[1]
        return [out]
    if name == "iota":
        shape, dim = p["shape"], p["dimension"]
        return [_rk_soa_const(np.broadcast_to(np.arange(shape[dim], dtype=p["dtype"]).reshape(
            [-1 if d == dim else 1 for d in range(len(shape))]), shape))]
    if name in ("reduce_sum", "reduce_max", "reduce_min", "reduce_prod", "reduce_and", "reduce_or"):
        op = _rk_soa_binop({"reduce_sum": "add", "reduce_max": "max", "reduce_min": "min", "reduce_prod": "mul",
                            "reduce_and": "and", "reduce_or": "or"}[name], lane_shape)
        axes = tuple(sorted(p["axes"]))
        keep = [d for d in range(x.ndim) if d not in axes]
        flat = np.transpose(x, keep + list(axes)).reshape([x.shape[d] for d in keep] + [-1])
        out = _rk_soa_obj(flat.shape[:-1])
        for i in np.ndindex(*out.shape):
            out[i] = _functools.reduce(op, list(flat[i]))
        return [out]
    if name == "dot_general":
        (lc, rc), (lb, rb) = p["dimension_numbers"]
        if lb or rb:
            raise _RKSoAUnsupported("`dot_general` with batch dimensions")
        mul, add = _rk_soa_binop("mul", lane_shape), _rk_soa_binop("add", lane_shape)
        lhs, rhs = ins
        lfree = [d for d in range(lhs.ndim) if d not in lc]
        rfree = [d for d in range(rhs.ndim) if d not in rc]
        out = _rk_soa_obj(tuple(lhs.shape[d] for d in lfree) + tuple(rhs.shape[d] for d in rfree))
        for i in np.ndindex(*out.shape):
            terms = []
            for c in np.ndindex(*[lhs.shape[d] for d in lc]):
                lidx, ridx = [0] * lhs.ndim, [0] * rhs.ndim
                for d, v in zip(lfree + list(lc), i[:len(lfree)] + c):
                    lidx[d] = v
                for d, v in zip(rfree + list(rc), i[len(lfree):] + c):
                    ridx[d] = v
                terms.append(mul(lhs[tuple(lidx)], rhs[tuple(ridx)]))
            out[i] = _functools.reduce(add, terms) if terms else np.zeros((), eqn.outvars[0].aval.dtype)
        return [out]
    if name in _RK_SOA_CALLS:
        for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
            if key in p:
                sub = p[key]
                j = getattr(sub, "jaxpr", sub)
                return _rk_soa_eval(j, [_rk_soa_const(c) for c in getattr(sub, "consts", [])],
                                    ins[len(ins) - len(j.invars):], lane_shape)
        raise _RKSoAUnsupported(f"`{name}` without a jaxpr")
    if name == "cond":
        index = ins[0]
        branches = [_rk_soa_eval(b.jaxpr, [_rk_soa_const(c) for c in b.consts], ins[1:], lane_shape)
                    for b in p["branches"]]
        if index.ndim == 0 and not isinstance(index[()], (_RKLane, jax.core.Tracer)):
            return branches[int(np.clip(int(index[()]), 0, len(branches) - 1))]
        last = np.int32(len(branches) - 1)
        to_int = {"new_dtype": np.dtype(np.int32), "weak_type": False, "sharding": None}

        def pick(i, *cs):
            i = _rk_soa_elem(jax.lax.convert_element_type_p, to_int, lane_shape, i)
            i = _rk_soa_elem(jax.lax.clamp_p, {}, lane_shape, np.int32(0), i, last)
            return _rk_soa_select(lane_shape, i, *cs)
        return [_rk_soa_apply(pick, [np.broadcast_to(index, v.aval.shape)] + [b[k] for b in branches], v.aval.shape)
                for k, v in enumerate(eqn.outvars)]
    raise _RKSoAUnsupported(f"primitive `{name}`")


# -- the lane solve: Dopri5 with sensitivity error control ------------------------

_RK_DOPRI5_A = ((1 / 5,), (3 / 40, 9 / 40), (44 / 45, -56 / 15, 32 / 9),
                (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
                (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
                (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84))
_RK_DOPRI5_C = (1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_RK_DOPRI5_E = (35 / 384 - 1951 / 21600, 0.0, 500 / 1113 - 22642 / 50085, 125 / 192 - 451 / 720,
                -2187 / 6784 - -12231 / 42400, 11 / 84 - 649 / 6300, -1.0 / 60.0)
_RK_DOPRI5_MID = (6025192743 / 30085553152 / 2, 0.0, 51252292925 / 65400821598 / 2,
                  -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
                  -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2)


def _rk_lane_dot(ws, xs):
    acc = None
    for w, x in zip(ws, xs):
        if w != 0:
            acc = w * x if acc is None else acc + w * x
    return acc


def _rk_lane_rms(xs):
    if len(xs) == 1:
        return jnp.abs(xs[0])
    # A Python float, weakly typed: a NumPy float64 would make float32 lanes float64
    # under 64-bit mode.
    return jnp.sqrt(_functools.reduce(lambda a, b: a + b, [x * x for x in xs])) / float(np.sqrt(len(xs)))


def _rk_lane_any(m):
    return jnp.max(m.astype(jnp.int32)) > 0


def _rk_lane_all(ms):
    return _functools.reduce(lambda a, b: a & b, ms)


class _RKLaneSpec:
    '''What a lane kernel needs that is fixed at trace time; hashed by identity.'''

    def __init__(self, jaxpr, S, W, dtype, leaves, interpret):
        self.jaxpr, self.S, self.W, self.dtype, self.interpret = jaxpr, S, W, dtype, interpret
        # leaves: one (shape, dtype, is_float) per argument leaf of the right-hand side
        self.leaves = leaves
        self.float_rows, self.int_rows, r_f, r_i = [], [], S + 2 + W, 1
        for shape, dt, is_float in leaves:
            k = int(np.prod(shape))
            if is_float:
                self.float_rows.append(list(range(r_f, r_f + k))); self.int_rows.append(None); r_f += k
            else:
                self.int_rows.append(list(range(r_i, r_i + k))); self.float_rows.append(None); r_i += k
        self.D, self.I = r_f, r_i


def _rk_pow2(n):
    return 1 << max(0, int(n - 1).bit_length())


def _rk_field_with_sensitivities(field, y, params, dy, dp):
    # Each row is one first variation. Only its value feeds error control.
    # The ordinary derivative of the state still flows through the original RHS.
    value, push = jax.linearize(field, y, params)
    slopes = jax.vmap(push)(dy, dp)
    return value, jax.tree.map(jax.lax.stop_gradient, slopes)


def _rk_coupled_system(rhs, t0, y0, args):
    # Lift captures into args as well: the controller must see the same basis
    # for a value and for every subset of inputs selected by an AD caller.
    leaves, tree = jax.tree.flatten(args)
    closed = jax.make_jaxpr(lambda t, y, *ls: rhs(t, y, jax.tree.unflatten(tree, ls)))(
        t0, y0, *leaves)
    narg = len(leaves)
    values = tuple(jnp.asarray(v) for v in (*leaves, *closed.consts))
    graph = closed.jaxpr

    def flat_rhs(t, y, values):
        return jax.core.eval_jaxpr(graph, values[narg:], t, y, *values[:narg])[0]

    if _RK_REVERSE_CALLS.intersection(_rk_soa_primitives(graph)):
        # A reverse-only primitive supplies a linear pullback. Transpose
        # that map to obtain the pushforward of the same authored rule.
        field = jax.custom_jvp(flat_rhs)

        @field.defjvp
        def rhs_jvp(primals, tangents):
            value, pullback = jax.vjp(flat_rhs, *primals)
            push = jax.linear_transpose(pullback, jnp.zeros_like(value))
            return value, push(tangents)[0]

        return field, values

    return flat_rhs, values


def _rk_diffrax_control(rhs, y0, values):
    floats = tuple(i for i, v in enumerate(values) if jnp.issubdtype(jnp.result_type(v), jnp.inexact))
    size = y0.size
    directions = size + sum(int(np.size(values[i])) for i in floats)
    index = jnp.arange(directions)[:, None]

    def seed(first, n, dtype):
        return (index == first + jnp.arange(n)).astype(dtype)

    init = jnp.concatenate((y0[None, :], seed(0, size, y0.dtype)))
    seeds = []
    first = size
    for i in floats:
        shape = jnp.shape(values[i])
        n = int(np.prod(shape))
        seeds.append(seed(first, n, jnp.result_type(values[i])).reshape((directions,) + shape))
        first += n

    def augmented(t, state, values):
        def field(y, params):
            args = list(values)
            for i, p in zip(floats, params):
                args[i] = p
            return rhs(t, y, tuple(args))

        ydot, sdot = _rk_field_with_sensitivities(field, state[0], tuple(values[i] for i in floats),
                                                  state[1:], tuple(seeds))
        return jnp.concatenate((ydot[None, :], sdot))

    return augmented, init


def _rk_control_norm(x):
    # The trial guard handles nonfinite candidates/errors. In a finite
    # trial with purely relative tolerance, a stationary zero has ratio 0/0.
    x = jnp.where(jnp.isnan(x), 0, x)
    return jnp.sqrt(jnp.maximum(jnp.mean(jnp.square(x[0])), jnp.mean(jnp.square(x[1:]))))


def _rk_lane_kernel(spec, rows):
    S, W, D, dtype = spec.S, spec.W, spec.D, spec.dtype
    BL, K = _RK_LANES, len(rows)
    KP = _rk_pow2(K) if K else 0
    T0, T1, TS = S, S + 1, S + 2
    trash_y, trash_j = W * S, W * S * KP
    param_rows = tuple(r for fr in spec.float_rows if fr is not None for r in fr)
    directions = _rk_pow2(S + len(param_rows))

    def kernel(xd_ref, xf_ref, xi_ref, ys_ref, code_ref, *jac):
        jac_ref = jac[0] if K else None
        cols = jnp.arange(BL, dtype=jnp.int32)
        lane = (BL,)
        x = {r: xd_ref[r, :] for r in list(range(TS)) + [r for fr in spec.float_rows if fr for r in fr]}
        rtol, atol, dt0, floor = (xf_ref[i, :] for i in range(4))
        limit = xi_ref[0, :]
        ints = {r: xi_ref[r, :] for ir in spec.int_rows if ir for r in ir}
        seed = {r: jnp.broadcast_to((jnp.arange(KP, dtype=jnp.int32) == k)[:, None], (KP, BL)).astype(dtype)
                for k, r in enumerate(rows)}
        pert = [r for r in rows if r < TS or r >= TS + W]  # inputs read by the steps (not output times)
        zeros_k = jnp.zeros((KP, BL), dtype) if K else None
        ys_ref[...] = jnp.full(ys_ref.shape, jnp.inf, dtype)
        if K:
            jac_ref[...] = jnp.zeros(jac_ref.shape, dtype)

        def get(dv, r):
            return dv[r] if r in dv else x[r]

        def field(t, y, params):
            args = []
            for (shape, dt, is_float), fr, ir in zip(spec.leaves, spec.float_rows, spec.int_rows):
                vals = [params[r] for r in fr] if is_float else [ints[r].astype(dt) for r in ir]
                args.append(_rk_soa_lanes(vals, shape))
            (out,) = _rk_soa_eval(spec.jaxpr.jaxpr, [_rk_soa_const(c) for c in spec.jaxpr.consts],
                                  [_rk_soa_lanes([t], ()), _rk_soa_lanes(y, (S,))] + args, lane)
            return [_rk_soa_lift(out[s], lane).astype(dtype) for s in range(S)]

        def basis(i):
            return jnp.broadcast_to((jnp.arange(directions) == i)[:, None], (directions, BL)).astype(dtype)

        dp = {r: basis(S + i) for i, r in enumerate(param_rows)}

        def rhs(t, y, dv):
            params = {r: get(dv, r) for r in param_rows}
            rate, slopes = _rk_field_with_sensitivities(lambda v, p: field(t, v, p),
                                                         y[:S], params, y[S:], dp)
            return rate + slopes

        def jvp(f, primals, tangents):
            # (outputs, their tangents along the K perturbed inputs, aux)
            if not K:
                out, aux = f(*primals)
                return out, None, aux
            return jax.vmap(lambda tg: jax.jvp(f, primals, tg, has_aux=True), in_axes=(0,),
                            out_axes=(None, 0, None))(tangents)

        def init(dv):
            t0, t1 = get(dv, T0), get(dv, T1)
            return (t0, jnp.minimum(t0 + dt0, t1), [get(dv, s) for s in range(S)]), None

        dvp = {r: x[r] for r in pert}
        if K and pert:
            (tprev, tnext, y), dstate, _ = jvp(init, (dvp,), ({r: seed[r] for r in pert},))
        else:
            (tprev, tnext, y), _ = init(dvp)
            dstate = (zeros_k, zeros_k, [zeros_k] * S) if K else None
        f0 = [jnp.zeros(lane, dtype) for _ in range(S)]
        df0 = [zeros_k] * S if K else None
        state = (tprev, tnext, y, f0)
        dstate = (dstate[0], dstate[1], dstate[2], df0) if K else None
        first = jnp.ones(lane, jnp.bool_)
        count = jnp.zeros(lane, jnp.int32)
        target = jnp.zeros(lane, jnp.int32)
        saved = jnp.zeros(lane, jnp.int32)
        code = jnp.zeros(lane, jnp.int32)
        dparams = {r: seed[r] for r in pert} if K else None

        control = ([basis(i) for i in range(S)], [jnp.zeros((directions, BL), dtype)] * S)

        def step(st, dv, flags, control):
            tprev, tnext, y, f0 = st
            y = y + control[0]
            f0 = f0 + control[1]
            count, target, saved, any_first = flags
            t1 = get(dv, T1)
            dt = tnext - tprev
            f0 = jax.lax.cond(any_first, lambda: rhs(tprev, y, dv), lambda: f0)
            k = [[v * dt for v in f0]]
            for a, c in zip(_RK_DOPRI5_A, _RK_DOPRI5_C):
                yi = [v + _rk_lane_dot(a, stages) for v, stages in zip(y, zip(*k))]
                fi = rhs(tprev + c * dt, yi, dv)
                k.append([v * dt for v in fi])
            y1, f1 = yi, fi
            err = [_rk_lane_dot(_RK_DOPRI5_E, stages) for stages in zip(*k)]
            # Triton supports integer reductions across the sensitivity axis.
            valid = [jnp.isfinite(v) for v in y1 + err]
            finite = _rk_lane_all([m if m.ndim == 1 else jnp.min(m.astype(jnp.int32), axis=0) != 0
                                   for m in valid])
            y1 = [jnp.where(finite, a, b) for a, b in zip(y1, y)]
            err = [jnp.where(finite, e, jnp.inf) for e in err]
            # PID control (I-controller, factor 0.2 to 10, safety 0.9, error order 5)
            ratios = [jnp.where(e == 0, 0, e / (atol + jnp.maximum(jnp.abs(a), jnp.abs(b)) * rtol))
                      for e, a, b in zip(err, y, y1)]
            sens_rms = jnp.sqrt(jnp.sum(jnp.square(_rk_lane_rms(ratios[S:])), axis=0) / (S + len(param_rows)))
            scaled = jax.lax.stop_gradient(jnp.maximum(_rk_lane_rms(ratios[:S]), sens_rms))
            keep = scaled < 1
            factor = jax.lax.stop_gradient(jnp.clip(0.9 * (1 / scaled) ** (1 / 5), jnp.where(keep, 1.0, 0.2),
                                                    jnp.where(keep, 10.0, 0.9)))
            dtn = (tnext - tprev) * factor
            next_t0 = jnp.where(keep, tnext, tprev)
            next_t1 = _rk_next_trial_time(keep, tprev, tnext, next_t0 + dtn)
            # the per-output step budget, and unusable states or steps
            count = jnp.where(saved == target, count, 0)
            count = jnp.where(keep, count + 1, count)
            unusable = ~jnp.isfinite(next_t1 - next_t0) | (next_t1 <= next_t0)
            unstable = ~_rk_lane_all([jnp.isfinite(v) for v in y[:S]]) | ((next_t0 < t1) & unusable)
            code = jnp.where(unstable, _RK_FAILED, jnp.where(count > limit, _RK_BUDGET, _RK_OK))
            code = jnp.where(keep & ~finite, _RK_FAILED, code)
            tprev_new = jnp.minimum(next_t0, t1)
            tclip = jnp.where(keep, t1, tprev_new + 0.5 * (t1 - tprev_new))
            tnext_new = jnp.where(next_t1 > floor, tclip, next_t1)
            code = jnp.where(~keep & ((tnext_new >= tnext) | (tnext_new <= tprev)), _RK_FAILED, code)
            # dense output over the step just taken
            ymid = [y[s] + _rk_lane_dot(_RK_DOPRI5_MID, [kj[s] for kj in k]) for s in range(S)]
            dense = []
            for s in range(S):
                a0, a1, km = y[s], y1[s], ymid[s]
                g0, g1 = k[0][s], k[-1][s]
                dense.append((2 * (g1 - g0) - 8 * (a1 + a0) + 16 * km, 5 * g0 - 3 * g1 + 18 * a0 + 14 * a1 - 32 * km,
                              g1 - 4 * g0 - 11 * a0 - 5 * a1 + 16 * km, g0, a0))
            ys = [jnp.where(keep, a, b) for a, b in zip(y1, y)]
            fs = [jnp.where(keep, a, b) for a, b in zip(f1, f0)]
            new = (tprev_new, tnext_new, ys[:S], fs[:S])
            return (new, (tprev, tnext, dense)), (keep, count, saved, code, (ys[S:], fs[S:]))

        def interp(span, t):
            tprev, tnext, dense = span
            same = tprev == tnext
            theta = jnp.where(same, 0.0, t - tprev) / jnp.where(same, 1.0, tnext - tprev)
            out = []
            for cs in dense:
                v = jnp.zeros_like(theta)
                for c in cs:
                    v = v * theta + c
                out.append(v)
            return out, None

        def live_of(c):
            (tprev, tnext, y, f0), dst, control, first, count, target, saved, code, nsteps = c
            return (tprev < x[T1]) & (code == _RK_OK)

        def cond_fun(c):
            return _rk_lane_any(live_of(c))

        def body_fun(c):
            st, dst, control, first, count, target, saved, code, nsteps = c
            live = live_of(c)
            any_first = _rk_lane_any(first & live)
            flags = (count, target, saved, any_first)
            if K:
                (new, span), (dnew, dspan), aux = jvp(lambda st, dv: step(st, dv, flags, control), (st, dvp), (dst, dparams))
            else:
                (new, span), _, aux = jvp(lambda st, dv: step(st, dv, flags, control), (st, dvp), ())
                dnew = dspan = None
            keep, count_new, target_new, code_new, new_control = aux
            sel = lambda a, b: jnp.where(live, a, b)
            st = jax.tree.map(sel, new, st)
            control = jax.tree.map(sel, new_control, control)
            if K:
                dst = jax.tree.map(lambda a, b: jnp.where(live[None, :], a, b), dnew, dst)
            # save every requested output in (tprev, tnext] of an accepted step
            saving = live & keep
            span_end = span[1]

            def pending(sv):
                safe = jnp.minimum(sv, W - 1)
                return saving & (sv < W) & (xd_ref[TS + safe, cols] <= span_end)

            def save_cond(sv):
                return _rk_lane_any(pending(sv))

            def save_body(sv):
                m = pending(sv)
                safe = jnp.minimum(sv, W - 1)
                tv = xd_ref[TS + safe, cols]
                if K:
                    # Each output depends only on its own interpolation time.
                    dtv = seed.get(TS, zeros_k)
                    yv, dyv, _ = jvp(interp, (span, tv), (dspan, dtv))
                else:
                    yv, _, _ = jvp(interp, (span, tv), ())
                for s in range(S):
                    ys_ref[jnp.where(m, safe * S + s, trash_y), cols] = yv[s]
                    if K:
                        base = jnp.where(m, (safe * S + s) * KP, trash_j)
                        jac_ref[base[None, :] + jnp.arange(KP, dtype=jnp.int32)[:, None], cols[None, :]] = dyv[s]
                return jnp.where(m, sv + 1, sv)

            saved_new = jax.lax.while_loop(save_cond, save_body, saved)
            first = jnp.where(live & keep, False, first)
            return (st, dst, control, first, jnp.where(live, count_new, count),
                    jnp.where(live, saved, target), saved_new, jnp.where(live, code_new, code),
                    jnp.where(live, nsteps + 1, nsteps))

        carry = (state, dstate, control, first, count, target, saved, code, jnp.zeros(lane, jnp.int32))
        carry = jax.lax.while_loop(cond_fun, body_fun, carry)
        code_ref[0, :] = carry[-2]
        code_ref[1, :] = carry[-1]

    return kernel


def _rk_lane_call(spec, rows, xd, xf, xi):
    '''Run the lane kernel over every lane: inputs with leading lane dimensions.'''
    lead = xd.shape[:-1]
    L = int(np.prod(lead)) if lead else 1
    S, W, BL = spec.S, spec.W, _RK_LANES
    K = len(rows)
    KP = _rk_pow2(K) if K else 0
    Lp = -(-L // BL) * BL
    take = jnp.minimum(jnp.arange(Lp), L - 1)

    def lay(a, rows_pad):
        a = a.reshape(L, a.shape[-1])[take].T
        return jnp.concatenate([a, jnp.zeros((rows_pad - a.shape[0], Lp), a.dtype)]) if rows_pad > a.shape[0] else a

    xd2 = lay(xd, _rk_pow2(spec.D))
    xf2 = lay(xf, 4)
    xi2 = lay(xi.astype(jnp.int32), _rk_pow2(spec.I))
    ny, nj = _rk_pow2(W * S + 1), _rk_pow2(W * S * KP + KP) if K else 0
    spec_block = lambda rows_n: _pl.BlockSpec((rows_n, BL), lambda i: (0, i))
    out_shape = [jax.ShapeDtypeStruct((ny, Lp), spec.dtype), jax.ShapeDtypeStruct((2, Lp), jnp.int32)]
    out_specs = [spec_block(ny), spec_block(2)]
    if K:
        out_shape.append(jax.ShapeDtypeStruct((nj, Lp), spec.dtype))
        out_specs.append(spec_block(nj))
    outs = _pl.pallas_call(
        _rk_lane_kernel(spec, rows), grid=(Lp // BL,),
        in_specs=[spec_block(xd2.shape[0]), spec_block(4), spec_block(xi2.shape[0])],
        out_specs=out_specs, out_shape=out_shape, interpret=spec.interpret,
        compiler_params=None if spec.interpret else _plt.CompilerParams(num_warps=_RK_LANE_WARPS, num_stages=1))(xd2, xf2, xi2)
    ys = outs[0][:W * S, :L].T.reshape(lead + (W, S))
    code = outs[1][0, :L].reshape(lead)
    if _RK_LANE_DEBUG is not None:
        jax.debug.callback(lambda v: _RK_LANE_DEBUG.append(np.asarray(v)), outs[1][1, :L].reshape(lead))
    if not K:
        return ys, code
    jac = outs[2][:W * S * KP, :L].T.reshape(lead + (W, S, KP))[..., :K]
    return ys, code, jac


def _rk_lane_batched(spec, rows):
    # One kernel call for all lanes: vmapped calls (plates, chains) join the lane axis.
    @jax.custom_batching.custom_vmap
    def solve(xd, xf, xi):
        return _rk_lane_call(spec, rows, xd, xf, xi)

    @solve.def_vmap
    def solve_vmap(axis_size, in_batched, xd, xf, xi):
        xs = [a if b else jnp.broadcast_to(a, (axis_size,) + a.shape) for a, b in zip((xd, xf, xi), in_batched)]
        out = solve(*xs)
        return out, tuple(True for _ in out)

    return solve


def _rk_lane_pack(diff, nondiff, spec):
    y0, t0, t1, ts, fleaves = diff
    rtol, atol, dt0, floor, limit, ileaves = nondiff
    xd = jnp.concatenate([jnp.ravel(y0), t0[None], t1[None], jnp.ravel(ts)] +
                         [jnp.ravel(l).astype(spec.dtype) for l in fleaves])
    xf = jnp.stack([jnp.asarray(v, spec.dtype) for v in (rtol, atol, dt0, floor)])
    limit = jnp.minimum(jnp.asarray(limit), 2 ** 31 - 1).astype(jnp.int32)  # an int32 kernel row
    xi = jnp.concatenate([limit[None]] + [jnp.ravel(l).astype(jnp.int32) for l in ileaves])
    return xd, xf, xi


@eqx.filter_custom_vjp
def _rk_lane_solve(diff, nondiff, spec):
    return _rk_lane_batched(spec, ())(*_rk_lane_pack(diff, nondiff, spec))


def _rk_lane_rows(perturbed, spec):
    y0, t0, t1, ts, fleaves = perturbed
    rows = []
    S, W = spec.S, spec.W
    if y0:
        rows += list(range(S))
    if t0:
        rows.append(S)
    if t1:
        rows.append(S + 1)
    if ts:
        rows.append(S + 2)
    for p, fr in zip(fleaves, [fr for fr in spec.float_rows if fr is not None]):
        if p:
            rows += fr
    return tuple(rows)


@_rk_lane_solve.def_fwd
def _rk_lane_solve_fwd(perturbed, diff, nondiff, spec):
    rows = _rk_lane_rows(perturbed, spec)
    if not rows:
        return _rk_lane_batched(spec, ())(*_rk_lane_pack(diff, nondiff, spec)), None
    ys, code, jac = _rk_lane_batched(spec, rows)(*_rk_lane_pack(diff, nondiff, spec))
    return (ys, code), jac


@_rk_lane_solve.def_bwd
def _rk_lane_solve_bwd(jac, grad_obj, perturbed, diff, nondiff, spec):
    rows = _rk_lane_rows(perturbed, spec)
    gys = grad_obj[0]
    y0, t0, t1, ts, fleaves = diff
    if jac is None or gys is None:
        return (None, None, None, None, tuple(None for _ in fleaves))
    g = jnp.einsum("ws,wsk->k", gys, jac)
    at = {r: i for i, r in enumerate(rows)}
    S, W = spec.S, spec.W

    def part(flag, rs, like):
        return g[jnp.asarray([at[r] for r in rs])].reshape(jnp.shape(like)).astype(jnp.result_type(like)) if flag else None

    frows = [fr for fr in spec.float_rows if fr is not None]
    py0, pt0, pt1, pts, pf = perturbed
    dts = jnp.sum(gys * jac[:, :, at[S + 2]], axis=1).astype(ts.dtype) if pts else None
    return (part(py0, range(S), y0), part(pt0, [S], t0), part(pt1, [S + 1], t1), dts,
            tuple(part(p, fr, l) for p, fr, l in zip(pf, frows, fleaves)))


def _rk_lane_prepare(rhs, y0, t0, args, W, interpret):
    if any(jnp.ndim(v) for v in args):
        # The structure-of-arrays interpreter expands such a leaf once per
        # element. Diffrax keeps its parameter and sensitivity axes as arrays.
        raise _RKSoAUnsupported("an array-valued right-hand-side argument")
    conv, consts = jax.closure_convert(lambda t, y, a: rhs(t, y, a), t0, y0, args)
    aleaves, atree = jax.tree.flatten(args)
    leaves = list(aleaves) + list(consts)
    na = len(aleaves)

    def flat(t, y, *ls):
        return conv(t, y, jax.tree.unflatten(atree, ls[:na]), *ls[na:])

    avals = [jax.ShapeDtypeStruct(jnp.shape(v), jnp.result_type(v)) for v in (t0, y0, *leaves)]
    jaxpr = jax.make_jaxpr(flat)(*avals)
    _rk_soa_check(jaxpr)
    # A vector-sized state may come from runtime data. Only a scalar state or
    # an RHS that explicitly constructs its components can be expanded here.
    # Other systems retain array operations in the Diffrax implementation.
    if y0.size > 1:
        (out,) = jaxpr.jaxpr.outvars
        explicit = any(out in e.outvars and (
            (e.primitive.name == "stack" and all(v.aval.shape == () for v in e.invars)) or
            (e.primitive.name == "concatenate" and all(v.aval.shape == (1,) for v in e.invars)))
            for e in jaxpr.jaxpr.eqns)
        if not explicit:
            raise _RKSoAUnsupported("an array-valued state without explicit scalar components")
    info = [(tuple(jnp.shape(v)), jnp.result_type(v), bool(jnp.issubdtype(jnp.result_type(v), jnp.inexact)))
            for v in leaves]
    S, dtype = int(y0.shape[0]), jnp.result_type(y0)

    def trial():
        # Evaluate element by element once, abstractly: what only shows when
        # run (an index at a traced position) is found here, not in the kernel.
        one = lambda dt: jnp.zeros((1,), dt)
        args = [_rk_soa_lanes([one(dt)] * int(np.prod(shape)), shape) for shape, dt, _ in info]
        (out,) = _rk_soa_eval(jaxpr.jaxpr, [_rk_soa_const(c) for c in jaxpr.consts],
                              [_rk_soa_lanes([one(dtype)], ()), _rk_soa_lanes([one(dtype)] * S, (S,))] + args, (1,))
        if out.shape != (S,):
            raise _RKSoAUnsupported(f"a right-hand side of shape {out.shape}")
        return [_rk_soa_lift(out[s], (1,)) for s in range(S)]

    jax.eval_shape(trial)
    spec = _RKLaneSpec(jaxpr, S, W, dtype, info, interpret)
    return spec, leaves


# -- forward-mode sensitivities through diffrax (Kvaerno5, unsupported right-hand sides) --

def _rk_ode_forward_core(diff, solver, setup):
    fn, y0, t0, last, ts, args = diff
    rtol, atol, dtmin, dt0, limit = setup
    term, initial, pid, saveat = _rk_diffrax_system(solver, fn, y0, args, ts, rtol, atol, dtmin)
    sol = diffrax.diffeqsolve(
        term, solver, t0, last, dt0, initial, args=args, saveat=saveat,
        stepsize_controller=_RKOutputBudget(pid, ts, limit, False),
        max_steps=2 ** 31 - 1, adjoint=diffrax.ForwardMode(), throw=False)
    r = sol.result
    return sol.ys, jnp.where(r == diffrax.RESULTS.successful, _RK_OK,
                             jnp.where(r == diffrax.RESULTS.max_steps_reached, _RK_BUDGET, _RK_FAILED)).astype(jnp.int32)


@eqx.filter_custom_vjp
def _rk_ode_forward(diff, solver, setup):
    return _rk_ode_forward_core(diff, solver, setup)


@_rk_ode_forward.def_fwd
def _rk_ode_forward_fwd(perturbed, diff, solver, setup):
    flags = jax.tree.leaves(perturbed)
    leaves, treedef = jax.tree.flatten(diff)
    active = [i for i, f in enumerate(flags) if f]
    if not active:
        return _rk_ode_forward_core(diff, solver, setup), None
    shapes = [jnp.shape(leaves[i]) for i in active]
    sizes = [int(np.prod(s)) for s in shapes]
    flat = jnp.concatenate([jnp.ravel(leaves[i]) for i in active])

    def at(v):
        new, o = list(leaves), 0
        for i, s, k in zip(active, shapes, sizes):
            new[i] = v[o:o + k].reshape(s).astype(jnp.result_type(leaves[i]))
            o += k
        return _rk_ode_forward_core(jax.tree.unflatten(treedef, new), solver, setup)

    def push(t):
        out, (dys, _) = jax.jvp(at, (flat,), (t,))
        return out, dys

    out, jac = jax.vmap(push, out_axes=((None, None), -1))(jnp.eye(flat.shape[0], dtype=flat.dtype))
    return out, jac


@_rk_ode_forward.def_bwd
def _rk_ode_forward_bwd(jac, grad_obj, perturbed, diff, solver, setup):
    flags = jax.tree.leaves(perturbed)
    leaves, treedef = jax.tree.flatten(diff)
    grads = [None] * len(leaves)
    gys = grad_obj[0]
    if jac is None or gys is None:
        return jax.tree.unflatten(treedef, grads)
    g = jnp.tensordot(gys, jac, axes=gys.ndim)
    o = 0
    for i, f in enumerate(flags):
        if f:
            k = int(np.prod(jnp.shape(leaves[i])))
            grads[i] = g[o:o + k].reshape(jnp.shape(leaves[i])).astype(jnp.result_type(leaves[i]))
            o += k
    return jax.tree.unflatten(treedef, grads)


# -- the entry -----------------------------------------------------------------

def _rk_ode_mode():
    if RK_ODE_KERNEL == "off" or _pl is None:
        return "diffrax"
    if RK_ODE_KERNEL == "interpret":
        return "interpret"
    return "cuda" if jax.default_backend() == "gpu" else "diffrax"


_rk_warned_ode_rtol = False


def _rk_warn_ode_rtol(reltol):
    global _rk_warned_ode_rtol
    if not _rk_warned_ode_rtol:
        _rk_warned_ode_rtol = True
        _warnings.warn(
            f"The ODE requested reltol={np.float32(reltol)!s} below Float32 epsilon (1.19e-7). "
            "Error control can take extra steps. For tighter relative accuracy use a Float64 "
            "export with JAX x64 enabled, or choose a looser reltol if the model permits it.",
            RuntimeWarning)


@jax.custom_batching.custom_vmap
def _rk_check_ode_rtol(reltol):
    def report(v):
        jax.debug.callback(_rk_warn_ode_rtol, v)
    jax.lax.cond(reltol < jnp.finfo(jnp.float32).eps, report, lambda v: None, reltol)
    return ()


@_rk_check_ode_rtol.def_vmap
def _rk_check_ode_rtol_batch(size, batched, reltol):
    # Report one offending value across all nested plate and chain axes.
    if size:
        _rk_check_ode_rtol(jnp.min(reltol))
    return (), ()


def _rk_diffrax_system(solver, rhs, y0, args, ts, rtol, atol, dtmin):
    if isinstance(solver, _RKDopri5):
        rhs, y0 = _rk_diffrax_control(rhs, y0, args)
        pid = diffrax.PIDController(rtol=rtol, atol=atol, norm=_rk_control_norm)
        saveat = diffrax.SaveAt(ts=ts, fn=lambda t, y, p: y[0])
    else:
        pid = diffrax.PIDController(rtol=rtol, atol=atol, dtmin=dtmin, force_dtmin=False)
        saveat = diffrax.SaveAt(ts=ts)
    return diffrax.ODETerm(rhs), y0, pid, saveat


def _rk_ode_solve(solver, rhs, y0, t0, ts, n, reltol, abstol, maxsteps, args, active=True):
    # ReactiveKernels' rk_ode_rk45_tol / rk_ode_bdf_tol: integrate
    # dy/dt = rhs(t, y, args) from t0, returning the state at each of the
    # first n output times (rows) and a status. The solve's times, state and
    # outputs are the module's float type (`_jax_ode_solve_source`).
    dtype = jnp.float64
    cap = 2 ** 31 - 1
    if isinstance(maxsteps, int):
        maxsteps = min(maxsteps, cap)
    y0 = jnp.asarray(y0, dtype=dtype)
    ts = jnp.asarray(ts, dtype=dtype)
    t0 = jnp.asarray(t0, dtype=dtype)
    width = ts.shape[0]
    if width == 0 or y0.size == 0:
        return jnp.empty((width, y0.size), dtype), jnp.where(n == 0, _RK_OK, 1).astype(jnp.int32)
    requested = jnp.arange(width) < n
    previous = jnp.concatenate([t0[None], ts[:-1]])
    valid = (jnp.all(jnp.isfinite(y0)) & jnp.isfinite(t0)
             & jnp.all(~requested | (jnp.isfinite(ts) & (ts > previous)))
             & jnp.isfinite(reltol) & (reltol >= 0) & jnp.isfinite(abstol) & (abstol >= 0)
             & ((reltol > 0) | (abstol > 0)) & (maxsteps > 0))
    # A row with no requested time (a padded lane) solves a unit interval
    # whose rows are padding.
    last = jnp.where(n > 0, ts[jnp.maximum(n, 1) - 1], t0 + 1)
    ok = valid & (n > 0)
    if dtype == jnp.float32:
        requested_rtol = jnp.where(active & ok & (reltol > 0), reltol, jnp.inf).astype(dtype)
        _rk_check_ode_rtol(jax.lax.stop_gradient(requested_rtol))
    last = jnp.where(ok, last, t0 + 1)
    ts = jnp.where(requested & ok, ts, last)
    # The stiff controller retains its endpoint-scaled minimum step.
    dtmin = jax.lax.stop_gradient(16 * jnp.finfo(dtype).eps * jnp.maximum(jnp.abs(t0), jnp.abs(last)))
    rtol = jnp.where(ok, reltol, 1.0)
    atol = jnp.where(ok, abstol, 1.0)
    limit = jnp.where(ok, maxsteps, cap)
    coupled = isinstance(solver, _RKDopri5)
    dt0 = None
    if coupled:
        rhs, args = _rk_coupled_system(rhs, t0, y0, args)
        origin = jax.lax.stop_gradient(t0)
        dt0 = jnp.maximum(jnp.asarray(0.01, dtype), jnp.nextafter(origin, jnp.inf) - origin)

    def diffrax_path():
        # RK45 has already lifted every capture into args for its controller.
        field = rhs if coupled else eqx.filter_closure_convert(rhs, t0, y0, args)
        term, initial, pid, saveat = _rk_diffrax_system(solver, field, y0, args, ts, rtol, atol, dtmin)

        def outputs(method, check_trials=False):
            sol = diffrax.diffeqsolve(
                term, method, t0, last, dt0, initial, args=args, saveat=saveat,
                stepsize_controller=_RKOutputBudget(pid, ts, limit, check_trials),
                max_steps=cap, adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=64), throw=False)
            return sol.ys, sol.result

        def plain_outputs():
            return outputs(solver.unguarded(), check_trials=True)

        if coupled:
            # Guard trials in the integration that produces the values. A
            # separate status probe can take different steps after rounding.
            ys, result = outputs(solver)
        else:
            _, probe_result = plain_outputs()
            all_finite = eqx.internal.unvmap_all(probe_result == diffrax.RESULTS.successful)
            ys, result = jax.lax.cond(all_finite, plain_outputs, lambda: outputs(solver))
        return ys, jnp.where(result == diffrax.RESULTS.successful, _RK_OK,
                             jnp.where(result == diffrax.RESULTS.max_steps_reached, _RK_BUDGET, _RK_FAILED)).astype(jnp.int32)

    def forward_path():
        fn = rhs if coupled else eqx.filter_closure_convert(rhs, t0, y0, args)
        return _rk_ode_forward((fn, y0, t0, last, ts, args), solver, (rtol, atol, dtmin, dt0, limit))

    def lane_path(interpret):
        if not isinstance(solver, _RKDopri5):
            return forward_path()
        try:
            spec, leaves = _rk_lane_prepare(rhs, y0, t0, args, width, interpret)
        except _RKSoAUnsupported as e:
            _warnings.warn(f"ReactiveKernels: an ODE right-hand side uses {e}, which the per-lane GPU solve "
                           "does not run; this solve uses diffrax with forward-mode sensitivities instead")
            return forward_path()
        floor = jax.lax.stop_gradient(last)
        for _ in range(100):
            floor = jnp.nextafter(floor, -jnp.inf)
        fl = tuple(v for v, (_, _, f) in zip(leaves, spec.leaves) if f)
        il = tuple(v for v, (_, _, f) in zip(leaves, spec.leaves) if not f)
        return _rk_lane_solve((y0, t0, last, ts, fl),
                              (rtol, atol, dt0, floor, limit, il), spec)

    mode = _rk_ode_mode()
    if mode == "interpret":
        ys, code = lane_path(True)
    elif mode == "cuda":
        ys, code = jax.lax.platform_dependent(cuda=lambda: lane_path(False), default=diffrax_path)
    else:
        ys, code = diffrax_path()
    status = jnp.where(valid, code, 1)
    return ys, jnp.where(n > 0, status, 0)
"""

# `_rk_ode_solve` of this module: it solves in the module's float type, never
# in JAX's default float, which is float64 whenever 64-bit mode is on, even in
# a float32 module (another module, or this one's `fma` helper, turned it on).
# Its float64 times and state would make the solution float64 (snag
# evaluating-the-l-662ff8c9).
_jax_ode_solve_source() = _jax_float_type() === Float64 ? _JAX_ODE_SOLVE :
    replace(_JAX_ODE_SOLVE, "    dtype = jnp.float64\n" => "    dtype = jnp." * _jax_float_name() * "\n")

_jax_ode_solver(::typeof(rk_ode_rk45_tol)) = "Dopri5"
_jax_ode_solver(::typeof(rk_ode_bdf_tol)) = "Kvaerno5"

for f in (rk_ode_rk45_tol, rk_ode_bdf_tol)
    @eval _jax_function_call(ctx::_JaxContext, f::typeof($f), args::Vector{_PyVal}) =
        _jax_ode_call(ctx, f, args)
end

# `rk_ode_*_tol(f, y0, t0, ts, reltol, abstol, maxsteps, args...)`: the
# right-hand side `f(t, y, args...)` becomes a nested Python function of
# `(t, y, args)`, diffrax's vector-field signature, translated like any
# helper; `args` carries the extra arguments that are runtime values, and the
# others (functions, static values) are read where `f` was called.
function _jax_ode_call(ctx::_JaxContext, f, args::Vector{_PyVal})
    name = nameof(f)
    ctx.host === nothing ||
        _jax_error("`", name, "` has no host (prepare_data) translation", _jax_where(ctx))
    length(args) >= 7 || _jax_error("`", name, "` takes f, y0, t0, ts, reltol, abstol, ",
                                    "maxsteps and the arguments of f", _jax_where(ctx))
    rhs, y0, t0, ts, rt, at, mx = args[1:7]
    extra = args[8:end]
    rhs.T isa _JaxCallable || rhs.T isa Type && rhs.T <: Function ||
        _jax_error("`", name, "`'s right-hand side is not a function", _jax_where(ctx))
    _jax_rank(y0.T) == 1 && isempty(y0.extent) ||
        _jax_error("`", name, "`'s initial state is a `", y0.T, "`, not a vector of ",
                   "one length", _jax_where(ctx))
    _jax_rank(ts.T) == 1 && !_jax_ragged(ts.T) ||
        _jax_error("`", name, "`'s output times are a `", ts.T, "`", _jax_where(ctx))
    for (what, v) in (("initial time", t0), ("relative tolerance", rt), ("absolute tolerance", at))
        v.T isa Type && v.T <: Real ||
            _jax_error("`", name, "`'s ", what, " is a `", v.T, "`", _jax_where(ctx))
    end
    mx.T isa Type && mx.T <: Integer ||
        _jax_error("`", name, "`'s maxsteps is a `", mx.T, "`", _jax_where(ctx))
    helper = _jax_helper!(ctx, "_rk_ode_solve", _jax_ode_solve_source())
    ts = _jax_bind!(ctx, :ts, ts)
    n = isempty(ts.extent) ? ts.code * ".shape[0]" : ts.extent
    fun = _jax_ode_rhs!(ctx, _jax_function_of(rhs), extra)
    passed = [a for a in extra if !_jax_ode_closed(a)]
    solution = _jax_fresh(ctx, :solution)
    status = _jax_fresh(ctx, :status)
    call_args = ["_RK" * _jax_ode_solver(f) * "()", fun, y0.code, t0.code, ts.code, n,
                 rt.code, at.code, mx.code, _jax_tuple_code(passed)]
    active = _jax_active_condition(ctx)
    active === nothing || push!(call_args, "active=" * active)
    _jax_emit!(ctx, solution * ", " * status * " = " * helper * "(" *
               join(call_args, ", ") * ")")
    label = string(name, " (", _jax_ode_solver(f), ")")
    # Invalid inputs raise, as natively (`ArgumentError`); an exceeded budget
    # or a failed solve is a numerical failure (`ODESolveFailure` natively),
    # where a density function answers -inf.
    for (code, message) in ((1, " has an invalid initial state, output times, tolerances or maxsteps"),
                            (2, " exceeded maxsteps between requested outputs"),
                            (3, " failed"))
        _jax_masked_check(ctx, _PyVal(status * " != " * string(code), Bool), repr(label * message);
                          reject = code != 1)
    end
    _PyVal(solution, Matrix{Float64}, false, ts.extent, "", isempty(ts.extent) ? 0 : 1)
end

# An extra argument the right-hand side reads where it was called instead of
# through diffrax's `args`: a function or type, an aggregate of translated
# values, or a static value (a shape or literal the body may branch on).
_jax_ode_closed(v::_PyVal) = _jax_translation_only(v) || v.static ||
    (v.T isa Type && v.T <: Function)

# The nested Python function `def <f>_rhs(t, y, args)` computing `f(t, y,
# extra...)`, emitted in the current scope before its call.
function _jax_ode_rhs!(ctx::_JaxContext, f, extra::Vector{_PyVal})
    outer = ctx.scope
    fname = _jax_fresh(ctx, Symbol(nameof_or(f), :_rhs))
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    t = _jax_identifier(inner.used, :t)
    y = _jax_identifier(inner.used, :y)
    a = _jax_identifier(inner.used, :args)
    inputs = copy(extra)
    params = String[]
    for (k, v) in enumerate(extra)
        _jax_ode_closed(v) && continue
        p = _jax_identifier(inner.used, Symbol(:arg, k))
        push!(params, p)
        inputs[k] = _PyVal(p, v.T, false, v.extent, v.limit, v.padded, v.origin, v.field_extents)
    end
    saved = copy(ctx.locals)
    ctx.scope = inner
    dy = try
        isempty(params) || _jax_emit!(ctx, join(params, ", ") * (length(params) == 1 ? "," : "") *
                                           " = " * a)
        _jax_call(ctx, f, vcat(_PyVal[_PyVal(t, Float64), _PyVal(y, Vector{Float64})], inputs))
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved)
    end
    _jax_rank(dy.T) == 1 && isempty(dy.extent) ||
        _jax_error("the right-hand side `", nameof_or(f), "` returns a `", dy.T,
                   "`, not a state vector", _jax_where(ctx))
    push!(outer.lines, "    "^outer.indent * "def " * fname * "(" * t * ", " * y * ", " * a * "):")
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return " * dy.code)
    union!(outer.used, inner.used)
    fname
end
