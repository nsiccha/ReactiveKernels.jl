"""JAX-only reproducer: `jax.grad` through `checkify` mishandles a `custom_vjp`
once checkify holds an error state (snag rk-ode-solve-fai-3c664466).

Every diffrax solve carries a `custom_vjp` (`RecursiveCheckpointAdjoint`), so
an emitted module whose failed-solve checks were `checkify.check`s could not
be differentiated where a solve sits in a `lax.scan` step, a `lax.cond` arm or
a `lax.fori_loop` body, and its gradient can be silently wrong at top level.

Each case compares `jax.grad` of the checkified value with the analytic
gradient. With jax 0.12.0.1 (the latest release, CPU) and with JAX main at
`daa254964d245e583392b467037f116851109a82` (2026-11-16, run from source over
jaxlib 0.12.0.1) the results are the same:

- `WRONG`: a top-level `custom_vjp` whose residual is its own input, after any
  earlier check, differentiates to 1.0 instead of cos(0.3) = 0.955, with no
  error. The bound inputs start with the error leaves, but the residuals'
  input-forwarding indices are not shifted past them, so the error predicate
  is used as the residual.
- `FAILS`: a scan step, cond arm or fori_loop body with a `custom_vjp` and a
  check, or after one (`ValueError: foreach() argument 2 is shorter than
  argument 1`, from `scan_error_check` / `_scan_linearize`). The recorded
  `custom_vjp_call`'s fwd returns no error outputs, while its checkified fun
  returns them.
- `FAILS`: inside a `vmap`, a check on the cell's data beside a `custom_vjp`
  (`IndexError: tuple index out of range`, in the batched bwd).

Both defects sit in `custom_vjp_call_rule` in `jax/_src/checkify.py`.

Run: `kb-jax-python -I benchmark/repro_jax_checkify_custom_vjp_control_flow.py`.
"""
import jax
import jax.numpy as jnp
from jax.experimental import checkify

jax.config.update("jax_enable_x64", True)


@jax.custom_vjp
def sin_input_residual(x):
    return jnp.sin(x)


sin_input_residual.defvjp(lambda x: (jnp.sin(x), x), lambda x, g: (g * jnp.cos(x),))


@jax.custom_vjp
def sin_computed_residual(x):
    return jnp.sin(x)


sin_computed_residual.defvjp(lambda x: (jnp.sin(x), jnp.cos(x)), lambda c, g: (g * c,))


@jax.custom_jvp
def sin_custom_jvp(x):
    return jnp.sin(x)


sin_custom_jvp.defjvp(lambda p, t: (jnp.sin(p[0]), jnp.cos(p[0]) * t[0]))

s = sin_input_residual
cs = jnp.arange(3.0)


def scan_of(fn, x, check_inside=False):
    def step(acc, c):
        y = fn(x * c)
        if check_inside:
            checkify.check(y < 10, "inside")
        return acc + y, None
    return jax.lax.scan(step, 0.0, cs)[0]


def loop_of(fn, x):
    def body(i, acc):
        y = fn(x * i)
        checkify.check(y < 10, "loop")
        return acc + y
    return jax.lax.fori_loop(0, 3, body, 0.0)


def cos_(x):
    return jnp.cos(x)


def scan_grad(x):
    return jnp.sum(cs * jnp.cos(x * cs))


CASES = {
    # name: (outcome with stock JAX, function of x, analytic gradient)
    "top level, input residual, no check": (
        "ok", lambda x: s(x), cos_),
    "top level, input residual, check before": (
        "WRONG", lambda x: (checkify.check(x < 10, "before"), s(x))[1], cos_),
    "top level, computed residual, check before": (
        "ok", lambda x: (checkify.check(x < 10, "before"), sin_computed_residual(x))[1], cos_),
    "two custom_vjps, check between": (
        "FAILS", lambda x: (lambda a: (checkify.check(a < 10, "mid"), a + s(2 * x))[1])(s(x)),
        lambda x: jnp.cos(x) + 2 * jnp.cos(2 * x)),
    "scan step, no check": (
        "ok", lambda x: scan_of(s, x), scan_grad),
    "scan step, check before the scan": (
        "FAILS", lambda x: (checkify.check(x < 10, "before"), scan_of(s, x))[1], scan_grad),
    "scan step, check inside": (
        "FAILS", lambda x: scan_of(s, x, check_inside=True), scan_grad),
    "scan step, check inside, computed residual": (
        "FAILS", lambda x: scan_of(sin_computed_residual, x, check_inside=True), scan_grad),
    "scan step with a custom_jvp, check before the scan": (
        "ok", lambda x: (checkify.check(x < 10, "before"), scan_of(sin_custom_jvp, x))[1], scan_grad),
    "cond arm, check inside": (
        "FAILS", lambda x: jax.lax.cond(
            x > 0, lambda: (checkify.check(x < 10, "arm"), s(x))[1], lambda: 0.0), cos_),
    "fori_loop body, check inside": (
        "FAILS", lambda x: loop_of(s, x), scan_grad),
    "vmapped body, check on the custom_vjp's value": (
        "ok", lambda x: jnp.sum(jax.vmap(
            lambda c: (lambda y: (checkify.check(y < 10, "cell"), y)[1])(s(x * c)))(cs)), scan_grad),
    "vmapped body, check on the cell's data": (
        "FAILS", lambda x: jnp.sum(jax.vmap(
            lambda c: (checkify.check(c < 10, "cell"), s(x * c))[1])(cs)), scan_grad),
}

x0 = 0.3
for name, (stock, f, analytic) in CASES.items():
    def value(x, f=f):
        _, v = checkify.checkify(lambda: f(x))()
        return v
    expected = float(analytic(x0))
    try:
        got = float(jax.grad(value)(x0))
        outcome = "ok" if abs(got - expected) < 1e-12 else "WRONG"
        detail = f"grad {got:.15f}, analytic {expected:.15f}"
    except Exception as e:
        outcome = "FAILS"
        detail = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    print(f"{outcome:5} (stock JAX: {stock:5}) {name}: {detail}")
