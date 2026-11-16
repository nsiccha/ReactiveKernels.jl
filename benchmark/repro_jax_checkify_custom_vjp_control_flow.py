"""JAX-only reproducer: `jax.grad` through `checkify` fails at a control-flow body
with a `custom_vjp` once checkify has an error state (snag
rk-ode-solve-fai-3c664466).

Every diffrax solve carries a `custom_vjp` (`RecursiveCheckpointAdjoint`), so
an emitted module whose failed-solve checks were `checkify.check`s could not
be differentiated where a solve sits in a `lax.scan` step, a `lax.cond` arm or
a `lax.fori_loop` body after any earlier check, or with a check in that body.
Inside a `vmap`, a check whose predicate reads only the cell's data beside a
`custom_vjp` fails too (`IndexError`), while one on a computed value works. A
`custom_jvp` in place of the `custom_vjp` differentiates.

Run: `kb-jax-python -I benchmark/repro_jax_checkify_custom_vjp_control_flow.py`.
Measured with jax 0.12.0.1 (CPU): the cases marked `fails` raise
`ValueError: foreach() argument 2 is shorter than argument 1` (from
`checkify.scan_error_check` / `_scan_linearize` / `eval_jaxpr`) or
`IndexError: tuple index out of range` (the vmapped data check).
"""
import jax
import jax.numpy as jnp
from jax.experimental import checkify

jax.config.update("jax_enable_x64", True)


@jax.custom_vjp
def custom_vjp_sin(x):
    return jnp.sin(x)


custom_vjp_sin.defvjp(lambda x: (jnp.sin(x), x), lambda x, g: (g * jnp.cos(x),))


@jax.custom_jvp
def custom_jvp_sin(x):
    return jnp.sin(x)


custom_jvp_sin.defjvp(lambda p, t: (jnp.sin(p[0]), jnp.cos(p[0]) * t[0]))


def scan_of(fn, x, check_inside=False):
    def step(s, c):
        y = fn(x * c)
        if check_inside:
            checkify.check(y < 10, "inside")
        return s + y, None
    return jax.lax.scan(step, 0.0, jnp.arange(3.0))[0]


CASES = {
    # (expected, function of x)
    "custom_vjp in a scan step, no check": ("works", lambda x: scan_of(custom_vjp_sin, x)),
    "custom_vjp in a scan step, check before the scan": (
        "fails", lambda x: (checkify.check(x < 10, "before"), scan_of(custom_vjp_sin, x))[1]),
    "custom_vjp and a check in a scan step": (
        "fails", lambda x: scan_of(custom_vjp_sin, x, check_inside=True)),
    "custom_vjp and a check in a cond arm": (
        "fails", lambda x: jax.lax.cond(
            x > 0, lambda: (checkify.check(x < 10, "arm"), custom_vjp_sin(x))[1], lambda: 0.0)),
    "custom_vjp and a check on its value in a vmapped body": (
        "works", lambda x: jnp.sum(jax.vmap(
            lambda c: (lambda y: (checkify.check(y < 10, "cell"), y)[1])(custom_vjp_sin(x * c)))(
                jnp.arange(3.0)))),
    "custom_vjp and a check on the cell's data in a vmapped body": (
        "fails", lambda x: jnp.sum(jax.vmap(
            lambda c: (checkify.check(c < 10, "cell"), custom_vjp_sin(x * c))[1])(jnp.arange(3.0)))),
    "custom_jvp in a scan step, check before the scan": (
        "works", lambda x: (checkify.check(x < 10, "before"), scan_of(custom_jvp_sin, x))[1]),
}

for name, (expected, f) in CASES.items():
    def value(x, f=f):
        _, v = checkify.checkify(lambda: f(x))()
        return v
    try:
        jax.grad(value)(0.3)
        got = "works"
    except Exception as e:
        got = "fails"
        detail = f"{type(e).__name__}: {e}"
    print(f"{got:5} (expected {expected}): {name}" + (f": {detail}" if got == "fails" else ""))
