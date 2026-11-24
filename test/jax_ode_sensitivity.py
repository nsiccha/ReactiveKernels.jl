"""Regression of the exported RK45 gradient at and away from an equilibrium."""
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, sys.argv[1])
import model


def exact(q, times):
    initial, p, start = q
    return jnp.sum(p + (initial - p) * jnp.exp(start - times))


@jax.custom_jvp
def identity(x):
    return jnp.where(x == 0, 0., x * x / x)


@identity.defjvp
def identity_jvp(primals, tangents):
    return identity(*primals), tangents[0]


@jax.custom_vjp
def reverse_identity(x):
    return identity(x)


reverse_identity.defvjp(lambda x: (identity(x), ()), lambda _, cotangent: (cotangent,))


def check(mode):
    model.RK_ODE_KERNEL = mode
    solve = jax.jit(lambda q, t: model.kernel(q, t))
    grad = jax.jit(jax.value_and_grad(lambda q, t: model.kernel(q, t), (0, 1)))
    analytic = jax.value_and_grad(exact, (0, 1))
    for q, times in ((jnp.zeros(3), jnp.array([100.])),
                     (jnp.array([-.2, .7, .1]), jnp.array([.5, 1., 3., 100.]))):
        value, (dq, dt) = grad(q, times)
        ref, (ref_dq, ref_dt) = analytic(q, times)
        np.testing.assert_allclose(value, solve(q, times), atol=2e-6)
        np.testing.assert_allclose(value, ref, atol=2e-6)
        np.testing.assert_allclose(dq, ref_dq, atol=2e-6)
        # Differentiating the dense interpolant with respect to the query
        # time divides its coefficient rounding by the accepted step length.
        time_atol = 2e-5 if times.dtype == jnp.float32 else 2e-6
        np.testing.assert_allclose(dt, ref_dt, atol=time_atol)
    q, t = jnp.zeros(3), jnp.array([100.])
    h = .01
    fd = (solve(q.at[1].add(h), t) - solve(q.at[1].add(-h), t)) / (2 * h)
    np.testing.assert_allclose(fd, 1., atol=2e-6)
    assert solve(q, jnp.array([])) == 0

    def vector_total(p):
        def field(t, state, args):
            return jnp.sum(args[0]) / args[0].size - state

        states, status = model._rk_ode_solve(
            model._RKDopri5(), field, jnp.zeros(1), 0., t, 1,
            1e-6, 1e-7, 10000, (p,))
        return states.sum(), status

    p = jnp.zeros(7)
    (value, status), g = jax.jit(jax.value_and_grad(vector_total, has_aux=True))(p)
    assert status == 0 and value == 0
    np.testing.assert_allclose(g, np.full(7, 1 / 7), atol=2e-6)

    values, (dq, dt) = jax.vmap(grad, in_axes=(0, None))(jnp.zeros((0, 3)), t)
    assert (values.shape, dq.shape, dt.shape) == ((0,), (0, 3), (0, 1))

    # The scalar-to-vector RHS must keep an authored derivative at the
    # removable singularity on both the checkpointed and forward paths.
    for op in (identity, reverse_identity):
        def total(p):
            def field(t, state, args):
                return op(args[0]) - state

            states, status = model._rk_ode_solve(
                model._RKDopri5(), field, jnp.zeros(3), 0., t, 1,
                1e-6, 1e-7, 10000, (p,))
            return states.sum(), status

        (value, status), g = jax.jit(jax.value_and_grad(total, has_aux=True))(0.)
        assert status == 0 and value == 0
        np.testing.assert_allclose(g, 3., atol=2e-6)


check("off")
if model._pl is not None:
    check("interpret")
    if jax.default_backend() == "gpu":
        check("auto")
