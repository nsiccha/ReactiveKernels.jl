"""Two float64 counterexamples for proposals to reuse a scan denominator."""
import jax
import jax.numpy as jnp


def direct(d, x, weight):
    return (x / d) * weight


def reciprocal(d, x, weight):
    return (x * (1.0 / d)) * weight


def scan_readouts(d, first, second, length, *, after):
    def divide(a, b):
        return a / d, b / d

    def step(carry, row):
        i, a, b = row
        outputs = jax.lax.cond(i < length, lambda: (a, b) if after else divide(a, b),
                               lambda: (0.0, 0.0))
        return carry, outputs

    positions = jnp.arange(first.size)
    _, (a, b) = jax.lax.scan(step, 0.0, (positions, first, second))
    if after:
        def scale(i, a, b):
            return jax.lax.cond(i < length, lambda: divide(a, b), lambda: (0.0, 0.0))
        a, b = jax.vmap(scale)(positions, a, b)
    return jnp.sum(a + b)


def main():
    jax.config.update("jax_enable_x64", True)
    print(jax.__version__, jax.default_backend())
    for fn in (direct, reciprocal):
        value, grad = jax.jit(jax.value_and_grad(fn))(2.0**400, 2.0**600, 2.0**450)
        print(fn.__name__, float(value), float(grad))

    first = jnp.tile(jnp.array([-2.0**903, 2.0**903]), 16)
    evaluate = jax.jit(jax.value_and_grad(scan_readouts), static_argnames=("after",))
    for after in (False, True):
        value, grad = evaluate(2.0**-60, first, jnp.zeros_like(first), first.size, after=after)
        print("after_scan" if after else "inside_scan", float(value), float(grad))


if __name__ == "__main__":
    main()
