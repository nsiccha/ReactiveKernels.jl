import importlib.util
import sys
from pathlib import Path

import jax
import numpy as np
from scipy.special import betaln, expit, gammaln


def model(name):
    spec = importlib.util.spec_from_file_location(name, Path(sys.argv[1]) / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = model("scalar")
y = np.array([1.2, 2.1])
data = m.prepare_data(y=y)
for u0 in (-1.2, 0.3, 1.0):
    u, p = np.array([u0]), expit(u0)
    for k in (0, 1):
        value, gradient = jax.jit(jax.value_and_grad(m.log_density))(u, {"k": k}, **data)
        expected = ((2 + k) * np.log(p) + (4 - k) * np.log1p(-p) - betaln(2, 3)
                    - 0.5 * np.sum((y - k) ** 2 + np.log(2 * np.pi)))
        np.testing.assert_allclose(value, expected, rtol=1e-12)
        np.testing.assert_allclose(gradient, [2 + k - 6 * p], rtol=1e-12)
for state in ({"k": 1.5}, {"k": np.array([1])}):
    try:
        m.log_density(np.array([0.0]), state, **data)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("invalid integer state accepted")


def equations(closed):
    graph = getattr(closed, "jaxpr", closed)
    count = len(graph.eqns)
    for equation in graph.eqns:
        for value in equation.params.values():
            for child in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    count += equations(child)
    return count


m = model("arrays")
inventories = {}
for n in (0, 2, 5, 50):
    u = np.linspace(-0.3, 0.3, n)
    k, y = np.arange(n) % 2, np.linspace(1.0, 2.0, n)
    data = m.prepare_data(x=np.zeros(n), p=0.4, y=y)
    assert m.dimension(data) == n
    value, gradient = jax.jit(jax.value_and_grad(m.log_density))(u, {"k": k}, **data)
    expected = (-0.5 * np.sum(u * u + np.log(2 * np.pi))
                + np.sum(k * np.log(0.4) + (1 - k) * np.log(0.6))
                - 0.5 * np.sum((y - u - k) ** 2 + np.log(2 * np.pi)))
    np.testing.assert_allclose(value, expected, rtol=1e-12)
    np.testing.assert_allclose(gradient, y - k - 2 * u, atol=1e-12)
    restored = jax.jit(m.constrain)(u, {"k": k}, **data)
    np.testing.assert_array_equal(restored["k"], k)
    inventories[n] = equations(jax.make_jaxpr(m.log_density)(u, {"k": k}, **data))
# A growth diagnostic in addition to value/gradient parity. The same emitted
# module is used above; increasing a data extent must not replicate its body.
assert inventories[50] < 2 * inventories[5], inventories
print("Integer JAX: values, gradients, restoration, validation and shape rebinding passed;",
      "trace equation counts", inventories)

m = model("categorical")
for count in (3, 5):
    alpha = np.linspace(1.5, 3.0, count)
    data = m.prepare_data(alpha=alpha, y=y[:2])
    u = np.linspace(-0.2, 0.3, count-1)
    assert m.dimension(data) == count-1
    for k in range(1, count+1):
        theta = np.asarray(jax.jit(m.constrain)(u, {"k": k}, **data)["theta"])
        value, gradient = jax.jit(jax.value_and_grad(m.log_density))(u, {"k": k}, **data)
        expected = (gammaln(sum(alpha))-sum(gammaln(alpha)) + sum(alpha*np.log(theta))
                    + np.log(theta[k-1])-0.5*np.sum((y[:2]-k)**2+np.log(2*np.pi)))
        reference_gradient = [alpha[i]+int(k==i+1)
                              -(sum(alpha[i:])+int(k>=i+1))*theta[i]/sum(theta[i:])
                              for i in range(count-1)]
        np.testing.assert_allclose(value, expected, rtol=1e-12)
        np.testing.assert_allclose(gradient, reference_gradient, atol=1e-12)
print("Categorical JAX: sampled simplex, ordinary gradients, and category-count rebinding passed")

m = model("uniform")
y = np.array([1.2, 2.1])
u = np.array([0.3])
for lo, hi in ((2, 7), (1, 3), (-3, 3)):
    data = m.prepare_data(lo=lo, hi=hi, y=y)
    for k in (lo, hi):
        value, gradient = jax.jit(jax.value_and_grad(m.log_density))(u, {"k": k}, **data)
        expected = (-0.5*(u[0]**2+np.log(2*np.pi))-np.log(hi-lo+1)
                    -0.5*np.sum((y-u[0]-k)**2+np.log(2*np.pi)))
        np.testing.assert_allclose(value, expected, rtol=1e-12)
        np.testing.assert_allclose(gradient, [-u[0]+sum(y-u[0]-k)], rtol=1e-12)
print("DiscreteUniform JAX: values, ordinary gradients, and endpoint rebinding passed")

m = model("binomial")
inventories = {}
for n, y in ((0, np.array([])), (5, np.array([1.2, 2.1])),
             (12, np.array([0.7])), (50, np.array([0.7]))):
    data = m.prepare_data(n=n, y=y)
    p = expit(u[0])
    for k in sorted({0, n//2, n}):
        value, gradient = jax.jit(jax.value_and_grad(m.log_density))(u, {"k": k}, **data)
        expected = ((2+k)*np.log(p)+(3+n-k)*np.log1p(-p)-betaln(2, 3)
                    + gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
                    - 0.5*np.sum((y-k)**2+np.log(2*np.pi)))
        np.testing.assert_allclose(value, expected, rtol=1e-12)
        np.testing.assert_allclose(gradient, [2+k-(5+n)*p], atol=1e-12)
        assert int(jax.jit(m.constrain)(u, {"k": k}, **data)["k"]) == k
    for k in (-1, n+1):
        assert np.isneginf(jax.jit(m.log_density)(u, {"k": k}, **data))
    inventories[n] = equations(jax.make_jaxpr(m.log_density)(u, {"k": 0}, **data))
assert inventories[50] < 2*inventories[12], inventories
print("Binomial JAX: values, gradients, restoration and support rebinding passed;",
      "trace equation counts", inventories)

m = model("rag")
for n, h in ((0, 0), (10, 0), (10, 8), (10, 10), (50, 40)):
    for a, b in ((1.0, 1.0), (2.0, 3.0)):
        data = m.prepare_data(n=n, h=h, a=a, b=b)
        p = expit(u[0])
        for c in sorted({0, h//2, h}):
            value, gradient = jax.jit(jax.value_and_grad(m.log_density))(u, {"c": c}, **data)
            expected = ((a+c)*np.log(p)+(b+n-c)*np.log1p(-p)-betaln(a, b)
                        + gammaln(n+1)-gammaln(c+1)-gammaln(n-c+1)
                        + gammaln(n-c+1)-gammaln(h-c+1)-gammaln(n-h+1)
                        - (n-c)*np.log(2))
            np.testing.assert_allclose(value, expected, rtol=1e-12)
            np.testing.assert_allclose(gradient, [a+c-(a+b+n)*p], atol=1e-12)
        if h < n:
            assert np.isneginf(jax.jit(m.log_density)(u, {"c": h+1}, **data))
print("RAG JAX: integer-dependent observation trials and impossible observations passed")

m = model("matrix")
p = np.array([0.2, 0.3, 0.5])
inventories = {}
for shape in ((2, 3), (1, 0), (0, 3), (4, 6)):
    k = (np.arange(np.prod(shape)).reshape(shape) % 3)+1
    y = np.full(shape, 0.4)
    data = m.prepare_data(x=np.zeros(shape), p=p, y=y)
    value, gradient = jax.jit(jax.value_and_grad(m.log_density))(u, {"k": k}, **data)
    expected = (-0.5*(u[0]**2+np.log(2*np.pi))+np.sum(np.log(p[k-1]))
                - 0.5*np.sum((y-u[0]-k)**2+np.log(2*np.pi)))
    np.testing.assert_allclose(value, expected, rtol=1e-12)
    np.testing.assert_allclose(gradient, [-u[0]+np.sum(y-u[0]-k)], atol=1e-12)
    np.testing.assert_array_equal(jax.jit(m.constrain)(u, {"k": k}, **data)["k"], k)
    inventories[shape] = equations(jax.make_jaxpr(m.log_density)(u, {"k": k}, **data))
assert inventories[(4, 6)] < 2*inventories[(2, 3)], inventories
print("Categorical matrix JAX: shape, empty axes, ordinary gradients and restoration passed;",
      "trace equation counts", inventories)
