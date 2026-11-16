import importlib.util
import sys
from pathlib import Path

import jax
import numpy as np
from scipy.special import betaln, expit, gammaln, logsumexp


def model(name):
    spec = importlib.util.spec_from_file_location(name, Path(sys.argv[1]) / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def field(value, name, position):
    # RK uses tuples for nested NamedTuples and dictionaries for named outputs.
    return value[name] if isinstance(value, dict) else value[position]


def recovered(output):
    parameters = field(output["_ppl_latents"], "parameters", 0)
    return field(parameters, "values", 0), np.asarray(field(parameters, "probabilities", 1))


def equations(closed):
    graph = getattr(closed, "jaxpr", closed)
    count = len(graph.eqns)
    for equation in graph.eqns:
        for value in equation.params.values():
            for child in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    count += equations(child)
    return count


def check_mixture():
    u = np.array([0.3])
    p = expit(u[0])
    m = model("mixture")
    inventories = {}
    for n, y in ((0, np.array([])), (2, np.array([0.7, 1.2])),
                 (20, np.array([0.7, 1.2])), (50, np.array([0.7, 1.2]))):
        data = m.prepare_data(n=n, y=y)
        states = np.arange(n+1)
        logprior = (gammaln(n+1)-gammaln(states+1)-gammaln(n-states+1)
                    + states*np.log(p)+(n-states)*np.log1p(-p))
        loglik = -0.5*np.sum((y[:, None]-states[None, :])**2+np.log(2*np.pi), axis=0)
        weights = np.exp(logprior+loglik-logsumexp(logprior+loglik))
        value, gradient = jax.jit(jax.value_and_grad(m.log_density))(u, **data)
        expected = 2*np.log(p)+3*np.log1p(-p)-betaln(2, 3)+logsumexp(logprior+loglik)
        np.testing.assert_allclose(value, expected, atol=1e-11)
        np.testing.assert_allclose(gradient, [2+sum(states*weights)-(5+n)*p], atol=1e-11)
        output = jax.jit(m.constrain)(u, **data)
        columns, actual_weights = recovered(output)
        np.testing.assert_array_equal(field(columns, "k", 0), states)
        np.testing.assert_allclose(actual_weights, weights, atol=1e-12)
        terms = np.asarray(field(output["pointwise"], "y", 0))
        assert terms.shape == y.shape
        np.testing.assert_allclose(sum(terms), logsumexp(logprior+loglik), atol=1e-11)
        inventories[n] = equations(jax.make_jaxpr(m.log_density)(u, **data))
    assert inventories[50] < 2*inventories[2], inventories
    print("Marginal Binomial JAX: density, ordinary gradient, joint state recovery,",
          "pointwise and empty observations passed; trace equation counts", inventories)


def check_joint():
    u = np.array([0.3])
    p = expit(u[0])
    m = model("joint")
    for y in (-0.2, 0.3, 1.8):
        data = m.prepare_data(y=y)
        diagonal = np.array([np.log1p(-p), np.log(p)])-0.5*((y-np.array([0, 2]))**2+np.log(2*np.pi))
        diagonal_weights = np.exp(diagonal-logsumexp(diagonal))
        expected_weights = np.diag(diagonal_weights)
        value, gradient = jax.jit(jax.value_and_grad(m.log_density))(u, **data)
        np.testing.assert_allclose(value, 2*np.log(p)+3*np.log1p(-p)-betaln(2, 3)+logsumexp(diagonal), atol=1e-11)
        np.testing.assert_allclose(gradient, [2+diagonal_weights[1]-6*p], atol=1e-11)
        output = jax.jit(m.constrain)(u, **data)
        columns, weights = recovered(output)
        np.testing.assert_allclose(weights, expected_weights, atol=1e-12)
        np.testing.assert_array_equal(field(columns, "k", 0), [[0, 0], [1, 1]])
        np.testing.assert_array_equal(field(columns, "z", 1), [[0, 1], [0, 1]])
        np.testing.assert_allclose(field(output["pointwise"], "y", 0), logsumexp(diagonal), atol=1e-11)
    print("Joint integer JAX: dependent two-axis support and zero-probability states passed")


def check_rag():
    u = np.array([0.3])
    p = expit(u[0])
    m = model("rag")
    for n, h in ((0, 0), (10, 0), (10, 8), (10, 10), (50, 40)):
        for a, b in ((1.0, 1.0), (2.0, 3.0)):
            data = m.prepare_data(n=n, h=h, a=a, b=b)
            value, gradient = jax.jit(jax.value_and_grad(m.log_density))(u, **data)
            expected_likelihood = (gammaln(n+1)-gammaln(h+1)-gammaln(n-h+1)
                                   + h*np.log1p(p)+(n-h)*np.log1p(-p)-n*np.log(2))
            expected = a*np.log(p)+b*np.log1p(-p)-betaln(a, b)+expected_likelihood
            expected_gradient = a-(a+b+n-h)*p+h*p*(1-p)/(1+p)
            np.testing.assert_allclose(value, expected, atol=1e-11)
            np.testing.assert_allclose(gradient, [expected_gradient], atol=1e-11)
            output = jax.jit(m.constrain)(u, **data)
            columns, weights = recovered(output)
            states = np.arange(n+1)
            np.testing.assert_array_equal(field(columns, "c", 0), states)
            conditional_p = 2*p/(1+p)
            expected_weights = np.zeros(n+1)
            allowed = np.arange(h+1)
            expected_weights[:h+1] = np.exp(gammaln(h+1)-gammaln(allowed+1)-gammaln(h-allowed+1)
                + allowed*np.log(conditional_p)+(h-allowed)*np.log1p(-conditional_p))
            np.testing.assert_allclose(weights, expected_weights, atol=1e-12)
            # r is a density on a deterministic definition, with no explicit
            # conditioning. It contributes to prior; the total target above is
            # still the RAG joint density. There are no observation terms here.
            assert len(output["pointwise"]) == 0
    print("Marginal RAG JAX: closed-form density, ordinary gradient and exact conditional passed")


checks = {"mixture": check_mixture, "joint": check_joint, "rag": check_rag}
for name in sys.argv[2:] or checks:
    checks[name]()
