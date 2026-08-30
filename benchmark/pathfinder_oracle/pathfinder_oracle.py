#!/usr/bin/env python3
"""Standalone Pathfinder Algorithms 3/4 oracle using only Python's stdlib."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


PAPER_URL = "https://jmlr.org/papers/volume23/21-0889/21-0889.pdf"
PAPER_SHA256 = "8fe38816d4953e5b4e01a8b531abb9f3ea1d1f92041f6c2a7ce5e9c7037c8435"


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def matmul(left, right):
    right_t = transpose(right)
    return [[dot(row, column) for column in right_t] for row in left]


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def outer(left, right):
    return [[a * b for b in right] for a in left]


def matrix_add(left, right):
    return [[a + b for a, b in zip(lrow, rrow)]
            for lrow, rrow in zip(left, right)]


def matrix_scale(scale, matrix):
    return [[scale * value for value in row] for row in matrix]


def matrix_subtract(left, right):
    return matrix_add(left, matrix_scale(-1.0, right))


def cholesky(matrix):
    dimension = len(matrix)
    factor = [[0.0] * dimension for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row + 1):
            residual = matrix[row][column] - sum(
                factor[row][k] * factor[column][k] for k in range(column)
            )
            if row == column:
                factor[row][column] = math.sqrt(residual)
            else:
                factor[row][column] = residual / factor[column][column]
    return factor


def columns(matrix):
    return transpose(matrix)


def add_vector_to_columns(vector, matrix):
    return [[vector[row] + matrix[row][column]
             for column in range(len(matrix[0]))]
            for row in range(len(vector))]


def gaussian_logdensity(draws, precision):
    dimension = len(precision)
    determinant = (precision[0][0] * precision[1][1]
                   - precision[0][1] * precision[1][0])
    normalizer = (-0.5 * dimension * math.log(2.0 * math.pi)
                  + 0.5 * math.log(determinant))
    return [normalizer - 0.5 * dot(draw, matvec(precision, draw))
            for draw in columns(draws)]


def candidate(logdensity, position, gradient, alpha, step, gradient_delta,
              identity, elbo_noise, output_noise, tolerance):
    curvature = dot(step, gradient_delta)
    accepted = curvature > tolerance * dot(gradient_delta, gradient_delta)
    if accepted:
        alpha_a = dot(gradient_delta,
                      [a * z for a, z in zip(alpha, gradient_delta)])
        alpha_c = dot(step, [s / a for s, a in zip(step, alpha)])
        recovered = [
            1.0 / ((alpha_a / curvature) * a + z * z / curvature
                   - (alpha_a / (curvature * alpha_c)) * s * s / (a * a))
            for a, s, z in zip(alpha, step, gradient_delta)
        ]
        alpha_next = recovered
        rho = 1.0 / curvature
    else:
        alpha_next = list(alpha)
        rho = 0.0

    diagonal = [[identity[row][column] * alpha_next[column]
                 for column in range(len(alpha_next))]
                for row in range(len(alpha_next))]
    transform = matrix_subtract(identity, matrix_scale(
        rho, outer(step, gradient_delta)))
    covariance = matrix_add(
        matmul(matmul(transform, diagonal), transpose(transform)),
        matrix_scale(rho, outer(step, step)),
    )
    mean = [theta + delta for theta, delta
            in zip(position, matvec(covariance, gradient))]
    factor = cholesky(covariance)
    elbo_draws = add_vector_to_columns(mean, matmul(factor, elbo_noise))
    output_draws = add_vector_to_columns(mean, matmul(factor, output_noise))
    half_logdet = sum(math.log(factor[i][i]) for i in range(len(factor)))
    log_q = [
        -0.5 * len(position) * math.log(2.0 * math.pi)
        - half_logdet - 0.5 * dot(noise, noise)
        for noise in columns(elbo_noise)
    ]
    target = logdensity(elbo_draws)
    elbo = sum(p - q for p, q in zip(target, log_q)) / len(log_q)
    return {
        "alpha_next": alpha_next,
        "curvature_accepted": accepted,
        "covariance": covariance,
        "mean": mean,
        "elbo_draws": elbo_draws,
        "log_q": log_q,
        "elbo": elbo,
        "output_draws": output_draws,
    }


def fixture():
    precision = [[1.5, 0.25], [0.25, 0.8]]
    positions = [[-1.0, 0.8], [-0.55, 0.40], [-0.22, 0.12], [-0.05, 0.02]]
    gradients = [[-value for value in matvec(precision, position)]
                 for position in positions]
    initial_alpha = [0.7, 1.2]
    identity = [[1.0, 0.0], [0.0, 1.0]]
    elbo_noise = [
        [[-1.0, 0.2, 1.1, -0.4], [0.5, -1.2, 0.3, 0.8]],
        [[0.7, -0.6, 0.1, 1.3], [-0.9, 0.4, 1.0, -0.2]],
        [[-0.3, 0.9, -1.1, 0.6], [1.2, -0.5, 0.2, -1.0]],
    ]
    output_noise = [
        [[0.15, -0.75, 1.25], [-0.35, 0.95, 0.45]],
        [[-1.1, 0.55, 0.25], [0.65, -0.15, 1.35]],
        [[0.8, -0.2, -0.95], [-0.6, 1.1, 0.05]],
    ]
    return {
        "precision": precision,
        "positions": positions,
        "gradients": gradients,
        "initial_alpha": initial_alpha,
        "identity": identity,
        "elbo_noise": elbo_noise,
        "output_noise": output_noise,
        "curvature_tolerance": 1e-12,
    }


def format_toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, int, float)):
        return json.dumps(value, separators=(",", ":"), allow_nan=False)
    raise TypeError(type(value))


def emit_section(name, values, *, trailing_blank=True):
    print(f"[{name}]")
    for key, value in values.items():
        print(f"{key} = {format_toml_value(value)}")
    if trailing_blank:
        print()


def main():
    # Keep the committed receipt byte-reproducible on Windows, where Python's
    # default text stdout otherwise expands each LF to CRLF.
    sys.stdout.reconfigure(newline="\n")
    inputs = fixture()
    alpha = list(inputs["initial_alpha"])
    results = []
    logdensity = lambda draws: gaussian_logdensity(draws, inputs["precision"])
    for index in range(1, len(inputs["positions"])):
        position = inputs["positions"][index]
        previous = inputs["positions"][index - 1]
        gradient = inputs["gradients"][index]
        previous_gradient = inputs["gradients"][index - 1]
        step = [current - old for current, old in zip(position, previous)]
        gradient_delta = [old - current for old, current
                          in zip(previous_gradient, gradient)]
        result = candidate(
            logdensity,
            position,
            gradient,
            alpha,
            step,
            gradient_delta,
            inputs["identity"],
            inputs["elbo_noise"][index - 1],
            inputs["output_noise"][index - 1],
            inputs["curvature_tolerance"],
        )
        results.append(result)
        alpha = result["alpha_next"]

    best_index = max(range(len(results)), key=lambda index: results[index]["elbo"])
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    emit_section("authority", {
        "paper_url": PAPER_URL,
        "paper_sha256": PAPER_SHA256,
        "algorithms": [1, 3, 4],
        "oracle_source_sha256": source_sha256,
        "oracle_engine": "python-stdlib",
    })
    emit_section("path", {
        "precision": inputs["precision"],
        "positions": inputs["positions"],
        "gradients": inputs["gradients"],
        "initial_alpha": inputs["initial_alpha"],
        "identity": inputs["identity"],
        "elbo_noise": inputs["elbo_noise"],
        "output_noise": inputs["output_noise"],
        "curvature_tolerance": inputs["curvature_tolerance"],
        "best_index": best_index + 1,
    })
    for index, result in enumerate(results, start=1):
        emit_section(
            f"candidate_{index}",
            result,
            trailing_blank=index != len(results),
        )


if __name__ == "__main__":
    main()
