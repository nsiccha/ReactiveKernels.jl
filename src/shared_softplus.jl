# The value-only recipe skips work in both tails and returns the joint
# recipe's Float64 bits there. For `x >= 34`, `exp(-x) < 2^-48 <= eps(x) / 2`,
# so `x + log1p(exp(-x))` rounds to `x`: no `exp`. For `x < -37`,
# `exp(x) < 2^-53`, where Julia's `log1p` returns its argument: no `log1p`.
# Between them it is the joint recipe's branch-free value.
@_record_source @inline _rk_log1pexp_value(x) =
    x >= 34.0 ? x : x < -37.0 ? exp(x) : max(x, 0.0) + log1p(exp(-abs(x)))

# A joint numerical graph keeps the expensive exponential available to both
# the stable value and its partial. Planning selects the value-only recipe for
# a primal call and the joint recipe for a derivative cut.
@kernel rk_log1pexp_graph(x::Float64) = begin
    value::Float64 = _rk_log1pexp_value(x)
    q::Float64 = exp(-abs(x))
    (value, partial::Float64) = (max(x, 0.0) + log1p(q), ifelse(x >= 0, 1.0, q) / (1 + q))
    return value, partial
end

"""
    rk_log1pexp(x::Float64)

Stable `log(1 + exp(x))` with a generated scalar derivative rule. The joint
value/partial graph shares `exp(-abs(x))`; neither tail evaluates `exp(abs(x))`.
A primal-only call selects a value recipe that returns `x` for `x >= 34`
without an `exp` and `exp(x)` for `x < -37` without a `log1p`, the same
Float64 values. Ordinary supported AD backends select the joint
value/partial recipe.
"""
const rk_log1pexp = scalar_derivative_rule(rk_log1pexp_graph;
    primal=:value, partials=(x=:partial,), name=:rk_log1pexp)
