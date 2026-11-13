# A joint numerical graph keeps the expensive exponential available to both
# the stable value and its partial. Backend adapters select cuts of this graph.
@kernel rk_log1pexp_graph(x::Float64) = begin
    q::Float64 = exp(-abs(x))
    value::Float64 = max(x, 0.0) + log1p(q)
    partial::Float64 = ifelse(x >= 0, 1.0, q) / (1 + q)
    return value, partial
end

"""
    rk_log1pexp(x::Float64)

Stable `log(1 + exp(x))` with a generated scalar derivative rule. The joint
value/partial graph shares `exp(-abs(x))`; neither tail evaluates `exp(abs(x))`.
The primal-only call selects the value cut. Ordinary supported AD backends
select the same graph's active derivative cut.
"""
const rk_log1pexp = scalar_derivative_rule(rk_log1pexp_graph;
    primal=:value, partials=(x=:partial,), name=:rk_log1pexp)
