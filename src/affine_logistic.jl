# A bounded affine logistic map. Its ordinary reverse path multiplies the
# incoming cotangent by the interval width before the sigmoid slope; that
# intermediate can overflow even when the derivative is representable. Keep
# the primal and its partials in one graph for the generated native AD adapter.
@kernel rk_affine_logistic_graph(lo::Float64, hi::Float64, u::Float64) = begin
    y::Float64 = lo + (hi - lo) / (1 + exp(-u))
    toward_hi::Float64 = exp(-log1pexp(-u))
    toward_lo::Float64 = exp(-log1pexp(u))
    dy_dlo::Float64 = toward_lo
    dy_dhi::Float64 = toward_hi
    dy_du::Float64 = ((hi - lo) * toward_hi) * toward_lo
    return y, dy_dlo, dy_dhi, dy_du
end

"""
    rk_affine_logistic(lo, hi, u)

Map an unconstrained scalar `u` into `(lo, hi)` with the affine logistic
primal. Its generated derivative rule multiplies an incoming cotangent by
the complete interval partial, avoiding overflow in a width-first reverse
chain when the final derivative is representable.
"""
const rk_affine_logistic = scalar_derivative_rule(rk_affine_logistic_graph;
    primal = :y,
    partials = (lo = :dy_dlo, hi = :dy_dhi, u = :dy_du),
    name = :rk_affine_logistic)
