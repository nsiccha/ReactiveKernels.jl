# Compiler-friendly lowering for the Lambert W branch used by the
# source-faithful ReactiveHMC RKE fixture. This is an explicit implementation
# supplied through `pure_callable_port`; neither the generic stateful compiler
# nor the authored RKE kernel contains LambertW-specific dispatch.
module ReactiveHMCRKEFunctionalLowering

"""
    lambertw_minus_one(x, branch)

Real-valued Lambert W on branch -1 for `-1/e <= x < 0`. A fixed Halley
schedule makes the implementation traceable without data-dependent host
control. The branch argument remains explicit so the callable has the same
surface as `LambertW.lambertw` at the captured source boundary.
"""
function lambertw_minus_one(x, branch::Int)
    branch == -1 || throw(ArgumentError(
        "the RKE functional lowering admits only Lambert W branch -1"))

    one_x = one(x)
    branchpoint = -exp(-one_x)
    p = -sqrt(max(zero(x), 2 * (one_x + exp(one_x) * x)))
    near = -one_x + p - p^2 / 3 + 11 * p^3 / 72 - 43 * p^4 / 540

    l1 = log(-x)
    l2 = log(-l1)
    far = l1 - l2 + l2 / l1
    w = ifelse(x < -one_x / 4, near, far)
    for _ in 1:16
        ew = exp(w)
        residual = w * ew - x
        w -= residual /
             (ew * (w + one_x) -
              (w + 2 * one_x) * residual / (2 * w + 2 * one_x))
    end

    # Preserve the real branch endpoints without making traced data control
    # host execution. Interior RKE quantiles use the converged Halley result.
    w = ifelse(x <= branchpoint, -one_x, w)
    ifelse(x == zero(x), -Inf * one_x, w)
end

end # module ReactiveHMCRKEFunctionalLowering
