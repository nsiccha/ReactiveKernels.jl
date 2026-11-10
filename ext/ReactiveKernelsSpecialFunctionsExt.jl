# JAX translations of SpecialFunctions functions for `jax_source`
# (`src/jax.jl`). Each is the `jax.scipy.special` counterpart, or a stable
# helper the emitted module defines.
module ReactiveKernelsSpecialFunctionsExt

using ReactiveKernels
using ReactiveKernels: _JaxContext, _PyVal, _jax_result, _jax_args
import ReactiveKernels: _jax_elementwise, _jax_elementwise_call, _jax_function_call
import SpecialFunctions

const _HELPERS = Dict{String,String}(
    "_rk_trigamma" => """
def _rk_trigamma(x):
    # SpecialFunctions.trigamma.
    return jsp.polygamma(1, x)
""",
    # `jax.scipy.special.betaln` (JAX 0.12) is accurate to about 1e-6 in float64
    # once an argument reaches 8: its Stirling correction is single-precision.
    "_rk_logbeta" => """
def _rk_lgamma_correction(x):
    # log(gamma(x)) - ((x - 0.5) * log(x) - x + 0.5 * log(2 pi)), for x >= 10:
    # the Stirling series through x^-15.
    z = 1.0 / (x * x)
    return (1.0 / x) * (1.0 / 12 + z * (-1.0 / 360 + z * (1.0 / 1260 + z * (-1.0 / 1680
        + z * (1.0 / 1188 + z * (-691.0 / 360360 + z * (1.0 / 156 + z * (-3617.0 / 122400))))))))


def _rk_logbeta(a, b):
    # SpecialFunctions.logbeta for positive arguments. With b the larger, a
    # large b subtracts log(gamma(b)) - log(gamma(a + b)) in Stirling form,
    # without cancellation.
    a, b = jnp.minimum(a, b), jnp.maximum(a, b)
    small = jsp.gammaln(a) + jsp.gammaln(b) - jsp.gammaln(a + b)
    big = jnp.maximum(b, 10.0)
    large = (jsp.gammaln(a) + _rk_lgamma_correction(big) - _rk_lgamma_correction(a + big)
             - (a + big - 0.5) * jnp.log1p(a / big) - a * jnp.log(big) + a)
    return jnp.where(b < 10, small, large)
""",
    "_rk_logerfc" => """
def _rk_logerfc(x):
    # log(erfc(x)) without overflow or cancellation (SpecialFunctions.logerfc):
    # log(erfcx(x)) - x^2 for x > 0.
    positive = x > 0
    xp = jnp.where(positive, x, 1.0)
    xn = jnp.where(positive, 0.0, x)
    return jnp.where(positive, jnp.log(jsp.erfcx(xp)) - xp * xp, jnp.log(jsp.erfc(xn)))
""",
    "_rk_logerfcx" => """
def _rk_logerfcx(x):
    # log(erfcx(x)) (SpecialFunctions.logerfcx): log(erfc(x)) + x^2 for x < 0.
    negative = x < 0
    xn = jnp.where(negative, x, -1.0)
    xp = jnp.where(negative, 0.0, x)
    return jnp.where(negative, jnp.log(jsp.erfc(xn)) + xn * xn, jnp.log(jsp.erfcx(xp)))
""",
)

_jax_elementwise(::typeof(SpecialFunctions.loggamma)) = "jsp.gammaln"
_jax_elementwise(::typeof(SpecialFunctions.gamma)) = "jsp.gamma"
_jax_elementwise(::typeof(SpecialFunctions.digamma)) = "jsp.digamma"
_jax_elementwise(::typeof(SpecialFunctions.trigamma)) = "_rk_trigamma" => _HELPERS["_rk_trigamma"]
_jax_elementwise(::typeof(SpecialFunctions.logbeta)) = "_rk_logbeta" => _HELPERS["_rk_logbeta"]
_jax_elementwise(::typeof(SpecialFunctions.erf)) = "jsp.erf"
_jax_elementwise(::typeof(SpecialFunctions.erfc)) = "jsp.erfc"
_jax_elementwise(::typeof(SpecialFunctions.erfcx)) = "jsp.erfcx"
_jax_elementwise(::typeof(SpecialFunctions.erfinv)) = "jsp.erfinv"
_jax_elementwise(::typeof(SpecialFunctions.logerfc)) = "_rk_logerfc" => _HELPERS["_rk_logerfc"]
_jax_elementwise(::typeof(SpecialFunctions.logerfcx)) = "_rk_logerfcx" => _HELPERS["_rk_logerfcx"]

# Modified Bessel functions of the first kind, `besseli(nu, x)`, and their
# exponentially scaled forms, `besselix(nu, x) = exp(-|x|) besseli(nu, x)`.
# A literal order 0 or 1 is `jax.scipy.special`'s own function. Any other
# order, a value of the data or the program included (the harmonic of a
# periodic Hilbert-space GP basis), is the module's `_rk_besselix`: JAX 0.12
# has no general-order function, and TensorFlow Probability's `bessel_ive`
# (0.25.0) does not import under it. Host code (`prepare_data`) and values
# known while emitting use SciPy's `ive` and `iv`.

# Debye's polynomials U_k(p) of the uniform expansion of I in large order
# (DLMF 10.41.10), exact from their recurrence U_{k+1}(p) = p^2 (1 - p^2)
# U_k'(p) / 2 + (1/8) int_0^p (1 - 5 t^2) U_k(t) dt; coefficient j + 1 of
# U_k is that of p^j.
function _debye_polynomials(n::Int)
    U = [Rational{BigInt}[1]]
    for _ in 1:n
        u = U[end]
        next = zeros(Rational{BigInt}, length(u) + 3)
        for j in 1:length(u)-1      # the derivative's p^(j-1) term, times p^2 (1 - p^2) / 2
            next[j+2] += j * u[j+1] / 2
            next[j+4] -= j * u[j+1] / 2
        end
        for j in 0:length(u)-1      # (1 - 5 t^2) t^j integrated from 0 to p, over 8
            next[j+2] += u[j+1] / (8 * (j + 1))
            next[j+4] -= 5 * u[j+1] / (8 * (j + 3))
        end
        while iszero(next[end])
            pop!(next)
        end
        push!(U, next)
    end
    U
end
_python_float_tuple(u) = "(" * join((repr(Float64(c)) for c in u), ", ") * (length(u) == 1 ? ",)" : ")")

# The expansion runs from order 25 through U_12, the series below that order
# up to x = 30 with 64 terms: relative error at most about 4e-13 against
# SciPy over orders 0 to 1000 and x from 1e-8 to 1e7.
_HELPERS["_rk_besselix"] = """
_RK_DEBYE_U = (
""" * join(("    " * _python_float_tuple(u) * "," for u in _debye_polynomials(12)), "\n") * """

)


def _rk_besselix_series(nu, x):
    # exp(-x) I_nu(x) for 0 <= x <= 30 and 0 <= nu < 25, by its power series
    # sum_k (x/2)^(nu + 2k) / (k! gamma(nu + k + 1)) through k = 63.
    k = jnp.arange(1, 64, dtype=jnp.float64)
    q = 0.25 * x[..., None] * x[..., None] / (k * (nu[..., None] + k))
    s = 1.0 + jnp.sum(jnp.cumprod(q, axis=-1), axis=-1)
    return jnp.power(0.5 * x, nu) * jnp.exp(-x - jsp.gammaln(nu + 1.0)) * s


def _rk_besselix_debye(nu, x):
    # exp(-x) I_nu(x) for nu >= 25 and x > 0, by the uniform expansion in large
    # order through U_12 (DLMF 10.41.3), with p = nu / sqrt(nu^2 + x^2).
    r = jnp.sqrt(nu * nu + x * x)
    p = nu / r
    t = x / nu
    s = jnp.zeros_like(p)
    for u in reversed(_RK_DEBYE_U):
        c = jnp.zeros_like(p)
        for a in reversed(u):
            c = c * p + a
        s = s / nu + c
    # nu (eta - x / nu) = nu / (sqrt(1 + t^2) + t) - nu asinh(1 / t), t = x / nu.
    return jnp.exp(nu / (jnp.sqrt(1.0 + t * t) + t) - nu * jnp.arcsinh(1.0 / t)) / jnp.sqrt(2.0 * jnp.pi * r) * s


def _rk_besselix_shifted(nu, x):
    # exp(-x) I_nu(x) for 0 <= nu < 25 and x > 30: the expansion at orders
    # mu = nu + m >= 25 and mu + 1, then m steps of the downward recurrence
    # I_(v-1) = (2v / x) I_v + I_(v+1), which is stable for I.
    m = jnp.ceil(25.0 - nu)
    mu = nu + m
    upper, lower = _rk_besselix_debye(mu + 1.0, x), _rk_besselix_debye(mu, x)

    def step(i, carry):
        upper, lower = carry
        taken = i < m
        below = 2.0 * (mu - i) / x * lower + upper
        return jnp.where(taken, lower, upper), jnp.where(taken, below, lower)

    return jax.lax.fori_loop(0, 25, step, (upper, lower))[1]


def _rk_besselix_nonnegative(nu, x):
    # exp(-x) I_nu(x) for nu >= 0 and x >= 0. Every branch is evaluated on
    # arguments valid for it and the value selected.
    series = (x <= 30.0) & (nu < 25.0)
    large = nu >= 25.0
    shifted = ~series & ~large
    return jnp.where(series, _rk_besselix_series(jnp.where(series, nu, 0.0), jnp.where(series, x, 1.0)),
                     jnp.where(large, _rk_besselix_debye(jnp.where(large, nu, 25.0), jnp.where(large, x, 1.0)),
                               _rk_besselix_shifted(jnp.where(shifted, nu, 0.0), jnp.where(shifted, x, 31.0))))


@jax.custom_jvp
def _rk_besselix(nu, x):
    # SpecialFunctions.besselix(nu, x) = exp(-|x|) I_nu(x), elementwise. An
    # integer order may be negative (I_-n = I_n) and so may x
    # (I_n(-x) = (-1)^n I_n(x)); a negative non-integer order is NaN here, and
    # so is a negative x with a non-integer order, where Julia throws.
    nu, x = jnp.broadcast_arrays(jnp.asarray(nu, dtype=jnp.float64), jnp.asarray(x, dtype=jnp.float64))
    integer = nu == jnp.round(nu)
    order = jnp.where(integer, jnp.abs(nu), nu)
    value = _rk_besselix_nonnegative(jnp.maximum(order, 0.0), jnp.abs(x))
    odd = integer & (jnp.mod(order, 2.0) == 1.0)
    sign = jnp.where(x < 0, jnp.where(integer, jnp.where(odd, -1.0, 1.0), jnp.nan), 1.0)
    return jnp.where(order < 0, jnp.nan, sign * value)


def _rk_besselix_jvp(primals, tangents):
    # d/dx exp(-|x|) I_nu(x) = exp(-|x|) (I_nu'(x) - sign(x) I_nu(x)), with
    # I_nu' = (I_(nu-1) + I_(nu+1)) / 2, or I_(nu+1) + (nu / x) I_nu for a
    # non-integer order below 1. The order has no derivative: one is NaN, as
    # Julia's derivative rule (DiffRules) gives.
    nu, x = primals
    dnu, dx = tangents
    nu, x = jnp.broadcast_arrays(jnp.asarray(nu, dtype=jnp.float64), jnp.asarray(x, dtype=jnp.float64))
    y = _rk_besselix(nu, x)
    out = jnp.zeros_like(y)
    if not isinstance(dx, jax.custom_derivatives.SymbolicZero):
        recurrence = (nu == jnp.round(nu)) | (jnp.abs(nu) >= 1.0)
        nr = jnp.where(recurrence, nu, 1.5)
        ns = jnp.where(recurrence, 0.5, nu)
        xs = jnp.where(recurrence | (x == 0), 1.0, x)
        derivative = jnp.where(recurrence, 0.5 * (_rk_besselix(nr - 1.0, x) + _rk_besselix(nr + 1.0, x)),
                               jnp.where(x == 0, jnp.inf, _rk_besselix(ns + 1.0, x) + ns / xs * _rk_besselix(ns, x)))
        out = out + (derivative - jnp.sign(x) * y) * dx
    if not isinstance(dnu, jax.custom_derivatives.SymbolicZero):
        out = out + jnp.nan * dnu
    return y, out


_rk_besselix.defjvp(_rk_besselix_jvp, symbolic_zeros=True)
"""
_HELPERS["_rk_besseli"] = """
def _rk_besseli(nu, x):
    # SpecialFunctions.besseli(nu, x) = exp(|x|) besselix(nu, x), elementwise.
    return jnp.exp(jnp.abs(x)) * _rk_besselix(nu, x)
"""

for (f, literal, host, helper) in
        ((SpecialFunctions.besseli, ("jsp.i0", "jsp.i1"), "ssp.iv", "_rk_besseli"),
         (SpecialFunctions.besselix, ("jsp.i0e", "jsp.i1e"), "ssp.ive", "_rk_besselix"))
    @eval function _jax_elementwise_call(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        length(args) == 2 || ReactiveKernels._jax_error("`", $f, "` with ", length(args),
                                                        " arguments")
        order, x = args
        order.code in ("0", "1") &&
            return _jax_result($f, args, $literal[parse(Int, order.code) + 1] * "(" * x.code * ")")
        # Host code, or values known while emitting: SciPy, whatever the order.
        if ctx.host !== nothing || all(a -> a.static, args)
            return _jax_result($f, args, $host * "(" * _jax_args(args) * ")")
        end
        ReactiveKernels._jax_helper!(ctx, "_rk_besselix", _HELPERS["_rk_besselix"])
        $helper == "_rk_besselix" || ReactiveKernels._jax_helper!(ctx, $helper, _HELPERS[$helper])
        _jax_result($f, args, $helper * "(" * _jax_args(args) * ")")
    end
end

# The regularized incomplete beta and gamma functions return their complement too.
function _jax_function_call(ctx::_JaxContext, ::typeof(SpecialFunctions.beta_inc),
                            args::Vector{_PyVal})
    length(args) == 3 || ReactiveKernels._jax_error("`beta_inc` with ", length(args), " arguments")
    p = "jsp.betainc(" * _jax_args(args) * ")"
    _jax_result(SpecialFunctions.beta_inc, args, "(" * p * ", 1.0 - " * p * ")")
end
function _jax_function_call(ctx::_JaxContext, ::typeof(SpecialFunctions.gamma_inc),
                            args::Vector{_PyVal})
    length(args) in (2, 3) || ReactiveKernels._jax_error("`gamma_inc` with ", length(args), " arguments")
    a = _jax_args(args[1:2])
    _jax_result(SpecialFunctions.gamma_inc, args,
                "(jsp.gammainc(" * a * "), jsp.gammaincc(" * a * "))")
end

end # module
