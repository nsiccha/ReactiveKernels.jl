# JAX translations of SpecialFunctions functions for `jax_source`
# (`src/jax.jl`). Each is the `jax.scipy.special` counterpart, or a stable
# helper the emitted module defines.
module ReactiveKernelsSpecialFunctionsExt

using ReactiveKernels
using ReactiveKernels: _JaxContext, _PyVal, _jax_result, _jax_args
import ReactiveKernels: _jax_elementwise, _jax_function_call
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
