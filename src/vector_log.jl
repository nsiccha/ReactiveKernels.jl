# SLEEFPirates.log vectorizes in a summed loop. Its negative-input behavior
# differs from Base.log, and a custom exp can return a negative Float64.
# Evaluate each exp, each + and each literal factor at its original position,
# carrying whether the leaves all used Float64's exponential. That bit is
# constant for concrete input types. A derivative cut uses Base's logarithm
# and its ordinary partial.
@inline _exp_with_sign_proof(f::F, x) where {F} = (f(x), false)
@inline _exp_with_sign_proof(f::F, x::Float64) where {F} = ((@inline f(x)), true)
@inline _add_with_sign_proof(a, b) = (a[1] + b[1], a[2] & b[2])
# A sum written as one call of three or more terms stays that one call.
@inline _add_with_sign_proof(a, b, c, rest...) =
    (+(a[1], b[1], c[1], map(first, rest)...), a[2] & b[2] & c[2] & all(map(last, rest)))
# A nonnegative literal factor keeps the sign of the sum it scales. The
# written multiplication `f` is called with the factor on its written side.
@inline _scale_with_sign_proof(f::F, c, a) where {F} = (f(c, a[1]), a[2])
@inline _scaled_with_sign_proof(f::F, a, c) where {F} = (f(a[1], c), a[2])

@_record_source @inline _vector_log_value(x::Float64) = SLEEFPirates.log(x)
@kernel _vector_log_graph(x::Float64) = begin
    value::Float64 = _vector_log_value(x)
    (value, partial::Float64) = (log(x), inv(x))
    return value, partial
end
const _vector_log = scalar_derivative_rule(_vector_log_graph;
    primal=:value, partials=(x=:partial,), name=:_vector_log)

@inline _log_exp_sum(t) = log(t[1])
@inline _log_exp_sum(t::Tuple{Float64,Bool}) = t[2] ? _vector_log(t[1]) : log(t[1])

# Match resolved constant bindings only. In particular a mutable global
# holding exp at preparation can name an entirely different function later.
_vector_log_callee(f::GlobalRef) = _rw_binding(f.mod, f.name)
_vector_log_callee(f::Function) = f
_vector_log_callee(f) = nothing

# A literal factor that cannot turn a nonnegative sum negative: a real number
# that is neither negative nor a negative zero (NaN stays NaN in the product).
_nonnegative_literal(c) = c isa Real && !(c isa Bool) && !signbit(c) && !(c < 0)
_vector_log_product(f) = f === (*) || f === _native_mul

# The argument of a logarithm as a sum of exponentials: an exponential, such a
# sum scaled by a nonnegative literal on either side, or one `+` call of two
# or more such terms.
function _exp_sum_with_sign_proof(ex)
    Meta.isexpr(ex, :call) || return nothing
    f = _vector_log_callee(ex.args[1])
    if _rw_is_exp(f) && length(ex.args) == 2
        return Expr(:call, _rw_helper(:_exp_with_sign_proof), ex.args...)
    end
    if _vector_log_product(f) && length(ex.args) == 3
        a, b = ex.args[2], ex.args[3]
        if _nonnegative_literal(a)
            inner = _exp_sum_with_sign_proof(b)
            return inner === nothing ? nothing :
                Expr(:call, _rw_helper(:_scale_with_sign_proof), ex.args[1], a, inner)
        elseif _nonnegative_literal(b)
            inner = _exp_sum_with_sign_proof(a)
            return inner === nothing ? nothing :
                Expr(:call, _rw_helper(:_scaled_with_sign_proof), ex.args[1], inner, b)
        end
        return nothing
    end
    f === (+) && length(ex.args) >= 3 || return nothing
    terms = Any[]
    for arg in @view ex.args[2:end]
        term = _exp_sum_with_sign_proof(arg)
        term === nothing && return nothing
        push!(terms, term)
    end
    Expr(:call, _rw_helper(:_add_with_sign_proof), terms...)
end

# A source softplus may still be present if its earlier pass rejected the
# cell. Only its already selected vector twin is safe beside a vector log.
_vector_log_safe(f) = f === _log_exp_sum || f === _add_with_sign_proof ||
    f === _exp_with_sign_proof || f === _scale_with_sign_proof || f === _scaled_with_sign_proof ||
    (f !== rk_log1pexp && f !== LogExpFunctions.log1pexp && _vector_safe_function(f))

function _vectorize_cell_logs(body)
    candidate = _index_run_replace(body, _RW_OPAQUE) do ex
        Meta.isexpr(ex, :call, 2) && _vector_log_callee(ex.args[1]) === log || return ex
        sum = ex.args[2]
        # A lone exponential is left to the rewrites that cancel it.
        Meta.isexpr(sum, :call) && length(sum.args) >= 3 || return ex
        proven = _exp_sum_with_sign_proof(sum)
        proven === nothing ? ex : Expr(:call, _rw_helper(:_log_exp_sum), proven)
    end
    candidate === body && return body
    _vector_safe_cell(candidate, Returns(nothing), _vector_log_safe) && _vector_doubles() >= 2 ? candidate : body
end
