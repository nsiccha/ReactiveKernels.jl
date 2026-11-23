# SLEEFPirates.log vectorizes in a summed loop. Its negative-input behavior
# differs from Base.log, and a custom exp can return a negative Float64.
# Evaluate each exp and + at its original position, carrying whether the
# leaves all used Float64's exponential. That bit is constant for concrete
# input types. A derivative cut uses Base's logarithm and its ordinary partial.
@inline _exp_with_sign_proof(f::F, x) where {F} = (f(x), false)
@inline _exp_with_sign_proof(f::F, x::Float64) where {F} = ((@inline f(x)), true)
@inline _add_with_sign_proof(a, b) = (a[1] + b[1], a[2] & b[2])

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

function _exp_sum_with_sign_proof(ex)
    Meta.isexpr(ex, :call) || return nothing
    f = _vector_log_callee(ex.args[1])
    if _rw_is_exp(f) && length(ex.args) == 2
        return Expr(:call, _rw_helper(:_exp_with_sign_proof), ex.args...)
    end
    f === (+) && length(ex.args) == 3 || return nothing
    left = _exp_sum_with_sign_proof(ex.args[2])
    left === nothing && return nothing
    right = _exp_sum_with_sign_proof(ex.args[3])
    right === nothing && return nothing
    Expr(:call, _rw_helper(:_add_with_sign_proof), left, right)
end

# A source softplus may still be present if its earlier pass rejected the
# cell. Only its already selected vector twin is safe beside a vector log.
_vector_log_safe(f) = f === _log_exp_sum || f === _add_with_sign_proof ||
    f === _exp_with_sign_proof ||
    (f !== rk_log1pexp && f !== LogExpFunctions.log1pexp && _vector_safe_function(f))

function _vectorize_cell_logs(body)
    candidate = _index_run_replace(body, _RW_OPAQUE) do ex
        Meta.isexpr(ex, :call, 2) && _vector_log_callee(ex.args[1]) === log || return ex
        sum = ex.args[2]
        Meta.isexpr(sum, :call, 3) && _vector_log_callee(sum.args[1]) === (+) || return ex
        proven = _exp_sum_with_sign_proof(sum)
        proven === nothing ? ex : Expr(:call, _rw_helper(:_log_exp_sum), proven)
    end
    candidate === body && return body
    _vector_safe_cell(candidate, Returns(nothing), _vector_log_safe) && _vector_doubles() >= 2 ? candidate : body
end
