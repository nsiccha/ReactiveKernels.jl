# JAX translations of Statistics' `mean`, `var` and `std` for `jax_source`
# (`src/jax.jl`). Statistics' own methods branch (on an empty argument, on
# `dims`), so the generic route refuses their lowered code. Each translation
# here is composed from the core translations of `sum`, `length`, `size` and
# broadcasting: a padded (ragged) value reduces over its valid entries, and a
# host (`prepare_data`) call takes NumPy's spelling, exactly as those do.
#
#     mean(x)                 sum(x) / length(x)
#     mean(f, x)              sum(f, x) / length(x)
#     mean(x; dims = d)       sum(x; dims = d) ./ size(x, d)
#     var(x; corrected, mean) sum(abs2.(x .- m)) / max(n - corrected, 0)
#     std(x; ...)             sqrt(var(x; ...))
#
# `max(n - corrected, 0)` keeps Statistics' NaN for an empty argument (and for
# one entry, corrected) instead of `0 / -1`.
module ReactiveKernelsStatisticsExt

using Statistics: Statistics
using ReactiveKernels: _JaxContext, _PyVal, _jax_error, _jax_where, _jax_is_array, _jax_known,
    _jax_is_colon, _jax_literal, _jax_broadcast, _jax_function_of, _jax_call_positional
import ReactiveKernels: _jax_function_call, _jax_call

const _Reduction = Union{typeof(Statistics.mean),typeof(Statistics.var),typeof(Statistics.std)}

# A value these reductions read whole: an array or a numeric tuple. `mean` of
# a number is the number, as a float.
_reducible(x::_PyVal) = _jax_is_array(x.T) || x.T isa Type && x.T <: Tuple
function _reduced_argument(ctx::_JaxContext, f, x::_PyVal)
    _reducible(x) || _jax_error("`", f, "` of a `", x.T, "` has no JAX translation",
                                _jax_where(ctx))
    x
end

function _jax_function_call(ctx::_JaxContext, ::typeof(Statistics.mean), args::Vector{_PyVal})
    length(args) in (1, 2) ||
        _jax_error("`mean` with ", length(args), " arguments has no JAX translation",
                   _jax_where(ctx))
    x = args[end]
    length(args) == 1 && x.T isa Type && x.T <: Number &&
        return _jax_call(ctx, /, [x, _jax_literal(1)])
    _reduced_argument(ctx, Statistics.mean, x)
    _jax_call(ctx, /, [_jax_call(ctx, sum, args), _jax_call(ctx, length, [x])])
end
_jax_function_call(ctx::_JaxContext, f::Union{typeof(Statistics.var),typeof(Statistics.std)},
                   args::Vector{_PyVal}) = _jax_spread(ctx, f, args, nothing, true, nothing)

function _jax_call(ctx::_JaxContext, f::_Reduction, args::Vector{_PyVal}, kw)
    isempty(kw) && return _jax_call_positional(ctx, f, args)
    names = first.(kw)
    allowed = f === Statistics.mean ? (:dims,) : (:dims, :corrected, :mean)
    unknown = filter(k -> !(k in allowed), names)
    isempty(unknown) && allunique(names) ||
        _jax_error("keyword arguments ", join(unknown, ", "), " to `", f,
                   "` have no JAX translation", _jax_where(ctx))
    keyword(k) = (i = findfirst(==(k), names); i === nothing ? nothing : last(kw[i]))
    dims = keyword(:dims)
    dims === nothing || _jax_is_colon(dims) || _jax_known(ctx, dims) && dims.T isa Type &&
        dims.T <: Integer ||
        _jax_error("`", f, "` over `dims` that are not one integer known while JAX traces",
                   _jax_where(ctx))
    dims !== nothing && _jax_is_colon(dims) && (dims = nothing)
    f === Statistics.mean && return _jax_mean(ctx, args, dims)
    corrected = keyword(:corrected)
    corrected === nothing || corrected.static && corrected.code in ("True", "False") ||
        _jax_error("`", f, "` with a `corrected` that is not a literal `true` or `false`",
                   _jax_where(ctx))
    _jax_spread(ctx, f, args, dims, corrected === nothing || corrected.code == "True",
                keyword(:mean))
end

# `mean(x; dims = d)` and `mean(f, x; dims = d)`.
function _jax_mean(ctx::_JaxContext, args::Vector{_PyVal}, dims)
    dims === nothing && return _jax_call_positional(ctx, Statistics.mean, args)
    length(args) in (1, 2) ||
        _jax_error("`mean` with ", length(args), " arguments has no JAX translation",
                   _jax_where(ctx))
    x = args[end]
    _jax_is_array(x.T) ||
        _jax_error("`mean` over `dims` of a `", x.T, "` has no JAX translation", _jax_where(ctx))
    length(args) == 2 && (x = _jax_broadcast(ctx, _jax_function_of(args[1]), [x]))
    total = _jax_call(ctx, sum, [x], Pair{Symbol,_PyVal}[:dims => dims])
    _jax_call(ctx, /, [total, _jax_call(ctx, size, [x, dims])])
end

# `var` and `std`: the mean of squared deviations from `m` (the mean unless
# given), divided by the count less one when `corrected`.
function _jax_spread(ctx::_JaxContext, f, args::Vector{_PyVal}, dims, corrected::Bool, m)
    length(args) == 1 ||
        _jax_error("`", f, "` with ", length(args), " arguments has no JAX translation",
                   _jax_where(ctx))
    x = _reduced_argument(ctx, f, only(args))
    dims === nothing || _jax_is_array(x.T) ||
        _jax_error("`", f, "` over `dims` of a `", x.T, "` has no JAX translation",
                   _jax_where(ctx))
    m === nothing && (m = _jax_mean(ctx, [x], dims))
    deviations = _jax_broadcast(ctx, abs2, [_jax_broadcast(ctx, -, [x, m])])
    total = dims === nothing ? _jax_call(ctx, sum, [deviations]) :
        _jax_call(ctx, sum, [deviations], Pair{Symbol,_PyVal}[:dims => dims])
    n = dims === nothing ? _jax_call(ctx, length, [x]) : _jax_call(ctx, size, [x, dims])
    corrected && (n = _jax_call(ctx, -, [n, _jax_literal(1)]))
    v = _jax_call(ctx, /, [total, _jax_call(ctx, max, [n, _jax_literal(0)])])
    f === Statistics.std || return v
    _jax_is_array(v.T) ? _jax_broadcast(ctx, sqrt, [v]) : _jax_call(ctx, sqrt, [v])
end

end
