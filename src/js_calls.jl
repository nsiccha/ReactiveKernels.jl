# The array translations of Base functions, gathered: each family answers the
# calls it translates and `nothing` for every other, and a function several
# families translate (`sum` of a generator and of a matrix) has this one method,
# which asks each family in turn and then the general table (`js.jl`).
#
#   `_js_array_call`  (`js_arrays.jl`):    literals and matrix reads
#   `_js_iter_call`   (`js_iterators.jl`): generators, filters and their reductions
#   `_js_linalg_call` (`js_linalg.jl`):    matrix arithmetic

const _JS_FAMILIES = (_js_array_call, _js_iter_call, _js_linalg_call)
const _JS_FAMILY_FUNCTIONS = (
    Base.vect, getindex, size, length, axes, firstindex, lastindex, copy,
    collect, sum, maximum, minimum, count, any, all, isempty, argmin, argmax, findmin, findmax, filter,
    Base.Iterators.filter, Core.kwcall,
    vec, transpose, adjoint, permutedims, *, /, +, -, prod, zeros, ones, fill)

for F in _JS_FAMILY_FUNCTIONS
    @eval function _js_function_call(ir::_JsIR, f::typeof($F), args, T)
        for family in _JS_FAMILIES
            v = family(ir, f, args, T)
            v === nothing || return v
        end
        invoke(_js_function_call, Tuple{_JsIR,Function,Any,Any}, ir, f, args, T)
    end
end
