# Array literals and matrices in `js_source` modules.
#
# A vector literal (`[a, b]`, `Float64[a, b]`) is a JavaScript array literal.
# A matrix is an array of row arrays: `m[i][j]` holds Julia's `m[i+1, j+1]`, as
# an input, an output and inside the module (decision `1r7ufgm`, prong
# `matrix`). Rows may be any array-like on input (`Float64Array`s under
# `js_run`'s `vectors = :float64array`). A matrix without rows keeps its
# column count in a `cols` property of the outer array, as
# `Object.assign([], { cols: 3 })` for Julia's `zeros(0, 3)`; a bare `[]` is
# 0×0.
#
# `_js_function_call` reaches these through `js_calls.jl`, which asks each
# family of array translations in turn, then the general table (`js.jl`).

_js_ismatrix(v) = (T = _js_type(v); T isa Type && T <: Matrix && eltype(T) <: _JsMatrixElement)
_js_iscolon(v) = v isa _JsVal && v.static && v.value isa Colon
_js_static_int(v) = v isa _JsVal && v.static && v.value isa Integer ? Int(v.value) : nothing

# The runtime helpers (`_rk_matrix`, `_rk_ncols`, `_rk_at2`, `_rk_col`,
# `_rk_row`, `_rk_mcopy`) are entries of `_JS_RUNTIME` in `js.jl`.

function _js_input_check(mod::_JsModule, name, ::Type{Matrix{E}}) where {E}
    E <: _JsMatrixElement || throw(JSEmissionError(
        "the input `$name` has type `Matrix{$E}`; JavaScript matrices hold Float64, Int64 or Bool"))
    el = E === Bool ? "boolean" : "number"
    "_rk_matrix(" * _js_string_literal(name) * ", " * name * ", \"" * el * "\"" *
        (E === Int64 ? ", true" : ", false") * ");", :_rk_matrix
end

# The element of a literal of element type `E`: Julia converts a `Bool` to `0`
# or `1` in a numeric array.
function _js_literal_element(mod, a, E)
    S = _js_type(a)
    if E === Bool
        S === Bool || return nothing
        return _js_runtime(mod, a).code
    end
    S === Bool && return "(" * _js_code(mod, a, _JS_COND + 1) * " ? 1 : 0)"
    S <: _JsNumber || return nothing
    E === Int64 && S === Float64 && return _js_use!(mod, :_rk_int) * "(" * _js_runtime(mod, a).code * ")"
    _js_runtime(mod, a).code
end

function _js_array_literal(ir, items, T)
    T isa Type && T <: Vector && eltype(T) <: _JsMatrixElement || return nothing
    mod = ir.fn.mod
    codes = String[]
    for a in items
        a isa _JsVal || return nothing
        c = _js_literal_element(mod, a, eltype(T))
        c === nothing && return nothing
        push!(codes, c)
    end
    _js_rt("[" * join(codes, ", ") * "]", T; pure = false)
end

function _js_array_call(ir, f, args, T)
    mod = ir.fn.mod
    if f === Base.vect
        return _js_array_literal(ir, args, T)
    elseif f === getindex && !isempty(args) && args[1] isa _JsVal && args[1].static && args[1].value isa Type
        # `Float64[a, b]`, a literal of a given element type
        return _js_array_literal(ir, args[2:end], T)
    end
    isempty(args) || _js_ismatrix(args[1]) || return nothing
    m = args[1]
    if f === getindex && length(args) == 3
        i, j = args[2], args[3]
        _js_isint(i) && _js_isint(j) && return _js_fcall(ir, _js_use!(mod, :_rk_at2), Any[m, i, j], T)
        _js_iscolon(i) && _js_isint(j) && return _js_fcall(ir, _js_use!(mod, :_rk_col), Any[m, j], T; pure = false)
        _js_isint(i) && _js_iscolon(j) && return _js_fcall(ir, _js_use!(mod, :_rk_row), Any[m, i], T; pure = false)
    elseif f === size && length(args) == 1
        x = _js_simple!(ir, m, "m")
        return _JsTuple(Any[_js_rt(x.code * ".length", Int64), _js_fcall(ir, _js_use!(mod, :_rk_ncols), Any[x], Int64)],
                        Tuple{Int64,Int64})
    elseif f in (size, lastindex) && length(args) == 2
        d = _js_static_int(args[2])
        d == 1 && return _js_rt(_js_code(mod, m, _JS_ATOM) * ".length", Int64)
        d == 2 && return _js_fcall(ir, _js_use!(mod, :_rk_ncols), Any[m], Int64)
        d !== nothing && d > 2 && return _js_static(1)
    elseif f === firstindex && length(args) == 2 && _js_static_int(args[2]) !== nothing
        return _js_static(1)
    elseif f === axes && length(args) == 2
        d = _js_static_int(args[2])
        d == 1 && return _js_rt(_js_use!(mod, :_rk_range) * "(1, " * _js_code(mod, m, _JS_ATOM) * ".length)", T)
        d == 2 && return _js_rt(_js_use!(mod, :_rk_range) * "(1, " * _js_use!(mod, :_rk_ncols) * "(" *
                                _js_runtime(mod, m).code * "))", T)
    elseif f === axes && length(args) == 1
        x = _js_simple!(ir, m, "m")
        return _JsTuple(Any[_js_rt(_js_use!(mod, :_rk_range) * "(1, " * x.code * ".length)", UnitRange{Int64}),
                            _js_rt(_js_use!(mod, :_rk_range) * "(1, " * _js_use!(mod, :_rk_ncols) * "(" * x.code * "))",
                                   UnitRange{Int64})], Tuple{UnitRange{Int64},UnitRange{Int64}})
    elseif f in (length, lastindex) && length(args) == 1
        x = _js_simple!(ir, m, "m")
        return _js_rt(x.code * ".length * " * _js_use!(mod, :_rk_ncols) * "(" * x.code * ")", Int64; prec = _JS_MUL)
    elseif f === copy && length(args) == 1
        return _js_fcall(ir, _js_use!(mod, :_rk_mcopy), Any[m], T; pure = false)
    end
    nothing
end
