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
# `_rk_row`, `_rk_mcopy`) are entries of `_JS_RUNTIME` in `js.jl`; those that
# write into a matrix are here. A write converts the value to the matrix's
# element type, as Julia's `setindex!` does, and checks its position (and a
# column's or row's length) as Julia does.
merge!(_JS_RUNTIME, Dict{Symbol,Tuple{String,Vector{Symbol}}}(
    :_rk_set2 => ("""
function _rk_set2(m, v, i, j) {
  const cols = _rk_ncols(m);
  if (!(Number.isInteger(i) && i >= 1 && i <= m.length && Number.isInteger(j) && j >= 1 && j <= cols))
    throw new JuliaError("BoundsError", "attempt to access " + m.length + "×" + cols + " array at index [" + i + ", " + j + "]");
  m[i - 1][j - 1] = v;
  return m;
}""", [:JuliaError, :_rk_ncols]),
    :_rk_setcol => ("""
function _rk_setcol(m, v, j) {
  const cols = _rk_ncols(m);
  if (!(Number.isInteger(j) && j >= 1 && j <= cols))
    throw new JuliaError("BoundsError", "attempt to access " + m.length + "×" + cols + " array at index [:, " + j + "]");
  if (v.length !== m.length)
    throw new JuliaError("DimensionMismatch", "tried to assign " + v.length + " elements to " + m.length + " destinations");
  for (let i = 0; i < m.length; i++) m[i][j - 1] = v[i];
  return m;
}""", [:JuliaError, :_rk_ncols]),
    :_rk_throw => ("""
function _rk_throw(e) { throw e; }""", Symbol[]),
    :_rk_mslice => ("""
function _rk_mslice(m, i, j) {
  const cols = _rk_ncols(m);
  if (i === null ? !(Number.isInteger(j) && j >= 1 && j <= cols) : !(Number.isInteger(i) && i >= 1 && i <= m.length))
    throw new JuliaError("BoundsError", "attempt to access " + m.length + "×" + cols + " array at index [" +
      (i === null ? ":" : i) + ", " + (j === null ? ":" : j) + "]");
}""", [:JuliaError, :_rk_ncols]),
    :_rk_setrow => ("""
function _rk_setrow(m, v, i) {
  const cols = _rk_ncols(m);
  if (!(Number.isInteger(i) && i >= 1 && i <= m.length))
    throw new JuliaError("BoundsError", "attempt to access " + m.length + "×" + cols + " array at index [" + i + ", :]");
  if (v.length !== cols)
    throw new JuliaError("DimensionMismatch", "tried to assign " + v.length + " elements to " + cols + " destinations");
  const r = m[i - 1];
  for (let j = 0; j < cols; j++) r[j] = v[j];
  return m;
}""", [:JuliaError, :_rk_ncols]),
))

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

# A matrix literal of numbers, `[a b; c d]` (`hvcat`, in a kernel's recipe RK's
# `_native_hvcat`), `[a b]` (`hcat`, `_native_hcat`) or `Float64[a b; c d]`
# (`typed_hvcat`, `typed_hcat`): a fresh row-array matrix, the operands filling
# it row by row. The layout, `lead` (the block-row counts, the element type),
# is known while translating, and so are the operands' types, so Julia's own
# concatenation of zeros of those types gives the shape and element type here,
# or the error a layout that does not fit raises, thrown where the literal is.
function _js_matrix_literal(ir, f, lead, items, T)
    all(a -> a isa _JsVal && a.static, lead) || return nothing
    all(a -> a isa _JsVal && _js_type(a) <: Union{_JsNumber,Bool}, items) || return nothing
    mod = ir.fn.mod
    standin = try
        f((a.value for a in lead)..., (zero(_js_type(a)) for a in items)...)
    catch err
        err isa Exception || rethrow()
        return _js_rt(_js_use!(mod, :_rk_throw) * "(" * _js_exception(mod, err) * ")", Union{}; pure = false)
    end
    S = typeof(standin)
    S <: Matrix && eltype(S) <: _JsMatrixElement && (T === Union{} || T === S) || return nothing
    rows, cols = size(standin)
    rows * cols == length(items) || return nothing
    E = eltype(S)
    lines = String[]
    for i in 1:rows
        codes = String[]
        for j in 1:cols
            c = _js_literal_element(mod, items[(i - 1) * cols + j], E)
            c === nothing && return nothing
            push!(codes, c)
        end
        push!(lines, "[" * join(codes, ", ") * "]")
    end
    _js_rt("[" * join(lines, ", ") * "]", S; pure = false)
end

# A matrix slice, `view(m, :, j)` (a column) or `view(m, i, :)` (a row), read
# where it is used. Julia checks its position where the view is made. An element
# read and its length read the matrix; a function of the whole slice
# (`extrema`, `sum`, `minimum`, `collect`, …) reads a copy of it taken there, so
# it sees the matrix as it is when it is read, as the view does. A write
# through a slice is not translated.
_js_isslice(v) = v isa _JsStruct && v.T === view && length(v.fields) == 3
const _JS_SLICE_READERS = (sum, prod, maximum, minimum, extrema, count, any, all, isempty, collect, copy,
                           argmin, argmax, findmin, findmax, findfirst, findlast, filter)

function _js_slice!(ir, m, i, j, T)
    mod = ir.fn.mod
    x = _js_simple!(ir, m, "m")
    k = _js_simple!(ir, _js_iscolon(i) ? j : i, "k")
    _js_flush!(ir)
    _js_line!(ir.fn, _js_use!(mod, :_rk_mslice) * "(" * x.code * ", " *
                     (_js_iscolon(i) ? "null, " * k.code : k.code * ", null") * ");")
    _JsStruct(view, _js_iscolon(i) ? Any[x, i, k] : Any[x, k, j], T)
end

function _js_slice_copy(ir, s)
    m, i, j = s.fields
    E = eltype(_js_type(m))
    _js_iscolon(i) ? _js_fcall(ir, _js_use!(ir.fn.mod, :_rk_col), Any[m, j], Vector{E}; pure = false) :
                     _js_fcall(ir, _js_use!(ir.fn.mod, :_rk_row), Any[m, i], Vector{E}; pure = false)
end

function _js_slice_call(ir, f, args, T)
    mod = ir.fn.mod
    s = args[1]
    if _js_isslice(s)
        m, i, j = s.fields
        len() = _js_call!(ir, size, Any[m, _js_static(_js_iscolon(i) ? 1 : 2)], Int64)
        if f === getindex && length(args) == 2 && _js_isint(args[2])
            return _js_fcall(ir, _js_use!(mod, :_rk_at2), _js_iscolon(i) ? Any[m, args[2], j] : Any[m, i, args[2]], T)
        elseif f in (length, lastindex) && length(args) == 1
            return len()
        elseif f === firstindex && length(args) == 1
            return _js_static(1)
        elseif f === size && length(args) == 1
            return _JsTuple(Any[len()], Tuple{Int64})
        end
    end
    f in _JS_SLICE_READERS || return nothing
    _js_call!(ir, f, Any[_js_isslice(a) ? _js_slice_copy(ir, a) : a for a in args], T)
end

# `extrema` of a vector: Julia's `minimum` and `maximum` of it, or the error
# `extrema` raises for an empty one.
function _js_extrema(ir, x, T)
    mod = ir.fn.mod
    E = eltype(_js_type(x))
    E <: _JsNumber && T === Tuple{E,E} || return nothing
    empty = try
        extrema(E[])
    catch e
        e
    end
    v = _js_simple!(ir, x, "x")
    pick(how) = _js_rt(_js_use!(mod, :_rk_extremum) * "(" * how * ", " * v.code * ", () => " *
                       _js_runtime(mod, _js_static(empty)).code * ")", E)
    _js_flush!(ir)
    lo = _js_simple!(ir, pick("Math.min"), "lo")
    hi = _js_simple!(ir, pick("Math.max"), "hi")
    _JsTuple(Any[lo, hi], T)
end

function _js_function_call(ir::_JsIR, f::typeof(extrema), args, T)
    length(args) == 1 || return nothing
    _js_isslice(args[1]) && return _js_slice_call(ir, f, args, T)
    _js_isvector(args[1]) ? _js_extrema(ir, args[1], T) : nothing
end
function _js_function_call(ir::_JsIR, f::typeof(view), args, T)
    if length(args) == 3 && _js_ismatrix(args[1]) &&
       ((_js_iscolon(args[2]) && _js_isint(args[3])) || (_js_isint(args[2]) && _js_iscolon(args[3])))
        return _js_slice!(ir, args[1], args[2], args[3], T)
    end
    invoke(_js_function_call, Tuple{_JsIR,Function,Any,Any}, ir, f, args, T)
end

function _js_array_call(ir, f, args, T)
    mod = ir.fn.mod
    if f === Base.vect
        return _js_array_literal(ir, args, T)
    elseif f === getindex && !isempty(args) && args[1] isa _JsVal && args[1].static && args[1].value isa Type
        # `Float64[a, b]`, a literal of a given element type
        return _js_array_literal(ir, args[2:end], T)
    elseif f in (hvcat, _native_hvcat) && length(args) >= 2
        return _js_matrix_literal(ir, f, args[1:1], args[2:end], T)
    elseif f in (hcat, _native_hcat) && !isempty(args)
        return _js_matrix_literal(ir, f, Any[], args, T)
    elseif f === Base.typed_hvcat && length(args) >= 3
        return _js_matrix_literal(ir, f, args[1:2], args[3:end], T)
    elseif f === Base.typed_hcat && length(args) >= 2
        return _js_matrix_literal(ir, f, args[1:1], args[2:end], T)
    end
    any(_js_isslice, args) && return _js_slice_call(ir, f, args, T)
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
    elseif f === setindex! && length(args) == 4
        # `m[i, j] = x`, `m[:, j] = v`, `m[i, :] = v`
        E = eltype(_js_type(m))
        v, i, j = args[2], args[3], args[4]
        if _js_isint(i) && _js_isint(j) && v isa _JsVal && _js_type(v) <: Union{_JsNumber,Bool}
            c = _js_literal_element(mod, v, E)
            c === nothing && return nothing
            return _js_fcall(ir, _js_use!(mod, :_rk_set2), Any[m, _js_rt(c, E), i, j], T; pure = false)
        end
        _js_isvector(v) && (eltype(_js_type(v)) === E || (E === Float64 && eltype(_js_type(v)) === Int64)) ||
            return nothing
        _js_iscolon(i) && _js_isint(j) &&
            return _js_fcall(ir, _js_use!(mod, :_rk_setcol), Any[m, v, j], T; pure = false)
        _js_isint(i) && _js_iscolon(j) &&
            return _js_fcall(ir, _js_use!(mod, :_rk_setrow), Any[m, v, i], T; pure = false)
    end
    nothing
end
