# Plain-helper idioms of Base in `js_source` modules: predicates made from a
# comparison, the functions that search with them or map before reducing,
# windows and copies of vectors and ranges, and merged named tuples.
#
# - `==(x)`, `!=(x)`, `<(x)`, `<=(x)`, `>(x)`, `>=(x)` and `isequal(x)` build
#   Julia's `Base.Fix2`. While translating it is the static `Fix2` when `x` is
#   known, or a `_JsStruct` holding the comparison and the runtime `x`; calling
#   either calls the comparison with `x` second, as Julia's `Fix2` does. Every
#   function that takes a predicate (`count`, `any`, `all`, `filter`, a
#   generator's condition, and `findfirst`/`findlast` below) takes one of them.
# - `findfirst(p, v)` and `findlast(p, v)` of a vector or a range, and of a
#   vector of `Bool`s, give the position or `nothing` (`null`), searching in
#   Julia's order and stopping at the first match. `isnothing` reads the result.
# - `sum(f, v)`, `maximum(f, v)` and `minimum(f, v)` of a vector or a range
#   compute `f` of every element in order, then reduce as Julia does: a sum
#   pairwise (`_rk_sum`, `Bool`s counted), an extremum with Julia's `NaN` and
#   signed-zero rules. An empty source gives Julia's result or error.
# - `x[a:b]` is a checked copy of the window; `collect` of a range or a vector
#   is a new vector. `abs2` of a number is its square.
# - `merge` of named tuples is a named tuple with the later fields winning, in
#   Julia's field order.
#
# `_js_function_call` reaches these through `js_calls.jl`, which asks each
# family of translations in turn, then the general table (`js.jl`).

const _JS_FIX2_COMPARISONS = (==, !=, <, <=, >, >=, isequal)

_js_findsource(v) = (_js_isvector(v) && !(v isa _JsVal && v.static)) || _js_isrange(v)
_js_elementtype(v) = _js_isrange(v) ? Int64 : eltype(_js_type(v))
_js_callable_name(g) = g isa _JsStruct ? g.T : g

# The empty source of the same kind as `v`, for asking Julia what an empty
# reduction gives.
_js_empty_like(v) = _js_isrange(v) ? (_js_type(v) === Base.OneTo{Int64} ? Base.OneTo(0) : 1:0) :
    _js_elementtype(v)[]

# The position of the first (or, with `forward = false`, the last) element of
# a vector or a range for which the predicate `g` is true, or `null`.
function _js_find!(ir::_JsIR, g, x, T; forward = true, what = forward ? findfirst : findlast)
    mod, fn = ir.fn.mod, ir.fn
    ET = _js_elementtype(x)
    RT = g === identity ? ET : _js_return_type(g, Any[ET])
    RT === Bool || _js_error(mod, "the predicate `", _js_callable_name(g), "` of `", what, "` returns a `", RT,
                             "` for a `", ET, "`; it must return a `Bool`")
    if _js_isrange(x)
        r = _js_range!(ir, x)
        _js_flush!(ir)
        a, b = _js_range_start(r), _js_range_stop(r)
        k = _js_local!(fn, "k")
        head = forward ? "for (let " * k * " = " * a * "; " * k * " <= " * b * "; " * k * "++)" :
            "for (let " * k * " = " * b * "; " * k * " >= " * a * "; " * k * "--)"
        index = r.static && first(r.value) == 1 ? k : k * " - " * (r.static ? _js_number(first(r.value) - 1) :
                                                                   "(" * a * " - 1)")
    else
        xs = _js_simple!(ir, x, "x")
        _js_flush!(ir)
        k = _js_local!(fn, "i")
        head = forward ? "for (let " * k * " = 0; " * k * " < " * xs.code * ".length; " * k * "++)" :
            "for (let " * k * " = " * xs.code * ".length - 1; " * k * " >= 0; " * k * "--)"
        index = k * " + 1"
    end
    out, label = _js_local!(fn, "found"), _js_local!(fn, "L")
    _js_line!(fn, "let " * out * " = null;")
    _js_line!(fn, label * ": " * head * " {")
    fn.indent += 1
    e = _js_isrange(x) ? _js_rt(k, Int64; ref = true) : _js_simple!(ir, _js_rt(xs.code * "[" * k * "]", ET), "e")
    c = g === identity ? e : _js_call!(ir, g, Any[e], Bool)
    if c isa _JsVal && c.static
        c.value === true && _js_line!(fn, out * " = " * index * "; break " * label * ";")
    else
        _js_line!(fn, "if (" * _js_runtime(mod, c).code * ") { " * out * " = " * index * "; break " * label * "; }")
    end
    fn.indent -= 1
    _js_line!(fn, "}")
    _js_rt(out, T; ref = true)
end

# Julia's result of `op(f, v)` for an empty `v` of `x`'s kind, or its error.
# A function holding run-time values is a stand-in, which Julia reduces as it
# reduces every function it does not special-case (`identity`, `abs`, `abs2`).
function _js_empty_mapreduction(op, g, x)
    h = g isa _JsStruct ? _js_standin : g
    try
        op(h, _js_empty_like(x))
    catch err
        err
    end
end

# `sum(f, v)`, `maximum(f, v)` or `minimum(f, v)`: `f` of every element in
# order into a new vector, then Julia's reduction of it.
function _js_mapreduce!(ir::_JsIR, op, g, x, T)
    mod = ir.fn.mod
    ET = _js_elementtype(x)
    RT = _js_return_type(g, Any[ET])
    RT <: _JsNumber || RT === Bool ||
        _js_error(mod, "`", op, "(", _js_callable_name(g), ", …)` of `", ET, "`s: its function returns a `", RT,
                  "`; a translated reduction maps to a Float64, an Int64 or a Bool")
    T isa DataType && (T <: _JsNumber || T === Bool) ||
        _js_error(mod, "`", op, "(", _js_callable_name(g), ", …)` gives a `", T,
                  "`; a translated reduction gives a Float64, an Int64 or a Bool")
    empty = _js_empty_mapreduction(op, g, x)
    v = _js_simple!(ir, _js_materialize!(ir, _JsBroadcast(g, Any[x], Nothing), Vector{RT}), "y")
    _js_flush!(ir)
    if op === sum
        whole = RT === Bool ? v.code * ".reduce((s, b) => b ? s + 1 : s, 0)" : _js_use!(mod, :_rk_sum) * "(" * v.code * ")"
    else
        pick = RT === Bool ? (op === maximum ? "(a, b) => a || b" : "(a, b) => a && b") :
            (op === maximum ? "Math.max" : "Math.min")
        whole = _js_use!(mod, :_rk_extremum) * "(" * pick * ", " * v.code * ", null)"
    end
    if empty isa Exception
        _js_line!(ir.fn, "if (" * v.code * ".length === 0) throw " * _js_exception(mod, empty) * ";")
        return _js_rt(whole, T)
    end
    lit = _js_runtime(mod, _js_static(convert(T, empty)))
    _js_rt(v.code * ".length === 0 ? " * _js_code(mod, lit, _JS_COND) * " : " * whole, T; prec = _JS_COND)
end

# A named tuple read field by field, bound to a name first when it is a
# runtime value that is not one already.
_js_fieldsource!(ir, a::_JsVal) = a.static ? a : _js_simple!(ir, a, "nt")
_js_fieldsource!(ir, a) = a

_js_concrete_namedtuple(S) = S isa DataType && S <: NamedTuple && isconcretetype(S)

function _js_functions_call(ir::_JsIR, f, args, T)
    mod = ir.fn.mod
    n = length(args)
    if f in _JS_FIX2_COMPARISONS && n == 1 && T isa DataType && T <: Base.Fix2 && isconcretetype(T)
        # `==(x)`: a predicate comparing its argument with `x`
        a = args[1]
        a isa _JsVal && a.static && return _js_static(f(a.value))
        return _JsStruct(T, Any[_js_static(f), a], T)
    elseif f === abs2 && n == 1 && _js_isnumber(args[1])
        # `x * x`, as Julia squares a real number
        x = _js_simple!(ir, args[1], "x")
        return _js_rt(_js_code(mod, x, _JS_MUL) * " * " * _js_code(mod, x, _JS_MUL + 1), T; prec = _JS_MUL)
    elseif f === isequal && n == 2 && (all(_js_isnumber, args) || all(_js_isbool, args))
        # `NaN` is equal to itself and `-0.0` is not `0.0`: `Object.is`
        return _js_fcall(ir, "Object.is", args, Bool)
    elseif f === isnothing && n == 1
        S = _js_type(args[1])
        args[1] isa _JsVal && args[1].static && return _js_static(isnothing(args[1].value))
        S === Nothing && return _js_static(true)
        Nothing <: S || return _js_static(false)
        return _js_rt(_js_code(mod, args[1], _JS_EQ) * " === null", Bool; prec = _JS_EQ)
    elseif f in (findfirst, findlast) && n == 2 && _js_findsource(args[2]) && T <: Union{Nothing,Int64}
        # (inference may type a search whose predicate is not a `Bool` as `nothing`:
        # its only path that does not throw; `_js_find!` names the predicate)
        return _js_find!(ir, _js_stagefn(args[1]), args[2], Union{Nothing,Int64}; forward = f === findfirst)
    elseif f in (findfirst, findlast) && n == 1 && _js_findsource(args[1]) && _js_elementtype(args[1]) === Bool &&
           T <: Union{Nothing,Int64}
        # the position of the first (last) `true`
        return _js_find!(ir, identity, args[1], Union{Nothing,Int64}; forward = f === findfirst, what = f)
    elseif f in (sum, maximum, minimum) && n == 2 && _js_findsource(args[2]) &&
           (args[1] isa _JsStruct || (args[1] isa _JsVal && args[1].static && args[1].value isa Base.Callable))
        return _js_mapreduce!(ir, f, _js_stagefn(args[1]), args[2], T)
    elseif f === getindex && n == 2 && _js_isvector(args[1]) && !(args[1] isa _JsVal && args[1].static) &&
           _js_isrange(args[2]) && T === _js_type(args[1])
        # `x[a:b]`: a new vector holding the window, its bounds checked first
        x, r = _js_simple!(ir, args[1], "x"), _js_range!(ir, args[2])
        _js_flush!(ir)
        _js_line!(ir.fn, _js_use!(mod, :_rk_window) * "(" * x.code * ", " * _js_range_start(r) * ", " *
                         _js_range_stop(r) * ");")
        lo = r.static ? _js_number(first(r.value) - 1) : _js_range_start(r) * " - 1"
        return _js_rt(x.code * ".slice(" * lo * ", " * _js_range_stop(r) * ")", T; pure = false)
    elseif f === collect && n == 1 && _js_isrange(args[1]) && T === Vector{Int64}
        r = _js_range!(ir, args[1])
        start = _js_range_start(r)
        return _js_rt("Array.from({ length: " * _js_range_length(r) * " }, (_, i) => " *
                      (start == "0" ? "i" : start * " + i") * ")", T; pure = false)
    elseif f === collect && n == 1 && _js_isvector(args[1]) && T === _js_type(args[1])
        return _js_rt(_js_code(mod, args[1], _JS_ATOM) * ".slice()", T; pure = false)
    elseif f === merge && n >= 1 && all(a -> _js_concrete_namedtuple(_js_type(a)), args) &&
           _js_concrete_namedtuple(T)
        # Julia's field order: the first tuple's names, then each later one's
        # new names; a later tuple's value wins.
        sources = Any[_js_fieldsource!(ir, a) for a in args]
        items = Any[]
        for k in fieldnames(T)
            j = findlast(a -> k in fieldnames(_js_type(a)), sources)
            j === nothing && return nothing
            item = _js_function_call(ir, Core.getfield, Any[sources[j], _js_static(k)], fieldtype(T, k))
            item === nothing && return nothing
            push!(items, item)
        end
        return _JsTuple(items, T)
    end
    nothing
end

# Calling a comparison predicate: the comparison with its value second.
function _js_function_call(ir::_JsIR, f::_JsStruct, args, T)
    if f.T <: Base.Fix2 && length(args) == 1 && f.fields[1] isa _JsVal && f.fields[1].static
        return _js_call!(ir, f.fields[1].value, Any[args[1], f.fields[2]], T)
    end
    nothing
end
function _js_function_call(ir::_JsIR, f::Base.Fix2, args, T)
    length(args) == 1 && return _js_call!(ir, f.f, Any[args[1], _js_static(f.x)], T)
    invoke(_js_function_call, Tuple{_JsIR,Function,Any,Any}, ir, f, args, T)
end
