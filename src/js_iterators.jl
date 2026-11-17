# Lazy iterators in `js_source` modules: generators, filters and the functions
# that consume them.
#
# A comprehension with a condition (`[f(x) for x in v if p(x)]`), a generator
# (`(f(x) for x in v if p(x))`) and `Iterators.filter(p, v)` build Julia's lazy
# `Base.Generator` and `Base.Iterators.Filter` over a vector or a range, nested
# as written. While translating they are `_JsStruct`s holding their function
# and the iterator they read (or, with nothing captured, values known while
# translating). A function that consumes one (`collect`, `sum`, `maximum`,
# `minimum`, each also with `init`, `count`, `any`, `all`, `isempty`) emits one
# loop over the source that passes each element through the stages in turn, so
# the functions run in Julia's order and an exception is the first one Julia
# meets. Reductions fold from the first element (or `init`) to the last, as
# Julia reduces an iterator; an empty iterator gives Julia's result or error.
# `argmin`, `argmax`, `findmin`, `findmax` and `filter` of a vector translate
# here too, with Julia's order (`NaN` first, then `-0.0` before `0.0`).
#
# `_js_function_call` reaches these through `js_calls.jl`, which asks each
# family of array translations in turn, then the general table (`js.jl`).

_JS_RUNTIME[:_rk_findmin] = ("""
function _rk_findmin(x, empty) {
  if (x.length === 0) throw empty();
  let m = x[0], k = 0;
  for (let i = 1; i < x.length && m === m; i++) {
    const v = x[i];
    if (v !== v || v < m || (v === 0 && m === 0 && Object.is(v, -0) && !Object.is(m, -0))) { m = v; k = i; }
  }
  return [m, k + 1];
}""", [:JuliaError])
_JS_RUNTIME[:_rk_findmax] = ("""
function _rk_findmax(x, empty) {
  if (x.length === 0) throw empty();
  let m = x[0], k = 0;
  for (let i = 1; i < x.length && m === m; i++) {
    const v = x[i];
    if (v !== v || v > m || (v === 0 && m === 0 && Object.is(m, -0) && !Object.is(v, -0))) { m = v; k = i; }
  }
  return [m, k + 1];
}""", [:JuliaError])

_js_isiterstruct(v) = v isa _JsStruct && (v.T === Base.Generator || v.T === Base.Iterators.Filter)
_js_isiterstatic(v) = v isa _JsVal && v.static && v.value isa Union{Base.Generator,Base.Iterators.Filter}
_js_isloopsource(v) = v isa _JsVal && ((!v.static && _js_isanyvector(v)) || _js_isrange(v))
_js_isiterable(v) = _js_isiterstruct(v) || _js_isiterstatic(v) || _js_isloopsource(v)
_js_stagefn(f) = f isa _JsVal && f.static ? f.value : f

# The source of an iterator and its stages, innermost first: `(:map, f)` for a
# generator, `(:filter, p)` for a filter. `nothing` when the innermost
# iterator is not a runtime vector or a range.
function _js_iterchain(v)
    stages = Tuple{Symbol,Any}[]
    while true
        if _js_isiterstruct(v)
            pushfirst!(stages, (v.T === Base.Generator ? :map : :filter, v.fields[1]))
            v = v.fields[2]
        elseif _js_isiterstatic(v)
            x = v.value
            pushfirst!(stages, x isa Base.Generator ? (:map, _js_static(x.f)) : (:filter, _js_static(x.flt)))
            v = _js_static(x isa Base.Generator ? x.iter : x.itr)
        else
            break
        end
    end
    _js_isloopsource(v) ? (v, stages) : nothing
end

# The type of the elements an iterator produces, each stage checked: a
# generator's function returns a concrete type, a filter's a `Bool`.
function _js_itereltype(mod, src, stages)
    ET = _js_isrange(src) ? Int64 : eltype(_js_type(src))
    for (kind, f) in stages
        g = _js_stagefn(f)
        kind === :map && g === identity && continue
        RT = _js_return_type(g, Any[ET])
        name = g isa _JsStruct ? g.T : g
        if kind === :map
            isconcretetype(RT) || _js_error(mod, "a generator's function `", name, "` returns a `", RT, "` for a `",
                                            ET, "`; a translated generator needs a concrete element type")
            ET = RT
        else
            RT === Bool || _js_error(mod, "a filter condition `", name, "` returns a `", RT, "` for a `", ET,
                                     "`; it must return a `Bool`")
        end
    end
    ET
end

# One loop over an iterator: each element of its source passes through the
# stages in turn, and `each(e)` translates what is done with an element every
# filter keeps. `label` names the loop for an early `break`.
function _js_iterloop!(ir::_JsIR, it, each; label = nothing)
    mod, fn = ir.fn.mod, ir.fn
    src, stages = _js_iterchain(it)
    _js_itereltype(mod, src, stages)
    head = label === nothing ? "for (let " : label * ": for (let "
    if _js_isrange(src)
        r = _js_range!(ir, src)
        _js_flush!(ir)
        k = _js_local!(fn, "k")
        _js_line!(fn, head * k * " = " * _js_range_start(r) * "; " * k * " <= " * _js_range_stop(r) * "; " *
                      k * "++) {")
        fn.indent += 1
        e = _js_rt(k, Int64; ref = true)
    else
        x = _js_simple!(ir, src, "x")
        _js_flush!(ir)
        i = _js_local!(fn, "i")
        _js_line!(fn, head * i * " = 0; " * i * " < " * x.code * ".length; " * i * "++) {")
        fn.indent += 1
        e = _js_simple!(ir, _js_rt(x.code * "[" * i * "]", eltype(x.T)), "e")
    end
    for (kind, f) in stages
        g = _js_stagefn(f)
        kind === :map && g === identity && continue
        RT = _js_return_type(g, Any[_js_argtype(e)])
        if kind === :map
            e = _js_simple!(ir, _js_call!(ir, g, Any[e], RT), "e")
        else
            c = _js_call!(ir, g, Any[e], Bool)
            c isa _JsVal && c.static && c.value === true && continue
            _js_line!(fn, "if (" * _js_not(mod, c).code * ") continue;")
        end
    end
    each(e)
    fn.indent -= 1
    _js_line!(fn, "}")
    nothing
end

# `collect` of an iterator: a new vector of the elements every filter keeps.
function _js_collect_iter!(ir::_JsIR, it, T)
    mod = ir.fn.mod
    _js_itereltype(mod, _js_iterchain(it)...)
    T isa DataType && T <: Vector && _js_isanyvector_type(T) ||
        _js_error(mod, "collecting an iterator into a `", T, "` has no JavaScript translation (vectors of ",
                  "Float64, Int64, Bool, vectors or matrices of them, and tuples or named tuples of those are translated)")
    _js_flush!(ir)
    out = _js_local!(ir.fn, "out")
    _js_line!(ir.fn, "const " * out * " = [];")
    _js_iterloop!(ir, it, e -> _js_line!(ir.fn, out * ".push(" * _js_runtime(mod, e).code * ");"))
    _js_rt(out, T; ref = true)
end

# Julia's result of `op` over an iterator that produces nothing, or its error.
# Julia folds a generator's function and a filter's condition into the
# reduction and reduces the innermost source, so the result is `op` of the
# same stages over an empty source: `sum(x for x in v if x > 0)` is `0.0`,
# `sum(2x for x in v)` the error of an empty reduction. A function capturing
# run-time values is replaced by a stand-in, which Julia reduces the same way.
_js_standin(x) = x
_js_keepall(x) = true
function _js_empty_reduction(op, it)
    src, stages = _js_iterchain(it)
    x = _js_isrange(src) ? (1:0) : eltype(_js_type(src))[]
    for (kind, f) in stages
        g = _js_stagefn(f)
        x = kind === :map ? Base.Generator(g isa _JsStruct ? _js_standin : g, x) :
            Base.Iterators.Filter(g isa _JsStruct ? _js_keepall : g, x)
    end
    try
        op(x)
    catch err
        err
    end
end

# `sum`, `maximum` or `minimum` of an iterator, from its first element or from
# `init`: a left fold, as Julia reduces an iterator.
function _js_reduce_iter!(ir::_JsIR, op, it, T; init = nothing)
    mod = ir.fn.mod
    src, stages = _js_iterchain(it)
    ET = _js_itereltype(mod, src, stages)
    T isa DataType && T <: _JsNumber ||
        _js_error(mod, "`", op, "` of an iterator of `", ET, "`", init === nothing ? "" : " with `init`",
                  " gives a `", T, "`; a translated reduction gives a Float64 or an Int64")
    ET <: _JsNumber || (op === sum && ET === Bool) ||
        _js_error(mod, "`", op, "` of an iterator of `", ET, "` has no JavaScript translation")
    _js_flush!(ir)
    acc = _js_local!(ir.fn, op === sum ? "s" : "m")
    _js_line!(ir.fn, "let " * acc * " = " * (init === nothing ? "null" : _js_runtime(mod, init).code) * ";")
    _js_iterloop!(ir, it, e -> begin
        x = ET === Bool ? "(" * _js_code(mod, e, _JS_COND + 1) * " ? 1 : 0)" : _js_code(mod, e, _JS_ADD + 1)
        step = op === sum ? acc * " + " * x : (op === maximum ? "Math.max(" : "Math.min(") * acc * ", " * x * ")"
        _js_line!(ir.fn, acc * " = " * (init === nothing ? acc * " === null ? " * x * " : " * step : step) * ";")
    end)
    if init === nothing
        empty = _js_empty_reduction(op, it)
        _js_line!(ir.fn, "if (" * acc * " === null) " *
                         (empty isa Exception ? "throw " * _js_exception(mod, empty) :
                          acc * " = " * _js_runtime(mod, _js_static(convert(T, empty))).code) * ";")
    end
    _js_rt(acc, T; ref = true)
end

# The values of an iterator of `Bool`s, checked to be `Bool`s while translating.
function _js_bool_iter(mod, it, what)
    src, stages = _js_iterchain(it)
    ET = _js_itereltype(mod, src, stages)
    ET === Bool || _js_error(mod, "`", what, "` of an iterator of `", ET, "`; its elements must be `Bool`s")
    nothing
end

# `count`, `any` or `all` of an iterator of `Bool`s; `any` and `all` stop at
# the first element that decides them, as Julia does.
function _js_count_iter!(ir::_JsIR, op, it)
    mod = ir.fn.mod
    _js_bool_iter(mod, it, op)
    _js_flush!(ir)
    if op === count
        c = _js_local!(ir.fn, "c")
        _js_line!(ir.fn, "let " * c * " = 0;")
        _js_iterloop!(ir, it, e -> _js_line!(ir.fn, "if (" * _js_runtime(mod, e).code * ") " * c * "++;"))
        return _js_rt(c, Int64; ref = true)
    end
    r, label = _js_local!(ir.fn, "r"), _js_local!(ir.fn, "L")
    _js_line!(ir.fn, "let " * r * " = " * (op === any ? "false" : "true") * ";")
    _js_iterloop!(ir, it, e -> _js_line!(ir.fn, "if (" *
        (op === any ? _js_runtime(mod, e).code : _js_not(mod, e).code) * ") { " * r * " = " *
        (op === any ? "true" : "false") * "; break " * label * "; }"); label)
    _js_rt(r, Bool; ref = true)
end

# `count`, `any` or `all` of a tuple (`all(>=(1), rows)`, `any(a -> length(a) != n,
# axes)`), or of a tuple of `Bool`s without a function: Julia calls the
# function on each element in turn, with that element's own type, and `any` and
# `all` stop at the first element that decides them. A result known while
# translating decides here; the elements after it are not translated, as Julia
# does not call the function on them.
function _js_tuple_count!(ir::_JsIR, op, f, t)
    mod = ir.fn.mod
    S = _js_type(t)
    decide = op === any
    r, label, open = "", "", false
    c = op === count ? 0 : nothing
    for k in 1:_js_tuple_length(t)
        e = _js_function_call(ir, Core.getfield, Any[t, _js_static(k)], fieldtype(S, k))
        e === nothing && return nothing
        v = if f === nothing
            e
        else
            RT = _js_return_type(_js_stagefn(f), Any[_js_argtype(e)])
            RT === Bool || _js_error(mod, "`", op, "` of a `", S, "`: the function `",
                                     f isa _JsStruct ? f.T : _js_stagefn(f), "` returns a `", RT,
                                     "` for element ", k, " (a `", _js_argtype(e), "`); it must return a `Bool`")
            g = _js_stagefn(f)
            folded = g isa _JsStruct ? nothing : _js_fold(_js_static(g), Any[e])
            folded === nothing ? _js_call!(ir, g, Any[e], Bool) : folded
        end
        _js_type(v) === Bool || _js_error(mod, "`", op, "` of a `", S, "`: element ", k, " is a `",
                                          _js_type(v), "`; its elements must be `Bool`s")
        if op === count
            if v isa _JsVal && v.static
                c isa Int ? (c += v.value) : (v.value && _js_line!(ir.fn, c * "++;"))
                continue
            end
            if c isa Int
                n0 = c
                _js_flush!(ir)
                c = _js_local!(ir.fn, "c")
                _js_line!(ir.fn, "let " * c * " = " * string(n0) * ";")
            end
            _js_line!(ir.fn, "if (" * _js_runtime(mod, v).code * ") " * c * "++;")
            continue
        end
        if v isa _JsVal && v.static
            v.value === decide || continue
            open || return _js_static(decide)
            _js_line!(ir.fn, r * " = " * string(decide) * ";")
            break
        end
        if !open
            _js_flush!(ir)
            r, label = _js_local!(ir.fn, "r"), _js_local!(ir.fn, "L")
            _js_line!(ir.fn, "let " * r * " = " * string(!decide) * ";")
            _js_line!(ir.fn, label * ": {")
            ir.fn.indent += 1
            open = true
        end
        _js_line!(ir.fn, "if (" * (decide ? _js_runtime(mod, v).code : _js_not(mod, v).code) * ") { " *
                         r * " = " * string(decide) * "; break " * label * "; }")
    end
    op === count && return c isa Int ? _js_static(c) : _js_rt(c, Int64; ref = true)
    open || return _js_static(!decide)
    ir.fn.indent -= 1
    _js_line!(ir.fn, "}")
    _js_rt(r, Bool; ref = true)
end

# `isempty` of a vector, a range or an iterator. A generator's function is not
# called: only its source and the filters before it decide.
function _js_isempty!(ir::_JsIR, it)
    mod = ir.fn.mod
    if _js_isloopsource(it) && _js_isanyvector(it)
        return _js_rt(_js_code(mod, it, _JS_ATOM) * ".length === 0", Bool; prec = _JS_EQ)
    elseif _js_isrange(it)
        r = _js_range!(ir, it)
        r.static && return _js_static(isempty(r.value))
        return _js_rt(r.code * ".stop < " * r.code * ".start", Bool; prec = _JS_REL)
    end
    src, stages = _js_iterchain(it)
    while !isempty(stages) && last(stages)[1] === :map
        pop!(stages)
    end
    isempty(stages) && return _js_isempty!(ir, src)
    inner = src
    for (kind, f) in stages
        inner = _JsStruct(kind === :map ? Base.Generator : Base.Iterators.Filter, Any[f, inner], Nothing)
    end
    _js_flush!(ir)
    r, label = _js_local!(ir.fn, "r"), _js_local!(ir.fn, "L")
    _js_line!(ir.fn, "let " * r * " = true;")
    _js_iterloop!(ir, inner, e -> _js_line!(ir.fn, r * " = false; break " * label * ";"); label)
    _js_rt(r, Bool; ref = true)
end

# `findmin`/`findmax` of a vector: `(value, index)`; `argmin`/`argmax`: the index.
function _js_findext!(ir::_JsIR, f, x, T)
    mod = ir.fn.mod
    empty = try
        f(eltype(_js_type(x))[])
    catch err
        err
    end
    helper = _js_use!(mod, f in (findmin, argmin) ? :_rk_findmin : :_rk_findmax)
    code = helper * "(" * _js_runtime(mod, x).code * ", () => " * _js_runtime(mod, _js_static(empty)).code * ")"
    f in (findmin, findmax) && return _js_rt(code, T)
    _js_rt(code * "[1]", Int64)
end

# Concatenate one vector level with runtime loops, retaining nested references.
function _js_reduce_vcat!(ir, value, T)
    mod = ir.fn.mod
    x = _js_simple!(ir, value, "rows")
    empty = _js_julia_error(reduce, vcat, _js_type(value)())
    _js_flush!(ir)
    _js_line!(ir.fn, "if (" * x.code * ".length === 0) throw " * _js_exception(mod, empty) * ";")
    out, row, i, j = (_js_local!(ir.fn, name) for name in ("out", "row", "i", "j"))
    at = _js_use!(mod, :_rk_at)
    _js_line!(ir.fn, "const " * out * " = [];")
    _js_line!(ir.fn, "for (let " * i * " = 0; " * i * " < " * x.code * ".length; " * i * "++) {")
    ir.fn.indent += 1
    _js_line!(ir.fn, "const " * row * " = " * at * "(" * x.code * ", " * i * " + 1);")
    _js_line!(ir.fn, "for (let " * j * " = 0; " * j * " < " * row * ".length; " * j * "++) {")
    ir.fn.indent += 1
    _js_line!(ir.fn, out * ".push(" * at * "(" * row * ", " * j * " + 1));")
    ir.fn.indent -= 1
    _js_line!(ir.fn, "}")
    ir.fn.indent -= 1
    _js_line!(ir.fn, "}")
    return _js_rt(out, T; pure = false)
end

function _js_iter_call(ir::_JsIR, f, args, T)
    if f === reduce && length(args) == 2 && args[1] isa _JsVal && args[1].static &&
       args[1].value === vcat && _js_isvector(args[2]) &&
       _js_isvector_type(eltype(_js_type(args[2]))) && T === eltype(_js_type(args[2]))
        return _js_reduce_vcat!(ir, args[2], T)
    end
    mod = ir.fn.mod
    n = length(args)
    if f === collect && n == 1 && _js_iterchain(args[1]) !== nothing
        # one generator over a vector or a range stays the general table's broadcast
        _, stages = _js_iterchain(args[1])
        isempty(stages) || (length(stages) == 1 && stages[1][1] === :map) ||
            return _js_collect_iter!(ir, args[1], T)
    elseif f in (sum, maximum, minimum) && n == 1 && (_js_isiterstruct(args[1]) || _js_isiterstatic(args[1]) ||
                                                      _js_isrange(args[1])) && _js_iterchain(args[1]) !== nothing
        return _js_reduce_iter!(ir, f, args[1], T)
    elseif f in (count, any, all) && n in (1, 2) && _js_istuple(args[n]) && !(_js_type(args[n]) <: NamedTuple)
        return _js_tuple_count!(ir, f, n == 1 ? nothing : args[1], args[n])
    elseif f in (count, any, all) && n in (1, 2) && _js_iterchain(args[n]) !== nothing
        it = n == 1 ? args[1] : _JsStruct(Base.Generator, Any[args[1], args[2]], Nothing)
        return _js_count_iter!(ir, f, it)
    elseif f === isempty && n == 1 && _js_iterchain(args[1]) !== nothing
        return _js_isempty!(ir, args[1])
    elseif f in (argmin, argmax, findmin, findmax) && n == 1 && _js_isloopsource(args[1]) && _js_isflatvector(args[1])
        return _js_findext!(ir, f, args[1], T)
    elseif f === filter && n == 2 && _js_isloopsource(args[2]) && _js_isanyvector(args[2])
        return _js_collect_iter!(ir, _JsStruct(Base.Iterators.Filter, Any[args[1], args[2]], Nothing), T)
    elseif f === Base.Iterators.filter && n == 2 && _js_iterchain(args[2]) !== nothing
        return _JsStruct(Base.Iterators.Filter, Any[args...], T)
    elseif f === Core.kwcall && n == 3 && args[2] isa _JsVal && args[2].static &&
           args[2].value in (sum, maximum, minimum) && _js_type(args[1]) <: NamedTuple &&
           isconcretetype(_js_type(args[1])) && fieldnames(_js_type(args[1])) === (:init,) &&
           _js_iterchain(args[3]) !== nothing
        # `sum(itr; init)`, `maximum(itr; init)`, `minimum(itr; init)`
        init = _js_function_call(ir, Core.getfield, Any[args[1], _js_static(:init)],
                                 fieldtype(_js_type(args[1]), :init))
        init === nothing && return nothing
        return _js_reduce_iter!(ir, args[2].value, args[3], T; init)
    end
    nothing
end

# `Iterators.Filter(p, itr)` and a generator over another iterator, built while
# translating and read by the functions above.
function _js_function_call(ir::_JsIR, f::Type{Base.Iterators.Filter}, args, T)
    length(args) == 2 && _js_iterchain(args[2]) !== nothing && return _JsStruct(f, Any[args...], T)
    invoke(_js_function_call, Tuple{_JsIR,Type,Any,Any}, ir, f, args, T)
end
function _js_function_call(ir::_JsIR, f::Type{Base.Generator}, args, T)
    length(args) == 2 && (_js_isiterstruct(args[2]) || _js_isiterstatic(args[2])) &&
        _js_iterchain(args[2]) !== nothing && return _JsStruct(f, Any[args...], T)
    invoke(_js_function_call, Tuple{_JsIR,Type,Any,Any}, ir, f, args, T)
end
