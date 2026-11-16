# Matrix arithmetic in `js_source` modules, on the row-array matrices of
# `js_arrays.jl`: products (`*` of matrices, vectors and their transposes),
# whole and per-dimension reductions, `vec`, `diag`, transposes, matrix fills, and
# broadcasts over matrices with Julia's shape rules.
#
# A lazy `transpose`/`adjoint` of a real matrix is its transpose's rows, made
# when it is taken; of a real vector it is the vector itself, read as a row.
# Products and broadcasts read both; indexing and `size` of a wrapper are not
# translated. `vec(m)` returns a fresh vector in Julia's column order, where
# Julia's shares the matrix's memory.
#
# `_js_function_call` reaches `_js_linalg_call` through `js_calls.jl`, which asks
# each family of array translations in turn, then the general table (`js.jl`).
# Broadcast materialization and RK's recipe product have their own methods here.

merge!(_JS_RUNTIME, Dict{Symbol,Tuple{String,Vector{Symbol}}}(
    :_rk_vec => ("""
function _rk_vec(m) {
  const rows = m.length, cols = _rk_ncols(m), out = new Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const r = m[i];
    for (let j = 0; j < cols; j++) out[j * rows + i] = r[j];
  }
  return out;
}""", [:_rk_ncols]),
    :_rk_transpose => ("""
function _rk_transpose(m) {
  const rows = m.length, cols = _rk_ncols(m), out = new Array(cols);
  for (let j = 0; j < cols; j++) {
    const r = new Array(rows);
    for (let i = 0; i < rows; i++) r[i] = m[i][j];
    out[j] = r;
  }
  if (cols === 0) out.cols = rows;
  return out;
}""", [:_rk_ncols]),
    :_rk_matvec => ("""
function _rk_matvec(m, x) {
  const cols = _rk_ncols(m);
  if (cols !== x.length)
    throw new JuliaError("DimensionMismatch", "second dimension of A, " + cols + ", does not match length of x, " + x.length);
  const out = new Array(m.length);
  for (let i = 0; i < m.length; i++) {
    const r = m[i];
    let s = 0;
    for (let j = 0; j < cols; j++) s += r[j] * x[j];
    out[i] = s;
  }
  return out;
}""", [:JuliaError, :_rk_ncols]),
    :_rk_vecmat => ("""
function _rk_vecmat(v, m) {
  const cols = _rk_ncols(m);
  if (m.length !== v.length)
    throw new JuliaError("DimensionMismatch", "second dimension of A, " + m.length + ", does not match length of x, " + v.length);
  const out = new Array(cols);
  for (let j = 0; j < cols; j++) {
    let s = 0;
    for (let i = 0; i < m.length; i++) s += v[i] * m[i][j];
    out[j] = s;
  }
  return out;
}""", [:JuliaError, :_rk_ncols]),
    :_rk_matmul => ("""
function _rk_matmul(a, b) {
  const rows = a.length, inner = _rk_ncols(a), cols = _rk_ncols(b);
  if (inner !== b.length)
    throw new JuliaError("DimensionMismatch", "A has dimensions (" + rows + "," + inner + ") but B has dimensions (" +
      b.length + "," + cols + ")");
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const r = a[i], o = new Array(cols);
    for (let j = 0; j < cols; j++) {
      let s = 0;
      for (let p = 0; p < inner; p++) s += r[p] * b[p][j];
      o[j] = s;
    }
    out[i] = o;
  }
  if (rows === 0) out.cols = cols;
  return out;
}""", [:JuliaError, :_rk_ncols]),
    :_rk_dot => ("""
function _rk_dot(v, w) {
  if (v.length !== w.length)
    throw new JuliaError("DimensionMismatch", "dot product arguments have lengths " + v.length + " and " + w.length);
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * w[i];
  return s;
}""", [:JuliaError]),
    :_rk_outer => ("""
function _rk_outer(v, w) {
  const out = new Array(v.length);
  for (let i = 0; i < v.length; i++) {
    const o = new Array(w.length);
    for (let j = 0; j < w.length; j++) o[j] = v[i] * w[j];
    out[i] = o;
  }
  if (v.length === 0) out.cols = w.length;
  return out;
}""", Symbol[]),
    :_rk_prod => ("""
function _rk_prod(x) {
  const n = x.length;
  if (n === 0) return 1;
  return _rk_pairprod(x, 0, n - 1);
}
function _rk_pairprod(x, first, last) {
  if (last - first < 1024) {
    let s = x[first];
    for (let i = first + 1; i <= last; i++) s *= x[i];
    return s;
  }
  const mid = first + Math.floor((last - first) / 2);
  return _rk_pairprod(x, first, mid) * _rk_pairprod(x, mid + 1, last);
}""", Symbol[]),
    :_rk_diag => ("""
function _rk_diag(m) {
  const n = Math.min(m.length, _rk_ncols(m)), out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = m[i][i];
  return out;
}""", [:_rk_ncols]),
    :_rk_mfill => ("""
function _rk_mfill(x, rows, cols) {
  if (!(rows >= 0 && cols >= 0)) throw new JuliaError("ArgumentError", "invalid Array dimensions");
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) out[i] = new Array(cols).fill(x);
  if (rows === 0) out.cols = cols;
  return out;
}""", [:JuliaError]),
    :_rk_bc2 => ("""
function _rk_bc2(m, i, j) {
  const r = m.length === 1 ? m[0] : m[i];
  return r.length === 1 ? r[0] : r[j];
}""", Symbol[]),
    :_rk_samesize => ("""
function _rk_samesize(a, b) {
  const ra = a.length, ca = _rk_ncols(a), rb = b.length, cb = _rk_ncols(b);
  if (ra !== rb || ca !== cb)
    throw new JuliaError("DimensionMismatch", "dimensions must match: a has dims (Base.OneTo(" + ra + "), Base.OneTo(" + ca +
      ")), b has dims (Base.OneTo(" + rb + "), Base.OneTo(" + cb + ")), mismatch at " + (ra !== rb ? 1 : 2));
}""", [:JuliaError, :_rk_ncols]),
    # Reductions over one dimension, in Julia's order: along a column, the
    # column's own reduction after the initial value; along a row, entry by
    # entry from it. A maximum or minimum over an empty dimension throws.
    :_rk_reduce1 => ("""
function _rk_reduce1(m, whole, init, empty) {
  const rows = m.length, cols = _rk_ncols(m), out = new Array(cols);
  if (empty && rows === 0) throw empty();
  const c = new Array(rows);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) c[i] = m[i][j];
    out[j] = init === null ? whole(c) : init(whole(c));
  }
  return [out];
}""", [:_rk_ncols]),
    :_rk_reduce2 => ("""
function _rk_reduce2(m, step, init, empty) {
  const rows = m.length, cols = _rk_ncols(m), out = new Array(rows);
  if (empty && cols === 0) throw empty();
  for (let i = 0; i < rows; i++) {
    const r = m[i];
    let s = init === null ? r[0] : init;
    for (let j = init === null ? 1 : 0; j < cols; j++) s = step(s, r[j]);
    out[i] = [s];
  }
  if (rows === 0) out.cols = 1;
  return out;
}""", [:_rk_ncols]),
))

# A real vector or matrix under a lazy transpose.
_js_wrapper(T) = T isa DataType && (T <: Adjoint || T <: Transpose) && T.parameters[2] isa Type &&
    eltype(T.parameters[2]) <: _JsMatrixElement ? T.parameters[2] : nothing
# A value whose JavaScript is an array of rows: a matrix or a transposed matrix.
_js_matlike(v) = _js_ismatrix(v) || ((P = _js_wrapper(_js_type(v))) !== nothing && P <: Matrix)
# A row: a transposed vector, whose JavaScript is the vector.
_js_rowlike(v) = (P = _js_wrapper(_js_type(v))) !== nothing && P <: Vector
_js_numeltype(v) = eltype(_js_type(v)) <: _JsNumber
_js_scalar(v) = _js_type(v) <: Union{_JsNumber,Bool}

# The error Julia throws for `f(args...)`, as this Julia throws it.
function _js_julia_error(f, args...; kw...)
    try
        f(args...; kw...)
    catch e
        return e
    end
    nothing
end
_js_thunk(mod, e) = "() => " * _js_runtime(mod, _js_static(e)).code

const _JS_EXTREMA = Dict{Any,String}(maximum => "Math.max", minimum => "Math.min")

function _js_linalg_call(ir, f, args, T)
    mod = ir.fn.mod
    n = length(args)
    use(s) = _js_use!(mod, s)
    fresh(name, xs) = _js_fcall(ir, use(name), Any[xs...], T; pure = false)
    if f === Core.kwcall
        return _js_kwcall(ir, args, T)
    elseif f === vec && n == 1 && _js_isvector(args[1]) && T === _js_type(args[1])
        # Julia's vec(::AbstractVector) returns the vector itself, including
        # nested vectors; it does not flatten its entries or copy them.
        return _js_runtime(mod, args[1])
    elseif f === vec && n == 1 && _js_matlike(args[1])
        return fresh(:_rk_vec, args)
    elseif f === LinearAlgebra.diag && n == 1 && _js_matlike(args[1])
        # the main diagonal, a fresh vector
        return fresh(:_rk_diag, args)
    elseif f in (transpose, adjoint, permutedims) && n == 1 && _js_matlike(args[1])
        return fresh(:_rk_transpose, args)
    elseif f in (transpose, adjoint) && n == 1 && (_js_isflatvector(args[1]) || _js_rowlike(args[1]))
        # a row is the vector it transposes, and the transpose of a row is its vector
        r = _js_runtime(mod, args[1])
        return _js_rt(r.code, T; prec = r.prec, ref = r.ref, pure = r.pure)
    elseif f === (*) && n > 2 && any(a -> _js_matlike(a) || _js_rowlike(a), args)
        return _js_mul_chain!(ir, args, T)
    elseif f === (*) && n == 2
        a, b = args
        _js_matlike(a) && _js_isflatvector(b) && return fresh(:_rk_matvec, args)
        _js_matlike(a) && _js_matlike(b) && return fresh(:_rk_matmul, args)
        _js_rowlike(a) && _js_matlike(b) && return fresh(:_rk_vecmat, args)
        _js_rowlike(a) && _js_isflatvector(b) && return _js_fcall(ir, use(:_rk_dot), args, T)
        _js_isflatvector(a) && _js_rowlike(b) && return fresh(:_rk_outer, args)
        if (_js_scalar(a) && _js_matlike(b)) || (_js_matlike(a) && _js_scalar(b))
            return _js_materialize2!(ir, _JsBroadcast(f, Any[a, b], Nothing), T)
        end
    elseif f === (/) && n == 2 && _js_matlike(args[1]) && _js_scalar(args[2])
        return _js_materialize2!(ir, _JsBroadcast(f, Any[args...], Nothing), T)
    elseif f in (+, -) && n == 2 && all(_js_matlike, args)
        x, y = _js_simple!(ir, args[1], "x"), _js_simple!(ir, args[2], "y")
        _js_flush!(ir)
        _js_line!(ir.fn, use(:_rk_samesize) * "(" * x.code * ", " * y.code * ");")
        return _js_materialize2!(ir, _JsBroadcast(f, Any[x, y], Nothing), T)
    elseif f === (-) && n == 1 && _js_matlike(args[1])
        return _js_materialize2!(ir, _JsBroadcast(f, Any[args[1]], Nothing), T)
    elseif f === sum && n == 1 && _js_matlike(args[1])
        code = use(:_rk_sum) * "(" * use(:_rk_vec) * "(" * _js_runtime(mod, args[1]).code * "))"
        # a sum of Booleans is a count
        eltype(_js_type(args[1])) === Bool && return _js_rt(code * " + 0", T; prec = _JS_ADD)
        return _js_rt(code, T)
    elseif f === prod && n == 1 && (_js_matlike(args[1]) || _js_isvector(args[1])) && _js_numeltype(args[1])
        x = _js_runtime(mod, args[1]).code
        return _js_rt(use(:_rk_prod) * "(" * (_js_isvector(args[1]) ? x : use(:_rk_vec) * "(" * x * ")") * ")", T)
    elseif haskey(_JS_EXTREMA, f) && n == 1 && _js_matlike(args[1]) && _js_numeltype(args[1])
        empty = _js_julia_error(f, Matrix{eltype(_js_type(args[1]))}(undef, 0, 0))
        return _js_rt(use(:_rk_extremum) * "(" * _JS_EXTREMA[f] * ", " * use(:_rk_vec) * "(" *
                      _js_runtime(mod, args[1]).code * "), " * _js_thunk(mod, empty) * ")", T)
    elseif f in (zeros, ones) && n in (2, 3) && _js_isint(args[end-1]) && _js_isint(args[end]) &&
           (n == 2 || (args[1] isa _JsVal && args[1].static && args[1].value in (Float64, Int64)))
        return fresh(:_rk_mfill, Any[_js_static(f === zeros ? 0 : 1), args[end-1], args[end]])
    elseif f === fill && n == 3 && _js_scalar(args[1]) && _js_isint(args[2]) && _js_isint(args[3])
        return fresh(:_rk_mfill, args)
    end
    nothing
end

# A product of several factors, `A * B * v`: from the right when it ends in a
# vector, as Julia's `*(A, B, x)` does, else from the left. Julia may pick
# another order for three matrices, which changes only rounding.
function _js_mul_chain!(ir, args, T)
    mod = ir.fn.mod
    step(a, b) = (R = Core.Compiler.return_type(*, Tuple{_js_argtype(a), _js_argtype(b)});
                  isconcretetype(R) || _js_error(mod, "the product of a `", _js_type(a), "` and a `", _js_type(b),
                                                 "` has no concrete type (", R, ")");
                  _js_call!(ir, *, Any[a, b], R))
    if _js_isvector(args[end]) && all(_js_matlike, args[1:end-1])
        acc = args[end]
        for a in reverse(args[1:end-1])
            acc = step(a, acc)
        end
    else
        acc = args[1]
        for b in args[2:end]
            acc = step(acc, b)
        end
    end
    _js_type(acc) == T || _js_error(mod, "a product of ", length(args), " factors of types (",
                                    join(map(_js_type, args), ", "), ") gives a `", _js_type(acc), "`, not a `", T, "`")
    acc
end

# `sum(m; dims = d)` and kin are keyword calls: `Core.kwcall((dims = d,), sum, m)`.
function _js_kwcall(ir, args, T)
    mod = ir.fn.mod
    length(args) == 3 || return nothing
    kw, g, m = args
    kw isa _JsVal && kw.static && kw.value isa NamedTuple && g isa _JsVal && g.static || return nothing
    f = g.value
    f in (sum, prod, maximum, minimum) && _js_matlike(m) || return nothing
    keys(kw.value) == (:dims,) && kw.value.dims in (1, 2) ||
        _js_error(mod, "`", f, "` of a matrix with keywords ", kw.value,
                  " has no JavaScript translation; `dims = 1` and `dims = 2` translate")
    f === sum || _js_numeltype(m) ||
        _js_error(mod, "`", f, "(…; dims)` of a `", _js_type(m), "` has no JavaScript translation")
    d = kw.value.dims
    x = _js_runtime(mod, m).code
    use(s) = _js_use!(mod, s)
    if f in (sum, prod)
        # From the initial value, as Julia's per-dimension reduction starts.
        init, op = f === sum ? ("0", "+") : ("1", "*")
        whole = use(f === sum ? :_rk_sum : :_rk_prod)
        call = d == 1 ? use(:_rk_reduce1) * "(" * x * ", " * whole * ", (s) => " * init * " " * op * " s, null)" :
               use(:_rk_reduce2) * "(" * x * ", (s, v) => s " * op * " v, " * init * ", null)"
    else
        empty = _js_julia_error(f, Matrix{eltype(_js_type(m))}(undef, 0, 0); dims = d)
        pick = _JS_EXTREMA[f]
        call = d == 1 ?
            use(:_rk_reduce1) * "(" * x * ", (c) => " * use(:_rk_extremum) * "(" * pick * ", c, null), null, " *
                _js_thunk(mod, empty) * ")" :
            use(:_rk_reduce2) * "(" * x * ", " * pick * ", null, " * _js_thunk(mod, empty) * ")"
    end
    _js_rt(call, T; pure = false)
end

# --- broadcasts over matrices ------------------------------------------------------------

_js_bc2d_leaf(l) = l isa _JsVal && !l.static && (_js_matlike(l) || _js_rowlike(l))
_js_bc2d(bc) = any(_js_bc2d_leaf, _js_leaves!(Any[], bc))

# A broadcast with a matrix or a row among its operands: one loop over rows and
# one over columns, each dimension of Julia's broadcast length (`dest`, when
# given, is written in place).
function _js_materialize2!(ir::_JsIR, bc::_JsBroadcast, T; dest = nothing)
    mod = ir.fn.mod
    bc = _js_bcvalues(mod, bc)
    leaves = _js_leaves!(Any[], bc)
    for l in leaves
        t = _js_type(l)
        (l isa _JsVal && (l.static || t <: Union{Number,Bool} || _js_isvector(l) || _js_isrange(l) ||
                          _js_matlike(l) || _js_rowlike(l) || t <: Base.RefValue)) ||
            _js_error(mod, "a broadcast over a `", t, "` has no JavaScript translation ",
                      "(scalars, vectors, ranges and matrices are supported)")
    end
    arrays = filter(l -> l isa _JsVal && ((!l.static && (_js_isvector(l) || _js_matlike(l) || _js_rowlike(l))) ||
                                          _js_isrange(l)), leaves)
    names = Dict{Any,_JsVal}()
    for l in arrays
        haskey(names, l) && continue
        names[l] = _js_isrange(l) ? _js_range!(ir, l, "x") : _js_simple!(ir, l, "x")
    end
    _js_flush!(ir)
    use(s) = _js_use!(mod, s)
    # each operand's extent along rows and along columns
    rowlen(l) = _js_isrange(l) ? _js_range_length(names[l]) : _js_rowlike(l) ? nothing : names[l].code * ".length"
    collen(l) = _js_isvector(l) || _js_isrange(l) ? nothing :
        _js_rowlike(l) ? names[l].code * ".length" : use(:_rk_ncols) * "(" * names[l].code * ")"
    rows = unique(filter(!isnothing, map(rowlen, arrays)))
    cols = unique(filter(!isnothing, map(collen, arrays)))
    if dest !== nothing
        pushfirst!(rows, dest.code * ".length")
        pushfirst!(cols, use(:_rk_ncols) * "(" * dest.code * ")")
        unique!(rows)
        unique!(cols)
    end
    extent(ls) = isempty(ls) ? "1" : length(ls) == 1 ? ls[1] : use(:_rk_bclen) * "(" * join(ls, ", ") * ")"
    nr, nc = _js_local!(ir.fn, "nr"), _js_local!(ir.fn, "nc")
    _js_line!(ir.fn, "const " * nr * " = " * extent(rows) * ", " * nc * " = " * extent(cols) * ";")
    out = dest === nothing ? _js_local!(ir.fn, "out") : dest.code
    dest === nothing && _js_line!(ir.fn, "const " * out * " = new Array(" * nr * ");")
    i, j, row = _js_local!(ir.fn, "i"), _js_local!(ir.fn, "j"), _js_local!(ir.fn, "row")
    _js_line!(ir.fn, "for (let " * i * " = 0; " * i * " < " * nr * "; " * i * "++) {")
    ir.fn.indent += 1
    _js_line!(ir.fn, "const " * row * " = " * (dest === nothing ? "new Array(" * nc * ")" : out * "[" * i * "]") * ";")
    _js_line!(ir.fn, "for (let " * j * " = 0; " * j * " < " * nc * "; " * j * "++) {")
    ir.fn.indent += 1
    single_rows, single_cols = length(rows) <= 1, length(cols) <= 1
    elements = Dict{Any,String}()
    for l in arrays
        a = names[l]
        elements[l] = if _js_isrange(l)
            # a range of length one is broadcast like any other
            single_rows ? "(" * _js_range_start(a) * " + " * i * ")" :
                "(" * _js_range_start(a) * " + (" * _js_range_length(a) * " === 1 ? 0 : " * i * "))"
        elseif _js_isvector(l)
            single_rows ? a.code * "[" * i * "]" : use(:_rk_bcat) * "(" * a.code * ", " * i * ")"
        elseif _js_rowlike(l)
            single_cols ? a.code * "[" * j * "]" : use(:_rk_bcat) * "(" * a.code * ", " * j * ")"
        else
            single_rows && single_cols ? a.code * "[" * i * "][" * j * "]" :
                use(:_rk_bc2) * "(" * a.code * ", " * i * ", " * j * ")"
        end
    end
    e = _js_element!(ir, bc, elements, i)
    _js_line!(ir.fn, row * "[" * j * "] = " * _js_runtime(mod, e).code * ";")
    ir.fn.indent -= 1
    _js_line!(ir.fn, "}")
    dest === nothing && _js_line!(ir.fn, out * "[" * i * "] = " * row * ";")
    ir.fn.indent -= 1
    _js_line!(ir.fn, "}")
    dest === nothing && _js_line!(ir.fn, "if (" * nr * " === 0) " * out * ".cols = " * nc * ";")
    _js_rt(out, dest === nothing ? T : dest.T; ref = true)
end

# --- the methods ---------------------------------------------------------------------------

# RK's own product in kernel recipes (`_native_mul(args...) = *(args...)`).
function _js_function_call(ir::_JsIR, f::typeof(_native_mul), args, T)
    v = _js_linalg_call(ir, *, args, T)
    v === nothing || return v
    v = invoke(_js_function_call, Tuple{_JsIR,Function,Any,Any}, ir, *, args, T)
    v === nothing || return v
    invoke(_js_function_call, Tuple{_JsIR,Function,Any,Any}, ir, f, args, T)
end
for F in (Base.Broadcast.materialize, _native_broadcast_materialize)
    @eval function _js_function_call(ir::_JsIR, f::typeof($F), args, T)
        length(args) == 1 && args[1] isa _JsBroadcast && _js_bc2d(args[1]) &&
            return _js_materialize2!(ir, args[1], T)
        invoke(_js_function_call, Tuple{_JsIR,Function,Any,Any}, ir, f, args, T)
    end
end
function _js_function_call(ir::_JsIR, f::typeof(Base.Broadcast.materialize!), args, T)
    if length(args) == 2 && args[2] isa _JsBroadcast && _js_ismatrix(args[1])
        _js_flush!(ir)
        return _js_materialize2!(ir, args[2], T; dest = _js_simple!(ir, args[1], "dest"))
    end
    invoke(_js_function_call, Tuple{_JsIR,Function,Any,Any}, ir, f, args, T)
end
