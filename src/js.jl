# JavaScript source target (prototype).
#
# `js_source` emits a self-contained ES module from a kernel plan. Each recipe's
# typed Julia code is translated statement by statement, and so are the plain
# Julia helpers it calls: a helper needs no annotation, because its control flow
# (branches, early returns, loops) is rebuilt as structured JavaScript from the
# typed code's basic blocks. Julia's own functions translate only through the
# table below (`_js_function_call`); a function without a translation, C calls
# among them, raises `JSEmissionError` naming it and the call chain.
#
# Values. `Float64` and `Int64` are JavaScript numbers (an `Int64` is exact
# while its magnitude stays below 2^53), `Bool` a boolean, a vector a JavaScript
# array (any array-like on input), a tuple an array and a named tuple an object.
# Indexing is checked and 1-based in the source, as in Julia. Sizes and numeric
# inputs stay runtime arguments; nothing is specialized on a value. An
# irrational constant written in the source (`π`) is known from its type: a
# call with it translates Julia's method for it, which rounds it to a float.

"""
    JSEmissionError(message)

`js_source` cannot translate a construct to JavaScript. The message names the
construct and where it was met.
"""
struct JSEmissionError <: Exception
    message::String
end
Base.showerror(io::IO, e::JSEmissionError) = print(io, "JSEmissionError: ", e.message)

# --- JavaScript values -------------------------------------------------------

const _JS_ATOM, _JS_UNARY, _JS_EXP, _JS_MUL, _JS_ADD = 20, 15, 14, 13, 12
const _JS_REL, _JS_EQ, _JS_AND, _JS_OR, _JS_COND = 10, 9, 5, 4, 3

# A value during translation: runtime JavaScript code of Julia type `T`, or a
# Julia value known while translating (`static`), such as a function, a type or
# a literal. `pure` code has no effect, so it may move to its single use;
# `prec` is the precedence of its outermost operator; `ref` marks a name or
# literal that is never reassigned, which may be read again freely.
struct _JsVal
    code::String
    T::Any
    static::Bool
    value::Any
    pure::Bool
    prec::Int
    ref::Bool
end
_js_rt(code, T; pure = true, prec = _JS_ATOM, ref = false) =
    _JsVal(code, T, false, nothing, pure, prec, ref)
_js_static(x) = _JsVal("", Core.Typeof(x), true, x, true, _JS_ATOM, true)

# Translation-time aggregates: a tuple of values, and a lazy broadcast.
struct _JsTuple
    items::Vector{Any}
    T::Any
end
struct _JsBroadcast
    f::Any
    args::Vector{Any}
    T::Any
end
struct _JsStruct
    T::Any          # the type constructed (Base.Generator, a closure type)
    fields::Vector{Any}
    RT::Any         # the Julia type of the value
end
_js_type(v::_JsStruct) = v.RT
_js_type(v::_JsVal) = v.T
_js_type(v::_JsTuple) = v.T
_js_type(v::_JsBroadcast) = v.T

# --- module and function contexts ----------------------------------------------

mutable struct _JsModule
    defs::Vector{String}
    helpers::Dict{Any,Tuple{String,Any}}  # specialization => (name, return type)
    taken::Set{String}
    runtime::Set{Symbol}
    stack::Vector{String}
end
_JsModule() = _JsModule(String[], Dict{Any,Tuple{String,Any}}(), Set{String}(),
                        Set{Symbol}(), String[])

mutable struct _JsFn
    mod::_JsModule
    lines::Vector{String}
    indent::Int
    taken::Set{String}
end
_JsFn(mod) = _JsFn(mod, String[], 1, Set{String}())

function _js_error(mod::_JsModule, parts...)
    where = isempty(mod.stack) ? "" : " (in " * join(reverse(mod.stack), ", called from ") * ")"
    throw(JSEmissionError(string(parts...) * where))
end

_js_line!(fn::_JsFn, s) = push!(fn.lines, "  "^fn.indent * s)

const _JS_RESERVED = Set(split("""
    abstract arguments await boolean break byte case catch char class const continue
    debugger default delete do double else enum eval export extends false final finally
    float for function goto if implements import in instanceof int interface let long
    native new null package private protected public return short static super switch
    synchronized this throw throws transient true try typeof undefined var void volatile
    while with yield Infinity NaN Math Number Array Object String Error JuliaError"""))

function _js_identifier(name)
    s = String(name)
    s = replace(s, r"^#+" => "", "#" => "_", "!" => "_b")
    s = join(c in ('_', '$') || isletter(c) || (isascii(c) && isdigit(c)) ? c : '_' for c in s)
    # RK's generated names (`__plate_value__`, `##captures#1`) read as words
    t = strip(s, '_')
    isempty(t) || isdigit(first(t)) || (s = String(t))
    isempty(s) && (s = "v")
    isdigit(first(s)) && (s = "_" * s)
    s in _JS_RESERVED && (s = s * "_")
    s
end
function _js_unique!(taken::Set{String}, name)
    base = _js_identifier(name)
    s, k = base, 1
    while s in taken
        k += 1
        s = base * "_" * string(k)
    end
    push!(taken, s)
    s
end
_js_local!(fn::_JsFn, name) = _js_unique!(fn.taken, name)

# --- literals ----------------------------------------------------------------------

function _js_number(x::Float64)
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Infinity" : "-Infinity"
    repr(x)
end
_js_number(x::Float32) = _js_number(Float64(x))
function _js_number(x::Integer)
    abs(x) <= 2^53 || throw(JSEmissionError("the integer $x is not exact as a JavaScript number"))
    string(x)
end
_js_string_literal(s) = "\"" * escape_string(String(s)) * "\""

# The runtime code of a value known while translating.
function _js_runtime(mod::_JsModule, v::_JsVal)
    v.static || return v
    x = v.value
    if x isa Bool
        return _js_rt(x ? "true" : "false", Bool; ref = true)
    elseif x isa Union{Float64,Float32,Int64,Int32,UInt8,UInt32}
        code = _js_number(x)
        return _js_rt(code, typeof(x); prec = startswith(code, "-") ? _JS_UNARY : _JS_ATOM, ref = true)
    elseif x === nothing
        return _js_rt("null", Nothing; ref = true)
    elseif x isa Union{String,Symbol}
        return _js_rt(_js_string_literal(x), typeof(x); ref = true)
    elseif x isa Tuple
        # Nested tuples use the same representation as runtime tuple values;
        # each leaf still goes through its ordinary translation/checks.
        items = [_js_runtime(mod, _js_static(e)).code for e in x]
        return _js_rt("[" * join(items, ", ") * "]", typeof(x); ref = false)
    elseif _js_isrange(v)
        # A range known while translating (`1:60`), where a value is needed:
        # the runtime range, as `a:b` builds it.
        return _js_rt(_js_use!(mod, :_rk_range) * "(" * _js_number(first(x)) * ", " * _js_number(last(x)) * ")",
                      typeof(x); ref = false)
    elseif x isa StepRange{Int64,Int64}
        return _js_rt(_js_use!(mod, :_rk_steprange) * "(" * _js_number(first(x)) * ", " * _js_number(step(x)) *
                      ", " * _js_number(last(x)) * ")", typeof(x); ref = false)
    elseif x isa Exception
        return _js_rt(_js_exception(mod, x), typeof(x); pure = true)
    elseif _js_isvector_type(typeof(x)) || (x isa Matrix && eltype(x) <: _JsMatrixElement)
        # An array known while translating (a `const` global the code reads):
        # a module-level array holding its value at emission, which the module
        # reads and writes as the kernel does the global.
        hint = get(mod.helpers, (:global, objectid(x)), ("constant", nothing))[1]
        return _js_constant(mod, x, "_rk_const_", hint; frozen = false)
    end
    _js_error(mod, "the value `", repr(x; context = :limit => true), "` of type `", typeof(x),
              "` is needed at run time and has no JavaScript translation; pass it as a kernel input")
end
function _js_runtime(mod::_JsModule, v::_JsTuple)
    v.T <: NamedTuple || return _js_rt("[" * join((_js_runtime(mod, e).code for e in v.items), ", ") * "]", v.T)
    fields = (_js_identifier(k) * ": " * _js_runtime(mod, e).code for (k, e) in zip(fieldnames(v.T), v.items))
    _js_rt("{ " * join(fields, ", ") * " }", v.T)
end
_js_runtime(mod::_JsModule, v::_JsStruct) =
    _js_error(mod, "a `", v.T, "` value is used at run time; only calls of it are translated")
_js_runtime(mod::_JsModule, v::_JsBroadcast) =
    _js_error(mod, "a lazy broadcast is used as a value; only its materialization is translated")

function _js_exception(mod::_JsModule, e::Exception)
    push!(mod.runtime, :JuliaError)
    msg = sprint(showerror, e)
    "new JuliaError(" * _js_string_literal(nameof(typeof(e))) * ", " * _js_string_literal(msg) * ")"
end

# The negation of a Boolean value, without a double negation.
function _js_not(mod, v)
    r = _js_runtime(mod, v)
    if r.prec == _JS_UNARY && startswith(r.code, "!")
        return _js_rt(r.code[2:end], Bool; prec = _JS_ATOM)
    end
    m = match(r"^([A-Za-z_$][A-Za-z_$0-9]*) (===|!==) null$", r.code)
    m === nothing || return _js_rt(m[1] * (m[2] == "===" ? " !== " : " === ") * "null", Bool; prec = _JS_EQ)
    _js_rt("!" * (r.prec < _JS_UNARY + 1 ? "(" * r.code * ")" : r.code), Bool; prec = _JS_UNARY)
end

_js_code(mod, v, prec) = (r = _js_runtime(mod, v); r.prec < prec ? "(" * r.code * ")" : r.code)

# --- the runtime ---------------------------------------------------------------------

# Special functions JavaScript lacks, as stdlib-js implements them (prebuilt and
# pinned in `js_vendor/`, whose README says how to regenerate them). The file
# defines `_rk_stdlib_<name>`; `ReactiveKernelsSpecialFunctionsExt` maps
# SpecialFunctions' functions to it.
function _js_vendored(name::Symbol)
    file = joinpath(@__DIR__, "js_vendor", string(name, ".js"))
    include_dependency(file)
    (String(rstrip(read(file, String))), Symbol[])
end

# Each helper: (definition, helpers it uses). Emitted only when used.
const _JS_RUNTIME = Dict{Symbol,Tuple{String,Vector{Symbol}}}(
    :JuliaError => ("""
export class JuliaError extends Error {
  constructor(type, message, value) {
    super(message);
    this.name = type;
    if (value !== undefined) this.value = value;
  }
}""", Symbol[]),
    :_rk_float_string => (raw"""
function _rk_float_string(x) {
  if (Number.isNaN(x)) return "NaN";
  if (!Number.isFinite(x)) return x < 0 ? "-Inf" : "Inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const sign = x < 0 ? "-" : "";
  // Reuse the engine's shortest round-trip digits. Only their presentation
  // differs from Julia Float64: decimal point, exponent threshold and sign.
  const parts = Math.abs(x).toString().split("e");
  const dot = parts[0].indexOf(".");
  let point = (dot < 0 ? parts[0].length : dot) + Number(parts[1] || 0);
  let digits = parts[0].replace(".", "");
  const leading = digits.match(/^0*/)[0].length;
  point -= leading;
  digits = digits.slice(leading).replace(/0+$/, "");
  if (point > -4 && point <= 6) {
    if (point <= 0) return sign + "0." + "0".repeat(-point) + digits;
    if (point >= digits.length)
      return sign + digits + "0".repeat(point - digits.length) + ".0";
    return sign + digits.slice(0, point) + "." + digits.slice(point);
  }
  return sign + digits[0] + "." + (digits.slice(1) || "0") + "e" + (point - 1);
}""", Symbol[]),
    :_rk_at => ("""
function _rk_at(x, i) {
  if (!(Number.isInteger(i) && i >= 1 && i <= x.length))
    throw new JuliaError("BoundsError", "attempt to access " + x.length + "-element array at index [" + i + "]");
  if (x[i - 1] === undefined) throw new JuliaError("UndefRefError", "access to undefined reference");
  return x[i - 1];
}""", [:JuliaError]),
    :_rk_set => ("""
function _rk_set(x, v, i) {
  if (!(Number.isInteger(i) && i >= 1 && i <= x.length))
    throw new JuliaError("BoundsError", "attempt to access " + x.length + "-element array at index [" + i + "]");
  x[i - 1] = v;
  return x;
}""", [:JuliaError]),
    :_rk_domain => ("""
function _rk_domain(name, x, condition) {
  throw new JuliaError("DomainError", name + " was called with the argument " + x + "; " + condition, x);
}""", [:JuliaError]),
    :_rk_log => ("""
function _rk_log(x) { return x < 0 ? _rk_domain("log", x, "it must be nonnegative") : Math.log(x); }""", [:_rk_domain]),
    :_rk_log2 => ("""
function _rk_log2(x) { return x < 0 ? _rk_domain("log2", x, "it must be nonnegative") : Math.log2(x); }""", [:_rk_domain]),
    :_rk_log10 => ("""
function _rk_log10(x) { return x < 0 ? _rk_domain("log10", x, "it must be nonnegative") : Math.log10(x); }""", [:_rk_domain]),
    :_rk_log1p => ("""
function _rk_log1p(x) { return x < -1 ? _rk_domain("log1p", x, "it must be at least -1") : Math.log1p(x); }""", [:_rk_domain]),
    :_rk_sqrt => ("""
function _rk_sqrt(x) { return x < 0 ? _rk_domain("sqrt", x, "it must be nonnegative") : Math.sqrt(x); }""", [:_rk_domain]),
    :_rk_asin => ("""
function _rk_asin(x) { return Math.abs(x) > 1 ? _rk_domain("asin", x, "it must be in [-1, 1]") : Math.asin(x); }""", [:_rk_domain]),
    :_rk_acos => ("""
function _rk_acos(x) { return Math.abs(x) > 1 ? _rk_domain("acos", x, "it must be in [-1, 1]") : Math.acos(x); }""", [:_rk_domain]),
    :_rk_acosh => ("""
function _rk_acosh(x) { return x < 1 ? _rk_domain("acosh", x, "it must be at least 1") : Math.acosh(x); }""", [:_rk_domain]),
    :_rk_atanh => ("""
function _rk_atanh(x) { return Math.abs(x) > 1 ? _rk_domain("atanh", x, "it must be in [-1, 1]") : Math.atanh(x); }""", [:_rk_domain]),
    :_rk_pow => ("""
function _rk_pow(x, y) {
  if (x === 1 || y === 0) return 1;
  if (x === -1 && (y === Infinity || y === -Infinity)) return 1;
  if (x < 0 && Number.isFinite(y) && !Number.isInteger(y))
    _rk_domain("^", x, "a negative number raised to a non-integer power is complex");
  return x ** y;
}""", [:_rk_domain]),
    :_rk_ipow => ("""
function _rk_ipow(x, n) {
  if (n < 0 && x !== 1 && x !== -1)
    _rk_domain("^", n, "an integer raised to a negative integer power is not an integer");
  return x ** n;
}""", [:_rk_domain]),
    :_rk_round => ("""
function _rk_round(x) {
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x);
}""", Symbol[]),
    :_rk_int => ("""
function _rk_int(x) {
  if (!(Number.isInteger(x) && Math.abs(x) <= 9007199254740992))
    throw new JuliaError("InexactError", "Int64(" + x + ")", x);
  return x;
}""", [:JuliaError]),
    :_rk_div => ("""
function _rk_div(a, b) {
  if (b === 0) throw new JuliaError("DivideError", "integer division error");
  return Math.trunc(a / b);
}""", [:JuliaError]),
    :_rk_rem => ("""
function _rk_rem(a, b) {
  if (b === 0) throw new JuliaError("DivideError", "integer division error");
  return a % b;
}""", [:JuliaError]),
    :_rk_mod => ("""
function _rk_mod(a, b) {
  if (b === 0) throw new JuliaError("DivideError", "integer division error");
  const r = a % b;
  return r !== 0 && (r < 0) !== (b < 0) ? r + b : r;
}""", [:JuliaError]),
    :_rk_fmod => ("""
function _rk_fmod(x, y) {
  const r = x % y;
  if (r === 0) return y < 0 || Object.is(y, -0) ? -0 : 0;
  return (r > 0) !== (y > 0) ? r + y : r;
}""", Symbol[]),
    :_rk_isinf => ("""
function _rk_isinf(x) { return x === Infinity || x === -Infinity; }""", Symbol[]),
    :_rk_sum => ("""
function _rk_sum(x) {
  const n = x.length;
  if (n === 0) return 0;
  return _rk_pairwise(x, 0, n - 1);
}
function _rk_pairwise(x, first, last) {
  if (last - first < 1024) {
    let s = x[first];
    for (let i = first + 1; i <= last; i++) s += x[i];
    return s;
  }
  const mid = first + Math.floor((last - first) / 2);
  return _rk_pairwise(x, first, mid) + _rk_pairwise(x, mid + 1, last);
}""", Symbol[]),
    :_rk_window => ("""
function _rk_window(x, start, stop) {
  if (stop >= start && !(Number.isInteger(start) && Number.isInteger(stop) && start >= 1 && stop <= x.length))
    throw new JuliaError("BoundsError", "attempt to access " + x.length + "-element array at index [" + start + ":" + stop + "]");
}""", [:JuliaError]),
    :_rk_sumrange => ("""
function _rk_sumrange(x, start, stop) {
  return stop < start ? 0 : _rk_pairwise(x, start - 1, stop - 1);
}""", [:_rk_sum]),
    :_rk_gather => ("""
function _rk_gather(x, index) {
  for (let k = 0; k < index.length; k++)
    if (!(index[k] >= 1 && index[k] <= x.length))
      throw new JuliaError("BoundsError", "attempt to access " + x.length + "-element array at index [[" + Array.from(index).join(", ") + "]]");
  const y = new Array(index.length);
  for (let k = 0; k < index.length; k++) y[k] = x[index[k] - 1];
  return y;
}""", [:JuliaError]),
    :_rk_extremum => ("""
function _rk_extremum(pick, x, empty) {
  if (x.length === 0) throw empty();
  let m = x[0];
  for (let i = 1; i < x.length; i++) m = pick(m, x[i]);
  return m;
}""", [:JuliaError]),
    :_rk_bclen => ("""
function _rk_bclen(...lengths) {
  let n = 1;
  for (const m of lengths) {
    if (m === 1 || m === n) continue;
    if (n !== 1) throw new JuliaError("DimensionMismatch", "arrays could not be broadcast to a common size; got a dimension with lengths " + n + " and " + m);
    n = m;
  }
  return n;
}""", [:JuliaError]),
    :_rk_samelen => ("""
function _rk_samelen(x, y) {
  if (x.length !== y.length)
    throw new JuliaError("DimensionMismatch", "dimensions must match: a has dims (" + x.length + ",), b has dims (" + y.length + ",)");
}""", [:JuliaError]),
    :_rk_bcat => ("""
function _rk_bcat(x, i) { return x.length === 1 ? x[0] : x[i]; }""", Symbol[]),
    :_rk_range => ("""
function _rk_range(start, stop) { return { start, stop: Math.max(start - 1, stop) }; }""", Symbol[]),
    # Matrices as arrays of row arrays (`js_arrays.jl`)
    :_rk_ncols => ("""
function _rk_ncols(m) { return m.length > 0 ? m[0].length : Number.isInteger(m.cols) ? m.cols : 0; }""", Symbol[]),
    :_rk_mcopy => ("""
function _rk_mcopy(m) {
  const c = m.map((r) => Array.from(r));
  if (m.length === 0) c.cols = _rk_ncols(m);
  return c;
}""", [:_rk_ncols]),
    :_rk_at2 => ("""
function _rk_at2(m, i, j) {
  const cols = _rk_ncols(m);
  if (!(Number.isInteger(i) && i >= 1 && i <= m.length && Number.isInteger(j) && j >= 1 && j <= cols))
    throw new JuliaError("BoundsError", "attempt to access " + m.length + "×" + cols + " array at index [" + i + ", " + j + "]");
  return m[i - 1][j - 1];
}""", [:JuliaError, :_rk_ncols]),
    :_rk_col => ("""
function _rk_col(m, j) {
  const cols = _rk_ncols(m);
  if (!(Number.isInteger(j) && j >= 1 && j <= cols))
    throw new JuliaError("BoundsError", "attempt to access " + m.length + "×" + cols + " array at index [:, " + j + "]");
  const out = new Array(m.length);
  for (let i = 0; i < m.length; i++) out[i] = m[i][j - 1];
  return out;
}""", [:JuliaError, :_rk_ncols]),
    :_rk_row => ("""
function _rk_row(m, i) {
  if (!(Number.isInteger(i) && i >= 1 && i <= m.length))
    throw new JuliaError("BoundsError", "attempt to access " + m.length + "×" + _rk_ncols(m) + " array at index [" + i + ", :]");
  return Array.from(m[i - 1]);
}""", [:JuliaError, :_rk_ncols]),
    :_rk_matrix => ("""
function _rk_matrix(name, x, element, integer) {
  if (!Array.isArray(x))
    throw new JuliaError("ArgumentError", "the input `" + name + "` must be an array of rows; got " + x);
  if (x.length === 0 && x.cols !== undefined && !(Number.isInteger(x.cols) && x.cols >= 0))
    throw new JuliaError("ArgumentError", "the input `" + name + "` has no rows and a column count `cols` of " + x.cols);
  for (let i = 0; i < x.length; i++) {
    const r = x[i];
    if (!(Array.isArray(r) || ArrayBuffer.isView(r)))
      throw new JuliaError("ArgumentError", "the input `" + name + "` must be an array of rows; row " + (i + 1) + " is " + r);
    if (r.length !== x[0].length)
      throw new JuliaError("ArgumentError", "the input `" + name + "` must have rows of one length; row " + (i + 1) +
        " has " + r.length + " entries and row 1 has " + x[0].length);
    for (let j = 0; j < r.length; j++)
      if (typeof r[j] !== element || (integer && !Number.isInteger(r[j])))
        throw new JuliaError("ArgumentError", "the input `" + name + "` must hold " + element + "s; entry [" +
          (i + 1) + ", " + (j + 1) + "] is " + r[j]);
  }
}""", [:JuliaError]),
    # `rk_ode_rk45_tol` (`js_ode.jl`)
    :_rk_ode_rk45 => ("""
function _rk_ode_rk45(f, y0, t0, ts, rt, at, mx) {
  const name = "JavaScript DP5";
  const domain = (x, what) => { throw new JuliaError("DomainError", name + " " + what + "; got " + x, x); };
  if (y0.length === 0) throw new JuliaError("ArgumentError", name + " initial state is empty");
  if (ts.length === 0) throw new JuliaError("ArgumentError", name + " output times are empty");
  if (!Number.isFinite(t0)) domain(t0, "initial time");
  for (let j = 0; j < y0.length; j++) if (!Number.isFinite(y0[j])) domain(y0[j], "initial state");
  if (!(Number.isFinite(rt) && rt >= 0)) domain(rt, "relative tolerance");
  if (!(Number.isFinite(at) && at >= 0)) domain(at, "absolute tolerance");
  if (!(rt > 0 || at > 0)) domain("(" + rt + ", " + at + ")", "tolerances are both zero");
  if (!(mx > 0)) domain(mx, "step limit");
  let previous = t0;
  for (let i = 0; i < ts.length; i++) {
    if (!(Number.isFinite(ts[i]) && ts[i] > previous)) domain(ts[i], "output times");
    previous = ts[i];
  }
  const w = ts[ts.length - 1] - t0, n = y0.length;
  if (!Number.isFinite(w)) domain(ts[ts.length - 1], "time span");
  // The library's step size turns NaN after a NaN error estimate, and reaches
  // the right-hand side as a NaN time.
  const nan = {};
  const rhs = (s, y, dy) => {
    if (s !== s) throw nan;
    const d = f(t0 + s * w, y);
    if (!((Array.isArray(d) || ArrayBuffer.isView(d)) && d.length === n))
      throw new JuliaError("DimensionMismatch", "the ODE right-hand side must return a state-length vector");
    for (let j = 0; j < n; j++) dy[j] = w * d[j];
  };
  const solver = new _rk_dopri.Dopri(rhs, n, { atol: 1, rtol: 1, maxSteps: Infinity,
    stepSizeMin: 0, stepSizeMax: 1, tcrit: [1] });
  // The constructor refuses a zero tolerance, which is valid here.
  solver._control.atol = at;
  solver._control.rtol = rt;
  const failure = (retcode) => {
    const t = t0 + solver._t * w;
    const e = new JuliaError("ODESolveFailure", retcode === "MaxIters" ?
      name + " exceeded maxsteps between requested outputs at t = " + t : name + " failed: " + retcode + " at t = " + t);
    e.method = name;
    e.retcode = retcode;
    e.t = t;
    return e;
  };
  const run = (step) => {
    try {
      step();
    } catch (e) {
      // Native's step size shrinks below eps(t) in both cases: a NaN error
      // estimate rejects the step with the smallest step factor there.
      if (e === nan || (e instanceof Error && !(e instanceof JuliaError) &&
                        String(e.message).startsWith("Integration failure: step size vanished")))
        throw failure("Unstable");
      throw e;
    }
  };
  run(() => solver.initialise(0, Array.from(y0)));
  const history = solver._history, out = new Array(ts.length);
  let k = 0;
  for (let i = 0; i < ts.length; i++) {
    const s = (ts[i] - t0) / w, last = i === ts.length - 1;
    let steps = 0;
    // The last step lands on 1 to within rounding of its length.
    while (solver._t < s && !(last && 1 - solver._t <= 2.220446049250313e-16)) {
      run(() => solver._step());
      history.push(solver._stepper.history.clone());
      if (++steps > mx) throw failure("MaxIters");
    }
    while (k < history.length - 1 && history[k].t + history[k].h < s) k++;
    out[i] = solver._stepper.interpolate(s, history[k]);
  }
  return out;
}""", [:JuliaError, :_rk_dopri]),
    :_rk_cumprod => ("""
function _rk_cumprod(x, start = 0, end = x.length) { const r = x.slice(start, end); for (let i = 1; i < r.length; i++) r[i] = r[i - 1] * r[i]; return r; }""", Symbol[]),
    :_rk_cumsum => ("""
function _rk_cumsum(x, start = 0, end = x.length) { const r = x.slice(start, end); for (let i = 1; i < r.length; i++) r[i] = r[i - 1] + r[i]; return r; }""", Symbol[]),
    # `start:step:stop` as Julia builds it: `stop` becomes the last element reached.
    :_rk_steprange => ("""
function _rk_steprange(start, step, stop) {
  if (step === 0) throw new JuliaError("ArgumentError", "step cannot be zero");
  let last = stop;
  if (stop !== start) {
    if ((step > 0) !== (stop > start)) last = start - step;
    else { const remain = Math.abs(stop - start) % Math.abs(step); last = stop > start ? stop - remain : stop + remain; }
  }
  return { start, step, stop: last };
}""", [:JuliaError]),
    :_rk_number => ("""
function _rk_number(name, x, integer) {
  if (typeof x !== "number" || (integer && !Number.isInteger(x)))
    throw new JuliaError("ArgumentError", "the input `" + name + "` must be " + (integer ? "an integer" : "a number") + "; got " + x);
}""", [:JuliaError]),
    :_rk_boolean => ("""
function _rk_boolean(name, x) {
  if (typeof x !== "boolean")
    throw new JuliaError("ArgumentError", "the input `" + name + "` must be a boolean; got " + x);
}""", [:JuliaError]),
    :_rk_vector => ("""
function _rk_vector(name, x, element, integer = false, depth = 1) {
  if (!(Array.isArray(x) || ArrayBuffer.isView(x)))
    throw new JuliaError("ArgumentError", "the input `" + name + "` must be an array; got " + x);
  for (let i = 0; i < x.length; i++) {
    if (depth > 1) {
      _rk_vector(name + "[" + (i + 1) + "]", x[i], element, integer, depth - 1);
    } else if (typeof x[i] !== element || (element === "number" && integer && !Number.isInteger(x[i])))
      throw new JuliaError("ArgumentError", "the input `" + name + "` must hold " + element + "s; entry " + (i + 1) + " is " + x[i]);
  }
}""", [:JuliaError]),
    # As Julia's `iseven`/`isodd` of a float: every integer above `maxintfloat` (2^53) is even.
    :_rk_iseven => ("""
function _rk_iseven(x) { return Number.isInteger(x) && (Math.abs(x) > 9007199254740992 || x % 2 === 0); }""", Symbol[]),
    :_rk_isodd => ("""
function _rk_isodd(x) { return Number.isInteger(x) && Math.abs(x) <= 9007199254740992 && x % 2 !== 0; }""", Symbol[]),
    :_rk_stdlib_erf => _js_vendored(:erf),
    :_rk_stdlib_erfc => _js_vendored(:erfc),
    :_rk_stdlib_erfcx => _js_vendored(:erfcx),
    :_rk_stdlib_gamma => _js_vendored(:gamma),
    :_rk_stdlib_gammaln => _js_vendored(:gammaln),
    # `rk_ode_rk45_tol`'s integrator, `@reside-ic/dopri` (`js_ode.jl`)
    :_rk_dopri => _js_vendored(:dopri),
    :_rk_stdlib_betainc => _js_vendored(:betainc),
    :_rk_stdlib_betaincinv => _js_vendored(:betaincinv),
    # SpecialFunctions' incomplete beta function and its inverse, with Julia's
    # argument checks and special cases. Each tail comes from its own
    # evaluation, so the smaller one keeps its precision.
    :_rk_beta_inc => ("""
function _rk_beta_inc(a, b, x, y) {
  if (a < 0 || b < 0) _rk_domain("beta_inc", [a, b], "a or b is negative");
  if (a === 0 && b === 0) _rk_domain("beta_inc", [a, b], "a and b are 0.0");
  if (x < 0 || x > 1) _rk_domain("beta_inc", x, "x < 0 or x > 1");
  if (y < 0 || y > 1) _rk_domain("beta_inc", y, "y < 0 or y > 1");
  if (Math.abs(x + y - 1) > 3 * 2.220446049250313e-16) _rk_domain("beta_inc", [x, y], "x + y != 1.0");
  if (Number.isNaN(x) || Number.isNaN(y) || Number.isNaN(a) || Number.isNaN(b)) return [NaN, NaN];
  if (x === 0) return [0, 1];
  if (y === 0 || a === 0) return [1, 0];
  if (b === 0) return [0, 1];
  return x <= 0.5 ? [_rk_stdlib_betainc(x, a, b, true, false), _rk_stdlib_betainc(x, a, b, true, true)]
                  : [_rk_stdlib_betainc(y, b, a, true, true), _rk_stdlib_betainc(y, b, a, true, false)];
}""", [:_rk_stdlib_betainc, :_rk_domain]),
    :_rk_beta_inc_inv => ("""
function _rk_beta_inc_inv(a, b, p, q) {
  if (p > 0.5) { const z = _rk_beta_inc_inv(b, a, q, p); return [z[1], z[0]]; }
  if (p === 0) return [0, 1];
  if (p < 0) _rk_domain("log", p, "it must be nonnegative");
  const x = _rk_stdlib_betaincinv(p, a, b, false);
  return [x, 1 - x];
}""", [:_rk_stdlib_betaincinv, :_rk_domain]),
    # `copysign` reads the sign of a NaN as positive: JavaScript keeps no sign of a NaN.
    :_rk_copysign => ("""
function _rk_copysign(x, y) { return y < 0 || Object.is(y, -0) ? -Math.abs(x) : Math.abs(x); }""", Symbol[]),
    # Julia's special cases of SpecialFunctions' `gamma`, `logabsgamma` and
    # `loggamma` (the latter two from FreeBSD `e_lgamma_r.c`, as stdlib's `gammaln`).
    :_rk_gamma => ("""
function _rk_gamma(x) {
  const y = _rk_stdlib_gamma(x);
  return Number.isNaN(y) && !Number.isNaN(x) ? _rk_domain("gamma", x, "NaN result for non-NaN input") : y;
}""", [:_rk_stdlib_gamma, :_rk_domain]),
    :_rk_logabsgamma => ("""
function _rk_logabsgamma(x) {
  if (!Number.isFinite(x)) return [x * x, 1];
  if (x === 0) return [Infinity, 1];
  if (x > 0) return [_rk_stdlib_gammaln(x), 1];
  if (x > -8.470329472543003e-22) return [-Math.log(-x), -1];
  const f = Math.floor(x);
  return f === x ? [Infinity, 1] : [_rk_stdlib_gammaln(x), f % 2 === 0 ? 1 : -1];
}""", [:_rk_stdlib_gammaln]),
    :_rk_loggamma => ("""
function _rk_loggamma(x) {
  const [y, s] = _rk_logabsgamma(x);
  return s < 0 ? _rk_domain("loggamma", x, "`gamma(x)` must be non-negative") : y;
}""", [:_rk_logabsgamma, :_rk_domain]),
)

function _js_runtime_closure(mod::_JsModule)
    seen = Set{Symbol}()
    order = Symbol[]
    visit(s) = s in seen ? nothing : (push!(seen, s); foreach(visit, _JS_RUNTIME[s][2]); push!(order, s))
    foreach(visit, sort!(collect(mod.runtime)))
    order
end

_js_use!(mod::_JsModule, s::Symbol) = (push!(mod.runtime, s); String(s))

# --- supported Julia types -----------------------------------------------------------

const _JsNumber = Union{Float64,Int64}
# an element of a vector or matrix the module holds (`js_arrays.jl`)
const _JsMatrixElement = Union{Float64,Int64,Bool}
_js_isnumber(v) = _js_type(v) <: _JsNumber
_js_isfloat(v) = _js_type(v) === Float64
_js_isint(v) = _js_type(v) === Int64
_js_isbool(v) = _js_type(v) === Bool
# A comparison broadcast (`x .> c`) is a `BitVector` natively: an array of
# Booleans here, read like a `Vector{Bool}`.
_js_isvector_type(::Any) = false
_js_isvector_type(::Type{BitVector}) = true
_js_isvector_type(::Type{Vector{E}}) where {E} = E in (Float64, Int64, Bool) || _js_isvector_type(E)
_js_isvector(v) = _js_isvector_type(_js_type(v))
# Numerical array helpers operate on scalar elements; indexing and iteration
# also accept nested vectors, whose entries remain arrays (possibly ragged).
_js_isflatvector(v) = _js_isvector(v) && eltype(_js_type(v)) <: _JsMatrixElement
_js_isrange(v) = _js_type(v) in (UnitRange{Int64}, Base.OneTo{Int64})
# A stepped range (`n:-1:1`): iterated, measured and read at its ends only.
_js_isstep(v) = _js_type(v) === StepRange{Int64,Int64}
_js_istuple(v) = v isa _JsTuple || (v isa _JsVal && (v.static ? v.value isa Tuple : v.T <: Tuple))
_js_tuple_length(v::_JsTuple) = length(v.items)
_js_tuple_length(v::_JsVal) = v.static ? length(v.value) : fieldcount(v.T)
# A nonempty tuple whose elements all have the type `T`, a value a module holds
# at run time: a number, a `Bool`, a vector or matrix of them, or a tuple of
# such values (`domain = ((lo1, hi1), (lo2, hi2))`, a tuple of axis vectors).
_js_runtime_type(T) = T isa DataType && isconcretetype(T) &&
    (T <: Union{_JsNumber,Bool} || _js_isvector_type(T) || _js_ismatrix_type(T) ||
     (T <: Tuple && !(T <: NamedTuple) && all(_js_runtime_type, fieldtypes(T))))
_js_uniform_tuple(S, T) = S <: Tuple && !(S <: NamedTuple) && isconcretetype(S) && fieldcount(S) > 0 &&
    _js_runtime_type(T) && all(==(T), fieldtypes(S))
# A nonempty tuple whose elements all have the number type `T`.
_js_number_tuple(S, T) = T <: _JsNumber && _js_uniform_tuple(S, T)

# A preceding plate, or a comprehension of a function returning a tuple or a
# named tuple (`[k(d) for d in dose]` with a prepared kernel `k` whose WANT is
# `(; a, b)`), can produce concrete records of runtime values: an array of the
# records' values. Such a vector (`_js_isanyvector`) is read, indexed and
# iterated with the same array and field operations as the vectors above
# (`length`, `x[i]`, `for`, generators, filters and their reductions,
# broadcasts, `map`, `findfirst`, copies, `push!`); numerical array helpers,
# constants and external input validation keep `_js_isvector`'s types.
_js_plate_element(T) = _js_runtime_type(T) ||
    (T isa DataType && isconcretetype(T) && T <: Union{Tuple,NamedTuple} &&
     all(_js_plate_element, fieldtypes(T)))
_js_isanyvector_type(T) = _js_isvector_type(T) || (T isa DataType && T <: Vector && _js_plate_element(eltype(T)))
_js_isanyvector(v) = _js_isanyvector_type(_js_type(v))
_js_vector_view(v) = v isa _JsStruct && v.T === view && length(v.fields) == 2
_js_gather_view(v) = v isa _JsStruct && v.T === _readonly_gather
_js_plate_vector(v) = _js_isanyvector(v) || _js_vector_view(v) || _js_gather_view(v)
_js_plate_length(v) = _js_isrange(v) ? _js_range_length(v) :
    _js_vector_view(v) ? _js_range_length(v.fields[2]) :
    _js_gather_view(v) ? v.fields[2].code * ".length" : v.code * ".length"

# A range is read through its first element, last element and length. A range
# known while translating (`for _ in 1:60` in a helper) is read as literals; a
# runtime one is a `{ start, stop }` object, bound to a name by `_js_range!`.
_js_range!(ir, v, hint = "r") = v isa _JsVal && v.static ? v : _js_simple!(ir, v, hint)
_js_range_start(r::_JsVal) = r.static ? _js_number(first(r.value)) : r.code * ".start"
_js_range_stop(r::_JsVal) = r.static ? _js_number(last(r.value)) : r.code * ".stop"
_js_range_length(r::_JsVal) =
    r.static ? _js_number(length(r.value)) : "(" * r.code * ".stop - " * r.code * ".start + 1)"

# --- typed code ------------------------------------------------------------------------

function _js_typed(mod::_JsModule, f, argtypes)
    tt = Tuple{f isa _JsStruct ? f.T : Core.Typeof(f),argtypes...}
    results = try
        Base.code_typed_by_type(tt; optimize = false)
    catch e
        _js_error(mod, "`", f, "` with argument types (", join(argtypes, ", "), ") could not be inferred: ",
                  sprint(showerror, e))
    end
    length(results) == 1 ||
        _js_error(mod, "`", f, "` has ", length(results), " methods for argument types (",
                  join(argtypes, ", "), "); a translated call needs exactly one")
    ci, rt = only(results)
    ci isa Core.CodeInfo ||
        _js_error(mod, "`", f, "` has no Julia code to translate for argument types (",
                  join(argtypes, ", "), ")")
    match = Base._which(tt)
    m = match.method
    for st in ci.code
        st isa Expr && st.head === :foreigncall || continue
        name = st.args[1]
        name isa QuoteNode && (name = name.value)
        _js_error(mod, "`", f isa _JsStruct ? f.T : f, "` calls the C function `", name,
                  "`, which has no JavaScript translation")
    end
    (ci, rt, m, match.sparams)
end

function _js_return_type(f, argtypes)
    f isa _JsStruct || return Core.Compiler.return_type(f, Tuple{argtypes...})
    results = Base.code_typed_by_type(Tuple{f.T,argtypes...}; optimize = false)
    length(results) == 1 ? last(only(results)) : Any
end

# Functions of these modules translate through the table only: their methods
# reach Julia's internals, C libraries or BLAS.
const _JS_TABLE_ONLY = (Base, Core, Base.Math, Base.Broadcast, Base.Iterators, Base.Checked,
                        Core.Intrinsics, LinearAlgebra, Random)
function _js_table_only(f)
    m = f isa Function && Base.issingletontype(typeof(f)) ? parentmodule(f) : parentmodule(typeof(f))
    while true
        m in _JS_TABLE_ONLY && return true
        p = parentmodule(m)
        p === m && return false
        m = p
    end
end

# --- the IR translator ---------------------------------------------------------------------

mutable struct _JsIR
    fn::_JsFn
    ci::Core.CodeInfo
    nargs::Int
    ssa::Vector{Any}
    slots::Vector{Any}
    slot_names::Vector{String}
    slot_assigns::Vector{Int}
    uses::Vector{Int}
    use_block::Vector{Int}
    block_of::Vector{Int}
    multiblock::Bool
    stable::Set{String}            # names every block of the code sees
    decl_at::Int
    decl_indent::Int
    pending::Vector{Int}
    result::Union{Nothing,String}  # inline: the variable a return assigns
    exit_label::String
    returned::Vector{Any}
    sparams::Core.SimpleVector     # the method's static parameters (`where T`)
end

# Julia 1.10's typed code reads a slot of narrowed type as a `TypedSlot`.
_js_typedslot(x) = isdefined(Core.Compiler, :TypedSlot) && x isa getfield(Core.Compiler, :TypedSlot)

_js_ssatype(ir::_JsIR, i) = ir.ci.ssavaluetypes isa Vector ? ir.ci.ssavaluetypes[i] : Any

function _js_operand(ir::_JsIR, x)
    mod = ir.fn.mod
    if x isa Core.SSAValue
        v = ir.ssa[x.id]
        v === nothing && _js_error(mod, "internal: SSA value %", x.id, " read before its definition")
        k = findfirst(==(x.id), ir.pending)
        k === nothing || deleteat!(ir.pending, k)
        return v
    elseif x isa Core.SlotNumber || x isa Core.Argument || _js_typedslot(x)
        n = x isa Core.Argument ? x.n : x.id
        v = ir.slots[n]
        v === nothing && _js_error(mod, "the local `", ir.ci.slotnames[n], "` may be read before it is assigned")
        return _js_typedslot(x) ? _js_narrow(v, x.typ) : v
    elseif x isa GlobalRef
        isdefined(x.mod, x.name) || _js_error(mod, "`", x, "` is not defined")
        isconst(x.mod, x.name) ||
            _js_error(mod, "the global `", x, "` is not constant; pass its value as a kernel input")
        v = getglobal(x.mod, x.name)
        # an array names the module constant it may become (`_js_runtime`)
        v isa Array && get!(mod.helpers, (:global, objectid(v)), (string(x.name), v))
        return _js_static(v)
    elseif x isa QuoteNode
        return _js_static(x.value)
    elseif x isa Expr && x.head === :static_parameter
        # `T` of a method `where T`: known for the argument types translated.
        k = x.args[1]
        k <= length(ir.sparams) && !(ir.sparams[k] isa TypeVar) ||
            _js_error(mod, "a static parameter whose value the argument types do not determine")
        return _js_static(ir.sparams[k])
    elseif x isa Expr
        _js_error(mod, "the expression `", x, "` as an operand has no JavaScript translation")
    else
        return _js_static(x)
    end
end

# A runtime value read where inference knows a narrower type than the local
# holding it (one slot reused for values of several types, as a keyword sorter
# reuses one for every keyword), or its constant value.
function _js_narrow(v, T)
    v isa _JsVal && !v.static || return v
    if T isa Core.Const
        x = T.val
        (isbits(x) || x isa Union{Symbol,String,Type}) && return _js_static(x)
    end
    RT = Core.Compiler.widenconst(T)
    RT !== v.T && RT <: v.T && isconcretetype(RT) || return v
    _JsVal(v.code, RT, false, nothing, v.pure, v.prec, v.ref)
end

# Bind a value to a name when its code is not simple, so it may be read twice.
function _js_simple!(ir::_JsIR, v::_JsVal, hint = "t")
    v.static && return _js_runtime(ir.fn.mod, v)
    v.ref && return v
    name = _js_local!(ir.fn, hint)
    _js_line!(ir.fn, "const " * name * " = " * v.code * ";")
    _js_rt(name, v.T; ref = true)
end
_js_simple!(ir::_JsIR, v, hint = "t") = v

# Before an effect, a pending pure value is computed where it was defined.
function _js_flush!(ir::_JsIR)
    for i in ir.pending
        v = ir.ssa[i]::_JsVal
        name = _js_local!(ir.fn, "t" * string(i))
        _js_line!(ir.fn, "const " * name * " = " * v.code * ";")
        ir.ssa[i] = _js_rt(name, v.T; ref = true)
    end
    empty!(ir.pending)
end

function _js_statement!(ir::_JsIR, i, st)
    mod = ir.fn.mod
    T = _js_ssatype(ir, i)
    if st isa Expr && st.head === :(=)
        lhs, rhs = st.args
        v = _js_expr!(ir, i, rhs, T)
        n = lhs isa Core.SlotNumber ? lhs.id : _js_error(mod, "assignment to `", lhs, "`")
        _js_assign_slot!(ir, n, v)
        ir.ssa[i] = v
        return
    end
    if st isa Expr && st.head in (:meta, :code_coverage_effect, :inbounds, :loopinfo,
                                   :aliasscope, :popaliasscope, :gc_preserve_end)
        ir.ssa[i] = _js_static(nothing)
        return
    end
    st === nothing && (ir.ssa[i] = _js_static(nothing); return)
    st isa Core.Const && (ir.ssa[i] = _js_static(nothing); return)
    v = _js_expr!(ir, i, st, T)
    _js_define_ssa!(ir, i, v, T)
end

function _js_define_ssa!(ir::_JsIR, i, v, T)
    # An SSA value is a snapshot, including when its expression is just a
    # read of a reassigned local. Keyword sorters reuse such a local for
    # several arguments; retaining its name would read the last argument.
    if v isa _JsVal && !v.static && (!v.ref || v.code in ir.slot_names)
        if T === Union{}
            _js_flush!(ir)
            _js_line!(ir.fn, v.code * ";")
        elseif ir.uses[i] == 0
            _js_flush!(ir)
            v.pure || _js_line!(ir.fn, v.code * ";")
        elseif v.pure && ir.uses[i] == 1 && ir.use_block[i] == ir.block_of[i]
            ir.ssa[i] = v
            push!(ir.pending, i)
            return
        else
            v.pure || _js_flush!(ir)
            if ir.multiblock && ir.use_block[i] != ir.block_of[i]
                name = _js_hoist!(ir, "t" * string(i))
                _js_line!(ir.fn, name * " = " * v.code * ";")
            else
                name = _js_local!(ir.fn, "t" * string(i))
                _js_line!(ir.fn, "const " * name * " = " * v.code * ";")
            end
            v = _js_rt(name, v.T; ref = true)
        end
    end
    ir.ssa[i] = v
end

function _js_assign_slot!(ir::_JsIR, n, v)
    name = ir.slot_names[n]
    if isempty(name)
        # Assigned once: alias a value that is safe to read again, else bind it.
        # In code of several blocks, only a value every block sees is aliased
        # (a JavaScript `const` lives in its block); another is declared on top.
        if v isa _JsVal && !v.static &&
           (!v.ref || v.code in ir.slot_names || (ir.multiblock && !(v.code in ir.stable)))
            _js_flush!(ir)
            if ir.multiblock
                name = _js_hoist!(ir, ir.ci.slotnames[n])
                _js_line!(ir.fn, name * " = " * v.code * ";")
                push!(ir.stable, name)
            else
                name = _js_local!(ir.fn, ir.ci.slotnames[n])
                _js_line!(ir.fn, "const " * name * " = " * v.code * ";")
            end
            v = _js_rt(name, v.T; ref = true)
        end
        ir.slots[n] = v
    elseif v isa _JsStruct && all(f -> f isa _JsVal, v.fields)
        # A closure assigned on every pass of a loop (`all(x -> lo <= x <= hi, v)`
        # inside `for j in 1:d`) stays a value known while translating, called
        # where it is read. Julia's closure holds the values it captured when it
        # was built, so each runtime capture is copied there, into a local of
        # this slot that every block sees (`<slot>$<field>`: `$` is in no
        # translated Julia name).
        _js_flush!(ir)
        fields = Any[]
        for (k, f) in enumerate(v.fields)
            if f.static
                push!(fields, f)
                continue
            end
            copy = name * "\$" * string(k)
            if !(copy in ir.fn.taken)
                push!(ir.fn.taken, copy)
                insert!(ir.fn.lines, ir.decl_at + 1, "  "^ir.decl_indent * "let " * copy * ";")
                ir.decl_at += 1
            end
            _js_line!(ir.fn, copy * " = " * _js_runtime(ir.fn.mod, f).code * ";")
            push!(fields, _js_rt(copy, f.T; ref = true))
        end
        ir.slots[n] = _JsStruct(v.T, fields, v.RT)
    else
        r = _js_runtime(ir.fn.mod, v)
        _js_flush!(ir)
        _js_line!(ir.fn, name * " = " * r.code * ";")
        ir.slots[n] = _js_rt(name, Core.Compiler.widenconst(ir.ci.slottypes[n]); ref = true)
    end
end

function _js_expr!(ir::_JsIR, i, ex, T)
    mod = ir.fn.mod
    ex isa Expr || return _js_narrow(_js_operand(ir, ex), T)
    if ex.head === :call
        fv = _js_operand(ir, ex.args[1])
        args = Any[_js_operand(ir, a) for a in ex.args[2:end]]
        if T isa Core.Const && _js_foldable(fv, args) && !_js_lazy_broadcast(fv, args)
            return _js_static(T.val)
        end
        folded = _js_lazy_broadcast(fv, args) ? nothing : _js_fold(fv, args)
        folded === nothing || return folded
        RT = Core.Compiler.widenconst(T)
        fv isa _JsVal && fv.static ||
            _js_error(mod, "a call through the runtime value `", fv isa _JsVal ? fv.code : fv,
                      "` has no JavaScript translation")
        return _js_call!(ir, fv.value, args, RT)
    elseif ex.head === :new
        S = _js_operand(ir, ex.args[1])
        S isa _JsVal && S.static && S.value isa DataType ||
            _js_error(mod, "constructing a value of a runtime type")
        fields = Any[_js_operand(ir, a) for a in ex.args[2:end]]
        S.value <: Function ||
            _js_error(mod, "constructing a `", S.value, "` has no JavaScript translation")
        all(a -> a isa _JsVal && a.static, fields) && return _js_static(ccall(:jl_new_structv, Any,
            (Any, Ptr{Any}, UInt32), S.value, Any[a.value for a in fields], length(fields)))
        # A closure over runtime values: its captures are read where it is called.
        return _JsStruct(S.value, fields, S.value)
    elseif ex.head === :foreigncall
        _js_error(mod, "a C call (`", ex.args[1], "`) has no JavaScript translation")
    elseif ex.head === :static_parameter
        T isa Core.Const && return _js_static(T.val)
        return _js_operand(ir, ex)
    elseif ex.head === :boundscheck
        return _js_static(true)
    elseif ex.head in (:enter, :leave, :pop_exception)
        _js_error(mod, "try/catch has no JavaScript translation")
    elseif ex.head === :isdefined
        return _js_static(true)
    elseif ex.head === :throw_undef_if_not
        return _js_static(nothing)
    end
    _js_error(mod, "the expression `", ex, "` has no JavaScript translation")
end

# A call inference folded to a constant, without an effect to keep. Its
# arguments may be runtime values: a constant that depends only on their types
# (`LogExpFunctions._log1pexp_thresholds(x)`, the keyword check of a keyword
# call) is the same constant for every input.
function _js_foldable(fv, args)
    fv isa _JsVal && fv.static || return false
    f = fv.value
    f isa Core.Builtin && return true
    f isa Type && return true
    types = Any[a isa _JsVal && a.static ? Core.Typeof(a.value) : _js_argtype(a) for a in args]
    effects = try
        Base.infer_effects(f, Tuple{types...})
    catch
        return false
    end
    Core.Compiler.is_removable_if_unused(effects)
end

# A call of a known function on known arguments that Julia's compiler may
# evaluate ahead of time (consistent, without effects): evaluated now. Reads
# of RK's own operation objects (`split.call`) are such calls.
function _js_fold(fv, args)
    fv isa _JsVal && fv.static || return nothing
    all(a -> a isa _JsVal && a.static, args) || return nothing
    f = fv.value
    vals = Any[a.value for a in args]
    ok = if f === getproperty || f === getfield || f === Core.getfield
        # a constant of a module (`Iterators.filter` in a helper), as a global reads
        !ismutable(vals[1]) || (vals[1] isa Module && vals[2] isa Symbol && isconst(vals[1], vals[2]))
    elseif f isa Core.Builtin || f isa Type
        f in (Core.tuple, Core.apply_type, typeof, fieldtype, nfields, (===), isa, Core.sizeof)
    else
        effects = try
            Base.infer_effects(f, Tuple{(Core.Typeof(v) for v in vals)...})
        catch
            return nothing
        end
        Core.Compiler.is_foldable(effects) && Core.Compiler.is_nothrow(effects)
    end
    ok || return nothing
    v = try
        f(vals...)
    catch
        return nothing
    end
    _js_static(v)
end

# --- control flow ------------------------------------------------------------------------

struct _JsCFG
    blocks::Vector{UnitRange{Int}}
    block_of::Vector{Int}
    succ::Vector{Vector{Int}}
    reach::Vector{Bool}
    loop_end::Dict{Int,Int}        # loop header => last block of its body
    labeled::Set{Int}              # blocks a `break` reaches
    ifs::Dict{Int,Int}             # block an `if` ends => its condition block
    scopes::Vector{Tuple{Int,Int,Symbol,Int}}  # (first block, last block, :loop/:block, target)
end

_js_noreturn(ci, i) = ci.ssavaluetypes isa Vector && ci.ssavaluetypes[i] === Union{} &&
    !(ci.code[i] isa Union{Core.GotoNode,Core.GotoIfNot,Core.ReturnNode})

# The value inference knows a branch condition to have (`if FLAG` on a constant
# global, the entry test of a loop over `1:0`), or `nothing`. Such a branch
# takes one edge only, and the code on the other is unreachable.
function _js_known_cond(ci::Core.CodeInfo, st::Core.GotoIfNot)
    c = st.cond
    c isa Bool && return c
    if c isa Core.SSAValue && ci.ssavaluetypes isa Vector
        t = ci.ssavaluetypes[c.id]
        t isa Core.Const && t.val isa Bool && return t.val
    elseif c isa GlobalRef && isdefined(c.mod, c.name) && isconst(c.mod, c.name)
        v = getglobal(c.mod, c.name)
        v isa Bool && return v
    end
    nothing
end

# A forward jump over unreachable blocks only is a fall-through.
_js_falls_to(reach, from, to) = all(b -> !reach[b], (from + 1):(to - 1))

# The basic blocks of typed code: each starts at a jump target or after a
# jump, a return or a call that never returns.
function _js_partition(ci::Core.CodeInfo)
    code = ci.code
    n = length(code)
    starts = Set{Int}(1)
    for (i, st) in enumerate(code)
        if st isa Core.GotoNode
            push!(starts, st.label)
            i < n && push!(starts, i + 1)
        elseif st isa Core.GotoIfNot
            push!(starts, st.dest)
            i < n && push!(starts, i + 1)
        elseif st isa Core.ReturnNode || _js_noreturn(ci, i)
            i < n && push!(starts, i + 1)
        end
    end
    s = sort!(collect(starts))
    blocks = [s[k]:(k < length(s) ? s[k+1] - 1 : n) for k in eachindex(s)]
    block_of = zeros(Int, n)
    for (b, r) in enumerate(blocks), i in r
        block_of[i] = b
    end
    blocks, block_of
end

function _js_cfg(mod::_JsModule, ci::Core.CodeInfo)
    code = ci.code
    any(st -> st isa Expr && st.head === :enter, code) &&
        _js_error(mod, "try/catch has no JavaScript translation")
    blocks, block_of = _js_partition(ci)
    nb = length(blocks)
    succ = [Int[] for _ in 1:nb]
    for (b, r) in enumerate(blocks)
        i = last(r)
        st = code[i]
        if st isa Core.GotoNode
            push!(succ[b], block_of[st.label])
        elseif st isa Core.GotoIfNot
            known = _js_known_cond(ci, st)
            known === true || push!(succ[b], block_of[st.dest])
            known === false || (b < nb && push!(succ[b], b + 1))
        elseif st isa Core.ReturnNode || _js_noreturn(ci, i)
        elseif b < nb
            push!(succ[b], b + 1)
        end
    end
    reach = falses(nb)
    work = [1]
    while !isempty(work)
        b = pop!(work)
        reach[b] && continue
        reach[b] = true
        append!(work, succ[b])
    end
    loop_end = Dict{Int,Int}()
    for b in 1:nb, t in succ[b]
        reach[b] && t <= b && (loop_end[t] = max(get(loop_end, t, t), b))
    end
    for (h, e) in loop_end, b in 1:nb, t in succ[b]
        reach[b] && h < t <= e && !(h <= b <= e) &&
            _js_error(mod, "control flow enters a loop other than at its start; it has no structured JavaScript translation")
    end
    # Forward edges that need a jump to their target.
    ends_loop(b) = any(e == b for e in values(loop_end))
    sources = Dict{Int,Set{Int}}()
    for b in 1:nb
        reach[b] || continue
        st = code[last(blocks[b])]
        for t in succ[b]
            t > b || continue
            fall = !(st isa Core.GotoIfNot && t == block_of[st.dest] && t != b + 1) && t == b + 1
            (fall && !ends_loop(b)) && continue
            (_js_falls_to(reach, b, t) && !ends_loop(b)) && continue
            push!(get!(Set{Int}, sources, t), b)
        end
    end
    # A target reached only by the conditional jump of one block `o` is the
    # end of `if (condition) { ... }` over the blocks after `o`; every other
    # target ends a labeled block a `break` leaves.
    ifs = Dict{Int,Int}()
    for (t, srcs) in sources
        length(srcs) == 1 || continue
        o = only(srcs)
        st = code[last(blocks[o])]
        st isa Core.GotoIfNot && block_of[st.dest] == t && t != o + 1 && _js_known_cond(ci, st) === nothing &&
            (ifs[t] = o)
    end
    function nest()
        # Open each labeled block early enough to nest with the loops, the
        # `if`s (which enclose their condition block) and the other labeled blocks.
        opens = Dict{Int,Int}(t => minimum(srcs) for (t, srcs) in sources if !haskey(ifs, t))
        fixed = [(h, e) for (h, e) in loop_end]
        append!(fixed, [(o, t - 1) for (t, o) in ifs])
        changed = true
        while changed
            changed = false
            for (t, o) in opens
                for (h, e) in fixed
                    if h < o <= e && e <= t - 1
                        opens[t] = h; changed = true; o = h
                    elseif o < h <= t - 1 < e
                        haskey(loop_end, h) && e == loop_end[h] && _js_error(mod,
                            "control flow enters a loop other than at its start; it has no structured JavaScript translation")
                        return nothing, findfirst(==(h), ifs)
                    end
                end
                for (t2, o2) in opens
                    if o < o2 <= t - 1 < t2 - 1
                        opens[t2] = o; changed = true
                    end
                end
            end
        end
        # An `if` opens at the end of its condition block and must not cross a
        # scope; the jump past it must not land inside a loop it ends with.
        others = vcat([(Float64(h), Float64(e), true) for (h, e) in loop_end],
                      [(Float64(o), Float64(t - 1), false) for (t, o) in opens],
                      [(o + 0.5, Float64(t - 1), false) for (t, o) in ifs])
        for (t, o) in ifs
            a, z = o + 0.5, Float64(t - 1)
            for (s1, s2, isloop) in others
                (s1, s2) == (a, z) && continue
                nested = s2 < a || z < s1 || (s1 <= a && z <= s2) || (a <= s1 && s2 <= z)
                (!nested || (isloop && s1 <= a && s2 == z)) && return nothing, t
            end
        end
        opens, nothing
    end
    opens = nothing
    while true
        opens, bad = nest()
        opens === nothing || break
        delete!(ifs, bad)
    end
    scopes = Tuple{Int,Int,Symbol,Int}[]
    for (h, e) in loop_end
        push!(scopes, (h, e, :loop, h))
    end
    for (t, o) in opens
        push!(scopes, (o, t - 1, :block, t))
    end
    # Outer scopes open first: larger extent; a block before a loop of the same extent.
    sort!(scopes; by = s -> (s[1], -s[2], s[3] === :block ? 0 : 1))
    _JsCFG(blocks, block_of, succ, reach, loop_end, Set(keys(opens)), ifs, scopes)
end

function _js_edge!(ir::_JsIR, cfg::_JsCFG, from, to; conditional = false)
    if to <= from
        _js_line!(ir.fn, "continue L" * string(to) * ";")
    elseif to in cfg.labeled && (conditional || to != from + 1 ||
                                 any(e == from for e in values(cfg.loop_end)))
        _js_line!(ir.fn, "break L" * string(to) * ";")
    elseif to != from + 1 && !_js_falls_to(cfg.reach, from, to)
        _js_error(ir.fn.mod, "internal: no label for a jump to block ", to)
    end
end

function _js_body!(ir::_JsIR)
    mod = ir.fn.mod
    ci = ir.ci
    cfg = _js_cfg(mod, ci)
    ir.block_of = cfg.block_of
    stack = Tuple{Int,Int,Symbol,Int}[]
    shown = Bool[]   # whether each open scope was written
    k = 1
    # The blocks an emitted edge reaches: a condition known only while
    # translating (a test of a value a helper is specialized on) takes one
    # edge. Control enters a scope only at its first block, so a scope that
    # starts at a block no edge reaches is dead throughout and is not written.
    live = falses(length(cfg.blocks))
    live[1] = true
    dead_until = 0
    for (b, r) in enumerate(cfg.blocks)
        while !isempty(stack) && last(stack)[2] == b - 1
            pop!(stack)
            if pop!(shown)
                ir.fn.indent -= 1
                _js_line!(ir.fn, "}")
            end
        end
        reached = cfg.reach[b] && live[b]
        while k <= length(cfg.scopes) && cfg.scopes[k][1] == b
            s = cfg.scopes[k]
            k += 1
            show = reached && b > dead_until
            if show
                label = "L" * string(s[4])
                _js_line!(ir.fn, s[3] === :loop ? label * ": while (true) {" : label * ": {")
                ir.fn.indent += 1
            else
                dead_until = max(dead_until, s[2])
            end
            push!(stack, s)
            push!(shown, show)
        end
        reached || continue
        b <= dead_until && _js_error(mod, "internal: block ", b, " is reached inside a scope that starts in dead code")
        ended = false
        for i in r
            st = ci.code[i]
            if st isa Core.GotoNode
                _js_flush!(ir)
                live[cfg.block_of[st.label]] = true
                _js_edge!(ir, cfg, b, cfg.block_of[st.label])
                ended = true
            elseif st isa Core.GotoIfNot
                c = _js_operand(ir, st.cond)
                dest = cfg.block_of[st.dest]
                known = _js_known_cond(ci, st)
                known === nothing && c isa _JsVal && c.static && (known = c.value::Bool)
                if known !== nothing
                    # An `if` planned over b+1:dest-1 needs no code: its body
                    # runs by falling through, or is skipped as dead code.
                    _js_flush!(ir)
                    if known === false
                        live[dest] = true
                        get(cfg.ifs, dest, 0) == b || _js_edge!(ir, cfg, b, dest)
                        ended = true
                    end
                elseif get(cfg.ifs, dest, 0) == b
                    code = _js_runtime(mod, c).code
                    _js_flush!(ir)
                    _js_line!(ir.fn, "if (" * code * ") {")
                    ir.fn.indent += 1
                    push!(stack, (b + 1, dest - 1, :if, dest))
                    push!(shown, true)
                    live[dest] = true
                else
                    code = _js_not(mod, c).code
                    _js_flush!(ir)
                    _js_line!(ir.fn, "if (" * code * ") {")
                    ir.fn.indent += 1
                    _js_edge!(ir, cfg, b, dest; conditional = true)
                    ir.fn.indent -= 1
                    _js_line!(ir.fn, "}")
                    live[dest] = true
                end
            elseif st isa Core.ReturnNode
                if !isdefined(st, :val)
                    _js_flush!(ir)
                    push!(mod.runtime, :JuliaError)
                    _js_line!(ir.fn, "throw new JuliaError(\"ErrorException\", \"unreachable code was reached\");")
                else
                    _js_return!(ir, _js_operand(ir, st.val))
                end
                ended = true
            else
                _js_statement!(ir, i, st)
                if _js_noreturn(ci, i)
                    ended = true
                    break
                end
            end
        end
        _js_flush!(ir)
        if !(ended || b == length(cfg.blocks))
            live[b+1] = true
            _js_edge!(ir, cfg, b, b + 1)
        end
    end
    while !isempty(stack)
        pop!(stack)
        if pop!(shown)
            ir.fn.indent -= 1
            _js_line!(ir.fn, "}")
        end
    end
end

function _js_return!(ir::_JsIR, v)
    push!(ir.returned, v)
    if ir.result === nothing
        code = v isa _JsVal && v.static && v.value === nothing ? "" : " " * _js_runtime(ir.fn.mod, v).code
        _js_flush!(ir)
        _js_line!(ir.fn, "return" * code * ";")
    else
        r = _js_runtime(ir.fn.mod, v)
        _js_flush!(ir)
        _js_line!(ir.fn, ir.result * " = " * r.code * ";")
        _js_line!(ir.fn, "break " * ir.exit_label * ";")
    end
end

function _js_ir(fn::_JsFn, ci::Core.CodeInfo, nargs::Int, sparams::Core.SimpleVector = Core.svec())
    n = length(ci.code)
    uses = zeros(Int, n)
    use_block = zeros(Int, n)
    blocks, cfg_blocks = _js_partition(ci)
    function count!(x, i)
        if x isa Core.SSAValue
            uses[x.id] += 1
            use_block[x.id] = use_block[x.id] == 0 ? cfg_blocks[i] : -1
        elseif x isa Expr
            foreach(a -> count!(a, i), x.args)
        elseif x isa Core.GotoIfNot
            count!(x.cond, i)
        elseif x isa Core.ReturnNode && isdefined(x, :val)
            count!(x.val, i)
        end
    end
    for (i, st) in enumerate(ci.code)
        count!(st, i)
    end
    for i in 1:n
        use_block[i] == -1 && (use_block[i] = 0)
    end
    nslots = length(ci.slotnames)
    assigns = zeros(Int, nslots)
    for st in ci.code
        st isa Expr && st.head === :(=) && st.args[1] isa Core.SlotNumber && (assigns[st.args[1].id] += 1)
    end
    _JsIR(fn, ci, nargs, Vector{Any}(nothing, n), Vector{Any}(nothing, nslots),
          fill("", nslots), assigns, uses, use_block, cfg_blocks, length(blocks) > 1, Set{String}(), 0, 0, Int[],
          nothing, "", Any[], sparams)
end

# Declare the locals assigned more than once (or in a loop) at the top.
function _js_static_slot(ir::_JsIR, n)
    t = ir.ci.slottypes isa Vector ? ir.ci.slottypes[n] : Any
    t isa Core.Const && return true
    T = Core.Compiler.widenconst(t)
    Base.issingletontype(T) || !(T <: Union{Number,Bool,Nothing,String,Vector,Tuple,NamedTuple,UnitRange})
end

function _js_declare_slots!(ir::_JsIR)
    for n in 1:min(ir.nargs, length(ir.slots))
        v = ir.slots[n]
        v isa _JsVal && v.ref && !v.static && push!(ir.stable, v.code)
        v isa _JsTuple && foreach(e -> e isa _JsVal && e.ref && !e.static && push!(ir.stable, e.code), v.items)
    end
    for n in 1:length(ir.ci.slotnames)
        multiple = ir.slot_assigns[n] > 1 || (n <= ir.nargs && ir.slot_assigns[n] > 0)
        if ir.slot_assigns[n] == 1 && n > ir.nargs
            # A single assignment inside a loop is reassigned on every iteration.
            for (i, st) in enumerate(ir.ci.code)
                st isa Expr && st.head === :(=) && st.args[1] == Core.SlotNumber(n) || continue
                any(s -> (s isa Core.GotoNode && s.label <= i) ||
                         (s isa Core.GotoIfNot && s.dest <= i), ir.ci.code[i:end]) && (multiple = true)
            end
        end
        multiple || continue
        if n <= ir.nargs && ir.slots[n] isa _JsVal && !ir.slots[n].static
            # a reassigned argument: copy it into a local
            name = _js_local!(ir.fn, ir.ci.slotnames[n])
            _js_line!(ir.fn, "let " * name * " = " * ir.slots[n].code * ";")
        else
            name = _js_local!(ir.fn, ir.ci.slotnames[n])
            _js_line!(ir.fn, "let " * name * ";")
        end
        ir.slot_names[n] = name
        n <= ir.nargs && (ir.slots[n] = _js_rt(name, Core.Compiler.widenconst(ir.ci.slottypes[n]); ref = true))
    end
    ir.decl_at = length(ir.fn.lines)
    ir.decl_indent = ir.fn.indent
end

# A local declared at the top of the code, for a value read in another block.
function _js_hoist!(ir::_JsIR, hint)
    name = _js_local!(ir.fn, hint)
    insert!(ir.fn.lines, ir.decl_at + 1, "  "^ir.decl_indent * "let " * name * ";")
    ir.decl_at += 1
    name
end

# --- calls -------------------------------------------------------------------------------------

"""
    _js_function_call(ir, f, args, T)

The JavaScript translation of the call `f(args...)` whose result has Julia type
`T`, or `nothing` when this method does not translate it. A package adds a
method for its own function type.
"""
_js_function_call(ir, f, args, T) = nothing

# The `MethodError` of a keyword call `f(x; a, b)` that no method of `f`
# takes: `args` are the keywords, `f` and the positional arguments.
function _js_kwerr(mod::_JsModule, args)
    NT = _js_type(args[1])
    f = args[2] isa _JsVal && args[2].static ? args[2].value : _js_type(args[2])
    msg = "no method matching " * string(f isa Function ? nameof(f) : f) * "(" *
          join(("::" * string(_js_argtype(a)) for a in args[3:end]), ", ") * "; " *
          join((string(k) * "::" * string(S) for (k, S) in zip(fieldnames(NT), fieldtypes(NT))), ", ") * ")"
    push!(mod.runtime, :JuliaError)
    _js_rt("throw new JuliaError(\"MethodError\", " * _js_string_literal(msg) * ")", Union{}; pure = false)
end

function _js_call!(ir::_JsIR, f, args::Vector{Any}, T)
    mod = ir.fn.mod
    v = _js_function_call(ir, f, args, T)
    v === nothing || return v
    if f === Core.kwcall && length(args) >= 2 && _js_type(args[1]) <: NamedTuple &&
       args[2] isa _JsVal && args[2].static && !_js_table_only(args[2].value)
        # `f(x; a, b)`: the keyword sorter of `f`, whose checks of the keyword
        # names inference decides, then calls `f`'s body function. Without a
        # keyword method, the call is Julia's `MethodError`.
        isempty(methods(Core.kwcall, Tuple{(_js_argtype(a) for a in args)...})) &&
            return _js_kwerr(mod, args)
        _js_flush!(ir)
        v, _ = _js_inline!(ir.fn, f, args, "value")
        return v
    end
    if f isa Type && f <: Exception
        push!(mod.runtime, :JuliaError)
        rargs = [_js_runtime(mod, a).code for a in args]
        return _js_rt("new JuliaError(" * _js_string_literal(nameof(f)) * ", [" * join(rargs, ", ") *
                      "].join(\", \"))", f)
    end
    _js_distributions_law(f, args) &&
        _js_error(mod, "`", f, "` is part of a Distributions.jl law, which has no JavaScript translation. ",
                  "Write the law with ReactiveKernels' distribution stack (ReactiveKernelsDistributionKernels), ",
                  "as RK, RKPPL and BRM models do: for example `standard_student_t(nu).cdf(z)`, ",
                  "`student_t(nu, location, scale).logpdf(x)`, `normal`, `lognormal`, `exponential` or `gamma`")
    if !(f isa Core.Builtin) && !(f isa Type) && !_js_table_only(f)
        # A function becomes a named JavaScript function; a callable object
        # (RK's own operations, closures) is inlined where it is called.
        _js_named_function(f) && return _js_helper_call!(ir, f, args, T)
        _js_flush!(ir)
        v, _ = _js_inline!(ir.fn, f, args, "value")
        return v
    end
    !(f isa Core.Builtin) && any(_js_isirrational, args) && return _js_irrational_call!(ir, f, args)
    !(f isa Core.Builtin) && _js_bool_operands(args) && return _js_bool_call!(ir, f, args)
    if f === Core.kwcall && length(args) >= 2 && args[2] isa _JsVal && args[2].static
        types = join((_js_type(a) isa Type ? _js_type(a) : typeof(_js_type(a)) for a in args[3:end]), ", ")
        _js_error(mod, "`", args[2].value, "` with argument types (", types, ") and keywords `",
                  _js_type(args[1]), "` has no JavaScript translation")
    end
    types = join((_js_type(a) isa Type ? _js_type(a) : typeof(_js_type(a)) for a in args), ", ")
    _js_error(mod, "`", f, "` with argument types (", types, ") has no JavaScript translation")
end

# An irrational constant (`π`, `ℯ`, `IrrationalConstants.log2π`) is a value
# known while translating; its type is its value.
_js_isirrational(a) = a isa _JsVal && a.static && a.value isa AbstractIrrational

# A function of Julia's own with an irrational constant among its operands,
# where the table has no entry: Julia's method for these operands, translated
# from its typed code. Most promote the constant to the float it rounds to
# (`tail * π` multiplies by `Float64(π)`); comparisons round it up or down
# (`x < π` holds for `x == Float64(π)`), `==` is false, and some are exact
# (`ℯ^x` is `exp(x)`, `sin(π)` is `0.0`), as natively.
function _js_irrational_call!(ir::_JsIR, f, args::Vector{Any})
    mod = ir.fn.mod
    _js_flush!(ir)
    push!(mod.stack, "`" * string(f) * "(" * join(("::" * string(_js_argtype(a)) for a in args), ", ") * ")`")
    v, _ = _js_inline!(ir.fn, f, args, "value")
    pop!(mod.stack)
    v
end

# A `Bool` is a number in Julia: `true + true == 2`, `0.2 * true == 0.2`. A
# function of Julia's own with a `Bool` among numeric operands, where the table
# has no entry, is Julia's method for these operands, translated from its typed
# code. Those methods convert the `Bool` to `0` or `1` (`Int(x) + Int(y)`) or
# choose between the other operand and its zero (`false * y` is `0.0` even for
# an infinite or `NaN` `y`, and `false + -0.0` stays `-0.0`), as natively.
_js_bool_operands(args) = any(_js_isbool, args) && all(a -> _js_isnumber(a) || _js_isbool(a), args)

function _js_bool_call!(ir::_JsIR, f, args::Vector{Any})
    mod = ir.fn.mod
    where = "`" * string(f) * "(" * join(("::" * string(_js_argtype(a)) for a in args), ", ") * ")`"
    # a method that calls itself for the same operands has no translation
    where in mod.stack && _js_error(mod, "`", f, "` with argument types (",
                                    join((_js_argtype(a) for a in args), ", "), ") has no JavaScript translation")
    _js_flush!(ir)
    push!(mod.stack, where)
    v, _ = _js_inline!(ir.fn, f, args, "value")
    pop!(mod.stack)
    v
end

# A constructor or function of Distributions.jl, or a call on one of its
# objects. Its laws reach C libraries and StatsFuns' tails; the browser computes
# RK's own laws instead (`reactivekernels-use` §4k).
function _js_distributions_law(f, args)
    root(m::Module) = nameof(Base.moduleroot(m))
    owner(x) = x isa Type ? root(parentmodule(x)) :
        x isa Function && Base.issingletontype(typeof(x)) ? root(parentmodule(x)) : root(parentmodule(typeof(x)))
    owner(f) === :Distributions && return true
    any(a -> a isa _JsVal && a.static && !(a.value isa Union{Number,Nothing,String,Symbol,Type}) &&
             owner(a.value) === :Distributions, args)
end

# A function bound to its name in its module (not a closure or callable object).
function _js_named_function(f)
    f isa Function && Base.issingletontype(typeof(f)) || return false
    m, n = parentmodule(f), nameof(f)
    isdefined(m, n) && getglobal(m, n) === f
end

_js_argtype(a::_JsVal) = a.static && !(a.value isa Union{Number,Nothing,String,Symbol}) ?
    Core.Typeof(a.value) : (a.static ? typeof(a.value) : a.T)
_js_argtype(a) = _js_type(a)

# A value known while translating that a helper is specialized on instead of
# taking it as a parameter: a function, a type, or an irrational constant such
# as `π`, whose type is its value. Other numbers, strings and symbols are passed.
_js_specialized(a) = a isa _JsVal && a.static &&
    (!(a.value isa Union{Number,Nothing,String,Symbol}) || a.value isa AbstractIrrational)

# A plain Julia function: its typed code becomes a JavaScript function, once
# per argument types (and static arguments such as functions or types).
function _js_helper_call!(ir::_JsIR, f, args::Vector{Any}, T)
    mod = ir.fn.mod
    argtypes = Any[_js_argtype(a) for a in args]
    if any(a -> _js_vector_view(a) || _js_gather_view(a) || _js_isslice(a), args)
        # A view carries its parent and indices as translated values. Keep that
        # representation through the call; an ordinary SubArray parameter would
        # lose it. Inlining this call still retains the helper's runtime loops.
        context = "`" * string(f) * "(" * join(("::" * string(S) for S in argtypes), ", ") * ")`"
        context in mod.stack && _js_error(mod, "recursive helper with a view argument: ", context)
        push!(mod.stack, context)
        _js_flush!(ir)
        value, _ = _js_inline!(ir.fn, f, args, "value")
        pop!(mod.stack)
        return value
    end
    statics = Tuple(_js_specialized(a) ? a.value : nothing for a in args)
    key = (f, Tuple(argtypes), statics)
    name, RT = get(mod.helpers, key, ("", nothing))
    if isempty(name)
        name, RT = _js_helper!(mod, f, args, argtypes, key)
    end
    rargs = String[]
    for a in args
        _js_specialized(a) && continue
        push!(rargs, _js_runtime(mod, a).code)
    end
    pure = all(a -> _js_type(a) <: Union{Number,Bool}, args)
    _js_rt(name * "(" * join(rargs, ", ") * ")", RT; pure)
end

function _js_helper!(mod::_JsModule, f, args, argtypes, key)
    ci, RT, m, sparams = _js_typed(mod, f, argtypes)
    name = _js_unique!(mod.taken, nameof(f))
    mod.helpers[key] = (name, RT)
    push!(mod.stack, "`" * string(nameof(f)) * "(" * join(("::" * string(T) for T in argtypes), ", ") * ")`")
    fn = _JsFn(mod)
    nargs = Int(m.nargs)
    ir = _js_ir(fn, ci, nargs, sparams)
    ir.slots[1] = _js_static(f)
    params = String[]
    nfixed = m.isva ? nargs - 2 : nargs - 1
    for k in 1:nfixed
        a = args[k]
        if _js_specialized(a)
            ir.slots[k+1] = a
        else
            p = _js_unique!(fn.taken, ci.slotnames[k+1])
            push!(params, p)
            ir.slots[k+1] = _js_rt(p, argtypes[k]; ref = true)
        end
    end
    if m.isva
        items = Any[]
        for k in nfixed+1:length(args)
            a = args[k]
            if _js_specialized(a)
                push!(items, a)
            else
                p = _js_unique!(fn.taken, string(ci.slotnames[nargs]) * "_" * string(k - nfixed))
                push!(params, p)
                push!(items, _js_rt(p, argtypes[k]; ref = true))
            end
        end
        ir.slots[nargs] = _JsTuple(items, Tuple{argtypes[nfixed+1:end]...})
    end
    lines = String[]
    _js_declare_slots!(ir)
    _js_body!(ir)
    pop!(mod.stack)
    push!(mod.defs, "function " * name * "(" * join(params, ", ") * ") {\n" * join(fn.lines, "\n") * "\n}")
    (name, RT)
end

# Inline a callable's typed code into the current function, assigning its
# result to a fresh local `hint`.
function _js_inline!(fn::_JsFn, f, args::Vector{Any}, hint)
    mod = fn.mod
    argtypes = Any[_js_argtype(a) for a in args]
    ci, RT, m, sparams = _js_typed(mod, f, argtypes)
    nargs = Int(m.nargs)
    ir = _js_ir(fn, ci, nargs, sparams)
    ir.slots[1] = f isa _JsStruct ? f : _js_static(f)
    nfixed = m.isva ? nargs - 2 : nargs - 1
    for k in 1:nfixed
        ir.slots[k+1] = args[k]
    end
    m.isva && (ir.slots[nargs] = _JsTuple(Any[args[nfixed+1:end]...], Tuple{argtypes[nfixed+1:end]...}))
    cfg_nblocks = count(st -> st isa Union{Core.GotoNode,Core.GotoIfNot}, ci.code)
    if cfg_nblocks == 0 && count(st -> st isa Core.ReturnNode, ci.code) == 1
        # straight-line code: its return value is the result
        _js_declare_slots!(ir)
        _js_straight!(ir)
        v = only(ir.returned)
        return v, RT
    end
    name = _js_local!(fn, hint)
    label = _js_local!(fn, "R_" * name)
    _js_line!(fn, "let " * name * ";")
    _js_line!(fn, label * ": {")
    fn.indent += 1
    ir.result = name
    ir.exit_label = label
    _js_declare_slots!(ir)
    _js_body!(ir)
    fn.indent -= 1
    # The last jump out of the region is its end; a region left by no other
    # jump needs no label.
    strip(last(fn.lines)) == "break " * label * ";" && pop!(fn.lines)
    k = findlast(l -> strip(l) == label * ": {", fn.lines)
    any(l -> occursin("break " * label * ";", l), fn.lines[k+1:end]) ||
        (fn.lines[k] = replace(fn.lines[k], label * ": {" => "{"))
    _js_line!(fn, "}")
    _js_rt(name, RT; ref = true), RT
end

# Straight-line inlined code: statements in order, the return value kept as a value.
function _js_straight!(ir::_JsIR)
    ir.block_of = ones(Int, length(ir.ci.code))
    fill!(ir.use_block, 1)
    for (i, st) in enumerate(ir.ci.code)
        if st isa Core.ReturnNode
            v = _js_operand(ir, st.val)
            _js_flush!(ir)
            push!(ir.returned, v)
            return
        end
        _js_statement!(ir, i, st)
        _js_noreturn(ir.ci, i) && (_js_flush!(ir); push!(ir.returned, _js_static(nothing)); return)
    end
end

# --- broadcasting ------------------------------------------------------------------------------------

function _js_broadcasted(ir, f, args, T)
    fv = first(args)
    fv isa _JsVal && fv.static || _js_error(ir.fn.mod, "a broadcast of a runtime function value")
    _JsBroadcast(fv.value, Any[args[2:end]...], T)
end

# An array known while translating (a `const` global), not a range.
_js_static_array(x) = x isa AbstractArray && !(x isa AbstractRange)
_js_holds_array(x) = _js_static_array(x) || (x isa Base.RefValue && _js_holds_array(x[])) ||
    (x isa Base.Broadcast.Broadcasted && any(_js_holds_array, x.args))
# A broadcast over such an array stays a broadcast while translating, to be
# read from the module constant the array becomes; one over numbers and
# ranges alone folds (`(1:3) .+ 1` is the range `2:4`).
_js_lazy_broadcast(fv, args) =
    fv isa _JsVal && fv.static && (fv.value === Base.Broadcast.broadcasted || fv.value === _native_broadcasted) &&
    any(a -> a isa _JsVal && a.static && _js_holds_array(a.value), args)

# A broadcast's operands that are arrays known while translating are read like
# runtime arrays, from the module constants they become (`_js_runtime`, which
# refuses an array it cannot emit), and a broadcast known while translating
# (`2 .* 3` inside `x .+ 2 .* 3`) is fused like any other; numbers, ranges
# and `Ref`s stay as they are.
function _js_bcvalues(mod::_JsModule, v)
    v isa _JsBroadcast && return _JsBroadcast(v.f, Any[_js_bcvalues(mod, a) for a in v.args], v.T)
    v isa _JsVal && v.static || return v
    x = v.value
    x isa Base.Broadcast.Broadcasted &&
        return _js_bcvalues(mod, _JsBroadcast(x.f, Any[_js_static(a) for a in x.args], Nothing))
    _js_static_array(x) ? _js_runtime(mod, v) : v
end

function _js_leaves!(leaves, v)
    if v isa _JsBroadcast
        foreach(a -> _js_leaves!(leaves, a), v.args)
    else
        push!(leaves, v)
    end
    leaves
end

function _js_materialize!(ir::_JsIR, bc::_JsBroadcast, T; dest = nothing)
    mod = ir.fn.mod
    bc = _js_bcvalues(mod, bc)
    dest === nothing && _js_bc2d(bc) && return _js_materialize2!(ir, bc, T)
    leaves = _js_leaves!(Any[], bc)
    for l in leaves
        t = _js_type(l)
        (l isa _JsVal && (l.static || t <: Union{Number,Bool} || _js_isanyvector(l) || _js_isrange(l) ||
                          t <: Base.RefValue)) ||
            _js_error(mod, "a broadcast over a `", t, "` has no JavaScript translation (scalars, vectors and ranges are supported)")
    end
    arrays = filter(l -> l isa _JsVal && ((!l.static && _js_isanyvector(l)) || _js_isrange(l)), leaves)
    if isempty(arrays) && dest === nothing
        return _js_element!(ir, bc, Dict{Any,String}(), "")
    end
    # bind every array operand to a name (a range known while translating is
    # read as literals)
    names = Dict{Any,_JsVal}()
    for l in arrays
        haskey(names, l) && continue
        names[l] = _js_isrange(l) ? _js_range!(ir, l, "x") : _js_simple!(ir, l, "x")
    end
    lengths = unique([_js_isrange(l) ? _js_range_length(names[l]) : names[l].code * ".length" for l in arrays])
    dest === nothing || pushfirst!(lengths, dest.code * ".length")
    single = length(lengths) == 1
    n = _js_local!(ir.fn, "n")
    _js_line!(ir.fn, "const " * n * " = " * (single ? lengths[1] :
              _js_use!(mod, :_rk_bclen) * "(" * join(lengths, ", ") * ")") * ";")
    out = dest === nothing ? _js_local!(ir.fn, "out") : dest.code
    dest === nothing && _js_line!(ir.fn, "const " * out * " = new Array(" * n * ");")
    i = _js_local!(ir.fn, "i")
    _js_line!(ir.fn, "for (let " * i * " = 0; " * i * " < " * n * "; " * i * "++) {")
    ir.fn.indent += 1
    elements = Dict{Any,String}()
    for l in arrays
        a = names[l]
        # a range of length one is broadcast like a vector of length one
        elements[l] = _js_isrange(l) ?
            (single ? "(" * _js_range_start(a) * " + " * i * ")" :
             "(" * _js_range_start(a) * " + (" * _js_range_length(a) * " === 1 ? 0 : " * i * "))") :
            single ? a.code * "[" * i * "]" : _js_use!(mod, :_rk_bcat) * "(" * a.code * ", " * i * ")"
    end
    e = _js_element!(ir, bc, elements, i)
    _js_line!(ir.fn, out * "[" * i * "] = " * _js_runtime(mod, e).code * ";")
    ir.fn.indent -= 1
    _js_line!(ir.fn, "}")
    _js_rt(out, dest === nothing ? T : dest.T; ref = true)
end

function _js_element!(ir::_JsIR, v, elements, i)
    if v isa _JsBroadcast
        args = Any[_js_element!(ir, a, elements, i) for a in v.args]
        ET = _js_bc_eltype(v.f, args)
        isconcretetype(ET) || _js_error(ir.fn.mod, "the broadcast element type of `", v.f,
                                        "` is not concrete (", ET, ")")
        return _js_call!(ir, v.f, args, ET)
    elseif haskey(elements, v)
        T = _js_type(v)
        ET = _js_isrange(v) ? Int64 : eltype(T)
        return _js_rt(elements[v], ET; pure = true)
    elseif v isa _JsVal && v.T <: Base.RefValue && !v.static
        _js_error(ir.fn.mod, "a runtime `Ref` in a broadcast has no JavaScript translation")
    elseif v isa _JsVal && v.static && v.value isa Base.RefValue
        return _js_static(v.value[])
    end
    v
end

# The element type of a broadcast call. A field read by a name or position
# written in the source (`getproperty.(records, :a)`) has that field's type,
# as Julia's constant propagation finds natively; the call's types alone give
# the union of every field's type.
function _js_bc_eltype(f, args)
    if f in (getproperty, getfield) && length(args) == 2 && args[2] isa _JsVal && args[2].static
        S, key = _js_argtype(args[1]), args[2].value
        S isa DataType && isconcretetype(S) && S <: Union{Tuple,NamedTuple} &&
            ((key isa Symbol && S <: NamedTuple && hasfield(S, key)) || (key isa Int && 1 <= key <= fieldcount(S))) &&
            return fieldtype(S, key)
    end
    _js_return_type(f, Any[_js_argtype(a) for a in args])
end

# --- the call table ---------------------------------------------------------------------------------

function _js_infix(ir, op, prec, a, b, T; right_prec = prec + 1)
    mod = ir.fn.mod
    _js_rt(_js_code(mod, a, prec) * " " * op * " " * _js_code(mod, b, right_prec), T; prec)
end
_js_fcall(ir, name, args, T; pure = true) =
    _js_rt(name * "(" * join((_js_runtime(ir.fn.mod, a).code for a in args), ", ") * ")", T; pure)

const _JS_MATH = Dict{Any,String}(
    exp => "Math.exp", expm1 => "Math.expm1", sin => "Math.sin", cos => "Math.cos",
    tan => "Math.tan", sinh => "Math.sinh", cosh => "Math.cosh", tanh => "Math.tanh",
    asinh => "Math.asinh", cbrt => "Math.cbrt", abs => "Math.abs", sign => "Math.sign",
    floor => "Math.floor", ceil => "Math.ceil", trunc => "Math.trunc")
const _JS_CHECKED = Dict{Any,Symbol}(
    log => :_rk_log, log2 => :_rk_log2, log10 => :_rk_log10, log1p => :_rk_log1p,
    sqrt => :_rk_sqrt, asin => :_rk_asin, acos => :_rk_acos, acosh => :_rk_acosh,
    atanh => :_rk_atanh, round => :_rk_round)
const _JS_COMPARE = Dict{Any,String}(Base.:< => "<", Base.:<= => "<=", Base.:> => ">",
                                     Base.:>= => ">=", Base.:(==) => "===", Base.:!= => "!==")

function _js_function_call(ir::_JsIR, f::Function, args, T)
    mod = ir.fn.mod
    nums = all(a -> _js_isnumber(a) && !(_js_type(a) === Bool), args)
    if f === Base.indexed_iterate && length(args) in (2, 3) && _js_type(args[1]) <: Union{Tuple,NamedTuple} &&
       args[2] isa _JsVal && args[2].static && args[2].value isa Int
        # `a, b = t` for a tuple or named tuple: element `i`, then the next index.
        i = args[2].value
        n = fieldcount(_js_type(args[1]))
        1 <= i <= n || _js_error(mod, "destructuring element ", i, " of a `", _js_type(args[1]), "`")
        item = _js_function_call(ir, Core.getfield, Any[args[1], _js_static(i)], fieldtype(_js_type(args[1]), i))
        item === nothing && return nothing
        return _JsTuple(Any[item, _js_static(i + 1)], Tuple{_js_type(item),Int})
    elseif f === Base.indexed_iterate && length(args) in (2, 3) && args[1] isa _JsStruct &&
           args[1].T === Base.Generator && args[1].fields[2] isa _JsVal && args[1].fields[2].static &&
           all(a -> a isa _JsVal && a.static, args[2:end])
        # `a, b = (f(k) for k in 1:2)` over a range written in the source: the
        # generator's next element, a call of its function.
        g, r = args[1], args[1].fields[2].value
        y = length(args) == 2 ? iterate(r) : iterate(r, args[3].value)
        if y === nothing
            push!(mod.runtime, :JuliaError)
            return _js_rt("throw new JuliaError(\"BoundsError\", " *
                          _js_string_literal("attempt to access a generator of " * string(length(r)) *
                                             " elements at index [" * string(args[2].value) * "]") * ")",
                          Union{}; pure = false)
        end
        fn = g.fields[1]
        fn = fn isa _JsVal && fn.static ? fn.value : fn
        item = _js_call!(ir, fn, Any[_js_static(y[1])], _js_return_type(fn, Any[typeof(y[1])]))
        return _JsTuple(Any[item, _js_static(y[2])], Tuple{_js_type(item),typeof(y[2])})
    elseif f === Base.structdiff && length(args) == 2 && _js_type(args[1]) <: NamedTuple && T isa DataType &&
           T <: NamedTuple && isconcretetype(T)
        # A keyword sorter's keywords that its method does not declare.
        items = Any[_js_function_call(ir, Core.getfield, Any[args[1], _js_static(k)], fieldtype(T, k))
                    for k in fieldnames(T)]
        any(isnothing, items) && return nothing
        return _JsTuple(items, T)
    elseif f === pairs && length(args) == 1 && _js_type(args[1]) <: NamedTuple
        # read only through what inference knows of it (`isempty` in a keyword sorter)
        return _JsStruct(pairs, Any[args[1]], T)
    elseif f === isempty && length(args) == 1 && args[1] isa _JsStruct && args[1].T === pairs
        return _js_static(fieldcount(_js_type(args[1].fields[1])) == 0)
    elseif f === Base.kwerr && length(args) >= 2 && _js_type(args[1]) <: NamedTuple
        return _js_kwerr(mod, args)
    elseif f === getproperty && length(args) == 2 && _js_type(args[1]) <: Union{Tuple,NamedTuple} &&
           args[2] isa _JsVal && args[2].static && args[2].value isa Union{Symbol,Int}
        # `x.name` of a named tuple, as Base reads it: `getfield(x, name)`.
        return _js_function_call(ir, Core.getfield, args, T)
    elseif f === similar && length(args) == 1 && _js_isvector(args[1]) && T === _js_type(args[1])
        # The new vector's values are unspecified, as in Julia.
        return _js_rt("new Array(" * _js_code(mod, args[1], _JS_ATOM) * ".length)" *
                      (_js_isflatvector(args[1]) ? ".fill(" * (eltype(T) === Bool ? "false" : "0") * ")" : ""),
                      T; pure = false)
    elseif f === Base.literal_pow && length(args) == 3 && _js_isnumber(args[2]) && args[3] isa _JsVal &&
       args[3].static && args[3].value isa Val
        p = typeof(args[3].value).parameters[1]
        x = _js_simple!(ir, args[2])
        p == 0 && return _js_static(one(_js_type(x)))
        p == 1 && return x
        p == 2 && return _js_infix(ir, "*", _JS_MUL, x, x, T)
        p == 3 && return _js_rt(_js_code(mod, x, _JS_MUL) * " * " * _js_code(mod, x, _JS_MUL + 1) *
                                " * " * _js_code(mod, x, _JS_MUL + 1), T; prec = _JS_MUL)
        return _js_call!(ir, ^, Any[x, _js_static(p)], T)
    elseif f === (^) && length(args) == 2 && nums
        if _js_isfloat(args[1]) && _js_isint(args[2])
            return _js_rt(_js_code(mod, args[1], _JS_EXP + 2) * " ** " * _js_code(mod, args[2], _JS_EXP), T; prec = _JS_EXP)
        elseif _js_isint(args[1]) && _js_isint(args[2])
            return _js_fcall(ir, _js_use!(mod, :_rk_ipow), args, T)
        end
        return _js_fcall(ir, _js_use!(mod, :_rk_pow), args, T)
    elseif f in (+, *) && nums && length(args) >= 2
        op, prec = f === (+) ? ("+", _JS_ADD) : ("*", _JS_MUL)
        code = _js_code(mod, args[1], prec)
        for a in args[2:end]
            code *= " " * op * " " * _js_code(mod, a, prec + 1)
        end
        return _js_rt(code, T; prec)
    elseif f in (+, -) && length(args) == 2 && all(_js_isvector, args)
        x, y = _js_simple!(ir, args[1], "x"), _js_simple!(ir, args[2], "y")
        _js_flush!(ir)
        _js_line!(ir.fn, _js_use!(mod, :_rk_samelen) * "(" * x.code * ", " * y.code * ");")
        return _js_materialize!(ir, _JsBroadcast(f, Any[x, y], Nothing), T)
    elseif f === (-) && length(args) == 1 && _js_isvector(args[1])
        return _js_materialize!(ir, _JsBroadcast(f, Any[args[1]], Nothing), T)
    elseif f === (*) && length(args) == 2 && count(_js_isvector, args) == 1 &&
           count(a -> _js_isnumber(a) || _js_isbool(a), args) == 1
        return _js_materialize!(ir, _JsBroadcast(f, Any[args...], Nothing), T)
    elseif f === (/) && length(args) == 2 && _js_isvector(args[1]) && (_js_isnumber(args[2]) || _js_isbool(args[2]))
        return _js_materialize!(ir, _JsBroadcast(f, Any[args...], Nothing), T)
    elseif f === (-) && nums && length(args) == 2
        return _js_infix(ir, "-", _JS_ADD, args[1], args[2], T)
    elseif f === (-) && nums && length(args) == 1
        return _js_rt("-" * _js_code(mod, args[1], _JS_UNARY + 1), T; prec = _JS_UNARY)
    elseif f === (+) && nums && length(args) == 1
        return args[1]
    elseif f === (/) && nums && length(args) == 2
        return _js_infix(ir, "/", _JS_MUL, args[1], args[2], T)
    elseif f === inv && nums && length(args) == 1
        return _js_infix(ir, "/", _JS_MUL, _js_static(1.0), args[1], T)
    elseif haskey(_JS_COMPARE, f) && length(args) == 2 && all(a -> _js_isnumber(a), args)
        return _js_infix(ir, _JS_COMPARE[f], f in (==, !=) ? _JS_EQ : _JS_REL, args[1], args[2], T)
    elseif haskey(_JS_COMPARE, f) && length(args) == 2 && all(_js_isbool, args)
        return _js_infix(ir, _JS_COMPARE[f], f in (==, !=) ? _JS_EQ : _JS_REL, args[1], args[2], T)
    elseif f === (!) && length(args) == 1 && _js_isbool(args[1])
        return _js_not(mod, args[1])
    elseif f in (&, |) && length(args) == 2 && all(_js_isbool, args)
        return _js_infix(ir, f === (&) ? "&&" : "||", f === (&) ? _JS_AND : _JS_OR, args[1], args[2], Bool)
    elseif f === xor && length(args) == 2 && all(_js_isbool, args)
        return _js_infix(ir, "!==", _JS_EQ, args[1], args[2], Bool)
    elseif f === ifelse && length(args) == 3 && _js_isbool(args[1])
        return _js_rt(_js_code(mod, args[1], _JS_COND + 1) * " ? " * _js_code(mod, args[2], _JS_COND) *
                      " : " * _js_code(mod, args[3], _JS_COND), T; prec = _JS_COND)
    elseif f in (min, max) && nums && length(args) >= 2
        return _js_fcall(ir, f === min ? "Math.min" : "Math.max", args, T)
    elseif haskey(_JS_MATH, f) && nums && length(args) == 1
        return _js_fcall(ir, _JS_MATH[f], args, T)
    elseif haskey(_JS_CHECKED, f) && nums && length(args) == 1
        return _js_fcall(ir, _js_use!(mod, _JS_CHECKED[f]), args, T)
    elseif f in (floor, ceil, trunc, round) && length(args) == 2 && args[1] isa _JsVal &&
           args[1].static && args[1].value === Int64 && _js_isfloat(args[2])
        inner = _js_call!(ir, f, Any[args[2]], Float64)
        return _js_fcall(ir, _js_use!(mod, :_rk_int), Any[inner], Int64)
    elseif f === atan && nums && length(args) in (1, 2)
        return _js_fcall(ir, length(args) == 1 ? "Math.atan" : "Math.atan2", args, T)
    elseif f === hypot && nums && length(args) >= 2
        return _js_fcall(ir, "Math.hypot", args, T)
    elseif f === exp2 && nums && length(args) == 1
        return _js_rt("2 ** " * _js_code(mod, args[1], _JS_EXP), T; prec = _JS_EXP)
    elseif f === exp10 && nums && length(args) == 1
        return _js_rt("10 ** " * _js_code(mod, args[1], _JS_EXP), T; prec = _JS_EXP)
    elseif f === isnan && length(args) == 1 && _js_isnumber(args[1])
        return _js_fcall(ir, "Number.isNaN", args, Bool)
    elseif f === isfinite && length(args) == 1 && _js_isnumber(args[1])
        return _js_fcall(ir, "Number.isFinite", args, Bool)
    elseif f === isinf && length(args) == 1 && _js_isnumber(args[1])
        return _js_fcall(ir, _js_use!(mod, :_rk_isinf), args, Bool)
    elseif f in (div, ÷) && length(args) == 2 && all(_js_isint, args)
        return _js_fcall(ir, _js_use!(mod, :_rk_div), args, T)
    elseif f in (rem, %) && length(args) == 2 && all(_js_isint, args)
        return _js_fcall(ir, _js_use!(mod, :_rk_rem), args, T)
    elseif f === rem && length(args) == 2 && args[2] isa _JsVal && args[2].static && args[2].value === Int64 &&
           (_js_isint(args[1]) || _js_isbool(args[1]))
        # `x % Int64`, as Julia's integer arithmetic converts its operands
        return _js_convert(ir, Int64, args[1])
    elseif f === Base.not_sametype && length(args) == 2 && _js_type(args[1]) !== _js_type(args[2])
        # promotion changed an operand's type, so there is no promotion error
        return _js_static(nothing)
    elseif f in (rem, %) && length(args) == 2 && all(_js_isfloat, args)
        return _js_infix(ir, "%", _JS_MUL, args[1], args[2], T)
    elseif f === Base.mod && length(args) == 2 && all(_js_isint, args)
        return _js_fcall(ir, _js_use!(mod, :_rk_mod), args, T)
    elseif f === Base.mod && length(args) == 2 && all(_js_isfloat, args)
        return _js_fcall(ir, _js_use!(mod, :_rk_fmod), args, T)
    elseif f === muladd && nums && length(args) == 3
        return _js_rt(_js_code(mod, args[1], _JS_MUL) * " * " * _js_code(mod, args[2], _JS_MUL + 1) *
                      " + " * _js_code(mod, args[3], _JS_ADD + 1), T; prec = _JS_ADD)
    elseif f === clamp && nums && length(args) == 3
        x = _js_simple!(ir, args[1])
        lo, hi = _js_simple!(ir, args[2]), _js_simple!(ir, args[3])
        return _js_rt(_js_code(mod, x, _JS_REL + 1) * " > " * _js_code(mod, hi, _JS_REL + 1) * " ? " *
                      _js_code(mod, hi, _JS_COND) * " : " * _js_code(mod, x, _JS_REL + 1) * " < " *
                      _js_code(mod, lo, _JS_REL + 1) * " ? " * _js_code(mod, lo, _JS_COND) * " : " *
                      _js_code(mod, x, _JS_COND), T; prec = _JS_COND)
    elseif f === float && length(args) == 1 && (_js_isnumber(args[1]) || _js_isbool(args[1]))
        return _js_convert(ir, Float64, args[1])
    elseif f in (identity, real) && length(args) == 1 && (_js_isnumber(args[1]) || _js_isbool(args[1]))
        return args[1]
    elseif f === isinteger && length(args) == 1 && _js_isnumber(args[1])
        _js_isint(args[1]) && return _js_static(true)
        return _js_fcall(ir, "Number.isInteger", args, Bool)
    elseif f === copysign && length(args) == 2 && all(_js_isfloat, args)
        return _js_fcall(ir, _js_use!(mod, :_rk_copysign), args, T)
    elseif f === promote && !isempty(args) && all(a -> _js_isnumber(a) || _js_isbool(a) || _js_isirrational(a), args)
        # an irrational constant becomes the number it rounds to, as `convert` makes it
        PT = promote_type(map(_js_type, args)...)
        return _JsTuple(Any[_js_isirrational(a) ? _js_static(convert(PT, a.value)) : _js_convert(ir, PT, a)
                            for a in args], T)
    elseif f === Base.afoldl && length(args) >= 2 && args[1] isa _JsVal && args[1].static
        # `2 * π * r` is `*(2, π, r)`, whose method folds the operator from the left.
        op, y = args[1].value, args[2]
        for b in args[3:end]
            y = _js_call!(ir, op, Any[y, b], _js_return_type(op, Any[_js_argtype(y), _js_argtype(b)]))
        end
        return y
    elseif f === oftype && length(args) == 2 && _js_isnumber(args[1]) && args[2] isa _JsVal &&
           (_js_isnumber(args[2]) || (args[2].static && args[2].value isa Real))
        # `oftype(x, loghalf)`: an irrational constant becomes the number it rounds to.
        args[2].static && return _js_static(convert(_js_type(args[1]), args[2].value))
        return _js_convert(ir, _js_type(args[1]), args[2])
    elseif f === getindex && length(args) == 2 && _js_istuple(args[1]) && args[2] isa _JsVal &&
           args[2].static && args[2].value isa Int
        # A tuple read at a position known while translating, such as the
        # operation table of an RK derivative rule: a field read.
        return _js_function_call(ir, Core.getfield, args, T)
    elseif f === getindex && length(args) == 2 && _js_istuple(args[1]) && _js_isint(args[2]) &&
           _js_uniform_tuple(_js_type(args[1]), T)
        # A tuple of values of one type read at a position computed at run time: as an array.
        return _js_fcall(ir, _js_use!(mod, :_rk_at), Any[_js_runtime(mod, args[1]), args[2]], T)
    elseif f === evalpoly && length(args) == 2 && _js_isnumber(args[1]) && _js_istuple(args[2]) &&
           _js_tuple_length(args[2]) > 0 && !(_js_type(args[2]) <: NamedTuple)
        # Horner's rule with `muladd`, as Julia evaluates a polynomial of a real argument.
        x = _js_simple!(ir, args[1], "x")
        n = _js_tuple_length(args[2])
        coefficient(k) = _js_function_call(ir, Core.getfield, Any[args[2], _js_static(k)],
                                           fieldtype(_js_type(args[2]), k))
        acc = coefficient(n)
        for k in (n - 1):-1:1
            c = coefficient(k)
            all(_js_isnumber, (acc, c)) || return nothing
            acc = _js_call!(ir, muladd, Any[x, acc, c], T)
        end
        return acc
    elseif f === map && length(args) == 2 && args[1] isa _JsVal && args[1].static && _js_istuple(args[2]) &&
           !(_js_type(args[2]) <: NamedTuple) && T <: Tuple && isconcretetype(T)
        # `map(float, (a, b, x))`: a tuple of the function's values, element by element.
        S = _js_type(args[2])
        items = Any[_js_call!(ir, args[1].value,
                              Any[_js_function_call(ir, Core.getfield, Any[args[2], _js_static(k)], fieldtype(S, k))],
                              fieldtype(T, k)) for k in 1:_js_tuple_length(args[2])]
        return _JsTuple(items, T)
    elseif f === minmax && length(args) == 2 && nums
        x, y = _js_simple!(ir, args[1], "x"), _js_simple!(ir, args[2], "y")
        lo = _js_runtime(mod, _js_call!(ir, min, Any[x, y], fieldtype(T, 1))).code
        hi = _js_runtime(mod, _js_call!(ir, max, Any[x, y], fieldtype(T, 2))).code
        return _js_rt("[" * lo * ", " * hi * "]", T; pure = false)
    elseif f in (first, last) && length(args) == 1 && _js_istuple(args[1]) && _js_tuple_length(args[1]) > 0
        k = f === first ? 1 : _js_tuple_length(args[1])
        return _js_function_call(ir, Core.getfield, Any[args[1], _js_static(k)], T)
    elseif f in (iseven, isodd) && length(args) == 1 && _js_isint(args[1])
        return _js_rt(_js_code(mod, args[1], _JS_MUL) * " % 2 " * (f === iseven ? "===" : "!==") * " 0",
                      Bool; prec = _JS_EQ)
    elseif f in (iseven, isodd) && length(args) == 1 && _js_isfloat(args[1])
        return _js_fcall(ir, _js_use!(mod, f === iseven ? :_rk_iseven : :_rk_isodd), args, Bool)
    elseif f === Base.indexed_iterate && length(args) in (2, 3) && _js_istuple(args[1]) &&
           args[2] isa _JsVal && args[2].static && args[2].value isa Int
        # Destructuring `(a, b) = t`: the element and the next position.
        item = _js_function_call(ir, Core.getfield, Any[args[1], args[2]], fieldtype(T, 1))
        return _JsTuple(Any[item, _js_static(args[2].value + 1)], T)
    elseif f in (zero, one, oneunit) && length(args) == 1 && (_js_isnumber(args[1]) || _js_isbool(args[1]))
        return _js_static(f(_js_type(args[1])))
    elseif f === convert && length(args) == 2 && args[1] isa _JsVal && args[1].static &&
           (_js_isnumber(args[2]) || _js_isbool(args[2]))
        return _js_convert(ir, args[1].value, args[2])
    elseif f === length && length(args) == 1 &&
           (_js_isanyvector(args[1]) || (_js_istuple(args[1]) && !(_js_type(args[1]) <: NamedTuple)))
        return _js_rt(_js_code(mod, args[1], _JS_ATOM) * ".length", Int64)
    elseif f === length && length(args) == 1 && _js_isrange(args[1])
        r = _js_range!(ir, args[1])
        r.static && return _js_static(length(r.value))
        return _js_rt(r.code * ".stop - " * r.code * ".start + 1", Int64; prec = _JS_ADD)
    elseif f === size && length(args) == 2 && _js_isanyvector(args[1]) && _js_isint(args[2])
        d = args[2]
        d isa _JsVal && d.static && d.value == 1 && return _js_call!(ir, length, Any[args[1]], Int64)
    elseif f === getindex && length(args) == 2 && _js_isanyvector(args[1]) && _js_isint(args[2])
        return _js_fcall(ir, _js_use!(mod, :_rk_at), args, T)
    elseif f === getindex && length(args) == 2 && _js_isanyvector(args[1]) && _js_type(args[2]) === Vector{Int64}
        # `x[index]` for a vector of indices: a new vector, every index checked first.
        return _js_fcall(ir, _js_use!(mod, :_rk_gather), args, T; pure = false)
    elseif f === view && length(args) == 2 && _js_isanyvector(args[1]) && _js_isrange(args[2])
        # `@view x[a:b]`, read where it is used (`sum`, `length`, `getindex`).
        x, r = _js_simple!(ir, args[1], "x"), _js_range!(ir, args[2])
        _js_flush!(ir)
        _js_line!(ir.fn, _js_use!(mod, :_rk_window) * "(" * x.code * ", " * _js_range_start(r) * ", " *
                         _js_range_stop(r) * ");")
        return _JsStruct(view, Any[x, r], T)
    elseif f === sum && length(args) == 1 && args[1] isa _JsStruct && args[1].T === view &&
           eltype(_js_type(args[1].fields[1])) <: _JsNumber
        x, r = args[1].fields
        return _js_rt(_js_use!(mod, :_rk_sumrange) * "(" * x.code * ", " * _js_range_start(r) * ", " *
                      _js_range_stop(r) * ")", T)
    elseif f === length && length(args) == 1 && args[1] isa _JsStruct && args[1].T === view
        return _js_call!(ir, length, Any[args[1].fields[2]], Int64)
    elseif f === getindex && length(args) == 2 && args[1] isa _JsStruct && args[1].T === view && _js_isint(args[2])
        x, r = args[1].fields
        i = _js_simple!(ir, args[2], "i")
        _js_flush!(ir)
        _js_line!(ir.fn, "if (!(" * i.code * " >= 1 && " * i.code * " <= " * _js_range_length(r) * ")) throw new " *
                         _js_use!(mod, :JuliaError) * "(\"BoundsError\",\"attempt to access \" + " *
                         _js_range_length(r) * " + \"-element view at index [\" + " * i.code * " + \"]\");")
        return _js_rt(x.code * "[" * _js_range_start(r) * " + " * _js_code(mod, i, _JS_ADD + 1) * " - 2]", T)
    elseif f === setindex! && length(args) == 3 && _js_isanyvector(args[1]) && _js_isint(args[3])
        _js_isflatvector(args[1]) || _js_type(args[2]) === eltype(_js_type(args[1])) || return nothing
        return _js_fcall(ir, _js_use!(mod, :_rk_set), args, T; pure = false)
    elseif f === sum && length(args) == 1 && _js_isvector(args[1]) && eltype(_js_type(args[1])) <: _JsNumber
        return _js_fcall(ir, _js_use!(mod, :_rk_sum), args, T)
    elseif f === sum && length(args) == 1 && _js_isvector(args[1]) && eltype(_js_type(args[1])) === Bool
        # a sum of Booleans is a count
        return _js_rt(_js_use!(mod, :_rk_sum) * "(" * _js_runtime(mod, args[1]).code * ") + 0", T; prec = _JS_ADD)
    elseif f in (maximum, minimum) && length(args) == 1 && _js_isflatvector(args[1])
        # The error Julia throws for an empty collection, as this Julia throws it.
        empty = try
            f(eltype(_js_type(args[1]))[])
        catch e
            e
        end
        return _js_rt(_js_use!(mod, :_rk_extremum) * "(" * (f === maximum ? "Math.max" : "Math.min") *
                      ", " * _js_runtime(mod, args[1]).code * ", () => " *
                      _js_runtime(mod, _js_static(empty)).code * ")", T)
    elseif f in (eachindex, LinearIndices) && length(args) == 1 && _js_isanyvector(args[1])
        return _js_rt(_js_use!(mod, :_rk_range) * "(1, " * _js_code(mod, args[1], _JS_ATOM) * ".length)", T)
    elseif f === axes && length(args) == 2 && _js_isanyvector(args[1]) && args[2] isa _JsVal &&
           args[2].static && args[2].value == 1
        return _js_rt(_js_use!(mod, :_rk_range) * "(1, " * _js_code(mod, args[1], _JS_ATOM) * ".length)", T)
    elseif f === lastindex && length(args) == 1 && _js_isanyvector(args[1])
        return _js_rt(_js_code(mod, args[1], _JS_ATOM) * ".length", Int64)
    elseif f === firstindex && length(args) == 1 && _js_isanyvector(args[1])
        return _js_static(1)
    elseif f === (:) && length(args) == 2 && all(_js_isint, args)
        return _js_fcall(ir, _js_use!(mod, :_rk_range), args, UnitRange{Int64})
    elseif f === (:) && length(args) == 3 && all(_js_isint, args)
        if all(a -> a isa _JsVal && a.static, args)
            r = try
                (:)((a.value for a in args)...)
            catch
                nothing
            end
            r === nothing || return _js_static(r)
        end
        return _js_fcall(ir, _js_use!(mod, :_rk_steprange), args, StepRange{Int64,Int64})
    elseif f === iterate && length(args) in (1, 2) && _js_isanyvector(args[1]) &&
           (length(args) == 1 || _js_isint(args[2]))
        x = _js_simple!(ir, args[1], "x")
        s = length(args) == 1 ? _js_static(1) : _js_simple!(ir, args[2], "s")
        index = _js_runtime(mod, s).code
        return _js_rt(index * " > " * x.code * ".length ? null : [" * _js_use!(mod, :_rk_at) *
                      "(" * x.code * ", " * index * "), " * index * " + 1]", T; prec = _JS_COND)
    elseif f === iterate && length(args) in (1, 2) && _js_isstep(args[1])
        r = _js_range!(ir, args[1], "r")
        if length(args) == 1
            r.static && return _js_static(iterate(r.value))
            return _js_rt("(" * r.code * ".start !== " * r.code * ".stop && (" * r.code * ".step > 0) !== (" *
                          r.code * ".stop > " * r.code * ".start)) ? null : [" * r.code * ".start, " * r.code *
                          ".start]", T; prec = _JS_COND)
        end
        s = _js_simple!(ir, args[2], "s")
        stop, stp = r.static ? (_js_number(last(r.value)), _js_number(step(r.value))) :
            (r.code * ".stop", r.code * ".step")
        return _js_rt(s.code * " === " * stop * " ? null : [" * s.code * " + " * stp * ", " * s.code * " + " * stp *
                      "]", T; prec = _JS_COND)
    elseif f in (first, last, step, length, isempty) && length(args) == 1 && _js_isstep(args[1])
        r = _js_range!(ir, args[1], "r")
        r.static && return _js_static(f(r.value))
        f === first && return _js_rt(r.code * ".start", Int64)
        f === last && return _js_rt(r.code * ".stop", Int64)
        f === step && return _js_rt(r.code * ".step", Int64)
        empty = "(" * r.code * ".start !== " * r.code * ".stop && (" * r.code * ".step > 0) !== (" * r.code *
                ".stop > " * r.code * ".start))"
        f === isempty && return _js_rt(empty, Bool; prec = _JS_ATOM)
        return _js_rt(empty * " ? 0 : (" * r.code * ".stop - " * r.code * ".start) / " * r.code * ".step + 1",
                      Int64; prec = _JS_COND)
    elseif f === iterate && length(args) in (1, 2) && _js_isrange(args[1])
        r = _js_range!(ir, args[1])
        if length(args) == 1
            r.static && return _js_static(iterate(r.value))
            return _js_rt(r.code * ".stop >= " * r.code * ".start ? [" * r.code * ".start, " * r.code *
                          ".start] : null", T; prec = _JS_COND)
        end
        s = _js_simple!(ir, args[2], "s")
        return _js_rt(s.code * " < " * _js_range_stop(r) * " ? [" * s.code * " + 1, " * s.code * " + 1] : null",
                      T; prec = _JS_COND)
    elseif f in (first, last) && length(args) == 1 && _js_isrange(args[1])
        r = args[1]
        r isa _JsVal && r.static && return _js_static(f(r.value))
        return _js_rt(_js_code(mod, r, _JS_ATOM) * (f === first ? ".start" : ".stop"), Int64)
    elseif f === Base.Broadcast.broadcasted || f === _native_broadcasted
        return _js_broadcasted(ir, f, args, T)
    elseif f === Base.broadcast_preserving_zero_d && length(args) >= 2
        # Julia's method for a number and an array, `π * v`, `A * π` or `v / ℯ`,
        # is this broadcast; JavaScript has no zero-dimensional arrays to preserve.
        return _js_call!(ir, Base.Broadcast.materialize, Any[_js_broadcasted(ir, f, args, Nothing)], T)
    elseif (f === Base.Broadcast.materialize || f === _native_broadcast_materialize) &&
           length(args) == 1 && args[1] isa _JsBroadcast
        return _js_materialize!(ir, args[1], T)
    elseif f === Base.Broadcast.materialize! && length(args) == 2 && args[2] isa _JsBroadcast &&
           _js_isvector(args[1])
        _js_flush!(ir)
        return _js_materialize!(ir, args[2], T; dest = _js_simple!(ir, args[1], "dest"))
    elseif f in (zeros, ones) && length(args) in (1, 2) && _js_isint(args[end]) &&
           (length(args) == 1 || (args[1] isa _JsVal && args[1].static && args[1].value === Float64))
        return _js_rt("new Array(" * _js_runtime(mod, args[end]).code * ").fill(" *
                      (f === zeros ? "0" : "1") * ")", T; pure = false)
    elseif f === fill && length(args) == 2 && _js_isnumber(args[1]) && _js_isint(args[2])
        return _js_rt("new Array(" * _js_runtime(mod, args[2]).code * ").fill(" *
                      _js_runtime(mod, args[1]).code * ")", T; pure = false)
    elseif f === copy && length(args) == 1 && _js_isanyvector(args[1])
        return _js_rt(_js_code(mod, args[1], _JS_ATOM) * ".slice()", T; pure = false)
    elseif f === collect && length(args) == 1 && args[1] isa _JsStruct && args[1].T === Base.Generator
        g = args[1]
        return _js_materialize!(ir, _JsBroadcast(g.fields[1] isa _JsVal ? g.fields[1].value : g.fields[1],
                                                 Any[g.fields[2]], Nothing), T)
    elseif f === collect && length(args) == 1 && args[1] isa _JsVal && args[1].static &&
           args[1].value isa Base.Generator && _js_isrange(_js_static(args[1].value.iter))
        # `[f(k) for k in 1:5]` with nothing captured: a generator known while translating
        g = args[1].value
        return _js_materialize!(ir, _JsBroadcast(g.f, Any[_js_static(g.iter)], Nothing), T)
    elseif f === map && length(args) == 2 && (_js_isanyvector(args[2]) || _js_isrange(args[2]))
        return _js_materialize!(ir, _JsBroadcast(args[1] isa _JsVal ? args[1].value : args[1],
                                                 Any[args[2]], Nothing), T)
    elseif f === push! && length(args) == 2 && _js_isanyvector(args[1]) &&
           (_js_isflatvector(args[1]) ? _js_isnumber(args[2]) : _js_type(args[2]) === eltype(_js_type(args[1])))
        x = _js_simple!(ir, args[1], "x")
        _js_flush!(ir)
        _js_line!(ir.fn, x.code * ".push(" * _js_runtime(mod, args[2]).code * ");")
        return x
    elseif f === throw && length(args) == 1
        return _js_rt("throw " * _js_runtime(mod, args[1]).code, Union{}; pure = false)
    elseif f === error && all(a -> a isa _JsVal, args)
        push!(mod.runtime, :JuliaError)
        parts = join((_js_runtime(mod, a).code for a in args), " + ")
        return _js_rt("throw new JuliaError(\"ErrorException\", " *
                      (isempty(args) ? "\"\"" : "\"\" + " * parts) * ")", Union{}; pure = false)
    elseif f in (vcat, _native_vcat) && T isa Type && T <: Vector && _js_isvector_type(T) &&
           !isempty(args) && all(a -> _js_type(a) === eltype(T) || _js_type(a) === T, args)
        # a vector concatenated from vectors and numbers of its element type: a new array
        items = (_js_type(a) === T ? "..." * _js_code(mod, a, _JS_UNARY) : _js_runtime(mod, a).code for a in args)
        return _js_rt("[" * join(items, ", ") * "]", T; pure = false)
    elseif f === (*) && length(args) >= 2 && all(a -> a isa _JsVal && _js_type(a) <: AbstractString, args)
        # string concatenation, as an error message builds it
        return _js_rt(join((_js_code(mod, a, _JS_ADD + 1) for a in args), " + "), String; prec = _JS_ADD)
    elseif f === findfirst && length(args) == 2 && args[1] isa _JsVal && args[1].static && _js_istuple(args[2]) &&
           !(_js_type(args[2]) <: NamedTuple) && T == Union{Nothing,Int64}
        # The first position of a tuple whose element satisfies a predicate known
        # while translating (`findfirst(<(1e-3), ts)`), or `null` for `nothing`.
        pred = args[1].value
        S = _js_type(args[2])
        test(v) = pred isa Base.Fix2 ? _js_call!(ir, pred.f, Any[v, _js_static(pred.x)], Bool) :
                  pred isa Base.Fix1 ? _js_call!(ir, pred.f, Any[_js_static(pred.x), v], Bool) :
                  _js_call!(ir, pred, Any[v], Bool)
        code = "null"
        for k in _js_tuple_length(args[2]):-1:1
            v = _js_simple!(ir, _js_function_call(ir, Core.getfield, Any[args[2], _js_static(k)], fieldtype(S, k)), "e")
            c = test(v)
            _js_isbool(c) || return nothing
            code = _js_code(mod, c, _JS_COND + 1) * " ? " * string(k) * " : " * code
        end
        return _js_rt("(" * code * ")", T)
    elseif f === something && length(args) == 2 && Nothing <: _js_type(args[1]) && _js_isnumber(args[2]) &&
           T === _js_type(args[2])
        return _js_rt(_js_code(mod, args[1], _JS_OR + 1) * " ?? " * _js_code(mod, args[2], _JS_OR + 1), T;
                      prec = _JS_OR)
    elseif f === iszero && length(args) == 1 && _js_isnumber(args[1])
        return _js_rt(_js_code(mod, args[1], _JS_EQ + 1) * " === 0", Bool; prec = _JS_EQ)
    elseif f in (cumprod, cumsum) && length(args) == 1 &&
           ((T == _js_type(args[1]) &&
             ((_js_istuple(args[1]) && _js_number_tuple(T, eltype(T))) ||
              (_js_isflatvector(args[1]) && eltype(T) <: _JsNumber))) ||
            (_js_vector_view(args[1]) && T === Vector{eltype(_js_type(args[1]))} && eltype(T) <: _JsNumber))
        # Running products or sums preserve the first value, including -0.0.
        helper = _js_use!(mod, f === cumprod ? :_rk_cumprod : :_rk_cumsum)
        if _js_vector_view(args[1])
            x, r = args[1].fields
            start = "(" * _js_range_start(r) * " - 1)"
            return _js_rt(helper * "(" * _js_runtime(mod, x).code * ", " * start *
                          ", " * start * " + " * _js_range_length(r) * ")", T; pure = false)
        end
        return _js_rt(helper * "(" * _js_runtime(mod, args[1]).code * ")", T)
    elseif f === ntuple && length(args) == 2 && args[2] isa _JsVal && args[2].static && args[2].value isa Val &&
           T <: Tuple && isconcretetype(T)
        # `ntuple(k -> c[k] * x, Val(21))`: the function at each position, a
        # tuple known in length while translating.
        n = typeof(args[2].value).parameters[1]
        n isa Int && n == fieldcount(T) || return nothing
        items = String[]
        for k in 1:n
            if args[1] isa _JsStruct
                _js_flush!(ir)
                push!(items, _js_runtime(mod, first(_js_inline!(ir.fn, args[1], Any[_js_static(k)], "e"))).code)
            else
                args[1] isa _JsVal && args[1].static || return nothing
                push!(items, _js_runtime(mod, _js_call!(ir, args[1].value, Any[_js_static(k)], fieldtype(T, k))).code)
            end
        end
        # An array, computed here: its elements are locals of this block.
        return _js_rt("[" * join(items, ", ") * "]", T; pure = false)
    elseif f === string && all(a -> a isa _JsVal, args)
        # JavaScript numbers erase the distinction between Int64 and Float64.
        # Select the float formatter while the Julia argument type is known.
        pieces = ["\"\""]
        for a in args
            code = _js_isfloat(a) ? _js_use!(mod, :_rk_float_string) * "(" * _js_runtime(mod, a).code * ")" :
                   _js_code(mod, a, _JS_ADD + 1)
            push!(pieces, code)
        end
        return _js_rt(join(pieces, " + "), String;
                      prec = _JS_ADD)
    elseif f === _native_exp && length(args) == 1 && _js_isnumber(args[1])
        return _js_call!(ir, exp, Any[args[1]], T)
    elseif f === _native_mul && all(a -> _js_isnumber(a), args)
        return _js_call!(ir, *, Any[args...], T)
    elseif f === _runtime_check && length(args) == 2 && _js_isbool(args[1])
        e = _js_runtime(mod, args[2]).code
        c = _js_code(mod, args[1], _JS_UNARY + 1)
        _js_flush!(ir)
        _js_line!(ir.fn, "if (!" * c * ") throw " * e * ";")
        return _js_static(nothing)
    end
    nothing
end

# Core builtins and intrinsics
const _JS_INTRINSIC_INFIX = Dict{Any,Tuple{String,Int}}(
    Core.Intrinsics.add_float => ("+", _JS_ADD), Core.Intrinsics.sub_float => ("-", _JS_ADD),
    Core.Intrinsics.mul_float => ("*", _JS_MUL), Core.Intrinsics.div_float => ("/", _JS_MUL),
    Core.Intrinsics.add_int => ("+", _JS_ADD), Core.Intrinsics.sub_int => ("-", _JS_ADD),
    Core.Intrinsics.lt_float => ("<", _JS_REL), Core.Intrinsics.le_float => ("<=", _JS_REL),
    Core.Intrinsics.eq_float => ("===", _JS_EQ), Core.Intrinsics.ne_float => ("!==", _JS_EQ),
    Core.Intrinsics.slt_int => ("<", _JS_REL), Core.Intrinsics.sle_int => ("<=", _JS_REL))
function _js_function_call(ir::_JsIR, f::Core.IntrinsicFunction, args, T)
    if f === Core.Intrinsics.not_int && length(args) == 1 && _js_isbool(args[1])
        return _js_not(ir.fn.mod, args[1])
    elseif f === Core.Intrinsics.neg_float && length(args) == 1 && _js_isfloat(args[1])
        return _js_rt("-" * _js_code(ir.fn.mod, args[1], _JS_UNARY + 1), Float64; prec = _JS_UNARY)
    elseif haskey(_JS_INTRINSIC_INFIX, f) && length(args) == 2 && all(_js_isnumber, args) &&
           (endswith(string(f), "_float") ? all(_js_isfloat, args) : all(_js_isint, args))
        op, prec = _JS_INTRINSIC_INFIX[f]
        return _js_infix(ir, op, prec, args[1], args[2], T)
    end
    nothing
end
# Whether a method exists for a signature known while translating, as RK's
# guarded native calls ask (`src/core.jl`): native inference folds it to a
# constant, and here it is answered in the world that translates.
function _js_function_call(ir::_JsIR, f::typeof(Core._hasmethod), args, T)
    length(args) == 1 && args[1] isa _JsVal && args[1].static || return nothing
    _js_static(Core._hasmethod(args[1].value))
end

function _js_function_call(ir::_JsIR, f::Core.Builtin, args, T)
    mod = ir.fn.mod
    if f === Core.getfield || f === getfield
        x, k = args[1], args[2]
        k isa _JsVal && k.static || _js_error(mod, "a field read with a runtime field index")
        key = k.value
        if x isa _JsTuple
            idx = key isa Symbol ? findfirst(==(key), fieldnames(x.T)) : key
            return x.items[idx]
        elseif x isa _JsStruct && x.T isa DataType
            idx = key isa Symbol ? findfirst(==(key), fieldnames(x.T)) : key
            return x.fields[idx]
        elseif x isa _JsVal && x.static
            return _js_static(getfield(x.value, key))
        elseif x isa _JsVal && x.T <: Tuple
            return _js_rt(_js_code(mod, x, _JS_ATOM) * "[" * string(key - 1) * "]", T)
        elseif x isa _JsVal && x.T <: NamedTuple
            name = key isa Symbol ? key : fieldnames(x.T)[key]
            return _js_rt(_js_code(mod, x, _JS_ATOM) * "." * _js_identifier(name), T)
        elseif x isa _JsVal && x.T <: Union{Nothing,Tuple}
            return _js_rt(_js_code(mod, x, _JS_ATOM) * "[" * string(key - 1) * "]", T)
        end
        _js_error(mod, "reading the field `", key, "` of a `", _js_type(x), "` has no JavaScript translation")
    elseif f === Core.tuple || f === tuple
        return _JsTuple(Any[args...], T)
    elseif f === (===)
        a, b = args
        if a isa _JsVal && a.static && b isa _JsVal && b.static
            return _js_static(a.value === b.value)
        end
        (a isa _JsVal && a.static && a.value === nothing) && ((a, b) = (b, a))
        if b isa _JsVal && b.static && b.value === nothing
            return _js_rt(_js_code(mod, a, _JS_EQ) * " === null", Bool; prec = _JS_EQ)
        end
        all(_js_isfloat, args) && return _js_fcall(ir, "Object.is", args, Bool)
        all(a -> _js_isint(a) || _js_isbool(a), args) && return _js_infix(ir, "===", _JS_EQ, a, b, Bool)
    elseif f === Core.ifelse
        return _js_function_call(ir, ifelse, args, T)
    elseif f === Core.typeassert
        return args[1]
    elseif f === isa
        x, t = args
        t isa _JsVal && t.static || _js_error(mod, "`isa` with a runtime type")
        S = _js_type(x)
        S <: t.value && return _js_static(true)
        typeintersect(S, t.value) === Union{} && return _js_static(false)
        if S isa Union && Nothing <: S
            t.value === Nothing && return _js_rt(_js_code(mod, x, _JS_EQ) * " === null", Bool; prec = _JS_EQ)
            return _js_rt(_js_code(mod, x, _JS_EQ) * " !== null", Bool; prec = _JS_EQ)
        end
    elseif f === Core.throw
        return _js_rt("throw " * _js_runtime(mod, args[1]).code, Union{}; pure = false)
    elseif f === Core._apply_iterate && length(args) >= 2 && args[1] isa _JsVal && args[1].static &&
           args[1].value === iterate && args[2] isa _JsVal && args[2].static &&
           all(a -> a isa _JsTuple || (a isa _JsVal && a.static && a.value isa Tuple) ||
                    (a isa _JsVal && !a.static && a.T <: Tuple && !(a.T <: NamedTuple) && isconcretetype(a.T)), args[3:end])
        items = Any[]
        for a in args[3:end]
            if a isa _JsTuple
                append!(items, a.items)
            elseif a.static
                append!(items, (_js_static(x) for x in a.value))
            else
                # a tuple computed at run time: its elements, read from one name
                t = _js_simple!(ir, a, "t")
                append!(items, (_js_rt(t.code * "[" * string(k - 1) * "]", fieldtype(a.T, k); ref = true)
                                for k in 1:fieldcount(a.T)))
            end
        end
        return _js_call!(ir, args[2].value, items, T)
    elseif f === Core.apply_type && all(a -> a isa _JsVal && a.static, args)
        return _js_static(Core.apply_type((a.value for a in args)...))
    end
    nothing
end

function _js_convert(ir, T, v)
    S = _js_type(v)
    S === T && return v
    if S === Bool && T in (Float64, Int64)
        # `0` or `1`, as `Int(x)` and `Float64(x)` give it
        v isa _JsVal && v.static && return _js_static(T(v.value))
        return _js_rt(_js_code(ir.fn.mod, v, _JS_COND + 1) * " ? 1 : 0", T; prec = _JS_COND)
    end
    T === Float64 && S <: _JsNumber && return _js_rt(v.code, Float64; prec = v.prec, ref = v.ref, pure = v.pure)
    T === Int64 && S === Float64 && return _js_fcall(ir, _js_use!(ir.fn.mod, :_rk_int), Any[v], Int64)
    _js_error(ir.fn.mod, "converting a `", S, "` to `", T, "` has no JavaScript translation")
end

# Number constructors (`Float64(x)`, `Int(x)`) are calls of the type.
function _js_function_call(ir::_JsIR, f::Type, args, T)
    if f === LazyString && all(a -> a isa _JsVal, args)
        # An exception message built lazily: a string, as `string` builds one.
        return _js_rt("\"\" + " * join((_js_code(ir.fn.mod, a, _JS_ADD + 1) for a in args), " + "), LazyString;
                      prec = _JS_ADD)
    end
    if f in (Float64, Int64) && length(args) == 1 && (_js_isnumber(args[1]) || _js_isbool(args[1]))
        return _js_convert(ir, f, args[1])
    elseif f in (Float64, Float32) && length(args) in (1, 2) && _js_isirrational(args[1]) &&
           (length(args) == 1 || (args[2] isa _JsVal && args[2].static && args[2].value isa RoundingMode))
        # An irrational constant rounded to a float (`Float64(x, RoundUp)` in
        # Julia's comparisons with it): the same float for every input.
        return _js_static(f((a.value for a in args)...))
    elseif f === Base.OneTo && length(args) == 1 && _js_isint(args[1])
        return _js_rt(_js_use!(ir.fn.mod, :_rk_range) * "(1, " * _js_runtime(ir.fn.mod, args[1]).code * ")", T)
    elseif f === Base.Generator && length(args) == 2 && (_js_isanyvector(args[2]) || _js_isrange(args[2]))
        return _JsStruct(f, Any[args...], T)
    elseif f <: Vector && _js_isvector_type(f) && length(args) == 2 && args[1] isa _JsVal &&
           args[1].static && args[1].value === undef && _js_isint(args[2])
        # `Vector{T}(undef, n)`: the values are unspecified, as in Julia.
        n = _js_simple!(ir, args[2], "n")
        _js_flush!(ir)
        _js_use!(ir.fn.mod, :JuliaError)
        _js_line!(ir.fn, "if (" * n.code * " < 0) throw new JuliaError(\"ArgumentError\", \"invalid Array dimensions\");")
        return _js_rt("new Array(" * n.code * ")" *
                      (eltype(f) <: _JsMatrixElement ? ".fill(" * (eltype(f) === Bool ? "false" : "0") * ")" : ""),
                      f; pure = false)
    elseif f in (Matrix{Float64}, Matrix{Int64}, Matrix{Bool}) && length(args) == 3 && args[1] isa _JsVal &&
           args[1].static && args[1].value === undef && _js_isint(args[2]) && _js_isint(args[3])
        # `Matrix{T}(undef, m, n)`: a row-array matrix whose values are unspecified, as in Julia.
        return _js_fcall(ir, _js_use!(ir.fn.mod, :_rk_mfill),
                         Any[_js_static(f === Matrix{Bool} ? false : 0), args[2], args[3]], f; pure = false)
    elseif f <: NamedTuple && length(args) <= 1 && T isa DataType && T <: NamedTuple && isconcretetype(T)
        # `(; a, b)` and the keywords of `f(x; a, b)`: `NamedTuple{names}(tuple)`.
        items = isempty(args) ? Any[] : _js_tuple_items(ir, args[1])
        (items === nothing || length(items) != fieldcount(T)) && return nothing
        for (k, v) in pairs(items)
            S = fieldtype(T, k)
            _js_type(v) === S && continue
            v isa _JsVal || return nothing
            items[k] = v.static ? _js_static(convert(S, v.value)) : _js_convert(ir, S, v)
        end
        return _JsTuple(items, T)
    end
    nothing
end

# The elements of a tuple value, or `nothing` for a value that is not a tuple.
function _js_tuple_items(ir::_JsIR, x)
    _js_type(x) <: Tuple || return nothing
    x isa _JsTuple && return copy(x.items)
    x isa _JsVal || return nothing
    x.static && return Any[_js_static(e) for e in x.value]
    fieldcount(x.T) == 0 && return Any[]
    t = _js_simple!(ir, x, "t")
    Any[_js_rt(t.code * "[" * string(k - 1) * "]", fieldtype(x.T, k)) for k in 1:fieldcount(x.T)]
end

# --- the module -----------------------------------------------------------------------------------------

function _js_input_check(mod::_JsModule, name, T)
    if T === Float64
        "_rk_number(" * _js_string_literal(name) * ", " * name * ", false);", :_rk_number
    elseif T === Int64
        "_rk_number(" * _js_string_literal(name) * ", " * name * ", true);", :_rk_number
    elseif T === Bool
        "_rk_boolean(" * _js_string_literal(name) * ", " * name * ");", :_rk_boolean
    elseif _js_isvector_type(T)
        E, depth = eltype(T), 1
        while _js_isvector_type(E)
            E, depth = eltype(E), depth + 1
        end
        el = E === Bool ? "boolean" : "number"
        "_rk_vector(" * _js_string_literal(name) * ", " * name * ", \"" * el * "\"" *
            (depth > 1 ? ", " * string(E === Int64) * ", " * string(depth) : E === Int64 ? ", true" : "") * ");", :_rk_vector
    else
        throw(JSEmissionError("the input `$name` has type `$T`; JavaScript inputs are Float64, Int64, Bool or nested vectors of them"))
    end
end

_js_noop() = nothing

# The returned value of a WANT: a value, an array for a tuple of names, an
# object for a named tuple (nested as given).
_js_shape(w::Symbol, out) = out[w]
_js_shape(w::Tuple, out) = "[" * join((_js_shape(x, out) for x in w), ", ") * "]"
function _js_shape(w::NamedTuple, out)
    fields = [(k === v && _js_identifier(k) == out[v] ? out[v] :
               (_js_identifier(k) == string(k) ? string(k) : _js_string_literal(k)) * ": " * _js_shape(v, out))
              for (k, v) in pairs(w)]
    "{ " * join(fields, ", ") * " }"
end

# The kernel function of a plan. An exported one checks its inputs; an inner
# one (a prepared kernel a recipe calls) is called with translated values.
function _js_plan_function!(mod::_JsModule, fname::String, p::Plan, argtypes, shape;
                            exported::Bool = true)
    fn = _JsFn(mod)
    values = Dict{ValueId,Any}()
    params = String[]
    for (v, T) in zip(p.have, argtypes)
        name = _js_local!(fn, v.name)
        push!(params, name)
        if exported
            check, helper = _js_input_check(mod, name, T)
            push!(mod.runtime, helper)
            _js_line!(fn, check)
        end
        values[canon_id(p.graph, v.id)] = _js_rt(name, T; ref = true)
    end
    wants = _js_plan_body!(fn, p, values)
    out = Dict(v.name => _js_runtime(mod, w).code for (v, w) in zip(p.want, wants))
    _js_line!(fn, "return " * _js_shape(shape, out) * ";")
    code = (exported ? "export function " : "function ") * fname * "(" * join(params, ", ") * ") {\n" *
           join(fn.lines, "\n") * "\n}"
    code, Any[_js_value_type(w) for w in wants]
end

# The WANT's shape as a JavaScript literal (`RETURNS`): a name, an array of
# shapes for a tuple, an object of shapes for a named tuple.
_js_returns(w::Symbol) = _js_string_literal(w)
_js_returns(w::Tuple) = "[" * join(map(_js_returns, w), ", ") * "]"
_js_returns(w::NamedTuple) =
    "{ " * join((_js_string_literal(k) * ": " * _js_returns(v) for (k, v) in pairs(w)), ", ") * " }"

# The recipes of a plan, in order, in the current function; `values` holds its
# HAVE values. Returns its WANT values.
function _js_plan_body!(fn::_JsFn, p::Plan, values::Dict{ValueId,Any})
    mod = fn.mod
    kir = _js_ir(fn, only(Base.code_lowered(_js_noop, Tuple{})), 0)
    # The plan's input checks, each as soon as its arguments are known.
    checks = collect(p.checks)
    function run_checks!()
        for c in copy(checks)
            all(a -> !(a isa Value) || haskey(values, canon_id(p.graph, a.id)), c.args) || continue
            args = Any[a isa Value ? values[canon_id(p.graph, a.id)] :
                       _js_bound(mod, a, "check_arg_" * string(i))
                       for (i, a) in enumerate(c.args)]
            v = _js_call!(kir, c.f, args, Nothing)
            _js_flush!(kir)
            # A check translated as a function (a package's named check, such
            # as RKPPL's packed-length check) is a call to run here; an
            # inlined one has already emitted its code.
            v isa _JsVal && !v.static && !v.ref && _js_line!(fn, v.code * ";")
            deleteat!(checks, findfirst(x -> x === c, checks))
        end
    end
    run_checks!()
    for recipe in p.recipes
        op = recipe.op
        where = recipe.source === _NO_KERNEL_SOURCE ? "" : string(Base.remove_linenums!(deepcopy(recipe.source)))
        outs = join((string(v.name) for v in recipe.outputs), ", ")
        push!(mod.stack, "recipe `" * outs * (isempty(where) ? "" : " = " * where) * "`")
        inputs = Any[values[canon_id(p.graph, v.id)] for v in recipe.inputs]
        hint = length(recipe.outputs) == 1 ? string(only(recipe.outputs).name) : "outputs"
        start = length(fn.lines)
        before = copy(fn.taken)
        result, RT = if op isa _BoundConstant
            # bound data: known while translating, emitted where read at run time
            _js_bound(mod, op.value, hint), typeof(op.value)
        elseif op isa _KernelSourceOp
            _js_inline!(fn, getfield(op, :f), inputs, hint)
        elseif op isa _AuthoredPlateOp
            _js_plate!(fn, op, inputs, hint)
        elseif op isa _AuthoredScanOp
            _js_scan!(fn, op, inputs, hint)
        else
            # A recipe whose operation is a function itself (`y = f(x)`).
            RT = Core.Compiler.return_type(op, Tuple{(_js_argtype(a) for a in inputs)...})
            isconcretetype(RT) || _js_error(mod, "the result type of `", op, "` is not concrete (", RT, ")")
            v = _js_call!(kir, op, inputs, RT)
            _js_flush!(kir)
            v, RT
        end
        if length(recipe.outputs) == 1
            v = only(recipe.outputs)
            if result isa _JsVal && !result.static && !result.ref
                name = _js_local!(fn, v.name)
                _js_line!(fn, "const " * name * " = " * result.code * ";")
                result = _js_rt(name, result.T; ref = true)
            elseif result isa _JsVal && result.ref && !result.static && !(result.code in before) &&
                   result.code in fn.taken && result.code != _js_identifier(v.name)
                # a local this recipe computed: name it after its value
                name = _js_local!(fn, v.name)
                for k in start+1:length(fn.lines)
                    fn.lines[k] = _js_rename(fn.lines[k], result.code, name)
                end
                result = _js_rt(name, result.T; ref = true)
            end
            values[canon_id(p.graph, v.id)] = result
        else
            for (k, v) in enumerate(recipe.outputs)
                item = result isa _JsTuple ? result.items[k] :
                    _js_rt(_js_code(mod, result, _JS_ATOM) * "[" * string(k - 1) * "]", fieldtype(RT, k))
                values[canon_id(p.graph, v.id)] = item
            end
        end
        run_checks!()
        pop!(mod.stack)
    end
    isempty(checks) || _js_error(mod, "an input check of `", first(checks).f, "` reads a value the plan does not compute")
    Any[values[canon_id(p.graph, v.id)] for v in p.want]
end

# The type a sub-plan's value has, for a WANT value of a cell or step.
_js_value_type(v) = v isa _JsVal ? (v.static ? typeof(v.value) : v.T) : _js_type(v)

# A module-level name `prefix * hint`, unique in the module. Function locals
# never start with `_`, so they cannot shadow it.
function _js_module_name!(mod::_JsModule, prefix, hint)
    base = prefix * _js_identifier(hint)
    name, k = base, 1
    while name in mod.taken
        k += 1
        name = base * "_" * string(k)
    end
    push!(mod.taken, name)
    name
end

# An array known while translating as a module-level constant `prefix * hint`,
# emitted once per array: a vector is an array, a matrix an array of row
# arrays (one without rows keeps its column count as `cols`), as values cross
# the module's boundary. Its numbers are exact (`Float64` literals round-trip,
# an `Int64` beyond 2^53 is refused). `frozen` freezes it and its rows.
function _js_constant(mod::_JsModule, x::Union{Vector,BitVector,Matrix}, prefix, hint;
                      frozen::Bool)
    key = (Symbol(prefix), objectid(x))
    name, _ = get(mod.helpers, key, ("", nothing))
    if isempty(name)
        name = _js_module_name!(mod, prefix, hint)
        wrap(code) = frozen ? "Object.freeze(" * code * ")" : code
        element(e) = e isa AbstractVector ? _js_constant(mod, e, prefix, string(hint, "_item"); frozen).code :
                                            _js_runtime(mod, _js_static(e)).code
        items(xs) = "[" * join((element(e) for e in xs), ", ") * "]"
        code = x isa AbstractVector ? wrap(items(x)) :
            size(x, 1) == 0 ? wrap("Object.assign([], { cols: " * string(size(x, 2)) * " })") :
            wrap("[" * join((wrap(items(r)) for r in eachrow(x)), ", ") * "]")
        push!(mod.defs, "const " * name * " = " * code * ";")
        mod.helpers[key] = (name, x)  # the array stays alive, so its id stays unique
    end
    _js_rt(name, typeof(x); ref = true)
end

# A value bound at preparation (`prepare(...; bound)`, an RKPPL binding's
# data): numerical values stay in module storage, like the arrays. They are
# runtime operands of the executable body, not specialization parameters.
function _js_bound(mod::_JsModule, x::Union{Float64,Int64,Bool,UnitRange{Int64},Base.OneTo{Int64}}, hint)
    name = _js_module_name!(mod, "_rk_bound_", hint)
    code = _js_runtime(mod, _js_static(x)).code
    x isa AbstractRange && (code = "Object.freeze(" * code * ")")
    push!(mod.defs, "const " * name * " = " * code * ";")
    _js_rt(name, typeof(x); ref = true)
end
_js_bound(mod::_JsModule, x::Union{Tuple,NamedTuple}, hint) =
    _JsTuple(Any[_js_bound(mod, v, string(hint, "_", k)) for (k, v) in pairs(x)], typeof(x))
_js_bound(mod::_JsModule, x::Union{Vector,BitVector,Matrix}, hint) =
    _js_isvector_type(typeof(x)) || (x isa Matrix && eltype(x) <: _JsMatrixElement) ?
    _js_constant(mod, x, "_rk_bound_", hint; frozen = true) : _js_static(x)
# Another bound value stays known while translating: a call reads its fields
# or folds it, and a run-time use of it raises as for any other such value.
_js_bound(mod::_JsModule, x, hint) = _js_static(x)

# A prepared kernel called as a function (a native model's prepared query
# that a recipe calls, such as an RKPPL `prepare_query`): a module-level
# function translated from the kernel's plan, its bound data included, once
# per kernel and argument types, as `jax_source` translates one. Its result is
# the kernel's: the value of a single WANT, else a tuple of them.
function _js_function_call(ir::_JsIR, k::PreparedKernel, args, T)
    mod = ir.fn.mod
    p = k.plan
    length(p.have) == length(args) ||
        _js_error(mod, "a prepared kernel of ", length(p.have), " inputs called with ", length(args))
    argtypes = Any[_js_argtype(a) for a in args]
    for (v, A) in zip(p.have, argtypes)
        A isa Type && A <: valtype(v) || _js_error(mod, "a prepared kernel's input `", v.name,
            "` (a `", valtype(v), "`) given a `", A, "`")
    end
    key = (:prepared, objectid(k), Tuple(argtypes))
    name, RT = get(mod.helpers, key, ("", nothing))
    if isempty(name)
        hint = length(p.want) == 1 ? only(p.want).name : :kernel
        name = _js_module_name!(mod, "_rk_prepared_", hint)
        shape = length(p.want) == 1 ? only(p.want).name : Tuple(v.name for v in p.want)
        push!(mod.stack, "prepared kernel `" * string(hint) * "`")
        code, out_types = _js_plan_function!(mod, name, p, argtypes, shape; exported = false)
        pop!(mod.stack)
        push!(mod.defs, code)
        RT = length(out_types) == 1 ? only(out_types) : Tuple{out_types...}
        mod.helpers[key] = (name, RT)
        # the kernel stays alive, so its id stays unique
        mod.helpers[(:prepared_kernel, objectid(k))] = ("", k)
    end
    rargs = String[_js_runtime(mod, a).code for a in args]
    _js_flush!(ir)
    _js_rt(name * "(" * join(rargs, ", ") * ")", RT; pure = false)
end

# An authored plate: a loop over its cells, each running the cell plan. A
# mapped argument is read at the cell's position (one of length 1 at every
# position, as broadcasting reads it); a shared argument is read whole.
function _js_plate!(fn::_JsFn, op::_AuthoredPlateOp{K,A}, inputs, hint) where {K,A}
    mod = fn.mod
    body = op.kernel.plan
    length(body.want) == 1 || _js_error(mod, "a plate whose cell computes several values has no JavaScript translation yet")
    mapped = Bool[!(i in A) && (_js_plate_vector(v) || _js_isrange(v)) for (i, v) in enumerate(inputs)]
    for (i, v) in enumerate(inputs)
        !(i in A) && !mapped[i] && !(_js_type(v) <: Union{Number,Bool}) &&
            _js_error(mod, "a plate over a `", _js_type(v), "` has no JavaScript translation (vectors and ranges are)")
    end
    any(mapped) || _js_error(mod, "a plate without a vector argument")
    kir = _js_ir(fn, only(Base.code_lowered(_js_noop, Tuple{})), 0)
    args = Any[!mapped[i] ? inputs[i] : _js_isrange(inputs[i]) ? _js_range!(kir, inputs[i], "x") :
               _js_simple!(kir, inputs[i], "x") for i in eachindex(inputs)]
    lengths = unique([_js_plate_length(a) for (i, a) in enumerate(args) if mapped[i]])
    single = length(lengths) == 1
    n = _js_local!(fn, "n")
    _js_line!(fn, "const " * n * " = " * (single ? lengths[1] :
              _js_use!(mod, :_rk_bclen) * "(" * join(lengths, ", ") * ")") * ";")
    out = _js_local!(fn, hint)
    _js_line!(fn, "const " * out * " = new Array(" * n * ");")
    i = _js_local!(fn, "i")
    _js_line!(fn, "for (let " * i * " = 0; " * i * " < " * n * "; " * i * "++) {")
    fn.indent += 1
    cell = Dict{ValueId,Any}()
    for (k, (v, a)) in enumerate(zip(body.have, args))
        cell[canon_id(body.graph, v.id)] = if !mapped[k]
            T = _js_type(a)
            a isa _JsVal && T <: Base.RefValue && a.static ? _js_static(a.value[]) : a
        elseif _js_gather_view(a)
            index = single ? i : "(" * _js_plate_length(a) * " === 1 ? 0 : " * i * ")"
            _js_call!(kir, getindex, Any[a, _js_rt("(" * index * " + 1)", Int64)], eltype(_js_type(a)))
        elseif _js_isrange(a) || _js_vector_view(a)
            index = single ? i : "(" * _js_plate_length(a) * " === 1 ? 0 : " * i * ")"
            if _js_isrange(a)
                _js_rt("(" * _js_range_start(a) * " + " * index * ")", Int64; prec = _JS_ATOM)
            else
                # The view's creation checked its window. Read its parent at
                # the cell position, retaining aliases without copying data.
                x, r = a.fields
                _js_rt(x.code * "[" * _js_range_start(r) * " - 1 + " * index * "]",
                    eltype(_js_type(a)))
            end
        else
            name = _js_local!(fn, v.name)
            el = single ? a.code * "[" * i * "]" : _js_use!(mod, :_rk_bcat) * "(" * a.code * ", " * i * ")"
            _js_line!(fn, "const " * name * " = " * el * ";")
            _js_rt(name, eltype(_js_type(a)); ref = true)
        end
    end
    w = only(_js_plan_body!(fn, body, cell))
    _js_line!(fn, out * "[" * i * "] = " * _js_runtime(mod, w).code * ";")
    fn.indent -= 1
    _js_line!(fn, "}")
    T = Vector{_js_value_type(w)}
    _js_rt(out, T; ref = true), T
end

# An authored scan: a loop over its sequences carrying the step's state; the
# step plan's WANT values are the next carry and the output collected.
function _js_scan!(fn::_JsFn, op::_AuthoredScanOp{K,A,I,H}, inputs, hint) where {K,A,I,H}
    mod = fn.mod
    H && _js_error(mod, "a scan with `history` has no JavaScript translation yet")
    body = op.kernel.plan
    length(body.want) == 2 || _js_error(mod, "a scan step without a (carry, output) pair")
    iterated = [i for i in 2:length(inputs) if !(i in A)]
    shared = [i for i in 2:length(inputs) if i in A]
    for i in iterated
        _js_isvector(inputs[i]) || _js_isrange(inputs[i]) ||
            _js_error(mod, "a scan over a `", _js_type(inputs[i]), "` has no JavaScript translation (vectors and ranges are)")
    end
    kir = _js_ir(fn, only(Base.code_lowered(_js_noop, Tuple{})), 0)
    seqs = Any[_js_isrange(inputs[i]) ? _js_range!(kir, inputs[i], "x") : _js_simple!(kir, inputs[i], "x")
               for i in iterated]
    lens = [_js_isrange(a) ? _js_range_length(a) : a.code * ".length" for a in seqs]
    n = _js_local!(fn, "n")
    _js_line!(fn, "const " * n * " = " * lens[1] * ";")
    for l in lens[2:end]
        _js_line!(fn, "if (" * l * " !== " * n * ") throw new JuliaError(\"DimensionMismatch\", " *
                      "\"all inputs to eachindex must have the same indices, got \" + " * n * " + \" and \" + " * l * ");")
        push!(mod.runtime, :JuliaError)
    end
    carry = _js_local!(fn, "carry")
    init = _js_runtime(mod, inputs[1])
    _js_line!(fn, "let " * carry * " = " * init.code * ";")
    out = _js_local!(fn, hint)
    _js_line!(fn, "const " * out * " = new Array(" * n * (I ? " + 1" : "") * ");")
    I && _js_line!(fn, out * "[0] = " * carry * ";")
    i = _js_local!(fn, "i")
    _js_line!(fn, "for (let " * i * " = 0; " * i * " < " * n * "; " * i * "++) {")
    fn.indent += 1
    step = Dict{ValueId,Any}()
    step[canon_id(body.graph, body.have[1].id)] = _js_rt(carry, _js_value_type(inputs[1]); ref = true)
    for (k, a) in enumerate(seqs)
        v = body.have[1+k]
        step[canon_id(body.graph, v.id)] = if _js_isrange(a)
            _js_rt("(" * _js_range_start(a) * " + " * i * ")", Int64; prec = _JS_ATOM)
        else
            name = _js_local!(fn, v.name)
            _js_line!(fn, "const " * name * " = " * a.code * "[" * i * "];")
            _js_rt(name, eltype(_js_type(a)); ref = true)
        end
    end
    for (k, j) in enumerate(shared)
        a = inputs[j]
        a = a isa _JsVal && a.static && a.value isa Base.RefValue ? _js_static(a.value[]) : a
        step[canon_id(body.graph, body.have[1+length(seqs)+k].id)] = a
    end
    next, output = _js_plan_body!(fn, body, step)
    o = _js_runtime(mod, output)
    _js_line!(fn, out * "[" * i * (I ? " + 1" : "") * "] = " * o.code * ";")
    _js_line!(fn, carry * " = " * _js_runtime(mod, next).code * ";")
    fn.indent -= 1
    _js_line!(fn, "}")
    T = Vector{_js_value_type(output)}
    _js_rt(out, T; ref = true), T
end

_js_typename(T) = string(T)

# Rename an identifier outside the string literals of a line.
function _js_rename(line, from, to)
    re = Regex("\\b" * from * "\\b")
    parts = String[]
    last = 1
    for m in eachmatch(r"\"(?:[^\"\\\\]|\\\\.)*\"", line)
        push!(parts, replace(line[last:prevind(line, m.offset)], re => to), m.match)
        last = m.offset + ncodeunits(m.match)
    end
    push!(parts, replace(line[last:end], re => to))
    join(parts)
end

"""
    js_source(spec::KernelSpec; have, want, types = nothing, name = :kernel) -> String

Emit a self-contained JavaScript (ES) module computing `want` from `have`,
for a browser or Node.js. Prototype; not exported.

The module exports the function `name`, which takes the HAVE values in order
and returns the WANT value: a single value for a name, an array for a tuple
of names, an object for a named tuple (`want = (share = :share, depth = :d)`).
It also exports `INPUTS` and `OUTPUTS` (names and Julia types), `RETURNS`
(the WANT's shape: a name, an array of shapes or an object of shapes), and
`JuliaError`, the class of the errors it throws; `error.name` is the Julia
exception type (`DomainError`, `BoundsError`, `DimensionMismatch`, …).
`types` gives each HAVE value's type, by name, where its port has none.

Values are `Float64` and `Int64` numbers (an `Int64` exact while it stays
below 2^53 in magnitude), `Bool`s, nested vectors of them (including empty
and ragged groups), and matrices of scalar numbers or Booleans as
arrays of row arrays (`m[i][j]` is Julia's `m[i+1, j+1]`; a matrix without
rows keeps its column count as `cols`, `Object.assign([], { cols: 3 })` for
`zeros(0, 3)`); sizes and numeric inputs stay runtime arguments. A `const`
global array of them that the code reads (`const LEVELS = [50.0, 100.0]`)
becomes a module-level array (`_rk_const_LEVELS`) holding its value at
emission, which the module reads and writes as the kernel does the global.
Translated:
ordinary kernel recipes, broadcasts (fused, with Julia's length rules),
`plate` and `scan` as loops, arithmetic and comparisons, `Base.Math`
functions with Julia's domain errors, checked 1-based indexing (of a matrix:
`m[i, j]`, `m[:, j]`, `m[i, :]`, `size`, `axes`), vector literals and
matrix literals of numbers (`[a b; c d]`), `sum`/`prod`/`maximum`/`minimum`,
ranges, comprehensions and `map` over vectors, including a vector of tuples or
named tuples that a recipe builds (`[k(d) for d in dose]` with a prepared
kernel `k` whose WANT is `(; a, b)`), read, indexed, iterated and projected
(`[r.a for r in records]`) like other vectors but never an input; `cumsum`/`cumprod` of numeric
vectors, contiguous vector views or tuples; tuples and named tuples
(objects) built, read and destructured, `all`/`any`/`count` of a tuple,
keyword calls, and `rk_ode_rk45_tol`, integrated by the vendored `@reside-ic/dopri`
Dormand–Prince 5(4) method with the native bridge's arguments, step budget
and `ODESolveFailure`s (values agree with native to the solvers' accuracy).
Matrix arithmetic: `*` of matrices, vectors, scalars and the transposes
`m'`/`transpose(m)`/`v'` (a chain `A * B * v` included); `+`/`-` of
matrices; `sum`/`prod`/`maximum`/`minimum` of a matrix, whole or with
`dims = 1` or `2`; `vec` (a fresh vector in column order); `permutedims`;
`zeros`/`ones`/`fill(x, m, n)`, `Matrix{T}(undef, m, n)` and writes
`m[i, j] = x`, `m[:, j] = v`, `m[i, :] = v`; `diag`; slices `view(m, :, j)`
and `view(m, i, :)` read in place; and broadcasts over matrices, columns and
rows with Julia's shape rules; `m[indices, j]` gathers a column's indexed rows.
`reshape(vector, rows, cols)`, including a vector view, shares
storage with the parent. Its dimensions must be nonnegative: Julia can
construct a `ReshapedArray` view with negative axes, which JavaScript's
array representation cannot hold and explicitly refuses.
Plain Julia helpers are translated from their
typed code, branches, early returns and loops included, without annotation;
helpers taking views retain the view's parent and index expressions by inlining
the call, including its loops (recursive calls with views are refused).
Julia's own functions translate only through RK's table. A construct without
a translation (a C call, `fma`, `\` and factorizations, indexing a transposed
matrix, a scan with `history`, `rk_ode_bdf_tol`) raises
`ReactiveKernels.JSEmissionError` naming it and its call chain. Elementary
functions use JavaScript's `Math` and products sum in their own order, so
results may differ from Julia's in the last bits.
"""
function js_source(spec::KernelSpec; have = _KERNEL_DEFAULT_BOUNDARY,
                   want = _KERNEL_DEFAULT_BOUNDARY, types = nothing, name = :kernel)
    p = plan(spec; have, want = _kernel_want_leaves(want))
    js_source(p; types, name, shape = want)
end

function js_source(p::Plan; types = nothing, name = :kernel,
                   shape = length(p.want) == 1 ? only(p.want).name : Tuple(v.name for v in p.want))
    argtypes = _js_have_types(p, types)
    mod = _JsModule()
    _js_identifier(name) == string(name) ||
        throw(ArgumentError("js_source: `$name` is not a JavaScript function name; pass another `name`"))
    fname = _js_unique!(mod.taken, name)
    main, out_types = _js_plan_function!(mod, fname, p, argtypes, shape)
    io = IOBuffer()
    println(io, "// Generated by ReactiveKernels.js_source from the kernel `", name, "`.")
    println(io, "// Inputs: ", join((string(v.name) * "::" * _js_typename(T) for (v, T) in zip(p.have, argtypes)), ", "),
            ". Outputs: ", join((string(v.name) for v in p.want), ", "), ".")
    println(io)
    push!(mod.runtime, :JuliaError)
    for s in _js_runtime_closure(mod)
        println(io, _JS_RUNTIME[s][1])
    end
    for d in mod.defs
        println(io)
        println(io, d)
    end
    println(io)
    println(io, main)
    println(io)
    println(io, "export const INPUTS = [", join(("{ name: " * _js_string_literal(v.name) * ", type: " *
                                                   _js_string_literal(_js_typename(T)) * " }"
                                                   for (v, T) in zip(p.have, argtypes)), ", "), "];")
    println(io, "export const OUTPUTS = [", join(("{ name: " * _js_string_literal(v.name) * ", type: " *
                                                    _js_string_literal(_js_typename(T)) * " }"
                                                    for (v, T) in zip(p.want, out_types)), ", "), "];")
    println(io, "export const RETURNS = ", _js_returns(shape), ";")
    String(take!(io))
end

"""
    js_source(kernel; types = nothing, name = :kernel) -> String

The module of a prepared kernel as [`prepare`](@ref) returns it (a flat or
shaped WANT), such as a native model's prepared query: its function takes the
kernel's HAVE values in order and returns what the kernel returns. Data bound
at preparation (`prepare(spec; have, want, bound)`, the data of an RKPPL
binding) are part of the module, numbers as literals and vectors and
matrices as frozen module constants, so the module computes what the kernel
computes. `types` defaults to the exemplar types the kernel was prepared
with, if any. A recipe of another kernel may call such a kernel too; its
function is then translated from the kernel's plan the same way.
"""
function js_source(k::PreparedKernel; types = nothing, name = :kernel)
    S = _exact_signature(k)
    types === nothing && S !== Nothing && (types = Tuple(fieldtypes(S)))
    js_source(k.plan; types, name)
end

function js_source(k::_ShapedKernel{S}; types = nothing, name = :kernel) where {S}
    target = k.target
    target isa PreparedKernel || throw(ArgumentError(
        "js_source: this kernel fills keyword arguments or defaults of its signature; " *
        "prepare it with an explicit `have` to emit it"))
    p = target.plan
    shape = _js_want_names(S, p)
    T = _exact_signature(target)
    types === nothing && T !== Nothing && (types = Tuple(fieldtypes(T)))
    js_source(p; types, name, shape)
end

# A shaped kernel's shape (positions in its flat WANT) as names of that WANT.
_js_want_names(position::Int, p::Plan) = p.want[position].name
_js_want_names(shape::Tuple, p::Plan) = map(x -> _js_want_names(x, p), shape)
_js_want_names(shape::NamedTuple, p::Plan) = map(x -> _js_want_names(x, p), shape)

function _js_have_types(p::Plan, types)
    names = [v.name for v in p.have]
    argtypes = if types === nothing
        Any[valtype(v) for v in p.have]
    elseif types isa NamedTuple
        Any[haskey(types, n) ? types[n] : valtype(v) for (n, v) in zip(names, p.have)]
    else
        length(types) == length(p.have) || throw(ArgumentError(
            "js_source: $(length(types)) types for $(length(p.have)) HAVE values"))
        Any[types...]
    end
    for (n, T) in zip(names, argtypes)
        isconcretetype(T) || throw(ArgumentError(
            "js_source: the HAVE value `$n` has type `$T`; pass a concrete type with `types`"))
    end
    argtypes
end
