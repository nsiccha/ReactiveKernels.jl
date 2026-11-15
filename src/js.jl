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
# inputs stay runtime arguments; nothing is specialized on a value.

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
    elseif x isa Tuple && all(e -> e isa Union{Number,Nothing,String,Symbol}, x)
        items = [_js_runtime(mod, _js_static(e)).code for e in x]
        return _js_rt("[" * join(items, ", ") * "]", typeof(x); ref = false)
    elseif _js_isrange(v)
        # A range known while translating (`1:60`), where a value is needed:
        # the runtime range, as `a:b` builds it.
        return _js_rt(_js_use!(mod, :_rk_range) * "(" * _js_number(first(x)) * ", " * _js_number(last(x)) * ")",
                      typeof(x); ref = false)
    elseif x isa Exception
        return _js_rt(_js_exception(mod, x), typeof(x); pure = true)
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
    :_rk_at => ("""
function _rk_at(x, i) {
  if (!(Number.isInteger(i) && i >= 1 && i <= x.length))
    throw new JuliaError("BoundsError", "attempt to access " + x.length + "-element array at index [" + i + "]");
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
function _rk_vector(name, x, element) {
  if (!(Array.isArray(x) || ArrayBuffer.isView(x)))
    throw new JuliaError("ArgumentError", "the input `" + name + "` must be an array; got " + x);
  for (let i = 0; i < x.length; i++)
    if (typeof x[i] !== element || (element === "number" && arguments[3] && !Number.isInteger(x[i])))
      throw new JuliaError("ArgumentError", "the input `" + name + "` must hold " + element + "s; entry " + (i + 1) + " is " + x[i]);
}""", [:JuliaError]),
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
_js_isnumber(v) = _js_type(v) <: _JsNumber
_js_isfloat(v) = _js_type(v) === Float64
_js_isint(v) = _js_type(v) === Int64
_js_isbool(v) = _js_type(v) === Bool
_js_isvector(v) = _js_type(v) <: Vector && eltype(_js_type(v)) <: Union{_JsNumber,Bool}
_js_isrange(v) = _js_type(v) in (UnitRange{Int64}, Base.OneTo{Int64})

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
    m = which(tt)
    for st in ci.code
        st isa Expr && st.head === :foreigncall || continue
        name = st.args[1]
        name isa QuoteNode && (name = name.value)
        _js_error(mod, "`", f isa _JsStruct ? f.T : f, "` calls the C function `", name,
                  "`, which has no JavaScript translation")
    end
    (ci, rt, m)
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
        return v
    elseif x isa GlobalRef
        isdefined(x.mod, x.name) || _js_error(mod, "`", x, "` is not defined")
        isconst(x.mod, x.name) ||
            _js_error(mod, "the global `", x, "` is not constant; pass its value as a kernel input")
        return _js_static(getglobal(x.mod, x.name))
    elseif x isa QuoteNode
        return _js_static(x.value)
    elseif x isa Expr
        _js_error(mod, "the expression `", x, "` as an operand has no JavaScript translation")
    else
        return _js_static(x)
    end
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
    if v isa _JsVal && !v.static && !v.ref
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
        if v isa _JsVal && !v.static && (!v.ref || (ir.multiblock && !(v.code in ir.stable)))
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
    else
        r = _js_runtime(ir.fn.mod, v)
        _js_flush!(ir)
        _js_line!(ir.fn, name * " = " * r.code * ";")
        ir.slots[n] = _js_rt(name, Core.Compiler.widenconst(ir.ci.slottypes[n]); ref = true)
    end
end

function _js_expr!(ir::_JsIR, i, ex, T)
    mod = ir.fn.mod
    ex isa Expr || return _js_operand(ir, ex)
    if ex.head === :call
        fv = _js_operand(ir, ex.args[1])
        args = Any[_js_operand(ir, a) for a in ex.args[2:end]]
        if T isa Core.Const && _js_foldable(fv, args)
            return _js_static(T.val)
        end
        folded = _js_fold(fv, args)
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
        _js_error(mod, "a static parameter whose value inference does not know")
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

# A call inference folded to a constant, without an effect to keep.
function _js_foldable(fv, args)
    fv isa _JsVal && fv.static || return false
    f = fv.value
    f isa Core.Builtin && return true
    f isa Type && return true
    all(a -> a isa _JsVal && a.static, args) || return false
    effects = try
        Base.infer_effects(f, Tuple{(Core.Typeof(a.value) for a in args)...})
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
        !ismutable(vals[1])
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

function _js_ir(fn::_JsFn, ci::Core.CodeInfo, nargs::Int)
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
          nothing, "", Any[])
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

function _js_call!(ir::_JsIR, f, args::Vector{Any}, T)
    mod = ir.fn.mod
    v = _js_function_call(ir, f, args, T)
    v === nothing || return v
    if f isa Type && f <: Exception
        push!(mod.runtime, :JuliaError)
        rargs = [_js_runtime(mod, a).code for a in args]
        return _js_rt("new JuliaError(" * _js_string_literal(nameof(f)) * ", [" * join(rargs, ", ") *
                      "].join(\", \"))", f)
    end
    if !(f isa Core.Builtin) && !(f isa Type) && !_js_table_only(f)
        # A function becomes a named JavaScript function; a callable object
        # (RK's own operations, closures) is inlined where it is called.
        _js_named_function(f) && return _js_helper_call!(ir, f, args, T)
        _js_flush!(ir)
        v, _ = _js_inline!(ir.fn, f, args, "value")
        return v
    end
    types = join((_js_type(a) isa Type ? _js_type(a) : typeof(_js_type(a)) for a in args), ", ")
    _js_error(mod, "`", f, "` with argument types (", types, ") has no JavaScript translation")
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

# A plain Julia function: its typed code becomes a JavaScript function, once
# per argument types (and static arguments such as functions or types).
function _js_helper_call!(ir::_JsIR, f, args::Vector{Any}, T)
    mod = ir.fn.mod
    argtypes = Any[_js_argtype(a) for a in args]
    statics = Tuple(a isa _JsVal && a.static && !(a.value isa Union{Number,Nothing,String,Symbol}) ?
                    a.value : nothing for a in args)
    key = (f, Tuple(argtypes), statics)
    name, RT = get(mod.helpers, key, ("", nothing))
    if isempty(name)
        name, RT = _js_helper!(mod, f, args, argtypes, key)
    end
    rargs = String[]
    for a in args
        a isa _JsVal && a.static && !(a.value isa Union{Number,Nothing,String,Symbol}) && continue
        push!(rargs, _js_runtime(mod, a).code)
    end
    pure = all(a -> _js_type(a) <: Union{Number,Bool}, args)
    _js_rt(name * "(" * join(rargs, ", ") * ")", RT; pure)
end

function _js_helper!(mod::_JsModule, f, args, argtypes, key)
    ci, RT, m = _js_typed(mod, f, argtypes)
    name = _js_unique!(mod.taken, nameof(f))
    mod.helpers[key] = (name, RT)
    push!(mod.stack, "`" * string(nameof(f)) * "(" * join(("::" * string(T) for T in argtypes), ", ") * ")`")
    fn = _JsFn(mod)
    nargs = Int(m.nargs)
    ir = _js_ir(fn, ci, nargs)
    ir.slots[1] = _js_static(f)
    params = String[]
    nfixed = m.isva ? nargs - 2 : nargs - 1
    for k in 1:nfixed
        a = args[k]
        if a isa _JsVal && a.static && !(a.value isa Union{Number,Nothing,String,Symbol})
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
            if a isa _JsVal && a.static && !(a.value isa Union{Number,Nothing,String,Symbol})
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
    ci, RT, m = _js_typed(mod, f, argtypes)
    nargs = Int(m.nargs)
    ir = _js_ir(fn, ci, nargs)
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
    leaves = _js_leaves!(Any[], bc)
    for l in leaves
        t = _js_type(l)
        (l isa _JsVal && (l.static || t <: Union{Number,Bool} || _js_isvector(l) || _js_isrange(l) ||
                          t <: Base.RefValue)) ||
            _js_error(mod, "a broadcast over a `", t, "` has no JavaScript translation (scalars, vectors and ranges are supported)")
    end
    arrays = filter(l -> l isa _JsVal && ((!l.static && _js_isvector(l)) || _js_isrange(l)), leaves)
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
        elements[l] = _js_isrange(l) ? "(" * _js_range_start(a) * " + " * i * ")" :
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
        ET = _js_return_type(v.f, Any[_js_argtype(a) for a in args])
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
    if f === Base.literal_pow && length(args) == 3 && _js_isnumber(args[2]) && args[3] isa _JsVal &&
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
    elseif f in (+, -) && length(args) == 2 && all(_js_isvector, args) && all(a -> eltype(_js_type(a)) <: _JsNumber, args)
        x, y = _js_simple!(ir, args[1], "x"), _js_simple!(ir, args[2], "y")
        _js_flush!(ir)
        _js_line!(ir.fn, _js_use!(mod, :_rk_samelen) * "(" * x.code * ", " * y.code * ");")
        return _js_materialize!(ir, _JsBroadcast(f, Any[x, y], Nothing), T)
    elseif f === (-) && length(args) == 1 && _js_isvector(args[1]) && eltype(_js_type(args[1])) <: _JsNumber
        return _js_materialize!(ir, _JsBroadcast(f, Any[args[1]], Nothing), T)
    elseif f === (*) && length(args) == 2 && count(_js_isvector, args) == 1 && count(_js_isnumber, args) == 1
        return _js_materialize!(ir, _JsBroadcast(f, Any[args...], Nothing), T)
    elseif f === (/) && length(args) == 2 && _js_isvector(args[1]) && _js_isnumber(args[2])
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
    elseif haskey(_JS_CHECKED, f) && nums && length(args) == 1 && _js_isfloat(args[1])
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
    elseif f in (float, identity) && length(args) == 1 && _js_isnumber(args[1])
        return args[1]
    elseif f in (zero, one) && length(args) == 1 && _js_isnumber(args[1])
        return _js_static(f(_js_type(args[1])))
    elseif f === convert && length(args) == 2 && args[1] isa _JsVal && args[1].static &&
           _js_isnumber(args[2])
        return _js_convert(ir, args[1].value, args[2])
    elseif f === length && length(args) == 1 && _js_isvector(args[1])
        return _js_rt(_js_code(mod, args[1], _JS_ATOM) * ".length", Int64)
    elseif f === length && length(args) == 1 && _js_isrange(args[1])
        r = _js_range!(ir, args[1])
        r.static && return _js_static(length(r.value))
        return _js_rt(r.code * ".stop - " * r.code * ".start + 1", Int64; prec = _JS_ADD)
    elseif f === size && length(args) == 2 && _js_isvector(args[1]) && _js_isint(args[2])
        d = args[2]
        d isa _JsVal && d.static && d.value == 1 && return _js_call!(ir, length, Any[args[1]], Int64)
    elseif f === getindex && length(args) == 2 && _js_isvector(args[1]) && _js_isint(args[2])
        return _js_fcall(ir, _js_use!(mod, :_rk_at), args, T)
    elseif f === setindex! && length(args) == 3 && _js_isvector(args[1]) && _js_isint(args[3])
        return _js_fcall(ir, _js_use!(mod, :_rk_set), args, T; pure = false)
    elseif f === sum && length(args) == 1 && _js_isvector(args[1]) && eltype(_js_type(args[1])) <: _JsNumber
        return _js_fcall(ir, _js_use!(mod, :_rk_sum), args, T)
    elseif f in (maximum, minimum) && length(args) == 1 && _js_isvector(args[1])
        # The error Julia throws for an empty collection, as this Julia throws it.
        empty = try
            f(eltype(_js_type(args[1]))[])
        catch e
            e
        end
        return _js_rt(_js_use!(mod, :_rk_extremum) * "(" * (f === maximum ? "Math.max" : "Math.min") *
                      ", " * _js_runtime(mod, args[1]).code * ", () => " *
                      _js_runtime(mod, _js_static(empty)).code * ")", T)
    elseif f in (eachindex, LinearIndices) && length(args) == 1 && _js_isvector(args[1])
        return _js_rt(_js_use!(mod, :_rk_range) * "(1, " * _js_code(mod, args[1], _JS_ATOM) * ".length)", T)
    elseif f === axes && length(args) == 2 && _js_isvector(args[1]) && args[2] isa _JsVal &&
           args[2].static && args[2].value == 1
        return _js_rt(_js_use!(mod, :_rk_range) * "(1, " * _js_code(mod, args[1], _JS_ATOM) * ".length)", T)
    elseif f === lastindex && length(args) == 1 && _js_isvector(args[1])
        return _js_rt(_js_code(mod, args[1], _JS_ATOM) * ".length", Int64)
    elseif f === firstindex && length(args) == 1 && _js_isvector(args[1])
        return _js_static(1)
    elseif f === (:) && length(args) == 2 && all(_js_isint, args)
        return _js_fcall(ir, _js_use!(mod, :_rk_range), args, UnitRange{Int64})
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
    elseif f === copy && length(args) == 1 && _js_isvector(args[1])
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
    elseif f === map && length(args) == 2 && (_js_isvector(args[2]) || _js_isrange(args[2]))
        return _js_materialize!(ir, _JsBroadcast(args[1] isa _JsVal ? args[1].value : args[1],
                                                 Any[args[2]], Nothing), T)
    elseif f === push! && length(args) == 2 && _js_isvector(args[1]) && _js_isnumber(args[2])
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
    elseif f === string && all(a -> a isa _JsVal, args)
        return _js_rt("\"\" + " * join((_js_code(mod, a, _JS_ADD + 1) for a in args), " + "), String;
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
function _js_function_call(ir::_JsIR, f::Core.Builtin, args, T)
    mod = ir.fn.mod
    if f === Core.getfield || f === getfield
        x, k = args[1], args[2]
        k isa _JsVal && k.static || _js_error(mod, "a field read with a runtime field index")
        key = k.value
        if x isa _JsTuple
            idx = key isa Symbol ? findfirst(==(key), fieldnames(x.T)) : key
            return x.items[idx]
        elseif x isa _JsStruct
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
           all(a -> a isa _JsTuple || (a isa _JsVal && a.static && a.value isa Tuple), args[3:end])
        items = Any[]
        for a in args[3:end]
            a isa _JsTuple ? append!(items, a.items) : append!(items, (_js_static(x) for x in a.value))
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
    T === Float64 && S <: _JsNumber && return _js_rt(v.code, Float64; prec = v.prec, ref = v.ref, pure = v.pure)
    T === Int64 && S === Float64 && return _js_fcall(ir, _js_use!(ir.fn.mod, :_rk_int), Any[v], Int64)
    _js_error(ir.fn.mod, "converting a `", S, "` to `", T, "` has no JavaScript translation")
end

# Number constructors (`Float64(x)`, `Int(x)`) are calls of the type.
function _js_function_call(ir::_JsIR, f::Type, args, T)
    if f in (Float64, Int64) && length(args) == 1 && _js_isnumber(args[1])
        return _js_convert(ir, f, args[1])
    elseif f === Base.OneTo && length(args) == 1 && _js_isint(args[1])
        return _js_rt(_js_use!(ir.fn.mod, :_rk_range) * "(1, " * _js_runtime(ir.fn.mod, args[1]).code * ")", T)
    elseif f === Base.Generator && length(args) == 2 && (_js_isvector(args[2]) || _js_isrange(args[2]))
        return _JsStruct(f, Any[args...], T)
    end
    nothing
end

# --- the module -----------------------------------------------------------------------------------------

function _js_input_check(mod::_JsModule, name, T)
    if T === Float64
        "_rk_number(" * _js_string_literal(name) * ", " * name * ", false);", :_rk_number
    elseif T === Int64
        "_rk_number(" * _js_string_literal(name) * ", " * name * ", true);", :_rk_number
    elseif T === Bool
        "_rk_boolean(" * _js_string_literal(name) * ", " * name * ");", :_rk_boolean
    elseif T <: Vector && eltype(T) in (Float64, Int64, Bool)
        el = eltype(T) === Bool ? "boolean" : "number"
        "_rk_vector(" * _js_string_literal(name) * ", " * name * ", \"" * el * "\"" *
            (eltype(T) === Int64 ? ", true" : "") * ");", :_rk_vector
    else
        throw(JSEmissionError("the input `$name` has type `$T`; JavaScript inputs are Float64, Int64, Bool or vectors of them"))
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
_js_want_names(w::Symbol) = (w,)
_js_want_names(w::Tuple) = Tuple(n for x in w for n in _js_want_names(x))
_js_want_names(w::NamedTuple) = Tuple(n for x in values(w) for n in _js_want_names(x))

# The kernel function of a plan.
function _js_plan_function!(mod::_JsModule, fname::String, p::Plan, argtypes, shape)
    fn = _JsFn(mod)
    values = Dict{ValueId,Any}()
    params = String[]
    for (v, T) in zip(p.have, argtypes)
        name = _js_local!(fn, v.name)
        push!(params, name)
        check, helper = _js_input_check(mod, name, T)
        push!(mod.runtime, helper)
        _js_line!(fn, check)
        values[canon_id(p.graph, v.id)] = _js_rt(name, T; ref = true)
    end
    wants = _js_plan_body!(fn, p, values)
    out = Dict(v.name => _js_runtime(mod, w).code for (v, w) in zip(p.want, wants))
    _js_line!(fn, "return " * _js_shape(shape, out) * ";")
    code = "export function " * fname * "(" * join(params, ", ") * ") {\n" * join(fn.lines, "\n") * "\n}"
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
            args = Any[a isa Value ? values[canon_id(p.graph, a.id)] : _js_static(a) for a in c.args]
            _js_call!(kir, c.f, args, Nothing)
            _js_flush!(kir)
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
        result, RT = if op isa _KernelSourceOp
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

# An authored plate: a loop over its cells, each running the cell plan. A
# mapped argument is read at the cell's position (one of length 1 at every
# position, as broadcasting reads it); a shared argument is read whole.
function _js_plate!(fn::_JsFn, op::_AuthoredPlateOp{K,A}, inputs, hint) where {K,A}
    mod = fn.mod
    body = op.kernel.plan
    length(body.want) == 1 || _js_error(mod, "a plate whose cell computes several values has no JavaScript translation yet")
    mapped = Bool[!(i in A) && (_js_isvector(v) || _js_isrange(v)) for (i, v) in enumerate(inputs)]
    for (i, v) in enumerate(inputs)
        !(i in A) && !mapped[i] && !(_js_type(v) <: Union{Number,Bool}) &&
            _js_error(mod, "a plate over a `", _js_type(v), "` has no JavaScript translation (vectors and ranges are)")
    end
    any(mapped) || _js_error(mod, "a plate without a vector argument")
    kir = _js_ir(fn, only(Base.code_lowered(_js_noop, Tuple{})), 0)
    args = Any[!mapped[i] ? inputs[i] : _js_isrange(inputs[i]) ? _js_range!(kir, inputs[i], "x") :
               _js_simple!(kir, inputs[i], "x") for i in eachindex(inputs)]
    lengths = unique([_js_isrange(a) ? _js_range_length(a) : a.code * ".length"
                      for (i, a) in enumerate(args) if mapped[i]])
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
        elseif _js_isrange(a)
            _js_rt("(" * _js_range_start(a) * " + " * i * ")", Int64; prec = _JS_ATOM)
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
below 2^53 in magnitude), `Bool`s and vectors of them; sizes and numeric
inputs stay runtime arguments. Translated: ordinary kernel recipes,
broadcasts (fused, with Julia's length rules), `plate` and `scan` as loops,
arithmetic and comparisons, `Base.Math` functions with Julia's domain errors,
checked 1-based indexing, `sum`/`maximum`/`minimum`, ranges, comprehensions
and `map` over vectors. Plain Julia helpers are translated from their typed
code, branches, early returns and loops included, without annotation;
Julia's own functions translate only through RK's table. A construct without
a translation (a C call, `fma`, a matrix, a scan with `history`) raises
`ReactiveKernels.JSEmissionError` naming it and its
call chain. Elementary functions use JavaScript's `Math`, which may differ
from Julia's in the last bit.
"""
function js_source(spec::KernelSpec; have = _KERNEL_DEFAULT_BOUNDARY,
                   want = _KERNEL_DEFAULT_BOUNDARY, types = nothing, name = :kernel)
    p = plan(spec; have, want = _js_want_names(want))
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
