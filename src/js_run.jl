# Executing a `js_source` module with Node.js: `js_run` calls the module's
# entry function on a batch of cases in one Node process and decodes each
# returned WANT into the Julia shape the native kernel returns. Consumers own
# their comparisons with native values; this file owns only execution.

"""
    JSRunError(message)

`js_run` could not execute a module: Node.js is missing, failed or timed out,
the module did not import, or it exports no function of the entry's name.
A call that throws inside the module is not a `JSRunError`; it is an
[`JSOutcome`](@ref ReactiveKernels.JSOutcome) with its error.
"""
struct JSRunError <: Exception
    message::String
end
Base.showerror(io::IO, e::JSRunError) = print(io, "JSRunError: ", e.message)

"""
    JSOutcome

The outcome of one call of a module's entry function by `js_run`.

- `value`: the returned WANT in the shape `prepare(spec; have, want)` returns
  for the same `want` (a value for a name or a one-name tuple, a `Tuple` for a
  tuple, a `NamedTuple` for a named one, nested as given), each value
  converted to the Julia type the module's `OUTPUTS` declares; `nothing` when
  the call threw.
- `error`: `nothing`, or the name of the error the call threw: the Julia
  exception type for a `JuliaError` (`"DomainError"`, …), else the JavaScript
  error's name (`"TypeError"`, …).
- `message`: the error's message (`""` without one).
- `julia_error`: whether the error was the module's `JuliaError`.
- `inputs_mutated`: whether the call changed any of the arrays it was given.
"""
struct JSOutcome
    value::Any
    error::Union{Nothing,String}
    message::String
    julia_error::Bool
    inputs_mutated::Bool
end

# The Node.js `js_run` uses by default: `ENV["RK_JS_NODE"]`, else `node` on the PATH.
function _js_default_node()
    node = get(ENV, "RK_JS_NODE", "")
    isempty(node) || return node
    Sys.which("node")
end

const _JS_RUN_DRIVER = raw"""
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [, , modulePath, casesPath] = process.argv;
const out = (x) => process.stdout.write(JSON.stringify(x) + "\n");
// A matrix is an array of rows; one without rows carries its column count as
// `cols`, which crosses JSON as `{ __cols: n }`.
const decode = (x) => typeof x === "string" ?
  (x === "NaN" ? NaN : x === "Infinity" ? Infinity : x === "-Infinity" ? -Infinity : x === "-0" ? -0 : x) :
  Array.isArray(x) ? x.map(decode) :
  x !== null && typeof x === "object" && Object.keys(x).length === 1 && Number.isInteger(x.__cols) ?
    Object.assign([], { cols: x.__cols }) : x;
const encode = (x) => typeof x === "number" ?
  (Number.isNaN(x) ? "NaN" : x === Infinity ? "Infinity" : x === -Infinity ? "-Infinity" : Object.is(x, -0) ? "-0" : x) :
  Array.isArray(x) && x.length === 0 && Number.isInteger(x.cols) && x.cols > 0 ? { __cols: x.cols } :
  Array.isArray(x) || ArrayBuffer.isView(x) ? Array.from(x, encode) :
  x !== null && typeof x === "object" ? Object.fromEntries(Object.entries(x).map(([k, v]) => [k, encode(v)])) :
  x === undefined ? null : x;
const { entry, vectors, cases } = JSON.parse(readFileSync(casesPath, "utf8"));
let m;
try {
  m = await import(pathToFileURL(modulePath).href);
} catch (e) {
  out({ fatal: "the module does not import: " + (e && e.stack || String(e)) });
  process.exit(0);
}
if (typeof m[entry] !== "function") {
  out({ fatal: "the module exports no function `" + entry + "`; it exports " +
    Object.keys(m).map((k) => "`" + k + "`").join(", ") });
  process.exit(0);
}
const inputs = Array.isArray(m.INPUTS) ? m.INPUTS : [];
const results = cases.map((raw) => {
  const args = raw.map(decode).map((a, i) =>
    vectors === "float64array" && Array.isArray(a) && /^Vector\{(Float64|Int64)\}$/.test(inputs[i]?.type ?? "") ?
      Float64Array.from(a) :
    vectors === "float64array" && Array.isArray(a) && /^Matrix\{(Float64|Int64)\}$/.test(inputs[i]?.type ?? "") ?
      Object.assign(a.map((r) => Float64Array.from(r)), a.length === 0 ? { cols: a.cols } : {}) : a);
  const before = JSON.stringify(encode(args));
  let r;
  try {
    r = { value: encode(m[entry](...args)) };
  } catch (e) {
    const julia = typeof m.JuliaError === "function" && e instanceof m.JuliaError;
    r = { error: e && e.name ? String(e.name) : typeof e, message: e && e.message !== undefined ? String(e.message) : String(e), julia };
  }
  r.mutated = JSON.stringify(encode(args)) !== before;
  return r;
});
out({ outputs: m.OUTPUTS ?? null, returns: m.RETURNS ?? null, results });
"""

_js_run_json(x::Bool) = x ? "true" : "false"
function _js_run_json(x::Integer)
    abs(x) <= 2^53 || throw(ArgumentError("js_run: the integer $x is not exact as a JavaScript number"))
    string(x)
end
_js_run_json(x::AbstractFloat) = isnan(x) ? "\"NaN\"" : isinf(x) ? (x > 0 ? "\"Infinity\"" : "\"-Infinity\"") :
    iszero(x) && signbit(x) ? "\"-0\"" : repr(Float64(x))
_js_run_json(x::AbstractVector) = "[" * join(map(_js_run_json, x), ",") * "]"
# A matrix as an array of rows; one without rows as its column count.
_js_run_json(x::AbstractMatrix) = size(x, 1) == 0 && size(x, 2) > 0 ? "{\"__cols\":" * string(size(x, 2)) * "}" :
    "[" * join((_js_run_json(collect(r)) for r in eachrow(x)), ",") * "]"
_js_run_json(x::Union{Tuple,AbstractString,Symbol}) = x isa Tuple ? _js_run_json(collect(Any, x)) :
    "\"" * escape_string(String(x)) * "\""
_js_run_json(x) = throw(ArgumentError(
    "js_run: an argument of type $(typeof(x)) has no JavaScript form; arguments are numbers, Booleans and vectors of them"))

"""
    js_run(source, entry, cases; node, vectors = :array, timeout = 60) -> Vector{JSOutcome}

Call the function `entry` of the JavaScript module `source` (the text
[`js_source`](@ref ReactiveKernels.js_source) returns) once per case in
`cases`, each a tuple of the HAVE values in the module's `INPUTS` order, in
one Node.js process, and return one [`JSOutcome`](@ref ReactiveKernels.JSOutcome)
per case, in order. Prototype; not exported.

Arguments are `Float64`, `Int64` and `Bool` values and vectors of them, also
nonfinite numbers and `-0.0`; they are copied, so the caller's arrays are never
changed. `vectors = :float64array` passes `Vector{Float64}` and
`Vector{Int64}` inputs as `Float64Array`s instead of arrays. A call that
throws inside the module is reported in its outcome; `js_run` throws a
[`JSRunError`](@ref ReactiveKernels.JSRunError) only when the run itself
fails: no Node.js, a module that does not import, a missing `entry`, a crash
or no answer within `timeout` seconds.

`node` is the Node.js executable or a `Cmd` running it (such as a
`NodeJS_*_jll` package's `node()`); by default `ENV["RK_JS_NODE"]`, else
`node` on the PATH. Each call writes the module to a fresh temporary
directory, runs Node.js there and removes the directory; no process outlives
the call.
"""
function js_run(source::AbstractString, entry::Union{Symbol,AbstractString}, cases::AbstractVector;
                node = _js_default_node(), vectors::Symbol = :array, timeout::Real = 60)
    vectors in (:array, :float64array) ||
        throw(ArgumentError("js_run: `vectors` is :array or :float64array; got :$vectors"))
    node === nothing && throw(JSRunError(
        "Node.js is not available: put `node` on the PATH, set ENV[\"RK_JS_NODE\"], or pass `node`"))
    cmd = node isa Cmd ? node : `$node`
    payload = "{\"entry\":" * _js_run_json(String(entry)) * ",\"vectors\":\"" * String(vectors) *
              "\",\"cases\":[" * join((_js_run_json(Tuple(c)) for c in cases), ",") * "]}"
    stdout_text, stderr_text = mktempdir() do dir
        modpath = joinpath(dir, "module.mjs")
        driver = joinpath(dir, "driver.mjs")
        casespath = joinpath(dir, "cases.json")
        write(modpath, source)
        write(driver, _JS_RUN_DRIVER)
        write(casespath, payload)
        out, err = IOBuffer(), IOBuffer()
        proc = try
            run(pipeline(`$cmd $driver $modpath $casespath`; stdout = out, stderr = err); wait = false)
        catch e
            throw(JSRunError("Node.js (`$(cmd)`) could not be started: " * sprint(showerror, e)))
        end
        timed_out = Ref(false)
        timer = Timer(timeout) do _
            process_running(proc) || return
            timed_out[] = true
            kill(proc)
        end
        try
            wait(proc)
        finally
            close(timer)
        end
        timed_out[] && throw(JSRunError("Node.js gave no answer within $(timeout) s"))
        success(proc) || throw(JSRunError("Node.js exited with " *
            (proc.termsignal != 0 ? "signal $(proc.termsignal)" : "code $(proc.exitcode)") *
            ": " * String(take!(err))))
        String(take!(out)), String(take!(err))
    end
    line = strip(stdout_text)
    reply = try
        _js_run_parse(line)
    catch
        throw(JSRunError("Node.js answered no result: " * repr(first(line, 500)) * " " * stderr_text))
    end
    reply isa Dict || throw(JSRunError("Node.js answered no result: " * repr(first(line, 500))))
    haskey(reply, "fatal") && throw(JSRunError(reply["fatal"]))
    types = _js_run_output_types(reply["outputs"])
    returns = reply["returns"]
    [_js_run_outcome(r, types, returns) for r in reply["results"]]
end

function _js_run_output_types(outputs)
    types = Dict{String,Any}()
    outputs isa Vector || return types
    for o in outputs
        o isa Dict && haskey(o, "type") || continue
        T = _js_run_type(o["type"])
        T === nothing || (types[o["name"]] = T)
    end
    types
end
const _JS_RUN_TYPES = Dict{String,Any}(string(T) => T for T in
    (Float64, Int64, Bool, Vector{Float64}, Vector{Int64}, Vector{Bool},
     Matrix{Float64}, Matrix{Int64}, Matrix{Bool}))
_js_run_type(name) = get(_JS_RUN_TYPES, name, nothing)

function _js_run_outcome(r::Dict, types, returns)
    mutated = get(r, "mutated", false) === true
    haskey(r, "error") &&
        return JSOutcome(nothing, string(r["error"]), string(get(r, "message", "")), get(r, "julia", false) === true, mutated)
    JSOutcome(_js_run_native_shape(returns, r["value"], types), nothing, "", false, mutated)
end

# A tuple of one name is a native kernel's flat boundary of one value, which it
# returns bare; the module returns a one-element array.
_js_run_native_shape(returns::Vector, x, types) =
    length(returns) == 1 && only(returns) isa String ?
    _js_run_shape(only(returns), only(x), types) : _js_run_shape(returns, x, types)
_js_run_native_shape(returns, x, types) = _js_run_shape(returns, x, types)

# The returned value in the WANT's shape, each output converted to its type.
_js_run_shape(returns::String, x, types) = _js_run_value(get(types, returns, nothing), x)
_js_run_shape(returns::Vector, x, types) = Tuple(_js_run_shape(s, e, types) for (s, e) in zip(returns, x))
_js_run_shape(returns::Dict, x, types) =
    (; (Symbol(k) => _js_run_shape(returns[k], x[k], types) for k in _js_run_keys(returns))...)
_js_run_shape(returns, x, types) = _js_run_value(nothing, x)
_js_run_keys(d::Dict) = get(d, :__order, collect(keys(d)))

_js_run_number(x::Real) = Float64(x)
_js_run_number(x::String) = x == "NaN" ? NaN : x == "Infinity" ? Inf : x == "-Infinity" ? -Inf :
    x == "-0" ? -0.0 : throw(JSRunError("the module returned the string $(repr(x)) where a number was expected"))
_js_run_value(::Type{Float64}, x) = _js_run_number(x)
_js_run_value(::Type{Int64}, x) = (v = _js_run_number(x); isinteger(v) ? Int64(v) :
    throw(JSRunError("the module returned $v where an Int64 was expected")))
_js_run_value(::Type{Bool}, x) = x isa Bool ? x :
    throw(JSRunError("the module returned $(repr(x)) where a Bool was expected"))
_js_run_value(::Type{Vector{T}}, x) where {T} = T[_js_run_value(T, e) for e in x]
_js_run_value(::Type{Matrix{T}}, x::Dict) where {T} = haskey(x, "__cols") ? Matrix{T}(undef, 0, Int(x["__cols"])) :
    throw(JSRunError("the module returned an object where a Matrix{$T} was expected"))
function _js_run_value(::Type{Matrix{T}}, x::Vector) where {T}
    rows = [T[_js_run_value(T, e) for e in r] for r in x]
    all(r -> length(r) == length(first(rows)), rows) ||
        throw(JSRunError("the module returned rows of different lengths where a Matrix{$T} was expected"))
    m = Matrix{T}(undef, length(rows), isempty(rows) ? 0 : length(first(rows)))
    for (i, r) in enumerate(rows), j in eachindex(r)
        m[i, j] = r[j]
    end
    m
end
# Without a declared type: numbers as Float64, arrays element by element.
_js_run_value(::Nothing, x::Union{Real,String}) = x isa Bool ? x : _js_run_number(x)
_js_run_value(::Nothing, x::Vector) = isempty(x) ? Float64[] : all(e -> e isa Bool, x) ? Bool[x...] :
    all(e -> e isa Union{Real,String}, x) ? Float64[_js_run_number(e) for e in x] : Any[_js_run_value(nothing, e) for e in x]
_js_run_value(::Nothing, x::Dict) = (; (Symbol(k) => _js_run_value(nothing, x[k]) for k in _js_run_keys(x))...)
_js_run_value(::Nothing, x) = x

# A JSON reader for the driver's one-line answer. Objects keep their key order
# (`:__order`), which a named-tuple WANT reads.
function _js_run_parse(s::AbstractString)
    i, v = _js_run_read(s, firstindex(s))
    i = _js_run_skip(s, i)
    i <= lastindex(s) && error("trailing text")
    v
end
function _js_run_skip(s, i)
    while i <= lastindex(s) && isspace(s[i])
        i = nextind(s, i)
    end
    i
end
function _js_run_read(s, i)
    i = _js_run_skip(s, i)
    c = s[i]
    if c == '['
        items = Any[]
        i = _js_run_skip(s, i + 1)
        s[i] == ']' && return i + 1, items
        while true
            i, v = _js_run_read(s, i)
            push!(items, v)
            i = _js_run_skip(s, i)
            s[i] == ']' && return i + 1, items
            s[i] == ',' || error("expected , in an array")
            i += 1
        end
    elseif c == '{'
        d = Dict{Any,Any}()
        order = String[]
        i = _js_run_skip(s, i + 1)
        s[i] == '}' && (d[:__order] = order; return i + 1, d)
        while true
            i, k = _js_run_read(s, i)
            i = _js_run_skip(s, i)
            s[i] == ':' || error("expected : in an object")
            i, v = _js_run_read(s, i + 1)
            d[k] = v
            push!(order, k)
            i = _js_run_skip(s, i)
            s[i] == '}' && (d[:__order] = order; return i + 1, d)
            s[i] == ',' || error("expected , in an object")
            i += 1
        end
    elseif c == '"'
        io = IOBuffer()
        i = nextind(s, i)
        while s[i] != '"'
            if s[i] == '\\'
                i = nextind(s, i)
                e = s[i]
                if e == 'u'
                    u = parse(UInt16, s[i+1:i+4]; base = 16)
                    i += 4
                    if 0xd800 <= u <= 0xdbff && i + 6 <= lastindex(s) && s[i+1] == '\\' && s[i+2] == 'u'
                        low = parse(UInt16, s[i+3:i+6]; base = 16)
                        if 0xdc00 <= low <= 0xdfff
                            print(io, Char(0x10000 + ((UInt32(u) - 0xd800) << 10) + (UInt32(low) - 0xdc00)))
                            i += 6
                        else
                            print(io, '�')
                        end
                    else
                        print(io, 0xd800 <= u <= 0xdfff ? '�' : Char(u))
                    end
                else
                    print(io, e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' :
                              e == 'f' ? '\f' : e)
                end
            else
                print(io, s[i])
            end
            i = nextind(s, i)
        end
        return nextind(s, i), String(take!(io))
    elseif startswith(SubString(s, i), "true")
        return i + 4, true
    elseif startswith(SubString(s, i), "false")
        return i + 5, false
    elseif startswith(SubString(s, i), "null")
        return i + 4, nothing
    end
    m = match(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?", s, i)
    (m === nothing || m.offset != i) && error("unexpected text")
    i + ncodeunits(m.match), parse(Float64, m.match)
end
