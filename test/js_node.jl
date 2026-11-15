# Run functions of a `js_source` module with Node.js and read their results
# back into Julia. Tests skip their Node checks when no `node` is on the PATH
# (or `RK_JS_NODE` names none).

const RK_JS_NODE = let node = get(ENV, "RK_JS_NODE", "")
    isempty(node) ? Sys.which("node") : node
end
rk_js_node_available() = RK_JS_NODE !== nothing && !isempty(RK_JS_NODE)

# JSON text of Julia arguments; nonfinite numbers travel as strings.
_rk_js_json(x::Bool) = x ? "true" : "false"
_rk_js_json(x::AbstractFloat) = isfinite(x) ? repr(Float64(x)) :
    "\"" * (isnan(x) ? "NaN" : x > 0 ? "Infinity" : "-Infinity") * "\""
_rk_js_json(x::Integer) = string(x)
_rk_js_json(x::AbstractVector) = "[" * join(map(_rk_js_json, x), ",") * "]"
_rk_js_json(x::Tuple) = _rk_js_json(collect(Any, x))

const _RK_JS_DRIVER = raw"""
import * as m from "./module.mjs";
import { readFileSync } from "node:fs";
const decode = (x) => typeof x === "string" ? Number(x) : Array.isArray(x) ? x.map(decode) : x;
const encode = (x) => typeof x === "number" ? (Number.isFinite(x) ? (Object.is(x, -0) ? "-0" : x) : String(x)) :
  Array.isArray(x) || ArrayBuffer.isView(x) ? Array.from(x, encode) :
  x !== null && typeof x === "object" ? Object.fromEntries(Object.entries(x).map(([k, v]) => [k, encode(v)])) : x;
const calls = JSON.parse(readFileSync(process.argv[2], "utf8"));
const results = calls.map(([name, args]) => {
  try { return { value: encode(m[name](...args.map(decode))) }; }
  catch (e) { return { error: e.name, message: e.message }; }
});
console.log(JSON.stringify(results));
"""

# `calls` is a vector of `fname => args`; each result is `(value = ...)` or
# `(error = name, message = ...)`.
function rk_js_run(source::AbstractString, calls)
    mktempdir() do dir
        write(joinpath(dir, "module.mjs"), source)
        write(joinpath(dir, "driver.mjs"), _RK_JS_DRIVER)
        write(joinpath(dir, "calls.json"),
              "[" * join(("[\"" * string(f) * "\"," * _rk_js_json(Tuple(a)) * "]" for (f, a) in calls), ",") * "]")
        out = read(`$RK_JS_NODE $(joinpath(dir, "driver.mjs")) $(joinpath(dir, "calls.json"))`, String)
        [_rk_js_result(r) for r in _rk_js_parse(strip(out))]
    end
end
rk_js_call(source::AbstractString, fname, args...) = only(rk_js_run(source, [fname => args]))

function _rk_js_result(r::Dict)
    haskey(r, "error") && return (; error = r["error"], message = r["message"])
    (; value = _rk_js_value(r["value"]))
end
_rk_js_value(x::String) = x == "-0" ? -0.0 : parse(Float64, x == "Infinity" ? "Inf" : x == "-Infinity" ? "-Inf" : x)
_rk_js_value(x::Vector) = isempty(x) ? Float64[] : all(e -> e isa Bool, x) ? Bool[e for e in x] :
    all(e -> e isa Union{Real,String}, x) ? Float64[_rk_js_value(e) for e in x] : Any[_rk_js_value(e) for e in x]
_rk_js_value(x::Dict) = Dict(k => _rk_js_value(v) for (k, v) in x)
_rk_js_value(x) = x isa Real && !(x isa Bool) ? Float64(x) : x

# A minimal JSON reader for the driver's output.
function _rk_js_parse(s::AbstractString)
    pos = Ref(firstindex(s))
    v = _rk_js_read(s, pos)
    v
end
function _rk_js_skip(s, pos)
    while pos[] <= lastindex(s) && isspace(s[pos[]])
        pos[] = nextind(s, pos[])
    end
end
function _rk_js_read(s, pos)
    _rk_js_skip(s, pos)
    c = s[pos[]]
    if c == '['
        pos[] += 1
        items = Any[]
        _rk_js_skip(s, pos)
        s[pos[]] == ']' && (pos[] += 1; return items)
        while true
            push!(items, _rk_js_read(s, pos))
            _rk_js_skip(s, pos)
            s[pos[]] == ']' && (pos[] += 1; return items)
            s[pos[]] == ',' || error("JSON: expected , at $(pos[])")
            pos[] += 1
        end
    elseif c == '{'
        pos[] += 1
        d = Dict{String,Any}()
        _rk_js_skip(s, pos)
        s[pos[]] == '}' && (pos[] += 1; return d)
        while true
            k = _rk_js_read(s, pos)
            _rk_js_skip(s, pos)
            s[pos[]] == ':' || error("JSON: expected : at $(pos[])")
            pos[] += 1
            d[k] = _rk_js_read(s, pos)
            _rk_js_skip(s, pos)
            s[pos[]] == '}' && (pos[] += 1; return d)
            s[pos[]] == ',' || error("JSON: expected , at $(pos[])")
            pos[] += 1
        end
    elseif c == '"'
        io = IOBuffer()
        i = nextind(s, pos[])
        while s[i] != '"'
            if s[i] == '\\'
                i = nextind(s, i)
                e = s[i]
                if e == 'u'
                    print(io, Char(parse(UInt16, s[i+1:i+4]; base = 16)))
                    i += 4
                else
                    print(io, get(Dict('n' => '\n', 't' => '\t', 'r' => '\r', 'b' => '\b', 'f' => '\f'), e, e))
                end
            else
                print(io, s[i])
            end
            i = nextind(s, i)
        end
        pos[] = nextind(s, i)
        return String(take!(io))
    elseif startswith(SubString(s, pos[]), "true")
        pos[] += 4
        return true
    elseif startswith(SubString(s, pos[]), "false")
        pos[] += 5
        return false
    elseif startswith(SubString(s, pos[]), "null")
        pos[] += 4
        return nothing
    end
    m = match(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", s, pos[])
    m === nothing && error("JSON: unexpected text at $(pos[])")
    pos[] += ncodeunits(m.match)
    parse(Float64, m.match)
end
