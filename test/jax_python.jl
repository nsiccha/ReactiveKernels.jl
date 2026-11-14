# Evaluate a `jax_source` module in Python from a test. `RK_JAX_PYTHON` names
# a Python interpreter with `jax` installed; without it the parity checks are
# reported as skipped (`@test_skip`), and the emission checks still run.
#
#     python -m venv jaxenv && jaxenv/bin/pip install "jax[cpu]"
#     RK_JAX_PYTHON=$PWD/jaxenv/bin/python julia --project=. test/test_jax_source.jl

const RK_JAX_PYTHON = get(ENV, "RK_JAX_PYTHON", "")
rk_jax_available() = !isempty(RK_JAX_PYTHON) && isfile(RK_JAX_PYTHON)

# A ragged input (one vector per group) crosses as its padded array and
# lengths, as the module's `pad_ragged` builds them. An empty row is zeros of
# the column's element type, so an integer index column stays integer.
function rk_jax_pad(x::AbstractVector{<:AbstractVector})
    lengths = length.(x)
    width = max(1, maximum(lengths; init = 0))
    E = eltype(eltype(x))
    E[isempty(r) ? zero(E) : r[min(i, length(r))] for r in x, i in 1:width], lengths
end
rk_jax_inputs(args) = Any[a for x in args
    for a in (x isa AbstractVector{<:AbstractVector} ? rk_jax_pad(x) : (x,))]

_rk_py(x::Bool) = x ? "True" : "False"
_rk_py(x::Integer) = string(x)
_rk_py(x::AbstractFloat) = isnan(x) ? "float('nan')" :
    isinf(x) ? (x > 0 ? "float('inf')" : "float('-inf')") : repr(Float64(x))
_rk_py(x::AbstractString) = repr(String(x))
_rk_py(::Missing) = "None"
# Any other value (a categorical level) crosses as its label.
_rk_py(x) = repr(string(x))
_rk_py(x::AbstractArray) = ndims(x) == 1 ? "[" * join(map(_rk_py, x), ", ") * "]" :
    "[" * join((_rk_py(selectdim(x, 1, i)) for i in axes(x, 1)), ", ") * "]"
_rk_py_input(x::AbstractArray{<:Number}) = "np.asarray(" * _rk_py(x) * ", dtype=" *
    (eltype(x) <: Bool ? "bool" : eltype(x) <: Integer ? "np.int64" : "np.float64") * ")" *
    # An empty array of rank two or more (no ragged rows, padded to width 1)
    # prints as `[]`; it keeps its shape.
    (isempty(x) && ndims(x) > 1 ? ".reshape(" * _rk_py(collect(size(x))) * ")" : "")
# A raw ragged input (for `prepare_data`) is a list of arrays.
_rk_py_input(x::AbstractVector{<:AbstractVector{<:Number}}) =
    "[" * join(map(_rk_py_input, x), ", ") * "]"
# A named tuple (the value of a definition `prepare_data` takes as an input,
# `UNTRANSLATED`) is a dict by field name.
_rk_py_input(x::NamedTuple) =
    "{" * join((repr(String(k)) * ": " * _rk_py_input(v) for (k, v) in pairs(x)), ", ") * "}"
_rk_py_input(x::Tuple) = "(" * join(map(_rk_py_input, x), ", ") * (length(x) == 1 ? ",)" : ")")
_rk_py_input(x) = _rk_py(x)

# The args file of a call through `prepare_data`: `ARGS`, and `NAMES`, the
# names `prepare_data` takes its inputs by, which `raw_data` pairs with the
# raw values (in that order) for the documented call `prepare_data(**raw)`.
function _rk_jax_args(inputs, src::Union{Nothing,JAXSource} = nothing)
    text = "ARGS = [" * join(map(_rk_py_input, inputs), ", ") * "]\n"
    src === nothing && return text
    names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
    text * "NAMES = [" * join(repr.(String.(names)), ", ") * "]\n" * raw"""
def raw_data(values):
    if len(values) != len(NAMES):
        raise ValueError("%d raw values for prepare_data's %d inputs %s" % (len(values), len(NAMES), NAMES))
    return dict(zip(NAMES, values))
"""
end

const _RK_JAX_DRIVER = raw"""
import importlib.util, math, sys
import numpy as np
spec = importlib.util.spec_from_file_location("rk_module", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import jax
exec(open(sys.argv[2]).read())
f = getattr(module, sys.argv[3])
first, rest = ARGS[0], ARGS[1:]
if sys.argv[5] in ("prepare", "arguments"):
    prepared = module.prepare_data(**raw_data(rest))
    call = lambda u: f(u, **prepared)
else:
    call = lambda u: f(u, *rest)
if sys.argv[5] == "arguments":
    # The prepared data traced as arguments, not closed over.
    value = jax.jit(f)(first, **prepared)
else:
    value = jax.jit(call)(first)

def jl(x):
    if isinstance(x, dict):
        return "Dict(" + ", ".join(repr(k).replace("'", '"') + " => " + jl(v) for k, v in x.items()) + ")"
    if isinstance(x, (tuple, list)):
        return "Any[" + ", ".join(jl(v) for v in x) + "]"
    x = np.asarray(x)
    if x.ndim > 0:
        return "Any[" + ", ".join(jl(v) for v in x) + "]"
    v = x.item()
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "NaN" if math.isnan(v) else ("Inf" if v == math.inf else ("-Inf" if v == -math.inf else repr(v)))
    return repr(v)

out = "(value = " + jl(value)
if sys.argv[4] == "grad":
    out += ", grad = " + jl(jax.jit(jax.grad(call))(first))
print(out + ",)")
"""

"""
    rk_jax_evaluate(src, name, args; grad = false, prepare = false) -> (; value, grad)

Call function `name` of the emitted module `src` on `args`, jitted over its
first argument with the others closed over, as a sampler calls it; with
`grad = true` also return `jax.grad` with respect to the first argument.
With `prepare = true` the other arguments are the raw data, which the
module's `prepare_data` turns into the function's inputs; `prepare =
:arguments` also passes those to the jitted function as traced arguments.
"""
function rk_jax_evaluate(src::JAXSource, name, args; grad::Bool = false, prepare = false)
    dir = mktempdir()
    module_path = joinpath(dir, "rk_module.py")
    write(module_path, src)
    args_path = joinpath(dir, "args.py")
    inputs = prepare === false ? rk_jax_inputs(args) : Any[args...]
    write(args_path, _rk_jax_args(inputs, prepare === false ? nothing : src))
    driver = joinpath(dir, "driver.py")
    write(driver, _RK_JAX_DRIVER)
    errors = joinpath(dir, "stderr.txt")
    cmd = `$RK_JAX_PYTHON $driver $module_path $args_path $(String(name)) $(grad ? "grad" : "value") $(prepare === :arguments ? "arguments" : prepare === false ? "plain" : "prepare")`
    output = try
        read(pipeline(cmd; stderr = errors), String)
    catch e
        e isa ProcessFailedException || rethrow()
        # The Python traceback names the failing line of the emitted module.
        error("the emitted module failed in Python (module at ", module_path, "):\n",
              join(last(readlines(errors), 12), "\n"))
    end
    _rk_jax_unarray(Base.eval(Module(), Meta.parse(output)))
end
const _RK_JAX_FLOAT32_DRIVER = raw"""
import importlib.util, math, sys
import numpy as np
def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
# The reference (float64) module first: it turns on 64-bit mode, so a float64
# value the float32 module leaks would be float64 in its trace.
reference = load(sys.argv[2], "rk_reference") if sys.argv[2] != "-" else None
module = load(sys.argv[1], "rk_module")
import jax
exec(open(sys.argv[3]).read())
name, prepare = sys.argv[4], sys.argv[5] == "prepare"
def single(x):
    if isinstance(x, np.ndarray) and x.dtype.kind == "f":
        return x.astype(np.float32)
    if isinstance(x, float):
        return np.float32(x)
    return x
def caller(m, cast):
    f = getattr(m, name)
    first, rest = ARGS[0], ARGS[1:]
    if prepare:
        p = m.prepare_data(**raw_data(rest))
        return (single(first) if cast else first), (lambda u: f(u, **p))
    rest = [single(a) for a in rest] if cast else rest
    return (single(first) if cast else first), (lambda u: f(u, *rest))
# A strongly typed float64 value anywhere in a trace, nested jaxprs included.
# A Python literal (`jnp.where(m, x, 0.0)`) is weakly typed: it takes the
# float32 of what it meets, so it is no float64 arithmetic.
def strong_f64(j):
    bad = lambda a: getattr(a, "dtype", None) == np.float64 and not getattr(a, "weak_type", False)
    if any(bad(v.aval) for v in list(j.invars) + list(j.constvars)):
        return True
    for e in j.eqns:
        if any(bad(v.aval) for v in list(e.invars) + list(e.outvars)):
            return True
        for value in e.params.values():
            for x in (value if isinstance(value, (list, tuple)) else [value]):
                sub = getattr(x, "jaxpr", x)
                if hasattr(sub, "eqns") and strong_f64(sub):
                    return True
    return False
u, call = caller(module, True)
f64 = strong_f64(jax.make_jaxpr(call)(u).jaxpr) or strong_f64(jax.make_jaxpr(jax.grad(call))(u).jaxpr)
value, grad = jax.jit(call)(u), jax.jit(jax.grad(call))(u)
def jl(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 0:
        return "Float64[" + ", ".join(jl(v) for v in x.reshape(-1)) + "]"
    v = float(x)
    return "NaN" if math.isnan(v) else ("Inf" if v == math.inf else ("-Inf" if v == -math.inf else repr(v)))
out = "(x64 = %s, f64 = %s, dtype = %r, grad_dtype = %r, value = %s, grad = %s" % (
    "true" if jax.config.read("jax_enable_x64") else "false", "true" if f64 else "false",
    str(value.dtype), str(grad.dtype), jl(value), jl(grad))
if reference is not None:
    u64, call64 = caller(reference, False)
    out += ", reference = %s, reference_grad = %s" % (jl(jax.jit(call64)(u64)), jl(jax.jit(jax.grad(call64))(u64)))
print(out.replace("'", '"') + ")")
"""

"""
    rk_jax_float32(src32, reference, name, args; prepare = false) -> NamedTuple

Call function `name` of the module `src32`, whose floats are float32 (a
`float_type = Float32` module, or one of a `Float32` program), on `args`,
with the float values a caller passes cast to `float32`, and return its value
and `jax.grad` with respect to the first argument, their dtypes, whether
JAX's 64-bit mode was on (`x64`) and whether the traced value or gradient
holds any strongly typed float64 (`f64`; a weakly typed Python literal takes
the float32 of what it meets). With a `reference` module (the `Float64` module of
the same program), it is imported first, which turns 64-bit mode on, and its
value and gradient on the original arguments are returned as `reference` and
`reference_grad`; with `reference = nothing` the float32 module runs alone.
`prepare = true` passes the arguments after the first through `prepare_data`.
"""
function rk_jax_float32(src32::JAXSource, reference, name, args; prepare::Bool = false)
    dir = mktempdir()
    module_path = joinpath(dir, "rk_module.py")
    write(module_path, src32)
    reference_path = "-"
    if reference !== nothing
        reference_path = joinpath(dir, "rk_reference.py")
        write(reference_path, reference)
    end
    args_path = joinpath(dir, "args.py")
    inputs = prepare ? Any[args...] : rk_jax_inputs(args)
    write(args_path, _rk_jax_args(inputs, prepare ? src32 : nothing))
    driver = joinpath(dir, "float32.py")
    write(driver, _RK_JAX_FLOAT32_DRIVER)
    errors = joinpath(dir, "stderr.txt")
    cmd = `$RK_JAX_PYTHON $driver $module_path $reference_path $args_path $(String(name)) $(prepare ? "prepare" : "plain")`
    output = try
        read(pipeline(cmd; stderr = errors), String)
    catch e
        e isa ProcessFailedException || rethrow()
        error("the emitted float32 module failed in Python (module at ", module_path, "):\n",
              join(last(readlines(errors), 12), "\n"))
    end
    Base.eval(Module(), Meta.parse(output))
end

const _RK_JAX_LAYOUT_DRIVER = raw"""
import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("rk_module", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
exec(open(sys.argv[2]).read())
prepared = module.prepare_data(**raw_data(ARGS))
params = module.parameters(prepared)
print("(dimension = %d, parameters = Dict(%s))" % (module.dimension(prepared), ", ".join(
    '"%s" => (start = %d, stop = %d, dims = (%s))' % (k, v["start"], v["stop"],
        "".join("%d," % d for d in v["dims"])) for k, v in params.items())))
"""

"""
    rk_jax_layout(src, data) -> (; dimension, parameters)

`dimension(prepared)` and `parameters(prepared)` of an RKPPL module for the
raw `data` (`prepare_data`'s inputs, in order): the length of
`unconstrained` and each parameter's `(start, stop, dims)`.
"""
function rk_jax_layout(src::JAXSource, data)
    dir = mktempdir()
    module_path = joinpath(dir, "rk_module.py")
    write(module_path, src)
    args_path = joinpath(dir, "args.py")
    write(args_path, _rk_jax_args(Any[data...], src))
    driver = joinpath(dir, "layout.py")
    write(driver, _RK_JAX_LAYOUT_DRIVER)
    Base.eval(Module(), Meta.parse(read(`$RK_JAX_PYTHON $driver $module_path $args_path`, String)))
end

_rk_jax_unarray(x::AbstractVector) = all(v -> v isa Number, x) ? Float64[v for v in x] :
    [_rk_jax_unarray(v) for v in x]
_rk_jax_unarray(x::NamedTuple) = map(_rk_jax_unarray, x)
_rk_jax_unarray(x::Dict) = Dict(k => _rk_jax_unarray(v) for (k, v) in x)
_rk_jax_unarray(x) = x

# Central finite differences of `f` at `u`.
rk_fd_gradient(f, u; h = 1e-6) = [(f(u .+ h .* (1:length(u) .== i)) -
                                   f(u .- h .* (1:length(u) .== i))) / 2h for i in eachindex(u)]

# A Python that parses an emitted module's text: `RK_JAX_PYTHON`, else any
# `python3` (parsing needs no JAX); `nothing` without one.
rk_python_parser() = rk_jax_available() ? RK_JAX_PYTHON : Sys.which("python3")

const _RK_PY_DOCSTRING_DRIVER = raw"""
import ast, sys, warnings
warnings.simplefilter("error")
with open(sys.argv[1], encoding="utf-8") as f:
    tree = ast.parse(f.read())
doc = ast.get_docstring(tree, clean=False)
if doc is None:
    sys.exit("the module has no docstring")
sys.stdout.buffer.write(doc.encode("utf-8"))
"""

"""
    rk_python_docstring(src) -> Union{String,Nothing}

The module docstring of the emitted module `src` as Python reads it, with
every warning an error (an invalid escape sequence among them); `nothing`
without a Python (`rk_python_parser`).
"""
function rk_python_docstring(src)
    py = rk_python_parser()
    py === nothing && return nothing
    mktempdir() do dir
        module_path = joinpath(dir, "module.py")
        write(module_path, string(src))
        driver = joinpath(dir, "docstring.py")
        write(driver, _RK_PY_DOCSTRING_DRIVER)
        read(`$py $driver $module_path`, String)
    end
end

# Python reads `expected` as the module docstring of `src`; a multi-line one
# ends in the newline before its closing quotes. Skipped without a Python.
function rk_test_python_docstring(src, expected)
    doc = rk_python_docstring(src)
    if doc === nothing
        @test_skip false
    else
        @test chomp(doc) == expected
    end
end
