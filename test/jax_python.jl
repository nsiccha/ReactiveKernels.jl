# Evaluate a `jax_source` module in Python from a test. `RK_JAX_PYTHON` names
# a Python interpreter with `jax` installed; without it the parity checks are
# reported as skipped (`@test_skip`), and the emission checks still run.
#
#     python -m venv jaxenv && jaxenv/bin/pip install "jax[cpu]"
#     RK_JAX_PYTHON=$PWD/jaxenv/bin/python julia --project=. test/test_jax_source.jl

const RK_JAX_PYTHON = get(ENV, "RK_JAX_PYTHON", "")
rk_jax_available() = !isempty(RK_JAX_PYTHON) && isfile(RK_JAX_PYTHON)

# A ragged input (one vector per group) crosses as its padded array and
# lengths, as the module's `pad_ragged` builds them.
function rk_jax_pad(x::AbstractVector{<:AbstractVector})
    lengths = length.(x)
    width = max(1, maximum(lengths; init = 0))
    [isempty(r) ? 0.0 : r[min(i, length(r))] for r in x, i in 1:width], lengths
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
    (eltype(x) <: Bool ? "bool" : eltype(x) <: Integer ? "np.int64" : "np.float64") * ")"
# A raw ragged input (for `prepare_data`) is a list of arrays.
_rk_py_input(x::AbstractVector{<:AbstractVector{<:Number}}) =
    "[" * join(map(_rk_py_input, x), ", ") * "]"
_rk_py_input(x) = _rk_py(x)

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
if sys.argv[5] == "prepare":
    prepared = module.prepare_data(*rest)
    call = lambda u: f(u, **prepared)
else:
    call = lambda u: f(u, *rest)
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
module's `prepare_data` turns into the function's inputs.
"""
function rk_jax_evaluate(src::JAXSource, name, args; grad::Bool = false, prepare::Bool = false)
    dir = mktempdir()
    module_path = joinpath(dir, "rk_module.py")
    write(module_path, src)
    args_path = joinpath(dir, "args.py")
    inputs = prepare ? Any[args...] : rk_jax_inputs(args)
    write(args_path, "ARGS = [" * join(map(_rk_py_input, inputs), ", ") * "]\n")
    driver = joinpath(dir, "driver.py")
    write(driver, _RK_JAX_DRIVER)
    errors = joinpath(dir, "stderr.txt")
    cmd = `$RK_JAX_PYTHON $driver $module_path $args_path $(String(name)) $(grad ? "grad" : "value") $(prepare ? "prepare" : "plain")`
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
_rk_jax_unarray(x::AbstractVector) = all(v -> v isa Number, x) ? Float64[v for v in x] :
    [_rk_jax_unarray(v) for v in x]
_rk_jax_unarray(x::NamedTuple) = map(_rk_jax_unarray, x)
_rk_jax_unarray(x::Dict) = Dict(k => _rk_jax_unarray(v) for (k, v) in x)
_rk_jax_unarray(x) = x

# Central finite differences of `f` at `u`.
rk_fd_gradient(f, u; h = 1e-6) = [(f(u .+ h .* (1:length(u) .== i)) -
                                   f(u .- h .* (1:length(u) .== i))) / 2h for i in eachindex(u)]
