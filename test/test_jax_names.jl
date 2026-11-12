module JAXNamesTests
using ReactiveKernels, Test
include("jax_python.jl")

# Data named like a Python keyword or a builtin the module's code uses
# (`class`, `lambda`, `id`; snag jax-source-renam-c36033de). The module's data
# interface keeps those names: `prepare_data` takes them by name through
# `**data`, its result, `**prepared`, `DATA`, `INPUTS`, `PREPARED` and
# `UNTRANSLATED` key them by the same names, and only the Python locals that
# hold them are respelled. `x` is an ordinary name, a parameter as before.
# The range `1:class[1]` reads an entry of the prepared `class`, so
# `prepare_data` also returns that array's bound, computed from `class`.
@kernel jax_python_names(mu::Float64, class::Vector{Int}, lambda::Vector{Float64},
                         id::Vector{Float64}, x::Vector{Float64}) = begin
    weight = lambda .* x ./ sum(lambda .* x)
    head = sum(mu * id[j] for j in 1:class[1])
    lp = head - 0.5 * sum(weight .* abs2.(id .- mu .* class))
end

const RAW = (; class = [2, 1, 3], lambda = [0.5, 1.5, 1.0], id = [0.3, -0.2, 0.9],
             x = [1.0, 2.0, 0.5])
const MU = 0.7

function expected(mu)
    (; class, lambda, id, x) = RAW
    weight = lambda .* x ./ sum(lambda .* x)
    value = mu * sum(id[1:class[1]]) - 0.5 * sum(weight .* abs2.(id .- mu .* class))
    grad = sum(id[1:class[1]]) + sum(weight .* (id .- mu .* class) .* class)
    value, grad
end

inputs(src, fname) = first.(only(f for f in src.functions if f.name == fname).inputs)

# Runs `call` (Python, after `import` of the module as `m`) and returns what
# it prints: the error a `prepare_data` call raises, or "accepted".
function python_prepare(src, call)
    dir = mktempdir()
    write(joinpath(dir, "m.py"), src)
    driver = joinpath(dir, "driver.py")
    write(driver, """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
try:
    $call
    print("accepted")
except TypeError as e:
    print("TypeError: " + str(e))
""")
    strip(read(`$RK_JAX_PYTHON $driver $(joinpath(dir, "m.py"))`, String))
end

@testset "jax_source keeps data names Python reserves (class, lambda, id)" begin
    have = (:mu, :class, :lambda, :id, :x)
    src = jax_source(jax_python_names; have, want = :lp, data = RAW)
    text = src.text
    # `prepare_data` takes the data by their own names; only `x` is a
    # parameter, the others are read from `**data` into respelled locals.
    @test Set(inputs(src, "prepare_data")) == Set(["class", "lambda", "id", "x"])
    @test occursin("def prepare_data(x: ArrayLike, **data: ArrayLike) -> dict:", text)
    @test occursin("_rk_keyword_inputs(data, (", text)
    # The functions read `**prepared` by the same names.
    kernel_inputs = inputs(src, "kernel")
    @test "class" in kernel_inputs && "id" in kernel_inputs
    @test occursin("class_ = prepared[\"class\"]", text)
    @test occursin("id_ = prepared[\"id\"]", text)
    # No key, in any dict or metadata, is a respelled name.
    for respelled in ("class_", "lambda_", "id_")
        @test !occursin("\"" * respelled * "\"", text)
        @test !(respelled in kernel_inputs)
    end
    inputs_line = match(r"^INPUTS = (.*)$"m, text)[1]
    @test occursin("\"class\"", inputs_line) && occursin("\"id\"", inputs_line)
    # The bound of the range `1:class[1]` is computed from the prepared
    # `class`, under its own key.
    @test occursin("_rk_prepared[\"class\"]", text)

    # With definitions: `DATA` and `UNTRANSLATED` name a reserved input alike.
    reason = JAXEmissionError("no host translation in this test")
    with_defs = jax_source(jax_python_names; have, want = :lp, data = RAW,
                           definitions = [:class => reason])
    data_line = match(r"^DATA = (.*)$"m, with_defs.text)[1]
    untranslated_line = match(r"^UNTRANSLATED = (.*)$"m, with_defs.text)[1]
    @test occursin("\"class\": {", data_line) && !occursin("\"class_\"", data_line)
    @test startswith(untranslated_line, "{\"class\": ")

    if rk_jax_available()
        value, grad = expected(MU)
        names = inputs(src, "prepare_data")
        args = (MU, (RAW[Symbol(n)] for n in names)...)
        result = rk_jax_evaluate(src, :kernel, args; grad = true, prepare = true)
        @test result.value ≈ value
        @test result.grad ≈ grad
        traced = rk_jax_evaluate(src, :kernel, args; grad = true, prepare = :arguments)
        @test traced.value ≈ value
        @test traced.grad ≈ grad
        # Passed by name, as `prepare_data(**raw)`; the respelled name, a
        # missing reserved input and a positional one raise as Python does.
        raw = "{'class': [2, 1, 3], 'lambda': [0.5, 1.5, 1.0], 'id': [0.3, -0.2, 0.9], 'x': [1.0, 2.0, 0.5]}"
        @test python_prepare(src, "m.prepare_data(**$raw)") == "accepted"
        @test python_prepare(src, "m.prepare_data(x=[1.0], class_=[1], **{'lambda': [1.0], 'id': [1.0]})") ==
              "TypeError: prepare_data() got unexpected keyword arguments: 'class_'"
        @test python_prepare(src, "m.prepare_data(x=[1.0], **{'lambda': [1.0], 'id': [1.0]})") ==
              "TypeError: prepare_data() missing required arguments: 'class'"
        @test startswith(python_prepare(src, "m.prepare_data([1.0], [1], [1.0], [1.0])"), "TypeError: ")
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end

end
