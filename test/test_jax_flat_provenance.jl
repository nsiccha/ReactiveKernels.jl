module JAXFlatProvenanceTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

retained_second(pair) = getfield(pair, 2)

# The plate iterates a numeric vector, while each retained field gets its
# shape from a different shared ragged input. Tuple projection must keep
# each field's own lengths through the next plate.
@kernel retained_rows(a::Float64, x::Vector{Float64},
                      R::Vector{Vector{Float64}}, S::Vector{Vector{Float64}}) = begin
    groups = eachindex(x)
    paired = plate(groups, Ref(x), Ref(R), Ref(S)) do i, xs, rs, ss
        (a .* rs[i] .+ xs[i], (a + xs[i]) .* ss[i], a * xs[i])
    end
    r_cells = plate(paired) do pair
        pair[1]
    end
    s_cells = plate(paired) do pair
        retained_second(pair)
    end
    r_flat = isempty(r_cells) ? Float64[] : reduce(vcat, r_cells)
    s_flat = isempty(s_cells) ? Float64[] : reduce(vcat, s_cells)
    total = sum(abs2, r_flat) + sum(abs2, s_flat)
end

# Walk nested call/branch jaxprs too: a short outer trace alone would miss
# data-sized expansion in a reachable body.
trace_inventory(src, data::NamedTuple) = trace_inventory(src, (0.4, data.x, data.R, data.S))
function trace_inventory(src, args::Tuple; reverse = false)
    mktempdir() do dir
        write(joinpath(dir, "model.py"), src)
        write(joinpath(dir, "args.py"), _rk_jax_args(args, src))
        driver = joinpath(dir, "trace.py")
        write(driver, raw"""
import collections, importlib.util, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location("model", sys.argv[1])
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
import jax
exec(open(sys.argv[2]).read())
prepared = model.prepare_data(**raw_data(ARGS[1:]))
counts = collections.Counter()
def visit(value):
    if hasattr(value, "eqns"):
        for eqn in value.eqns:
            counts[str(eqn.primitive)] += 1
            visit(eqn.params)
    elif hasattr(value, "jaxpr"):
        visit(value.jaxpr)
    elif isinstance(value, dict):
        for item in value.values(): visit(item)
    elif isinstance(value, (tuple, list)):
        for item in value: visit(item)
f = jax.value_and_grad(model.total) if sys.argv[3] == "true" else model.total
visit(jax.make_jaxpr(f)(ARGS[0], **prepared))
print(json.dumps(counts, sort_keys=True))
""")
        read(`$RK_JAX_PYTHON $driver $(joinpath(dir, "model.py")) $(joinpath(dir, "args.py")) $reverse`, String)
    end
end

@testset "retained ragged plate fields keep their prepared lengths" begin
    data = [
        (; x = [0.2, -0.5, 0.9], R = [[1., 2.], [3.], [4., 5., 6.]],
           S = [[0.5], [1.5, 2.5], [3.5]]),
        (; x = [0.1, 0.7], R = [Float64[], [2., 4., 8., 16.]],
           S = [[1., 2.], Float64[]]),
        (; x = [0.1], R = [Float64[]], S = [Float64[]]),
        (; x = Float64[], R = Vector{Float64}[], S = Vector{Float64}[]),
        (; x = collect(range(-0.4, 0.7; length = 23)),
           R = [collect(1.0:(i % 7)) for i in 1:23],
           S = [collect(0.5:(i % 5)) for i in 1:23]),
    ]
    have = (:a, :x, :R, :S)
    sources = [jax_source(retained_rows; have,
        functions = (r = :r_flat, s = :s_flat, total = :total), data = d) for d in data]
    # CONTRIBUTING.md: binding more rows or longer cells never replicates
    # the plate body in RK emission. One module serves every binding below.
    @test all(src -> src.text == first(sources).text, sources)
    @test occursin("jax.vmap(", first(sources).text)
    src = first(sources)
    native = prepare(retained_rows; have, want = :total)
    if !rk_jax_available()
        @test_skip "RK_JAX_PYTHON not set"
    else
        # This fixture keeps the same primitive inventory with more rows
        # and wider cells; equality is evidence here, not a general policy
        # against legitimate backend simplification for other shapes.
        @test trace_inventory(src, first(data)) == trace_inventory(src, last(data))
        for d in data, a in (-0.7, 0.4)
            args = (a, d.x, d.R, d.S)
            original = deepcopy(args)
            expected_r = [a * r + x for (x, row) in zip(d.x, d.R) for r in row]
            expected_s = [(a + x) * s for (x, row) in zip(d.x, d.S) for s in row]
            @test rk_jax_evaluate(src, :r, args; prepare = :arguments).value ≈ expected_r
            @test rk_jax_evaluate(src, :s, args; prepare = :arguments).value ≈ expected_s
            result = rk_jax_evaluate(src, :total, args; grad = true, prepare = :arguments)
            @test result.value ≈ native(args...)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                Enzyme.Active(a), map(Enzyme.Const, Base.tail(args))...)[1][1]
            @test result.grad ≈ derivative
            @test args == original
        end
    end
end

# A plate cell returning a vector whose length its own expression fixes
# (`vcat` of scalars, a vector literal, a tuple field of that kind) fills its
# stored row in every cell, so `reduce(vcat, cells)` reads the stored array in
# order. No prepared ragged input carries those lengths (snag
# jax-source-value-3bf4c69e: BRM's `mean_y[i] = vcat(level[i], level[i] - fall[i])`).
# The helper is translated from its authored body, as BRM's `values` recipe is.
@traceable flatten_cells(cells) = isempty(cells) ? Float64[] : reduce(vcat, cells)
@traceable prepend_scalar(a, row) = vcat(a, row)

@kernel broadcast_rows(a::Float64, picks::Vector{Int},
                       X::Vector{Vector{Float64}}, Y::Vector{Vector{Float64}}) = begin
    records = plate(picks, Ref(X), Ref(Y)) do i, xs, ys
        (a, (a .* xs[i]) .* ys[i])
    end
    rows = plate(records) do record
        retained_second(record)
    end
    flat = flatten_cells(rows)
    total = sum(abs2, flat)
end

@traceable row_product(x, y) = x .* y
@kernel mapped_broadcast(a::Float64, X::Vector{Vector{Float64}}, Y::Vector{Vector{Float64}}) = begin
    rows = plate(X, Y) do x, y
        row_product(a .* x, y)
    end
    total = a > 0 ? sum(flatten_cells(rows)) : 0.0
end

@testset "flattening broadcast rows uses the selected lengths of every operand" begin
    # Neither operand supplies the flat size: singleton expansion changes
    # sides between groups; repeated, reordered and omitted rows count as read.
    data = [(; picks = [2, 1, 2], X = [[2.], [1., 3., 4.], [9.]], Y = [[5., 6.], [4.], [9.]]),
            (; picks = [2, 1, 3], X = [[2.], Float64[], [3.]], Y = [Float64[], [4.], [5.]]),
            (; picks = Int[], X = Vector{Float64}[], Y = Vector{Float64}[]),
            (; picks = collect(17:-1:1),
               X = [isodd(i) ? Float64[i] : collect(1.0:7) for i in 1:17],
               Y = [isodd(i) ? collect(1.0:(i % 6)) : Float64[i] for i in 1:17])]
    have = (:a, :picks, :X, :Y)
    functions = (flat = :flat, total = :total)
    region = ReactiveKernels._JaxSourceRegion(:combine_rows, collect(have), [:records],
        Dict(:records => :combined), "Combine the selected rows.")
    options = (; have, functions, source_regions = [region])
    src = jax_source(broadcast_rows; options..., data = first(data))
    @test occursin("def combine_rows", src.text)
    @test src.text == jax_source(broadcast_rows; options..., data = last(data)).text
    @test occursin("jax.vmap(", src.text)
    if rk_jax_available()
        native = prepare(broadcast_rows; have, want = :total)
        for reverse in (false, true)
            @test trace_inventory(src, (0.5, values(first(data))...); reverse) ==
                  trace_inventory(src, (0.5, values(last(data))...); reverse)
        end
        for d in data
            a = 0.5
            args = (a, values(d)...)
            expected = Float64[v for i in d.picks for v in (a .* d.X[i]) .* d.Y[i]]
            result = rk_jax_evaluate(src, :flat, args; prepare = :arguments)
            @test result.value == expected
            result = rk_jax_evaluate(src, :total, args; grad = true, prepare = :arguments)
            @test result.value ≈ native(args...)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active, Enzyme.Active(a),
                                         map(Enzyme.Const, Base.tail(args))...)[1][1]
            @test result.grad ≈ derivative
        end
        d = first(data)
        mapped = jax_source(mapped_broadcast; have = (:a, :X, :Y), want = :total,
                            name = :total, data = (:X, :Y))
        @test rk_jax_evaluate(mapped, :total, (0.5, d.X, d.Y); prepare = :arguments).value ≈
              sum(sum((0.5 .* x) .* y) for (x, y) in zip(d.X, d.Y))
        @test rk_jax_evaluate(mapped, :total, (-0.5, d.X, d.Y); prepare = :arguments).value == 0
        bad = ([[1., 2.]], [[1., 2., 3.]])
        @test_throws "reduce(vcat, cells)" rk_jax_evaluate(mapped, :total, (0.5, bad...);
                                                           prepare = :arguments)
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end

# A record field and a separate plate pass different per-row prefixes to
# the same helper. A prefix survives an empty tail.
@kernel prefixed_rows(a::Float64, x::Vector{Float64}, R::Vector{Vector{Float64}}) = begin
    records = plate(eachindex(x), Ref(x), Ref(R)) do i, xs, rs
        (a + xs[i], vcat(a * xs[i], a .* rs[i]))
    end
    singles = plate(records) do record
        retained_second(record)
    end
    doubles = plate(eachindex(x), Ref(x), Ref(R)) do i, xs, rs
        vcat(a, prepend_scalar(xs[i], a .* rs[i]))
    end
    first_flat = flatten_cells(singles)
    second_flat = flatten_cells(doubles)
    total = sum(abs2, first_flat) + sum(abs2, second_flat)
end

@testset "scalar prefixes count once per ragged row" begin
    data = [(; x = [0.2, -0.5, 0.9], R = [[1., 2.], [3.], Float64[]]),
            (; x = collect(range(-0.4, 0.7; length = 23)),
               R = [collect(1.0:(i % 7)) for i in 1:23]),
            (; x = [0.3], R = [Float64[]]),
            (; x = Float64[], R = Vector{Float64}[])]
    have = (:a, :x, :R)
    functions = (firsts = :first_flat, seconds = :second_flat, total = :total)
    sources = [jax_source(prefixed_rows; have, functions, data = d) for d in data]
    src = first(sources)
    @test all(s -> s.text == src.text, sources)
    @test occursin("jax.vmap(", src.text)
    @test occursin("do not add up", src.text)
    if rk_jax_available()
        native = prepare(prefixed_rows; have, want = :total)
        a = -0.7
        @test trace_inventory(src, (a, data[1].x, data[1].R)) ==
              trace_inventory(src, (a, data[2].x, data[2].R))
        for d in data
            args = (a, d.x, d.R)
            for (name, prefix) in ((:firsts, x -> [a * x]), (:seconds, x -> [a, x]))
                expected = Float64[v for (x, r) in zip(d.x, d.R) for v in vcat(prefix(x), a .* r)]
                actual = rk_jax_evaluate(src, name, args; prepare = :arguments).value
                @test length(actual) == length(expected)
                @test isempty(expected) || actual ≈ expected
            end
            result = rk_jax_evaluate(src, :total, args; grad = true, prepare = :arguments)
            @test result.value ≈ native(args...)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                Enzyme.Active(a), Enzyme.Const(d.x), Enzyme.Const(d.R))[1][1]
            @test result.grad ≈ derivative
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end

@kernel fixed_rows(a::Float64, x::Vector{Float64}) = begin
    groups = eachindex(x)
    pairs = plate(groups, Ref(x)) do i, xs
        vcat(a * xs[i], a - xs[i])
    end
    triples = plate(groups, Ref(x)) do i, xs
        [a + xs[i], a * xs[i], xs[i] - a]
    end
    tupled = plate(groups, Ref(x)) do i, xs
        (vcat(xs[i], a * xs[i]), a + xs[i])
    end
    fields = plate(tupled) do t
        t[1]
    end
    p_flat = isempty(pairs) ? Float64[] : reduce(vcat, pairs)
    q_flat = flatten_cells(triples)
    f_flat = flatten_cells(fields)
    total = sum(abs2, p_flat) + sum(q_flat) + sum(abs2, f_flat)
end

# One helper flattens full rows and rows of a prepared ragged input: each
# keeps its own translation.
@kernel shared_flatten(a::Float64, x::Vector{Float64}, R::Vector{Vector{Float64}}) = begin
    groups = eachindex(x)
    pairs = plate(groups, Ref(x)) do i, xs
        vcat(a * xs[i], a - xs[i])
    end
    rows = plate(groups, Ref(x), Ref(R)) do i, xs, rs
        a .* rs[i] .+ xs[i]
    end
    p_flat = flatten_cells(pairs)
    r_flat = flatten_cells(rows)
    total = sum(abs2, p_flat) + sum(abs2, r_flat)
end

@testset "fixed-length plate cells flatten row by row" begin
    data = [(; x = [0.2, -0.5, 0.9]), (; x = [0.1]), (; x = Float64[]),
            (; x = collect(range(-0.4, 0.7; length = 23)))]
    sources = [jax_source(fixed_rows; have = (:a, :x), data = d,
        functions = (p = :p_flat, q = :q_flat, f = :f_flat, total = :total)) for d in data]
    # CONTRIBUTING.md: more rows never replicate the plate bodies.
    @test all(src -> src.text == first(sources).text, sources)
    src = first(sources)
    @test occursin("jax.vmap(", src.text)
    # Full rows need neither the masked gather nor its runtime check.
    @test !occursin("jnp.nonzero", src.text)
    @test !occursin("do not add up", src.text)
    native = prepare(fixed_rows; have = (:a, :x), want = :total)
    if !rk_jax_available()
        @test_skip "RK_JAX_PYTHON not set"
    else
        @test trace_inventory(src, (0.4, first(data).x)) ==
              trace_inventory(src, (0.4, last(data).x))
        for d in data, a in (-0.7, 0.4)
            args = (a, d.x)
            original = deepcopy(args)
            expected_p = [v for x in d.x for v in (a * x, a - x)]
            expected_q = [v for x in d.x for v in (a + x, a * x, x - a)]
            expected_f = [v for x in d.x for v in (x, a * x)]
            for (name, expected) in ((:p, expected_p), (:q, expected_q), (:f, expected_f))
                value = rk_jax_evaluate(src, name, args; prepare = :arguments).value
                @test length(value) == length(expected)
                @test isempty(expected) || value ≈ expected
            end
            result = rk_jax_evaluate(src, :total, args; grad = true, prepare = :arguments)
            @test result.value ≈ native(args...)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                Enzyme.Active(a), Enzyme.Const(d.x))[1][1]
            @test result.grad ≈ derivative
            @test args == original
        end
    end

    ragged = [(; x = [0.2, -0.5, 0.9], R = [[1., 2.], [3.], Float64[]]),
              (; x = collect(range(-0.4, 0.7; length = 23)),
                 R = [collect(1.0:(i % 7)) for i in 1:23])]
    shared = [jax_source(shared_flatten; have = (:a, :x, :R), data = d,
        functions = (p = :p_flat, r = :r_flat, total = :total)) for d in ragged]
    @test first(shared).text == last(shared).text
    # The ragged rows keep their checked flattening beside the full rows.
    @test occursin("do not add up", first(shared).text)
    if rk_jax_available()
        native = prepare(shared_flatten; have = (:a, :x, :R), want = :total)
        for d in ragged, a in (-0.7, 0.4)
            args = (a, d.x, d.R)
            @test rk_jax_evaluate(first(shared), :p, args; prepare = :arguments).value ≈
                  [v for x in d.x for v in (a * x, a - x)]
            @test rk_jax_evaluate(first(shared), :r, args; prepare = :arguments).value ≈
                  [a * r + x for (x, row) in zip(d.x, d.R) for r in row]
            result = rk_jax_evaluate(first(shared), :total, args; grad = true, prepare = :arguments)
            @test result.value ≈ native(args...)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                Enzyme.Active(a), Enzyme.Const(d.x), Enzyme.Const(d.R))[1][1]
            @test result.grad ≈ derivative
        end
    end
end
end
