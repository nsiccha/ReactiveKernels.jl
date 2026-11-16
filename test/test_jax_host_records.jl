using ReactiveKernels, Test, Enzyme

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))
_rk_py_input(rows::AbstractVector{<:Tuple}) = "[" * join(map(_rk_py_input, rows), ", ") * "]"

@traceable function jhrec_rows(labels, values)
    labels == collect(labels) || error("collection changed labels")
    rows = collect(zip(labels, values))
    order = sortperm(rows)
    ordered = sort!(unique(rows))
    keyed = sort(rows; by = r -> (r[2], r[1]))
    sort!(rows; by = r -> (r[2], r[1]))
    rank = sortperm(rows; by = r -> r[1])
    total = 0.0
    for (i, row) in enumerate(rows)
        total += i * row[1] + row[2]
    end
    for i in eachindex(order)
        total += i * order[i] + i^2 * rank[i] + i * keyed[i][1]
    end
    for (i, row) in enumerate(ordered)
        total += i * (row[1] + row[2])
    end
    total
end

@traceable function jhrec_dict(labels, values)
    length(labels) == length(values) == length(values) || error("unequal input lengths")
    d = Dict{Tuple{Int,Int},Float64}()
    for i in eachindex(labels)
        key = (labels[i], i % 2)
        d[key] = get(d, key, 0.0) + values[i]
    end
    keys_in_order = sort!(collect(keys(d)))
    lookup = Dict(key => i for (i, key) in enumerate(keys_in_order))
    positions = getindex.(Ref(lookup), keys_in_order)
    seen = Set(key[1] for key in keys_in_order)
    result = 0.0
    for key in keys_in_order
        haskey(lookup, key) && (key[1] in seen) || error("missing key")
        result += lookup[key] * d[key]
    end
    result + sum(positions)
end

@traceable function jhrec_capture(labels, values)
    out = Float64[]
    append_value!(i, value) = begin
        value >= 0 || error("negative entry $value")
        push!(out, i * value)
    end
    i = 1
    while i <= length(values)
        if labels[i] == 0
            i += 1
            continue
        end
        append_value!(i, values[i])
        i += 1
    end
    sum(out)
end

@traceable function jhrec_chain(labels, values)
    visits = Int[]
    marker = 1
    middle() = begin
        push!(visits, marker)
        length(values)
    end
    last_value() = begin
        push!(visits, 2)
        values[1]
    end
    marker = 2
    good = 0 < middle() < last_value()
    (good ? 10.0 : 0.0) + sum(visits)
end

@traceable function jhrec_predicates(values)
    visits = Int[]
    positive(i) = begin
        push!(visits, i)
        i <= length(values) && values[i] > 0
    end
    good = all(positive, 1:(length(values) + 2))
    some = any(positive, 1:(length(values) + 2))
    finite = all(isfinite, values)
    nonnegative = all(>=(0), values)
    sum(visits) + (good ? 1 : 0) + (some ? 2 : 0) + (finite ? 4 : 0) + (nonnegative ? 8 : 0)
end

@traceable function jhrec_defaults(values; labels = Int[], extra = Float64[])
    sum(extra[labels .== length(values)])
end

@kernel jhrec_model(theta::Float64, labels::Vector{Int}, values::Vector{Float64}) = begin
    rows = jhrec_rows(labels, values)
    dictionaries = jhrec_dict(labels, values)
    captured = jhrec_capture(labels, values)
    chain = jhrec_chain(labels, values)
    predicates = jhrec_predicates(values)
    defaults = jhrec_defaults(values)
    result = theta * (rows + dictionaries + captured + predicates + defaults) + theta^2 * chain
    return result
end

@traceable function jhrec_lex(keys, values)
    rows = collect(zip(keys, values))
    order = sortperm(rows)
    sum((i * order[i] for i in eachindex(order)); init = 0) + length(unique(rows))
end
@kernel jhrec_lex_model(theta::Float64, keys::Vector{Float64}, values::Vector{Int}) = begin
    coefficient = jhrec_lex(keys, values)
    result = theta * coefficient
    return result
end

@traceable function jhrec_nested(labels, values)
    result = 0.0
    for i in eachindex(values)
        result += jhrec_rows(labels[i], values[i]) + jhrec_chain(labels[i], values[i])
    end
    result
end
@kernel jhrec_nested_model(theta::Float64, labels::Vector{Vector{Int}}, values::Vector{Vector{Float64}}) = begin
    coefficient = jhrec_nested(labels, values)
    result = theta * coefficient
    return result
end

@traceable function jhrec_raw_rows(input)
    Row = Tuple{Int,Float64}
    rows = Vector{Row}(undef, length(input))
    for (i, row) in enumerate(input)
        rows[i] = row
    end
    alias = rows
    sort!(rows; by = row -> (row[2], row[1]))
    unique!(rows)
    result = 0.0
    for (i, row) in enumerate(alias)
        result += i * row[1] + row[2]
    end
    result
end
@kernel jhrec_raw_model(theta::Float64, input::Vector{Tuple{Int,Float64}}) = begin
    coefficient = jhrec_raw_rows(input)
    result = theta * coefficient
    return result
end

@kernel jhrec_equal_model(theta::Float64, left::Vector{Int}, right::Vector{Float64}) = begin
    same = left == right
    different = left != right
    result = theta * (same ? 1 : 2) + theta^2 * (different ? 3 : 4)
    return result
end

@traceable function jhrec_effectful_key(input)
    visits = Int[]
    rows = sort(input; by = row -> begin
        push!(visits, 1)
        row[1]
    end)
    length(rows) + length(visits)
end
@kernel jhrec_effectful_model(theta::Float64, input::Vector{Tuple{Int,Float64}}) = begin
    coefficient = jhrec_effectful_key(input)
    result = theta * coefficient
    return result
end

@traceable function jhrec_rebind(values)
    total = 0
    change() = (total = length(values))
    change()
    total
end
@traceable jhrec_apply(f, values) = f(length(values))
@traceable function jhrec_pass_capture(values)
    read_at(i) = values[i]
    jhrec_apply(read_at, values)
end
@kernel jhrec_rebind_model(theta::Float64, values::Vector{Float64}) = begin
    coefficient = jhrec_rebind(values)
    result = theta * coefficient
    return result
end
@kernel jhrec_pass_capture_model(theta::Float64, values::Vector{Float64}) = begin
    coefficient = jhrec_pass_capture(values)
    result = theta * coefficient
    return result
end

function jhrec_parity(spec, data)
    have = (:theta, keys(data)...)
    source = jax_source(spec; have, want = :result, data)
    native = prepare(spec; have, want = :result)
    if rk_jax_available()
        args = (0.7, values(data)...)
        got = rk_jax_evaluate(source, :kernel, args; grad = true, prepare = :arguments)
        @test got.value ≈ native(args...)
        expected_grad = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
            Enzyme.Active(first(args)), map(Enzyme.Const, Base.tail(args))...)[1][1]
        @test only(got.grad) ≈ expected_grad
    end
    string(source)
end

function jhrec_structure(source, datasets)
    mktempdir() do dir
        write(joinpath(dir, "records.py"), source)
        write(joinpath(dir, "inputs.py"), "DATASETS = [" * join(map(_rk_py_input, datasets), ", ") * "]\n")
        driver = joinpath(dir, "check.py")
        write(driver, raw"""
import numpy as np
import jax
import records
exec(open(__file__.replace("check.py", "inputs.py")).read())

def regions(value):
    if hasattr(value, "eqns"):
        return len(value.eqns) + sum(regions(e.params) for e in value.eqns)
    if hasattr(value, "jaxpr"):
        return regions(value.jaxpr)
    if isinstance(value, dict):
        return sum(map(regions, value.values()))
    if isinstance(value, (tuple, list)):
        return sum(map(regions, value))
    return 0

counts = []
nonempty_counts = []
for raw in DATASETS:
    before = {k: v.copy() for k, v in raw.items()}
    prepared = records.prepare_data(**raw)
    for k in raw:
        np.testing.assert_array_equal(raw[k], before[k])
    counts.append(tuple(regions(jax.make_jaxpr(f)(0.7, **prepared))
                        for f in (records.kernel, jax.grad(records.kernel))))
    if len(raw["values"]):
        nonempty_counts.append(counts[-1])
# Empty reductions may add a scalar dtype conversion. Growth from one to
# four to twelve rows must not replicate any primal or derivative body.
assert len(set(nonempty_counts)) == 1, counts
for raw in ({"labels": [1], "values": [-2.0]}, {"labels": [1], "values": []}):
    try:
        records.prepare_data(**raw)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input was accepted")
print("HOST_RECORD_TRACE_COUNTS", counts)
""")
        read(`$RK_JAX_PYTHON $driver`, String)
    end
end

@testset "host tuple records, dictionaries, closures and comparisons" begin
    datasets = [(; labels = [2, 1, 2, 0], values = [4.0, 2.0, 3.0, 1.0]),
                (; labels = Int[], values = Float64[]),
                (; labels = [1], values = [4.0]),
                (; labels = repeat([2, 1, 2, 0], 3), values = repeat([4.0, 2.0, 3.0, 1.0], 3))]
    source = jax_source(jhrec_model; have = (:theta, :labels, :values), want = :result,
                        data = first(datasets))
    @test occursin("def prepare_data", string(source))
    @test occursin("while ", string(source))
    @test occursin("for ", string(source))
    @test occursin(r"def _append_value.*\(i, value, out\):", string(source))
    for data in datasets
        @test string(jax_source(jhrec_model; have = (:theta, :labels, :values), want = :result, data)) == string(source)
        native = prepare(jhrec_model; have = (:theta, :labels, :values), want = :result)
        args = (0.7, data.labels, data.values)
        before = deepcopy(args)
        expected = native(args...)
        if rk_jax_available()
            result = rk_jax_evaluate(source, :kernel, args; grad = true, prepare = :arguments)
            @test result.value ≈ expected
            expected_grad = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                Enzyme.Active(args[1]), Enzyme.Const(args[2]), Enzyme.Const(args[3]))[1][1]
            @test only(result.grad) ≈ expected_grad
            @test isequal(args, before)
        end
    end
    if rk_jax_available()
        trace_report = jhrec_structure(source, datasets)
        @test startswith(trace_report, "HOST_RECORD_TRACE_COUNTS")
        print(trace_report)
    end
    native = prepare(jhrec_model; have = (:theta, :labels, :values), want = :result)
    @test_throws ErrorException native(0.7, [1], [-2.0])
    @test_throws ErrorException native(0.7, [1], Float64[])
    lex = [(; keys = [NaN, 0.0, -0.0, NaN, 0.0, 0.0],
              values = [3, 2, 2, 3, 2^53 + 1, 2^53]),
           (; keys = Float64[], values = Int[])]
    @test jhrec_parity(jhrec_lex_model, first(lex)) == jhrec_parity(jhrec_lex_model, last(lex))
    ragged = [(; labels = [[2, 1], Int[], [1]], values = [[4.0, 2.0], Float64[], [3.0]]),
              (; labels = Vector{Int}[], values = Vector{Float64}[])]
    @test jhrec_parity(jhrec_nested_model, first(ragged)) == jhrec_parity(jhrec_nested_model, last(ragged))
    @test jhrec_parity(jhrec_raw_model, (; input = [(2, 4.0), (1, 2.0), (2, 4.0)])) ==
          jhrec_parity(jhrec_raw_model, (; input = Tuple{Int,Float64}[]))
    for data in ((; left=[2^53 + 1], right=[Float64(2^53)]),
                 (; left=[1], right=[1.0]), (; left=Int[], right=Float64[]),
                 (; left=[1], right=Float64[]), (; left=[0], right=[NaN]))
        jhrec_parity(jhrec_equal_model, data)
    end
    refusal = try
        string(jax_source(jhrec_effectful_model; have=(:theta,:input), want=:result,
                          data=(; input=[(1, 2.0)])))
    catch e
        e isa ReactiveKernels.JAXEmissionError || rethrow()
        sprint(showerror, e)
    end
    @test occursin("total tuple-field projection", refusal)
    for (spec, reason) in ((jhrec_rebind_model, "rebinding the captured local"),
                           (jhrec_pass_capture_model, "local named function with captures"))
        refusal = try
            string(jax_source(spec; have=(:theta,:values), want=:result, data=(; values=[2.0])))
        catch e
            e isa ReactiveKernels.JAXEmissionError || rethrow()
            sprint(showerror, e)
        end
        @test occursin(reason, refusal)
    end
    rk_jax_available() || @test_skip false
end
