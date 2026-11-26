module JAXNestedInputTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@traceable function mapped_score(a, group)
    scores = map(group) do row
        isempty(row) ? a : a * sum(row)
    end
    sum(scores)
end

@kernel nested_rows(a::Float64, x::Vector{Vector{Vector{Int}}}) = begin
    by_group = plate(x) do group
        rows = plate(group) do row
            a + a^2 * sum(row)
        end
        sum(rows)
    end
    shared = plate(eachindex(x), Ref(x)) do i, groups
        group = groups[i]
        parts = plate(eachindex(group), Ref(group)) do j, rows
            row = rows[j]
            a * length(row) + sum(row)
        end
        length(group) + sum(parts)
    end
    ordered = plate(x) do rows
        joined = isempty(rows) ? Int[] : reduce(vcat, rows)
        sum(a .* joined .* eachindex(joined))
    end
    mapped = plate(x) do group
        mapped_score(a, group)
    end
    total = sum(by_group) + sum(shared) + sum(ordered) + sum(mapped)
end

@kernel selected_group(a::Float64, x::Vector{Vector{Vector{Int}}}, i::Int) = begin
    groups = plate(x) do rows
        joined = isempty(rows) ? Int[] : reduce(vcat, rows)
        a .* joined
    end
    flat = groups[i]
end

@kernel first_row(a::Float64, x::Vector{Vector{Vector{Int}}}) = begin
    score = a * sum(x[1][1])
end

@kernel record_groups(gain::Float64, batches::Vector{Vector{Vector{Float64}}}, base::Vector{Float64}) = begin
    cells = plate(batches, Ref(base), Ref(gain)) do rows, b, g
        (g .* b, rows)
    end
    terms = plate(cells) do cell
        sum(cell[1]) + sum(cell[2][1])
    end
    score = sum(terms)
    counts = map(cells) do cell
        length(cell[2])
    end
end

@testset "independent group and row lengths" begin
    data = [
        [[[2, 4], Int[]], [[3]], Vector{Int}[]],
        [Vector{Int}[], [Int[]]],
        Vector{Vector{Int}}[],
        [[collect(1:(j % 5)) for j in 1:(i % 7)] for i in 1:17],
    ]
    have = (:a, :x)
    outputs = (; by_group = :by_group, shared = :shared, ordered = :ordered, total = :total)
    plain = jax_source(nested_rows; have, functions = outputs)
    prepared(x) = jax_source(nested_rows; have, functions = outputs, data = (; x))
    bound = prepared(first(data))
    @test bound.text == prepared(last(data)).text
    @test occursin("jax.vmap(", bound.text)
    if rk_jax_available()
        native = prepare(nested_rows; have, want = :total)
        by_group = prepare(nested_rows; have, want = :by_group)
        shared = prepare(nested_rows; have, want = :shared)
        ordered = prepare(nested_rows; have, want = :ordered)
        mktempdir() do dir
            write(joinpath(dir, "direct.py"), plain)
            write(joinpath(dir, "bound.py"), bound)
            write(joinpath(dir, "chains.py"), jax_source(nested_rows; have, functions = outputs,
                                                        data = (; x = first(data)), plate_layout = :chains))
            selected = (; have = (:a, :x, :i), functions = (flat = :flat, named = (; flat = :flat)))
            write(joinpath(dir, "selected.py"), jax_source(selected_group; selected...))
            write(joinpath(dir, "selected_bound.py"), jax_source(selected_group; selected..., data = (; x = first(data))))
            write(joinpath(dir, "first_row.py"), jax_source(first_row; have, want = :score, name = :score))
            a = 0.7
            cases = map(data) do x
                derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                                             Enzyme.Active(a), Enzyme.Const(x))[1][1]
                (x, by_group(a, x), shared(a, x), ordered(a, x), native(a, x), derivative)
            end
            write(joinpath(dir, "cases.py"), "import numpy as np\na = $a\ncases = [" *
                  join(map(_rk_py_input, cases), ", ") * "]\n")
            @test success(pipeline(`$RK_JAX_PYTHON $(joinpath(@__DIR__, "jax_nested_inputs.py")) $dir`;
                                   stdout, stderr))
            @test success(pipeline(`$RK_JAX_PYTHON $(joinpath(@__DIR__, "jax_nested_chain_inputs.py")) $dir`;
                                   stdout, stderr))
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end

@testset "groups of ragged rows in a plate's tuple field" begin
    gain = 2.0
    base = [0.5, 1.5]
    batches = [[[1.0, 2.0], Float64[]], [[3.0]]]
    larger = [[collect(1.0:j) for j in 0:i] for i in 1:9]
    have = (:gain, :batches, :base)
    functions = (; score = :score, details = (; cells = :cells, counts = :counts))
    bound(x) = jax_source(record_groups; have, functions, data = (; batches = x, base))
    src = bound(batches)
    @test src.text == bound(larger).text
    f = prepare(record_groups; have, want = :score)
    @test f(gain, batches, base) == 14.0
    if rk_jax_available()
        mktempdir() do dir
            write(joinpath(dir, "bound.py"), src)
            write(joinpath(dir, "direct.py"), jax_source(record_groups; have, functions))
            cases = map((batches, [[Float64[]]], Vector{Vector{Float64}}[], larger)) do x
                grad = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active,
                                       Enzyme.Active(gain), Enzyme.Const(x), Enzyme.Const(base))[1][1]
                (x, f(gain, x, base), grad)
            end
            @test first(cases)[3] == 4.0
            write(joinpath(dir, "cases.py"), "import numpy as np\ncases = " * _rk_py_input(cases))
            write(joinpath(dir, "chains_records.py"), jax_source(record_groups; have, functions,
                  data = (; batches, base), plate_layout = :chains))
            driver = joinpath(@__DIR__, "jax_tuple_groups.py")
            @test success(pipeline(`$RK_JAX_PYTHON $driver $dir --chains`; stdout, stderr))
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
end
