module JAXEmptyPlateTests
using ReactiveKernels, Test
include("jax_python.jl")

@kernel ragged_gather(x::Vector{Float64}, rows::Vector{Vector{Int}}) = begin
    values = plate(rows, Ref(x)) do row, shared
        sum(shared[row])
    end
    total = sum(values)
end

@kernel ragged_scales(a::Float64, rows::Vector{Vector{Float64}}, offsets::Vector{Float64}) = begin
    scales = plate(eachindex(rows), Ref(rows), offsets) do i, rs, offset
        a ./ sqrt.(rs[i] .- offset)
    end
    totals = plate(scales) do row
        terms = plate(row) do scale
            a / scale + log(scale)
        end
        sum(terms)
    end
    total = sum(totals)
end

@traceable function checked_log(x)
    x > 0 || throw(ArgumentError("expected a positive entry"))
    log(x)
end

@kernel ragged_log(a::Float64, rows::Vector{Vector{Float64}}) = begin
    totals = plate(rows) do row
        sum(checked_log.(a .* row))
    end
    total = sum(totals)
end

@testset "empty JAX rows contribute zero through broadcasts and mapped cells" begin
    # Borrowing 4 from the first row would still divide by zero in the
    # second group, whose offset is 4. Both groups read the shared a.
    cases = (([[4.0, 9.0], Float64[], [2.0]], [0.0, 4.0, 1.0]),
             ([Float64[], Float64[]], [0.0, 4.0]),
             ([collect(1.0:32.0), [2.0]], [0.0, 1.0]))
    native = prepare(ragged_scales; want = :total)
    source(rows, offsets) = jax_source(ragged_scales; want = :total, data = (; rows, offsets))
    prepared = source(first(cases)...)
    raw = jax_source(ragged_scales; want = :total)
    a = 1.7
    for (rows, offsets) in cases
        @test source(rows, offsets).text == prepared.text
        value = native(a, rows, offsets)
        # On a real entry a/scale is independent of a, and d(log(scale))/da = 1/a.
        grad = sum(length, rows) / a
        if rk_jax_available()
            for (src, mode) in ((raw, false), (prepared, :arguments))
                got = rk_jax_evaluate(src, :kernel, (a, rows, offsets); grad = :joint, prepare = mode)
                @test got.joint_value ≈ value
                @test got.grad ≈ grad
            end
        end
    end
    if rk_jax_available()
        # The same zero denominator is observable when it belongs to a real row.
        @test isinf(rk_jax_evaluate(prepared, :kernel, (a, [[4.0]], [4.0]); prepare = :arguments).value)
        rows = [[1.0, 2.0], Float64[]]
        checked = jax_source(ragged_log; want = :total, data = (; rows))
        got = rk_jax_evaluate(checked, :kernel, (a, rows); grad = true, prepare = :arguments)
        @test got.value ≈ log(a) + log(2a)
        @test got.grad ≈ 2/a
        @test_throws ErrorException rk_jax_evaluate(checked, :kernel, (a, [[0.0]]); prepare = :arguments)
        mktempdir() do dir
            path = joinpath(dir, "model.py")
            write(path, prepared)
            driver = joinpath(@__DIR__, "jax_empty_rows.py")
            run(`$RK_JAX_PYTHON $driver $path`)
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end

@testset "empty JAX plates and ragged gathers perform no indexed reads" begin
    datasets = (([0.3, -0.7], [[1, 2], Int[]]),
                (Float64[], [Int[], Int[]]),
                (Float64[], Vector{Int}[]))
    srcs = [jax_source(ragged_gather; want = :total, data = (; rows)) for (_, rows) in datasets]
    @test all(s -> s.text == first(srcs).text, srcs)
    native = prepare(ragged_gather; want = :total)
    if rk_jax_available()
        for (x, rows) in datasets
            result = rk_jax_evaluate(first(srcs), :kernel, (x, rows);
                                     grad = true, prepare = :arguments)
            @test result.value ≈ native(x, rows)
            @test result.grad ≈ Float64[sum(count(==(i), row) for row in rows; init = 0)
                                 for i in eachindex(x)]
        end
        # CONTRIBUTING.md's lazy-branch contract: skipping an empty domain
        # must not suppress a real indexed read in a nonempty domain.
        @test_throws BoundsError native(Float64[], [[1]])
        @test_throws ErrorException rk_jax_evaluate(first(srcs), :kernel,
            (Float64[], [[1]]); prepare = :arguments)
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
end
