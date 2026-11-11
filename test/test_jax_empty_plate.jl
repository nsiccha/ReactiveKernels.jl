module JAXEmptyPlateTests
using ReactiveKernels, Test
include("jax_python.jl")

@kernel ragged_gather(x::Vector{Float64}, rows::Vector{Vector{Int}}) = begin
    values = plate(rows, Ref(x)) do row, shared
        sum(shared[row])
    end
    total = sum(values)
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
