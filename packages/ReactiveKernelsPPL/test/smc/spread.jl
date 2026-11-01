module SMCSpreadTests
using LinearAlgebra, Random, Statistics, Test
include("../../ext/smc_spread.jl")

function dense_scale(data)
    # A small dense test oracle; production never constructs this matrix.
    L = Matrix(Diagonal(data.diagonal))
    for j in reverse(eachindex(data.reflections))
        r = zeros(length(data.diagonal))
        r[data.indices[j]] = data.reflections[j]
        L -= 2r * (transpose(r) * L)
    end
    L
end

@testset "Native-coordinate reference spread" begin
    rng = Xoshiro(52)
    for d in (2, 5, 23), k in (0, 1, 4, 16)
        indices = [sort(randperm(rng, d)[1:rand(rng, 1:d)]) for _ in 1:k]
        data = (; diagonal=exp.(3randn(rng, d)), indices,
            reflections=[normalize(randn(rng, length(i))) for i in indices])
        original = deepcopy(data)
        L = dense_scale(data)
        @test _reference_sd(data) ≈ [norm(view(L, i, :)) for i in 1:d] rtol=1e-11
        @test data == original
        @test _reference_sd(_inflate_reference_scale(data, 3.0)) ≈ 3 .* _reference_sd(data)
    end
    # A coordinate permutation with a very narrow direction must not lose
    # that direction to subtraction of nearly equal covariance diagonals.
    data = (; diagonal=[1e-100, 1.0], indices=[[1, 2]],
        reflections=[[inv(sqrt(2)), inv(sqrt(2))]])
    L = dense_scale(data)
    @test all(isapprox.(_reference_sd(data), [norm(view(L, i, :)) for i in 1:2]; rtol=1e-12))
    @test _reference_sd(Float64[]) == Float64[]
    @test _reference_sd([0.1, 2.0]) == [0.1, 2.0]
    cloud = [1.0 3.0 6.0; -4.0 1.0 2.0]
    @test _cloud_sd(cloud) ≈ vec(std(cloud; dims=2, corrected=false))
end

@testset "Reference diagnostics stay linear in dimension" begin
    function allocated(d)
        rng = Xoshiro(159)
        data = (; diagonal=exp.(randn(rng, d)), indices=[collect(1:d) for _ in 1:4],
            reflections=[normalize(randn(rng, d)) for _ in 1:4])
        _reference_sd(data)
        @allocated _reference_sd(data)
    end
    small, large = allocated(512), allocated(1024)
    @test large < 2.5small
    @info "Reference spread allocations" small large
end
end
