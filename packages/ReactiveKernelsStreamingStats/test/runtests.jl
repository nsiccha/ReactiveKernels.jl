using ReactiveKernelsStreamingStats
using Statistics
using Test

@testset "mergeable streaming moments" begin
    left = fit([1.0, 2.0])
    right = fit([4.0, 8.0])
    combined = merge(left, right)
    direct = fit([1.0, 2.0, 4.0, 8.0])

    @test combined.n == 4
    @test mean(combined) == mean(direct)
    @test var(combined) == var(direct)
    @test merge(MomentsAccumulator(), combined) === combined
    @test_throws ArgumentError MomentsAccumulator{Float64}(2, 0.0, -1.0)
end

@testset "compiled method-bearing moments" begin
    statistics = online_moments(2)
    step!(statistics, [1.0, 3.0])
    step!(statistics, [3.0, 7.0])

    @test statistics.n == 2.0
    @test statistics.mean == [2.0, 5.0]
    @test statistics.var == [1.0, 4.0]
    @test var(statistics) == [2.0, 8.0]
    @test snapshot(statistics).mean == [2.0, 5.0]

    reset!(statistics)
    @test statistics.n == 0.0
end
