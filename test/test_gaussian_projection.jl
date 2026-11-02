using Test, Random, LinearAlgebra, Statistics
import ReactiveKernels as RK

@testset "Compact Gaussian marginal and conditional algebra" begin
    rng = Xoshiro(164)
    mean = randn(rng, 9)
    diagonal = exp.(randn(rng, 9))
    factors = randn(rng, 9, 3)
    core = [0.3 0.1 0.0; 0.1 0.6 0.0; 0.0 0.0 0.2]
    covariance = Diagonal(diagonal) + factors * core * factors'
    g, diagnostics = RK._compact_gaussian(mean, diagonal, factors, core)
    dense(g) = Diagonal(g.diagonal) + RK._gaussian_factors(g) *
        Diagonal(g.values) * RK._gaussian_factors(g)'
    @test dense(g) ≈ covariance
    @test diagnostics.relative_loss == 0
    x = randn(rng, 9)
    @test RK._gaussian_covariance_mul(g, x) ≈ covariance * x
    @test RK._gaussian_solve(g, x) ≈ covariance \ x
    @test RK._gaussian_logdensity(g, x) ≈
        -(9log(2pi) + logdet(covariance) + dot(x-mean, covariance \ (x-mean))) / 2
    L = hcat([RK._gaussian_color(g, Float64.(1:9 .== i)) for i in 1:9]...)
    @test L * L' ≈ covariance
    @test RK._gaussian_whiten(g.diagonal, g.vectors, g.values, L*x) ≈ x
    @testset "Reflection scale covariance, including non-prefix selections" begin
        scale = [0.3, 1.2, 2.0, 0.7]
        indices = [[1,3,4], [2,3]]
        reflections = [normalize([0.3,0.1,0.7]), normalize([0.4,0.8])]
        L = Matrix(Diagonal(scale))
        for j in 2:-1:1
            u = zeros(4)
            u[indices[j]] = reflections[j]
            L = (I - 2u*u') * L
        end
        projected = RK._gaussian_from_reflections(zeros(4), scale, indices, reflections)
        @test dense(projected) ≈ L*L'
        @test dense(RK._gaussian_marginal(projected,[4,1])) ≈ (L*L')[[4,1],[4,1]]
        @test !(L[[4,1],[4,1]] * L[[4,1],[4,1]]' ≈ (L*L')[[4,1],[4,1]])
    end
    for active in ([5, 2, 8], Int[], collect(1:9))
        marginal = RK._gaussian_marginal(g, active)
        @test marginal.mean == mean[active]
        @test dense(marginal) ≈ covariance[active, active]
        conditional = RK._gaussian_conditional(g, active)
        inactive = conditional.inactive
        B = covariance[inactive, active] / covariance[active, active]
        @test conditional.factors * conditional.regression ≈ B atol=1e-12
        @test dense(conditional.residual) ≈
            covariance[inactive, inactive] - B * covariance[active, inactive]
        same, _ = RK._gaussian_replace_marginal(g, active, marginal)
        @test same.mean ≈ mean
        @test dense(same) ≈ covariance
    end
    @test_throws ArgumentError RK._gaussian_marginal(g, [2,2])
    @test_throws ArgumentError RK._gaussian_marginal(g, [0])
    @test_throws ArgumentError RK._compact_gaussian(zeros(2), [1.0, 0.0])
    @test_throws ArgumentError RK._compact_gaussian(zeros(2), ones(2),
        ones(2,1), fill(-1.0,1,1))

    # Reordered active coordinates and fresh coordinates interleaved in the
    # replacement are deliberately unlike the full-joint output ordering.
    active, slots = [5,2,8], [4,1,3]
    replacement, _ = RK._compact_gaussian(randn(rng,5), exp.(randn(rng,5)),
        randn(rng,5,2), [0.3 0.1;0.1 0.2])
    next, info = RK._gaussian_replace_marginal(g, active, replacement; new_active=slots)
    inactive = setdiff(1:9, active)
    target = [2,10,8,5,11]
    B = covariance[inactive, active] / covariance[active, active]
    expected_mean = zeros(11)
    expected_mean[target] = replacement.mean
    expected_mean[inactive] = mean[inactive] + B * (replacement.mean[slots]-mean[active])
    expected = zeros(11,11)
    expected[target,target] = dense(replacement)
    expected[inactive,target] = B * dense(replacement)[slots,:]
    expected[target,inactive] = expected[inactive,target]'
    expected[inactive,inactive] = covariance[inactive,inactive] +
        B * (dense(replacement)[slots,slots]-covariance[active,active]) * B'
    @test next.mean ≈ expected_mean
    @test dense(next) ≈ expected
    @test info.relative_loss == 0
    @test RK._gaussian_solve(next, ones(11)) ≈ expected \ ones(11)
    @test mean == g.mean # inputs own their copies
    mean[1] += 100
    @test mean[1] != g.mean[1]

    @testset "Independent and correlated inactive oracle" for rho in (0.0, 0.8)
        U = [1.0 1.0; 1.0 -1.0] / sqrt(2)
        old, _ = RK._compact_gaussian(zeros(2), ones(2), U, Diagonal([rho,-rho]))
        new, _ = RK._compact_gaussian([1.0], [0.5])
        posterior, _ = RK._gaussian_replace_marginal(old, [1], new)
        @test posterior.mean ≈ [1.0, rho]
        @test dense(posterior) ≈ [0.5 rho/2; rho/2 1-rho^2/2]
        c = RK._gaussian_conditional(old, [1])
        draws = [only(RK._gaussian_recover(rng,c,[1.0])) for _ in 1:10000]
        @test abs(Statistics.mean(draws) - rho) < 0.04
        @test abs(var(draws; corrected=false) - (1-rho^2)) < 0.04
    end

    @testset "Bounded repeated replacement and untouched independent factors" begin
        g, _ = RK._compact_gaussian(zeros(100), ones(100))
        replacement, _ = RK._compact_gaussian([1.0,2.0], [0.5,0.4],
            reshape([0.1,0.2],2,1), ones(1,1))
        for _ in 1:30
            g, _ = RK._gaussian_replace_marginal(g, [3,8], replacement; maxrank=4)
            @test length(g.values) <= 4
            @test g.mean[setdiff(1:100,[3,8])] == zeros(98)
            @test diag(dense(g))[setdiff(1:100,[3,8])] ≈ ones(98)
        end
        compressed, info = RK._gaussian_replace_marginal(g, [3,8], replacement; maxrank=0)
        @test isempty(compressed.values)
        @test info.discarded_rank > 0
        @test info.relative_loss > 0
    end
end

@testset "Gaussian retained storage and boundary allocation are linear" begin
    function boundary(n)
        factors = zeros(n, 2)
        factors[:,1] .= 1 / sqrt(n)
        factors[1:2:end,2] .= 1 / sqrt(n)
        old, _ = RK._compact_gaussian(zeros(n), ones(n), factors, [0.2 0.1;0.1 0.3])
        replacement, _ = RK._compact_gaussian([0.1,0.2], [0.4,0.5])
        RK._gaussian_replace_marginal(old, [2,5], replacement; maxrank=4)
    end
    boundary(16)
    small = first(boundary(1000))
    large = first(boundary(10000))
    @test Base.summarysize(large) < 10.1Base.summarysize(small)
    a = @allocated boundary(1000)
    b = @allocated boundary(10000)
    @test b < 11a
    @info "Gaussian boundary scaling" small_bytes=a large_bytes=b retained_small=Base.summarysize(small) retained_large=Base.summarysize(large)
end
