# Bound-data least-squares statistics (snag bound-data-suffi-7c8369bd, decision
# 1gq53h7): the collapsed residual sum of squares against a 256-bit row-wise
# reference, on toy data only.
module BoundStatisticsTests
using ReactiveKernels, LinearAlgebra, Random, Test
import Enzyme
using Enzyme: Const, Active, Duplicated
const RK = ReactiveKernels

relerr(a, ref) = Float64(abs(big(a) - ref) / abs(ref))
rss_ref(A, y, t) = setprecision(BigFloat, 256) do
    sum((big.(y) .- big.(A) * big.(t)) .^ 2)
end
rowwise(A, y, t) = sum(abs2, y .- A * t)

@testset "bound-data least-squares statistics" begin
    rng = Xoshiro(20261112)

    @testset "one block over every row: $(label)" for (label, N, K, shift) in (
            ("unshifted", 400, 3, 0.0), ("shifted response", 400, 3, 1e4))
        # An uncentred covariate: an ill-conditioned design, as a raw height is.
        X = [ones(N) randn(rng, N, K - 1) .+ 60]
        y = X * randn(rng, K) .+ 0.8 .* randn(rng, N) .+ shift
        s = RK._ls_statistics_dense(X, y)
        @test s.usable && s.blocks == 1 && s.k == K && s.rows == N
        fit = X \ y
        sd = 0.8 .* sqrt.(diag(inv(X' * X)))
        for offset in (1.0, 0.1, 0.0)
            t = fit .+ offset .* sd .* randn(rng, K)
            ref = rss_ref(X, y, t)
            v = RK._ls_rss_dense(s, (), t)
            # At least as accurate as the row-wise sum, within a small factor.
            @test relerr(v, ref) <= max(4 * relerr(rowwise(X, y, t), ref), 1e-15)
            # Live scalars come first: the intercept read as a scalar.
            @test RK._ls_rss_dense(s, (t[1],), t[2:end]) == v
        end
    end

    @testset "a rank-deficient design" begin
        N = 60
        Z = randn(rng, N, 2)
        A = [Z Z[:, 1]]
        y = randn(rng, N) .+ 3
        s = RK._ls_statistics_dense(A, y)
        t = randn(rng, 3)
        @test relerr(RK._ls_rss_dense(s, (), t), rss_ref(A, y, t)) < 1e-14
    end

    @testset "one block per index value: $(label)" for (label, shift) in (
            ("unshifted", 0.0), ("shifted response", 1e4))
        J, N = 30, 600
        g = rand(rng, 1:J, N)          # unsorted, not contiguous
        g[1] = J + 2                   # a value with one row
        f = Float64.(rand(rng, 0:1, N))
        f[g .== 3] .= 0.0              # a constant coefficient within one group
        y = randn(rng, N) .+ 1.3 .+ 0.5 .* f .+ shift
        alpha = randn(rng, J + 2) .+ 1.3 .+ shift
        beta = randn(rng, J + 2)
        s = RK._ls_statistics_grouped(g, (ones(N), f), y)
        @test s.usable && s.groups == sort(unique(g)) && s.blocks == length(unique(g))
        A = zeros(N, 2 * (J + 2))
        for i in 1:N
            A[i, g[i]] = 1.0
            A[i, J + 2 + g[i]] = f[i]
        end
        t = vcat(alpha, beta)
        ref = rss_ref(A, y, t)
        v = RK._ls_rss_grouped(s, (alpha, beta))
        @test relerr(v, ref) <= max(4 * relerr(rowwise(A, y, t), ref), 1e-15)
        # A gathered source too short for an index value raises, as the gather does.
        @test_throws BoundsError RK._ls_rss_grouped(s, (alpha[1:J], beta))
    end

    @testset "unusable bindings return NaN" begin
        s = RK._ls_statistics_dense([1.0 2.0; 3.0 Inf; 1.0 1.0; 2.0 2.0; 5.0 1.0], ones(5))
        @test !s.usable && isnan(RK._ls_rss_dense(s, (), [1.0, 2.0]))
        # Fewer rows than the statistics would cost: kept as rows.
        s = RK._ls_statistics_grouped(collect(1:10), (ones(10),), randn(rng, 10))
        @test !s.usable && isnan(RK._ls_rss_grouped(s, (zeros(10),)))
        s = RK._ls_statistics_dense(randn(rng, 3, 2), randn(rng, 3))
        @test !s.usable
    end

    @testset "ordinary Enzyme Reverse differentiates the collapsed sum" begin
        N, K = 300, 3
        X = [ones(N) randn(rng, N, K - 1) .+ 60]
        y = X * randn(rng, K) .+ 0.8 .* randn(rng, N) .+ 100
        s = RK._ls_statistics_dense(X, y)
        t = (X \ y) .+ 0.01 .* randn(rng, K)
        ref = setprecision(BigFloat, 256) do
            -2 .* (big.(X)' * (big.(y) .- big.(X) * big.(t)))
        end
        dv = zeros(K - 1)
        da = Enzyme.autodiff(Enzyme.Reverse, Const(RK._ls_rss_dense), Active, Const(s),
            Active((t[1],)), Duplicated(t[2:end], dv))[1][2][1]
        g = vcat(da, dv)
        @test Float64(norm(big.(g) .- ref) / norm(ref)) <=
            max(4 * Float64(norm(big.(-2 .* (X' * (y .- X * t))) .- ref) / norm(ref)), 1e-14)

        J, n = 12, 240
        idx = rand(rng, 1:J, n)
        f = randn(rng, n)
        yy = randn(rng, n) .+ 2
        sg = RK._ls_statistics_grouped(idx, (ones(n), f), yy)
        alpha, beta = randn(rng, J), randn(rng, J)
        dalpha, dbeta = zeros(J), zeros(J)
        Enzyme.autodiff(Enzyme.Reverse, Const(RK._ls_rss_grouped), Active, Const(sg),
            Duplicated((alpha, beta), (dalpha, dbeta)))
        r = yy .- (alpha[idx] .+ f .* beta[idx])
        @test dalpha ≈ [-2 * sum(r[idx .== j]) for j in 1:J] rtol = 1e-12
        @test dbeta ≈ [-2 * sum((r .* f)[idx .== j]) for j in 1:J] rtol = 1e-12
    end

    @testset "parameter counts are checked" begin
        X = randn(rng, 20, 2)
        s = RK._ls_statistics_dense(X, randn(rng, 20))
        @test_throws DimensionMismatch RK._ls_rss_dense(s, (), zeros(3))
        @test_throws DimensionMismatch RK._ls_statistics_dense(X, zeros(19))
    end
end

end
