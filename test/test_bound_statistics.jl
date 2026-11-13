# Bound-data least-squares statistics (snag bound-data-suffi-7c8369bd, decision
# 1gq53h7): the collapsed residual sum of squares against a 256-bit row-wise
# reference, on toy data only.
module BoundStatisticsTests
using ReactiveKernels, LinearAlgebra, Random, Test
import Enzyme
using Enzyme: Const, Active, Duplicated
using DifferentiationInterface: AutoEnzyme
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, student_t
include("type_stability_audit.jl")
using .TypeStabilityAudit: findings, ad_call_findings
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
            v = RK._ls_rss_dense(s, (t,))
            # At least as accurate as the row-wise sum, within a small factor.
            @test relerr(v, ref) <= max(4 * relerr(rowwise(X, y, t), ref), 1e-15)
            # Live scalars come first: the intercept read as a scalar.
            @test RK._ls_rss_dense(s, (t[1], t[2:end])) == v
        end
    end

    @testset "a rank-deficient design" begin
        N = 60
        Z = randn(rng, N, 2)
        A = [Z Z[:, 1]]
        y = randn(rng, N) .+ 3
        s = RK._ls_statistics_dense(A, y)
        t = randn(rng, 3)
        @test relerr(RK._ls_rss_dense(s, (t,)), rss_ref(A, y, t)) < 1e-14
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
        @test !s.usable && isnan(RK._ls_rss_dense(s, ([1.0, 2.0],)))
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
        rss(a, v) = RK._ls_rss_dense(s, (a, v))
        da = Enzyme.autodiff(Enzyme.Reverse, Const(rss), Active,
            Active(t[1]), Duplicated(t[2:end], dv))[1][1]
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
        @test_throws DimensionMismatch RK._ls_rss_dense(s, (zeros(3),))
        @test_throws DimensionMismatch RK._ls_statistics_dense(X, zeros(19))
    end
end

# --- the rewrite in a bound preparation ------------------------------------------

@kernel ls_dense(X::Matrix{Float64}, y::Vector{Float64}, beta::Vector{Float64},
                 sigma::Float64) = begin
    mu = X * beta
    lp = plate(y, mu) do yi, mi
        normal(mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
# A live intercept and a bound offset beside the product.
@kernel ls_intercept(X::Matrix{Float64}, offset::Vector{Float64}, y::Vector{Float64},
                     alpha::Float64, beta::Vector{Float64}, sigma::Float64) = begin
    mu = alpha .+ X * beta .+ offset
    lp = plate(y, mu) do yi, mi
        normal(mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
# Varying intercepts and slopes gathered through one bound index.
@kernel ls_gathered(idx::Vector{Int}, f::Vector{Float64}, y::Vector{Float64},
                    alpha::Vector{Float64}, beta::Vector{Float64}, sigma::Float64) = begin
    mu = alpha[idx] .+ f .* beta[idx]
    lp = plate(y, mu) do yi, mi
        normal(mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
# The live vector is computed for the location alone: its producer stays.
@kernel ls_derived(X::Matrix{Float64}, y::Vector{Float64}, theta::Vector{Float64}) = begin
    beta = theta[1:3]
    sigma = exp(theta[4])
    mu = X * beta
    lp = plate(y, mu) do yi, mi
        normal(mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
# Ports without declared types.
@kernel ls_untyped(X, y, beta, sigma) = begin
    mu = X * beta
    lp = plate(y, mu) do yi, mi
        normal(mi, sigma).logpdf(yi)
    end
    total = sum(lp)
    return total
end
# Not a squared residual: the plate keeps its location.
@kernel ls_student(X::Matrix{Float64}, y::Vector{Float64}, beta::Vector{Float64},
                   sigma::Float64) = begin
    mu = X * beta
    lp = plate(y, mu) do yi, mi
        student_t(4.0, mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end

_code(k) = string(code_expr(k))
# The program up to its gensym numbering.
_shape(k) = replace(_code(k), r"#\d+" => "#")
_collapsed(k) = occursin("_ls_rss", _code(k))
_reads_usable(x) = false
_reads_usable(x::QuoteNode) = x.value === :usable
_reads_usable(x::Expr) = any(_reads_usable, x.args)
function _statistics_arms(ex)
    ex isa Expr || return Tuple{Any,Any}[]
    Meta.isexpr(ex, :if, 3) && _reads_usable(ex.args[1]) &&
        return [(ex.args[2], ex.args[3])]
    reduce(vcat, (_statistics_arms(a) for a in ex.args); init = Tuple{Any,Any}[])
end
function _loops(ex)
    ex isa Expr || return 0
    (ex.head === :while) + sum(_loops, ex.args; init = 0)
end
# The density and its ordinary reverse gradient in every live argument.
function _value_gradient(k, args...)
    shadows = map(a -> a isa AbstractArray ? zero(a) : nothing, args)
    annotated = map((a, s) -> s === nothing ? Active(a) : Duplicated(copy(a), s), args, shadows)
    result = Enzyme.autodiff(Enzyme.ReverseWithPrimal, Const(k), Active, annotated...)
    actives = result[1]
    grads = map((a, s, g) -> s === nothing ? g : s, args, shadows, actives)
    (result[2], grads)
end
_agree(a, b; rtol) = all(isapprox.(a, b; rtol, atol = rtol * maximum(abs, b; init = 0.0)))

const RNG = Xoshiro(20261113)
dense_data(n, k; shift = 0.0) = (X = [ones(n) randn(RNG, n, k - 1) .+ 2],
                                 y = randn(RNG, n) .+ shift)

@testset "dense locations read frozen statistics" begin
    n = 200
    (; X, y) = dense_data(n, 3; shift = 50.0)
    offset = 0.1 .* randn(RNG, n)
    beta = X \ y .+ 0.05 .* randn(RNG, 3)
    for (kernel, bound, live) in (
            (ls_dense, (; X, y), (beta,)),
            (ls_intercept, (; X, offset, y), (0.3, beta)))
        k = prepare(kernel; bound)
        literal = prepare(kernel; bound, bound_statistics = false)
        @test _collapsed(k)
        @test !_collapsed(literal)
        # Eligibility is a runtime choice. Only its selected statistics arm
        # computes the collapsed sum; the other retains authored row work.
        arms = _statistics_arms(code_expr(k))
        @test length(arms) == 1
        @test occursin("_ls_rss", string(only(arms)[1]))
        @test !occursin("_ls_rss", string(only(arms)[2]))
        for sigma in (0.7, 3.0, 1e-3)
            v, g = _value_gradient(k, live..., sigma)
            vl, gl = _value_gradient(literal, live..., sigma)
            @test v ≈ vl rtol = 1e-12
            for (a, b) in zip(g, gl)
                @test _agree(a, b; rtol = 1e-10)
            end
        end
        # Where the factored total is not kept, the plate's own loop decides,
        # reading the location row by row.
        for sigma in (0.0, 1e-160, Inf)
            v, vl = k(live..., sigma), literal(live..., sigma)
            @test isequal(v, vl) || isapprox(v, vl; rtol = 1e-12)
        end
    end
end

@testset "ports without declared types" begin
    (; X, y) = dense_data(60, 2)
    k = prepare(ls_untyped; bound = (; X, y))
    literal = prepare(ls_untyped; bound = (; X, y), bound_statistics = false)
    @test _collapsed(k)
    @test k([0.1, 0.2], 0.8) ≈ literal([0.1, 0.2], 0.8) rtol = 1e-12
end

@testset "a live vector computed for the location alone" begin
    (; X, y) = dense_data(80, 3)
    theta = [X \ y; -0.2]
    k = prepare(ls_derived; bound = (; X, y))
    literal = prepare(ls_derived; bound = (; X, y), bound_statistics = false)
    @test _collapsed(k)
    v, g = _value_gradient(k, theta)
    vl, gl = _value_gradient(literal, theta)
    @test v ≈ vl rtol = 1e-12
    @test _agree(g[1], gl[1]; rtol = 1e-10)
end

@testset "gathered locations: one block per index value" begin
    J, n = 12, 300
    idx = rand(RNG, 1:J, n)
    f = Float64.(rand(RNG, 0:1, n)) .+ 0.1 .* randn(RNG, n)
    y = randn(RNG, n) .+ 1.3
    alpha, beta = randn(RNG, J) .+ 1.3, randn(RNG, J)
    k = prepare(ls_gathered; bound = (; idx, f, y))
    literal = prepare(ls_gathered; bound = (; idx, f, y), bound_statistics = false)
    @test _collapsed(k)
    # The optimized arm does not gather or group authored rows. Their source
    # remains in the lazy fallback for bindings where statistics cannot apply.
    arms = _statistics_arms(code_expr(k))
    @test length(arms) == 1
    fast = string(only(arms)[1])
    @test !occursin("_readonly_gather", fast) && !occursin("index_runs", fast)
    for sigma in (0.5, 2.0)
        v, g = _value_gradient(k, alpha, beta, sigma)
        vl, gl = _value_gradient(literal, alpha, beta, sigma)
        @test v ≈ vl rtol = 1e-12
        for (a, b) in zip(g, gl)
            @test _agree(a, b; rtol = 1e-10)
        end
    end
    # A source too short for an index value raises, as the gather does.
    @test_throws BoundsError k(alpha[1:J-1], beta, 0.5)
end

@testset "with_bound_statistics scopes the setting" begin
    (; X, y) = dense_data(60, 2)
    @test !_collapsed(with_bound_statistics(() -> prepare(ls_dense; bound = (; X, y)), false))
    @test _collapsed(with_bound_statistics(() -> prepare(ls_dense; bound = (; X, y))))
    # A keyword inside the scope decides for its own preparation.
    @test _collapsed(with_bound_statistics(false) do
        prepare(ls_dense; bound = (; X, y), bound_statistics = true)
    end)
end

@testset "rebinding recomputes the statistics" begin
    cache = PreparationCache()
    for shift in (0.0, 10.0)
        (; X, y) = dense_data(150, 3; shift)
        beta = X \ y
        k = prepare!(cache, ls_dense; bound = (; X, y))
        literal = prepare(ls_dense; bound = (; X, y), bound_statistics = false)
        @test _collapsed(k)
        @test k(beta, 0.9) ≈ literal(beta, 0.9) rtol = 1e-12
    end
end

@testset "an unusable binding selects authored arithmetic at runtime" begin
    template = prepare(ls_dense; bound = dense_data(20, 3))
    # Fewer rows than the statistics would cost, and data that are not finite.
    for (X, y) in ((randn(RNG, 4, 3), randn(RNG, 4)),
                   ([ones(20) [NaN; randn(RNG, 19)]], randn(RNG, 20)))
        k = prepare(ls_dense; bound = (; X, y))
        literal = prepare(ls_dense; bound = (; X, y), bound_statistics = false)
        @test _collapsed(k)
        @test typeof(k) === typeof(template)
        @test _shape(k) == _shape(template)
        beta = zeros(size(X, 2))
        @test isequal(k(beta, 0.8), literal(beta, 0.8))
    end
end

@testset "other laws keep their location" begin
    (; X, y) = dense_data(100, 3)
    k = prepare(ls_student; bound = (; X, y))
    @test !_collapsed(k)
    @test _shape(k) == _shape(prepare(ls_student; bound = (; X, y), bound_statistics = false))
end

@testset "data length does not grow the program" begin
    sizes = map((20, 2000)) do n
        (; X, y) = dense_data(n, 3)
        k = prepare(ls_dense; bound = (; X, y))
        @test _collapsed(k)
        _loops(k.ast)
    end
    @test sizes[1] == sizes[2]
end

@testset "concrete program and ordinary reverse" begin
    (; X, y) = dense_data(100, 3)
    k = prepare(ls_dense; bound = (; X, y))
    @test isempty(findings(k, (Vector{Float64}, Float64)))
    beta = zeros(3)
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), beta, 0.7;
        active = (:beta, :sigma), native_forward_batch = false)
    @test isempty(ad_call_findings(ad, beta, 0.7))
end

end
