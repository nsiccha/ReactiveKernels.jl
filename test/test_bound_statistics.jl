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

    @testset "a rank-deficient design keeps its rows" begin
        N = 60
        Z = randn(rng, N, 2)
        A = [Z Z[:, 1]]
        y = randn(rng, N) .+ 3
        s = RK._ls_statistics_dense(A, y)
        @test !s.usable
        @test isnan(RK._ls_rss_dense(s, (randn(rng, 3),)))
    end

    @testset "one block per index value: $(label)" for (label, shift) in (
            ("unshifted", 0.0), ("shifted response", 1e4))
        J, N = 30, 600
        g = rand(rng, 1:J, N)          # unsorted, not contiguous
        f = Float64.(rand(rng, 0:1, N)) .+ 0.1 .* randn(rng, N)
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
        @test_throws BoundsError RK._ls_rss_grouped(s, (alpha[1:J-1], beta))
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

@testset "rank-deficient gathered blocks use the authored fallback" begin
    cache = PreparationCache()
    kernels = []
    # Include a full-rank control, a constant column, and a singleton group.
    for kind in (:independent, :constant, :singleton)
        idx = repeat(collect(1:3), 40)
        f = collect(range(-1.0, 1.0; length = length(idx)))
        kind === :constant && (f[idx .== 3] .= 0.0)
        kind === :singleton && (idx[1] = 4)
        y = 1.3 .+ 0.5 .* f
        alpha, beta = fill(1.1, 4), fill(0.4, 4)
        original = deepcopy((idx, f, y))
        k = prepare!(cache, ls_gathered; bound = (; idx, f, y))
        push!(kernels, k)
        s = only(filter(x -> x isa RK._LSStatistics, k.constants))
        @test s.usable == (kind === :independent)
        literal = prepare(ls_gathered; bound = (; idx, f, y), bound_statistics = false)
        v, g = _value_gradient(k, alpha, beta, 0.7)
        vl, gl = _value_gradient(literal, alpha, beta, 0.7)
        @test v ≈ vl rtol = 1e-12
        for (a, b) in zip(g, gl)
            @test a ≈ b rtol = 1e-12 atol = 1e-12
        end
        @test (idx, f, y) == original
        @test_throws BoundsError k(alpha[1:2], beta, 0.7)
    end
    @test allequal(typeof.(kernels))
    @test allequal(_shape.(kernels))
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

@kernel ls_weighted(x::Vector{Float64}, w::Vector{Float64}, y::Vector{Float64},
                    a::Float64, b::Float64, sigma::Float64) = begin
    mu = muladd.(b, x, a)
    scales = sigma .* w
    lp = plate(y, mu, scales) do yi, mi, si
        cell::Float64 = si > 0 ? normal(mi, si).logpdf(yi) : -Inf
        cell
    end
    total::Float64 = sum(lp)
    return total
end

@kernel ls_precision(x::Vector{Float64}, w::Vector{Float64}, y::Vector{Float64},
                     a::Float64, b::Float64, sigma::Float64) = begin
    mu = plate(a, b, x) do a, b, x
        a + *(b, x, x)
    end
    scales = plate(sigma, w) do s, n
        s / sqrt(n)
    end
    lp = plate(y, mu, scales) do yi, mi, si
        1.25 + 0.75 * ((yi - mi) / si)^2 - 2.0 * log(si)
    end
    total::Float64 = sum(lp)
    return total
end

@kernel ls_bounded_scale(w::Vector{Float64}, y::Vector{Float64}, a::Float64, sigma::Float64) = begin
    scales = sigma .* w
    lp = plate(y, a, scales) do yi, mi, si
        si > 1.0 ? -((yi - mi) / si)^2 : 0.0
    end
    total::Float64 = sum(lp)
    return total
end

@testset "separable row scales bind a weighted design" begin
    for (model, precision) in ((ls_weighted, false), (ls_precision, true))
        kernels = Any[]
        for n in (0, 1, 40, 400)
            x = randn(RNG, n)
            y = randn(RNG, n)
            w = [isodd(i) ? 0.75 : 1.5 for i in 1:n]
            k = prepare(model; bound = (; x, w, y))
            push!(kernels, k)
            stats = only(filter(v -> v isa RK._LSWeightedStatistics, k.constants))
            @test stats.fit.usable == (n >= 40)
            a, b = 0.2, -0.5
            if n == 40
                @test isempty(findings(k, (Float64, Float64, Float64)))
                ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), a, b, 1.1;
                    active = (:a, :b, :sigma), native_forward_batch = false)
                @test isempty(ad_call_findings(ad, a, b, 1.1))
            end
            for sigma in (1e-6, 1.1, 500.0)
                value, gradient = _value_gradient(k, a, b, sigma)
                ref, da, db, ds = 0.0, 0.0, 0.0, 0.0
                for i in eachindex(x)
                    feature = precision ? x[i]^2 : x[i]
                    s = precision ? sigma/sqrt(w[i]) : sigma*w[i]
                    residual = y[i] - (a + b*feature)
                    c, l, q = precision ? (1.25, -2.0, 0.75) : (-log(2pi)/2, -1.0, -0.5)
                    ref += c + l*log(s) + q*(residual/s)^2
                    slope = -2q*residual/s^2
                    da += slope
                    db += slope*feature
                    ds += (l - 2q*(residual/s)^2)/sigma
                end
                @test value ≈ ref rtol=1e-12
                @test collect(gradient) ≈ [da, db, ds] rtol=1e-12
            end
        end
        @test allequal(typeof.(kernels))
        @test allequal(_shape.(kernels))
    end
    x = randn(RNG, 35)
    y = randn(RNG, 35)
    for w in (fill(0.7,35), [0.0; ones(34)], [-1.0; ones(34)], [Inf; ones(34)])
        k = prepare(ls_weighted; bound=(; x,w,y))
        original = prepare(ls_weighted; bound=(; x,w,y), bound_statistics=false)
        for s in (-1.0, 0.0, 1e-160, 0.8, Inf)
            v, ref = k(0.1, 0.3, s), original(0.1, 0.3, s)
            @test isequal(v,ref) || isapprox(v,ref)
        end
    end
    # This guard changes truth within the positive-scale domain. The plate
    # must still test it on each row, in both the primal and reverse.
    w = [isodd(i) ? 0.5 : 2.0 for i in eachindex(y)]
    bounded = prepare(ls_bounded_scale; bound = (; w, y))
    @test !any(s -> s isa RK._LSWeightedStatistics, bounded.constants)
    value, gradient = _value_gradient(bounded, 0.2, 1.0)
    active = findall(>(1), w)
    residual = y[active] .- 0.2
    @test value ≈ -sum(abs2, residual ./ w[active])
    @test collect(gradient) ≈ [2sum(residual ./ w[active].^2), -2value]

    cache = PreparationCache()
    bindings = ((; x, w, y), (; x, w = [0.0; w[2:end]], y), (; x = 2x, w, y = -y))
    rebound = map(data -> prepare!(cache, ls_weighted; bound = data), bindings)
    @test allequal(typeof.(rebound)) && allequal(_shape.(rebound))
    @test map(k -> only(v.fit.usable for v in k.constants if v isa RK._LSWeightedStatistics),
              rebound) == (true, false, true)
    for (k, data) in zip(rebound, bindings)
        literal = prepare(ls_weighted; bound = data, bound_statistics = false)
        value, grad = _value_gradient(k, 0.2, -0.3, 0.8)
        ref, dref = _value_gradient(literal, 0.2, -0.3, 0.8)
        @test value ≈ ref
        @test collect(grad) ≈ collect(dref)
    end
end

@kernel ls_mixed(X::Matrix{Float64}, idx::Vector{Int}, y::Vector{Float64},
                 alpha::Vector{Float64}, beta::Vector{Float64}, sigma::Float64) = begin
    mu = alpha[idx] .+ X * beta
    lp = plate(y, mu) do yi, mi
        normal(mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end

@kernel ls_mixed_weighted(X::Matrix{Float64}, idx::Vector{Int}, w::Vector{Float64}, y::Vector{Float64},
                         alpha::Vector{Float64}, beta::Vector{Float64}, sigma::Float64) = begin
    mu = alpha[idx] .+ X * beta
    scales = sigma .* w
    lp = plate(y, mu, scales) do yi, mi, si
        si > 0 ? normal(mi, si).logpdf(yi) : -Inf
    end
    total::Float64 = sum(lp)
    return total
end

@testset "one grouped coefficient and a shared design" begin
    for weighted in (false, true)
        model = weighted ? ls_mixed_weighted : ls_mixed
        kernels = Any[]
        for (n, levels) in ((0, 5), (1, 1), (60, 15), (400, 100))
            X = randn(RNG, n, 3)
            y = randn(RNG, n)
            idx = shuffle(RNG, repeat(1:levels, cld(n, levels))[1:n])
            w = [isodd(i) ? 0.75 : 1.5 for i in 1:n]
            data = weighted ? (; X, idx, w, y) : (; X, idx, y)
            k = prepare(model; bound = data)
            push!(kernels, k)
            stats = weighted ? only(v.fit for v in k.constants if v isa RK._LSWeightedStatistics) :
                only(v for v in k.constants if v isa RK._LSMixedStatistics)
            @test stats.usable == (n >= 60)
            if stats.usable
                @test size(stats.cross) == (levels, 3)
                @test n > 100 || isempty(findings(k, (Vector{Float64}, Vector{Float64}, Float64)))
            end
            alpha, beta, sigma = randn(RNG, levels), randn(RNG, 3), 0.8
            value, grad = _value_gradient(k, alpha, beta, sigma)
            expected, da, db, ds = 0.0, zero(alpha), zero(beta), 0.0
            for i in eachindex(y)
                sd = weighted ? sigma*w[i] : sigma
                r = y[i] - (alpha[idx[i]] + dot(view(X, i, :), beta))
                expected += -0.5 * log(2pi) - log(sd) - 0.5 * (r/sd)^2
                da[idx[i]] += r/sd^2
                db .+= (r/sd^2) .* view(X,i,:)
                ds += ((r/sd)^2 - 1) / sigma
            end
            @test value ≈ expected
            @test grad[1] ≈ da
            @test grad[2] ≈ db
            @test grad[3] ≈ ds
        end
        @test allequal(typeof.(kernels))
        @test allequal(_shape.(kernels))
    end

    # A shared intercept belongs to the span of the grouped intercepts.
    # For these large opposite coefficients the literal rows cancel exactly.
    idx, X, y = repeat(1:4, 20), ones(80, 1), zeros(80)
    k = prepare(ls_mixed; bound = (; idx, X, y))
    @test !only(v for v in k.constants if v isa RK._LSMixedStatistics).usable
    value, grad = _value_gradient(k, fill(1e14, 4), [-1e14], 0.8)
    @test value ≈ length(y) * (-log(2pi)/2 - log(0.8))
    @test iszero(grad[1]) && iszero(grad[2])

    # A large coordinate can have a moderate contribution through a tiny
    # local column. Scale the residual by its column norm before squaring.
    local_column = fill(1e-80, length(idx))
    tiny = RK._ls_statistics_mixed(idx, local_column, 1e-80 .* randn(RNG, length(idx), 1),
                                   y, nothing)
    @test tiny.usable
    @test RK._ls_rss_mixed(tiny, fill(1e180, 4), (0.0,)) ≈ length(y) * 1e200
end

@kernel ls_product_scale(x::Vector{Int}, z::Vector{Int}, w::Vector{Float64},
                         v::Vector{Float64}, y::Vector{Float64}, b::Float64, sigma::Float64) = begin
    mu = plate(b, x, z) do b, x, z
        *(b, x, z)
    end
    scales = plate(sigma, w, v) do s, w, v
        *(s, w, v)
    end
    lp = plate(y, mu, scales) do yi, mi, si
        si > 0 ? normal(mi, si).logpdf(yi) : -Inf
    end
    total::Float64 = sum(lp)
    return total
end

@testset "moving data multipliers across a live value" begin
    n = 40
    # The authored multiplication promotes to Float64 before x*z can wrap.
    x = fill(isqrt(typemax(Int)) + 1, n)
    y = ones(n)
    b = 0.1 / Float64(first(x))^2
    for (w, v, sigma, usable) in ((2.0, 0.5, 0.8, true), (1e200, 1e-200, 1e120, false))
        data = (; x, z=x, w=fill(w,n), v=fill(v,n), y)
        k = prepare(ls_product_scale; bound = data)
        literal = prepare(ls_product_scale; bound = data, bound_statistics = false)
        @test only(s.fit.usable for s in k.constants if s isa RK._LSWeightedStatistics) == usable
        @test k(b, sigma) ≈ literal(b, sigma)
        if usable
            _, grad = _value_gradient(k, b, sigma)
            _, ref = _value_gradient(literal, b, sigma)
            @test collect(grad) ≈ collect(ref)
        end
    end
end

end
