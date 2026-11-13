# Runtime sizes and numerical values must not choose a compiled program type.
# Primary authority: ODLPlaus:precompile decision 0ikeidi; regression 1sq8f1k.
module BoundStatisticsRuntimeTests
using ReactiveKernels, Enzyme, DifferentiationInterface, MutatingFunctions, Test, LinearAlgebra
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
const RK = ReactiveKernels
const backend = AutoEnzyme(; mode = Enzyme.Reverse)

_structure(x) = (1, 0, 0)
_structure(x::RK._PlateProgram) = _structure(x.ast)
_structure(x::RK._PlateCall) = _structure(RK._program_expression(x.program))
function _structure(x::Expr)
    parts = map(_structure, x.args)
    (1 + sum(first, parts; init = 0),
     (x.head in (:for, :while)) + sum(p -> p[2], parts; init = 0),
     (x.head === :if) + sum(last, parts; init = 0))
end

@kernel dense(X::Matrix{Float64}, y::Vector{Float64}, beta::Vector{Float64}, sigma::Float64) = begin
    mu = X * beta
    lp = plate(y, mu) do yi, mi
        normal(mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
@testset "bound statistics use runtime eligibility" begin
    @testset "core dense" begin
        cache = PreparationCache()
        kernels = []
        structures = []
        beta = [0.1, -0.2]
        for n in (2, 20, 0, 200)
            X = hcat(ones(n), collect(1.0:n) ./ max(n, 1))
            y = fill(0.4, n)
            original = deepcopy((X, y))
            k = prepare!(cache, dense; bound = (; X, y))
            push!(kernels, k)
            push!(structures, _structure(k.ast))
            literal = prepare(dense; bound = (; X, y), bound_statistics = false)
            value = k(beta, 0.7)
            @test value ≈ literal(beta, 0.7)
            dbeta = zeros(2)
            result = Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(k), Enzyme.Active,
                Enzyme.Duplicated(copy(beta), dbeta), Enzyme.Active(0.7))
            residual = y - X * beta
            @test result[2] ≈ value
            @test dbeta ≈ X' * residual / 0.7^2
            @test result[1][2] ≈ sum(abs2, residual) / 0.7^3 - n / 0.7
            statistics = filter(x -> x isa RK._LSStatistics, k.constants)
            @test length(statistics) == 1
            @test only(statistics).usable == (n >= 4)
            @test (X, y) == original
        end
        @test allequal(typeof.(kernels))
        @test allequal(structures)
    end
end

@kernel two_dense(X::Matrix{Float64}, y::Vector{Float64}, W::Matrix{Float64},
                  z::Vector{Float64}, beta::Vector{Float64}, sigma::Float64) = begin
    m1 = X * beta
    p1 = plate(y, m1) do yi, mi
        normal(mi, sigma).logpdf(yi)
    end
    t1::Float64 = sum(p1)
    m2 = W * beta
    p2 = plate(z, m2) do zi, mi
        normal(mi, sigma).logpdf(zi)
    end
    t2::Float64 = sum(p2)
    total::Float64 = t1 + t2
    return total
end
@testset "independent statistics regions" begin
    cache = PreparationCache()
    kernels = []
    beta = [0.2, 0.3]
    for (n, m) in ((2, 20), (20, 2), (20, 20), (2, 2), (0, 20))
        X = hcat(ones(n), fill(0.2, n))
        W = hcat(ones(m), fill(0.3, m))
        y, z = fill(0.4, n), fill(0.5, m)
        k = prepare!(cache, two_dense; bound = (; X, y, W, z))
        push!(kernels, k)
        literal = prepare(two_dense; bound = (; X, y, W, z), bound_statistics = false)
        @test k(beta, 0.7) ≈ literal(beta, 0.7)
        dbeta = zeros(2)
        result = Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(k), Enzyme.Active,
            Enzyme.Duplicated(copy(beta), dbeta), Enzyme.Active(0.7))
        @test dbeta ≈ (X' * (y - X * beta) + W' * (z - W * beta)) / 0.7^2
    end
    @test allequal(typeof.(kernels))
end

@kernel typed_location(X::Matrix{Float64}, y::Vector{Float64}, beta::Vector{Float64}) = begin
    mu::Vector{Float32} = X * beta
    lp = plate(y, mu) do yi, mi
        normal(mi, 0.7).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end

@testset "a declared conversion is not affine producer arithmetic" begin
    beta = [0.123456789, -0.345678912]
    for n in (2, 20)
        X = hcat(ones(n), collect(1.0:n) ./ n)
        y = fill(0.4, n)
        k = prepare(typed_location; bound = (; X, y))
        literal = prepare(typed_location; bound = (; X, y), bound_statistics = false)
        residual = y .- Float32.(X * beta)
        reference = sum(-abs2(r) / (2 * 0.7^2) - log(0.7) - log(2pi)/2 for r in residual)
        @test k(beta) ≈ reference rtol = 1e-14
        @test k(beta) == literal(beta)
    end
end

const check_calls = Ref(0)
function checked_location(mu)
    check_calls[] += 1
    length(mu) <= 20 || throw(DimensionMismatch("too many location rows"))
    nothing
end
@kernel checked_dense(X::Matrix{Float64}, y::Vector{Float64}, beta::Vector{Float64}) = begin
    mu = X * beta
    lp = plate(y, mu) do yi, mi
        normal(mi, 0.7).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
RK.input_check!(checked_dense, :beta, checked_location, :mu)
@testset "checks inside a candidate region retain their ordering and count" begin
    for n in (2, 20, 21), statistics in (false, true)
        X = hcat(ones(n), fill(0.3, n))
        y = fill(0.4, n)
        native = prepare(checked_dense; bound = (; X, y), bound_statistics = statistics)
        kernels = n > 20 ? (native,) :
            (native, prepare_nonallocating(native, [0.1, 0.2]))
        for k in kernels
            check_calls[] = 0
            if n > 20
                @test_throws "too many location rows" k([0.1, 0.2])
            else
                @test isfinite(k([0.1, 0.2]))
            end
            @test check_calls[] == 1
            # The matrix product must fail before this check can see its result.
            @test_throws DimensionMismatch k([0.1, 0.2, 0.3])
            @test check_calls[] == 1
        end
    end
end

@kernel entry_checked_dense(X::Matrix{Float64}, y::Vector{Float64}, beta::Vector{Float64}) = begin
    mu = X * beta
    lp = plate(y, mu) do yi, mi
        normal(mi, 0.7).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
function checked_point(beta)
    check_calls[] += 1
    length(beta) == 2 || throw(DimensionMismatch("expected two coefficients"))
    nothing
end
RK.input_check!(entry_checked_dense, :beta, checked_point, :beta)
@testset "entry checks run once across retained statistics arms" begin
    for n in (2, 20), statistics in (false, true)
        X = hcat(ones(n), fill(0.3, n))
        y = fill(0.4, n)
        native = prepare(entry_checked_dense; bound = (; X, y), bound_statistics = statistics)
        if statistics
            @test count(r -> r.op isa RK._LSRegion, native.plan.recipes) == 1
        end
        reused = prepare_nonallocating(native, [0.1, 0.2])
        for k in (native, reused)
            check_calls[] = 0
            @test isfinite(k([0.1, 0.2]))
            @test check_calls[] == 1
            @test_throws "expected two coefficients" k([0.1, 0.2, 0.3])
            @test check_calls[] == 2
        end
    end
end

# The first region starts before z is produced, and the second starts before
# the first is summed. The merged region must expose z's storage to reuse.
@kernel overlapping(X::Matrix{Float64}, y::Vector{Float64}, W::Matrix{Float64},
                    w::Vector{Float64}, beta::Vector{Float64}) = begin
    m1 = X * beta
    z = plate(beta) do b
        exp(b)
    end
    m2 = W * z
    p1 = plate(y, m1) do yi, mi
        normal(mi, 0.7).logpdf(yi)
    end
    p2 = plate(w, m2) do wi, mi
        normal(mi, 0.7).logpdf(wi)
    end
    t1::Float64 = sum(p1)
    t2::Float64 = sum(p2)
    total::Float64 = t1 + t2
    return total
end
function ad_bytes(prepared, beta)
    g = similar(beta)
    for _ in 1:4
        ad_value_and_gradient!(prepared, g, beta)
    end
    @allocated ad_value_and_gradient!(prepared, g, beta)
end
include("type_stability_audit.jl")
using .TypeStabilityAudit: findings, ad_call_findings

@testset "overlapping regions preserve storage and concrete programs" begin
    beta = [0.1, -0.2]
    kernels = []
    gradient_types = []
    structures = []
    for (n, m) in ((2, 20), (20, 2), (20, 20), (0, 20))
        X = hcat(ones(n), fill(0.2, n))
        W = hcat(ones(m), fill(0.3, m))
        y, w = fill(0.4, n), fill(0.5, m)
        bound = (; X, y, W, w)
        native = prepare(overlapping; bound)
        push!(kernels, native)
        push!(structures, _structure(native.ast))
        regions = [r.op for r in native.plan.recipes if r.op isa RK._LSRegion]
        @test length(regions) == 1
        @test length(only(regions).statistics) == 2
        @test any(r -> r.op isa RK._AuthoredPlateOp &&
                       any(v -> v.name === :z, r.outputs), only(regions).fast.recipes)
        @test RK._nonallocating_plan_supported(native.plan)
        @test isempty(findings(native, (typeof(beta),)))
        literal = prepare(overlapping; bound, bound_statistics = false)
        @test native(beta) ≈ literal(beta)
        ads = map((literal, native)) do k
            reused = prepare_nonallocating(k, beta)
            @test reused(beta) ≈ literal(beta)
            @test isempty(findings(reused, (typeof(beta),)))
            ad = prepare_ad(reused, backend, beta; active = :beta)
            @test isempty(ad_call_findings(ad, beta))
            g = similar(beta)
            value, _ = ad_value_and_gradient!(ad, g, beta)
            z = exp.(beta)
            expected = (X' * (y - X * beta) + z .* (W' * (w - W * z))) / 0.7^2
            @test value ≈ literal(beta)
            @test g ≈ expected
            ad
        end
        push!(gradient_types, typeof.(ads))
        @test ad_bytes(ads[2], beta) <= ad_bytes(ads[1], beta)
    end
    @test allequal(typeof.(kernels))
    @test allequal(first.(gradient_types))
    @test allequal(last.(gradient_types))
    @test allequal(structures)
end

@kernel conditional_storage(x::Vector{Float64}, weights::Vector{Float64}, valid::Bool) = begin
    total::Float64 = if valid
        sum(weights .* log.(x))
    else
        sum(abs2, x)
    end
    return total
end
@testset "unseeded array destinations preserve lazy ordinary reverse" begin
    prepared = map((false, true)) do valid
        x = fill(valid ? 0.5 : -0.5, 3)
        weights = [1.0, 2.0, 3.0]
        k = prepare_nonallocating(prepare(conditional_storage), x, weights, valid)
        prepare_ad(k, backend, x, weights, valid; active = :x)
    end
    @test allequal(typeof.(prepared))
    for ad in prepared, n in (0, 3, 7), valid in (false, true)
        x = fill(valid ? 0.5 : -0.5, n)
        weights = collect(1.0:n)
        original = deepcopy((x, weights))
        g = similar(x)
        value, _ = ad_value_and_gradient!(ad, g, x, weights, valid)
        @test value ≈ (valid ? sum(weights .* log.(x)) : sum(abs2, x))
        @test g ≈ (valid ? weights ./ x : 2 .* x)
        @test (x, weights) == original
    end
end

end # module
