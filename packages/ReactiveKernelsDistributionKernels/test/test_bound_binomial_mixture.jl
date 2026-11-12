module BoundBinomialMixtureTests
using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    binomial, binomial_product_mixture_logdensity
using DifferentiationInterface: AutoEnzyme
import Enzyme

@kernel common_probability(counts::Vector{Int}, trials::Matrix{Int},
                           weights::Vector{Float64}, p::Float64) = begin
    probabilities = fill(p, size(trials, 1), size(trials, 2))
    return binomial_product_mixture_logdensity(counts, trials, probabilities, weights)
end

function reference(counts, nmax, p)
    setprecision(BigFloat, 512) do
        probability = big(p)
        ns = max(1, maximum(counts; init=0)):nmax
        scores = [sum(log(big(Base.binomial(big(n), y))) +
            y * log(probability) + (n - y) * log1p(-probability)
            for y in counts; init=big(0.0)) for n in ns]
        top = maximum(scores)
        masses = exp.(scores .- top)
        value = top + log(sum(masses)) - log(big(nmax))
        slope = sum(masses .* [sum(counts) / probability -
            (length(counts) * n - sum(counts)) / (1 - probability) for n in ns]) /
            sum(masses)
        Float64(value), Float64(slope)
    end
end
function structure(ex)
    ex isa Expr || return (0, 0)
    result = (1, Int(ex.head in (:for, :while)))
    for child in ex.args
        result = result .+ structure(child)
    end
    result
end

@testset "bound mixture coefficients retain invalid-component guards" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    shapes = Tuple{Int,Int}[]
    for nmax in (17, 37)
        counts = [3, 5, 8, 11]
        trials = repeat(reshape(collect(1:nmax), 1, :), length(counts), 1)
        weights = fill(inv(nmax), nmax)
        saved = deepcopy((counts, trials, weights))
        k = prepare(common_probability; bound=(; counts, trials, weights),
            type_stability=:error)
        ad = prepare_ad(k, backend, 0.4; active=:p)
        for p in (0.2, 0.4, 0.7)
            value, slope = ad_value_and_gradient(ad, p)
            expected, expected_slope = reference(counts, nmax, p)
            @test value ≈ expected rtol=1e-12 atol=1e-12
            @test slope ≈ expected_slope rtol=1e-12 atol=1e-12
        end
        @test (counts, trials, weights) == saved
        # The coefficient cache includes finite guarded values at invalid
        # counts. It is computed at binding, outside the live scalar law.
        caches = [r.op.value for r in k.plan.recipes
            if r.op isa ReactiveKernels._BoundConstant &&
                r.op.value isa Vector{Vector{Float64}}]
        expected_coefficients = [[0 <= y <= n ?
            Float64(log(big(Base.binomial(big(n), y)))) : 0.0 for y in counts]
            for n in 1:nmax]
        @test any(cache -> cache ≈ expected_coefficients, caches)
        push!(shapes, structure(code_expr(k)))
    end
    @test shapes[1] == shapes[2]
    @test last(first(shapes)) >= 2
    for counts in (Int[], [0], [0, 1, 2])
        nmax = 4
        trials = repeat(reshape(collect(1:nmax), 1, :), length(counts), 1)
        weights = fill(inv(nmax), nmax)
        k = prepare(common_probability; bound=(; counts, trials, weights))
        ad = prepare_ad(k, backend, 0.4; active=:p)
        value, slope = ad_value_and_gradient(ad, 0.4)
        expected, expected_slope = reference(counts, nmax, 0.4)
        @test value ≈ expected atol=1e-13
        @test slope ≈ expected_slope atol=1e-13
    end
end

@testset "probability coefficient preserves lazy support and endpoints" begin
    k = prepare(binomial.logpdf; have=(:n, :p, :observed))
    for (n, p, y, expected) in ((0, NaN, 0, 0.0), (3, 0.0, 0, 0.0),
            (3, 1.0, 3, 0.0), (3, NaN, 4, -Inf), (3, NaN, -1, -Inf),
            (-1, NaN, 0, -Inf))
        @test isequal(k(n, p, y), expected)
    end
    for (n, p, y, expected_slope) in ((0, 0.4, 0, 0.0), (3, 0.0, 0, -3.0),
            (3, 1.0, 3, 3.0), (3, 0.4, 4, 0.0))
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), n, p, y; active=:p)
        @test ad_gradient(ad, n, p, y) ≈ expected_slope
    end
end
end
