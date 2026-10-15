using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    zero_inflated_binomial_logit_product_mixture_logdensity
using Distributions: Binomial, logpdf
import DifferentiationInterface as DI
import Enzyme

# Runs after `test_product_mixture.jl`, whose 512-bit fold
# `product_mixture_logit_reference(cell, observed, args, weights)` it reuses.

# Zero-inflated Binomial cells with both probabilities on the log-odds scale.
# The reference cell is the exact log mass at success log-odds `l` and
# zero-inflation log-odds `z`, in 512-bit arithmetic (`Base.binomial`, as in
# `test_product_mixture.jl`).
function zib_logit_reference_cell(y, n, l, z)
    l, z = big(l), big(z)
    logzi, log1mzi = -log1p(exp(-z)), -log1p(exp(z))
    count = log(Base.binomial(big(n), big(y))) +
        (y == 0 ? big(0.0) : y * -log1p(exp(-l))) +
        (y == n ? big(0.0) : (n - y) * -log1p(exp(l)))
    y > 0 && return log1mzi + count
    a, b = logzi, log1mzi + count
    top = max(a, b)
    top == -Inf ? top : top + log(exp(a - top) + exp(b - top))
end

@testset "zero-inflated Binomial product mixture streams component scores" begin
    law = prepare(zero_inflated_binomial_logit_product_mixture_logdensity)
    for (positions, components) in ((4, 6), (9, 13), (0, 6))
        counts = mod.(collect(1:positions), 5)
        trials = fill(4, positions, components)
        logits = [sin(i + 2k) for i in 1:positions, k in 1:components]
        zi_logits = [cos(2i - k) for i in 1:positions, k in 1:components]
        weights = fill(1 / components, components)
        expected = product_mixture_logit_reference(zib_logit_reference_cell,
            counts, (trials, logits, zi_logits), weights)
        @test law(counts, trials, logits, zi_logits, weights) ≈ Float64(expected) atol=1e-14 rtol=1e-13
        @test law_allocated(law, counts, trials, logits, zi_logits, weights) == 0
    end
    @test law([0], zeros(Int, 1, 0), zeros(1, 0), zeros(1, 0), Float64[]) == -Inf
end

@testset "zero-inflated Binomial whole-event law keeps both log-odds tails" begin
    law = prepare(zero_inflated_binomial_logit_product_mixture_logdensity)
    weights = [0.7, 0.3]
    counts = [0, 2, 0, 4, 1]
    trials = fill(4, 5, 2)
    logits = [0.3 -1.0; -0.8 0.4; 1.2 0.0; 0.5 -0.2; -1.5 0.9]
    zi_logits = [-0.4 1.1; 0.2 -0.6; -1.3 0.0; 0.7 0.3; -0.1 -2.0]

    # Moderate values against an independent probability-space oracle that
    # composes Distributions' Binomial mass with the zero-inflation mixture.
    function probability_oracle(counts, trials, logits, zi_logits, weights)
        scores = map(eachindex(weights)) do k
            log(weights[k]) + sum(eachindex(counts)) do i
                p = 1 / (1 + exp(-logits[i, k]))
                zi = 1 / (1 + exp(-zi_logits[i, k]))
                binom = Binomial(trials[i, k], p)
                counts[i] == 0 ? log(zi + (1 - zi) * exp(logpdf(binom, 0))) :
                    log1p(-zi) + logpdf(binom, counts[i])
            end
        end
        top = maximum(scores)
        top + log(sum(exp.(scores .- top)))
    end
    actual = law(counts, trials, logits, zi_logits, weights)
    @test actual ≈ probability_oracle(counts, trials, logits, zi_logits, weights) rtol=1e-13
    # One component governs the whole vector: not a per-cell mixture.
    independent = sum(eachindex(counts)) do i
        law(counts[i:i], trials[i:i, :], logits[i:i, :], zi_logits[i:i, :], weights)
    end
    @test abs(actual - independent) > 0.05

    # Saturated success and zero-inflation log-odds beside a structural-zero
    # component (success logit -Inf), against the 512-bit reference.
    for (l, z) in ((40.0, -40.0), (-40.0, 40.0), (38.0, 38.0), (-45.0, -45.0))
        sat_logits = hcat([0.3, l, 1.2, 0.5, -1.5], fill(-Inf, 5))
        sat_zi = hcat([-0.4, z, -1.3, z, -0.1], zeros(5))
        value = law(counts, trials, sat_logits, sat_zi, weights)
        reference = product_mixture_logit_reference(zib_logit_reference_cell,
            counts, (trials, sat_logits, sat_zi), weights)
        @test isfinite(value)
        @test abs(big(value) - reference) <= 1e-14 * abs(reference)
    end
    # An all-zero event is carried partly by the structural-zero component;
    # with a positive count that component contributes nothing.
    zeros_event = [0, 0, 0]
    present = hcat([0.4, -0.2, 1.0], fill(-Inf, 3))
    zi_present = hcat([-0.5, 0.3, 30.0], fill(0.0, 3))
    @test law(zeros_event, fill(4, 3, 2), present, zi_present, weights) ≈
        Float64(product_mixture_logit_reference(zib_logit_reference_cell,
            zeros_event, (fill(4, 3, 2), present, zi_present), weights)) rtol=1e-14
    @test law([0, 1, 0], fill(4, 3, 2), present, zi_present, weights) ≈
        log(weights[1]) + sum(Float64(zib_logit_reference_cell(y, 4, l, z))
            for (y, l, z) in zip([0, 1, 0], present[:, 1], zi_present[:, 1])) rtol=1e-14
    # An empty event leaves log(sum(weights)), zero up to rounding.
    @test law(Int[], zeros(Int, 0, 2), zeros(0, 2), zeros(0, 2), weights) ≈ 0.0 atol=1e-15
    @test_throws DimensionMismatch law(counts, trials, logits, zi_logits, [1.0])
    @test_throws DimensionMismatch law(counts, trials, logits[1:2, :],
        zi_logits[1:2, :], weights)
    @test_throws DimensionMismatch law(counts, trials[1:2, :], logits,
        zi_logits, weights)
    @test_throws DimensionMismatch law(counts, trials, logits,
        zi_logits[:, 1:1], weights)

    # Ordinary Enzyme Reverse through saturated log-odds and a structural-zero
    # column, for both parameter matrices, against 512-bit central differences.
    backend = DI.AutoEnzyme(; mode=Enzyme.Reverse)
    grad_logits = hcat([0.3, 35.0, 1.2, 0.5, -1.5], fill(-Inf, 5))
    grad_zi = hcat([-0.4, -33.0, -1.3, 36.0, -0.1], zeros(5))
    point = (counts, trials, grad_logits, grad_zi, weights)
    for (active, position) in ((:logits, 3), (:zi_logits, 4))
        op = prepare_ad(law, backend, point...; active)
        gradient = ad_gradient(op, point...)
        @test all(isfinite, gradient)
        @test all(iszero, gradient[:, 2])
        step = big"1e-40"
        for i in axes(grad_logits, 1)
            plus = Any[big.(grad_logits), big.(grad_zi)]
            minus = Any[big.(grad_logits), big.(grad_zi)]
            plus[position - 2][i, 1] += step
            minus[position - 2][i, 1] -= step
            expected = (product_mixture_logit_reference(zib_logit_reference_cell,
                    counts, (trials, plus...), weights) -
                product_mixture_logit_reference(zib_logit_reference_cell,
                    counts, (trials, minus...), weights)) / (2step)
            @test gradient[i, 1] ≈ Float64(expected) atol=1e-14 rtol=1e-13
        end
    end
end
