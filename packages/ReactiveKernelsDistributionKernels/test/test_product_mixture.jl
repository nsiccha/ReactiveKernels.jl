using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_product_mixture_logdensity, binomial_product_mixture_logdensity,
    bernoulli_logit_product_mixture_logdensity,
    binomial_logit_product_mixture_logdensity,
    zero_inflated_binomial_logit_product_mixture_logdensity
using Distributions: Bernoulli, Binomial, MixtureModel, logpdf, product_distribution
import DifferentiationInterface as DI
import Enzyme

@testset "one component governs each whole observed vector" begin
    weights = [0.4, 0.6]
    observed = Bool[true, false, true]
    probabilities = [0.8 0.1; 0.2 0.7; 0.5 0.3]
    prepared = prepare(bernoulli_product_mixture_logdensity)
    reference = MixtureModel([product_distribution(Bernoulli.(probabilities[:, k]))
        for k in 1:2], weights)
    actual = prepared(observed, probabilities, weights)
    @test actual ≈ logpdf(reference, Int.(observed))
    independent = sum(logpdf(MixtureModel(Bernoulli.(probabilities[i, :]),
        weights), Int(observed[i])) for i in eachindex(observed))
    @test abs(actual - independent) > 0.05

    trials = [2 3; 1 2; 3 4]
    counts = [1, 0, 2]
    binomial_prepared = prepare(binomial_product_mixture_logdensity)
    binomial_reference = MixtureModel([product_distribution(
        Binomial.(trials[:, k], probabilities[:, k])) for k in 1:2], weights)
    @test binomial_prepared(counts, trials, probabilities, weights) ≈
        logpdf(binomial_reference, counts)

    # The same prepared graph covers other event lengths and an empty event.
    @test prepared(Bool[false], reshape([0.3, 0.7], 1, 2), weights) ≈
        logpdf(MixtureModel([Bernoulli(0.3), Bernoulli(0.7)], weights), 0)
    @test prepared(Bool[], zeros(0, 2), weights) ≈ 0.0
    @test_throws DimensionMismatch prepared(observed, probabilities, [1.0])
    @test_throws DimensionMismatch prepared(observed, probabilities[1:2, :], weights)
    @test_throws DimensionMismatch binomial_prepared(counts, trials[1:2, :],
        probabilities, weights)

    backend = DI.AutoEnzyme(; mode=Enzyme.Reverse)
    op = prepare_ad(prepared, backend, observed, probabilities, weights;
        active=:probabilities)
    grad = ad_gradient(op, observed, probabilities, weights)
    delta = 1e-5
    plus, minus = copy(probabilities), copy(probabilities)
    plus[1, 1] += delta
    minus[1, 1] -= delta
    expected = (prepared(observed, plus, weights) -
        prepared(observed, minus, weights)) / (2delta)
    @test grad[1, 1] ≈ expected rtol=2e-6
end

# Independent 512-bit reference for the logit-scale laws: each cell is the
# exact Bernoulli or Binomial log mass at logit `l`, and the outer fold is a
# log-sum-exp over weighted component sums.
function product_mixture_logit_reference(cell, observed, args, weights)
    setprecision(BigFloat, 512) do
        scores = [log(big(weights[k])) +
            sum(cell(observed[i], map(a -> a[i, k], args)...)
                for i in eachindex(observed); init = big(0.0))
            for k in eachindex(weights)]
        top = maximum(scores)
        top == -Inf ? top : top + log(sum(exp(s - top) for s in scores))
    end
end
bernoulli_logit_reference_cell(y, l) = y ? -log1p(exp(-big(l))) : -log1p(exp(big(l)))
binomial_logit_reference_cell(y, n, l) = log(binomial(big(n), big(y))) +
    (y == 0 ? big(0.0) : y * -log1p(exp(-big(l)))) +
    (y == n ? big(0.0) : (n - y) * -log1p(exp(big(l))))

@testset "logit-scale whole-event laws keep saturated tails" begin
    weights = [0.6, 0.4]
    observed = Bool[true, false, true, true, false]
    bernoulli_law = prepare(bernoulli_logit_product_mixture_logdensity)
    probability_law = prepare(bernoulli_product_mixture_logdensity)
    # Moderate logits: the same mixture as the probability law.
    moderate = [0.5 -1.2; -1.0 0.3; 2.0 0.8; 1.0 -0.4; -0.3 1.1]
    @test bernoulli_law(observed, moderate, weights) ≈
        probability_law(observed, 1 ./ (1 .+ exp.(-moderate)), weights) rtol=1e-13
    # Saturated logits beside a structural-zero component (logit -Inf): the
    # absent outcome at logit 38 is -38 in log space, where
    # log1p(-logistic(38)) is -Inf.
    for saturated in (18.0, 25.0, 38.0, 60.0)
        logits = hcat([0.5, saturated, 2.0, 1.0, -0.3], fill(-Inf, 5))
        actual = bernoulli_law(observed, logits, weights)
        reference = product_mixture_logit_reference(
            bernoulli_logit_reference_cell, observed, (logits,), weights)
        @test isfinite(actual)
        @test abs(big(actual) - reference) <= 1e-14 * abs(reference)
    end
    # An event without captures is carried by the structural-zero component.
    @test bernoulli_law(Bool[false, false], [40.0 -Inf; -2.0 -Inf], weights) ≈
        Float64(product_mixture_logit_reference(bernoulli_logit_reference_cell,
            Bool[false, false], ([40.0 -Inf; -2.0 -Inf],), weights)) rtol=1e-14
    @test bernoulli_law(Bool[], zeros(0, 2), weights) ≈ 0.0
    @test_throws DimensionMismatch bernoulli_law(observed, moderate, [1.0])
    @test_throws DimensionMismatch bernoulli_law(observed, moderate[1:2, :], weights)

    counts = [1, 0, 3, 2]
    trials = [3 3; 2 2; 4 4; 2 2]
    binomial_law = prepare(binomial_logit_product_mixture_logdensity)
    @test binomial_law(counts, trials, [0.2 -1.0; 0.4 0.5; 1.0 2.0; -0.5 0.0],
        weights) ≈ logpdf(MixtureModel([product_distribution(Binomial.(
            trials[:, k], 1 ./ (1 .+ exp.(-[0.2 -1.0; 0.4 0.5; 1.0 2.0; -0.5 0.0][:, k]))))
            for k in 1:2], weights), counts) rtol=1e-13
    for saturated in (20.0, 40.0)
        logits = hcat([0.2, saturated, 1.0, -0.5], fill(-Inf, 4))
        actual = binomial_law(counts, trials, logits, weights)
        reference = product_mixture_logit_reference(
            binomial_logit_reference_cell, counts, (trials, logits), weights)
        @test abs(big(actual) - reference) <= 1e-14 * abs(reference)
    end
    @test_throws DimensionMismatch binomial_law(counts, trials[1:2, :],
        zeros(4, 2), weights)

    # Ordinary Enzyme Reverse through a saturated logit and a structural-zero
    # column, against 512-bit central differences.
    backend = DI.AutoEnzyme(; mode=Enzyme.Reverse)
    for (law, cell, observations, data, logits) in (
            (bernoulli_law, bernoulli_logit_reference_cell, observed, (),
                hcat([0.5, 39.0, 2.0, 1.0, -0.3], fill(-Inf, 5))),
            (binomial_law, binomial_logit_reference_cell, counts, (trials,),
                hcat([0.2, 35.0, 1.0, -0.5], fill(-Inf, 4))))
        op = prepare_ad(law, backend, observations, data..., logits, weights;
            active=:logits)
        gradient = ad_gradient(op, observations, data..., logits, weights)
        @test all(isfinite, gradient)
        @test all(iszero, gradient[:, 2])
        step = big"1e-40"
        for i in axes(logits, 1)
            plus, minus = big.(logits), big.(logits)
            plus[i, 1] += step
            minus[i, 1] -= step
            expected = (product_mixture_logit_reference(cell, observations,
                    (data..., plus), weights) -
                product_mixture_logit_reference(cell, observations,
                    (data..., minus), weights)) / (2step)
            @test gradient[i, 1] ≈ Float64(expected) atol=1e-14 rtol=1e-13
        end
    end
end

# Zero-inflated Binomial cells with both probabilities on the log-odds scale.
# The reference cell is the exact log mass at success log-odds `l` and
# zero-inflation log-odds `z`, in 512-bit arithmetic.
function zib_logit_reference_cell(y, n, l, z)
    l, z = big(l), big(z)
    logzi, log1mzi = -log1p(exp(-z)), -log1p(exp(z))
    count = log(binomial(big(n), big(y))) +
        (y == 0 ? big(0.0) : y * -log1p(exp(-l))) +
        (y == n ? big(0.0) : (n - y) * -log1p(exp(l)))
    y > 0 && return log1mzi + count
    a, b = logzi, log1mzi + count
    top = max(a, b)
    top == -Inf ? top : top + log(exp(a - top) + exp(b - top))
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
