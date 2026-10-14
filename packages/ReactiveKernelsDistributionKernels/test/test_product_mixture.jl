using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_product_mixture_logdensity, binomial_product_mixture_logdensity
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
