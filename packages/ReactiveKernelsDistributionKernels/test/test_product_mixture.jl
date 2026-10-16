using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_product_mixture_logdensity, binomial_product_mixture_logdensity,
    bernoulli_logit_product_mixture_logdensity,
    binomial_logit_product_mixture_logdensity
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
# log-sum-exp over weighted component sums. The coefficient is spelled
# `Base.binomial`: `runtests.jl` imports DK's `binomial` kernel object.
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
binomial_logit_reference_cell(y, n, l) = log(Base.binomial(big(n), big(y))) +
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

law_allocated(k::K, args::Vararg{Any,N}) where {K,N} = @allocated k(args...)

@testset "whole-event laws keep no per-component vector" begin
    prepared = prepare(bernoulli_product_mixture_logdensity)
    observed = Bool[true, false, true, true]
    probabilities = reshape(0.05 .+ 0.9 .* abs.(sin.(1:24)), 4, 6)
    weights = fill(1 / 6, 6)
    reference = MixtureModel([product_distribution(Bernoulli.(probabilities[:, k]))
        for k in 1:6], weights)
    @test prepared(observed, probabilities, weights) ≈
        logpdf(reference, Int.(observed)) rtol=1e-13
    # The component scores stream through `logsumexp` instead of filling a
    # vector per event. The weights' logs are one vector per call, taken
    # before the weighting loop: nothing allocated grows with the event.
    @test !occursin("similar", string(code_expr(prepared)))
    longer = repeat(observed, 10)
    longer_probabilities = repeat(probabilities, 10, 1)
    @test law_allocated(prepared, observed, probabilities, weights) ==
          law_allocated(prepared, longer, longer_probabilities, weights)
    @test prepared(observed, zeros(4, 0), Float64[]) == -Inf

    trials = fill(3, 4, 6)
    counts = [0, 2, 3, 1]
    binomial_prepared = prepare(binomial_product_mixture_logdensity)
    binomial_reference = MixtureModel([product_distribution(
        Binomial.(trials[:, k], probabilities[:, k])) for k in 1:6], weights)
    @test binomial_prepared(counts, trials, probabilities, weights) ≈
        logpdf(binomial_reference, counts) rtol=1e-13
    # Each law allocates at most that one weights-log vector.
    @test law_allocated(binomial_prepared, counts, trials, probabilities, weights) <=
          law_allocated(prepared, observed, probabilities, weights)
    @test_throws DimensionMismatch binomial_prepared(counts, trials, probabilities,
        weights[1:5])

    logits = log.(probabilities ./ (1 .- probabilities))
    logit_prepared = prepare(bernoulli_logit_product_mixture_logdensity)
    @test logit_prepared(observed, logits, weights) ≈
        logpdf(reference, Int.(observed)) rtol=1e-13
    @test law_allocated(logit_prepared, observed, logits, weights) <=
          law_allocated(prepared, observed, probabilities, weights)
    binomial_logit_prepared = prepare(binomial_logit_product_mixture_logdensity)
    @test binomial_logit_prepared(counts, trials, logits, weights) ≈
        logpdf(binomial_reference, counts) rtol=1e-13
    @test law_allocated(binomial_logit_prepared, counts, trials, logits, weights) <=
          law_allocated(prepared, observed, probabilities, weights)
    @test_throws DimensionMismatch logit_prepared(observed, logits, weights[1:5])
end

# The `for` depth of each assignment whose target name contains `name`.
function loop_depths(ex, name, depth = 0, found = Int[])
    ex isa Expr || return found
    ex.head === :(=) && ex.args[1] isa Symbol && occursin(name, String(ex.args[1])) &&
        push!(found, depth)
    foreach(a -> loop_depths(a, name, ex.head === :for ? depth + 1 : depth, found),
            ex.args)
    found
end

@kernel shared_weight_events(histories, probabilities::Matrix{Float64},
        omega::Float64) = begin
    weights = [omega, 1 - omega]
    cells = plate(histories) do observed
        bernoulli_product_mixture_logdensity(observed, probabilities, weights)
    end
    return sum(cells)
end
@kernel per_event_probabilities(histories, present, absent::Vector{Float64},
        omega::Float64) = begin
    weights = [omega, 1 - omega]
    cells = plate(histories, present) do observed, p
        bernoulli_product_mixture_logdensity(observed, hcat(p, absent), weights)
    end
    return sum(cells)
end

@testset "events that share the weights take their logs once per call" begin
    histories = [Bool[isodd(i + j) for j in 1:3] for i in 1:7]
    present = [0.2 .+ 0.1 .* [i, i + 1, i + 2] ./ 10 for i in 1:7]
    absent = zeros(3)
    probabilities = hcat(present[1], absent)
    backend = DI.AutoEnzyme(; mode = Enzyme.Reverse)
    reference(omega, probabilities_of) = sum(eachindex(histories)) do i
        P = probabilities_of(i)
        logpdf(MixtureModel([product_distribution(Bernoulli.(P[:, k])) for k in 1:2],
                            [omega, 1 - omega]), Int.(histories[i]))
    end
    for (spec, args, probabilities_of) in (
            (shared_weight_events, omega -> (histories, probabilities, omega),
             i -> probabilities),
            (per_event_probabilities, omega -> (histories, present, absent, omega),
             i -> hcat(present[i], absent)))
        k = prepare(spec)
        # The logs sit outside the event loop, whether or not the
        # probabilities vary by event.
        depths = loop_depths(code_expr(k), "log_weights")
        @test !isempty(depths) && all(==(0), depths)
        for omega in (0.3, 0.8)
            @test k(args(omega)...) ≈ reference(omega, probabilities_of) rtol = 1e-12
            op = prepare_ad(k, backend, args(omega)...; active = :omega)
            value, gradient = ad_value_and_gradient(op, args(omega)...)
            h = 1e-6
            @test value ≈ reference(omega, probabilities_of) rtol = 1e-12
            @test gradient ≈ (reference(omega + h, probabilities_of) -
                reference(omega - h, probabilities_of)) / 2h rtol = 1e-6
        end
    end
end

