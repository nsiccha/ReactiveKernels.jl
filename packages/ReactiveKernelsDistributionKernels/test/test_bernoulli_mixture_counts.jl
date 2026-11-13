module BernoulliMixtureCountsTests
using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using DifferentiationInterface: AutoEnzyme
import Enzyme
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_logit_product_mixture_logdensity

# Independent per-observation arithmetic and analytical derivatives. In
# particular, this oracle does not summarize a singleton row into counts.
function reference(observed, logits, weights)
    setprecision(BigFloat, 512) do
        scores = log.(big.(weights))
        for k in eachindex(weights), i in eachindex(observed)
            row = size(logits, 1) == 1 ? 1 : i
            eta = big(logits[row, k])
            scores[k] -= log1p(exp(observed[i] ? -eta : eta))
        end
        top = maximum(scores)
        value = top + log(sum(exp.(scores .- top)))
        responsibilities = exp.(scores .- value)
        gradient = zeros(BigFloat, size(logits))
        for k in eachindex(weights), i in eachindex(observed)
            row = size(logits, 1) == 1 ? 1 : i
            eta = big(logits[row, k])
            slope = observed[i] ? inv(1 + exp(eta)) : -inv(1 + exp(-eta))
            gradient[row, k] += responsibilities[k] * slope
        end
        Float64(value), Float64.(gradient), Float64.(responsibilities ./ big.(weights))
    end
end

@testset "Bernoulli mixture counts preserve empty and saturated derivatives" begin
    law = prepare(bernoulli_logit_product_mixture_logdensity; type_stability=:error)
    weights = [0.3, 0.7]
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    logits_ad = prepare_ad(law, backend, Bool[true, false], [0.2 0.7], weights;
        active=:logits)
    weights_ad = prepare_ad(law, backend, Bool[true, false], [0.2 0.7], weights;
        active=:weights)
    # Reuse the same prepared kernels across changed values and row extents.
    cases = [
        (Bool[], [NaN NaN]),
        (Bool[], [Inf -Inf]),
        (Bool[], zeros(0, 2)),
        (Bool[true], [Inf 0.4]),
        (Bool[false], [-Inf -0.4]),
        (Bool[true, false, true], [Inf 0.4]),
        (Bool[true, false, true], [-Inf -0.4]),
        (Bool[false, false, false], [-700.0 700.0]),
        (Bool[true, true, true], [700.0 -700.0]),
        (Bool[true, false, true], [0.2 -0.7; -0.3 0.6; 1.2 -1.8]),
    ]
    for (observed, logits) in cases
        saved = deepcopy((observed, logits, weights))
        expected, dlogits, dweights = reference(observed, logits, weights)
        @test law(observed, logits, weights) ≈ expected rtol=2e-13 atol=2e-13
        for (ad, derivative) in ((logits_ad, dlogits), (weights_ad, dweights))
            value, gradient = ad_value_and_gradient(ad, observed, logits, weights)
            @test value ≈ expected rtol=2e-13 atol=2e-13
            @test all(isfinite, gradient)
            @test gradient ≈ derivative rtol=2e-13 atol=2e-13
        end
        # Bound data selects the zero-count branches during preparation.
        # Its empty/NaN cases must retain the same lazy derivative semantics.
        bound = prepare(bernoulli_logit_product_mixture_logdensity;
            bound=(; observed, weights), type_stability=:error)
        bound_ad = prepare_ad(bound, backend, logits; active=:logits)
        value, gradient = ad_value_and_gradient(bound_ad, logits)
        @test value ≈ expected rtol=2e-13 atol=2e-13
        @test all(isfinite, gradient)
        @test gradient ≈ dlogits rtol=2e-13 atol=2e-13
        @test isequal((observed, logits, weights), saved)
    end
end

function structure(ex)
    ex isa ReactiveKernels._PlateProgram && return structure(ex.ast)
    ex isa Expr || return (0, 0)
    foldl(.+, (structure(a) for a in ex.args);
        init=(1, Int(ex.head in (:for, :while))))
end

@testset "full-row Bernoulli mixtures retain runtime iteration" begin
    bodies = Tuple{Int,Int}[]
    for n in (5, 51, 501)
        observed = Bool[isodd(i) for i in 1:n]
        weights = [0.35, 0.65]
        logits = [0.7sin(i + k) for i in 1:n, k in 1:2]
        law = prepare(bernoulli_logit_product_mixture_logdensity;
            bound=(; observed, weights), type_stability=:error)
        expected, _, _ = reference(observed, logits, weights)
        @test law(logits) ≈ expected rtol=2e-13 atol=2e-13
        push!(bodies, structure(code_expr(law)))
    end
    # Check bounded growth over a 100-fold data increase, allowing fixed
    # bookkeeping differences. Equal raw operation inventories are not a
    # semantic requirement (CONTRIBUTING.md, Acceptance evidence).
    @test maximum(first, bodies) <= minimum(first, bodies) + 24
    @test all(body -> last(body) >= 2, bodies)
end
end
