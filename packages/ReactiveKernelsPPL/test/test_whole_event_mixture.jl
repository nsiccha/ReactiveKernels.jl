module WholeEventMixtureTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_product_mixture_logdensity, binomial_product_mixture_logdensity
using Distributions: Beta, Bernoulli, Binomial, MixtureModel, Normal, logpdf,
    product_distribution
import DifferentiationInterface as DI
import Enzyme

function independent_logdensity(unconstrained, events, offsets)
    a, raw_weight = unconstrained
    weight = 1 / (1 + exp(-raw_weight))
    likelihood = sum(eachindex(events); init=0.0) do i
        observed = events[i]
        probability = 1 / (1 + exp(-(a + offsets[i])))
        law = MixtureModel([
            product_distribution(Bernoulli.(fill(probability, length(observed)))),
            product_distribution(Bernoulli.(fill(0.35, length(observed))))
        ], [weight, 1 - weight])
        logpdf(law, Int.(observed))
    end
    posterior = logpdf(Normal(), a) + logpdf(Beta(2, 2), weight) +
        log(weight) + log1p(-weight) + likelihood
    (; posterior, likelihood)
end

@testset "whole-event graph mixture inside retained observation cells" begin
    events = [Bool[true, false], Bool[false, false, true]]
    offsets = [0.1, -0.3]
    data = (; events, offsets)
    bound = bind_data(lower_rkppl(quote
        a ~ Normal(0, 1)
        w ~ Beta(2, 2)
        @plate for i in eachindex(events)
            events[i] ~ LogDensity(bernoulli_product_mixture_logdensity,
                hcat(fill(1 / (1 + exp(-(a + offsets[i]))), length(events[i])),
                    fill(0.35, length(events[i]))),
                [w, 1 - w])
        end
    end, data; mod=@__MODULE__, conditioned=(:events,)), data)
    built = build_kernel(bound)
    @test !occursin("sampling_logdensity", string(kernel_expr(bound, built.layout)))
    likelihood = prepare_query(built, bound, :likelihood)
    sampler = prepare_sampler(built, bound, [0.2, -0.4];
        backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    for unconstrained in ([0.2, -0.4], [-0.3, 0.5])
        actual, gradient = sampler_value_and_gradient!(sampler,
            similar(unconstrained), unconstrained)
        expected = independent_logdensity(unconstrained, events, offsets)
        @test actual ≈ expected.posterior rtol=1e-11
        @test likelihood(unconstrained) ≈ expected.likelihood rtol=1e-11
        step = 1e-5
        for i in eachindex(unconstrained)
            plus, minus = copy(unconstrained), copy(unconstrained)
            plus[i] += step
            minus[i] -= step
            target = (independent_logdensity(plus, events, offsets).posterior -
                independent_logdensity(minus, events, offsets).posterior) / (2step)
            @test gradient[i] ≈ target atol=2e-7 rtol=2e-6
        end
    end
end

@testset "whole-event Binomial graph mixture with bound component data" begin
    counts = [0, 1, 2]
    trial_options = [2, 4]
    weights = [0.4, 0.6]
    nobs = length(counts)
    data = (; counts, trial_options, weights, nobs)
    bound = bind_data(lower_rkppl(quote
        rate ~ Beta(2, 2)
        counts ~ LogDensity(binomial_product_mixture_logdensity,
            repeat(reshape(trial_options, 1, length(trial_options)), nobs, 1),
            fill(rate, nobs, length(trial_options)), weights)
    end, data; mod=@__MODULE__, conditioned=(:counts,)), data)
    built = build_kernel(bound)
    @test !occursin("sampling_logdensity", string(kernel_expr(bound, built.layout)))
    likelihood = prepare_query(built, bound, :likelihood)
    sampler = prepare_sampler(built, bound, [0.2];
        backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    function oracle(raw)
        rate = 1 / (1 + exp(-raw))
        laws = [product_distribution(Binomial.(fill(n, nobs), fill(rate, nobs)))
            for n in trial_options]
        ll = logpdf(MixtureModel(laws, weights), counts)
        posterior = logpdf(Beta(2, 2), rate) + log(rate) + log1p(-rate) + ll
        (; ll, posterior)
    end
    for raw in (0.2, -0.5)
        actual, gradient = sampler_value_and_gradient!(sampler, zeros(1), [raw])
        expected = oracle(raw)
        @test likelihood([raw]) ≈ expected.ll rtol=1e-11
        @test actual ≈ expected.posterior rtol=1e-11
        step = 1e-5
        target = (oracle(raw + step).posterior - oracle(raw - step).posterior) /
            (2step)
        @test only(gradient) ≈ target atol=2e-7 rtol=2e-6
    end
end
end
