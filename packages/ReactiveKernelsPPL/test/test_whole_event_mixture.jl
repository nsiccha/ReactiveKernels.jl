module WholeEventMixtureTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_product_mixture_logdensity, binomial_product_mixture_logdensity,
    bernoulli_logit_product_mixture_logdensity
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

# Membership mixture whose present component is defined on the logit scale,
# with a lag effect that can saturate it. The absent component is a structural
# zero: logit -Inf at every position.
function logit_membership_reference(unconstrained, events, previous)
    setprecision(BigFloat, 512) do
        a, g, raw_weight = big.(unconstrained)
        weight = 1 / (1 + exp(-raw_weight))
        cell(y, l) = y ? -log1p(exp(-l)) : -log1p(exp(l))
        likelihood = sum(eachindex(events); init=big(0.0)) do i
            present = log(weight) + sum(cell(events[i][t], a + g * previous[i][t])
                for t in eachindex(events[i]); init=big(0.0))
            any(events[i]) ? present :
                log(exp(present) + exp(log1p(-weight)))
        end
        prior = -log(2 * big(pi)) - a^2 / 2 - log(big(10)) - g^2 / 200 +
            log(big(6)) + log(weight) + log1p(-weight)
        (; likelihood, posterior = prior + log(weight) + log1p(-weight) + likelihood)
    end
end

@testset "logit-scale whole-event mixture with a structural-zero component" begin
    events = [Bool[true, false, true, false], Bool[false, false, false, false],
        Bool[false, true, true, false], Bool[false, false, false]]
    previous = [vcat(0.0, Float64.(e[1:end-1])) for e in events]
    absent = [fill(-Inf, length(e)) for e in events]
    data = (; events, previous, absent)
    bound = bind_data(lower_rkppl(quote
        a ~ Normal(0, 1)
        g ~ Normal(0, 10)
        w ~ Beta(2, 2)
        @plate for i in eachindex(events)
            eta = a .+ g .* previous[i]
            events[i] ~ LogDensity(bernoulli_logit_product_mixture_logdensity,
                hcat(eta, absent[i]), [w, 1 - w])
        end
    end, data; mod=@__MODULE__, conditioned=(:events,)), data)
    built = build_kernel(bound)
    @test !occursin("sampling_logdensity", string(kernel_expr(bound, built.layout)))
    likelihood = prepare_query(built, bound, :likelihood)
    sampler = prepare_sampler(built, bound, [0.3, 1.5, 0.2];
        backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    # g = 40 puts the logit after each capture at 40.3; the probability law
    # rounds logistic(40.3) to 1 there and its absent outcome to -Inf.
    for unconstrained in ([0.3, 1.5, 0.2], [0.3, 40.0, 0.2], [-0.4, -45.0, -0.6])
        actual, gradient = sampler_value_and_gradient!(sampler,
            similar(unconstrained), unconstrained)
        expected = logit_membership_reference(unconstrained, events, previous)
        @test actual ≈ Float64(expected.posterior) rtol=1e-13
        @test likelihood(unconstrained) ≈ Float64(expected.likelihood) rtol=1e-13
        step = big"1e-40"
        for i in eachindex(unconstrained)
            plus, minus = big.(unconstrained), big.(unconstrained)
            plus[i] += step
            minus[i] -= step
            target = (logit_membership_reference(plus, events, previous).posterior -
                logit_membership_reference(minus, events, previous).posterior) /
                (2step)
            @test gradient[i] ≈ Float64(target) atol=1e-12 rtol=1e-12
        end
    end
end
end
