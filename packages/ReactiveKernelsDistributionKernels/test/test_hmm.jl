using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    hmm_logdensity, normal_hmm_logdensity, exponential_hmm_logdensity
using Distributions: Exponential, Normal, logpdf
import DifferentiationInterface as DI
import Enzyme

# Independent oracle: sum the joint density over every state path, in
# 512-bit arithmetic. `step_weight(i, j, t)` weighs the move into step t.
function hmm_path_reference(log_emissions, step_weight, initial)
    setprecision(BigFloat, 512) do
        T, K = size(log_emissions)
        T == 0 && return big(0.0)
        total = big(0.0)
        for path in Iterators.product(ntuple(_ -> 1:K, T)...)
            p = big(initial[path[1]]) * exp(big(log_emissions[1, path[1]]))
            for t in 2:T
                p *= big(step_weight(path[t - 1], path[t], t)) *
                    exp(big(log_emissions[t, path[t]]))
            end
            total += p
        end
        log(total)
    end
end

function rand_stochastic(K, seed)
    raw = [abs(sin(seed * 7 + 3i + j)) + 0.05 for i in 1:K, j in 1:K]
    raw ./ sum(raw; dims = 2)
end

@testset "forward algorithm against state-path enumeration" begin
    law = prepare(hmm_logdensity)
    log_emissions = [-1.2 -0.4 -2.0; -0.3 -1.9 -0.8; -2.2 -0.6 -1.1;
        -0.9 -0.9 -0.2; -1.5 -0.1 -3.0]
    transition = [0.7 0.2 0.1; 0.1 0.8 0.1; 0.3 0.3 0.4]
    initial = [0.5, 0.3, 0.2]
    reference = hmm_path_reference(log_emissions, (i, j, t) -> transition[i, j],
        initial)
    @test law(log_emissions, transition, initial) ≈ reference rtol=1e-13

    # Unnormalized initial weights and structural zeros are used as given.
    zeroed = [0.6 0.4 0.0; 0.0 0.5 0.5; 0.2 0.0 0.8]
    @test law(log_emissions, zeroed, ones(3)) ≈
        hmm_path_reference(log_emissions, (i, j, t) -> zeroed[i, j], ones(3)) rtol=1e-13

    # A third axis indexes the move into step t at t - 1; singleton axes
    # broadcast, so a states × 1 × (steps - 1) array weighs the move by the
    # previous state only.
    varying = cat((rand_stochastic(3, seed) for seed in 1:4)...; dims = 3)
    @test law(log_emissions, varying, initial) ≈
        hmm_path_reference(log_emissions, (i, j, t) -> varying[i, j, t - 1],
            initial) rtol=1e-13
    previous_only = reshape([0.2 0.5 0.3 0.9; 0.7 0.1 0.6 0.05; 0.1 0.4 0.1 0.05],
        3, 1, 4)
    @test law(log_emissions, previous_only, initial) ≈
        hmm_path_reference(log_emissions, (i, j, t) -> previous_only[i, 1, t - 1],
            initial) rtol=1e-13

    # One step is the initial-weighted mixture; an empty sequence has density 1.
    @test law(log_emissions[1:1, :], transition, initial) ≈
        log(sum(initial .* exp.(log_emissions[1, :]))) rtol=1e-14
    @test law(zeros(0, 3), transition, initial) == 0.0
    # An observation impossible under every state has log density -Inf.
    impossible = copy(log_emissions)
    impossible[3, :] .= -Inf
    @test law(impossible, transition, initial) == -Inf

    @test_throws DimensionMismatch law(log_emissions, transition, [0.5, 0.5])
    @test_throws DimensionMismatch law(log_emissions, transition[1:2, :], initial)
    @test_throws DimensionMismatch law(log_emissions, varying[:, :, 1:3], initial)
end

@testset "forward derivatives: finite ties and a shared incoming term" begin
    law = prepare(hmm_logdensity)
    backend = DI.AutoEnzyme(; mode = Enzyme.Reverse)
    # Identical states tie every pair of predecessor terms. The derivative of
    # the log density in a log emission is that state's posterior probability,
    # exactly 1/2 here.
    tied, uniform = zeros(5, 2), fill(0.5, 2, 2)
    op = prepare_ad(law, backend, tied, uniform, ones(2); active = :log_emissions)
    @test ad_gradient(op, tied, uniform, ones(2)) ≈ fill(0.5, 5, 2) rtol=1e-14

    # A singleton destination axis: every state of a step receives one shared
    # incoming term. Reverse gradients against central differences.
    log_emissions = [-1.2 -0.4 -2.0; -0.3 -1.9 -0.8; -2.2 -0.6 -1.1;
        -0.9 -0.9 -0.2; -1.5 -0.1 -3.0]
    previous_only = reshape([0.2 0.5 0.3 0.9; 0.7 0.1 0.6 0.05; 0.1 0.4 0.1 0.05],
        3, 1, 4)
    initial = [0.5, 0.3, 0.2]
    step = 1e-6
    for (active, position) in ((:log_emissions, 1), (:transition, 2), (:initial, 3))
        args = (log_emissions, previous_only, initial)
        op = prepare_ad(law, backend, args...; active)
        gradient = ad_gradient(op, args...)
        for i in eachindex(args[position])
            plus, minus = map(copy, args), map(copy, args)
            plus[position][i] += step
            minus[position][i] -= step
            @test gradient[i] ≈ (law(plus...) - law(minus...)) / (2step) rtol=1e-6 atol=1e-8
        end
    end
end


@testset "family emission laws" begin
    transition = [0.8 0.2; 0.3 0.7]
    initial = [0.4, 0.6]
    y = [0.3, 2.9, 3.1, 0.2, -0.1]
    normal_law = prepare(normal_hmm_logdensity)
    mu, sigma = [0.0, 3.0], [1.0, 0.5]
    emissions = [logpdf(Normal(mu[k], sigma[k]), y[t]) for t in eachindex(y), k in 1:2]
    reference = hmm_path_reference(emissions, (i, j, t) -> transition[i, j], initial)
    @test normal_law(y, mu, sigma, transition, initial) ≈ reference rtol=1e-13
    @test normal_law(y, mu, 1.0, transition, initial) ≈ hmm_path_reference(
        [logpdf(Normal(mu[k], 1), y[t]) for t in eachindex(y), k in 1:2],
        (i, j, t) -> transition[i, j], initial) rtol=1e-13

    # Two streams, independent given the state: states × streams locations
    # and one shared scale per stream.
    streams = hcat(y, [1.0, 0.3, 0.7, 0.2, 1.4])
    locations = [0.0 1.2; 3.0 0.1]
    scales = [0.7 1.3]
    two_stream = [sum(logpdf(Normal(locations[k, s], scales[s]), streams[t, s])
        for s in 1:2) for t in 1:5, k in 1:2]
    @test normal_law(streams, locations, scales, transition, ones(2)) ≈
        hmm_path_reference(two_stream, (i, j, t) -> transition[i, j], ones(2)) rtol=1e-13

    # Locations that change with the step use the third axis.
    by_step = reshape([0.1t + 3.0 * (k - 1) for k in 1:2, t in 1:5], 2, 1, 5)
    varying = [logpdf(Normal(by_step[k, 1, t], sigma[k]), y[t]) for t in 1:5, k in 1:2]
    @test normal_law(y, by_step, sigma, transition, initial) ≈
        hmm_path_reference(varying, (i, j, t) -> transition[i, j], initial) rtol=1e-13

    exponential_law = prepare(exponential_hmm_logdensity)
    positive = abs.(streams)
    rate_scales = [2.0 0.5; 0.7 1.4]
    exponential_emissions = [sum(logpdf(Exponential(rate_scales[k, s]), positive[t, s])
        for s in 1:2) for t in 1:5, k in 1:2]
    @test exponential_law(positive, rate_scales, transition, ones(2)) ≈
        hmm_path_reference(exponential_emissions, (i, j, t) -> transition[i, j],
            ones(2)) rtol=1e-13

    @test_throws DimensionMismatch normal_law(y, [0.0, 1.0, 2.0], sigma,
        transition, initial)
    @test_throws DimensionMismatch normal_law(streams, locations, [0.7 1.3 2.0],
        transition, initial)

    # Native Enzyme Reverse through the emission plate and the forward scan.
    backend = DI.AutoEnzyme(; mode = Enzyme.Reverse)
    op = prepare_ad(normal_law, backend, y, mu, sigma, transition, initial;
        active = :locations)
    gradient = ad_gradient(op, y, mu, sigma, transition, initial)
    step = 1e-6
    for k in 1:2
        plus, minus = copy(mu), copy(mu)
        plus[k] += step
        minus[k] -= step
        expected = (normal_law(y, plus, sigma, transition, initial) -
            normal_law(y, minus, sigma, transition, initial)) / (2step)
        @test gradient[k] ≈ expected rtol=1e-6
    end
    op = prepare_ad(normal_law, backend, y, mu, sigma, transition, initial;
        active = :transition)
    gradient = ad_gradient(op, y, mu, sigma, transition, initial)
    plus, minus = copy(transition), copy(transition)
    plus[1, 2] += step
    minus[1, 2] -= step
    expected = (normal_law(y, mu, sigma, plus, initial) -
        normal_law(y, mu, sigma, minus, initial)) / (2step)
    @test gradient[1, 2] ≈ expected rtol=1e-6
end
