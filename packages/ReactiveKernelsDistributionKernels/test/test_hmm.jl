using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    hmm_logdensity, normal_hmm_logdensity, exponential_hmm_logdensity,
    hmm_log_weights_logdensity, normal_hmm_log_weights_logdensity,
    exponential_hmm_log_weights_logdensity
using Distributions: Exponential, Normal, logpdf
import DifferentiationInterface as DI
import Enzyme

# Independent oracle: sum the joint density over every state path, in
# 512-bit arithmetic. `step_weight(i, j, t)` weighs the move into step t.
hmm_path_reference(log_emissions, step_weight, initial) =
    setprecision(BigFloat, 512) do
        hmm_log_path_reference(log_emissions,
            (i, j, t) -> log(big(step_weight(i, j, t))), log.(big.(initial)))
    end

# The same sum from log weights, exponentiated only in 512-bit arithmetic, so
# a weight below Float64's smallest positive number still counts.
function hmm_log_path_reference(log_emissions, log_step_weight, log_initial)
    setprecision(BigFloat, 512) do
        T, K = size(log_emissions)
        T == 0 && return big(0.0)
        total = big(0.0)
        for path in Iterators.product(ntuple(_ -> 1:K, T)...)
            p = exp(big(log_initial[path[1]]) + big(log_emissions[1, path[1]]))
            for t in 2:T
                p *= exp(big(log_step_weight(path[t - 1], path[t], t)) +
                    big(log_emissions[t, path[t]]))
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

# The log-weights laws take the transition and initial weights on the log
# scale: no probability is formed, so a weight below Float64's smallest
# positive number keeps its value. Each probability law is its log-weights law
# at the logarithms of its weights.
@testset "log-weight laws" begin
    law = prepare(hmm_log_weights_logdensity)
    log_emissions = [-1.2 -0.4 -2.0; -0.3 -1.9 -0.8; -2.2 -0.6 -1.1;
        -0.9 -0.9 -0.2; -1.5 -0.1 -3.0]
    transition = [0.7 0.2 0.1; 0.1 0.8 0.1; 0.3 0.3 0.4]
    initial = [0.5, 0.3, 0.2]
    @test law(log_emissions, log.(transition), log.(initial)) ≈
        hmm_path_reference(log_emissions, (i, j, t) -> transition[i, j], initial) rtol=1e-13
    @test law(log_emissions, log.(transition), log.(initial)) ==
        prepare(hmm_logdensity)(log_emissions, transition, initial)

    # A per-step log-softmax over the previous state (states × 1 × (steps - 1)),
    # with unnormalized initial log weights, as an input-output model states it.
    scores = [0.4 -1.3 2.2 0.1; 1.7 0.3 -0.6 -2.4; -0.8 0.9 0.5 1.1]
    log_softmax = reshape(scores .- log.(sum(exp.(scores); dims = 1)), 3, 1, 4)
    log_initial = [0.3, -1.1, 0.0]
    @test law(log_emissions, log_softmax, log_initial) ≈ hmm_log_path_reference(
        log_emissions, (i, j, t) -> log_softmax[i, 1, t - 1], log_initial) rtol=1e-13
    step_dependent = cat((log.(rand_stochastic(3, seed)) for seed in 1:4)...; dims = 3)
    @test law(log_emissions, step_dependent, log_initial) ≈ hmm_log_path_reference(
        log_emissions, (i, j, t) -> step_dependent[i, j, t - 1], log_initial) rtol=1e-13

    # Weights whose probabilities underflow Float64: every path starts with
    # exp(-800), which is 0.0 as a probability weight.
    tiny_initial = [-800.0, -790.0, -805.0]
    tiny_transition = [0.0 -760.0 -1.0; -2.0 -0.5 -745.0; -0.1 -1.0 -3.0]
    tiny = law(log_emissions, tiny_transition, tiny_initial)
    @test isfinite(tiny)
    @test tiny ≈ hmm_log_path_reference(log_emissions,
        (i, j, t) -> tiny_transition[i, j], tiny_initial) rtol=1e-13
    @test prepare(hmm_logdensity)(log_emissions, exp.(tiny_transition),
        exp.(tiny_initial)) == -Inf

    # -Inf is a structural zero; one step is the initial-weighted mixture and
    # an empty sequence has density 1.
    zeroed = log.([0.6 0.4 0.0; 0.0 0.5 0.5; 0.2 0.0 0.8])
    @test law(log_emissions, zeroed, zeros(3)) ≈ hmm_log_path_reference(
        log_emissions, (i, j, t) -> zeroed[i, j], zeros(3)) rtol=1e-13
    @test law(log_emissions[1:1, :], log.(transition), log_initial) ≈
        log(sum(exp.(log_initial .+ log_emissions[1, :]))) rtol=1e-14
    @test law(zeros(0, 3), log.(transition), log_initial) == 0.0

    # refused: weights whose state count or step count disagrees with the
    # emissions describe a different chain, as for the probability laws above.
    @test_throws DimensionMismatch law(log_emissions, log.(transition), log.([0.5, 0.5]))
    @test_throws DimensionMismatch law(log_emissions, log.(transition[1:2, :]), log_initial)
    @test_throws DimensionMismatch law(log_emissions, log_softmax[:, :, 1:3], log_initial)

    # Family laws: emissions from the RK Normal and Exponential families.
    y = [0.3, 2.9, 3.1, 0.2, -0.1]
    streams = hcat(y, [1.0, 0.3, 0.7, 0.2, 1.4])
    locations, scales = [0.0 1.2; 3.0 0.1], [0.7 1.3]
    log_moves = reshape([0.2 -1.0 -0.3 -2.1; -1.6 -0.4 -1.2 -0.1], 2, 1, 4)
    log_start = [-0.2, -1.7]
    normal_log_law = prepare(normal_hmm_log_weights_logdensity)
    two_stream = [sum(logpdf(Normal(locations[k, s], scales[s]), streams[t, s])
        for s in 1:2) for t in 1:5, k in 1:2]
    @test normal_log_law(streams, locations, scales, log_moves, log_start) ≈
        hmm_log_path_reference(two_stream, (i, j, t) -> log_moves[i, 1, t - 1],
            log_start) rtol=1e-13
    @test normal_log_law(streams, locations, scales, log_moves, log_start) ==
        prepare(normal_hmm_logdensity)(streams, locations, scales, exp.(log_moves),
            exp.(log_start))
    exponential_log_law = prepare(exponential_hmm_log_weights_logdensity)
    positive, rate_scales = abs.(streams), [2.0 0.5; 0.7 1.4]
    exponential_emissions = [sum(logpdf(Exponential(rate_scales[k, s]), positive[t, s])
        for s in 1:2) for t in 1:5, k in 1:2]
    @test exponential_log_law(positive, rate_scales, log_moves, log_start) ≈
        hmm_log_path_reference(exponential_emissions,
            (i, j, t) -> log_moves[i, 1, t - 1], log_start) rtol=1e-13

    # Native Enzyme Reverse in each log weight, against central differences.
    backend = DI.AutoEnzyme(; mode = Enzyme.Reverse)
    step = 1e-6
    for (active, position) in ((:log_transition, 2), (:log_initial, 3))
        args = (log_emissions, log_softmax, log_initial)
        op = prepare_ad(law, backend, args...; active)
        gradient = ad_gradient(op, args...)
        for i in eachindex(args[position])
            plus, minus = map(copy, args), map(copy, args)
            plus[position][i] += step
            minus[position][i] -= step
            @test gradient[i] ≈ (law(plus...) - law(minus...)) / (2step) rtol=1e-6 atol=1e-8
        end
    end
    args = (streams, locations, scales, log_moves, log_start)
    op = prepare_ad(normal_log_law, backend, args...; active = :log_transition)
    gradient = ad_gradient(op, args...)
    for i in eachindex(log_moves)
        plus, minus = copy(log_moves), copy(log_moves)
        plus[i] += step
        minus[i] -= step
        expected = (normal_log_law(streams, locations, scales, plus, log_start) -
            normal_log_law(streams, locations, scales, minus, log_start)) / (2step)
        @test gradient[i] ≈ expected rtol=1e-6 atol=1e-8
    end
end
