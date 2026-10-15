module HiddenMarkovTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal_hmm_logdensity, exponential_hmm_logdensity
using Distributions: Exponential, Normal, logpdf
import DifferentiationInterface as DI
import Enzyme

# Independent oracle: the sequence's log density summed over every state
# path in 512-bit arithmetic. `emission(t, k)` is step t's log density under
# state k; `move(i, j, t)` weighs the move from i to j into step t.
function path_sum(emission, move, initial, T)
    setprecision(BigFloat, 512) do
        K = length(initial)
        total = big(0.0)
        for path in Iterators.product(ntuple(_ -> 1:K, T)...)
            p = big(initial[path[1]]) * exp(big(emission(1, path[1])))
            for t in 2:T
                p *= big(move(path[t - 1], path[t], t)) * exp(big(emission(t, path[t])))
            end
            total += p
        end
        Float64(log(total))
    end
end

# The observed sequence is one whole-event `LogDensity` observation; the
# forward recursion composes into the model as a retained RK scan.
function check_model(ast, data, oracle; points)
    bound = bind_data(lower_rkppl(ast, data; mod = @__MODULE__,
        conditioned = (first(keys(data)),)), data)
    built = build_kernel(bound)
    @test !occursin("sampling_logdensity", string(kernel_expr(bound, built.layout)))
    @test any(e -> e.kind === :scan, recipe_inventory(built.spec))
    likelihood = prepare_query(built, bound, :likelihood)
    sampler = prepare_sampler(built, bound, first(points);
        backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
    for u in points
        @test Base.invokelatest(likelihood, u) ≈ oracle(constrain(built.layout, u)) rtol=1e-11
        gradient = similar(u)
        sampler_value_and_gradient!(sampler, gradient, u)
        step = 1e-6
        for i in eachindex(u)
            plus, minus = copy(u), copy(u)
            plus[i] += step
            minus[i] -= step
            expected = (Base.invokelatest(sampler, plus) -
                Base.invokelatest(sampler, minus)) / (2step)
            @test gradient[i] ≈ expected atol=1e-6 rtol=1e-6
        end
    end
end

rows(theta...) = permutedims(reduce(hcat, theta))

y = [0.3, 2.9, 3.1, 0.2, -0.1, 3.3]

@testset "per-state Gaussian emissions, simplex transition rows" begin
    check_model(quote
        theta1 ~ Dirichlet([1.0, 1.0])
        theta2 ~ Dirichlet([1.0, 1.0])
        mu ~ Ordered(Normal(0, 5), 2)
        Gamma = permutedims(hcat(theta1, theta2))
        y ~ LogDensity(normal_hmm_logdensity, mu, 1.0, Gamma, [1.0, 1.0])
    end, (; y), c -> path_sum((t, k) -> logpdf(Normal(c.mu[k], 1), y[t]),
        (i, j, t) -> rows(c.theta1, c.theta2)[i, j], [1.0, 1.0], length(y));
        points = ([0.3, -0.4, 0.1, 0.5], [-0.2, 0.6, 1.0, 0.9]))
end

@testset "per-state scales, sliced simplex rows and an initial simplex" begin
    check_model(quote
        rho ~ Dirichlet(ones(3))
        eachrow(A[1:3, 1:3]) .~ Dirichlet(ones(3))
        mu ~ Ordered(Normal(0, 5), 3)
        sigma[1:3] .~ LogNormal.(0, 1)
        y ~ LogDensity(normal_hmm_logdensity, mu, sigma, A, rho)
    end, (; y), c -> path_sum((t, k) -> logpdf(Normal(c.mu[k], c.sigma[k]), y[t]),
        (i, j, t) -> c.A[i, j], c.rho, length(y));
        points = (collect(range(-0.6, 0.7; length = 14)),))
end

@testset "input-driven transitions and emission locations" begin
    u = [0.5 1.0; 1.2 -0.3; 0.1 0.4; 2.0 1.1; -0.4 0.2; 0.9 -1.0]
    function oracle(c)
        P = exp.(u * c.W) ./ sum(exp.(u * c.W); dims = 2)
        means = u * c.B
        # The move into step t weighs only the previous state, by P[t, i].
        path_sum((t, k) -> logpdf(Normal(means[t, k], c.sigma[k]), y[t]),
            (i, j, t) -> P[t, i], c.rho, length(y))
    end
    check_model(quote
        rho ~ Dirichlet(ones(2))
        W[axes(u, 2), 1:2] .~ Normal.(0, 5)
        B[axes(u, 2), 1:2] .~ Normal.(0, 5)
        sigma[1:2] .~ LogNormal.(0, 1)
        P = exp.(u * W) ./ sum.(eachrow(exp.(u * W)))
        moves = reshape(permutedims(P[2:end, :]), 2, 1, size(u, 1) - 1)
        means = reshape(permutedims(u * B), 2, 1, size(u, 1))
        y ~ LogDensity(normal_hmm_logdensity, means, sigma, moves, rho)
    end, (; y, u), oracle;
        points = (collect(range(-0.5, 0.5; length = 11)),))
end

# The same softmax over `hcat` of declared per-state columns: an `hcat` of
# model-level values is a model-level matrix, so the named, two-step and
# one-line spellings mean what the declared matrix `W` does above.
@testset "input-driven transitions from hcat columns" begin
    u = [0.5 1.0; 1.2 -0.3; 0.1 0.4; 2.0 1.1; -0.4 0.2; 0.9 -1.0]
    function oracle(c)
        R = u * hcat(c.w1, c.w2)
        P = exp.(R) ./ sum(exp.(R); dims = 2)
        path_sum((t, k) -> logpdf(Normal(c.mu[k], 1), y[t]),
            (i, j, t) -> P[t, i], [0.5, 0.5], length(y))
    end
    softmaxes = (
        :(begin
            R = u * hcat(w1, w2)
            E = exp.(R)
            P = E ./ sum.(eachrow(E))
        end),
        :(begin
            R = u * hcat(w1, w2)
            P = exp.(R .- log.(sum.(eachrow(exp.(R)))))
        end),
        :(begin
            Wm = hcat(w1, w2)
            P = exp.(u * Wm) ./ sum.(eachrow(exp.(u * Wm)))
        end),
    )
    for softmax in softmaxes
        check_model(quote
            w1[axes(u, 2)] .~ Normal.(0, 5)
            w2[axes(u, 2)] .~ Normal.(0, 5)
            mu ~ Ordered(Normal(0, 5), 2)
            $(softmax.args...)
            moves = reshape(permutedims(P[2:end, :]), 2, 1, size(u, 1) - 1)
            y ~ LogDensity(normal_hmm_logdensity, mu, 1.0, moves, [0.5, 0.5])
        end, (; y, u), oracle;
            points = ([0.3, -0.4, 0.1, 0.5, -0.2, 0.6],
                [-0.7, 0.2, 0.9, -0.3, 0.4, 1.1]))
    end
end

streams = hcat([0.5, 1.2, 0.1, 2.0, 0.3, 0.8], [1.0, 0.3, 0.7, 0.2, 1.4, 0.5])

@testset "two exponential streams" begin
    check_model(quote
        theta1 ~ Dirichlet([4.0, 2.0])
        theta2 ~ Dirichlet([2.0, 4.0])
        phi ~ Ordered(Normal(0, 1), 2)
        lambda ~ Ordered(Normal(0, 1), 2)
        Gamma = permutedims(hcat(theta1, theta2))
        streams ~ LogDensity(exponential_hmm_logdensity,
            1 ./ exp.(hcat(phi, lambda)), Gamma, [1.0, 1.0])
    end, (; streams), c -> path_sum((t, k) ->
            logpdf(Exponential(exp(-c.phi[k])), streams[t, 1]) +
            logpdf(Exponential(exp(-c.lambda[k])), streams[t, 2]),
        (i, j, t) -> rows(c.theta1, c.theta2)[i, j], [1.0, 1.0], size(streams, 1));
        points = ([0.3, -0.4, 0.1, 0.5, -0.2, 0.3],))
end

# The same matrix as an in-model `hcat` definition of its columns binds once
# and is observed as the same whole value (snag rkppl-undotted-c-f517b2f1).
@testset "two exponential streams defined in the model" begin
    check_model(quote
        theta1 ~ Dirichlet([4.0, 2.0])
        theta2 ~ Dirichlet([2.0, 4.0])
        phi ~ Ordered(Normal(0, 1), 2)
        lambda ~ Ordered(Normal(0, 1), 2)
        Gamma = permutedims(hcat(theta1, theta2))
        streams = hcat(s1, s2)
        streams ~ LogDensity(exponential_hmm_logdensity,
            1 ./ exp.(hcat(phi, lambda)), Gamma, [1.0, 1.0])
    end, (; s1 = streams[:, 1], s2 = streams[:, 2]), c -> path_sum((t, k) ->
            logpdf(Exponential(exp(-c.phi[k])), streams[t, 1]) +
            logpdf(Exponential(exp(-c.lambda[k])), streams[t, 2]),
        (i, j, t) -> rows(c.theta1, c.theta2)[i, j], [1.0, 1.0], size(streams, 1));
        points = ([0.3, -0.4, 0.1, 0.5, -0.2, 0.3],))
end

@testset "two Gaussian streams with per-stream scales" begin
    check_model(quote
        theta1 ~ Dirichlet([4.0, 2.0])
        theta2 ~ Dirichlet([2.0, 4.0])
        phi ~ Ordered(Normal(0, 1), 2)
        lambda ~ Ordered(Normal(0, 1), 2)
        Gamma = permutedims(hcat(theta1, theta2))
        streams ~ LogDensity(normal_hmm_logdensity, hcat(phi, lambda),
            [tau kappa], Gamma, [1.0, 1.0])
    end, (; streams, tau = 0.8, kappa = 1.3), c -> path_sum((t, k) ->
            logpdf(Normal(c.phi[k], 0.8), streams[t, 1]) +
            logpdf(Normal(c.lambda[k], 1.3), streams[t, 2]),
        (i, j, t) -> rows(c.theta1, c.theta2)[i, j], [1.0, 1.0], size(streams, 1));
        points = ([0.3, -0.4, 0.1, 0.5, -0.2, 0.3],))
end
end
