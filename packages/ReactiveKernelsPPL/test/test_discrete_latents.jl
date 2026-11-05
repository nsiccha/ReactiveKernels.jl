module DiscreteLatentTests
using Test, Random, ReactiveKernels, ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal_hmm_logdensity, normal_hmm_log_weights_logdensity, draw_hmm_states
using Distributions: Normal, logpdf

# Independent reference: forward-backward in 512-bit arithmetic, written out
# from the definition (the path posterior's per-step marginals). `emission(t,
# k)` is step t's log density under state k; `move(i, j, t)` weighs the move
# from i to j into step t.
function marginals_reference(emission, move, initial, T, K)
    setprecision(BigFloat, 512) do
        alpha = zeros(BigFloat, T, K)
        beta = ones(BigFloat, T, K)
        for k in 1:K
            alpha[1, k] = big(initial[k]) * exp(big(emission(1, k)))
        end
        for t in 2:T, j in 1:K
            alpha[t, j] = sum(alpha[t - 1, i] * big(move(i, j, t)) for i in 1:K) *
                exp(big(emission(t, j)))
        end
        for t in (T - 1):-1:1, i in 1:K
            beta[t, i] = sum(big(move(i, j, t + 1)) * exp(big(emission(t + 1, j))) *
                beta[t + 1, j] for j in 1:K)
        end
        Float64.(alpha .* beta ./ sum(alpha[T, :]))
    end
end

function bound_model(src, data; conditioned = ())
    bound = bind_data(lower_rkppl(src, data; mod = @__MODULE__, conditioned), data)
    return bound, build_kernel(bound)
end

y = [0.3, 2.9, 3.1, 0.2, -0.1, 3.3]
hmm_source = quote
    eachrow(A[1:2, 1:2]) .~ Dirichlet(ones(2))
    m[1:2] .~ Normal.([0.0, 3.0], 1.0)
    y ~ LogDensity(normal_hmm_logdensity, m, 0.5, A, [0.4, 0.6])
end

# The same sequence observed as a conditioned declaration and as a data
# definition the model observes whole (`obs = …; obs ~ LogDensity(…)`).
derived_source = quote
    eachrow(A[1:2, 1:2]) .~ Dirichlet(ones(2))
    m[1:2] .~ Normal.([0.0, 3.0], 1.0)
    obs = y .* 1.0
    obs ~ LogDensity(normal_hmm_logdensity, m, 0.5, A, [0.4, 0.6])
end

@testset "hidden Markov state posteriors: conditioned and derived observations" begin
    for (source, conditioned, name) in ((hmm_source, (:y,), :y), (derived_source, (), :obs))
        bound, built = bound_model(source, (; y); conditioned)
        latents = prepare_discrete_latents(built, bound)
        A = [0.8 0.2; 0.35 0.65]
        m = [0.1, 2.9]
        u = unconstrain(built.layout, (; A, m))
        posterior = getproperty(latents(u), name)
        @test keys(latents(u)) == (name,)
        reference = marginals_reference((t, k) -> logpdf(Normal(m[k], 0.5), y[t]),
            (i, j, t) -> A[i, j], [0.4, 0.6], length(y), 2)
        @test posterior.probabilities ≈ reference rtol=1e-10
        @test posterior.log_transition ≈ log.(A) rtol=1e-14
        # The latents node sits beside the likelihood, which is still the law.
        likelihood = prepare_query(built, bound, :likelihood)
        @test Base.invokelatest(likelihood, u) ≈
            prepare(normal_hmm_logdensity)(y, m, 0.5, A, [0.4, 0.6]) rtol=1e-13
        draws = getproperty(discrete_latent_draws(latents, u, Xoshiro(3)), name)
        @test draws isa Vector{Int} && length(draws) == length(y)
        @test draws == getproperty(discrete_latent_draws(latents, u, Xoshiro(3)), name)
        @test draws == draw_hmm_states(Xoshiro(3), posterior)
    end
end

@testset "step-dependent log-weight transitions" begin
    T = length(y)
    src = quote
        a ~ Normal(0, 1)
        m[1:2] .~ Normal.([0.0, 3.0], 1.0)
        logits = reshape(a .* x[1:(end - 1)], 1, 1, length(x) - 1)
        stay = -log1p.(exp.(-logits))
        move = -log1p.(exp.(logits))
        log_transition = vcat(hcat(stay, move), hcat(move, stay))
        y ~ LogDensity(normal_hmm_log_weights_logdensity, m, 0.5, log_transition,
            log.([0.5, 0.5]))
    end
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    bound, built = bound_model(src, (; y, x); conditioned = (:y,))
    latents = prepare_discrete_latents(built, bound)
    a, m = 1.3, [0.2, 2.8]
    u = unconstrain(built.layout, (; a, m))
    stay(t) = 1 / (1 + exp(-a * x[t - 1]))
    move(i, j, t) = i == j ? stay(t) : 1 - stay(t)
    reference = marginals_reference((t, k) -> logpdf(Normal(m[k], 0.5), y[t]), move,
        [0.5, 0.5], T, 2)
    @test latents(u).y.probabilities ≈ reference rtol=1e-10
end

@testset "drawn paths follow the exact path posterior" begin
    bound, built = bound_model(hmm_source, (; y = y[1:4]); conditioned = (:y,))
    latents = prepare_discrete_latents(built, bound)
    A = [0.7 0.3; 0.45 0.55]
    m = [0.4, 2.5]
    u = unconstrain(built.layout, (; A, m))
    weights = Dict{Vector{Int},Float64}()
    for path in Iterators.product(ntuple(_ -> 1:2, 4)...)
        w = [0.4, 0.6][path[1]] * exp(logpdf(Normal(m[path[1]], 0.5), y[1]))
        for t in 2:4
            w *= A[path[t - 1], path[t]] * exp(logpdf(Normal(m[path[t]], 0.5), y[t]))
        end
        weights[collect(path)] = w
    end
    total = sum(values(weights))
    rng = Xoshiro(11)
    posterior = latents(u)
    counts = Dict{Vector{Int},Int}()
    draws = 100_000
    for _ in 1:draws
        path = discrete_latent_draws(posterior, rng).y
        counts[path] = get(counts, path, 0) + 1
    end
    for (path, w) in weights
        p = w / total
        @test abs(get(counts, path, 0) / draws - p) <= 4 * sqrt(p * (1 - p) / draws)
    end
end

@testset "mixture memberships" begin
    ym = [-1.0, 0.2, 2.9, 3.4, 0.1]
    src = quote
        w ~ Dirichlet(ones(2))
        mu ~ Ordered(Normal(0, 5), 2)
        sigma ~ Exponential(1.0)
        y .~ MixtureModel.(vcat.(Normal.(mu[1], sigma), Normal.(mu[2], sigma)), Ref(w))
    end
    bound, built = bound_model(src, (; y = ym); conditioned = (:y,))
    latents = prepare_discrete_latents(built, bound)
    w, mu, sigma = [0.35, 0.65], [-0.5, 3.0], 0.9
    u = unconstrain(built.layout, (; w, mu, sigma))
    P = latents(u).y.probabilities
    reference = [w[k] * exp(logpdf(Normal(mu[k], sigma), x)) for x in ym, k in 1:2]
    @test P ≈ reference ./ sum(reference; dims = 2) rtol=1e-12
    rng = Xoshiro(2)
    draws = 50_000
    counts = zeros(length(ym), 2)
    for _ in 1:draws
        z = discrete_latent_draws(latents, u, rng).y
        for i in eachindex(z)
            counts[i, z[i]] += 1
        end
    end
    @test all(abs.(counts ./ draws .- P) .<= 4 .* sqrt.(P .* (1 .- P) ./ draws) .+ 1e-12)

    # Literal weights and a third component.
    src3 = quote
        mu ~ Ordered(Normal(0, 5), 3)
        y .~ MixtureModel.(vcat.(Normal.(mu[1], 1.0), Normal.(mu[2], 1.0),
            Normal.(mu[3], 1.0)), Ref([0.2, 0.3, 0.5]))
    end
    bound3, built3 = bound_model(src3, (; y = ym); conditioned = (:y,))
    mu3 = [-1.0, 0.5, 3.0]
    u3 = unconstrain(built3.layout, (; mu = mu3))
    P3 = prepare_discrete_latents(built3, bound3)(u3).y.probabilities
    reference3 = [[0.2, 0.3, 0.5][k] * exp(logpdf(Normal(mu3[k], 1.0), x))
        for x in ym, k in 1:3]
    @test P3 ≈ reference3 ./ sum(reference3; dims = 2) rtol=1e-12
end

@testset "the latents node in a caller's vectorized cut" begin
    bound, built = bound_model(hmm_source, (; y); conditioned = (:y,))
    latents = prepare_discrete_latents(built, bound)
    U = hcat(unconstrain(built.layout, (; A = [0.8 0.2; 0.35 0.65], m = [0.1, 2.9])),
        unconstrain(built.layout, (; A = [0.6 0.4; 0.5 0.5], m = [-0.2, 3.2])))
    node = discrete_latents_node(built)
    ports = ReactiveKernelsPPL._query_columns(bound)
    data = Tuple(bound.columns[k] for k in ports)
    cut = Base.invokelatest(ReactiveKernels.prepare, built.spec;
        have = (:unconstrained, ports...), want = node)
    batch = Base.invokelatest(ReactiveKernels.vectorize, cut; batched = :unconstrained)
    value = Base.invokelatest(batch, U, data...)
    for k in 1:2
        single = latents(U[:, k])
        @test selectdim(value.y.probabilities, 3, k) ≈ single.y.probabilities rtol=1e-14
        @test discrete_latent_draws(value, Xoshiro(k); position = k) ==
            discrete_latent_draws(single, Xoshiro(k))
    end
end

@testset "programs without a recoverable latent are refused" begin
    plain = quote
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1.0)
    end
    bound, built = bound_model(plain, (; y); conditioned = (:y,))
    @test_throws ContractValidationError prepare_discrete_latents(built, bound)

    # A weighted mixture has no single membership per observation yet.
    weighted = quote
        mu ~ Ordered(Normal(0, 5), 2)
        y .~ weighted.(MixtureModel.(vcat.(Normal.(mu[1], 1.0), Normal.(mu[2], 1.0)),
            Ref([0.5, 0.5])), wt)
    end
    bound, built = bound_model(weighted, (; y, wt = fill(2.0, length(y))); conditioned = (:y,))
    err = try
        Base.invokelatest(prepare_discrete_latents(built, bound),
            unconstrain(built.layout, (; mu = [0.0, 3.0])))
        nothing
    catch e
        e
    end
    @test err isa ContractValidationError
    @test occursin("y", sprint(showerror, err))
end

# The docs page "Discrete latent variables" runs these files as displayed.
function run_example(file)
    mod = Module(gensym(:DiscreteLatentExample))
    Base.include_string(mod, read(joinpath(@__DIR__, "discrete_latent_examples", file),
        String), file)
    return mod
end

@testset "docs examples against independent references" begin
    hmm = run_example("hmm.jl")
    T, m = hmm.true_T, hmm.true_m
    reference = marginals_reference((t, k) -> logpdf(Normal(m[k], 0.1), hmm.y[t]),
        (i, j, t) -> T[i, j], fill(1 / 3, 3), length(hmm.y), 3)
    @test hmm.posterior.probabilities ≈ reference rtol=1e-9 atol=1e-300
    @test hmm.most_probable == hmm.true_s
    @test length(hmm.path) == 60

    mixture = run_example("mixture.jl")
    weights = [w * exp(logpdf(Normal(mu, 0.6), x))
        for x in mixture.y, (w, mu) in zip([0.4, 0.6], [-1.0, 2.0])]
    @test mixture.memberships ≈ weights ./ sum(weights; dims = 2) rtol=1e-12
    @test length(mixture.components) == 80

    changepoint = run_example("changepoint.jl")
    y = changepoint.y
    scores = setprecision(BigFloat, 256) do
        [sum(big(y[t]) * log(big(3.0)) - 3 for t in 1:(s - 1); init = big(0.0)) +
         sum(big(y[t]) * log(big(1.0)) - 1 for t in s:length(y); init = big(0.0))
         for s in eachindex(y)]
    end
    expected = Float64.(exp.(scores .- maximum(scores)) ./ sum(exp.(scores .- maximum(scores))))
    @test changepoint.probabilities ≈ expected rtol=1e-10
end
end
