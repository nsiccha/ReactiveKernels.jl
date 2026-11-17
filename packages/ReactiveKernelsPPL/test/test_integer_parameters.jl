module IntegerParameterTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme
using DifferentiationInterface: AutoEnzyme
using Random: Xoshiro
include("static_enzyme.jl")

exprnodes(x) = 1
exprnodes(x::Expr) = 1 + sum(exprnodes, x.args; init=0)

@rkppl integer_child(n, p) = begin
    k ~ Binomial(n, p)
    return k
end

const MODEL = @rkppl begin
    p ~ Beta(2, 3)
    k ~ Binomial(n, p)
    y .~ Normal.(k, 1.0)
end

@testset "integer parameters are runtime state outside continuous coordinates" begin
    plan = MODEL(; n = 5) | (; y = [1.2, 2.1])
    built = build_kernel(plan)
    u = [0.3]
    p = 1 / (1 + exp(-only(u)))
    q = prepare_query(built, plan, :sampler)
    @test built.layout.total == 1
    for k in (0, 2, 5, -1, 6)
        expected = logpdf(Beta(2, 3), p) + logpdf(Binomial(5, p), k) +
            sum(logpdf.(Normal(k, 1), [1.2, 2.1])) + log(p) + log1p(-p)
        @test q(u, (; k)) ≈ expected
    end
    @test constrain(built.layout, u, (; k = 2)).k === 2
    @test unconstrain(built.layout, (; p, k = 2)) ≈ u
    @test restore_draws(built.layout, hcat(u, u); discrete = [(; k=1), (; k=3)]).k == [1, 3]
    @test eltype(restore_draws(built.layout, zeros(1, 0)).k) === Int
    @test_throws ArgumentError q(u, (; k = 1.5))
    sampler = prepare_sampler(built, plan, u; backend = AutoEnzyme(; mode = Enzyme.Reverse),
        discrete = (; k = 2))
    value, gradient = sampler_value_and_gradient!(sampler, zeros(1), u)
    @test value ≈ q(u, (; k = 2))
    @test only(gradient) ≈ 4 - 10p
    sampler.discrete = (; k = 4)
    @test sampler(u) ≈ q(u, (; k = 4))
    @test only(last(sampler_value_and_gradient!(sampler, zeros(1), u))) ≈ 6 - 10p
    derivative = prepare_query_ad(prepare_ad, built, plan, :sampler,
        AutoEnzyme(; mode=Enzyme.Reverse), u; discrete=(; k=2))
    @test only(ad_gradient(derivative, u)) ≈ 4 - 10p
end

@testset "finite-support marginalization" begin
    model = marginalize(MODEL, :k)
    plan = model(; n = 5) | (; y = [1.2, 2.1])
    built = build_kernel(plan)
    u = [0.3]
    p = 1 / (1 + exp(-only(u)))
    weights = [logpdf(Binomial(5, p), k) + sum(logpdf.(Normal(k, 1), [1.2, 2.1])) for k in 0:5]
    marginal = ReactiveKernelsPPL.logsumexp(weights)
    q = prepare_query(built, plan, :sampler)
    @test q(u) ≈ marginal + logpdf(Beta(2, 3), p) + log(p) + log1p(-p)
    @test prepare_query(built, plan, :prior)(u) ≈ logpdf(Beta(2, 3), p)
    @test prepare_query(built, plan, :likelihood)(u) ≈ marginal
    @test sum(prepare_query(built, plan, :pointwise)(u).y) ≈ marginal
    @test_throws ContractValidationError density_factors(built, plan)
    latents = prepare_discrete_latents(built, plan)(u)
    @test vec(latents.parameters.probabilities) ≈ exp.(weights .- marginal)
    @test keys(latents.parameters.values) == (:k,)
    @test vec(latents.parameters.values.k) == 0:5
    latent_query = prepare_discrete_latents(built, plan)
    @test discrete_latent_draws(latent_query, u, Xoshiro(42)) ==
        discrete_latent_draws(latent_query, u, Xoshiro(42))
    sampler = prepare_sampler(built, plan, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    v, g = sampler_value_and_gradient!(sampler, zeros(1), u)
    @test v ≈ q(u)
    expected_k = sum((0:5) .* exp.(weights .- marginal))
    @test only(g) ≈ 2 + expected_k - 10p
    # Reuse the same generated graph at other support and observation lengths.
    # This exercises runtime loops rather than binding-sized source emission.
    emitted = Int[]
    for (n, y) in ((0, Float64[]), (12, [0.7, 1.1, 1.4]), (50, [2.0]))
        rebound = model(; n) | (; y)
        reference = ReactiveKernelsPPL.logsumexp([
            logpdf(Binomial(n, p), k) + sum(logpdf.(Normal(k, 1), y)) for k in 0:n])
        reused = prepare_query(built, rebound, :sampler)
        @test reused(u) ≈ reference + logpdf(Beta(2, 3), p) + log(p) + log1p(-p)
        push!(emitted, exprnodes(code_expr(reused)))
    end
    @test maximum(emitted) < 2minimum(emitted)
end

@testset "joint integer recovery and vectorized latent queries" begin
    model = @rkppl begin
        p ~ Beta(2, 3)
        k ~ Bernoulli(p)
        z ~ Bernoulli(k)
        y ~ Normal(k + z, 1)
    end
    plan = marginalize(model, :k, :z)() | (; y=0.3)
    built = build_kernel(plan)
    latent = prepare_discrete_latents(built, plan)
    U = [0.0 0.4]
    ports = ReactiveKernelsPPL._query_columns(plan)
    data = Tuple(plan.columns[k] for k in ports)
    cut = Base.invokelatest(ReactiveKernels.prepare, built.spec;
        have=(:unconstrained, ports...), want=discrete_latents_node(built))
    batch = Base.invokelatest(ReactiveKernels.vectorize, cut; batched=:unconstrained)
    value = Base.invokelatest(batch, U, data...)
    for i in 1:2
        single = latent(U[:, i])
        @test selectdim(value.parameters.probabilities, 3, i) ≈ single.parameters.probabilities
        @test discrete_latent_draws(value, Xoshiro(i); position=i) ==
            discrete_latent_draws(single, Xoshiro(i))
        for seed in 1:20
            draw = discrete_latent_draws(single, Xoshiro(seed)).parameters
            @test draw.k == draw.z
        end
    end
end

@testset "restricted integer support excludes unsafe indices" begin
    model = @rkppl begin
        k ~ restricted(Binomial(n, p), lo, hi)
        y ~ Normal(means[k], 1)
    end
    plan = marginalize(model, :k)(; n=4, p=0.4, lo=1.0, hi=2.0,
        means=[0.2, 1.4]) | (; y=0.7)
    built = build_kernel(plan)
    expected = ReactiveKernelsPPL.logsumexp([
        logpdf(Binomial(4, 0.4), k) + logpdf(Normal([0.2, 1.4][k], 1), 0.7)
        for k in 1:2])
    @test prepare_query(built, plan, :sampler)(Float64[]) ≈ expected
    @test vec(prepare_discrete_latents(built, plan)(Float64[]).parameters.values.k) == 1:2
end

@testset "other finite integer laws" begin
    for (rhs, data, values, law) in (
        (:(Bernoulli(p)), (; p = 0.3), 0:1, Bernoulli(0.3)),
        (:(Categorical(p)), (; p = [0.2, 0.3, 0.5]), 1:3, Categorical([0.2, 0.3, 0.5])),
        (:(DiscreteUniform(lo, hi)), (; lo = -2, hi = 3), -2:3, DiscreteUniform(-2, 3)))
        model = RKPPLModel(quote k ~ $rhs end, @__MODULE__)
        plan = bind_data(model(; data...))
        built = build_kernel(plan)
        q = prepare_query(built, plan, :prior)
        @test built.layout.total == 0
        for k in values
            @test q(Float64[], (; k)) ≈ logpdf(law, k)
        end
    end
end

@testset "categorical probabilities can be sampled simplexes" begin
    model = @rkppl begin
        theta ~ Dirichlet(alpha)
        k ~ Categorical(theta)
        y ~ Normal(k, 1)
    end
    alpha = [1.5, 2.0, 3.0]
    plan = model(; alpha) | (; y=1.4)
    built = build_kernel(plan)
    u, state = [0.3, -0.2], (; k=2)
    theta = constrain(built.layout, u, state).theta
    q = prepare_query(built, plan, :sampler)
    expected = logpdf(Dirichlet(alpha), theta) + log(theta[2]) +
        logpdf(Normal(2, 1), 1.4) + logjac(built.layout, u)
    @test q(u, state) ≈ expected
    sampler = prepare_sampler(built, plan, u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse), discrete=state, nonallocating=false)
    value, gradient = sampler_value_and_gradient!(sampler, zeros(2), u)
    @test value ≈ expected
    @test gradient ≈ [alpha[1] - (sum(alpha)+1)*theta[1],
        alpha[2]+1 - (alpha[2]+alpha[3]+1)*theta[2]/(1-theta[1])]
end

@testset "RAG model with an integer cheater count" begin
    model = @rkppl begin
        p ~ Beta(2, 3)
        c ~ Binomial(n, p)
        honest_claims = h - c
        honest_claims ~ Binomial(n - c, 0.5)
    end
    plan = bind_data(model(; n=10, h=8))
    built = build_kernel(plan)
    u = [0.3]
    p = 1 / (1 + exp(-only(u)))
    conditional = prepare_query(built, plan, :sampler)
    for c in 0:10
        reference = logpdf(Beta(2, 3), p) + logpdf(Binomial(10, p), c) +
            logpdf(Binomial(10-c, 0.5), 8-c) + log(p) + log1p(-p)
        @test conditional(u, (; c)) ≈ reference
    end
    marginal_plan = bind_data(marginalize(model, :c)(; n=10, h=8))
    marginal_built = build_kernel(marginal_plan)
    expected = logpdf(Beta(2, 3), p) + logpdf(Binomial(10, (1+p)/2), 8) +
        log(p) + log1p(-p)
    @test prepare_query(marginal_built, marginal_plan, :sampler)(u) ≈ expected
    # A density on a deterministic definition is a prior factor unless
    # explicitly conditioned; no observation field was conditioned here.
    @test prepare_query(marginal_built, marginal_plan, :pointwise)(u) === NamedTuple()
    @test prepare_query(marginal_built, marginal_plan, :likelihood)(u) ≈ 0.0 atol=1e-12
    sampler = prepare_sampler(marginal_built, marginal_plan, u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse), nonallocating=false)
    value, gradient = sampler_value_and_gradient!(sampler, zeros(1), u)
    @test value ≈ expected
    @test only(gradient) ≈ 2 - 7p + 8p * (1-p) / (1+p)
    latent = prepare_discrete_latents(marginal_built, marginal_plan)(u)
    @test vec(latent.parameters.probabilities) ≈ pdf.(Binomial(8, 2p/(1+p)), 0:10)
end

@testset "integer arrays and runtime queries" begin
    model = @rkppl begin
        p ~ Beta(2, 3)
        k[axes(x, 1)] .~ Binomial.(n, p)
        y .~ Normal.(k .+ x, 1.0)
    end
    plan = model(; n = 4, x = [0.1, 0.2]) | (; y = [1.0, 2.0])
    built = build_kernel(plan)
    u, discrete = [0.0], (; k = [1, 2])
    @test built.layout.total == 1
    saved = constrain(built.layout, u, discrete)
    @test saved.k == discrete.k
    discrete.k[1] = 3
    @test saved.k == [1, 2]
    saved.k[2] = 0
    @test discrete.k == [3, 2]
    discrete.k[1] = 1
    query = prepare_query(built, plan, :sampler; inputs = (:x,))
    @test query(u, discrete, [0.2, 0.3]) ≈
        logpdf(Beta(2, 3), 0.5) + sum(logpdf.(Binomial(4, 0.5), discrete.k)) +
        sum(logpdf.(Normal.([1.2, 2.3], 1), [1.0, 2.0])) - 2log(2)
    pred = prepare_predictive(built, plan)
    @test mean.(predictive_laws(pred, u, discrete).y) ≈ [1.1, 2.2]
    @test_throws ArgumentError prepare_query(built, plan, :sampler)(u, (; k = [1]))
    empty_plan = model(; n = 4, x = Float64[]) | (; y = Float64[])
    @test prepare_query(built, empty_plan, :sampler)(u, (; k = Int[])) ≈
        logpdf(Beta(2, 3), 0.5) - 2log(2)
end

@testset "integer indices and observation trials" begin
    model = @rkppl begin
        k ~ DiscreteUniform(lo, hi)
        mu = means[k]
        y ~ Normal(mu, 1)
    end
    plan = model(; lo = 1, hi = 3, means = [0.2, 1.4, -0.5]) | (; y = 0.7)
    built = build_kernel(plan)
    q = prepare_query(built, plan, :sampler)
    @test q(Float64[], (; k = 2)) ≈ -log(3) + logpdf(Normal(1.4, 1), 0.7)
    counts = @rkppl begin
        p ~ Beta(2, 3)
        k ~ Binomial(n, p)
        remaining = n - k
        h ~ Binomial(remaining, 0.5)
    end
    plan = counts(; n = 6) | (; h = 2)
    built = build_kernel(plan)
    @test prepare_query(built, plan, :likelihood)([0.0], (; k = 3)) ≈
        logpdf(Binomial(3, 0.5), 2)
end

@testset "conditional cell, factor and partition queries" begin
    model = @rkppl begin
        p ~ Beta(2, 3)
        k[axes(x, 1)] .~ Binomial.(n, p)
        y .~ Normal.(k .* p .+ x, 1.0)
    end
    x, y, group = [0.1, 0.2], [1.0, 2.0], [1, 1]
    plan = model(; n=4, x, group) | (; y)
    built = build_kernel(plan)
    u, state = [0.0], (; k=[1, 2])
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    expected = logpdf.(Normal.(state.k .* 0.5 .+ x, 1), y)
    cell = prepare_cell_query(built, plan, :y; discrete=state)
    @test [cell(u, i) for i in 1:2] ≈ expected
    partition = prepare_partition(built, plan, :y; by=:group, discrete=state)
    @test partition(u, 1) ≈ sum(expected)
    _static_enzyme() do
        gradient = prepare_compact_cell_sampler(built, plan, :y, u, [1]; backend, discrete=state)
        value, g = cell_value_and_gradient!(gradient, zeros(1), u, copy(u), [1], 2)
        @test value ≈ expected[2]
        @test only(g) ≈ (y[2] - (state.k[2] * 0.5 + x[2])) * state.k[2] * 0.25
    end
    catalogue = density_factors(built, plan; discrete=state)
    selected = select_factors(catalogue; kind=:prior)
    integers = select_factors(catalogue; kind=:prior, site=:k)
    @test length(integers) == 2
    @test isempty(factor_coordinates(integers))
    @test factor_coordinates(select_factors(catalogue; kind=:prior, site=:p)) == [1]
    @test prepare_query(built, plan, selected)(u) ≈
        logpdf(Beta(2, 3), 0.5) + sum(logpdf.(Binomial(4, 0.5), state.k))
    # Preparation snapshots integer index state and its row plans.
    state.k[2] = 4
    @test cell(u, 2) ≈ expected[2]
end

@testset "nested integer parameter names" begin
    model = @rkppl begin
        p ~ Beta(2, 3)
        z ~ integer_child(n, p)
        y ~ Normal(z, 1)
    end
    plan = model(; n=5) | (; y=1.2)
    built = build_kernel(plan)
    u, state = [0.0], (; z=(; k=2))
    @test constrain(built.layout, u, state).z.k === 2
    @test prepare_query(built, plan, :sampler)(u, state) ≈
        logpdf(Beta(2, 3), 0.5) + logpdf(Binomial(5, 0.5), 2) +
        logpdf(Normal(2, 1), 1.2) - 2log(2)
    @test eltype(restore_draws(built.layout, zeros(1, 0)).z.k) === Int
    marginal_plan = marginalize(model, Symbol("z.k"))(; n=5) | (; y=1.2)
    marginal_built = build_kernel(marginal_plan)
    latent = prepare_discrete_latents(marginal_built, marginal_plan)
    @test vec(latent(u).parameters.values.z.k) == 0:5
    @test discrete_latent_draws(latent, u, Xoshiro(12)).parameters.z.k in 0:5
end

end
