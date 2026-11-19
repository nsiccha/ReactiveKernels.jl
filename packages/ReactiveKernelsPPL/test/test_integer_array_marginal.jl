module IntegerArrayMarginalTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme
using DifferentiationInterface: AutoEnzyme
using Random: Xoshiro
using LogExpFunctions: logsumexp
const PPL = ReactiveKernelsPPL

# `marginalize(model, :z)` sums an integer array out element by element when
# every density term reading it sums cells that read it only at their own
# element (snag marginalize-refu-4b5b7092: the guide's mixture spelling).
const MIXTURE = @rkppl begin
    w ~ Dirichlet(ones(K))
    mu[1:K] .~ Normal.(0, 5)
    z[axes(y, 1)] .~ Categorical(w)
    y .~ Normal.(mu[z], 1.0)
end
const LOOP_MIXTURE = @rkppl begin
    w ~ Dirichlet(ones(K))
    mu[1:K] .~ Normal.(0, 5)
    @plate for i in eachindex(y)
        z[i] ~ Categorical(w)
        y[i] ~ Normal(mu[z[i]], 1.0)
    end
end
# The route-around the reporter wrote by hand, components spelled per K.
const WRITTEN_OUT = @rkppl begin
    w ~ Dirichlet(ones(K))
    mu[1:K] .~ Normal.(0, 5)
    y .~ MixtureModel.(vcat.(Normal.(mu[1], 1.0), Normal.(mu[2], 1.0),
        Normal.(mu[3], 1.0)), Ref(w))
end
const Y = [-2.1, -1.9, 2.0, 2.2, 1.8]
const W, MU = [0.2, 0.3, 0.5], [-2.0, 0.1, 2.0]
const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

@testset "an integer array sums out element by element ($label)" for (label, model) in
        (("declared", MIXTURE), ("loop cells", LOOP_MIXTURE))
    plan = marginalize(model(; K = 3), :z) | (; y = Y)
    built = build_kernel(plan)
    @test coordinate_names(built.layout) ==
        [Symbol("w.1"), Symbol("w.2"), Symbol("mu.1"), Symbol("mu.2"), Symbol("mu.3")]
    u = unconstrain(built.layout, (; w = W, mu = MU))
    law = MixtureModel(Normal.(MU, 1.0), W)
    @test prepare_query(built, plan, :likelihood)(u) ≈ sum(logpdf.(Ref(law), Y))
    @test prepare_query(built, plan, :prior)(u) ≈
        logpdf(Dirichlet(ones(3)), W) + sum(logpdf.(Normal(0, 5), MU))
    @test prepare_query(built, plan, :pointwise)(u).y ≈ logpdf.(Ref(law), Y)
    # The same density and gradient as the components written out by hand.
    reference = WRITTEN_OUT(; K = 3) | (; y = Y)
    reference_built = build_kernel(reference)
    @test prepare_query(built, plan, :sampler)(u) ≈
        prepare_query(reference_built, reference, :sampler)(u)
    value, gradient = sampler_value_and_gradient!(
        prepare_sampler(built, plan, u; backend = BACKEND), zeros(5), u)
    expected_value, expected_gradient = sampler_value_and_gradient!(
        prepare_sampler(reference_built, reference, u; backend = BACKEND), zeros(5), u)
    @test value ≈ expected_value
    @test gradient ≈ expected_gradient
    # Each element's posterior over the components, drawn independently.
    latents = prepare_discrete_latents(built, plan)
    record = latents(u).parameters.elements.z
    expected = [W[k] * pdf(Normal(MU[k], 1), y) for y in Y, k in 1:3]
    @test record.values == 1:3
    @test record.probabilities ≈ expected ./ sum(expected; dims = 2)
    draw = discrete_latent_draws(latents, u, Xoshiro(1)).parameters.z
    @test draw isa Vector{Int} && length(draw) == length(Y) && all(in(1:3), draw)
    @test draw == discrete_latent_draws(latents, u, Xoshiro(1)).parameters.z
end

@testset "elements with their own supports" begin
    trials, y = [2, 5, 1], [1.2, 3.9, 0.1]
    model = @rkppl begin
        k[axes(trials, 1)] .~ Binomial.(trials, 0.35)
        y .~ Normal.(k, 0.7)
    end
    plan = marginalize(model(; trials), :k) | (; y)
    built = build_kernel(plan)
    expected = sum(logsumexp([logpdf(Binomial(n, 0.35), s) + logpdf(Normal(s, 0.7), yi)
        for s in 0:n]) for (n, yi) in zip(trials, y))
    @test prepare_query(built, plan, :sampler)(Float64[]) ≈ expected
    latents = prepare_discrete_latents(built, plan)
    record = latents(Float64[]).parameters.elements.k
    @test record.values == 0:5
    # Zero outside each element's own support.
    @test record.probabilities[1, 4:6] == zeros(3) && record.probabilities[3, 3:6] == zeros(4)
    @test sum(record.probabilities; dims = 2) ≈ ones(3, 1)
    draws = [discrete_latent_draws(latents, Float64[], Xoshiro(s)).parameters.k for s in 1:100]
    @test all(d -> all(0 .<= d .<= trials), draws)
    empty_plan = marginalize(model(; trials = Int[]), :k) | (; y = Float64[])
    q = prepare_predictive(build_kernel(empty_plan), empty_plan)
    @test size(predictive_laws(q, Float64[]).y) == (0,)
    @test q(Float64[], Xoshiro(4)).y == Float64[]
end

@testset "a matrix of integers sums out entry by entry" begin
    model = @rkppl begin
        z[1:2, 1:3] .~ Bernoulli(0.3)
        y .~ Normal.(z, 1.0)
    end
    Y = [0.1 0.9 1.5; 0.3 2.2 1.0]
    plan = marginalize(model(), :z) | (; y = Y)
    built = build_kernel(plan)
    terms = [logsumexp([logpdf(Bernoulli(0.3), s) + logpdf(Normal(s, 1), y) for s in 0:1])
        for y in Y]
    @test prepare_query(built, plan, :sampler)(Float64[]) ≈ sum(terms)
    @test prepare_query(built, plan, :pointwise)(Float64[]).y ≈ terms
    latents = prepare_discrete_latents(built, plan)
    probabilities = latents(Float64[]).parameters.elements.z.probabilities
    @test size(probabilities) == (2, 3, 2)
    @test probabilities[:, :, 2] ≈ [0.3 * pdf(Normal(1, 1), y) /
        (0.3 * pdf(Normal(1, 1), y) + 0.7 * pdf(Normal(0, 1), y)) for y in Y]
    draw = discrete_latent_draws(latents, Float64[], Xoshiro(1)).parameters.z
    @test draw isa Matrix{Int} && size(draw) == (2, 3)
end

@testset "a scalar summed out beside an array" begin
    model = @rkppl begin
        a ~ Bernoulli(0.4)
        z[axes(y, 1)] .~ Bernoulli(0.3)
        y .~ Normal.(z .+ a, 1.0)
    end
    y = [0.1, 1.3, 2.2]
    plan = marginalize(model(), :a, :z) | (; y)
    built = build_kernel(plan)
    given(a) = logpdf(Bernoulli(0.4), a) + sum(logsumexp([logpdf(Bernoulli(0.3), s) +
        logpdf(Normal(s + a, 1), yi) for s in 0:1]) for yi in y)
    @test prepare_query(built, plan, :sampler)(Float64[]) ≈ logsumexp(given.(0:1))
    @test sum(prepare_query(built, plan, :pointwise)(Float64[]).y) ≈ logsumexp(given.(0:1))
    latents = prepare_discrete_latents(built, plan)
    record = latents(Float64[]).parameters
    @test record.probabilities ≈ exp.(given.(0:1) .- logsumexp(given.(0:1)))
    # Each element's posterior given each state of `a`, on a trailing axis.
    elements = record.elements.z
    @test size(elements.probabilities) == (3, 2, 2)
    for a in 0:1
        expected = [pdf(Bernoulli(0.3), s) * pdf(Normal(s + a, 1), yi) for yi in y, s in 0:1]
        @test elements.probabilities[:, :, a + 1] ≈ expected ./ sum(expected; dims = 2)
    end
    draw = discrete_latent_draws(latents, Float64[], Xoshiro(3)).parameters
    @test draw.a in 0:1 && draw.z isa Vector{Int} && all(in(0:1), draw.z)
end

@testset "two responses read each element: sequential pointwise factors" begin
    model = @rkppl begin
        z[axes(a, 1)] .~ Bernoulli(0.3)
        a .~ Normal.(z, 1.0)
        b .~ Normal.(2 .* z, 1.0)
    end
    av, bv = [0.1, 0.9, 1.5], [0.3, 2.2, 1.0]
    plan = marginalize(model(), :z) | (; a = av, b = bv)
    built = build_kernel(plan)
    first_factor = [logsumexp([logpdf(Bernoulli(0.3), s) + logpdf(Normal(s, 1), x)
        for s in 0:1]) for x in av]
    both = [logsumexp([logpdf(Bernoulli(0.3), s) + logpdf(Normal(s, 1), x) +
        logpdf(Normal(2s, 1), y) for s in 0:1]) for (x, y) in zip(av, bv)]
    pointwise = prepare_query(built, plan, :pointwise)(Float64[])
    @test pointwise.a ≈ first_factor
    @test pointwise.b ≈ both .- first_factor
    @test prepare_query(built, plan, :likelihood)(Float64[]) ≈ sum(both)
end

@testset "a submodel's array, named by its author path" begin
    @eval @rkppl assign_child(y, w, mu) = begin
        z[axes(y, 1)] .~ Categorical(w)
        y .~ Normal.(mu[z], 1.0)
        return 0.0
    end
    parent = @rkppl begin
        w ~ Dirichlet(ones(K))
        mu[1:K] .~ Normal.(0, 5)
        c ~ assign_child(y, w, mu)
    end
    plan = marginalize(parent(; K = 3), Symbol("c.z")) | (; y = Y)
    built = build_kernel(plan)
    u = unconstrain(built.layout, (; w = W, mu = MU))
    @test prepare_query(built, plan, :likelihood)(u) ≈
        sum(logpdf.(Ref(MixtureModel(Normal.(MU, 1.0), W)), Y))
    latents = prepare_discrete_latents(built, plan)
    @test size(latents(u).parameters.elements.c.z.probabilities) == (5, 3)
    @test length(discrete_latent_draws(latents, u, Xoshiro(1)).parameters.c.z) == 5
end

@testset "vectorized latent cuts and Gibbs beside a summed-out array" begin
    plan = marginalize(MIXTURE(; K = 3), :z) | (; y = Y)
    built = build_kernel(plan)
    u = unconstrain(built.layout, (; w = W, mu = MU))
    U = hcat(u, u .+ 0.1)
    ports = PPL._query_columns(plan)
    data = Tuple(plan.columns[k] for k in ports)
    cut = Base.invokelatest(ReactiveKernels.prepare, built.spec;
        have = (:unconstrained, ports...), want = discrete_latents_node(built))
    batch = Base.invokelatest(ReactiveKernels.vectorize, cut; batched = :unconstrained)
    value = Base.invokelatest(batch, U, data...)
    single = prepare_discrete_latents(built, plan)(U[:, 2])
    @test PPL._at_position(value, 2).parameters.elements.z.probabilities ≈
        single.parameters.elements.z.probabilities
    @test discrete_latent_draws(value, Xoshiro(2); position = 2) ==
        discrete_latent_draws(single, Xoshiro(2))

    # An explicit scalar's collapsed conditional sums the array out.
    model = @rkppl begin
        k ~ Bernoulli(0.4)
        z[axes(y, 1)] .~ Bernoulli(0.3)
        y .~ Normal.(z .+ 2k, 1.0)
    end
    y = [0.1, 1.3, 2.2]
    gplan = marginalize(model(), :z) | (; y)
    problem = prepare_gibbs(build_kernel(gplan), gplan, Float64[]; discrete = (; k = 0))
    conditional = only(problem.integers).kernel(Float64[], (; k = 0))
    exact = [pdf(Bernoulli(0.4), k) * prod(sum(pdf(Bernoulli(0.3), s) *
        pdf(Normal(s + 2k, 1), yi) for s in 0:1) for yi in y) for k in 0:1]
    @test PPL._gibbs_weights(conditional.logweights) ≈ exact ./ sum(exact) rtol = 1e-12
end

@testset "predictive mixtures use each element's posterior" begin
    plan = marginalize(MIXTURE(; K = 3), :z) | (; y = Y)
    built = build_kernel(plan)
    u = unconstrain(built.layout, (; w = W, mu = MU))
    q = prepare_predictive(built, plan)
    laws = predictive_laws(q, u).y
    @test size(laws) == size(Y)
    for (law, y) in zip(laws, Y)
        weights = W .* pdf.(Normal.(MU, 1), y)
        expected = MixtureModel(Normal.(MU, 1), weights ./ sum(weights))
        @test pdf.(Ref(law), MU) ≈ pdf.(Ref(expected), MU)
    end
    @test q(u, Xoshiro(123)) == q(u, Xoshiro(123))
end

@testset "responses draw a shared value once per element" begin
    model = @rkppl begin
        z[axes(a, 1)] .~ Bernoulli(0.5)
        a .~ Normal.(z, 0.01)
        b .~ Normal.(z, 0.01)
        c .~ Normal.(3.0, 1.0)
    end
    data = fill(0.5, 40)
    plan = marginalize(model(), :z) | (; a = data, b = data, c = [3.0])
    built = build_kernel(plan)
    q = prepare_predictive(built, plan)
    u = Float64[]
    @test only(predictive_laws(q, u).c) == Normal(3.0, 1.0)
    draw = q(u, Xoshiro(31))
    @test (draw.a .> 0.5) == (draw.b .> 0.5)
    @test 0 < count(>(0.5), draw.a) < length(data)
    @test predictive_draws!(deepcopy(draw), q, u, Xoshiro(31)) == draw
end

@testset "named values reading an eliminated integer" begin
    # A value reading the array has no single value once it is summed out.
    model = @rkppl begin
        w ~ Dirichlet(ones(K))
        mu[1:K] .~ Normal.(0, 5)
        z[axes(y, 1)] .~ Categorical(w)
        loc = mu[z]
        y .~ Normal.(loc, 1.0)
    end
    vplan = marginalize(model(; K = 3), :z) | (; y = Y)
    vbuilt = build_kernel(vplan)
    err = try prepare_query(vbuilt, vplan, (:loc,)); nothing catch e; e end
    @test err isa ContractValidationError && occursin("eliminated integer", sprint(showerror, err))
end

@testset "elements read together are refused" begin
    model = @rkppl begin
        z[axes(y, 1)] .~ Bernoulli(0.3)
        s = sum(z)
        y .~ Normal.(s, 1.0)
    end
    plan = marginalize(model(), :z) | (; y = [0.2, 1.0])
    err = try build_kernel(plan); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("the likelihood of `y` reads several of its elements together",
        sprint(showerror, err))
end
end
