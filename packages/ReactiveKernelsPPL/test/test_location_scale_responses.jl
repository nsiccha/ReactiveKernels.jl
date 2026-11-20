module LocationScaleResponses
using Test, Random, ReactiveKernels, ReactiveKernelsPPL
using DifferentiationInterface: AutoEnzyme
import Distributions as D
import Enzyme
import Serialization
import ReactiveKernelsDistributionKernels.DistributionKernelSources as Laws
using ReactiveKernelsDistributionKernels.DistributionKernelSources: student_t

# A new spelling bound to an RK law in the model module. Neither this name
# nor Gumbel has a row in the PPL's built-in prior constructor table.
const ExtremeValue = Laws.gumbel
@kernel scale_first(scale::Float64, logdf::Float64, location::Float64) = begin
    logpdf(y::Float64)::Float64 = student_t(exp(logdf), location, scale).logpdf(y)
end

const backend = AutoEnzyme(; mode = Enzyme.Reverse)

function check_response(head, distribution, x, y; standardized = false)
    rhs = standardized ? :($head.(s)) : :($head.(exp.(a .+ x), s))
    body = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        y .~ $rhs
    end
    model = ReactiveKernelsPPL.RKPPLModel(body, @__MODULE__)
    fit = (standardized ? model() : model(; x)) | (; y)
    built = build_kernel(fit)
    u = [0.2, -0.3]
    posterior(v) = begin
        a, s = v[1], exp(v[2])
        prior = D.logpdf(D.Normal(), a) + D.logpdf(D.Exponential(), s) + v[2]
        prior + sum((D.logpdf(standardized ? distribution(s) :
            distribution(exp(a + xi), s), yi) for (xi, yi) in zip(x, y)); init = 0.0)
    end
    sampler = prepare_sampler(built, fit, u; backend)
    grad = similar(u)
    value, _ = sampler_value_and_gradient!(sampler, grad, u)
    @test value ≈ posterior(u) rtol = 2e-12
    h = 1e-5
    for i in eachindex(u)
        hi, lo = copy(u), copy(u)
        hi[i] += h
        lo[i] -= h
        @test grad[i] ≈ (posterior(hi) - posterior(lo)) / (2h) rtol = 2e-6 atol = 2e-6
    end
    isempty(y) && return
    query = prepare_predictive(built, fit)
    laws = predictive_laws(query, u).y
    rng = Xoshiro(158)
    refs = [standardized ? distribution(exp(u[2])) :
        distribution(exp(u[1] + xi), exp(u[2])) for xi in x]
    @test D.logpdf.(laws, y) ≈ D.logpdf.(refs, y)
    @test query(u, Xoshiro(158)).y ≈ rand.(Ref(rng), refs)
end

@testset "location-scale response densities and draws" begin
    x = [-1.0, 0.0, 0.25, 2.0]
    @test Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLDistributionsExt) !== nothing
    for (head, dist, y) in ((:Cauchy, D.Cauchy, [-1e100, -0.5, 1.0, 1e100]),
            (:Laplace, D.Laplace, [-90.0, -0.5, 1.0, 90.0]),
            (:Logistic, D.Logistic, [-900.0, -0.5, 1.0, 900.0]),
            (:Gumbel, D.Gumbel, [-3.0, -0.5, 1.0, 900.0]),
            (:(Laws.gumbel), D.Gumbel, [-3.0, -0.5, 1.0, 900.0]),
            (:ExtremeValue, D.Gumbel, [-3.0, -0.5, 1.0, 900.0]),
            (:TDist, D.TDist, [-1e40, -0.5, 1.0, 1e40]))
        @testset "$head" begin
            check_response(head, dist, x, y; standardized = head === :TDist)
        end
    end
    check_response(:Laplace, D.Laplace, Float64[], Float64[])

    # With no model parameters the same query draws the unconditional law.
    for (head, law) in ((:Cauchy, D.Cauchy()), (:Laplace, D.Laplace()),
            (:Logistic, D.Logistic()), (:TDist, D.TDist(4.0)))
        args = head === :TDist ? (4.0,) : ()
        model = ReactiveKernelsPPL.RKPPLModel(:(begin y .~ $head.($(args...)) end), @__MODULE__)
        fit = model() | (; y = zeros(8))
        query = prepare_predictive(build_kernel(fit), fit)
        rng = Xoshiro(568)
        draws = query(Float64[], Xoshiro(568)).y
        @test draws ≈ [rand(rng, law) for i in 1:8]

        prior = ReactiveKernelsPPL.RKPPLModel(:(begin z ~ $head($(args...)) end), @__MODULE__)
        prior_fit = prior()
        built = build_kernel(prior_fit)
        density = prepare_query(built, prior_fit, :prior)
        @test all(z -> Base.invokelatest(density, [z]) ≈ D.logpdf(law, z), draws)
    end
end

@testset "standardized Student-t with row-varying degrees of freedom" begin
    model = @rkppl begin
        a ~ Normal()
        y .~ TDist.(exp.(a .* x))
    end
    x, y = [-1.2, 0.4, 1.5], [-3.0, 0.1, 8.0]
    fit = model(; x) | (; y)
    a = 0.35
    q = prepare_sampler(build_kernel(fit), fit, [a]; backend)
    value, gradient = sampler_value_and_gradient!(q, zeros(1), [a])
    reference(t) = D.logpdf(D.Normal(), t) + sum(D.logpdf.(D.TDist.(exp.(t .* x)), y))
    @test value ≈ reference(a)
    @test only(gradient) ≈ (reference(a + 1e-5) - reference(a - 1e-5)) / 2e-5
end

@testset "location-scale evidence and weighted observations" begin
    y, w = [-1.0, 0.2, 1.0], [1.0, 2.0, 0.5]
    for (head, dist) in ((:Cauchy, D.Cauchy), (:Laplace, D.Laplace),
            (:Logistic, D.Logistic), (:TDist, D.TDist))
        law = head === :TDist ? :($head.(exp(a))) : :($head.(a, 1.25))
        distribution(t) = head === :TDist ? dist(exp(t)) : dist(t, 1.25)
        for (rhs, loglik) in (
                (:(truncated.($law, -1, 1)), d -> sum(D.logpdf.(D.truncated(d, -1, 1), y))),
                (:(censored.($law, -1, 1)), d -> sum(D.logpdf.(D.censored(d, -1, 1), y))),
                (:(interval_censored.($law, 1.7)), d -> sum(log.(D.cdf(d, 1.7) .- D.cdf.(d, y)))),
                (:(weighted.($law, w)), d -> sum(w .* D.logpdf.(d, y))))
            @testset "$rhs" begin
                model = ReactiveKernelsPPL.RKPPLModel(quote
                    a ~ Normal()
                    y .~ $rhs
                end, @__MODULE__)
                fit = (rhs.args[1] === :weighted ? model(; w) : model()) | (; y)
                u = [0.3]
                q = prepare_sampler(build_kernel(fit), fit, u; backend)
                value, gradient = sampler_value_and_gradient!(q, zeros(1), u)
                reference(t) = D.logpdf(D.Normal(), t) + loglik(distribution(t))
                @test value ≈ reference(u[1])
                @test only(gradient) ≈ (reference(u[1] + 1e-5) - reference(u[1] - 1e-5)) / 2e-5 atol=1e-6
            end
        end
    end
end

@testset "hand-built location-scale response with a scale predictor" begin
    # As in the extension serializer, construct the structural plan directly.
    # The family obtained from lowering identifies the RK kernel across I/O.
    family = only(lower_rkppl(:(begin y .~ Laplace.(0, 1) end), (:y,);
        conditioned = (:y,)).responses).family
    x, y = [-0.4, 0.25, 1.2], [-2.0, 1.0, 3.0]
    r = LikelihoodSpec(family, ReactiveKernelsPPL.IdentityLink, :y, :mu,
        ScalePredictorRef(:sigma, ReactiveKernelsPPL.LogLink), nothing,
        ResponseEvidence(:none, nothing, nothing), :obs)
    mu = PredictorSpec(:mu, ReactiveKernelsPPL.IdentityLink, [
        TermSpec(InterceptTerm, ColumnRef[], (;), :Intercept, :mean_intercept)], :mu)
    scale = PredictorSpec(:sigma, ReactiveKernelsPPL.LogLink, [
        TermSpec(ContinuousTerm, [:x], (;), :x, :scale_x)], :sigma)
    plan = StructuralPlan([r], [mu, scale],
        [PopulationPrior(:mu, :Intercept, 0.0, 1.0), PopulationPrior(:sigma, :x, 0.0, 1.0)],
        SampledParameter[], AssignmentSpec[], Dict{Symbol,AbstractVector}(:x => x, :y => y), length(y))
    io = IOBuffer()
    Serialization.serialize(io, plan)
    seekstart(io)
    restored = Serialization.deserialize(io)
    built = build_kernel(restored)
    u = [0.2, -0.3]
    q = prepare_sampler(built, restored, u; backend)
    value, gradient = sampler_value_and_gradient!(q, zeros(2), u)
    reference(t) = sum(D.logpdf.(D.Normal(), t)) +
        sum(D.logpdf.(D.Laplace.(t[1], exp.(t[2] .* x)), y))
    @test value ≈ reference(u)
    for i in eachindex(u)
        hi, lo = copy(u), copy(u)
        hi[i] += 1e-5
        lo[i] -= 1e-5
        @test gradient[i] ≈ (reference(hi) - reference(lo)) / 2e-5
    end
end

@testset "a caller's density with scale before location" begin
    model = @rkppl begin
        a ~ Normal()
        y .~ scale_first.(1.5, a .* x, a .+ x)
    end
    x, y = [0.1, -0.8], [-1.0, 2.0]
    fit = model(; x) | (; y)
    built = build_kernel(fit)
    u = [0.2]
    reference(a) = D.logpdf(D.Normal(), a) + sum(
        D.logpdf((a + xi) + 1.5 * D.TDist(exp(a * xi)), yi) for (xi, yi) in zip(x, y))
    obs = Base.invokelatest(prepare_query(built, fit, :observations), u).y
    @test obs.logdf ≈ only(u) .* x
    q = prepare_sampler(built, fit, u; backend)
    value, grad = sampler_value_and_gradient!(q, similar(u), u)
    @test value ≈ reference(only(u))
    @test only(grad) ≈ (reference(only(u) + 1e-5) - reference(only(u) - 1e-5)) / 2e-5
end

@testset "scale guard leaves the invalid density inactive" begin
    model = @rkppl begin
        s ~ Normal()
        y .~ Laplace.(0.0, s)
    end
    fit = model() | (; y = [0.1, 0.4])
    q = prepare_sampler(build_kernel(fit), fit, [1.0]; backend)
    v, g = sampler_value_and_gradient!(q, zeros(1), [-0.5])
    @test v == -Inf
    @test only(g) ≈ 0.5
end

@testset "repeated location-scale observations retain a grouped sum" begin
    model = @rkppl begin
        a ~ Normal()
        s ~ Exponential()
        y .~ Logistic.(a .+ x, s)
    end
    for n in (100, 400)
        x = repeat([0.1, 0.5], n)
        y = repeat([-0.2, 0.4], n)
        fit = model(; x) | (; y)
        u = [0.2, -0.3]
        q = prepare_sampler(build_kernel(fit), fit, u; backend)
        @test occursin("_multiplicity", string(ReactiveKernels.code_expr(q.kernel)))
        value = D.logpdf(D.Normal(), u[1]) + D.logpdf(D.Exponential(), exp(u[2])) + u[2] +
            sum(D.logpdf.(D.Logistic.(u[1] .+ x, exp(u[2])), y))
        @test q(u) ≈ value
    end
end
end
