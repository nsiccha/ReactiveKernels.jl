module IntegerGatherTests
# An integer parameter indexes an array as in Julia: a scalar one reads one
# entry (`mu[k]`), an integer array one entry per element (`mu[z]`). Its
# values are runtime state, so Julia checks their bounds when the program
# runs. Values, priors and native Enzyme reverse gradients are compared with
# closed forms and central differences.
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme
using DifferentiationInterface: AutoEnzyme
const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

const Y = [-2.1, -1.9, 2.0, 2.2, 1.8]
const MU, W, Z = [-2.0, 2.0], [0.4, 0.6], [1, 1, 2, 2, 2]

fdgrad(f, u; h=1e-6) = [(e = zeros(length(u)); e[j] = h; (f(u .+ e) - f(u .- e)) / 2h)
    for j in eachindex(u)]

# Likelihood against `reference`; the sampler's gradient against central
# differences of the joint density at the same integer state.
function check_model(model, data, discrete, values, reference)
    plan = model(; data...) | (; y=Y)
    built = build_kernel(plan)
    u = unconstrain(built.layout, merge(values, discrete))
    @test prepare_query(built, plan, :likelihood)(u, discrete) ≈ reference rtol=1e-12
    joint = prepare_query(built, plan, :sampler)
    sampler = prepare_sampler(built, plan, u; backend=BACKEND, discrete)
    g = similar(u)
    value, _ = sampler_value_and_gradient!(sampler, g, u)
    @test value ≈ joint(u, discrete) rtol=1e-12
    @test g ≈ fdgrad(x -> joint(x, discrete), u) rtol=1e-5 atol=1e-7
    return built, plan, u
end

mixture_likelihood(z) = sum(logpdf(Normal(MU[z[i]], 1.0), Y[i]) for i in eachindex(Y))

@testset "a scalar integer parameter indexes a parameter vector" begin
    inline = @rkppl begin
        mu[1:2] .~ Normal.(0, 5)
        k ~ Categorical(p)
        y .~ Normal.(mu[k], 1.0)
    end
    named = @rkppl begin
        mu[1:2] .~ Normal.(0, 5)
        k ~ Categorical(p)
        m = mu[k]
        y .~ Normal.(m, 1.0)
    end
    for model in (inline, named), k in (1, 2)
        check_model(model, (; p=W), (; k), (; mu=MU), sum(logpdf.(Normal(MU[k], 1.0), Y)))
    end
    simplex = @rkppl begin
        w ~ Dirichlet(ones(K))
        mu[1:K] .~ Normal.(0, 5)
        k ~ Categorical(w)
        y .~ Normal.(mu[k], 1.0)
    end
    built, plan, u = check_model(simplex, (; K=2), (; k=1), (; mu=MU, w=W),
        sum(logpdf.(Normal(MU[1], 1.0), Y)))
    @test prepare_query(built, plan, :prior)(u, (; k=1)) ≈
        logpdf(Dirichlet(ones(2)), W) + sum(logpdf.(Normal(0, 5), MU)) + log(W[1]) rtol=1e-12
end

@testset "an integer array indexes one entry per element" begin
    dotted = @rkppl begin
        w ~ Dirichlet(ones(K))
        mu[1:K] .~ Normal.(0, 5)
        z[axes(y, 1)] .~ Categorical(w)
        y .~ Normal.(mu[z], 1.0)
    end
    named = @rkppl begin
        w ~ Dirichlet(ones(K))
        mu[1:K] .~ Normal.(0, 5)
        z[axes(y, 1)] .~ Categorical(w)
        m = mu[z]
        y .~ Normal.(m, 1.0)
    end
    for model in (dotted, named), z in (Z, [2, 1, 1, 2, 1])
        built, plan, u = check_model(model, (; K=2), (; z), (; mu=MU, w=W), mixture_likelihood(z))
        d = (; z)
        prior = prepare_query(built, plan, :prior)(u, d)
        @test prior ≈ logpdf(Dirichlet(ones(2)), W) + sum(logpdf.(Normal(0, 5), MU)) +
            sum(log(W[zi]) for zi in z) rtol=1e-12
        @test prepare_query(built, plan, :sampler)(u, d) ≈ prior +
            prepare_query(built, plan, :likelihood)(u, d) +
            prepare_query(built, plan, :log_jacobian)(u, d) rtol=1e-12
        @test constrain(built.layout, u, d).z == z
    end
    # Data probabilities shared by every element.
    shared = @rkppl begin
        mu[1:2] .~ Normal.(0, 5)
        z[axes(y, 1)] .~ Categorical(p)
        y .~ Normal.(mu[z], 1.0)
    end
    check_model(shared, (; p=W), (; z=Z), (; mu=MU), mixture_likelihood(Z))
end

@testset "integer indices are checked when the program runs" begin
    model = @rkppl begin
        mu[1:2] .~ Normal.(0, 5)
        k ~ DiscreteUniform(lo, hi)
        y .~ Normal.(mu[k], 1.0)
    end
    plan = model(; lo=1, hi=3) | (; y=Y)
    built = build_kernel(plan)
    q = prepare_query(built, plan, :likelihood)
    u = zeros(2)
    @test q(u, (; k=2)) ≈ sum(logpdf.(Normal(0.0, 1.0), Y))
    @test_throws BoundsError q(u, (; k=3))
end
end
