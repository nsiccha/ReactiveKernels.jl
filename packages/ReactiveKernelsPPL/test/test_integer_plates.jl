module IntegerPlateTests
# Integer parameters written as `@plate` loop cells (`z[i] ~ Categorical(w)`
# with `mu[z[i]]` in the same cell) mean the same model as the declared-array
# spelling (`z[axes(y, 1)] .~ Categorical(w)`, `mu[z]`). A data array is
# indexed by an integer as in Julia (`centers[z]`). Values, priors, joint
# densities and native Enzyme reverse gradients are compared between the
# spellings, with closed forms and with central differences; Gibbs draws of
# the two spellings are compared draw for draw.
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Random
using WarmupHMC, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

const Y = [-2.1, -1.9, 2.0, 2.2, 1.8]
const MU, W, Z = [-2.0, 2.0], [0.4, 0.6], [1, 1, 2, 2, 2]

fdgrad(f, u; h=1e-6) = [(e = zeros(length(u)); e[j] = h; (f(u .+ e) - f(u .- e)) / 2h)
    for j in eachindex(u)]

const LOOP = @rkppl begin
    w ~ Dirichlet(ones(2))
    mu[1:2] .~ Normal.(0, 5)
    @plate for i in eachindex(y)
        z[i] ~ Categorical(w)
        y[i] ~ Normal(mu[z[i]], 1.0)
    end
end
const ARRAY = @rkppl begin
    w ~ Dirichlet(ones(2))
    mu[1:2] .~ Normal.(0, 5)
    z[axes(y, 1)] .~ Categorical(w)
    y .~ Normal.(mu[z], 1.0)
end

# Queries, gradient and constrained values of one model at `values`.
function evaluate(model, data, values, discrete, y=Y)
    plan = model(; data...) | (; y)
    built = build_kernel(plan)
    u = unconstrain(built.layout, merge(values, discrete))
    q(preset) = prepare_query(built, plan, preset)(u, discrete)
    sampler = prepare_sampler(built, plan, u; backend=BACKEND, discrete)
    g = similar(u)
    value, _ = sampler_value_and_gradient!(sampler, g, u)
    joint = prepare_query(built, plan, :sampler)
    return (; u, prior=q(:prior), likelihood=q(:likelihood), joint=q(:sampler), value, g,
        fd=fdgrad(x -> joint(x, discrete), u), constrained=constrain(built.layout, u, discrete))
end

@testset "an integer loop cell means the declared integer array" begin
    for z in (Z, [2, 1, 1, 2, 1])
        d = (; z)
        loop = evaluate(LOOP, (;), (; w=W, mu=MU), d)
        array = evaluate(ARRAY, (;), (; w=W, mu=MU), d)
        likelihood = sum(logpdf(Normal(MU[z[i]], 1.0), Y[i]) for i in eachindex(Y))
        prior = logpdf(Dirichlet(ones(2)), W) + sum(logpdf.(Normal(0, 5), MU)) +
            sum(log(W[zi]) for zi in z)
        @test loop.likelihood ≈ likelihood rtol=1e-12
        @test loop.prior ≈ prior rtol=1e-12
        for field in (:u, :prior, :likelihood, :joint, :value, :g)
            @test getfield(loop, field) ≈ getfield(array, field) rtol=1e-12
        end
        @test loop.g ≈ loop.fd rtol=1e-5 atol=1e-7
        @test loop.constrained.z == z
        @test eltype(loop.constrained.z) == Int
    end
end

@testset "a loop cell's Categorical may read data probabilities whole" begin
    model = @rkppl begin
        mu[1:2] .~ Normal.(0, 5)
        @plate for i in eachindex(y)
            z[i] ~ Categorical(p)
            y[i] ~ Normal(mu[z[i]], 1.0)
        end
    end
    r = evaluate(model, (; p=W), (; mu=MU), (; z=Z))
    @test r.prior ≈ sum(logpdf.(Normal(0, 5), MU)) + sum(log(W[zi]) for zi in Z) rtol=1e-12
    @test r.likelihood ≈ sum(logpdf(Normal(MU[Z[i]], 1.0), Y[i]) for i in eachindex(Y)) rtol=1e-12
    @test r.g ≈ r.fd rtol=1e-5 atol=1e-7
end

@testset "a data array is indexed by an integer as in Julia" begin
    centers = [-2.0, 2.0]
    gathered = @rkppl begin
        w ~ Dirichlet(ones(2))
        s ~ Exponential(1)
        z[axes(y, 1)] .~ Categorical(w)
        y .~ Normal.(centers[z], s)
    end
    named = @rkppl begin
        w ~ Dirichlet(ones(2))
        s ~ Exponential(1)
        z[axes(y, 1)] .~ Categorical(w)
        m = centers[z]
        y .~ Normal.(m, s)
    end
    for model in (gathered, named)
        r = evaluate(model, (; centers), (; w=W, s=0.8), (; z=Z))
        @test r.likelihood ≈ sum(logpdf(Normal(centers[Z[i]], 0.8), Y[i]) for i in eachindex(Y)) rtol=1e-12
        @test r.g ≈ r.fd rtol=1e-5 atol=1e-7
    end
    scalar = @rkppl begin
        s ~ Exponential(1)
        k ~ Categorical(p)
        y .~ Normal.(centers[k], s)
    end
    for k in (1, 2)
        r = evaluate(scalar, (; centers, p=W), (; s=0.8), (; k))
        @test r.likelihood ≈ sum(logpdf.(Normal(centers[k], 0.8), Y)) rtol=1e-12
        @test r.prior ≈ logpdf(Exponential(1), 0.8) + log(W[k]) rtol=1e-12
    end
end

@testset "Gibbs draws a loop cell's integers like the declared array" begin
    rng = Xoshiro(20261117)
    n = 200
    y = [-2.0, 2.0][rand(rng, 1:2, n)] .+ randn(rng, n)
    z0 = [yi < 0 ? 1 : 2 for yi in y]
    function draws(model)
        plan = model() | (; y)
        built = build_kernel(plan)
        u0 = unconstrain(built.layout, (; w=[0.5, 0.5], mu=[-1.0, 1.0], z=z0))
        problem = prepare_gibbs(built, plan, u0; discrete=(; z=z0), backend=BACKEND)
        state = initialize_gibbs(Xoshiro(1), problem, u0; discrete=(; z=z0), warmup=200)
        return last(problem.integers).elementwise, gibbs_draws(state, 300)
    end
    loop_elementwise, loop = draws(LOOP)
    array_elementwise, array = draws(ARRAY)
    # Each cell reads its own element, so all elements are drawn at once.
    @test loop_elementwise && array_elementwise
    @test loop.discrete == array.discrete
    @test loop.positions ≈ array.positions rtol=1e-10
    @test count(loop.divergent) == 0
end
end
