using DifferentiationInterface: AutoEnzyme
using Distributions: Normal, Exponential, Uniform, Logistic, Beta, Dirichlet,
    cdf, logpdf
using Enzyme
using ReactiveKernelsPPL
using Test

# These oracles use the authored priors and scalar probability formulas.
# They do not inspect emitted expressions or reuse the generated density.
function _admission_findiff(f, u; h = 1e-5)
    g = similar(u)
    for i in eachindex(u)
        hi, lo = copy(u), copy(u)
        hi[i] += h
        lo[i] -= h
        g[i] = (f(hi) - f(lo)) / (2h)
    end
    return g
end

function _admission_check(program, data, oracle; names = nothing)
    original = deepcopy(data)
    plan = program isa StructuralPlan ? program : lower_rkppl(program, data)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    names === nothing || @test coordinate_names(built.layout) == names
    u = collect(range(-0.2, 0.3; length = built.layout.total))
    reference(v) = oracle(constrain(built.layout, v))
    expected = reference(u)
    for (preset, value) in ((:likelihood, expected.ll), (:prior, expected.pr),
            (:log_jacobian, expected.jac),
            (:sampler, expected.ll + expected.pr + expected.jac))
        query = prepare_query(built, bound, preset)
        @test Base.invokelatest(query, u) ≈ value rtol = 1e-12 atol = 1e-12
    end
    sampler = prepare_sampler(built, bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    grad = similar(u)
    value, _ = sampler_value_and_gradient!(sampler, grad, u)
    ref(v) = sum(values(reference(v)))
    @test value ≈ ref(u) rtol = 1e-12
    @test grad ≈ _admission_findiff(ref, u) rtol = 2e-5 atol = 2e-7
    @test data == original
    return (; bound, built, u)
end

@testset "authored factor priors with an intercept" begin
    data = Dict{Symbol,Any}(:y => [0.2, -0.1, 0.4], :g => [1, 2, 1],
        :x => [-1.0, 0.0, 1.0])
    factor(prior; intercept_prior = :(Normal(0, 1))) = quote
        a ~ $intercept_prior
        c[levels(g)] .~ $prior
        mu = a .+ c[g]
        y .~ Normal.(mu, 1)
    end
    ll(a, c; sigma = 1.0) = sum(logpdf(Normal(a + c[g], sigma), y)
        for (g, y) in zip(data[:g], data[:y]))
    fixed(q) = (; ll = ll(q.a, q.c),
        pr = logpdf(Normal(), q.a) + sum(logpdf.(Normal(0, 2), q.c)),
        jac = 0.0)
    literal = factor(:(Normal.(0, 2)))
    alias = Expr(:block, :(sd = 2.0), factor(:(Normal.(0, sd))).args...)
    sized = quote
        a ~ Normal(0, 1)
        c[1:2] .~ Normal.(0, 2)
        mu = a .+ c[g]
        y .~ Normal.(mu, 1)
    end
    for ast in (literal, alias, sized)
        _admission_check(ast, data, fixed;
            names = [:a, Symbol("c.1"), Symbol("c.2")])
    end
    hierarchical = Expr(:block, :(m ~ Normal(0, 1)),
        :(s ~ Exponential(1)), factor(:(Normal.(m, s))).args...)
    _admission_check(hierarchical, data, q -> (;
        ll = ll(q.a, q.c),
        pr = logpdf(Normal(), q.a) + logpdf(Normal(), q.m) +
            logpdf(Exponential(1), q.s) + sum(logpdf.(Normal(q.m, q.s), q.c)),
        jac = log(q.s)))

    # An improper joint density is still translated. No statistical-quality
    # check substitutes for the removed identifiability refusals.
    _admission_check(factor(:(Flat.()); intercept_prior = :(Flat())), data,
        q -> (; ll = ll(q.a, q.c), pr = 0.0, jac = 0.0))
    _admission_check(factor(:(Uniform.(-2, 2))), data, q -> (;
        ll = ll(q.a, q.c),
        pr = logpdf(Normal(), q.a) + sum(logpdf.(Uniform(-2, 2), q.c)),
        jac = sum(log((c + 2) * (2 - c) / 4) for c in q.c)))

    matrix = quote
        b[axes(X, 2)] .~ Normal.(0, 1)
        c[levels(g)] .~ Normal.(0, 2)
        X = hcat(ones(length(x)), x)
        mu = X * b .+ c[g]
        y .~ Normal.(mu, 1)
    end
    _admission_check(matrix, data, q -> (;
        ll = sum(logpdf(Normal(q.b[1] + q.b[2] * x + q.c[g], 1), y)
            for (x, g, y) in zip(data[:x], data[:g], data[:y])),
        pr = sum(logpdf.(Normal(), q.b)) + sum(logpdf.(Normal(0, 2), q.c)),
        jac = 0.0))

    # Exercise the legacy PopulationPrior route directly, independently of
    # the surface's ordinary array-parameter route.
    plan = StructuralPlan(
        [LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            ResponseEvidence(:none, nothing, nothing), :y_resp)],
        [PredictorSpec(:mu, IdentityLink, [
            TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept, :a),
            TermSpec(FactorTerm, [:g], NamedTuple(), :g, :c)], :mu)],
        [PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :g, 0.0, 2.0)],
        [SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing, :sigma)],
        AssignmentSpec[], Dict{Symbol,Any}(), 0;
        levelmaps = [LevelMap(:mu, :g, Int[], :levels, Colon())])
    _admission_check(plan, data, q -> (;
        ll = ll(q.mu[1], q.mu[2:3]; sigma = q.sigma),
        pr = logpdf(Normal(), q.mu[1]) +
            sum(logpdf.(Normal(0, 2), q.mu[2:3])) +
            logpdf(Exponential(1), q.sigma), jac = log(q.sigma)))

    # The likelihood ridge exists, but the proper authored prior changes
    # under that shift; preserving both effects is intentional.
    a, c, shift = 0.2, [-0.3, 0.4], 0.7
    pr(a, c) = logpdf(Normal(), a) + sum(logpdf.(Normal(0, 2), c))
    @test ll(a, c) ≈ ll(a + shift, c .- shift)
    @test !isapprox(pr(a, c), pr(a + shift, c .- shift))
end

@testset "full-cover factor in an R2D2 predictor" begin
    # R2D2 has a separate surface-lowering path that used to apply the same
    # identifiability refusal. Its variance decomposition remains unchanged.
    data = Dict{Symbol,Any}(:y => [0.2, -0.1, 0.4], :g => [1, 2, 1])
    ast = quote
        R2 ~ Beta(1, 1)
        phi ~ Dirichlet([1.0, 1.0])
        mu = a .+ c[g]
        r2d2(mu, R2, phi, 1.0)
        y .~ Normal.(mu, 1)
    end
    _admission_check(ast, data, q -> begin
        # Each indicator has sample variance 1/3 in this three-row data set.
        scales = sqrt.(q.phi .* q.R2 ./ (1 / 3))
        (; ll = sum(logpdf(Normal(q.mu[1] + q.mu[1 + g], 1), y)
                for (g, y) in zip(data[:g], data[:y])),
            pr = logpdf(Normal(), q.mu[1]) +
                sum(logpdf.(Normal.(0, scales), q.mu[2:3])) +
                logpdf(Beta(1, 1), q.R2) +
                logpdf(Dirichlet([1.0, 1.0]), q.phi),
            jac = log(q.R2 * (1 - q.R2)) + sum(log, q.phi))
    end)
end

@testset "authored ordinal intercept and thresholds" begin
    data = Dict{Symbol,Any}(:y => [1, 2, 3, 2], :x => [-1.0, 0.0, 1.0, 0.5])
    cumulative = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Ordered(Normal(0, 1), 2)
        eta = a .+ b .* x
        y .~ Ordinal.(Cumulative(), LogitLink(), eta, Ref(c))
    end
    ordered = Expr(:block, cumulative.args[1:end-1]...,
        :(y .~ OrderedLogistic.(eta, Ref(c))))
    stopping = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c[1:2] .~ Normal.(0, 1)
        eta = a .+ b .* x
        y .~ Ordinal.(StoppingRatio(), LogitLink(), eta, Ref(c))
    end
    cumulative_ll(a, b, c) = sum(begin
            eta = a + b * x
            lo = y == 1 ? 0.0 : cdf(Logistic(), c[y - 1] - eta)
            hi = y == 3 ? 1.0 : cdf(Logistic(), c[y] - eta)
            log(hi - lo)
        end for (x, y) in zip(data[:x], data[:y]))
    for ast in (cumulative, ordered)
        _admission_check(ast, data, q -> (;
            ll = cumulative_ll(q.a, q.b, q.c),
            pr = logpdf(Normal(), q.a) + logpdf(Normal(), q.b) +
                sum(logpdf.(Normal(), q.c)),
            jac = sum(log, diff(q.c))))
    end
    _admission_check(stopping, data, q -> (;
        ll = sum(begin
                eta = q.a + q.b * x
                p = y == 3 ? 1.0 : cdf(Logistic(), q.c[y] - eta)
                for j in 1:y-1
                    p *= 1 - cdf(Logistic(), q.c[j] - eta)
                end
                log(p)
            end for (x, y) in zip(data[:x], data[:y])),
        pr = logpdf(Normal(), q.a) + logpdf(Normal(), q.b) +
            sum(logpdf.(Normal(), q.c)), jac = 0.0))
    a, b, c, shift = 0.2, -0.3, [-0.7, 0.6], 0.7
    @test cumulative_ll(a, b, c) ≈ cumulative_ll(a + shift, b, c .+ shift)
end
