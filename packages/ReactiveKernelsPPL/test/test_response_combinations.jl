using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Synthetic public models. Independent Distributions oracles check the
# authored response expression; differences check ordinary native reverse AD.
function _rc_check(ast, data, oracle; observations = (:y,))
    plan = bind_data(lower_rkppl(ast, data; conditioned = observations),
        Dict{Symbol,Any}(pairs(data)))
    return _rc_check_bound(plan, oracle)
end

function _rc_check_bound(plan, oracle)
    validate_plan(plan)
    built = build_kernel(plan)
    u = [0.2sin(i) for i in 1:built.layout.total]
    nt = constrain(built.layout, u)
    likelihood = prepare_query(built, plan, :likelihood)
    f = prepare_query(built, plan, :sampler)
    @test Base.invokelatest(likelihood, u) ≈ oracle(nt)
    if !isempty(u)
        ad = Base.invokelatest(prepare_ad, f, AutoEnzyme(; mode = Enzyme.Reverse), u;
            active = :unconstrained)
        g = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, ad, similar(u), u)[2]
        h = cbrt(eps(Float64))
        fd = map(eachindex(u)) do i
            up, down = copy(u), copy(u)
            up[i] += h
            down[i] -= h
            (Base.invokelatest(f, up) - Base.invokelatest(f, down)) / (2h)
        end
        @test all(isfinite, g)
        @test g ≈ fd rtol=2e-5 atol=2e-7
    end
    return (; plan, built, u, f)
end

function _rc_mi_plan(family, link, y; scale=nothing, weights=nothing,
        evidence=ResponseEvidence(:none, nothing, nothing), range=nothing,
        trials=nothing)
    cols = Dict{Symbol,AbstractVector}(:y=>y, :Jobs=>[1, 3],
        :wt=>[1.0, 8.0, 2.0, 9.0], :lo=>fill(-1.0, 4),
        :hi=>[1.0, 2.0, 1.5, 3.0], :n=>[2, 4, 5, 6])
    r = LikelihoodSpec(family, link, :y, :mu, scale, weights,
        evidence, :y_resp, trials, range; mi_jobs=:Jobs)
    return StructuralPlan([r],
        [PredictorSpec(:mu, link === LogLink ? LogLink : IdentityLink,
            [TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept, :intercept)], :mu)],
        [PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        SampledParameter[], AssignmentSpec[], cols, 4)
end

@testset "packed missing-data response combinations" begin
    plan = _rc_mi_plan(GaussianFam, IdentityLink, [0.2, -0.4];
        scale=0.5, weights=:wt,
        evidence=ResponseEvidence(:truncated, :lo, :hi), range=2:4)
    _rc_check_bound(plan, nt -> 2logpdf(truncated(Normal(nt.mu[1], 0.5),
        -1.0, 1.5), -0.4))
    # An intercept needs no full-length predictor column or artificial data.
    plan = _rc_mi_plan(GaussianFam, IdentityLink, [0.2, -0.4]; scale=0.5)
    for key in (:wt, :lo, :hi, :n)
        delete!(plan.columns, key)
    end
    _rc_check_bound(plan, nt -> sum(logpdf.(Normal(nt.mu[1], 0.5), [0.2, -0.4])))
    # A packed selection may include every row of its own observation axis.
    plan.columns[:Jobs] = collect(1:4)
    plan.columns[:y] = [0.2, -0.4, 0.1, 0.3]
    _rc_check_bound(plan, nt -> sum(logpdf.(Normal(nt.mu[1], 0.5), plan.columns[:y])))
    for (family, link, y, dist, trials) in (
            (BernoulliLogitFam, LogitLink, [true, false],
                m -> Bernoulli(inv(1+exp(-m))), nothing),
            (PoissonLogFam, LogLink, [0, 2], m -> Poisson(exp(m)), nothing),
            (BinomialLogitFam, LogitLink, [1, 3],
                m -> Binomial(2, inv(1+exp(-m))), :n))
        plan = _rc_mi_plan(family, link, y; weights=:wt, trials)
        _rc_check_bound(plan, nt -> begin
            m = nt.mu[1]
            d1 = dist(m)
            d2 = trials === nothing ? d1 : Binomial(5, inv(1+exp(-m)))
            logpdf(d1, y[1]) + 2logpdf(d2, y[2])
        end)
    end
end

function _rc_plate_plan(family, link, y; scale=nothing, weights=nothing,
        evidence=ResponseEvidence(:none, nothing, nothing), range=nothing)
    cols = Dict{Symbol,AbstractVector}(:y=>y, :wt=>[1.0, 2.0, 0.5, 1.5])
    r = LikelihoodSpec(family, link, :y, :z, scale, weights,
        evidence, :y_resp, nothing, range)
    params = scale === :s ? [SampledParameter(:s, :exponential,
        (arg1=1.0,), nothing, :s)] : SampledParameter[]
    return StructuralPlan([r], PredictorSpec[], PopulationPrior[],
        params, AssignmentSpec[], cols, 4;
        plate_parameters=[PlateParameter(:z, :normal, (arg1=0.0,arg2=1.0), nothing)])
end

@testset "direct plate-location response combinations" begin
    y = [0.1, 0.2, -0.3, 0.4]
    plan = _rc_plate_plan(GaussianFam, IdentityLink, y; scale=:s,
        weights=:wt, evidence=ResponseEvidence(:truncated, -1.0, 1.0), range=1:4)
    _rc_check_bound(plan, nt -> sum(plan.columns[:wt] .* map(eachindex(y)) do i
        logpdf(truncated(Normal(nt.z[i], nt.s), -1.0, 1.0), y[i])
    end))
    for (family, link, y, dist) in (
            (BernoulliLogitFam, LogitLink, [true, false, true, false],
                m -> Bernoulli(inv(1+exp(-m)))),
            (PoissonLogFam, LogLink, [0, 2, 1, 3], m -> Poisson(exp(m))))
        plan = _rc_plate_plan(family, link, y; weights=:wt)
        _rc_check_bound(plan, nt -> sum(plan.columns[:wt] .* logpdf.(dist.(nt.z), y)))
    end
end

@testset "mixture response combinations" begin
    data = (; y = [0.2, -0.4, 0.6], x = [0.1, -0.2, 0.5],
        w = [0.3, 0.7], wt = [1.0, 2.0, 0.5])
    ast = quote
        s ~ Exponential(1)
        y .~ weighted.(MixtureModel.(vcat.(Normal.(x, s),
            Normal.(0.0, s)), Ref(w)), wt)
    end
    _rc_check(ast, data, nt -> sum(data.wt .* map(eachindex(data.y)) do i
        logpdf(MixtureModel([Normal(data.x[i], nt.s), Normal(0, nt.s)],
            data.w), data.y[i])
    end))
    # Shared probabilities are whole values; their width is K, not n.
    for w in ([1.0, 0.0], [0.0, 1.0])
        d = merge(data, (; w))
        _rc_check(ast, d, nt -> sum(d.wt .* map(eachindex(d.y)) do i
            logpdf(MixtureModel([Normal(d.x[i], nt.s), Normal(0, nt.s)],
                d.w), d.y[i])
        end))
    end
    for w in ([0.2, 0.2], [-0.1, 1.1], [NaN, 0.0], [1.0])
        # refused: malformed probability vector, violating MixtureModel's domain.
        @test_throws ContractValidationError bind_data(
            lower_rkppl(ast, merge(data, (; w)); conditioned=(:y,)),
            Dict{Symbol,Any}(pairs(merge(data, (; w)))))
    end

    ast = quote
        p ~ Beta(2, 2)
        y .~ MixtureModel.(vcat.(Binomial.(n1, p), Binomial.(n2, 0.3)),
            Ref([0.3, 0.7]))
    end
    data = (; y = [0, 3, 2], n1 = [1, 2, 1], n2 = [4, 4, 4])
    _rc_check(ast, data, nt -> sum(map(eachindex(data.y)) do i
        logpdf(MixtureModel([Binomial(data.n1[i], nt.p),
            Binomial(data.n2[i], 0.3)], [0.3, 0.7]), data.y[i])
    end))

    ast = quote
        m ~ Normal(0, 1)
        y[1:2] .~ MixtureModel.(vcat.(Normal.(m, 0.5), Normal.(1.0, 0.5)),
            Ref([0.4, 0.6]))
    end
    # Complete literal selection; partial coverage is refused at binding (1uhcm3b).
    data = (; y = [0.2, -0.4])
    _rc_check(ast, data, nt -> sum(logpdf.(
        MixtureModel([Normal(nt.m, 0.5), Normal(1, 0.5)], [0.4, 0.6]),
        data.y)))

    for wrapper in (:truncated, :censored)
        ast = quote
            m ~ Normal(0, 1)
            y .~ $wrapper.(MixtureModel.(vcat.(Normal.(m, 0.5),
                Normal.(1.0, 0.5)), Ref([0.4, 0.6])), -0.5, 1.5)
        end
        data = (; y = [-0.5, 0.2, 1.5])
        fun = wrapper === :truncated ? truncated : censored
        _rc_check(ast, data, nt -> sum(logpdf.(fun(
            MixtureModel([Normal(nt.m, 0.5), Normal(1, 0.5)], [0.4, 0.6]),
            -0.5, 1.5), data.y)))
    end
    ast = quote
        m ~ Normal(0, 1)
        y .~ interval_censored.(MixtureModel.(vcat.(Normal.(m, 0.5),
            Normal.(1.0, 0.5)), Ref([0.4, 0.6])), hi)
    end
    data = (; y=[0.1, 0.2, 0.3], hi=[1.0, 1.5, 2.0])
    _rc_check(ast, data, nt -> begin
        d = MixtureModel([Normal(nt.m,0.5),Normal(1,0.5)],[0.4,0.6])
        sum(log.(cdf.(d,data.hi) .- cdf.(d,data.y)))
    end)
end
