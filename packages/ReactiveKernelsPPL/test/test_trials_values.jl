# Trials are data values. A bound number broadcasts over the observations
# exactly as `fill(n, length(y))` does, and a definition reading only data
# (a number or an array) binds as data, as `weighted` weights do. Densities
# and Enzyme reverse gradients match Distributions oracles, and the number,
# literal and column spellings agree. Self-contained: runs standalone.
using DifferentiationInterface
using Distributions: Beta, BetaBinomial, Binomial, Normal, logpdf
using Enzyme
using LogExpFunctions: log1pexp, logaddexp, logistic, logit
using ReactiveKernels
using ReactiveKernelsPPL
using Test

const _TV_X = [-1.2, -0.4, 0.1, 0.6, 1.3, 0.0, 0.8, -0.7]
# Zero, full, interior counts; every count is at most five.
const _TV_Y = [0, 5, 0, 2, 4, 1, 3, 1]
const _TV_U = [0.3, -0.2, 0.4]

function _tv_model(ast, data; conditioned = (:y,))
    values = Dict{Symbol,Any}(pairs(data))
    plan = bind_data(lower_rkppl(ast, values; conditioned), values)
    built = build_kernel(plan)
    kernel = prepare_query(built, plan, :sampler)
    return (; plan, built, kernel)
end

_tv_value(f, u) = Base.invokelatest(f.kernel, u)
_tv_pointwise(f, u) =
    Base.invokelatest(prepare_query(f.built, f.plan, :pointwise), u)

function _tv_gradient(f, u)
    ad = Base.invokelatest(prepare_ad, f.kernel,
        AutoEnzyme(; mode = Enzyme.Reverse), u; active = :unconstrained)
    return Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
        ad, similar(u), u)
end

function _tv_findiff(f, u)
    h = cbrt(eps(Float64))
    return [(f(u .+ [j == i ? h : 0.0 for j in eachindex(u)]) -
             f(u .- [j == i ? h : 0.0 for j in eachindex(u)])) / (2h)
        for i in eachindex(u)]
end

function _tv_check(f, oracle, u)
    @test _tv_value(f, u) ≈ oracle(u) rtol = 2e-12
    value, gradient = _tv_gradient(f, u)
    @test value ≈ oracle(u) rtol = 2e-12
    @test gradient ≈ _tv_findiff(oracle, u) rtol = 2e-5 atol = 2e-7
end

_tv_eta(u) = u[1] .+ u[2] .* _TV_X
_tv_normal_prior(u) = logpdf(Normal(), u[1]) + logpdf(Normal(), u[2])
# A Beta(2, 2) declaration: constrained by `logistic`, plus its Jacobian.
_tv_beta22(v) = (p = logistic(v); logpdf(Beta(2, 2), p) + log(p) + log1p(-p))

_tv_zib_cell(y, n, p, zi) = y == 0 ?
    logaddexp(log(zi), log1p(-zi) + logpdf(Binomial(n, p), y)) :
    log1p(-zi) + logpdf(Binomial(n, p), y)

const _TV_BINOMIAL = quote
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    eta = a .+ b .* x
    y .~ Binomial.(n, logistic.(eta))
end
_tv_binomial_oracle(n) = u ->
    sum(logpdf.(Binomial.(n, logistic.(_tv_eta(u))), _TV_Y)) + _tv_normal_prior(u)

const _TV_ZIB = quote
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    zi ~ Beta(2, 2)
    eta = a .+ b .* x
    y .~ ZeroInflatedBinomial.(n, logistic.(eta), zi)
end
_tv_zib_oracle(n) = u -> sum(_tv_zib_cell.(_TV_Y, n, logistic.(_tv_eta(u)),
    logistic(u[3]))) + _tv_normal_prior(u) + _tv_beta22(u[3])

@testset "a bound number is the trials of every observation" begin
    u2 = _TV_U[1:2]
    for (name, ast, oracle, u) in (
            ("Binomial logit", _TV_BINOMIAL, _tv_binomial_oracle, u2),
            ("ZeroInflatedBinomial", _TV_ZIB, _tv_zib_oracle, _TV_U))
        @testset "$name" begin
            number = _tv_model(ast, (; x = _TV_X, y = _TV_Y, n = 5))
            _tv_check(number, oracle(5), u)
            column = _tv_model(ast, (; x = _TV_X, y = _TV_Y, n = fill(5, 8)))
            @test _tv_value(number, u) ≈ _tv_value(column, u) rtol = 1e-13
            @test _tv_gradient(number, u)[2] ≈ _tv_gradient(column, u)[2] rtol = 1e-12
            @test _tv_pointwise(number, u).y ≈ _tv_pointwise(column, u).y rtol = 1e-13
        end
    end
    # The fused Binomial-logit form reads a number like a literal: one
    # `sum(log1pexp, eta)` and a folded binomial-coefficient constant.
    number = _tv_model(_TV_BINOMIAL, (; x = _TV_X, y = _TV_Y, n = 5))
    source = string(kernel_expr(number.plan, number.built.layout))
    @test occursin("_ppl_lchoose_y_resp", source)
    @test occursin("sum(log1pexp", source)
    @test !occursin("loggamma",
        sprint(print, Base.remove_linenums!(deepcopy(code_expr(number.kernel)))))
    # A Beta-sampled probability (the probability-space Binomial plate).
    f = _tv_model(quote
        p ~ Beta(2, 2)
        y .~ Binomial.(n, p)
    end, (; y = _TV_Y, n = 5))
    _tv_check(f, u -> sum(logpdf.(Binomial(5, logistic(u[1])), _TV_Y)) +
        _tv_beta22(u[1]), [0.2])
end

@testset "the reported capture-recapture model takes a number of occasions" begin
    # A zero-inflated count over individual detection logits, with the
    # occasion count shared by every individual.
    m = @rkppl begin
        omega ~ Uniform(0, 1)
        mean_p ~ Uniform(0, 1)
        sigma ~ Uniform(0, 5)
        eps_raw[axes(captures_all, 1)] .~ Normal.(0, 1)
        logit_p = logit(mean_p) .+ sigma .* eps_raw
        captures_all .~ ZeroInflatedBinomial.(trials, logistic.(logit_p), 1 - omega)
    end
    captures_all = [0, 2, 5, 0, 1, 0, 3, 0, 4, 0, 1, 0]
    u = collect(range(-0.6, 0.7; length = 3 + length(captures_all)))
    fs = map((5, fill(5, length(captures_all)))) do trials
        plan = m(; trials) | (; captures_all)
        built = build_kernel(plan)
        (; plan, built, kernel = prepare_query(built, plan, :sampler))
    end
    # The inline array product belongs to the predictor broadcast. There is
    # no intermediate sampled-size vector between it and the scalar logit.
    terms = only(p for p in fs[1].plan.predictors if p.name === :logit_p).terms
    @test any(t -> t.kind === ComposedTerm &&
        t.options.tree == :(sigma .* eps_raw), terms)
    @test !any(a -> a.expr == :(sigma .* eps_raw), fs[1].plan.assignments)
    oracle = u -> begin
        omega, mean_p = logistic(u[1]), logistic(u[2])
        sigma = 5 * logistic(u[3])
        eps_raw = u[4:end]
        p = logistic.(logit(mean_p) .+ sigma .* eps_raw)
        sum(_tv_zib_cell.(captures_all, 5, p, 1 - omega)) +
            sum(logpdf.(Normal(), eps_raw)) +
            # Uniform densities cancel their interval widths in the Jacobian.
            sum(log(logistic(v)) + log1p(-logistic(v)) for v in u[1:3])
    end
    _tv_check(fs[1], oracle, u)
    @test _tv_value(fs[1], u) ≈ _tv_value(fs[2], u) rtol = 1e-13
    @test _tv_gradient(fs[1], u)[2] ≈ _tv_gradient(fs[2], u)[2] rtol = 1e-12
end

@testset "anonymous array predictor keeps broadcast axes" begin
    m = @rkppl begin
        a ~ Normal(0, 1)
        s ~ Normal(0, 1)
        z[axes(x, 1)] .~ Normal.(0, 1)
        eta = a .+ s .* z
        y .~ Normal.(eta, 1)
    end
    function density(x, y, u)
        plan = m(; x) | (; y)
        built = build_kernel(plan)
        f = prepare_query(built, plan, :sampler)
        return Base.invokelatest(f, u), f, plan
    end
    v, f, _ = density(Float64[], Float64[], [0.2, -0.3])
    @test v ≈ logpdf(Normal(), 0.2) + logpdf(Normal(), -0.3)
    ad = Base.invokelatest(prepare_ad, f,
        AutoEnzyme(; mode = Enzyme.Reverse), [0.2, -0.3];
        active = :unconstrained)
    _, g = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, ad,
        zeros(2), [0.2, -0.3])
    @test g ≈ [-0.2, 0.3]
    v1, _, _ = density([0.0], [0.4, -0.2, 0.1], [0.2, -0.3, 0.1])
    @test v1 ≈ sum(logpdf.(Normal.([0.17], 1), [0.4, -0.2, 0.1])) +
        sum(logpdf.(Normal(), [0.2, -0.3, 0.1]))
    @test_throws DimensionMismatch density([0.0, 0.0], [0.0, 0.0, 0.0],
        [0.2, -0.3, 0.1, -0.1])
    named = @rkppl begin
        a ~ Normal(0, 1)
        s ~ Normal(0, 1)
        z[axes(x, 1)] .~ Normal.(0, 1)
        v = s .* z
        eta = a .+ v
        y .~ Normal.(eta, 1)
        y2 .~ Normal.(v, 1)
    end
    plan = named(; x = [0.0, 0.0]) | (; y = [0.4, -0.2], y2 = [0.1, 0.3])
    @test any(a -> a.name === :v, plan.assignments) ||
        any(d -> d.name === :v, plan.derived)
end

@testset "a definition reading only data supplies trials" begin
    u = _TV_U[1:2]
    for (name, def, data) in (
            ("literal definition", :(n = 5), (;)),
            ("scalar computed from data", :(n = k + 2), (; k = 3)),
            ("array computed from data", :(n = t .+ 1), (; t = fill(4, 8))),
            ("module call on data", :(n = fill(5, length(y))), (;)))
        @testset "$name" begin
            ast = Expr(:block, def, _TV_BINOMIAL.args...)
            f = _tv_model(ast, merge((; x = _TV_X, y = _TV_Y), data))
            _tv_check(f, _tv_binomial_oracle(5), u)
        end
    end
end

@testset "BetaBinomial2, Multinomial and Binomial mixtures read a number" begin
    u = [0.3, -0.2, 0.5]
    bb = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        lphi ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ BetaBinomial2.(n, logistic.(eta), exp.(lphi))
    end
    f = _tv_model(bb, (; x = _TV_X, y = _TV_Y, n = 5))
    _tv_check(f, u -> begin
        mu, phi = logistic.(_tv_eta(u)), exp(u[3])
        sum(logpdf.(BetaBinomial.(5, mu .* phi, (1 .- mu) .* phi), _TV_Y)) +
            _tv_normal_prior(u) + logpdf(Normal(), u[3])
    end, u)

    c1, c2, c3 = [1, 0, 2, 3], [2, 1, 0, 0], [0, 2, 1, 0]
    multinomial = quote
        s ~ Dirichlet([1.0, 2.0, 1.5])
        eachrow(hcat(c1, c2, c3)) .~ Multinomial.(N, Ref(s))
    end
    fs = [_tv_model(multinomial, (; c1, c2, c3, N); conditioned = (:c1, :c2, :c3))
        for N in (3, fill(3, 4))]
    lit = _tv_model(:(begin
        s ~ Dirichlet([1.0, 2.0, 1.5])
        eachrow(hcat(c1, c2, c3)) .~ Multinomial.(3, Ref(s))
    end), (; c1, c2, c3); conditioned = (:c1, :c2, :c3))
    us = [0.2, -0.4]
    @test _tv_value(fs[1], us) ≈ _tv_value(lit, us) rtol = 1e-13
    @test _tv_value(fs[1], us) ≈ _tv_value(fs[2], us) rtol = 1e-13
    @test _tv_gradient(fs[1], us)[2] ≈ _tv_gradient(lit, us)[2] rtol = 1e-12

    mixture = quote
        p ~ Beta(2, 2)
        y .~ MixtureModel.(vcat.(Binomial.(n1, p), Binomial.(n2, 0.3)), Ref([0.4, 0.6]))
    end
    y = [0, 3, 5, 1, 2]
    f = _tv_model(mixture, (; y, n1 = 5, n2 = 3))
    _tv_check(f, u -> begin
        p = logistic(u[1])
        sum(logaddexp.(log(0.4) .+ logpdf.(Binomial(5, p), y),
            log(0.6) .+ logpdf.(Binomial(3, 0.3), y))) + _tv_beta22(u[1])
    end, [0.25])
end

@testset "names-only plans bind a number, and a build serves other numbers" begin
    plan = lower_rkppl(_TV_BINOMIAL, (:x, :y, :n); conditioned = (:y,))
    bound = bind_data(plan, (; x = _TV_X, y = _TV_Y, n = 5))
    built = build_kernel(bound)
    u = _TV_U[1:2]
    q = prepare_query(built, bound, :sampler)
    @test Base.invokelatest(q, u) ≈ _tv_binomial_oracle(5)(u) rtol = 2e-12
    # Another number binds without lowering or building again.
    six = bind_data(plan, (; x = _TV_X, y = _TV_Y, n = 6))
    q6 = prepare_query(built, six, :sampler)
    @test Base.invokelatest(q6, u) ≈ _tv_binomial_oracle(6)(u) rtol = 2e-12
    # The value-aware plan rebinds the same way.
    m = _tv_model(_TV_BINOMIAL, (; x = _TV_X, y = _TV_Y, n = 5))
    rebound = bind_data(m.plan, (; x = _TV_X, y = _TV_Y, n = 7))
    q7 = prepare_query(m.built, rebound, :sampler)
    @test Base.invokelatest(q7, u) ≈ _tv_binomial_oracle(7)(u) rtol = 2e-12
end

@testset "trials stay nonnegative integer data" begin
    # refused: a trial count is discrete data; a value computed from a
    # sampled parameter has no Binomial meaning (Binomial support)
    @test_throws SurfaceLoweringError _tv_model(
        Expr(:block, :(c ~ Normal(0, 1)), :(n = 5 + c), _TV_BINOMIAL.args...),
        (; x = _TV_X, y = _TV_Y))
    # refused: Binomial trials must be a nonnegative integer (Binomial support)
    @test_throws ContractValidationError _tv_model(_TV_BINOMIAL,
        (; x = _TV_X, y = _TV_Y, n = 5.0))
    @test_throws ContractValidationError _tv_model(_TV_BINOMIAL,
        (; x = _TV_X, y = _TV_Y, n = -1))
    @test_throws ContractValidationError _tv_model(_TV_BINOMIAL,
        (; x = _TV_X, y = _TV_Y, n = true))
    # refused: an observed count above its trials is outside the support (wrong data)
    @test_throws ContractValidationError _tv_model(_TV_BINOMIAL,
        (; x = _TV_X, y = _TV_Y, n = 4))
    @test_throws ContractValidationError _tv_model(
        Expr(:block, :(n = k - 2), _TV_BINOMIAL.args...),
        (; x = _TV_X, y = _TV_Y, k = 3))
end
