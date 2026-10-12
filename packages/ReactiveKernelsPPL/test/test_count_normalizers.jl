# A count likelihood evaluates its data-only normalizer once, at preparation.
# The Binomial-logit GLM base case is one whole-vector reduction whose log
# binomial coefficients are a named data-only recipe. The Binomial,
# zero-inflated Binomial and Poisson observation plates sum a `logkernel`
# plate and a `lognormalizer` plate over the bound response and trials; the
# normalizer sum folds to one constant and the pointwise plate keeps the
# whole `logpdf`. Values, gradients and pointwise densities match
# Distributions oracles. (`_distributional_model` / `_distributional_check`
# come from test_distributional_links.jl and `_zib_oracle_cell` from
# test_zib.jl, both included first.)
using Distributions: Beta, Binomial, Gamma, Normal, Poisson, ccdf, cdf, logpdf
using LogExpFunctions: logistic
using ReactiveKernels
using ReactiveKernelsPPL
using StatsFuns: normcdf
using Test

const _CN_X = [-1.2, -0.4, 0.1, 0.6, 1.3, 0.0, 0.8, -0.7]
const _CN_N = [3, 5, 0, 4, 6, 2, 7, 1]
# Zero, full, zero-trial and interior counts.
const _CN_Y = [0, 5, 0, 2, 6, 1, 3, 1]
const _CN_W = [1.0, 0.5, 2.0, 1.0, 1.5, 1.0, 0.25, 3.0]

_cn_source(f) = string(kernel_expr(f.plan, f.built.layout))
_cn_prepared(f) = sprint(show, MIME"text/plain"(), readable_code(f.kernel))
_cn_folded(code, name) = occursin(Regex("\\b$(name) = -?[0-9]"), code)
_cn_eta(u) = u[1] .+ u[2] .* _CN_X
_cn_normal_prior(u) = sum(logpdf.(Normal(), u))

function _cn_linear(response, data)
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
    end
    push!(ast.args, response)
    return _distributional_model(ast, merge((; x = _CN_X), data))
end

function _cn_pointwise(f, u)
    pointwise = prepare_query(f.built, f.plan, :pointwise)
    return Base.invokelatest(pointwise, u).y
end

@testset "binomial-logit GLM base case folds its binomial coefficients" begin
    u = [0.3, -0.2]
    y5 = min.(_CN_Y, 5)
    for (trials, data, counts, n) in (
            (:n, (; n = _CN_N), _CN_Y, _CN_N),
            (5, (;), y5, fill(5, length(y5))),
            (:m, (; m = 5), y5, fill(5, length(y5))))
        f = _cn_linear(:(y .~ Binomial.($trials, logistic.(eta))),
            merge(data, (; y = counts)))
        source = _cn_source(f)
        @test occursin("dot(", source)
        @test occursin("_ppl_lchoose_y_resp", source)
        code = _cn_prepared(f)
        @test !occursin("loggamma", code)
        @test _cn_folded(code, "_ppl_lchoose_y_resp")
        oracle = u -> sum(logpdf.(Binomial.(n, logistic.(_cn_eta(u))), counts)) +
            _cn_normal_prior(u)
        _distributional_check(f, oracle, u)
        @test _cn_pointwise(f, u) ≈ logpdf.(Binomial.(n, logistic.(_cn_eta(u))), counts)
    end
    # One trial count stretching over the rows keeps the plate path, whose
    # normalizer still folds.
    f = _cn_linear(:(y .~ Binomial.(n, logistic.(eta))), (; n = [5], y = y5))
    @test !occursin("dot(", _cn_source(f))
    @test _cn_folded(_cn_prepared(f), "_ppl_lnorm_y_resp")
    _distributional_check(f,
        u -> sum(logpdf.(Binomial.(5, logistic.(_cn_eta(u))), y5)) + _cn_normal_prior(u), u)
end

@testset "count observation plates fold their normalizer" begin
    u = [0.3, -0.2]
    eta = _cn_eta(u)
    cases = (
        ("weighted binomial logit",
            :(y .~ weighted.(Binomial.(n, logistic.(eta)), w)), (; n = _CN_N, w = _CN_W),
            eta -> _CN_W .* logpdf.(Binomial.(_CN_N, logistic.(eta)), _CN_Y)),
        ("binomial probit", :(y .~ Binomial.(n, normcdf.(eta))), (; n = _CN_N),
            eta -> logpdf.(Binomial.(_CN_N, normcdf.(eta)), _CN_Y)),
        ("zero-inflated binomial logit",
            :(y .~ ZeroInflatedBinomial.(n, logistic.(eta), 0.3)), (; n = _CN_N),
            eta -> _zib_oracle_cell.(_CN_Y, _CN_N, logistic.(eta), 0.3)),
        ("weighted poisson log",
            :(y .~ weighted.(Poisson.(exp.(eta)), w)), (; w = _CN_W),
            eta -> _CN_W .* logpdf.(Poisson.(exp.(eta)), _CN_Y)),
    )
    for (name, response, data, pointwise) in cases
        @testset "$name" begin
            f = _cn_linear(response, merge(data, (; y = _CN_Y)))
            source = _cn_source(f)
            @test occursin(".logkernel(", source)
            @test occursin(".lognormalizer(", source)
            @test _cn_folded(_cn_prepared(f), "_ppl_lnorm_y_resp")
            _distributional_check(f, u -> sum(pointwise(_cn_eta(u))) + _cn_normal_prior(u), u)
            @test _cn_pointwise(f, u) ≈ pointwise(eta)
        end
    end
    # The plate cell reads the logit directly: no probability vector.
    f = _cn_linear(:(y .~ ZeroInflatedBinomial.(n, logistic.(eta), 0.3)),
        (; n = _CN_N, y = _CN_Y))
    @test occursin("zero_inflated_binomial(; n = n, logit = eta", _cn_source(f))
    @test !occursin("_ppl_mu_y_resp", _cn_source(f))
    # A visible Binomial kernel plate evaluates no `loggamma`.
    f = _cn_linear(:(y .~ weighted.(Binomial.(n, logistic.(eta)), w)),
        (; n = _CN_N, w = _CN_W, y = _CN_Y))
    @test !occursin("loggamma", _cn_prepared(f))
end

@testset "count plates over sampled probabilities and rates fold their normalizer" begin
    # Bare sampled locations take the probability and rate HAVE routes.
    beta_prior(v) = (p = logistic(v); (p, logpdf(Beta(2, 2), p) + log(p) + log1p(-p)))
    cases = (
        ("binomial", :(p ~ Beta(2, 2)), :(y .~ Binomial.(n, p)),
            v -> ((p, lp) = beta_prior(v); lp + sum(logpdf.(Binomial.(_CN_N, p), _CN_Y)))),
        ("zero-inflated binomial", :(p ~ Beta(2, 2)), :(y .~ ZeroInflatedBinomial.(n, p, 0.3)),
            v -> ((p, lp) = beta_prior(v); lp + sum(_zib_oracle_cell.(_CN_Y, _CN_N, p, 0.3)))),
        ("poisson", :(lambda ~ Gamma(2, 1)), :(y .~ Poisson.(lambda)),
            v -> logpdf(Gamma(2, 1), exp(v)) + v + sum(logpdf.(Poisson(exp(v)), _CN_Y))),
    )
    for (name, prior, response, oracle) in cases
        @testset "$name" begin
            f = _distributional_model(Expr(:block, prior, response), (; n = _CN_N, y = _CN_Y))
            @test occursin(".logkernel(", _cn_source(f))
            @test _cn_folded(_cn_prepared(f), "_ppl_lnorm_y_resp")
            for v in (-0.4, 0.7)
                _distributional_check(f, u -> oracle(only(u)), [v])
            end
        end
    end
end

@testset "invalid zero inflation stays outside the support" begin
    f = _distributional_model(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        zi = 1.5 * logistic(c)
        eta = a .+ b .* x
        y .~ ZeroInflatedBinomial.(n, logistic.(eta), zi)
    end, (; x = _CN_X, n = _CN_N, y = _CN_Y))
    @test _cn_folded(_cn_prepared(f), "_ppl_lnorm_y_resp")
    @test Base.invokelatest(f.kernel, [0.3, -0.2, 3.0]) == -Inf
    valid(u) = (zi = 1.5 * logistic(u[3]);
        sum(_zib_oracle_cell.(_CN_Y, _CN_N, logistic.(_cn_eta(u)), zi)) + _cn_normal_prior(u))
    _distributional_check(f, valid, [0.3, -0.2, -3.0])
end

@testset "evidence keeps the whole-logpdf plate" begin
    counts = [1, 4, 2, 3, 4, 1, 2, 3]
    f = _cn_linear(:(y .~ censored.(Binomial.(6, logistic.(eta)), 1, 4)), (; y = counts))
    @test !occursin("_ppl_lnorm_", _cn_source(f))
    function cell(y, p)
        d = Binomial(6, p)
        y == 1 && return log(cdf(d, 1))
        y == 4 && return log(ccdf(d, 3))
        return logpdf(d, y)
    end
    _distributional_check(f,
        u -> sum(cell.(counts, logistic.(_cn_eta(u)))) + _cn_normal_prior(u), [0.3, -0.2])
end
