using DifferentiationInterface
using Enzyme
using LogExpFunctions: log1pexp, logaddexp, logit
using ReactiveKernels
using ReactiveKernelsPPL
using Test

_upt_logsigmoid(x) = -log1pexp(-x)
_upt_logcomplement(x) = -log1pexp(x)
_upt_jacobian(x) = _upt_logsigmoid(x) + _upt_logcomplement(x)

function _upt_model(ast, data)
    values = Dict{Symbol,Any}(pairs(data))
    plan = bind_data(lower_rkppl(ast, values; conditioned = (:y,)), values)
    built = build_kernel(plan)
    return (; plan, built, kernel = prepare_query(built, plan, :sampler))
end

function _upt_value_gradient(model, u)
    ad = Base.invokelatest(prepare_ad, model.kernel,
        AutoEnzyme(; mode = Enzyme.Reverse), u; active = :unconstrained)
    return Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
        ad, similar(u), u)
end

@testset "unit parameter logit and complement retain finite tails" begin
    # A sampled probability's inverse link must read its source coordinate,
    # even after the constrained Float64 value rounds to one.
    bin = _upt_model(quote
        p ~ Uniform(0, 1)
        eta = logit(p) .+ x
        y .~ Binomial.(3, logistic.(eta))
    end, (; x = [0.0], y = [1]))
    for z in (-40.0, 40.0)
        u = [z]
        value, gradient = _upt_value_gradient(bin, u)
        expected = log(3.0) + _upt_logsigmoid(z) +
            2 * _upt_logcomplement(z) + _upt_jacobian(z)
        expected_gradient = z < 0 ? 2.0 : -3.0
        @test isfinite(value)
        @test logjac(bin.built.layout, u) ≈ _upt_jacobian(z) rtol = 2e-12
        @test value ≈ expected rtol = 2e-12
        @test gradient[1] ≈ expected_gradient atol = 1e-12
    end

    # A non-unit interval has the same source-coordinate Jacobian, plus its
    # width. Its constrained value also rounds to a bound at the positive tail.
    interval = _upt_model(quote
        q ~ Uniform(2, 5)
        y .~ Normal.(q, 1)
    end, (; y = [3.0]))
    for z in (-40.0, 40.0)
        u = [z]
        value, gradient = _upt_value_gradient(interval, u)
        @test isfinite(value)
        @test all(isfinite, gradient)
        @test logjac(interval.built.layout, u) ≈ log(3.0) +
            _upt_jacobian(z) rtol = 2e-12
    end

    # The zero-inflation argument is the complement of a sampled unit
    # probability. Positive and zero counts exercise opposite log tails.
    zib = _upt_model(quote
        q ~ Uniform(0, 1)
        y .~ ZeroInflatedBinomial.(3, logistic.(x), 1 - q)
    end, (; x = [0.0], y = [1]))
    value, gradient = _upt_value_gradient(zib, [-40.0])
    @test isfinite(value)
    @test value ≈ _upt_logsigmoid(-40.0) + log(3 / 8) +
        _upt_jacobian(-40.0) rtol = 2e-12
    @test gradient[1] ≈ 2.0 atol = 1e-12

    zib_zero = _upt_model(quote
        q ~ Uniform(0, 1)
        y .~ ZeroInflatedBinomial.(3, logistic.(x), 1 - q)
    end, (; x = [1000.0], y = [0]))
    value, gradient = _upt_value_gradient(zib_zero, [40.0])
    @test isfinite(value)
    @test value ≈ logaddexp(_upt_logcomplement(40.0),
        _upt_logsigmoid(40.0) - 3 * log1pexp(1000.0)) +
        _upt_jacobian(40.0) rtol = 2e-12
    @test gradient[1] ≈ -2.0 atol = 1e-12
end
