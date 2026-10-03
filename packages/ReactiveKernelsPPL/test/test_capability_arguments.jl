using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Test
using ReactiveKernelsDistributionKernels.DistributionKernelSources: gamma

@kernel _ARG_GAMMA_LOG_ROUTE(u::Vector{Float64}) = begin
    log_rate::Float64 = sum(u)
    density::Float64 = gamma(; shape=2.0, log_rate=log_rate).logpdf(1.0)
end

const _ARG_MODELS = [
    (:fixed, quote k .~ Binomial.(n, 0.3) end, Float64[]),
    (:zib, quote
        zi ~ Beta(2, 3)
        k .~ ZeroInflatedBinomial.(n, 0.3, zi)
    end, [0.2]),
    (:weibull, quote
        a ~ Normal(0, 1)
        y .~ Weibull.(a, 1.5)
    end, [1.7]),
    (:gamma_scalar, quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        alpha ~ Exponential(1)
        alpha2 ~ Exponential(1)
        eta = a .+ b .* x
        y .~ Gamma.(alpha, exp.(eta) ./ alpha2)
    end, [0.2, 0.1, 0.3, -0.2]),
    (:gamma_mixed, quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        s = c .+ d .* z
        y .~ Gamma.(s, exp.(eta) ./ exp.(s))
    end, [0.2, 0.1, 1.4, 0.1]),
    (:gamma_other, quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        e ~ Normal(0, 1)
        f ~ Normal(0, 1)
        eta = a .+ b .* x
        s = c .+ d .* z
        t = e .+ f .* x
        y .~ Gamma.(exp.(s), exp.(eta) ./ exp.(t))
    end, [0.2, 0.1, 0.3, 0.1, -0.2, 0.15]),
    (:sqrt_scale, quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        mu = a .+ b .* x
        sigma = c .+ d .* z
        y .~ Normal.(mu, sqrt.(sigma))
    end, [0.2, 0.1, 1.4, 0.1]),
    (:shared_scale, quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, exp.(mu))
    end, [0.2, 0.1]),
    (:value_scales, quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        tau ~ Exponential(1)
        mu = a .+ b .* x
        y1 .~ Normal.(mu, exp.(1.5))
        y2 .~ Normal.(mu, exp.(tau))
        y3 .~ Normal.(mu, exp.(x))
    end, [0.2, 0.1, -0.3]),
    (:gamma_raw_scale, quote
        a ~ Normal(0, 1)
        y .~ Gamma.(2.0, a)
    end, [1.5]),
]

function _arg_data(n)
    y = [0.5+0.05i for i in 1:n]
    return (; y, y1=y, y2=y, y3=y, y4=y,
        x=collect(range(-0.3, 0.4; length=n)), z=collect(range(0.1, 0.6; length=n)),
        k=[mod(i, 3) for i in 1:n], n=fill(3, n))
end

@testset "Gamma log-rate owner retains its numerical range" begin
    kernel = prepare(_ARG_GAMMA_LOG_ROUTE; have=(:u,), want=:density)
    ad = prepare_ad(kernel, AutoEnzyme(; mode=Enzyme.Reverse), [0.4]; active=:u)
    for log_rate in (-1000.0, 0.4)
        u = [log_rate]
        @test Base.invokelatest(kernel, u) ≈ 2log_rate-exp(log_rate)
        value, grad = ad_value_and_gradient(ad, u)
        @test value ≈ 2log_rate-exp(log_rate)
        @test grad ≈ [2-exp(log_rate)]
    end
end
function _arg_build(ast, data)
    plan = lower_rkppl(ast, data; conditioned=keys(data))
    bound = bind_data(plan, Dict{Symbol,Any}(pairs(data)))
    return (; bound, built=build_kernel(bound))
end
function _arg_oracle(kind, layout, u, data)
    th = constrain(layout, u)
    if kind === :fixed
        return sum(logpdf.(Binomial.(data.n, 0.3), data.k))
    elseif kind === :zib
        zi = th.zi
        return logpdf(Beta(2, 3), zi)+log(zi)+log1p(-zi)+sum(
            y == 0 ? log(zi+(1-zi)*pdf(Binomial(n, 0.3), y)) :
                log1p(-zi)+logpdf(Binomial(n, 0.3), y)
            for (n, y) in zip(data.n, data.k))
    elseif kind === :weibull
        return logpdf(Normal(), th.a)+sum(logpdf.(Weibull(th.a, 1.5), data.y))
    elseif kind === :gamma_raw_scale
        return logpdf(Normal(), th.a)+sum(logpdf.(Gamma(2.0, th.a), data.y))
    end
    mu = th.a .+ th.b .* data.x
    if kind === :value_scales
        prior = logpdf(Normal(), th.a)+logpdf(Normal(), th.b)+
            logpdf(Exponential(1), th.tau)+log(th.tau)
        return prior+sum(logpdf.(Normal.(mu, exp(1.5)), data.y1))+
            sum(logpdf.(Normal.(mu, exp(th.tau)), data.y2))+
            sum(logpdf.(Normal.(mu, exp.(data.x)), data.y3))
    end
    prior = sum(logpdf(Normal(), getproperty(th, nm)) for nm in
        (kind === :shared_scale ? (:a, :b) : kind === :gamma_scalar ?
            (:a, :b) : kind === :gamma_other ? (:a, :b, :c, :d, :e, :f) :
            (:a, :b, :c, :d)))
    if kind === :gamma_scalar
        prior += logpdf(Exponential(1), th.alpha)+log(th.alpha)+
            logpdf(Exponential(1), th.alpha2)+log(th.alpha2)
        return prior+sum(logpdf.(Gamma.(th.alpha, exp.(mu)./th.alpha2), data.y))
    elseif kind === :shared_scale
        return prior+sum(logpdf.(Normal.(mu, exp.(mu)), data.y))
    end
    s = th.c .+ th.d .* data.z
    if kind === :sqrt_scale
        return prior+sum(logpdf.(Normal.(mu, sqrt.(s)), data.y))
    end
    alpha = kind === :gamma_mixed ? s : exp.(s)
    divisor = kind === :gamma_mixed ? exp.(s) : exp.(th.e .+ th.f .* data.x)
    return prior+sum(logpdf.(Gamma.(alpha, exp.(mu)./divisor), data.y))
end
function _arg_findiff(f, u)
    h = cbrt(eps(Float64))
    return [(f(u+h*e)-f(u-h*e))/(2h)
        for e in eachcol(Matrix{Float64}(I, length(u), length(u)))]
end

@testset "ordinary distribution argument values" begin
    data = _arg_data(4)
    for (kind, ast, u) in _ARG_MODELS
        @testset "$kind" begin
            fx = _arg_build(ast, data)
            q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
            oracle(v) = _arg_oracle(kind, fx.built.layout, v, data)
            @test Base.invokelatest(q.kernel, u) ≈ oracle(u)
            if !isempty(u)
                value, grad = sampler_value_and_gradient!(q, similar(u), u)
                @test value ≈ oracle(u)
                @test grad ≈ _arg_findiff(oracle, u) rtol=3e-6 atol=3e-6
            else
                @test fx.built.layout.total == 0
            end
        end
    end
    fx = _arg_build(_ARG_MODELS[3][2], data)
    q = prepare_sampler(fx.built, fx.bound, [-0.5]; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    @test Base.invokelatest(q.kernel, [-0.5]) == -Inf
    fx = _arg_build(_ARG_MODELS[5][2], data)
    u = unconstrain(fx.built.layout, (; a=0.2, b=0.1, c=-1.4, d=0.1))
    q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    @test Base.invokelatest(q.kernel, u) == -Inf
    fx = _arg_build(last(_ARG_MODELS)[2], data)
    u = [-0.5]
    q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(q, similar(u), u)
    @test value == -Inf
    @test gradient ≈ -u
end
