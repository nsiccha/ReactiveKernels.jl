using DifferentiationInterface
using Distributions
using Enzyme
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

function _cs_build(ast, data)
    plan = lower_rkppl(ast, data; conditioned=keys(data))
    bound = bind_data(plan, Dict{Symbol,Any}(pairs(data)))
    built = build_kernel(bound)
    return (; bound, built)
end

function _cs_findiff(f, u)
    h = cbrt(eps(Float64))
    return [(f(u + h * e) - f(u - h * e)) / (2h)
        for e in eachcol(Matrix{Float64}(I, length(u), length(u)))]
end

const _CS_POINTWISE = quote
    b ~ Normal(0, 1)
    eta_z = b .* x
    eta_k = b .* x
    eta_y = b .* x
    z .~ Bernoulli.(logistic.(eta_z))
    k .~ Poisson.(exp.(eta_k))
    y .~ Normal.(eta_y, 1.2)
end
_cs_pointwise_data(n) = (; x=[0.3sin(i) for i in 1:n],
    z=[isodd(i) for i in 1:n], k=[mod(i, 4) for i in 1:n],
    y=[0.2cos(i) for i in 1:n])
function _cs_pointwise_oracle(u, data)
    eta = u[1] .* data.x
    return (; z=logpdf.(Bernoulli.(1 ./ (1 .+ exp.(-eta))), data.z),
        k=logpdf.(Poisson.(exp.(eta)), data.k),
        y=logpdf.(Normal.(eta, 1.2), data.y))
end

@testset "pointwise likelihood retains observation names and shapes" begin
    data = _cs_pointwise_data(6)
    fx = _cs_build(_CS_POINTWISE, data)
    u = [0.4]
    kern = prepare_query(fx.built, fx.bound, :pointwise)
    values = Base.invokelatest(kern, u)
    expected = _cs_pointwise_oracle(u, data)
    @test keys(values) == (:z, :k, :y)
    @test all(size(values[n]) == size(data[n]) for n in keys(values))
    @test all(values[n] ≈ expected[n] for n in keys(values))
    lik = prepare_query(fx.built, fx.bound, :likelihood)
    @test sum(sum, values) ≈ Base.invokelatest(lik, u)
    q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    oracle(v) = logpdf(Normal(), v[1]) + sum(sum, _cs_pointwise_oracle(v, data))
    value, gradient = sampler_value_and_gradient!(q, similar(u), u)
    @test value ≈ oracle(u)
    @test gradient ≈ _cs_findiff(oracle, u) rtol=1e-5

    # A scalar declaration and elementwise matrix declarations become
    # likelihood terms when conditioned, keeping scalar and matrix shape.
    ast = quote
        b ~ Normal(0, 1)
        s ~ Exponential(1)
        Y[1:2, 1:3] .~ Normal.(b, s)
    end
    data = (; s=1.2, Y=reshape(collect(-0.3:0.1:0.2), 2, 3))
    fx = _cs_build(ast, data)
    values = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u)
    @test values.s ≈ logpdf(Exponential(1), data.s)
    @test size(values.Y) == size(data.Y)
    @test values.Y ≈ logpdf.(Normal(u[1], data.s), data.Y)
    lik = prepare_query(fx.built, fx.bound, :likelihood)
    @test values.s + sum(values.Y) ≈ Base.invokelatest(lik, u)

    fx = _cs_build(quote b ~ Normal(0, 1) end, NamedTuple())
    @test Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u) == NamedTuple()
end

@testset "pointwise joint slice draws and stopping stages" begin
    data = (; Z=[0.2 0.8; 0.4 0.6; 0.5 0.5])
    fx = _cs_build(quote
        b ~ Normal(0, 1)
        eachrow(Z[1:3, 1:2]) .~ Dirichlet([2.0, 3.0])
    end, data)
    u = [0.3]
    values = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u)
    @test values.Z ≈ [logpdf(Dirichlet([2.0, 3.0]), row) for row in eachrow(data.Z)]
    @test sum(values.Z) ≈ Base.invokelatest(prepare_query(fx.built, fx.bound, :likelihood), u)

    # A broadcast of joint normal draws has one scalar per row.
    data = (; data..., F=Matrix{Float64}(I, 2, 2))
    fx = _cs_build(quote
        b ~ Normal(0, 1)
        eachrow(Z[1:3, 1:2]) .~ MvNormal(zeros(2), F)
    end, data)
    values = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u)
    @test values.Z ≈ [logpdf(MvNormal(zeros(2), I), row) for row in eachrow(data.Z)]
    @test sum(values.Z) ≈ Base.invokelatest(prepare_query(fx.built, fx.bound, :likelihood), u)

    ast = quote
        b ~ Normal(0, 1)
        c[1:2] .~ Normal.(0, 1)
        y .~ Ordinal.(StoppingRatio(), LogitLink(), b .* x, Ref(c))
    end
    data = (; x=[0.1, -0.2, 0.3, 0.4], y=[1, 3, 2, 3])
    fx = _cs_build(ast, data)
    u = [0.3, -0.4, 0.2]
    th = constrain(fx.built.layout, u)
    values = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u)
    expected = [sum(j < data.y[i] ? -log1p(exp(th.c[j] - th.b*data.x[i])) :
        -log1p(exp(th.b*data.x[i] - th.c[j])) for j in 1:min(data.y[i], 2))
        for i in eachindex(data.y)]
    @test size(values.y) == size(data.y)
    @test values.y ≈ expected
    @test sum(values.y) ≈ Base.invokelatest(prepare_query(fx.built, fx.bound, :likelihood), u)
end

const _CS_SHARED_THRESHOLDS = quote
    b ~ Normal(0, 1)
    c ~ Ordered(Normal(0, 1), 2)
    y1 .~ OrderedLogistic.(b .* x, Ref(c))
    y2 .~ OrderedLogistic.(b .* x, Ref(c))
end
_cs_ordinal_data(n) = (; x=[sin(i) for i in 1:n],
    y1=[mod1(i, 3) for i in 1:n], y2=[mod1(2i, 3) for i in 1:n])
function _cs_ordinal_oracle(layout, u, data)
    q = constrain(layout, u)
    lp = logpdf(Normal(), q.b) + sum(logpdf.(Normal(), q.c))
    # c = [u1, u1 + exp(u2)], with Jacobian exp(u2).
    lp += u[end]
    for y in (data.y1, data.y2), i in eachindex(y)
        eta = q.b * data.x[i]
        hi = y[i] == 3 ? 1.0 : cdf(Logistic(), q.c[y[i]] - eta)
        lo = y[i] == 1 ? 0.0 : cdf(Logistic(), q.c[y[i]-1] - eta)
        lp += log(hi-lo)
    end
    return lp
end

@testset "shared thresholds count their prior once" begin
    data = _cs_ordinal_data(6)
    original = deepcopy(data)
    fx = _cs_build(_CS_SHARED_THRESHOLDS, data)
    @test fx.built.layout.total == 3
    @test length(fx.bound.vector_parameters) == 1
    @test all(r -> r.thresholds === :c, fx.bound.responses)
    u = [0.3, -0.4, 0.2]
    oracle(v) = _cs_ordinal_oracle(fx.built.layout, v, data)
    q = prepare_sampler(fx.built, fx.bound, u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(q, similar(u), u)
    @test value ≈ oracle(u)
    @test grad ≈ _cs_findiff(oracle,u) rtol=1e-5 atol=1e-7
    @test data == original
end
