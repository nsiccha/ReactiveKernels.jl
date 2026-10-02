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
