using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Test

function _csi_build(ast, data)
    plan = lower_rkppl(ast, data; conditioned=keys(data))
    bound = bind_data(plan, Dict{Symbol,Any}(pairs(data)))
    return (; bound, built=build_kernel(bound))
end
function _csi_findiff(f, u)
    h = cbrt(eps(Float64))
    return [(f(u+h*e)-f(u-h*e))/(2h)
        for e in eachcol(Matrix{Float64}(I, length(u), length(u)))]
end

const _CSI_PRIOR_ONLY = quote
    sigma ~ Exponential(1)
    c ~ Ordered(Normal(0, 1), 2)
    p ~ Dirichlet([2.0, 3.0])
end
function _csi_prior_oracle(layout, u)
    th = constrain(layout, u)
    return logpdf(Exponential(1), th.sigma) + sum(logpdf.(Normal(), th.c)) +
        logpdf(Dirichlet([2.0, 3.0]), th.p) + log(th.sigma) +
        log(th.c[2]-th.c[1]) + sum(log, th.p)
end

const _CSI_SHARED_MONOTONIC = quote
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    d ~ Normal(0, 1)
    s ~ Dirichlet([2.0, 3.0])
    m1 = cumsum(vcat(0.0, s))[c]
    m2 = cumsum(vcat(0.0, s))[c]
    mu = a .+ b .* m1
    nu = d .+ m2
    y .~ Normal.(mu, 1.0)
    z .~ Normal.(nu, 1.0)
end
_csi_mo_data(n) = (; c=[mod1(i, 3) for i in 1:n],
    y=[0.2sin(i) for i in 1:n], z=[0.3cos(i) for i in 1:n])
function _csi_mo_oracle(layout, u, data)
    th = constrain(layout, u)
    m = cumsum(vcat(0.0, th.s))[data.c]
    return logpdf(Normal(), th.a) + logpdf(Normal(), th.b) + logpdf(Normal(), th.d) +
        logpdf(Dirichlet([2.0, 3.0]), th.s) + sum(log, th.s) +
        sum(logpdf.(Normal.(th.a .+ th.b .* m, 1), data.y)) +
        sum(logpdf.(Normal.(th.d .+ m, 1), data.z))
end

function _csi_index_model(centered)
    steps = centered ? [:(h[t] ~ Normal(phi * h[t-1] + 0.1t, 1))] :
        [:(eps ~ Normal(0.1t, 1)), :(h[t] = phi * h[t-1] + t * eps + 0.1t)]
    return quote
        phi ~ Normal(0, 1)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                $(steps...)
            end
        end
        y .~ Normal.(h, 0.8)
    end
end
function _csi_index_oracle(layout, u, data, centered)
    th = constrain(layout, u)
    entry = only(e for e in layout.entries if e.kind === :scan)
    z = u[entry.offset:entry.offset+entry.size-1]
    lp = logpdf(Normal(), th.phi) + logpdf(Normal(), z[1])
    h = copy(z)
    for t in 2:length(z)
        if centered
            lp += logpdf(Normal(th.phi*h[t-1] + 0.1t, 1), h[t])
        else
            lp += logpdf(Normal(0.1t, 1), z[t])
            h[t] = th.phi*h[t-1] + t*z[t] + 0.1t
        end
    end
    return lp + sum(logpdf.(Normal.(h, 0.8), data.y))
end

@testset "standalone vector priors and shared declared simplex" begin
    fx = _csi_build(_CSI_PRIOR_ONLY, NamedTuple())
    @test isempty(fx.bound.responses)
    @test fx.built.layout.total == 4
    u = [0.2, -0.3, 0.1, 0.4]
    oracle(v) = _csi_prior_oracle(fx.built.layout, v)
    q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(q, similar(u), u)
    @test value ≈ oracle(u)
    @test gradient ≈ _csi_findiff(oracle, u) rtol=1e-5
    @test Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u) == NamedTuple()
    @test Base.invokelatest(prepare_query(fx.built, fx.bound, :likelihood), u) == 0

    data = _csi_mo_data(6)
    fx = _csi_build(_CSI_SHARED_MONOTONIC, data)
    @test length(fx.bound.vector_parameters) == 1
    u = [0.2, -0.3, 0.1, 0.4]
    oracle_mo(v) = _csi_mo_oracle(fx.built.layout, v, data)
    q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(q, similar(u), u)
    @test value ≈ oracle_mo(u)
    @test gradient ≈ _csi_findiff(oracle_mo, u) rtol=1e-5
end

@testset "scan indices are values in carry and innovation density" begin
    for centered in (true, false)
        data = (; y=[0.2sin(i) for i in 1:4])
        fx = _csi_build(_csi_index_model(centered), data)
        u = 0.05cos.(1:fx.built.layout.total)
        oracle(v) = _csi_index_oracle(fx.built.layout, v, data, centered)
        q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(q, similar(u), u)
        @test value ≈ oracle(u)
        @test gradient ≈ _csi_findiff(oracle, u) rtol=1e-5 atol=1e-7
    end
end
