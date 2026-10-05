using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test

module CapabilityGenerativeStreams
using ReactiveKernelsPPL
@rkppl vector_stream(x, b) = begin
    a ~ Normal(0, 1)
    s ~ Exponential(1)
    mu = a .+ b .* x
    slot .~ Normal.(mu, s)
    slot
end

@rkppl scalar_stream() = begin
    a ~ Normal(0, 1)
    slot .~ Normal.(a, 0.8)
    slot
end
@rkppl cell_stream(a, s) = begin
    b ~ Normal(0, 1)
    mu = a + b
    slot .~ Normal.(mu, s)
    slot
end
end

function _cap_empty_conditioned_prior()
    data = Dict(:y => Float64[], :z => Float64[])
    plan = lower_rkppl(quote
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            b[i] ~ Normal(a, 1)
            z[i] ~ Normal(a + b[i], 0.8)
        end
    end, data; conditioned=(:z,))
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; a=0.2, b=Float64[]))
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    (; data, bound, built, u, sampler)
end

@testset "empty conditioned cell priors skip indexed latent arguments" begin
    fx = _cap_empty_conditioned_prior()
    value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
    @test value ≈ logpdf(Normal(), only(fx.u))
    @test gradient ≈ -fx.u
    @test Base.invokelatest(prepare_query(fx.built, fx.bound, :likelihood), fx.u) == 0.0
    @test Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u) == (; z=Float64[])
end

function _cap_stream_fixture(kind, n)
    data = Dict(:x => collect(range(-0.5, 0.7; length=n)),
        :y => collect(range(-0.3, 0.4; length=n)))
    ast = kind === :vector ? quote
        b ~ Normal(0, 1)
        z ~ vector_stream(x, b)
        y .~ Normal.(z, 0.7)
    end : kind === :scalar ? quote
        z ~ scalar_stream()
        y .~ Normal.(z, 0.7)
    end : quote
        a ~ Normal(0, 1)
        s ~ Exponential(1)
    end
    if kind in (:cell, :observed)
        cell = kind === :cell ? quote
            z[i] ~ cell_stream(a, s)
            y[i] ~ Normal(z[i], 0.7)
        end : quote
            y[i] ~ cell_stream(a, s)
        end
        push!(ast.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, :(i = eachindex(y)), cell)))
    end
    plan = lower_rkppl(ast, data; mod=CapabilityGenerativeStreams, conditioned=keys(data))
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    a, s = 0.2, 1.1
    b = collect(range(-0.1, 0.2; length=n))
    z = collect(range(-0.2, 0.3; length=n))
    values = kind === :vector ? (; b=0.35, z=(; a, s, slot=z)) :
        kind === :scalar ? (; z=(; a, slot=0.1)) :
        kind === :cell ? (; a, s, z=(; b, slot=z)) : (; a, s, y=(; b))
    u = unconstrain(built.layout, values)
    function pointwise(v)
        q = constrain(built.layout, v)
        mu = kind === :observed ? q.a .+ q.y.b : q.z.slot
        sigma = kind === :observed ? q.s : 0.7
        logpdf.(Normal.(mu, sigma), data[:y])
    end
    function oracle(v)
        q = constrain(built.layout, v)
        prior = if kind === :vector
            logpdf(Normal(), q.b) + logpdf(Normal(), q.z.a) +
                logpdf(Exponential(), q.z.s) + log(q.z.s) +
                sum(logpdf.(Normal.(q.z.a .+ q.b .* data[:x], q.z.s), q.z.slot); init=0.0)
        elseif kind === :scalar
            logpdf(Normal(), q.z.a) + logpdf(Normal(q.z.a, 0.8), q.z.slot)
        elseif kind === :cell
            logpdf(Normal(), q.a) + logpdf(Exponential(), q.s) + log(q.s) +
                sum(logpdf.(Normal(), q.z.b); init=0.0) +
                sum(logpdf.(Normal.(q.a .+ q.z.b, q.s), q.z.slot); init=0.0)
        else
            logpdf(Normal(), q.a) + logpdf(Exponential(), q.s) + log(q.s) +
                sum(logpdf.(Normal(), q.y.b); init=0.0)
        end
        prior + sum(pointwise(v); init=0.0)
    end
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    return (; kind, ast, data, plan, bound, built, u, sampler, oracle, pointwise)
end

function _cap_stream_fd(f, u)
    h = cbrt(eps(Float64))
    [(f(u+h*e)-f(u-h*e))/(2h) for e in eachcol(Matrix{Float64}(I, length(u), length(u)))]
end

@testset "stream submodels generate latent scalar and cell values" begin
    for kind in (:vector, :scalar, :cell, :observed)
        fx = _cap_stream_fixture(kind, 4)
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _cap_stream_fd(fx.oracle, fx.u) rtol=6e-6
        @test Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u).y ≈ fx.pointwise(fx.u)
        if kind === :observed
            @test length(constrain(fx.built.layout, fx.u).y.b) == 4
        else
            slot = constrain(fx.built.layout, fx.u).z.slot
            @test kind === :scalar ? slot isa Float64 : length(slot) == 4
        end
    end
end

@testset "empty generative stream plates retain their scalar priors" begin
    for kind in (:vector, :scalar, :cell, :observed)
        fx = _cap_stream_fixture(kind, 0)
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _cap_stream_fd(fx.oracle, fx.u) rtol=6e-6
        @test Base.invokelatest(prepare_query(fx.built, fx.bound, :likelihood), fx.u) == 0
        @test isempty(Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u).y)
    end
end
