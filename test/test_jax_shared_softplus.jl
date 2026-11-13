module JAXSharedSoftplusTests
using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using LogExpFunctions: log1pexp, logistic
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
include("jax_python.jl")

@kernel total(x::Vector{Float64}) = begin
    values = plate(x) do xi
        rk_log1pexp(xi)
    end
    value::Float64 = sum(values)
    return value
end

@testset "shared softplus and Bernoulli through ordinary JAX reverse" begin
    src=jax_source(total;have=(:x,),want=:value)
    bern=jax_source(bernoulli.logpdf;have=(:logit,:observed),want=:logpdf)
    if rk_jax_available()
        for x in (Float64[],[-1000.0,-40.0,0.0,40.0,1000.0],sin.(collect(1:17)))
            got=rk_jax_evaluate(src,"kernel",(x,);grad=true)
            @test got.value ≈ sum(log1pexp,x;init=0.0)
            @test got.grad ≈ logistic.(x)
        end
        for x in (-1000.0,-40.0,0.0,40.0,1000.0), y in (false,true)
            got=rk_jax_evaluate(bern,"kernel",(x,y);grad=true)
            @test got.value ≈ -log1pexp(y ? -x : x)
            @test got.grad ≈ (y ? logistic(-x) : -logistic(x))
        end
    else
        @test_skip false
    end
end
end
