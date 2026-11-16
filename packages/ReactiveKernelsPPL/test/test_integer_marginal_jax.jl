module IntegerMarginalJAXTests
using ReactiveKernels, ReactiveKernelsPPL, Test
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

@testset "finite integer marginalization in emitted JAX" begin
    mixture = @rkppl begin
        p ~ Beta(2,3)
        k ~ Binomial(n,p)
        y .~ Normal.(k,1)
    end
    joint = @rkppl begin
        p ~ Beta(2,3)
        k ~ Bernoulli(p)
        z ~ Bernoulli(k)
        y ~ Normal(k+z,1)
    end
    rag = @rkppl begin
        p ~ Beta(a,b)
        c ~ Binomial(n,p)
        r = h-c
        r ~ Binomial(n-c,0.5)
    end
    mktempdir() do dir
        plans = (
            ("mixture",marginalize(mixture,:k)(;n=5) | (;y=[1.2,2.1])),
            ("joint",marginalize(joint,:k,:z)() | (;y=0.3)),
            ("rag",bind_data(marginalize(rag,:c)(;n=10,h=8,a=2.0,b=3.0))))
        for (name,plan) in plans
            @testset "$name marginal outputs" begin
                source = jax_source(build_kernel(plan),plan;
                    values=(:pointwise,:_ppl_latents))
                @test occursin("def log_density",source.text)
                write(joinpath(dir,name*".py"),source)
                if rk_jax_available()
                    driver = joinpath(@__DIR__,"integer_marginal_jax.py")
                    @test success(pipeline(`$RK_JAX_PYTHON $driver $dir $name`;stdout,stderr))
                else
                    @test_skip "RK_JAX_PYTHON not set"
                end
            end
        end
    end
end
end
