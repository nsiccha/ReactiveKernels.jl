module IntegerJAXTests
using ReactiveKernels, ReactiveKernelsPPL, Test
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

@testset "integer runtime state in emitted JAX" begin
    scalar = @rkppl begin
        p ~ Beta(2, 3)
        k ~ Bernoulli(p)
        y .~ Normal.(k, 1.0)
    end
    arrays = @rkppl begin
        theta[axes(x, 1)] .~ Normal.(0, 1)
        k[axes(x, 1)] .~ Bernoulli.(p)
        y .~ Normal.(theta .+ k, 1)
    end
    categorical = @rkppl begin
        theta ~ Dirichlet(alpha)
        k ~ Categorical(theta)
        y .~ Normal.(k,1)
    end
    uniform = @rkppl begin
        mu ~ Normal(0,1)
        k ~ DiscreteUniform(lo,hi)
        y .~ Normal.(mu+k,1)
    end
    binomial = @rkppl begin
        p ~ Beta(2,3)
        k ~ Binomial(n,p)
        y .~ Normal.(k,1)
    end
    rag = @rkppl begin
        p ~ Beta(a,b)
        c ~ Binomial(n,p)
        r = h-c
        r ~ Binomial(n-c,0.5)
    end
    matrix = @rkppl begin
        mu ~ Normal(0,1)
        k[axes(x,1),axes(x,2)] .~ Categorical(p .* 1)
        y .~ Normal.(mu .+ k,1)
    end
    # Sized by the observation, which the program may not read: the module
    # holds no binding size for `k` and checks its rank.
    observed = @rkppl begin
        mu ~ Normal(0, 1)
        k[axes(y, 1)] .~ Bernoulli(p)
        y .~ Normal.(mu .+ k, 1)
    end
    mktempdir() do dir
        for (name, plan) in (("scalar", scalar() | (; y=[1.2, 2.1])),
                ("arrays", arrays(; x=[0.1, 0.2], p=0.4) | (; y=[1.2, 2.1])),
                ("categorical",categorical(;alpha=[1.5,2.0,3.0]) | (;y=[1.2,2.1])),
                ("uniform",uniform(;lo=2,hi=7) | (;y=[1.2,2.1])),
                ("binomial",binomial(;n=5) | (;y=[1.2,2.1])),
                ("rag",bind_data(rag(;n=10,h=8,a=2.0,b=3.0))),
                ("matrix",matrix(;x=zeros(2,3),p=[0.2,0.3,0.5]) | (;y=fill(0.4,2,3))),
                ("observed", observed(; p=0.4) | (; y=[1.2, 2.1])))
            source = jax_source(build_kernel(plan), plan)
            @test occursin("def log_density", source.text)
            write(joinpath(dir, name * ".py"), source)
        end
        if rk_jax_available()
            driver = joinpath(@__DIR__, "integer_jax.py")
            @test success(pipeline(`$RK_JAX_PYTHON $driver $dir`; stdout, stderr))
        else
            @test_skip "RK_JAX_PYTHON not set"
        end
    end
end
end
