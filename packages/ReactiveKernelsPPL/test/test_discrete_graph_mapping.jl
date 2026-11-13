module DiscreteGraphMappingTests
using Test, ReactiveKernels, ReactiveKernelsPPL
import Distributions as D
import Enzyme, DifferentiationInterface as DI
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

function mapped(constructor, args; plated=false, endpoint=:logpdf)
    object = sampling_distribution_graph(constructor, args)
    @test object isa Expr
    density = Expr(:call, Expr(:., object, QuoteNode(endpoint)), :x)
    definition = if plated
        :(mapped_plate(q, xs) = begin
            cells = plate(xs, Ref(q)) do x, q
                $density
            end
            density = sum(cells)
            return density
        end)
    else
        :(mapped_density(q, x) = begin
            return $density
        end)
    end
    return ReactiveKernels._kernel_eval_definition(@__MODULE__, definition)
end

const CASES = (
    (D.Categorical, (:q,), ([0.2,0.0,0.8], [0.7,0.2,0.1]), q -> D.Categorical(q)),
    (D.Poisson, (:(q[1]),), ([0.0], [2.3]), q -> D.Poisson(q[1])),
    (D.Binomial, (5, :(q[1])), ([0.0], [0.3], [1.0]), q -> D.Binomial(5,q[1])),
    (D.Bernoulli, (:(q[1]),), ([0.0], [0.3], [1.0]), q -> D.Bernoulli(q[1])),
)

@testset "discrete constructor graphs: density, support and ownership" begin
    @test Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLDistributionsExt) !== nothing
    for (constructor, args, inputs, oracle) in CASES
        f = prepare(mapped(constructor,args))
        for q in inputs
            saved = copy(q)
            for x in (-Inf,-2.0,-1.0,0.0,0.2,1.0,2.0,3.0,5.0,6.0,1e20,Inf,NaN)
                @test isequal(f(q,x), D.logpdf(oracle(q),x)) ||
                    isapprox(f(q,x), D.logpdf(oracle(q),x); rtol=2e-14,atol=2e-14)
            end
            for x in (-1,0,1,2,5,6,false,true)
                @test f(q,x) ≈ D.logpdf(oracle(q),x)
            end
            @test q == saved
            @test (@inferred f(q,1)) isa Float64
        end
    end
    for (constructor,args,oracle) in ((D.Poisson,(),D.Poisson()),
            (D.Bernoulli,(),D.Bernoulli()), (D.Binomial,(),D.Binomial()),
            (D.Binomial,(3,),D.Binomial(3)), (D.Categorical,(3,),D.Categorical(3)))
        f = prepare(mapped(constructor,args))
        for x in (-1,0,1,2,3,4)
            @test f(nothing,x) ≈ D.logpdf(oracle,x)
        end
    end
    @test sampling_distribution_graph(D.Poisson,(:a,:b)) === nothing
    @test sampling_distribution_graph(D.Binomial,(:a,:b,:c)) === nothing
    @test sampling_distribution_graph(D.Bernoulli,(:a,:b)) === nothing
    @test sampling_distribution_graph(D.Categorical,()) === nothing
    for constructor in (D.Poisson,D.Binomial,D.Bernoulli)
        args = constructor === D.Binomial ? (3,:(q[1])) : (:(q[1]),)
        f = prepare(mapped(constructor,args))
        for q in ([-0.1],[NaN])
            @test_throws DomainError f(q,1)
            @test_throws DomainError f(q,-1)
        end
    end
    f = prepare(mapped(D.Categorical,(:q,)))
    for q in (Float64[],[-0.1,1.1],[0.2,0.2],[NaN,1.0])
        @test_throws DomainError f(q,1)
    end
    @test_throws DomainError prepare(mapped(D.Binomial,(-1,0.5)))(nothing,0)
end

@testset "probability tolerance and shared Poisson observation port" begin
    for T in (Float32, Float64), scale in (0, 0.5, 2.0, -0.5, -2.0)
        q = T[0.5, 0.5 + scale * sqrt(eps(T))]
        f = prepare(mapped(D.Categorical, (:q,)))
        valid = all(>=(0),q) && isapprox(sum(q),one(T))
        if valid
            @test f(q,1) == Float64(D.logpdf(D.Categorical(q),1))
        else
            @test_throws DomainError f(q,1)
        end
    end
    for endpoint in (:cdf,:ccdf)
        f = prepare(mapped(D.Poisson, (:(q[1]),); endpoint))
        oracle = getproperty(D,endpoint)
        for q in ([0.0],[2.3]), x in (-Inf,-1.0,0.0,0.2,1.0,1.8,Inf,NaN)
            @test isapprox(f(q,x),oracle(D.Poisson(q[1]),x);nans=true)
        end
    end
end

@testset "discrete graph plates: native reverse and runtime extents" begin
    for (constructor,args,_,oracle) in CASES
        q = constructor === D.Categorical ? [0.2,0.3,0.5] : [0.3]
        spec = mapped(constructor,args;plated=true)
        for n in (0,1,9)
            xs = constructor === D.Categorical ? [1 + mod(i,3) for i in 1:n] :
                constructor === D.Bernoulli ? [mod(i,2) for i in 1:n] :
                [mod(i,5) for i in 1:n]
            f = prepare(spec;bound=(;xs))
            saved = copy(q)
            value, gradient = DI.value_and_gradient(f, DI.AutoEnzyme(;mode=Enzyme.Reverse),q)
            @test value ≈ sum(x -> D.logpdf(oracle(q),x),xs;init=0.0)
            expected = if constructor === D.Categorical
                [count(==(i),xs)/q[i] for i in eachindex(q)]
            elseif constructor === D.Poisson
                [sum(xs;init=0)/q[1] - n]
            else
                trials = constructor === D.Bernoulli ? 1 : 5
                [sum(xs;init=0)/q[1] - (trials*n-sum(xs;init=0))/(1-q[1])]
            end
            @test gradient ≈ expected
            @test q == saved
            @test f(q) == value
        end
    end
end


@testset "discrete mapped graphs emit and execute JAX" begin
    for (constructor,args,_,oracle) in CASES
        q = constructor === D.Categorical ? [0.2,0.3,0.5] : [0.3]
        spec = mapped(constructor,args;plated=true)
        for T in (Int,Float64)
            src = jax_source(spec; want=:density, name=:log_density,
                types=(q=typeof(q),xs=Vector{T}))
            cases = [T[],T[1],T[1,0,1,1,0]]
            constructor === D.Categorical && (cases = [xs .+ 1 for xs in cases])
            T === Float64 && push!(cases, [1.0,0.2,-1.0,Inf,NaN])
            for xs in cases
                if rk_jax_available()
                    got = rk_jax_evaluate(src,"log_density",(q,xs);grad=true)
                    native = prepare(spec;bound=(;xs))
                    @test got.value ≈ native(q)
                    @test got.grad ≈ DI.gradient(native,DI.AutoEnzyme(;mode=Enzyme.Reverse),q)
                else
                    @test_skip "RK_JAX_PYTHON is not configured"
                end
            end
        end
    end
end

@testset "lazy discrete boundaries retain ordinary derivatives" begin
    for (constructor,args,q,xs,expected) in (
            (D.Categorical,(:q,),[0.0,0.4,0.6],[2,3],[0.0,2.5,1/0.6]),
            (D.Poisson,(:(q[1]),),[0.0],[0],[-1.0]),
            (D.Binomial,(5,:(q[1])),[0.0],[0],[-5.0]),
            (D.Binomial,(5,:(q[1])),[1.0],[5],[5.0]),
            (D.Bernoulli,(:(q[1]),),[0.0],[0],[-1.0]),
            (D.Bernoulli,(:(q[1]),),[1.0],[1],[1.0]))
        spec = mapped(constructor,args;plated=true)
        f = prepare(spec;bound=(;xs))
        @test DI.gradient(f,DI.AutoEnzyme(;mode=Enzyme.Reverse),q) ≈ expected
        if rk_jax_available()
            src = jax_source(spec;want=:density,name=:log_density,
                types=(q=typeof(q),xs=Vector{Int}))
            got = rk_jax_evaluate(src,"log_density",(q,xs);grad=true)
            @test got.value ≈ f(q)
            @test got.grad ≈ expected
        end
    end
end
end
