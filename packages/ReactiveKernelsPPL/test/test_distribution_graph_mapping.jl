module DistributionGraphMappingTests
using Test, ReactiveKernels, ReactiveKernelsPPL
import Distributions as D
import Enzyme, DifferentiationInterface as DI

const mapped_normal = D.Normal
const mapped_lognormal = D.LogNormal

function endpoints(constructor, arguments)
    object = sampling_distribution_graph(constructor, arguments)
    @test object isa Expr
    call(endpoint, value=:x) = Expr(:call, Expr(:., object, QuoteNode(endpoint)), value)
    name = gensym(:family)
    Core.eval(@__MODULE__, :(@kernel $name(x::Float64, q::Vector{Float64}) = begin
        density::Float64 = $(call(:logpdf))
        lower::Float64 = $(call(:logcdf))
        upper::Float64 = $(call(:logccdf))
        normalizer::Float64 = $(call(:logccdf, :(q[3])))
        restricted::Float64 = x >= q[3] ? density : -Inf
        normalized::Float64 = restricted - normalizer
        return density
    end))
    getglobal(@__MODULE__,name)
end

@testset "resolved distribution graphs preserve base-family arguments" begin
    for constructor in (D.Normal,D.LogNormal,D.Gamma,D.Exponential,D.Cauchy,D.Laplace,D.Logistic)
        maxargs = constructor === D.Exponential ? 1 : 2
        for count in 0:maxargs
            args = Tuple(:(q[$i]) for i in 1:count)
            spec = endpoints(constructor,args)
            q = constructor in (D.Normal,D.LogNormal,D.Cauchy,D.Laplace,D.Logistic) ? [0.2,0.7] : [1.3,0.7]
            law = constructor(q[1:count]...)
            joint = prepare(spec;want=(:density,:lower,:upper))
            for x in (-Inf,-1.0,-0.0,0.0,0.2,0.7,2.0,Inf,NaN)
                got = Base.invokelatest(joint,x,q)
                expected = (D.logpdf(law,x),D.logcdf(law,x),D.logccdf(law,x))
                @test all(isapprox(a,b;rtol=5e-12,atol=1e-14,nans=true) for (a,b) in zip(got,expected))
            end
            count == 0 && continue
            for (output, reference) in ((:density,D.logpdf),(:lower,D.logcdf),(:upper,D.logccdf))
                f = prepare(spec;want=output,bound=(x=0.7,))
                g = Base.invokelatest(DI.gradient,f,DI.AutoEnzyme(;mode=Enzyme.Reverse),q)
                oracle(w)=reference(constructor(w[1:count]...),0.7)
                h=1e-5
                fd=[(oracle(q .+ (1:2 .== i).*h)-oracle(q .- (1:2 .== i).*h))/(2h) for i in 1:2]
                @test g ≈ fd rtol=2e-8 atol=1e-9
            end
        end
    end
    @test sampling_distribution_graph(identity,(:x,)) === nothing
    @test sampling_distribution_graph(D.Normal,(:a,:b,:c)) === nothing
end

const added_families = (
    (D.TDist, 1:1, [2.7,0.7]),
    (D.Gumbel, 0:2, [0.2,0.7]),
    (D.Weibull, 0:2, [1.3,0.7]),
    (D.InverseGamma, 0:2, [1.3,0.7]),
    (D.InverseGaussian, 0:2, [1.3,0.7]),
    (D.Rayleigh, 0:1, [1.3,0.7]),
    (D.Chi, 1:1, [2.7,0.7]),
    (D.Chisq, 1:1, [2.7,0.7]),
    (D.FDist, 2:2, [2.7,4.1]),
    (D.BetaPrime, 0:2, [1.3,0.7]),
)

@testset "BRM admitted families preserve constructor arities and defaults" begin
    for (constructor, arities, q) in added_families
        for count in 0:3
            args = Tuple(:(q[$i]) for i in 1:count)
            if !(count in arities)
                @test sampling_distribution_graph(constructor, args) === nothing
                continue
            end
            spec = endpoints(constructor, args)
            law = constructor(q[1:count]...)
            joint = prepare(spec; want=(:density,:lower,:upper))
            for x in (0.03,0.7,1.8,10.0)
                got = Base.invokelatest(joint,x,q)
                expected = (D.logpdf(law,x),D.logcdf(law,x),D.logccdf(law,x))
                @test all(isapprox(a,b;rtol=2e-10,atol=1e-12) for (a,b) in zip(got,expected))
            end
            count == 0 && continue
            positive_positions = constructor === D.Gumbel ? (count == 2 ? (2,) : ()) : 1:count
            for i in positive_positions, invalid in (0.0,-1.0,NaN)
                bad = copy(q)
                bad[i] = invalid
                if constructor === D.Weibull && count == 1
                    # Unlike Weibull(shape, scale), Weibull(shape) constructs
                    # without checking shape; its endpoints retain that behavior.
                    unchecked = constructor(bad[1])
                    @test unchecked isa D.Weibull
                    if invalid < 0
                        @test_throws DomainError D.logpdf(unchecked,0.7)
                        @test_throws DomainError Base.invokelatest(joint,0.7,bad)
                    else
                        expected = (D.logpdf(unchecked,0.7),D.logcdf(unchecked,0.7),D.logccdf(unchecked,0.7))
                        got = Base.invokelatest(joint,0.7,bad)
                        @test all(isapprox(a,b;nans=true) for (a,b) in zip(got,expected))
                    end
                    continue
                end
                @test_throws DomainError constructor(bad[1:count]...)
                @test_throws DomainError Base.invokelatest(joint,0.7,bad)
            end
            for (output,reference) in ((:density,D.logpdf),(:lower,D.logcdf),(:upper,D.logccdf))
                f = prepare(spec;want=output,bound=(x=0.7,))
                g = Base.invokelatest(DI.gradient,f,DI.AutoEnzyme(;mode=Enzyme.Reverse),q)
                oracle(w) = reference(constructor(w[1:count]...),0.7)
                h = 1e-5
                fd = [(oracle(q .+ (1:2 .== i).*h)-oracle(q .- (1:2 .== i).*h))/(2h) for i in 1:2]
                @test g ≈ fd rtol=3e-6 atol=2e-8
            end
        end
    end
end

@testset "restricted and normalized lower tails remain different graphs" begin
    for (constructor, arities, parameters) in added_families
        count = last(arities)
        spec = endpoints(constructor, Tuple(:(q[$i]) for i in 1:count))
        joint = prepare(spec; want=(:restricted,:normalized))
        for lower in (0.0,0.4,5.0)
            q = [parameters;lower]
            law = constructor(q[1:count]...)
            for x in (lower-0.1,lower,lower+0.2)
                # The inverse-Gamma library evaluates log(x) outside support;
                # the owned graph instead retains its existing support guard.
                density = x < lower ? -Inf : x == 0 && constructor === D.InverseGamma ?
                    -Inf : D.logpdf(law,x)
                expected = (density,density-D.logccdf(law,lower))
                got = Base.invokelatest(joint,x,q)
                @test all(isapprox(a,b;rtol=3e-10,atol=2e-11) for (a,b) in zip(got,expected))
            end
        end
        q = [parameters;0.4]
        f = prepare(spec; want=:normalized,bound=(x=1.0,))
        g = Base.invokelatest(DI.gradient,f,DI.AutoEnzyme(;mode=Enzyme.Reverse),q)
        oracle(w) = D.logpdf(constructor(w[1:count]...),1.0) - D.logccdf(constructor(w[1:count]...),w[3])
        h=1e-5
        fd=[(oracle(q .+ (1:3 .== i).*h)-oracle(q .- (1:3 .== i).*h))/(2h) for i in 1:3]
        @test g ≈ fd rtol=5e-6 atol=1e-7
    end
end

@testset "new qualified family graphs replay in observation plates" begin
    for (constructor, arities, q) in added_families, n in (1,7)
        family_name = nameof(constructor)
        head = Expr(:., :D, QuoteNode(family_name))
        count = last(arities)
        args = count == 1 ? [:(exp(a))] :
            constructor === D.Gumbel ? [:a, :(exp(b))] : [:(exp(a)), :(exp(b))]
        observed = [0.3 + 0.2i for i in 1:n]
        data = (; y=observed)
        saved = deepcopy(data)
        ast = quote
            a ~ Normal(0,1)
            b ~ Normal(0,1)
            y .~ $head.($(args...))
        end
        plan = bind_data(lower_rkppl(ast,data;mod=@__MODULE__,conditioned=(:y,)),data)
        built = build_kernel(plan)
        source = kernel_expr(plan,built.layout)
        @test !occursin("sampling_logdensity",string(source))
        replayed = ReactiveKernelsPPL._eval_kernel_def(source)
        replay = prepare_query((;spec=replayed,layout=built.layout),plan,:sampler)
        position = [0.2,-0.3]
        sampler = prepare_sampler(built,plan,position;backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
        value,g = sampler_value_and_gradient!(sampler,similar(position),position)
        law(w) = count == 1 ? constructor(exp(w[1])) :
            constructor === D.Gumbel ? constructor(w[1],exp(w[2])) :
            constructor(exp(w[1]),exp(w[2]))
        oracle(w) = sum(D.logpdf.(law(w),observed)) + sum(D.logpdf.(D.Normal(),w))
        @test value ≈ oracle(position)
        @test Base.invokelatest(replay,position) ≈ oracle(position)
        h=1e-5
        fd=[(oracle(position .+ (1:2 .== i).*h)-oracle(position .- (1:2 .== i).*h))/(2h) for i in 1:2]
        @test g ≈ fd rtol=3e-6 atol=2e-8
        @test data == saved
        @test position == [0.2,-0.3]
    end
end

@testset "zero-scale Normal and LogNormal graph semantics" begin
    for constructor in (D.Normal,D.LogNormal)
        spec = endpoints(constructor, (:(q[1]), :(q[2])))
        f = prepare(spec;want=(:density,:lower,:upper))
        q = [0.0,0.0]
        law = constructor(q...)
        for x in (-Inf,-1.0,0.0,0.5,1.0,2.0,Inf,NaN)
            actual = Base.invokelatest(f,x,q)
            expected = (D.logpdf(law,x),D.logcdf(law,x),D.logccdf(law,x))
            @test all(isapprox(a,b;nans=true) for (a,b) in zip(actual,expected))
        end
    end
end

@testset "qualified and aliased distribution graphs replay inside plates" begin
    for (head,family) in ((:mapped_normal,D.Normal), (:(D.Normal),D.Normal),
                          (:mapped_lognormal,D.LogNormal), (:(D.LogNormal),D.LogNormal)),
        count in (1,8)
        observations = collect(range(-0.4,0.8;length=count+1))[1:count]
        data = (; y = family === D.LogNormal ? exp.(observations) : observations)
        saved = deepcopy(data)
        ast = quote
            mu ~ Normal(0,1)
            log_scale ~ Normal(0,1)
            y .~ $head.(mu,exp(log_scale))
        end
        plan = bind_data(lower_rkppl(ast,data;mod=@__MODULE__,conditioned=(:y,)),data)
        built = build_kernel(plan)
        source = kernel_expr(plan,built.layout)
        @test !occursin("sampling_logdensity",string(source))
        replayed = ReactiveKernelsPPL._eval_kernel_def(source)
        replay = prepare_query((;spec=replayed,layout=built.layout),plan,:sampler)
        q = [0.2,-0.3]
        sampler = prepare_sampler(built,plan,q;backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
        value,g = sampler_value_and_gradient!(sampler,similar(q),q)
        oracle(w)=sum(D.logpdf.(family(w[1],exp(w[2])),data.y))+sum(D.logpdf.(D.Normal(),w))
        @test value ≈ oracle(q)
        @test Base.invokelatest(replay,q) ≈ oracle(q)
        h=1e-5
        fd=[(oracle(q .+ (1:2 .== i).*h)-oracle(q .- (1:2 .== i).*h))/(2h) for i in 1:2]
        @test g ≈ fd rtol=2e-8 atol=1e-9
        @test data == saved
        @test q == [0.2,-0.3]
    end
end
end
