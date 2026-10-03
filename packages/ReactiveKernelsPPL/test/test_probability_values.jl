using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

function _probability_value_fixture(family,n; named=false, scalar=false)
    x=repeat([-0.4,0.1,0.6],cld(n,3))[1:n]
    y=repeat([0,1,1],cld(n,3))[1:n]
    data=(; x,y,trials=fill(3,n))
    eta=scalar ? :a : :eta
    probability=:(1 .- exp.(-exp.($eta)))
    ast=quote
        a ~ Normal(0,1)
        b ~ Normal(0,1)
        eta = a .+ b .* x
    end
    named && push!(ast.args,:(p = $probability))
    arg=named ? :p : probability
    response=family===:Bernoulli ? :(y .~ Bernoulli.($arg)) :
        :(y .~ Binomial.(trials,$arg))
    push!(ast.args,response)
    model=_distributional_model(ast,data)
    oracle=u->begin
        etas=scalar ? fill(u[1],n) : u[1] .+ u[2].*x
        probabilities=1 .- exp.(-exp.(etas))
        ds=family===:Bernoulli ? Bernoulli.(probabilities) : Binomial.(data.trials,probabilities)
        sum(logpdf.(ds,y))+sum(logpdf.(Normal(),u))
    end
    (; model...,oracle,u=[0.2,-0.3])
end

@testset "inline and named probability arithmetic preserve Julia values" begin
    for family in (:Bernoulli,:Binomial), named in (false,true), scalar in (false,true)
        f=_probability_value_fixture(family,9; named,scalar)
        @test f.built.layout.total == 2
        _distributional_check(f,f.oracle,f.u)
    end
    # An ordinary expression can leave the law's support at runtime.
    data=(; x=[-0.4,0.1,0.6],y=[false,true,true])
    f=_distributional_model(quote
        a ~ Normal(0,1)
        b ~ Normal(0,1)
        eta = a .+ b .* x
        y .~ Bernoulli.(eta .+ 0.1)
    end,data)
    u=[-0.5,0.0]
    ad=Base.invokelatest(prepare_ad,f.kernel,AutoEnzyme(; mode=Enzyme.Reverse),u;active=:unconstrained)
    value,gradient=Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,ad,similar(u),u)
    @test value == -Inf
    @test gradient ≈ -u
    # The value parser still resolves names and functions in the model module.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0,1)
        y .~ Bernoulli.(missing_probability_function.(a .+ x))
    end,data; conditioned=(:y,))
end
