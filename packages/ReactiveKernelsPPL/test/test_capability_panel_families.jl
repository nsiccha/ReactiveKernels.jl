using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test

function _cap_panel_family(kind, subjects)
    t = repeat([0.2,0.7,1.3]; outer=subjects)
    trials = [isodd(s) ? 2 : 4 for s in 1:subjects]
    obs = kind in (:scalar_poisson,:scalar_nb) ? trials :
        kind in (:binomial,:trials) ? [i % 3 for i in eachindex(t)] :
        0.2 .+ 0.03 .* collect(eachindex(t))
    rhs = kind === :binomial ? :(Binomial.(2,logistic.(mu))) :
        kind === :trials ? :(Binomial.(nn,logistic.(mu))) :
        kind === :cauchy ? :(Cauchy.(mu,sigma)) :
        kind === :scalar_poisson ? :(Poisson.(exp.(mu))) :
        kind === :scalar_nb ? :(NegativeBinomial2.(exp.(mu),2.0)) :
        :(Normal.(mu .+ 1.0,sigma))
    ast = quote
        a ~ Normal(0,1)
        sigma ~ Exponential(1)
        pred ~ plate(t,obs,trials; subjects=kernel_nsub_pred) do ts,yy,nn
            mu = a .* ts
            yy .~ $rhs
            mu
        end
    end
    data = Dict{Symbol,Any}(:t=>t,:obs=>obs,:trials=>trials)
    before = deepcopy(data)
    bound = bind_data(lower_rkppl(ast,data; conditioned=keys(data)),data;
        dims=Dict(:kernel_nsub_pred=>subjects,:kernel_T_pred=>3))
    built = build_kernel(bound)
    u = unconstrain(built.layout,(; a=0.3,sigma=0.9))
    q = prepare_sampler(built,bound,u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    function pointwise(v)
        p = constrain(built.layout,v)
        mu = p.a .* t
        if kind in (:binomial,:trials)
            n = kind === :binomial ? fill(2,length(t)) : repeat(trials; inner=3)
            return logpdf.(Binomial.(n,1 ./ (1 .+ exp.(-mu))),obs)
        elseif kind === :scalar_poisson
            return logpdf.(Poisson.(exp.(mu)),repeat(obs; inner=3))
        elseif kind === :scalar_nb
            return logpdf.(NegativeBinomial.(2.0,2.0 ./ (2.0 .+ exp.(mu))),repeat(obs; inner=3))
        elseif kind === :cauchy
            return logpdf.(Cauchy.(mu,p.sigma),obs)
        end
        return logpdf.(Normal.(mu .+ 1.0,p.sigma),obs)
    end
    oracle(v) = let p=constrain(built.layout,v)
        logpdf(Normal(),p.a)+logpdf(Exponential(),p.sigma)+log(p.sigma)+sum(pointwise(v))
    end
    return (; kind,ast,data,before,bound,built,u,q,pointwise,oracle)
end

function _cap_panel_family_fd(f,u)
    h = cbrt(eps(Float64))
    return [(f(u+h*e)-f(u-h*e))/(2h) for e in eachcol(Matrix{Float64}(I,length(u),length(u)))]
end

@testset "panel Binomial trials, Cauchy and inline arguments" begin
    for kind in (:binomial,:trials,:cauchy,:inline,:scalar_poisson,:scalar_nb)
        fx = _cap_panel_family(kind,3)
        value,grad = sampler_value_and_gradient!(fx.q,similar(fx.u),fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test grad ≈ _cap_panel_family_fd(fx.oracle,fx.u) rtol=3e-6
        pw = Base.invokelatest(prepare_query(fx.built,fx.bound,:pointwise),fx.u)
        @test pw.obs ≈ fx.pointwise(fx.u)
        @test coordinate_names(fx.built.layout) == [:a,:sigma]
        @test fx.data == fx.before
    end
    fx = _cap_panel_family(:trials,3)
    # Trial counts beyond Float64's integer precision retain their Julia value.
    large = merge(fx.data,Dict(:trials=>[2^53+1,2^53+3,2^53+5]))
    large_bound = bind_data(lower_rkppl(fx.ast,large; conditioned=keys(large)),large;
        dims=Dict(:kernel_nsub_pred=>3,:kernel_T_pred=>3))
    @test large_bound.columns[:pred_kexp_trials] == repeat(large[:trials]; inner=3)
    @test eltype(large_bound.columns[:pred_kexp_trials]) === Int
    bad = merge(fx.data,Dict(:trials=>Float64.(fx.data[:trials])))
    @test_throws "integer data" bind_data(lower_rkppl(fx.ast,bad; conditioned=keys(bad)),bad;
        dims=Dict(:kernel_nsub_pred=>3,:kernel_T_pred=>3))
    collision = deepcopy(fx.ast)
    cell = collision.args[end].args[3].args[2].args[2]
    insert!(cell.args,1,:(yy_arg2 = 0.5))
    @test_throws "already a model name" lower_rkppl(collision,fx.data; conditioned=keys(fx.data))
end
