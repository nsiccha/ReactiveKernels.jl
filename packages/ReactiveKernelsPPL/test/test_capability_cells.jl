using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test

function _cap_cell_fixture(kind,n)
    data = Dict(:y=>collect(range(-0.2,0.4; length=n)))
    body = kind === :index ? quote y[i] ~ Normal(i,0.7) end :
        kind === :index_math ? quote mu = a+0.1*i; y[i] ~ Normal(mu,0.7) end :
        kind === :broadcast ? quote y[i] .~ Normal.(a,0.7) end :
        kind === :dotted_latent ? quote theta[i] ~ Normal.(a,0.8); y[i] ~ Normal(theta[i],0.7) end :
        kind === :free_latent ? quote theta[i] ~ Normal.(a,0.8) end : quote end
    ast = quote
        a ~ Normal(0,1)
        sigma ~ Exponential(1)
    end
    push!(ast.args,Expr(:macrocall,Symbol("@plate"),LineNumberNode(1),
        Expr(:for,:(i = eachindex(y)),body)))
    plan = lower_rkppl(ast,data; conditioned=keys(data))
    bound = bind_data(plan,data)
    built = build_kernel(bound)
    latent = kind in (:dotted_latent,:free_latent)
    values = latent ? (; a=0.25,sigma=0.9,theta=collect(range(-0.1,0.3; length=n))) :
        (; a=0.25,sigma=0.9)
    u = unconstrain(built.layout,values)
    sampler = prepare_sampler(built,bound,u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    function pointwise(v)
        th = constrain(built.layout,v)
        kind in (:free_latent,:empty) && return Float64[]
        mu = kind === :index ? collect(1:n) : kind === :index_math ? th.a .+ 0.1 .* collect(1:n) :
            kind === :dotted_latent ? th.theta : fill(th.a,n)
        return logpdf.(Normal.(mu,0.7),data[:y])
    end
    function oracle(v)
        th = constrain(built.layout,v)
        prior = logpdf(Normal(),th.a)+logpdf(Exponential(),th.sigma)+log(th.sigma)
        latent && (prior += sum(logpdf.(Normal(th.a,0.8),th.theta)))
        return prior+sum(pointwise(v))
    end
    return (; kind,plan,bound,built,u,sampler,pointwise,oracle,data)
end

function _cap_cell_fd(f,u)
    h=cbrt(eps(Float64))
    return [(f(u+h*e)-f(u-h*e))/(2h) for e in eachcol(Matrix{Float64}(I,length(u),length(u)))]
end

@testset "numeric cell indices, scalar broadcast and observation-free plates" begin
    for kind in (:index,:index_math,:broadcast,:dotted_latent,:free_latent,:empty)
        fx = _cap_cell_fixture(kind,3)
        value,gradient = sampler_value_and_gradient!(fx.sampler,similar(fx.u),fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _cap_cell_fd(fx.oracle,fx.u) rtol=4e-6
        pw = Base.invokelatest(prepare_query(fx.built,fx.bound,:pointwise),fx.u)
        if kind in (:free_latent,:empty)
            @test pw == (;)
            @test isempty(fx.bound.responses)
        else
            @test pw.y ≈ fx.pointwise(fx.u)
            @test length(pw.y) == 3
        end
    end
end
