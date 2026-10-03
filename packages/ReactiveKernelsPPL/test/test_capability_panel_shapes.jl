using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test

function _cap_panel_fixture(kind, subjects)
    t = repeat([0.2,0.7,1.3]; outer=subjects)
    dose = collect(range(0.7,1.5; length=subjects))
    obs = 0.3 .+ 0.02 .* collect(eachindex(t))
    data = Dict{Symbol,Any}(:t=>t,:dose=>dose,:obs=>obs)
    free = kind in (:free,:mixed_free)
    ast = quote
        a ~ Normal(0,1)
        sigma ~ Exponential(1)
        pred ~ plate(t,dose,obs; subjects=kernel_nsub_pred) do ts,d,yy
            mu = (a .* d) .* ts
            mu
        end
    end
    if !free
        cell = ast.args[end].args[3].args[2].args[2]
        insert!(cell.args,length(cell.args),:(yy .~ Normal.(mu,sigma)))
    end
    if kind in (:mixed,:mixed_free)
        data[:x] = collect(range(-0.4,0.4; length=subjects+2))
        data[:y] = fill(0.2,subjects+2)
        push!(ast.args,:(top_mu = a .+ 0.2 .* x),:(y .~ Normal.(top_mu,sigma)))
    end
    plan = lower_rkppl(ast,data; conditioned=keys(data))
    bound = bind_data(plan,data; dims=Dict(:kernel_nsub_pred=>subjects,:kernel_T_pred=>3))
    built = build_kernel(bound)
    u = unconstrain(built.layout,(; a=0.6,sigma=0.9))
    sampler = prepare_sampler(built,bound,u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    function oracle(v)
        th = constrain(built.layout,v)
        prior = logpdf(Normal(),th.a)+logpdf(Exponential(),th.sigma)+log(th.sigma)
        cell = free ? 0.0 : sum(logpdf.(Normal.(th.a .* repeat(dose; inner=3) .* t,th.sigma),obs))
        top = haskey(data,:y) ? sum(logpdf.(Normal.(th.a .+ 0.2 .* data[:x],th.sigma),data[:y])) : 0.0
        return prior+cell+top
    end
    return (; bound,built,u,sampler,oracle,data,free)
end
function _cap_panel_findiff(f,u)
    h=cbrt(eps(Float64))
    return [(f(u+h*e)-f(u-h*e))/(2h) for e in eachcol(Matrix{Float64}(I,length(u),length(u)))]
end

@testset "deterministic panels and ordinary responses compose" begin
    for kind in (:free,:mixed_free,:mixed)
        fx = _cap_panel_fixture(kind,3)
        @test Base.invokelatest(fx.sampler.kernel,fx.u) ≈ fx.oracle(fx.u)
        value,gradient = sampler_value_and_gradient!(fx.sampler,similar(fx.u),fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _cap_panel_findiff(fx.oracle,fx.u) rtol=3e-6
        pointwise = Base.invokelatest(prepare_query(fx.built,fx.bound,:pointwise),fx.u)
        @test Set(keys(pointwise)) == Set(kind === :free ? () : kind === :mixed_free ? (:y,) : (:obs,:y))
        likelihood = Base.invokelatest(prepare_query(fx.built,fx.bound,:likelihood),fx.u)
        @test sum((sum(v) for v in values(pointwise)); init=0.0) ≈ likelihood
        collected = Base.invokelatest(prepare,fx.built.spec;
            have=ReactiveKernelsPPL._query_have(fx.bound),want=:pred,
            bound=ReactiveKernelsPPL._query_bound(fx.bound))
        @test Base.invokelatest(collected,fx.u) ≈ 0.6 .* repeat(fx.data[:dose]; inner=3) .* fx.data[:t]
        @test isempty(only(fx.bound.kernel_plates).obs) == fx.free
        if haskey(fx.data,:y)
            bad = copy(fx.data)
            bad[:x] = [0.1,0.2]
            @test_throws ContractValidationError bind_data(lower_rkppl(
                quote
                    a ~ Normal(0,1)
                    sigma ~ Exponential(1)
                    pred ~ plate(t,dose,obs; subjects=kernel_nsub_pred) do ts,d,yy
                        mu = (a .* d) .* ts
                        mu
                    end
                    top_mu = a .+ 0.2 .* x
                    y .~ Normal.(top_mu,sigma)
                end,bad; conditioned=keys(bad)),bad;
                dims=Dict(:kernel_nsub_pred=>3,:kernel_T_pred=>3))
        end
    end
end
