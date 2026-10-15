using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test

function scalar_prior_fixture(family, separated, width=2)
    prior = Dict(:normal => :(Normal(0, 1)), :cauchy => :(Cauchy(0, 1)),
        :laplace => :(Laplace(0, 1)), :logistic => :(Logistic(0, 1)),
        :student_t => :(StudentT(3, 0, 1)))[family]
    extras = [:( $(Symbol(:extra_,i)) ~ $prior ) for i in 1:width-2]
    middle = separated ? [:(scale ~ Exponential(1)), :(slope ~ $prior)] :
        [:(slope ~ $prior), :(scale ~ Exponential(1))]
    ast = quote
        center ~ $prior
        $(middle...)
        $(extras...)
        levels[1:J] .~ Normal.(center, scale)
        location = levels[group] .+ slope .* x
        y .~ Normal.(location, 0.9)
    end
    RKPPLModel(ast, @__MODULE__)
end

_scalar_prior_normal(x, m, s) = -0.5log(2pi)-log(s)-0.5*((x-m)/s)^2
function scalar_prior_reference(family, values, group, x, y)
    scalar(v) = family === :normal ? _scalar_prior_normal(v,0,1) :
        family === :cauchy ? -log(pi)-log1p(v^2) :
        family === :laplace ? -log(2)-abs(v) :
        family === :logistic ? -v-2log1p(exp(-v)) :
        log(2)-log(pi)-0.5log(3)-2log1p(v^2/3)
    (;center,scale,slope,levels) = values
    mu = levels[group] .+ slope .* x
    extra_prior = sum(scalar(getproperty(values,k)) for k in keys(values)
        if startswith(string(k),"extra_"); init=0.0)
    scalar(center)+scalar(slope)+extra_prior-scale+log(scale)+
        sum(z -> _scalar_prior_normal(z,center,scale),levels;init=0.0)+
        sum(i -> _scalar_prior_normal(y[i],mu[i],0.9),eachindex(y);init=0.0)
end

@testset "scalar prior packing preserves declarations and gradients" begin
    cases = [(family,separated,2,n,J) for family in (:normal,:cauchy)
        for separated in (false,true) for (n,J) in ((0,3),(97,7))]
    append!(cases, [(family,true,2,97,7) for family in (:laplace,:logistic,:student_t)])
    push!(cases, (:normal,true,17,97,7))
    for (family,separated,width,n,J) in cases
        model=scalar_prior_fixture(family,separated,width)
        group=[mod1(div(i-1,8)+1,J) for i in 1:n]
        x=[sin(i/13)+0.2 for i in 1:n]
        y=[cos(i/9)/3 for i in 1:n]
        before=deepcopy((group,x,y))
        plan=model(;J,group,x) | (;y)
        built=build_kernel(plan)
        params=(;center=0.2,scale=0.8,slope=-0.3,levels=[0.1cos(i) for i in 1:J],
            (Symbol(:extra_,i) => 0.1sin(i) for i in 1:width-2)...)
        u=collect(Float64,unconstrain(built.layout,params))
        q=prepare_sampler(built,plan,u;backend=AutoEnzyme(;mode=Enzyme.Reverse),native_forward_batch=false)
        @test q.ad.backend.mode === Enzyme.Reverse
        for shift in (0.0,0.07)
            v=u .+ shift
            saved=copy(v)
            reference(w)=scalar_prior_reference(family,constrain(built.layout,w),group,x,y)
            h=cbrt(eps(Float64))
            expected=[begin
                up,down=copy(v),copy(v)
                up[k]+=h;down[k]-=h
                (reference(up)-reference(down))/(2h)
            end for k in eachindex(v)]
            g=similar(v)
            value,_=sampler_value_and_gradient!(q,g,v)
            @test value ≈ reference(v) rtol=1e-12 atol=1e-12
            @test g ≈ expected rtol=2e-6 atol=1e-7
            @test v == saved
            @test (group,x,y) == before
        end
    end
end
