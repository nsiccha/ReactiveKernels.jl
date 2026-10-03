using Test, Reactant, Enzyme, ReactiveKernelsDistributionKernels
import Distributions as OwnedTailD
const OwnedTailSources=ReactiveKernelsDistributionKernels.DistributionKernelSources

function _owned_tail_measure(f,v,expected,data...)
    native_f(v)=f(v,data...)
    @test native_f(v)≈expected atol=2e-12
    native=only(Enzyme.gradient(Enzyme.Reverse,native_f,v))
    finite=map(eachindex(v)) do i
        up,down=copy(v),copy(v);up[i]+=1e-5;down[i]-=1e-5
        (native_f(up)-native_f(down))/2e-5
    end
    @test native≈finite atol=1e-7 rtol=1e-6
    rv=Reactant.to_rarray(v)
    rdata=map(Reactant.to_rarray,data)
    hlo=repr(Reactant.@code_hlo optimize=false f(rv,rdata...))
    @test Float64((Reactant.@compile f(rv,rdata...))(rv,rdata...))≈expected atol=2e-12
    gradient(v,d...)=only(Enzyme.gradient(Enzyme.Reverse,x->f(x,d...),v))
    @test Array((Reactant.@compile gradient(rv,rdata...))(rv,rdata...))≈native atol=1e-9 rtol=1e-8
    return count("stablehlo.while",hlo),count("stablehlo.if",hlo)
end

@testset "owned beta-binomial tails retain count loops" begin
    structures=Tuple{Int,Int}[]
    for n in (8,16)
        f(v)=OwnedTailSources.rk_beta_binomial_cdf(n,exp(sum(view(v,1:1))),exp(sum(view(v,2:2))),n÷2)
        v=[.2,-.1];expected=OwnedTailD.cdf(OwnedTailD.BetaBinomial(n,exp(v[1]),exp(v[2])),n÷2)
        push!(structures,_owned_tail_measure(f,v,expected))
    end
    @test all(first(s)>0 for s in structures)
    @test structures[1]==structures[2]
end

@testset "owned stopping-ordinal tails retain level loops" begin
    structures=Tuple{Int,Int}[]
    for K in (4,8)
        t=collect(range(-1.,1.;length=K-1));k=K÷2
        f(v,t)=OwnedTailSources.rk_ordinal_stopping_tail(t,sum(v),1.,nothing,1,k,Val(:logit),false)
        v=[.2];expected=1-prod(OwnedTailD.ccdf.(Ref(OwnedTailD.Logistic()),t[1:k].-v[1]))
        push!(structures,_owned_tail_measure(f,v,expected,t))
    end
    @test all(first(s)>0 for s in structures)
    @test structures[1]==structures[2]
end

@testset "owned inverse-Gaussian tails" begin
    for upper in (false,true)
        f(v)=OwnedTailSources.rk_inverse_gaussian_tail(exp(sum(view(v,1:1))),exp(sum(view(v,2:2))),.8,upper)
        v=[.2,.5];d=OwnedTailD.InverseGaussian(exp(v[1]),exp(v[2]))
        _owned_tail_measure(f,v,upper ? OwnedTailD.ccdf(d,.8) : OwnedTailD.cdf(d,.8))
    end
end

@testset "categorical tails retain support loops" begin
    structures=Tuple{Int,Int}[]
    for K in (4,8)
        f(v)=OwnedTailSources.rk_logprob_tail(v .- log(sum(exp.(v))),K÷2,false)
        v=collect(range(-.4,.7;length=K));p=exp.(v)./sum(exp.(v))
        push!(structures,_owned_tail_measure(f,v,sum(p[1:K÷2])))
    end
    @test all(first(s)>0 for s in structures)
    @test structures[1]==structures[2]
end
