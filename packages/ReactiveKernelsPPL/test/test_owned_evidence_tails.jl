using Test, Enzyme, ReactiveKernelsDistributionKernels
import Distributions as OwnedTailD
const OwnedTailSources=ReactiveKernelsDistributionKernels.DistributionKernelSources

# Each case is `(f, v, expected, data)`: `f(v, data...)` evaluates the tail.
_owned_beta_binomial_tail_cases()=map((8,16)) do n
    f(v)=OwnedTailSources.rk_beta_binomial_cdf(n,exp(sum(view(v,1:1))),exp(sum(view(v,2:2))),n÷2)
    v=[.2,-.1]
    (f,v,OwnedTailD.cdf(OwnedTailD.BetaBinomial(n,exp(v[1]),exp(v[2])),n÷2),())
end

_owned_stopping_ordinal_tail_cases()=map((4,8)) do K
    t=collect(range(-1.,1.;length=K-1));k=K÷2
    f(v,t)=OwnedTailSources.rk_ordinal_stopping_tail(t,sum(v),1.,nothing,1,k,Val(:logit),false)
    v=[.2]
    (f,v,1-prod(OwnedTailD.ccdf.(Ref(OwnedTailD.Logistic()),t[1:k].-v[1])),(t,))
end

_owned_inverse_gaussian_tail_cases()=map((false,true)) do upper
    f(v)=OwnedTailSources.rk_inverse_gaussian_tail(exp(sum(view(v,1:1))),exp(sum(view(v,2:2))),.8,upper)
    v=[.2,.5];d=OwnedTailD.InverseGaussian(exp(v[1]),exp(v[2]))
    (f,v,upper ? OwnedTailD.ccdf(d,.8) : OwnedTailD.cdf(d,.8),())
end

_owned_categorical_tail_cases()=map((4,8)) do K
    f(v)=OwnedTailSources.rk_logprob_tail(v .- log(sum(exp.(v))),K÷2,false)
    v=collect(range(-.4,.7;length=K));p=exp.(v)./sum(exp.(v))
    (f,v,sum(p[1:K÷2]),())
end

const _OWNED_TAIL_CASES=(
    "beta-binomial"=>_owned_beta_binomial_tail_cases,
    "stopping-ordinal"=>_owned_stopping_ordinal_tail_cases,
    "inverse-Gaussian"=>_owned_inverse_gaussian_tail_cases,
    "categorical"=>_owned_categorical_tail_cases)

@testset "owned $name tails: values and native reverse gradients" for (name,cases) in _OWNED_TAIL_CASES
    for (f,v,expected,data) in cases()
        native_f(v)=f(v,data...)
        @test native_f(v)≈expected atol=2e-12
        native=only(Enzyme.gradient(Enzyme.Reverse,native_f,v))
        finite=map(eachindex(v)) do i
            up,down=copy(v),copy(v);up[i]+=1e-5;down[i]-=1e-5
            (native_f(up)-native_f(down))/2e-5
        end
        @test native≈finite atol=1e-7 rtol=1e-6
    end
end
