module BernoulliSharedSoftplusTests
using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using DifferentiationInterface: AutoEnzyme
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli, bernoulli_logit_glm
using LogExpFunctions: log1pexp, logistic
import Enzyme

@testset "Bernoulli shared primitive preserves both input routes" begin
    k=prepare(bernoulli.logpdf;have=(:observed,:logit),want=:logpdf)
    backend=AutoEnzyme(;mode=Enzyme.Reverse)
    op=prepare_ad(k,backend,true,0.0;active=:logit)
    setprecision(BigFloat,512) do
        for x in (-1000.0,-700.0,-40.0,-1.0,-0.0,0.0,1.0,40.0,700.0,1000.0),
                observed in (false,true)
            b=BigFloat(x)
            expected=observed ? -log1p(exp(-b)) : -log1p(exp(b))
            derivative=observed ? inv(1+exp(b)) : -inv(1+exp(-b))
            v,g=ad_value_and_gradient(op,observed,x)
            @test v ≈ Float64(expected) rtol=4eps(Float64)
            @test g ≈ Float64(derivative) rtol=4eps(Float64)
        end
    end
    @test k(true,Inf)==0.0
    @test k(false,-Inf)==0.0
    @test k(false,Inf)==-Inf
    @test k(true,-Inf)==-Inf

    direct=prepare(bernoulli.logpdf;have=(:observed,:p),want=:logpdf)
    direct_ad=prepare_ad(direct,backend,true,0.5;active=:p)
    @test ad_value_and_gradient(direct_ad,true,1.0)==(0.0,1.0)
    @test ad_value_and_gradient(direct_ad,false,0.0)==(0.0,-1.0)
    @test direct(false,1.0)==-Inf
    @test direct(true,0.0)==-Inf

    # Whole-vector GLM uses the same primitive in its signed-margin formula.
    glm=prepare(bernoulli_logit_glm.logpdf;have=(:y,:X,:beta),want=:logpdf)
    X=[1.0 -2.0; 1.0 0.0; 1.0 3.0]; y=[0,1,1]; beta=[.2,-.3]
    glm_ad=prepare_ad(glm,backend,y,X,beta;active=:beta)
    for b in (beta,[0.0,1000.0])
        v,g=ad_value_and_gradient(glm_ad,y,X,b)
        eta=X*b
        @test v ≈ sum(-log1pexp.((1 .- 2 .* y).*eta))
        @test g ≈ X'*(y .- logistic.(eta))
    end
end
end
