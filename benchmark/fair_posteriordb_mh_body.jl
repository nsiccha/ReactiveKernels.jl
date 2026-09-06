# Fair benchmark: Mh (capture-recapture heterogeneity, dim 3+M ≈ 303) — the large-dim
# case. Random-effects binomial-logit with data augmentation (log_sum_exp over the
# unobserved). RK vs Stan (the posteriordb reference — already about optimal; the
# augmentation loop is inherent) vs optimized Turing.
using Random, LinearAlgebra, Statistics, SpecialFunctions, LogExpFunctions
using BenchmarkTools
using ReactiveKernels
import BridgeStan
import Turing, DynamicPPL, Distributions
import ADTypes, Mooncake
using DifferentiationInterface
import Enzyme
const LOG2PI=log(2π); const ROUNDS=10
jnum(x::Integer)=string(x); jnum(x::Real)=(v=Float64(x); isinteger(v) ? string(Int(v))*".0" : repr(v))
jvec(v)="["*join(jnum.(v),",")*"]"; jobj(p)="{"*join(["\"$k\":$v" for (k,v) in p],",")*"}"
stan_model(src,dj,tag)=(d=mktempdir();f=joinpath(d,"$tag.stan");write(f,src);BridgeStan.StanModel(f,dj))
function stable_ldf(model;adtype=nothing,logdensity=DynamicPPL.getlogjoint_internal)
    vi=DynamicPPL.OnlyAccsVarInfo(DynamicPPL.VectorValueAccumulator())
    _,vi=DynamicPPL.init!!(model,vi,DynamicPPL.InitFromUniform(-2.1,-1.9),DynamicPPL.LinkAll())
    fvi=DynamicPPL.OnlyAccsVarInfo(DynamicPPL.FixedTransformAccumulator())
    _,fvi=DynamicPPL.init!!(model,fvi,DynamicPPL.InitFromUniform(-2.1,-1.9),DynamicPPL.LinkAll())
    tr=DynamicPPL.get_fixed_transforms(fvi); vv=DynamicPPL.getacc(vi,Val(:VectorValue)).values
    vv=DynamicPPL.update_transforms!!(vv,tr); DynamicPPL.LogDensityFunction(model,logdensity,vv;adtype,fix_transforms=false)
end
bench(t;rounds=ROUNDS)=(b=@benchmarkable $t(); ts=Float64[]; by=Int[]; for _ in 1:rounds; e=minimum(run(b;samples=200,seconds=0.2)); push!(ts,e.time); push!(by,e.memory); end; (;median_ns=median(ts),median_bytes=Int(median(by))))
fmt(ns)= ns<1e3 ? "$(round(ns;digits=1)) ns" : ns<1e6 ? "$(round(ns/1e3;digits=2)) µs" : "$(round(ns/1e6;digits=3)) ms"
fd_grad(f,q)=(g=similar(q);h=1e-6; for i in eachindex(q); qp=copy(q);qp[i]+=h;qm=copy(q);qm[i]-=h; g[i]=(f(qp)-f(qm))/(2h); end; g)
lchoose(N,c)=loggamma(N+1.0)-loggamma(c+1.0)-loggamma(N-c+1.0)

function gen_mh(M,T;seed=20260906)
    rng=MersenneTwister(seed); y=zeros(Int,M)
    for i in 1:M
        if rand(rng)<0.6                          # present
            p=logistic(logit(0.4)+0.8*randn(rng)); y[i]=rand(rng,Distributions.Binomial(T,p))
        end
    end
    y
end
# constraint transforms matching Stan <lower=0,upper=1>/<lower=0,upper=5>
mh_unpack(q,M)=(logistic(q[1]), logistic(q[2]), 5*logistic(q[3]), @view q[4:3+M])
mh_jac(om,mp,sg)= log(om)+log1p(-om) + log(mp)+log1p(-mp) + log(5)+log(sg/5)+log1p(-sg/5)
function mh_loglik(om, eps, y, T, cterm)
    lo=log(om); l1=log1p(-om); s=cterm
    @inbounds for i in eachindex(y)
        e=eps[i]; lp=-T*log1pexp(e)
        s += y[i]>0 ? (lo + y[i]*e + lp) : logaddexp(lo+lp, l1)
    end
    s
end
mh_eps(mp,sg,er)= logit(mp) .+ sg.*er
mh_prioreps(er,M)= -0.5*M*LOG2PI - 0.5*dot(er,er)
mh_post(jac,pe,ll)=jac+pe+ll
function oracle_mh(q,y,T,cterm,M); om,mp,sg,er=mh_unpack(q,M); eps=mh_eps(mp,sg,er)
    mh_jac(om,mp,sg) + mh_prioreps(er,M) + mh_loglik(om,eps,y,T,cterm)
end
const STAN_MH=raw"""
data{int<lower=0> M;int<lower=0> T;array[M] int<lower=0,upper=T> y;}
parameters{real<lower=0,upper=1> omega;real<lower=0,upper=1> mean_p;real<lower=0,upper=5> sigma;vector[M] eps_raw;}
transformed parameters{vector[M] eps=logit(mean_p)+sigma*eps_raw;}
model{eps_raw~normal(0,1);
 for(i in 1:M){
   if(y[i]>0) target+=bernoulli_lpmf(1|omega)+binomial_logit_lpmf(y[i]|T,eps[i]);
   else target+=log_sum_exp(bernoulli_lpmf(1|omega)+binomial_logit_lpmf(0|T,eps[i]), bernoulli_lpmf(0|omega));}}
"""
Turing.@model function turing_mh(M,T,y)
    omega~Distributions.Uniform(0,1); mean_p~Distributions.Uniform(0,1); sigma~Distributions.Uniform(0,5)
    eps_raw~Distributions.MvNormal(zeros(M), I)
    eps=logit(mean_p).+sigma.*eps_raw
    for i in 1:M
        if y[i]>0
            DynamicPPL.@addlogprob! log(omega)+ (loggamma(T+1.0)-loggamma(y[i]+1.0)-loggamma(T-y[i]+1.0)) + y[i]*eps[i] - T*log1pexp(eps[i])
        else
            DynamicPPL.@addlogprob! logaddexp(log(omega)-T*log1pexp(eps[i]), log1p(-omega))
        end
    end
end
mh_om(q)=logistic(q[1]); mh_mp(q)=logistic(q[2]); mh_sg(q)=5*logistic(q[3]); mh_er(q,M)=@view q[4:3+M]
rk_mh=@kernel model(unconstrained,y,T,cterm,M)=begin
    om=mh_om(unconstrained); mp=mh_mp(unconstrained); sg=mh_sg(unconstrained); er=mh_er(unconstrained,M)
    eps=mh_eps(mp,sg,er); jac=mh_jac(om,mp,sg); pe=mh_prioreps(er,M); ll=mh_loglik(om,eps,y,T,cterm)
    posterior=mh_post(jac,pe,ll); return posterior
end
let
    M=300; T=7; y=gen_mh(M,T); cterm=sum(lchoose(T,y[i]) for i in 1:M if y[i]>0)
    q=vcat([0.0,0.0,0.0], fill(0.0,M))
    ref=oracle_mh(q,y,T,cterm,M); gref=fd_grad(qq->oracle_mh(qq,y,T,cterm,M),q); gsc=maximum(abs,gref)
    dj=jobj(["M"=>M,"T"=>T,"y"=>jvec(y)])
    sm=stan_model(STAN_MH,dj,"mh")
    ldf=stable_ldf(turing_mh(M,T,y);adtype=ADTypes.AutoMooncake(config=nothing))
    kb=prepare(rk_mh;have=(:unconstrained,:y,:T,:cterm,:M),want=:posterior,bound=(;y=y,T=T,cterm=cterm,M=M))
    prep=prepare_ad(kb,AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse),function_annotation=Enzyme.Const),q;active=:unconstrained)
    gbuf=similar(q); gerr(g)=maximum(abs,g.-gref)/gsc
    println("stan grad err ", gerr(BridgeStan.log_density_gradient(sm,q;propto=false,jacobian=true)[2]))
    println("turing grad err ", gerr(DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,q)[2]))
    println("rk grad err ", gerr(ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q)[2]))
    println("\n===== Mh (capture-recapture, M=$M, dim $(3+M)) =====")
    for (nm,f) in (("stan_primal",()->BridgeStan.log_density(sm,q;propto=false,jacobian=true)),
                   ("turing_primal",()->DynamicPPL.LogDensityProblems.logdensity(ldf,q)),
                   ("rk_primal",()->kb(q)),
                   ("stan_grad",()->BridgeStan.log_density_gradient(sm,q;propto=false,jacobian=true)),
                   ("turing_grad",()->DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,q)),
                   ("rk_grad",()->ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q)))
        r=bench(f); println("  $(rpad(nm,14)) $(fmt(r.median_ns))  $(r.median_bytes)B")
    end
end
println("\nMH_DONE")
