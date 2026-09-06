# Fair all-sides benchmark, batch 2: arK (AR(K)) and GLMM_Poisson (hierarchical Poisson).
# RK @kernel vs Stan (naive loop / vectorized / fused-GLM where it exists) vs optimized Turing.
# Weak-normal unbounded priors; native primal+gradient; parity vs FD oracle.
using Random, LinearAlgebra, Statistics, SpecialFunctions, LogExpFunctions, Dates
using BenchmarkTools
import TOML
using ReactiveKernels
import BridgeStan
import Turing, DynamicPPL, Distributions
import ADTypes, Mooncake
using DifferentiationInterface
import Enzyme
const ROUNDS = 12
const LOG2PI = log(2π)
nlp(x,m,s) = -0.5*LOG2PI - log(s) - 0.5*((x-m)/s)^2

jnum(x::Integer)=string(x); jnum(x::Real)=(v=Float64(x); isinteger(v) ? string(Int(v))*".0" : repr(v))
jvec(v)="["*join(jnum.(v),",")*"]"; jmat(M)="["*join([jvec(view(M,i,:)) for i in 1:size(M,1)],",")*"]"
jobj(p)="{"*join(["\"$k\":$v" for (k,v) in p],",")*"}"
stan_model(src,dj,tag)=(d=mktempdir();f=joinpath(d,"$tag.stan");write(f,src);BridgeStan.StanModel(f,dj))
function stable_ldf(model;adtype=nothing,logdensity=DynamicPPL.getlogjoint_internal)
    vi=DynamicPPL.OnlyAccsVarInfo(DynamicPPL.VectorValueAccumulator())
    _,vi=DynamicPPL.init!!(model,vi,DynamicPPL.InitFromUniform(-2.1,-1.9),DynamicPPL.LinkAll())
    fvi=DynamicPPL.OnlyAccsVarInfo(DynamicPPL.FixedTransformAccumulator())
    _,fvi=DynamicPPL.init!!(model,fvi,DynamicPPL.InitFromUniform(-2.1,-1.9),DynamicPPL.LinkAll())
    tr=DynamicPPL.get_fixed_transforms(fvi); vv=DynamicPPL.getacc(vi,Val(:VectorValue)).values
    vv=DynamicPPL.update_transforms!!(vv,tr); DynamicPPL.LogDensityFunction(model,logdensity,vv;adtype,fix_transforms=false)
end
function bench(t;rounds=ROUNDS); b=@benchmarkable $t(); ts=Float64[]; by=Int[]
    for _ in 1:rounds; e=minimum(run(b;samples=200,seconds=0.2)); push!(ts,e.time); push!(by,e.memory); end
    (;median_ns=median(ts),median_bytes=Int(median(by))); end
fmt(ns)= ns<1e3 ? "$(round(ns;digits=1)) ns" : ns<1e6 ? "$(round(ns/1e3;digits=2)) µs" : "$(round(ns/1e6;digits=3)) ms"
fd_grad(f,q)=(g=similar(q);h=1e-6; for i in eachindex(q); qp=copy(q);qp[i]+=h;qm=copy(q);qm[i]-=h; g[i]=(f(qp)-f(qm))/(2h); end; g)

function run_model(name, q, oracle, stan_specs, ldf, kern_b, prep, gbuf, opt_baseline)
    ref=oracle(q); gref=fd_grad(oracle,q); gscale=maximum(abs,gref); gerr(g)=maximum(abs,g.-gref)/gscale
    rows=Dict{String,Any}[]; add!(k,i,r)=push!(rows,Dict("kind"=>k,"impl"=>i,"median_ns"=>r.median_ns,"median_bytes"=>r.median_bytes))
    for (impl,sm) in stan_specs
        gerr(BridgeStan.log_density_gradient(sm,q;propto=false,jacobian=true)[2])<2e-3 || error("parity $name/$impl")
        add!("primal",impl,bench(()->BridgeStan.log_density(sm,q;propto=false,jacobian=true)))
    end
    gerr(DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,q)[2])<2e-3 || error("parity $name/turing")
    gerr(ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q)[2])<2e-3 || error("parity $name/rk")
    add!("primal","turing",bench(()->DynamicPPL.LogDensityProblems.logdensity(ldf,q)))
    add!("primal","rk",bench(()->kern_b(q)))
    base_sm=Dict(stan_specs)[opt_baseline]
    add!("gradient","stan",bench(()->BridgeStan.log_density_gradient(base_sm,q;propto=false,jacobian=true)))
    add!("gradient","turing",bench(()->DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,q)))
    add!("gradient","rk",bench(()->ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q)))
    println("\n===== $name  dim=$(length(q)) =====")
    for kind in ("primal","gradient")
        bkey= kind=="primal" ? opt_baseline : "stan"
        base=first(r["median_ns"] for r in rows if r["kind"]==kind && r["impl"]==bkey)
        println("  -- $kind (baseline optimized Stan) --")
        for r in rows; r["kind"]==kind||continue
            println("     $(rpad(r["impl"],12)) $(lpad(fmt(r["median_ns"]),10))  $(lpad(string(round(r["median_ns"]/base;digits=2))*"x",8))  $(r["median_bytes"])B"); end
    end
    Dict("model"=>name,"dim"=>length(q),"measurements"=>rows)
end

results=Any[]

# ===================== arK  (AR(K) → normal regression on a lag matrix) =====================
# q = [alpha, beta[1..K], log_sigma]; yt ~ Normal(alpha + Ylag*beta, sigma). Weak N(0,10)/N(0,5) priors.
function gen_ark(T,K;seed=20260906)
    rng=MersenneTwister(seed); y=zeros(T); y[1:K].=randn(rng,K)
    b=[0.5,-0.2,0.15,0.1,-0.05,0.05,-0.03][1:K]
    for t in K+1:T; y[t]=0.3 + sum(b[k]*y[t-k] for k in 1:K) + 0.4*randn(rng); end
    Ylag=zeros(T-K,K); for t in K+1:T, k in 1:K; Ylag[t-K,k]=y[t-k]; end
    yt=y[K+1:T]; Ylag, yt
end
function oracle_ark(q,Ylag,yt); K=size(Ylag,2); α=q[1]; β=@view q[2:K+1]; ls=q[K+2]; σ=exp(ls); m=length(yt)
    η=α.+Ylag*β; r=yt.-η
    sum(x->nlp(x,0,10), @view q[1:K+1]) + nlp(σ,0,5) + ls +
      (-0.5*m*LOG2PI - m*ls - 0.5*dot(r,r)/σ^2)
end
const STAN_ARK_NAIVE=raw"""
data{int<lower=0> m;int<lower=0> K;matrix[m,K] Ylag;vector[m] yt;}
parameters{real alpha;vector[K] beta;real<lower=0> sigma;}
model{target+=normal_lpdf(alpha|0,10);target+=normal_lpdf(beta|0,10);target+=normal_lpdf(sigma|0,5);
 for(i in 1:m){real mu=alpha; for(k in 1:K) mu+=beta[k]*Ylag[i,k]; target+=normal_lpdf(yt[i]|mu,sigma);}}
"""
const STAN_ARK_GLM=raw"""
data{int<lower=0> m;int<lower=0> K;matrix[m,K] Ylag;vector[m] yt;}
parameters{real alpha;vector[K] beta;real<lower=0> sigma;}
model{target+=normal_lpdf(alpha|0,10);target+=normal_lpdf(beta|0,10);target+=normal_lpdf(sigma|0,5);
 target+=normal_id_glm_lpdf(yt|Ylag,alpha,beta,sigma);}
"""
Turing.@model function turing_ark(Ylag,yt,K)
    alpha~Distributions.Normal(0,10); beta~Distributions.MvNormal(zeros(K),100.0*I)
    sigma~Distributions.truncated(Distributions.Normal(0,5);lower=0)
    yt~Distributions.MvNormal(alpha.+Ylag*beta, sigma^2*I)
end
ark_a(q)=q[1]; ark_b(q)=@view q[2:end-1]; ark_ls(q)=q[end]
ark_eta(a,b,Y)=a.+Y*b
ark_ll(yt,eta,ls,s,m)= -0.5*m*LOG2PI - m*ls - 0.5*dot(yt.-eta, yt.-eta)/s^2
ark_prior(q,s,ls)=sum(x->nlp(x,0,10), @view q[1:end-1]) + nlp(s,0,5) + ls
rk_ark=@kernel model(unconstrained,Ylag,yt,m)=begin
    a=ark_a(unconstrained); b=ark_b(unconstrained); ls=ark_ls(unconstrained); s=exp(ls)
    eta=ark_eta(a,b,Ylag); prior=ark_prior(unconstrained,s,ls); likelihood=ark_ll(yt,eta,ls,s,m)
    posterior=prior+likelihood; return posterior
end
let
    K=5; T=200; Ylag,yt=gen_ark(T,K); m=size(Ylag,1); q=[0.3; fill(0.1,K); log(0.5)]
    dj=jobj(["m"=>m,"K"=>K,"Ylag"=>jmat(Ylag),"yt"=>jvec(yt)])
    stans=["stan_naive"=>stan_model(STAN_ARK_NAIVE,dj,"ark_naive"),"stan_glm"=>stan_model(STAN_ARK_GLM,dj,"ark_glm")]
    ldf=stable_ldf(turing_ark(Ylag,yt,K);adtype=ADTypes.AutoMooncake(config=nothing))
    kb=prepare(rk_ark;have=(:unconstrained,:Ylag,:yt,:m),want=:posterior,bound=(;Ylag=Ylag,yt=yt,m=m))
    prep=prepare_ad(kb,AutoEnzyme(mode=Enzyme.Reverse,function_annotation=Enzyme.Const),q;active=:unconstrained)
    push!(results, run_model("arK (AR($K), T=$T)", q, qq->oracle_ark(qq,Ylag,yt), stans, ldf, kb, prep, similar(q), "stan_glm"))
end

# ===================== GLMM_Poisson (hierarchical Poisson-log with per-obs random effects) =====================
# q = [alpha, b1, b2, b3, eps[1..n], log_sigma]; log_lambda = alpha + X*bfix + eps; C~poisson_log; eps~N(0,sigma).
function gen_glmm(n;seed=20260906)
    rng=MersenneTwister(seed); yr=collect(range(-1.5,1.5;length=n)); X=hcat(yr,yr.^2,yr.^3)
    bf=[0.6,0.3,-0.15,0.08]; eps=0.3.*randn(rng,n); ll=bf[1].+X*bf[2:4].+eps
    C=[rand(rng,Distributions.Poisson(exp(min(e,6.0)))) for e in ll]; X,C
end
# layout matches Turing's natural declaration order: q = [alpha, beta(3), log_sigma, eps(n)]
function oracle_glmm(q,X,C,cterm); n=length(C); α=q[1]; bf=@view q[2:4]; ls=q[5]; σ=exp(ls); eps=@view q[6:5+n]
    η=α.+X*bf.+eps
    prior_fix=sum(x->nlp(x,0,10), @view q[1:4]); prior_sig=nlp(σ,0,5)+ls
    ll_eps=-0.5*n*LOG2PI - n*ls - 0.5*dot(eps,eps)/σ^2
    ll_pois=dot(C,η)-sum(exp,η)-cterm
    prior_fix+prior_sig+ll_eps+ll_pois
end
const STAN_GLMM_NAIVE=raw"""
data{int<lower=0> n;int<lower=0> K;matrix[n,K] X;array[n] int<lower=0> C;}
parameters{real alpha;vector[K] beta;real<lower=0> sigma;vector[n] eps;}
model{target+=normal_lpdf(alpha|0,10);target+=normal_lpdf(beta|0,10);target+=normal_lpdf(sigma|0,5);
 vector[n] ll=alpha+X*beta+eps; for(i in 1:n) target+=poisson_log_lpmf(C[i]|ll[i]);
 for(i in 1:n) target+=normal_lpdf(eps[i]|0,sigma);}
"""
const STAN_GLMM_VEC=raw"""
data{int<lower=0> n;int<lower=0> K;matrix[n,K] X;array[n] int<lower=0> C;}
parameters{real alpha;vector[K] beta;real<lower=0> sigma;vector[n] eps;}
model{target+=normal_lpdf(alpha|0,10);target+=normal_lpdf(beta|0,10);target+=normal_lpdf(sigma|0,5);
 target+=poisson_log_lpmf(C|alpha+X*beta+eps); target+=normal_lpdf(eps|0,sigma);}
"""
Turing.@model function turing_glmm(X,C,K,n)
    alpha~Distributions.Normal(0,10); beta~Distributions.MvNormal(zeros(K),100.0*I)
    sigma~Distributions.truncated(Distributions.Normal(0,5);lower=0)
    eps~Distributions.MvNormal(zeros(n), sigma^2*I)
    C~Distributions.product_distribution([Distributions.Poisson(exp(e)) for e in (alpha.+X*beta.+eps)])
end
glmm_a(q)=q[1]; glmm_bf(q)=@view q[2:4]; glmm_ls(q)=q[5]; glmm_eps(q,n)=@view q[6:5+n]; glmm_s(ls)=exp(ls)
glmm_eta(a,bf,X,eps)=a.+X*bf.+eps
glmm_prior(q,s,ls)=sum(x->nlp(x,0,10), @view q[1:4]) + nlp(s,0,5) + ls
glmm_lleps(eps,ls,s,n)= -0.5*n*LOG2PI - n*ls - 0.5*dot(eps,eps)/s^2
glmm_llpois(C,eta,cterm)=dot(C,eta)-sum(exp,eta)-cterm
glmm_post(p,le,lp)=p+le+lp
rk_glmm=@kernel model(unconstrained,X,C,cterm,n)=begin
    a=glmm_a(unconstrained); bf=glmm_bf(unconstrained); eps=glmm_eps(unconstrained,n)
    ls=glmm_ls(unconstrained); s=glmm_s(ls)
    eta=glmm_eta(a,bf,X,eps); prior=glmm_prior(unconstrained,s,ls)
    ll_eps=glmm_lleps(eps,ls,s,n); ll_pois=glmm_llpois(C,eta,cterm)
    posterior=glmm_post(prior,ll_eps,ll_pois)
    return posterior
end
let
    n=40; X,C=gen_glmm(n); K=3; cterm=sum(c->loggamma(c+1.0),C)
    q=vcat([0.5,0.25,-0.1,0.06,log(0.4)], fill(0.0,n)); Cf=Float64.(C)
    dj=jobj(["n"=>n,"K"=>K,"X"=>jmat(X),"C"=>jvec(C)])
    stans=["stan_naive"=>stan_model(STAN_GLMM_NAIVE,dj,"glmm_naive"),"stan_vec"=>stan_model(STAN_GLMM_VEC,dj,"glmm_vec")]
    ldf=stable_ldf(turing_glmm(X,C,K,n);adtype=ADTypes.AutoMooncake(config=nothing))
    kb=prepare(rk_glmm;have=(:unconstrained,:X,:C,:cterm,:n),want=:posterior,bound=(;X=X,C=Cf,cterm=cterm,n=n))
    prep=prepare_ad(kb,AutoEnzyme(mode=Enzyme.Reverse,function_annotation=Enzyme.Const),q;active=:unconstrained)
    push!(results, run_model("GLMM_Poisson (n=$n)", q, qq->oracle_glmm(qq,X,C,cterm), stans, ldf, kb, prep, similar(q), "stan_vec"))
end

outp=get(ENV,"AS_OUTPUT","")
if outp!=""
    receipt=Dict("schema"=>"fair-posteriordb-more-v1","generated_at"=>string(now()),
        "methodology"=>"median over $ROUNDS rounds of BenchmarkTools minimum(200 samples); RK vs optimized Stan vs optimized Turing; weak-normal priors; parity vs FD oracle",
        "environment"=>Dict("julia"=>string(VERSION),"arch"=>string(Sys.ARCH),"cpu"=>Sys.cpu_info()[1].model,"stan"=>"2.39.0 (BridgeStan 2.9.0)"),
        "models"=>results)
    open(outp,"w") do io; TOML.print(io,receipt); end; println("\nwrote receipt: $outp")
end
println("\nFAIR_MORE_DONE")
