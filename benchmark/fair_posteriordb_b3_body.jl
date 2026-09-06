# Fair all-sides benchmark, batch 3: Rate_1 (beta-binomial, dim 1, bounded param)
# and eight_schools_centered (hierarchical normal, dim J+2). RK vs optimized Stan
# vs optimized Turing; native primal+gradient; parity vs FD oracle.
using Random, LinearAlgebra, Statistics, SpecialFunctions, LogExpFunctions, Dates
using BenchmarkTools
import TOML
using ReactiveKernels
import BridgeStan
import Turing, DynamicPPL, Distributions
import ADTypes, Mooncake
using DifferentiationInterface
import Enzyme
const ROUNDS=12; const LOG2PI=log(2π)
nlp(x,m,s)= -0.5*LOG2PI - log(s) - 0.5*((x-m)/s)^2
cauchylp(x,l,s)= -log(π) - log(s) - log1p(((x-l)/s)^2)
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
function bench(t;rounds=ROUNDS); b=@benchmarkable $t(); ts=Float64[]; by=Int[]
    for _ in 1:rounds; e=minimum(run(b;samples=200,seconds=0.2)); push!(ts,e.time); push!(by,e.memory); end
    (;median_ns=median(ts),median_bytes=Int(median(by))); end
fmt(ns)= ns<1e3 ? "$(round(ns;digits=1)) ns" : ns<1e6 ? "$(round(ns/1e3;digits=2)) µs" : "$(round(ns/1e6;digits=3)) ms"
fd_grad(f,q)=(g=similar(q);h=1e-6; for i in eachindex(q); qp=copy(q);qp[i]+=h;qm=copy(q);qm[i]-=h; g[i]=(f(qp)-f(qm))/(2h); end; g)
function run_model(name,q,oracle,stan_specs,ldf,kern_b,prep,gbuf,opt_baseline)
    ref=oracle(q); gref=fd_grad(oracle,q); gscale=max(maximum(abs,gref),1e-8); gerr(g)=maximum(abs,g.-gref)/gscale
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

# ===================== Rate_1 (beta-binomial rate; dim 1, bounded theta) =====================
lchoose(N,c)=loggamma(N+1.0)-loggamma(c+1.0)-loggamma(N-c+1.0)
function oracle_rate1(q,n,k,cterm); u=q[1]; th=logistic(u); logjac=log(th)+log1p(-th)
    ll=cterm + k*log(th) + (n-k)*log1p(-th)
    0.0 + logjac + ll                          # Beta(1,1) log prior = 0
end
const STAN_RATE1=raw"""
data{int<lower=1> n;int<lower=0> k;}
parameters{real<lower=0,upper=1> theta;}
model{theta~beta(1,1); k~binomial(n,theta);}
"""
Turing.@model function turing_rate1(n,k)
    theta~Distributions.Beta(1,1); k~Distributions.Binomial(n,theta)
end
r1_theta(q)=logistic(q[1]); r1_jac(th)=log(th)+log1p(-th)
r1_ll(th,n,k,cterm)=cterm + k*log(th) + (n-k)*log1p(-th)
r1_post(jac,ll)=jac+ll
rk_rate1=@kernel model(unconstrained,n,k,cterm)=begin
    th=r1_theta(unconstrained); jac=r1_jac(th); likelihood=r1_ll(th,n,k,cterm); posterior=r1_post(jac,likelihood)
    return posterior
end
let
    n=100; k=34; cterm=lchoose(n,k); q=[0.0]
    dj=jobj(["n"=>n,"k"=>k])
    stans=["stan"=>stan_model(STAN_RATE1,dj,"rate1")]
    ldf=stable_ldf(turing_rate1(n,k);adtype=ADTypes.AutoMooncake(config=nothing))
    kb=prepare(rk_rate1;have=(:unconstrained,:n,:k,:cterm),want=:posterior,bound=(;n=n,k=k,cterm=cterm))
    prep=prepare_ad(kb,AutoEnzyme(mode=Enzyme.Reverse,function_annotation=Enzyme.Const),q;active=:unconstrained)
    push!(results, run_model("Rate_1 (beta-binomial)", q, qq->oracle_rate1(qq,n,k,cterm), stans, ldf, kb, prep, similar(q), "stan"))
end

# ===================== eight_schools_centered (hierarchical normal; dim J+2) =====================
# q = [mu, log_tau, theta(J)]. mu~N(0,5), tau~HalfCauchy(0,5), theta~N(mu,tau), y~N(theta,sigma).
function oracle_es(q,y,sig); J=length(y); mu=q[1]; lt=q[2]; tau=exp(lt); th=@view q[3:2+J]
    prior=nlp(mu,0,5)+cauchylp(tau,0,5)+lt+sum(t->nlp(t,mu,tau), th)
    ll=sum(nlp(y[i],th[i],sig[i]) for i in 1:J)
    prior+ll
end
const STAN_ES=raw"""
data{int<lower=0> J;vector[J] y;vector<lower=0>[J] sigma;}
parameters{real mu;real<lower=0> tau;vector[J] theta;}
model{target+=normal_lpdf(mu|0,5); target+=cauchy_lpdf(tau|0,5);
 target+=normal_lpdf(theta|mu,tau); target+=normal_lpdf(y|theta,sigma);}
"""
Turing.@model function turing_es(J,y,sig)
    mu~Distributions.Normal(0,5); tau~Distributions.truncated(Distributions.Cauchy(0,5);lower=0)
    theta~Distributions.MvNormal(fill(mu,J), tau^2*I)
    y~Distributions.MvNormal(theta, Distributions.Diagonal(sig.^2))
end
es_mu(q)=q[1]; es_lt(q)=q[2]; es_th(q)=@view q[3:end]; es_tau(lt)=exp(lt)
es_prior(mu,tau,lt,th)=nlp(mu,0,5)+cauchylp(tau,0,5)+lt+sum(t->nlp(t,mu,tau), th)
es_ll(y,th,sig)=sum(nlp(y[i],th[i],sig[i]) for i in eachindex(y))
es_post(p,l)=p+l
rk_es=@kernel model(unconstrained,y,sig)=begin
    mu=es_mu(unconstrained); lt=es_lt(unconstrained); th=es_th(unconstrained); tau=es_tau(lt)
    prior=es_prior(mu,tau,lt,th); likelihood=es_ll(y,th,sig); posterior=es_post(prior,likelihood)
    return posterior
end
let
    J=8; y=[28.,8.,-3.,7.,-1.,1.,18.,12.]; sig=[15.,10.,16.,11.,9.,11.,10.,18.]
    q=vcat([1.0, log(5.0)], fill(2.0,J))
    dj=jobj(["J"=>J,"y"=>jvec(y),"sigma"=>jvec(sig)])
    stans=["stan"=>stan_model(STAN_ES,dj,"es")]
    ldf=stable_ldf(turing_es(J,y,sig);adtype=ADTypes.AutoMooncake(config=nothing))
    kb=prepare(rk_es;have=(:unconstrained,:y,:sig),want=:posterior,bound=(;y=y,sig=sig))
    prep=prepare_ad(kb,AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse),function_annotation=Enzyme.Const),q;active=:unconstrained)
    push!(results, run_model("eight_schools_centered", q, qq->oracle_es(qq,y,sig), stans, ldf, kb, prep, similar(q), "stan"))
end

outp=get(ENV,"AS_OUTPUT","")
if outp!=""
    receipt=Dict("schema"=>"fair-posteriordb-b3-v1","generated_at"=>string(now()),
        "methodology"=>"median over $ROUNDS rounds of BenchmarkTools minimum(200 samples); RK vs optimized Stan vs optimized Turing; parity vs FD oracle",
        "environment"=>Dict("julia"=>string(VERSION),"arch"=>string(Sys.ARCH),"cpu"=>Sys.cpu_info()[1].model,"stan"=>"2.39.0 (BridgeStan 2.9.0)"),
        "models"=>results)
    open(outp,"w") do io; TOML.print(io,receipt); end; println("\nwrote receipt: $outp")
end
println("\nFAIR_B3_DONE")
