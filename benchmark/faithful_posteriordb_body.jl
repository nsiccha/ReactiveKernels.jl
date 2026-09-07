# Faithful posteriordb comparison — idiomatic RK-builtin graphs vs optimized/reference
# Stan vs optimized Turing, ALL sides on the SAME faithful real model, real priors
# (real bounded-interval transforms + Jacobians, improper flat where the .stan is flat)
# and the SAME FULL real posteriordb data. The RK side consumes the rich
# distribution-kernel `@kernel` graphs from packages/ReactiveKernelsPPLExamples. Every
# side is parity-gated against an independent finite-difference oracle BEFORE timing; a
# prior/likelihood/support mismatch is a HARD parity FAILURE, not a footnote.
#
# Stan sides per model: the posteriordb REFERENCE `.stan` (verbatim artifact bytes) AND,
# where a faster idiom exists and computes the identical density, an OPTIMIZED `.stan`
# (fused `poisson_log_glm` / `normal_id_glm`, vectorized). The real priors are preserved
# in BOTH Stan variants (same density), so "optimized" is only a faster implementation.
# eight_schools carries a 4th RK side: the same model authored in the experimental @ppl
# front-end (documentation-only machine-precision migration evidence).
using Random, LinearAlgebra, Statistics, Dates
using BenchmarkTools
import TOML, SHA, JSON
using ReactiveKernels, ReactiveKernelsPPLExamples
import BridgeStan
import Turing, DynamicPPL, Distributions
import ADTypes, Mooncake
using DifferentiationInterface
import Enzyme
const LOG2PI = log(2π); const ROUNDS = 12
logistic(u)=1/(1+exp(-u)); log1pexp(x)= x>0 ? x+log1p(exp(-x)) : log1p(exp(x))
logit(p)=log(p)-log1p(-p); logaddexp(a,b)=(m=max(a,b); m+log1p(exp(-abs(a-b)))); ijac(u,w)=log(w)-log1pexp(-u)-log1pexp(u)
jnum(x::Integer)=string(x); jnum(x::Real)=(v=Float64(x); isinteger(v) ? string(Int(v))*".0" : repr(v))
jvec(v)="["*join(jnum.(v),",")*"]"; jmat(M)="["*join([jvec(view(M,i,:)) for i in 1:size(M,1)],",")*"]"; jobj(p)="{"*join(["\"$k\":$v" for (k,v) in p],",")*"}"
stan_model(src,dj,tag)=(d=mktempdir();f=joinpath(d,"$tag.stan");write(f,src);BridgeStan.StanModel(f,dj))
const PDB = "/home/n/.julia/artifacts/f45397e9120c2d7e26bf6aff8c7aa810779a7e1b/posteriordb-1.0.0/posterior_database"
const ART = joinpath(PDB, "models", "stan")
stan_shas=Dict{String,String}()
# posteriordb REFERENCE Stan, compiled verbatim from the artifact bytes (SHA recorded).
function stanA(name,dj); src=read(joinpath(ART,"$name.stan"),String); stan_shas["reference:"*name]=bytes2hex(SHA.sha256(src)); stan_model(src,dj,name); end
# OPTIMIZED Stan (hand-written fused/vectorized, same density; SHA of the source recorded).
function stanS(tag,src,dj); stan_shas["optimized:"*tag]=bytes2hex(SHA.sha256(src)); stan_model(src,dj,tag); end
# FULL real posteriordb data from the on-host artifact (data/data/<name>.json.zip).
function load_data(name)
    j = read(pipeline(`unzip -p $(joinpath(PDB,"data","data","$name.json.zip"))`), String)
    JSON.parse(j)
end
fvec(d,k)=Float64.(d[k]); ivec(d,k)=Int.(d[k])
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
const AE = AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)
results=Any[]

# Multi-Stan-variant report. `stanv` = Vector of (label, StanModel), e.g.
# [("reference",sm_ref),("optimized",sm_opt)] or [("stan",sm)]. `extra` = additional
# RK-like sides (label, primal_fn, ad_prep, gbuf). `stan_perm` permutes the RK-order q
# into an artifact Stan's own parameter order (eight_schools). Strict parity gate.
function report(name,q,oracle,stanv,ldf,kb,prep,gbuf; extra=Tuple[], stan_perm=nothing, data_note="")
    sd(qq)= stan_perm===nothing ? qq : qq[stan_perm]
    function sg(g); stan_perm===nothing && return g; out=similar(g); out[stan_perm]=g; out; end
    qs=sd(q)
    gref=fd_grad(oracle,q); gsc=maximum(abs,gref); gerr(g)=maximum(abs,g.-gref)/gsc
    for (lbl,sm) in stanv; e=gerr(sg(BridgeStan.log_density_gradient(sm,qs;propto=false,jacobian=true)[2])); e<2e-3 || error("PARITY FAIL $name/stan:$lbl=$e"); end
    et=gerr(DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,q)[2]); et<2e-3 || error("PARITY FAIL $name/turing=$et")
    er=gerr(ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q)[2]); er<2e-3 || error("PARITY FAIL $name/rk=$er")
    exp2=[(lbl,gerr(ReactiveKernels.ad_value_and_gradient!(pp,gb,q)[2])) for (lbl,_,pp,gb) in extra]
    all(e<2e-3 for (_,e) in exp2) || error("PARITY FAIL $name/extra")
    meas=Dict{String,Any}[]
    pm(kind,impl,r)=push!(meas,Dict("kind"=>kind,"impl"=>impl,"median_ns"=>r.median_ns,"median_bytes"=>r.median_bytes))
    for (lbl,sm) in stanv; pm("primal","stan_"*lbl,bench(()->BridgeStan.log_density(sm,qs;propto=false,jacobian=true))); end
    pm("primal","turing",bench(()->DynamicPPL.LogDensityProblems.logdensity(ldf,q)))
    pm("primal","rk",bench(()->kb(q)))
    for (lbl,f,_,_) in extra; pm("primal",lbl,bench(()->f(q))); end
    for (lbl,sm) in stanv; pm("gradient","stan_"*lbl,bench(()->BridgeStan.log_density_gradient(sm,qs;propto=false,jacobian=true))); end
    pm("gradient","turing",bench(()->DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,q)))
    pm("gradient","rk",bench(()->ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q)))
    for (lbl,_,pp,gb) in extra; pm("gradient",lbl,bench(()->ReactiveKernels.ad_value_and_gradient!(pp,gb,q))); end
    println("\n===== $name  dim=$(length(q))  $data_note  (parity stan/turing/rk = $(join([string(round(gerr(sg(BridgeStan.log_density_gradient(sm,qs;propto=false,jacobian=true)[2]));sigdigits=2)) for (_,sm) in stanv],",")) / $(round(et;sigdigits=2)) / $(round(er;sigdigits=2))) =====")
    for m in meas; println("  $(rpad(m["kind"]*"/"*m["impl"],22)) $(lpad(fmt(m["median_ns"]),10))  $(m["median_bytes"])B"); end
    push!(results,Dict("model"=>name,"dim"=>length(q),"measurements"=>meas,"rk_builtin"=>true,"data_note"=>data_note))
end

# ---- optimized (fused) Stan sources, real priors preserved ----
const STAN_GP_OPT = raw"""
data { int<lower=0> n; int<lower=0> K; matrix[n,K] X; array[n] int<lower=0> C; }
parameters { real<lower=-20,upper=20> alpha; vector<lower=-10,upper=10>[K] beta; }
model { C ~ poisson_log_glm(X, alpha, beta); }
"""
const STAN_KID_OPT = raw"""
data { int<lower=0> N; int<lower=0> P; matrix[N,P] Xd; vector[N] kid_score; }
parameters { vector[P+1] beta; real<lower=0> sigma; }
model { sigma ~ cauchy(0, 2.5); kid_score ~ normal_id_glm(Xd, beta[1], beta[2:P+1], sigma); }
"""
const STAN_BLR_OPT = raw"""
data { int<lower=0> N; int<lower=0> D; matrix[N,D] X; vector[N] y; }
parameters { vector[D] beta; real<lower=0> sigma; }
model { beta ~ normal(0,10); sigma ~ normal(0,10); y ~ normal_id_glm(X, 0, beta, sigma); }
"""
const STAN_ARK_OPT = raw"""
data { int<lower=0> R; int<lower=0> K; matrix[R,K] Ylag; vector[R] yt; }
parameters { real alpha; vector[K] beta; real<lower=0> sigma; }
model { alpha ~ normal(0,10); beta ~ normal(0,10); sigma ~ cauchy(0,2.5); yt ~ normal_id_glm(Ylag, alpha, beta, sigma); }
"""

# ================= eight_schools (RK-builtin graph + @ppl 4th side; full data J=8) =================
const ES = ReactiveKernelsPPLExamples.EightSchoolsExample
function oracle_es(q,y,sigma)
    mu=q[1]; lt=q[2]; th=@view q[3:end]; tau=exp(lt); J=length(y)
    lp=-0.5*LOG2PI-log(5.0)-0.5*(mu/5)^2 + log(2.0)-log(π*5.0)-log1p((tau/5.0)^2) + lt
    lp+=sum(i-> -0.5*LOG2PI-log(tau)-0.5*((th[i]-mu)/tau)^2, 1:J)
    lp+=sum(i-> -0.5*LOG2PI-log(sigma[i])-0.5*((y[i]-th[i])/sigma[i])^2, 1:J); lp
end
Turing.@model function turing_es(y,sigma,J)
    mu ~ Distributions.Normal(0,5); tau ~ Distributions.truncated(Distributions.Cauchy(0,5); lower=0)
    theta ~ Distributions.MvNormal(fill(mu,J), tau^2*I); y ~ Distributions.MvNormal(theta, Diagonal(sigma.^2))
end
let
    d=load_data("eight_schools"); y=fvec(d,"y"); sigma=fvec(d,"sigma"); J=Int(d["J"])
    q=vcat([0.0,log(5.0)], zeros(J)); speq=vcat(3:(2+J),1,2)
    kb=prepare(ES.build_eight_schools_graph(); have=(:unconstrained,:observations,:observation_scales), want=:posterior, bound=(observations=y, observation_scales=sigma))
    prep=prepare_ad(kb, AE, q; active=:unconstrained)
    kbp=prepare(ReactiveKernelsPPLExamples.PPLEightSchoolsExample.build_ppl_eight_schools(); have=(:unconstrained,:observations,:observation_scales), want=:posterior, bound=(observations=y, observation_scales=sigma))
    prepp=prepare_ad(kbp, AE, q; active=:unconstrained)
    sm=stanA("eight_schools_centered", jobj(["J"=>J,"y"=>jvec(y),"sigma"=>jvec(sigma)]))
    ldf=stable_ldf(turing_es(y,sigma,J); adtype=ADTypes.AutoMooncake(config=nothing))
    report("eight_schools", q, qq->oracle_es(qq,y,sigma), [("stan",sm)], ldf, kb, prep, similar(q);
        extra=[("rk_ppl", kbp, prepp, similar(q))], stan_perm=speq, data_note="full J=$J")
end

# ================= GLM_Poisson (peregrine; reference poisson_log + optimized poisson_log_glm) =================
const GP = ReactiveKernelsPPLExamples.GLMPoissonExample
oracle_glmpois(q,year,C)=(a=-20+40*logistic(q[1]); b1=-10+20*logistic(q[2]); b2=-10+20*logistic(q[3]); b3=-10+20*logistic(q[4]);
    lp=(log(40.0)-log1pexp(-q[1])-log1pexp(q[1]))+(log(20.0)-log1pexp(-q[2])-log1pexp(q[2]))+(log(20.0)-log1pexp(-q[3])-log1pexp(q[3]))+(log(20.0)-log1pexp(-q[4])-log1pexp(q[4]));
    for i in eachindex(year); z=a+b1*year[i]+b2*year[i]^2+b3*year[i]^3; lp+=C[i]*z-exp(z); end; lp)
Turing.@model function turing_glmpois(year,C)
    alpha~Distributions.Uniform(-20,20); beta1~Distributions.Uniform(-10,10); beta2~Distributions.Uniform(-10,10); beta3~Distributions.Uniform(-10,10)
    z=alpha .+ beta1.*year .+ beta2.*year.^2 .+ beta3.*year.^3
    DynamicPPL.@addlogprob! sum(C.*z .- exp.(z))
end
let
    d=load_data("GLM_Poisson_Data"); year=fvec(d,"year"); C=ivec(d,"C"); n=length(year); q=[0.2,0.1,-0.05,0.03]
    kb=prepare(GP.build_glm_poisson_graph(); have=(:unconstrained,:year,:counts), want=:posterior, bound=(year=year, counts=C))
    prep=prepare_ad(kb, AE, q; active=:unconstrained)
    X=hcat(year, year.^2, year.^3)
    smr=stanA("GLM_Poisson_model", jobj(["n"=>n,"C"=>jvec(C),"year"=>jvec(year)]))
    smo=stanS("GLM_Poisson_opt", STAN_GP_OPT, jobj(["n"=>n,"K"=>3,"X"=>jmat(X),"C"=>jvec(C)]))
    ldf=stable_ldf(turing_glmpois(year,C); adtype=ADTypes.AutoMooncake(config=nothing))
    report("GLM_Poisson", q, qq->oracle_glmpois(qq,year,C), [("reference",smr),("optimized",smo)], ldf, kb, prep, similar(q); data_note="full n=$n")
end

# ================= GLM_Binomial (peregrine; vectorized binomial_logit = reference = optimum) =================
const GB = ReactiveKernelsPPLExamples.GLMBinomialExample
oracle_glmbin(q,year,C,N)=(a=q[1];b1=q[2];b2=q[3]; lp=sum(p-> -0.5*LOG2PI-log(100.0)-0.5*(p/100)^2,(a,b1,b2));
    for i in eachindex(year); z=a+b1*year[i]+b2*year[i]^2; lp+=C[i]*z-N[i]*log1pexp(z); end; lp)
Turing.@model function turing_glmbin(year,C,N)
    alpha~Distributions.Normal(0,100); beta1~Distributions.Normal(0,100); beta2~Distributions.Normal(0,100)
    z=alpha .+ beta1.*year .+ beta2.*year.^2
    DynamicPPL.@addlogprob! sum(C.*z .- N.*log1pexp.(z))
end
let
    d=load_data("GLM_Binomial_data"); year=fvec(d,"year"); C=ivec(d,"C"); N=ivec(d,"N"); n=length(year); q=[0.1,0.2,-0.1]
    kb=prepare(GB.build_glm_binomial_graph(); have=(:unconstrained,:year,:counts,:totals), want=:posterior, bound=(year=year,counts=C,totals=N))
    prep=prepare_ad(kb, AE, q; active=:unconstrained)
    sm=stanA("GLM_Binomial_model", jobj(["nyears"=>n,"C"=>jvec(C),"N"=>jvec(N),"year"=>jvec(year)]))
    ldf=stable_ldf(turing_glmbin(year,C,N); adtype=ADTypes.AutoMooncake(config=nothing))
    report("GLM_Binomial", q, qq->oracle_glmbin(qq,year,C,N), [("stan",sm)], ldf, kb, prep, similar(q); data_note="full n=$n")
end

# ================= kidscore_interaction (kidiq; reference vec normal + optimized normal_id_glm; FLAT beta) =================
const KI = ReactiveKernelsPPLExamples.KidscoreInteractionExample
oracle_kid(q,hs,iq,inter,kid)=(b=@view q[1:4]; sig=exp(q[5]); lp=-log(π*2.5)-log1p((sig/2.5)^2)+q[5];
    for i in eachindex(kid); mu=b[1]+b[2]*hs[i]+b[3]*iq[i]+b[4]*inter[i]; lp+=-0.5*LOG2PI-log(sig)-0.5*((kid[i]-mu)/sig)^2; end; lp)
Turing.@model function turing_kid(hs,iq,inter,kid)
    beta ~ Distributions.product_distribution([DynamicPPL.Flat() for _ in 1:4])   # improper flat, identical to RK/Stan
    sigma~Distributions.truncated(Distributions.Cauchy(0,2.5);lower=0)
    mu=beta[1] .+ beta[2].*hs .+ beta[3].*iq .+ beta[4].*inter
    DynamicPPL.@addlogprob! sum(-0.5*LOG2PI .- log(sigma) .- 0.5.*((kid.-mu)./sigma).^2)
end
let
    d=load_data("kidiq"); hs=fvec(d,"mom_hs"); iq=fvec(d,"mom_iq"); kid=fvec(d,"kid_score"); inter=hs.*iq; N=length(kid); q=[25.0,-5.0,0.5,0.05,log(15.0)]
    kb=prepare(KI.build_kidscore_interaction_graph(); have=(:unconstrained,:kid_score,:mom_hs,:mom_iq), want=:posterior, bound=(kid_score=kid,mom_hs=hs,mom_iq=iq))
    prep=prepare_ad(kb, AE, q; active=:unconstrained)
    Xd=hcat(hs,iq,inter)
    smr=stanA("kidscore_interaction", jobj(["N"=>N,"kid_score"=>jvec(kid),"mom_iq"=>jvec(iq),"mom_hs"=>jvec(hs)]))
    smo=stanS("kidscore_opt", STAN_KID_OPT, jobj(["N"=>N,"P"=>3,"Xd"=>jmat(Xd),"kid_score"=>jvec(kid)]))
    ldf=stable_ldf(turing_kid(hs,iq,inter,kid); adtype=ADTypes.AutoMooncake(config=nothing))
    report("kidscore_interaction", q, qq->oracle_kid(qq,hs,iq,inter,kid), [("reference",smr),("optimized",smo)], ldf, kb, prep, similar(q); data_note="full N=$N")
end

# ================= blr / sblri (reference vec normal + optimized normal_id_glm; full D=5) =================
const BL = ReactiveKernelsPPLExamples.BLRExample
oracle_blr(q,X,y)=(D=size(X,2); beta=@view q[1:D]; sig=exp(q[D+1]); mu=X*beta;
    lp=sum(b-> -0.5*LOG2PI-log(10.0)-0.5*(b/10)^2,beta) + (-0.5*LOG2PI-log(10.0)-0.5*(sig/10)^2) + q[D+1];
    for i in eachindex(y); lp+=-0.5*LOG2PI-log(sig)-0.5*((y[i]-mu[i])/sig)^2; end; lp)
Turing.@model function turing_blr(X,y,D)
    beta~Distributions.MvNormal(zeros(D), Diagonal(fill(100.0,D)))
    sigma~Distributions.truncated(Distributions.Normal(0,10);lower=0)
    mu=X*beta; DynamicPPL.@addlogprob! sum(-0.5*LOG2PI .- log(sigma) .- 0.5.*((y.-mu)./sigma).^2)
end
let
    d=load_data("sblri"); N=Int(d["N"]); D=Int(d["D"]); X=reduce(vcat,[permutedims(Float64.(r)) for r in d["X"]]); y=fvec(d,"y")
    q=vcat(fill(0.1,D), log(0.6))
    kb=prepare(BL.build_blr_graph(); have=(:unconstrained,:predictors,:responses), want=:posterior, bound=(predictors=X,responses=y))
    prep=prepare_ad(kb, AE, q; active=:unconstrained)
    smr=stanA("blr", jobj(["N"=>N,"D"=>D,"X"=>jmat(X),"y"=>jvec(y)]))
    smo=stanS("blr_opt", STAN_BLR_OPT, jobj(["N"=>N,"D"=>D,"X"=>jmat(X),"y"=>jvec(y)]))
    ldf=stable_ldf(turing_blr(X,y,D); adtype=ADTypes.AutoMooncake(config=nothing))
    report("blr_sblri", q, qq->oracle_blr(qq,X,y), [("reference",smr),("optimized",smo)], ldf, kb, prep, similar(q); data_note="full N=$N D=$D")
end

# ================= GLMM_Poisson (peregrine; vectorized poisson_log = reference = optimum; support boundary) =================
const GM = ReactiveKernelsPPLExamples.GLMMPoissonExample
function oracle_glmm(q,year,C,n)
    a=-20+40*logistic(q[1]); b1=-10+20*logistic(q[2]); b2=-10+30*logistic(q[3]); b3=-10+20*logistic(q[4])
    eps=@view q[5:4+n]; sig=5*logistic(q[5+n])
    lp=(log(40.0)-log1pexp(-q[1])-log1pexp(q[1]))+(log(20.0)-log1pexp(-q[2])-log1pexp(q[2]))+(log(30.0)-log1pexp(-q[3])-log1pexp(q[3]))+(log(20.0)-log1pexp(-q[4])-log1pexp(q[4]))+(log(5.0)-log1pexp(-q[5+n])-log1pexp(q[5+n]))
    for i in 1:n; z=a+b1*year[i]+b2*year[i]^2+b3*year[i]^3+eps[i]; lp+=C[i]*z-exp(z); end
    lp += sum(-0.5*LOG2PI-log(sig)-0.5*(eps[i]/sig)^2 for i in 1:n); lp
end
Turing.@model function turing_glmm(year,C,n)
    alpha~Distributions.Uniform(-20,20); beta1~Distributions.Uniform(-10,10); beta2~Distributions.Uniform(-10,20); beta3~Distributions.Uniform(-10,10)
    sigma~Distributions.Uniform(0,5); eps~Distributions.MvNormal(zeros(n), sigma^2*I)
    DynamicPPL.@addlogprob! (beta2<=10.0 ? log(30.0)-log(20.0) : -Inf)
    z=alpha .+ beta1.*year .+ beta2.*year.^2 .+ beta3.*year.^3 .+ eps
    DynamicPPL.@addlogprob! sum(C.*z .- exp.(z))
end
let
    d=load_data("GLMM_Poisson_data"); year=fvec(d,"year"); C=ivec(d,"C"); n=length(year)
    q=vcat([0.2,0.1,-0.05,0.03], fill(0.0,n), [0.0]); tperm=vcat(1:4, 5+n, 5:4+n); qb=copy(q); qb[3]=5.0
    kb=prepare(GM.build_glmm_poisson_graph(); have=(:unconstrained,:year,:counts), want=:posterior, bound=(year=year,counts=C))
    prep=prepare_ad(kb, AE, q; active=:unconstrained)
    sm=stanA("GLMM_Poisson_model", jobj(["n"=>n,"C"=>jvec(C),"year"=>jvec(year)]))
    ldf=stable_ldf(turing_glmm(year,C,n); adtype=ADTypes.AutoMooncake(config=nothing))
    gref=fd_grad(q->oracle_glmm(q,year,C,n),q); gsc=maximum(abs,gref); gerr(gg)=maximum(abs,gg.-gref)/gsc
    vr=kb(q); vs=BridgeStan.log_density(sm,q;propto=false,jacobian=true); vdiff=abs(vr-vs)
    es=gerr(BridgeStan.log_density_gradient(sm,q;propto=false,jacobian=true)[2])
    tq=q[tperm]; tgt=DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,tq)[2]; tg=similar(tgt); tg[tperm]=tgt; et=gerr(tg)
    gbuf=similar(q); er=gerr(ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q)[2])
    vrb=kb(qb); vsb=BridgeStan.log_density(sm,qb;propto=false,jacobian=true)
    println("\n===== GLMM_Poisson  dim=$(length(q))  full n=$n  VALUE |rk-stan|=$(round(vdiff;sigdigits=3))  grad stan/turing/rk=$(round(es;sigdigits=2))/$(round(et;sigdigits=2))/$(round(er;sigdigits=2))  boundary(beta2>10) rk=$vrb stan=$vsb =====")
    (vdiff<1e-3 && es<2e-3 && et<2e-3 && er<2e-3) || error("GLMM value/grad parity FAIL")
    (vrb==-Inf && vsb==-Inf) || error("GLMM boundary FAIL")
    meas=Dict{String,Any}[]
    for (kind,fs) in ("primal"=>[("stan_stan",()->BridgeStan.log_density(sm,q;propto=false,jacobian=true)),("turing",()->DynamicPPL.LogDensityProblems.logdensity(ldf,tq)),("rk",()->kb(q))],
                      "gradient"=>[("stan_stan",()->BridgeStan.log_density_gradient(sm,q;propto=false,jacobian=true)),("turing",()->DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,tq)),("rk",()->ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q))])
        for (nm,f) in fs; r=bench(f); push!(meas,Dict("kind"=>kind,"impl"=>nm,"median_ns"=>r.median_ns,"median_bytes"=>r.median_bytes))
            println("  $(rpad(kind*"/"*nm,22)) $(lpad(fmt(r.median_ns),10))  $(r.median_bytes)B"); end
    end
    push!(results,Dict("model"=>"GLMM_Poisson","dim"=>length(q),"measurements"=>meas,"value_reldiff_rk_stan"=>vdiff,"beta2_boundary_both_neginf"=>(vrb==-Inf&&vsb==-Inf),"rk_builtin"=>true,"data_note"=>"full n=$n"))
end

# ================= arK (reference naive double-loop + optimized normal_id_glm; full T=200) =================
const AK = ReactiveKernelsPPLExamples.ARKExample
function ark_lag(y,K); T=length(y); yl=zeros(T-K,K); for i in 1:(T-K),k in 1:K; yl[i,k]=y[K+i-k]; end; yl; end
oracle_ark(q,y,K)=(T=length(y); a=q[1]; beta=@view q[2:K+1]; sig=exp(q[K+2]);
    lp=-0.5*LOG2PI-log(10.0)-0.5*(a/10)^2 + sum(b-> -0.5*LOG2PI-log(10.0)-0.5*(b/10)^2,beta) + (-log(π*2.5)-log1p((sig/2.5)^2)) + q[K+2];
    for t in (K+1):T; mu=a; for k in 1:K; mu+=beta[k]*y[t-k]; end; lp+=-0.5*LOG2PI-log(sig)-0.5*((y[t]-mu)/sig)^2; end; lp)
Turing.@model function turing_ark(y,K,T)
    alpha~Distributions.Normal(0,10); beta~Distributions.MvNormal(zeros(K),Diagonal(fill(100.0,K))); sigma~Distributions.truncated(Distributions.Cauchy(0,2.5);lower=0)
    s=0.0; for t in (K+1):T; mu=alpha; for k in 1:K; mu+=beta[k]*y[t-k]; end; s+= -0.5*LOG2PI-log(sigma)-0.5*((y[t]-mu)/sigma)^2; end
    DynamicPPL.@addlogprob! s
end
let
    d=load_data("arK"); y=fvec(d,"y"); K=Int(d["K"]); T=length(y); q=[0.3,0.5,-0.2,0.15,0.1,-0.05,log(0.4)]
    ylag=ark_lag(y,K); yt=y[(K+1):end]
    kb=prepare(AK.build_ark_graph(); have=(:unconstrained,:ylag,:yt), want=:posterior, bound=(ylag=ylag,yt=yt))
    prep=prepare_ad(kb, AE, q; active=:unconstrained)
    smr=stanA("arK", jobj(["K"=>K,"T"=>T,"y"=>jvec(y)]))
    smo=stanS("arK_opt", STAN_ARK_OPT, jobj(["R"=>size(ylag,1),"K"=>K,"Ylag"=>jmat(ylag),"yt"=>jvec(yt)]))
    ldf=stable_ldf(turing_ark(y,K,T); adtype=ADTypes.AutoMooncake(config=nothing))
    report("arK", q, qq->oracle_ark(qq,y,K), [("reference",smr),("optimized",smo)], ldf, kb, prep, similar(q); data_note="full T=$T K=$K")
end

# ================= Mh (capture-recapture; log_sum_exp loop = reference = optimum; full M=385) =================
const MH = ReactiveKernelsPPLExamples.MhExample
lchoose_mh(N,c)=0.0  # binomial normalizer folds into data-only prefix; Mh y in {0..T}
oracle_mh(q,y,M,T)=(om=logistic(q[1]);mp=logistic(q[2]);sig=5*logistic(q[3]); er=@view q[4:3+M];
    lp=ijac(q[1],1)+ijac(q[2],1)+ijac(q[3],5) + sum(-0.5*LOG2PI-0.5*er[i]^2 for i in 1:M);
    lo=log(om);l1=log1p(-om);lmp=logit(mp);
    for i in 1:M; e=lmp+sig*er[i]; bl=-T*log1pexp(e); lp+= y[i]>0 ? (lo+y[i]*e+bl) : logaddexp(lo+bl,l1); end; lp)
Turing.@model function turing_mh(y,M,T)
    omega~Distributions.Uniform(0,1); mean_p~Distributions.Uniform(0,1); sigma~Distributions.Uniform(0,5); eps_raw~Distributions.MvNormal(zeros(M), I)
    lo=log(omega);l1=log1p(-omega);lmp=logit(mean_p); s=0.0
    for i in 1:M; e=lmp+sigma*eps_raw[i]; bl=-T*log1pexp(e); s+= y[i]>0 ? (lo+y[i]*e+bl) : logaddexp(lo+bl,l1); end
    DynamicPPL.@addlogprob! s
end
let
    d=load_data("Mh_data"); y=ivec(d,"y"); M=Int(d["M"]); T=Int(d["T"]); q=vcat(zeros(3), zeros(M))
    lc=[Float64(Base.binomial(T, yi)) > 0 ? log(Float64(Base.binomial(T, yi))) : 0.0 for yi in y]
    kb=prepare(MH.build_mh_graph(); have=(:unconstrained,:y,:lchoose,:T,:M), want=:posterior, bound=(y=y,lchoose=lc,T=T,M=M))
    prep=prepare_ad(kb, AE, q; active=:unconstrained)
    sm=stanA("Mh_model", jobj(["M"=>M,"T"=>T,"y"=>jvec(y)]))
    ldf=stable_ldf(turing_mh(y,M,T); adtype=ADTypes.AutoMooncake(config=nothing))
    report("Mh", q, qq->oracle_mh(qq,y,M,T), [("stan",sm)], ldf, kb, prep, similar(q); data_note="full M=$M T=$T")
end

# ================= Rate_1 (beta-binomial; trivial; full n=10) =================
const R1 = ReactiveKernelsPPLExamples.Rate1Example
oracle_rate1(q,n,k)=(th=logistic(q[1]); ijac(q[1],1) + k*log(th) + (n-k)*log1p(-th))
Turing.@model function turing_rate1(n,k); theta~Distributions.Beta(1,1); k~Distributions.Binomial(n,theta); end
let
    d=load_data("Rate_1_data"); n=Int(d["n"]); k=Int(d["k"]); q=[0.1]
    kb=prepare(R1.build_rate_1_graph(); have=(:unconstrained,:n,:k), want=:posterior, bound=(n=n,k=k))
    prep=prepare_ad(kb, AutoEnzyme(mode=Enzyme.Reverse,function_annotation=Enzyme.Const), q; active=:unconstrained)
    sm=stanA("Rate_1_model", jobj(["n"=>n,"k"=>k]))
    ldf=stable_ldf(turing_rate1(n,k); adtype=ADTypes.AutoMooncake(config=nothing))
    report("Rate_1", q, qq->oracle_rate1(qq,n,k), [("stan",sm)], ldf, kb, prep, similar(q); data_note="full n=$n k=$k")
end

outp=get(ENV,"AS_OUTPUT","")
if outp!=""
    receipt=Dict("schema"=>"faithful-posteriordb-v1","generated_at"=>string(now()),
        "status"=>"canonical (RK consumes canonical ReactiveKernelsPPLExamples modules; FULL real posteriordb data)",
        "posteriordb_source_main"=>"3399636c47ef2721a7f5ddca2a5cab7a54ff13b4",
        "methodology"=>"Idiomatic RK-builtin distribution-kernel graphs vs Stan (posteriordb REFERENCE .stan verbatim from the artifact, AND a hand-written OPTIMIZED .stan — fused poisson_log_glm/normal_id_glm, vectorized — where a faster idiom exists and computes the identical density) vs optimized Turing. ALL sides the SAME faithful model, SAME real priors/support/Jacobians (improper flat where the .stan is flat), and the SAME FULL real posteriordb data (loaded from the artifact). Native primal+gradient at the fair boundary; median over $ROUNDS rounds of minimum(run(@benchmarkable;samples=200,seconds=0.2)); every side parity-gated vs an independent central FD oracle before timing (mismatch = hard failure). eight_schools carries a 4th 'rk_ppl' side (experimental @ppl front-end, documentation-only migration evidence).",
        "environment"=>Dict("julia"=>string(VERSION),"arch"=>string(Sys.ARCH),"cpu"=>Sys.cpu_info()[1].model,"stan"=>"2.39.0 (BridgeStan 2.9.0)"),
        "stan_artifact_dir"=>ART,"stan_shas"=>stan_shas,"models"=>results)
    open(outp,"w") do io; TOML.print(io,receipt); end; println("\nwrote receipt: $outp")
end
println("\nFAITHFUL_POSTERIORDB_DONE")
