# Fair benchmark: kidscore_interaction (dim 5) + sblri-blr (dim 6) — the linear
# family, completing the 10 named posteriordb models. Conjugate sufficient-stat
# form (like the Gaussian case). RK vs optimized Stan vs optimized Turing.
using Random, LinearAlgebra, Statistics, SpecialFunctions, Dates
using BenchmarkTools
import TOML
using ReactiveKernels
import BridgeStan
import Turing, DynamicPPL, Distributions
import ADTypes, Mooncake
using DifferentiationInterface
import Enzyme
const LOG2PI=log(2π); const ROUNDS=12
nlp(x,m,s)= -0.5*LOG2PI - log(s) - 0.5*((x-m)/s)^2
jnum(x::Integer)=string(x); jnum(x::Real)=(v=Float64(x); isinteger(v) ? string(Int(v))*".0" : repr(v))
jvec(v)="["*join(jnum.(v),",")*"]"; jmat(M)="["*join([jvec(view(M,i,:)) for i in 1:size(M,1)],",")*"]"; jobj(p)="{"*join(["\"$k\":$v" for (k,v) in p],",")*"}"
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
function report(name,q,oracle,stanspecs,ldf,kb,prep,gbuf,optbase)
    ref=oracle(q); gref=fd_grad(oracle,q); gsc=maximum(abs,gref); gerr(g)=maximum(abs,g.-gref)/gsc
    for (nm,sm) in stanspecs; gerr(BridgeStan.log_density_gradient(sm,q;propto=false,jacobian=true)[2])<2e-3 || error("parity $name/$nm"); end
    gerr(DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,q)[2])<2e-3 || error("parity $name/turing")
    gerr(ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q)[2])<2e-3 || error("parity $name/rk")
    meas=Dict{String,Any}[]
    base=Dict(stanspecs)[optbase]
    for (kind,fs) in ("primal"=>vcat([(nm,()->BridgeStan.log_density(sm,q;propto=false,jacobian=true)) for (nm,sm) in stanspecs],
                                      [("turing",()->DynamicPPL.LogDensityProblems.logdensity(ldf,q)),("rk",()->kb(q))]),
                      "gradient"=>[("stan",()->BridgeStan.log_density_gradient(base,q;propto=false,jacobian=true)),
                                   ("turing",()->DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,q)),
                                   ("rk",()->ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q))])
        for (nm,f) in fs; r=bench(f); push!(meas,Dict("kind"=>kind,"impl"=>nm,"median_ns"=>r.median_ns,"median_bytes"=>r.median_bytes)); end
    end
    bp=first(m["median_ns"] for m in meas if m["kind"]=="primal"&&m["impl"]==optbase)
    bg=first(m["median_ns"] for m in meas if m["kind"]=="gradient"&&m["impl"]=="stan")
    println("\n===== $name  dim=$(length(q)) =====")
    for m in meas; b=m["kind"]=="primal" ? bp : bg
        println("  $(rpad(m["kind"]*"/"*m["impl"],16)) $(lpad(fmt(m["median_ns"]),10))  $(lpad(string(round(m["median_ns"]/b;digits=2))*"x",8))  $(m["median_bytes"])B"); end
    Dict("model"=>name,"dim"=>length(q),"measurements"=>meas)
end
results=Any[]

# conjugate Gaussian regression density (intercept + K coeffs). q=[alpha,beta(K),log_sigma].
rk_a(q)=q[1]; rk_b(q)=@view q[2:end-1]; rk_ls(q)=q[end]; rk_s(ls)=exp(ls)
rk_sse(a,b,XtX,Xty,xsum,ysum,yty,N)= yty - 2*a*ysum - 2*dot(b,Xty) + N*a^2 + 2*a*dot(xsum,b) + dot(b,XtX*b)
rk_like(sse,ls,s,N)= -0.5*N*LOG2PI - N*ls - 0.5*sse/s^2
rk_prior(q,s,ls,ps)=sum(x->nlp(x,0,ps), @view q[1:end-1]) + nlp(s,0,10) + ls
rk_post(p,l)=p+l
rk_lin=@kernel model(unconstrained,XtX,Xty,xsum,ysum,yty,N,ps)=begin
    a=rk_a(unconstrained); b=rk_b(unconstrained); ls=rk_ls(unconstrained); s=rk_s(ls)
    sse=rk_sse(a,b,XtX,Xty,xsum,ysum,yty,N); prior=rk_prior(unconstrained,s,ls,ps); likelihood=rk_like(sse,ls,s,N)
    posterior=rk_post(prior,likelihood); return posterior
end
oracle_lin(q,X,y,ps)=(K=size(X,2); a=q[1]; b=@view q[2:K+1]; ls=q[K+2]; s=exp(ls); mu=a.+X*b;
    sum(x->nlp(x,0,ps), @view q[1:K+1]) + nlp(s,0,10) + ls + sum(i->nlp(y[i],mu[i],s), eachindex(y)))
const STAN_LIN=raw"""
data{int<lower=0> N;int<lower=0> K;matrix[N,K] X;vector[N] y;real ps;}
parameters{real alpha;vector[K] beta;real<lower=0> sigma;}
model{target+=normal_lpdf(alpha|0,ps);target+=normal_lpdf(beta|0,ps);target+=normal_lpdf(sigma|0,10);
 target+=normal_id_glm_lpdf(y|X,alpha,beta,sigma);}
"""
Turing.@model function turing_lin(XtX,Xty,xsum,ysum,yty,N,K,ps)
    alpha~Distributions.Normal(0,ps); beta~Distributions.MvNormal(zeros(K),ps^2*I); sigma~Distributions.truncated(Distributions.Normal(0,10);lower=0)
    sse=yty-2*alpha*ysum-2*dot(beta,Xty)+N*alpha^2+2*alpha*dot(xsum,beta)+dot(beta,XtX*beta)
    DynamicPPL.@addlogprob! -0.5*N*LOG2PI - N*log(sigma) - 0.5*sse/sigma^2
end
function run_lin(name,X,y,ps)
    N,K=size(X); q=vcat([0.3],fill(0.1,K),[log(0.6)])
    XtX=X'X; Xty=X'y; xsum=vec(sum(X;dims=1)); ysum=sum(y); yty=dot(y,y)
    dj=jobj(["N"=>N,"K"=>K,"X"=>jmat(X),"y"=>jvec(y),"ps"=>ps])
    stans=["stan_glm"=>stan_model(STAN_LIN,dj,name)]
    ldf=stable_ldf(turing_lin(XtX,Xty,xsum,ysum,yty,N,K,ps);adtype=ADTypes.AutoMooncake(config=nothing))
    kb=prepare(rk_lin;have=(:unconstrained,:XtX,:Xty,:xsum,:ysum,:yty,:N,:ps),want=:posterior,bound=(;XtX=XtX,Xty=Xty,xsum=xsum,ysum=ysum,yty=yty,N=N,ps=ps))
    prep=prepare_ad(kb,AutoEnzyme(mode=Enzyme.Reverse,function_annotation=Enzyme.Const),q;active=:unconstrained)
    push!(results, report(name,q,qq->oracle_lin(qq,X,y,ps),stans,ldf,kb,prep,similar(q),"stan_glm"))
end
let  # kidscore_interaction: intercept + [mom_hs, mom_iq, inter]; N~434
    rng=MersenneTwister(1); N=434; mom_hs=Float64.(rand(rng,0:1,N)); mom_iq=80 .+ 30 .*randn(rng,N)
    X=hcat(mom_hs, mom_iq, mom_hs.*mom_iq); y=25 .+ 6 .*mom_hs .+ 0.5 .*mom_iq .+ 18 .*randn(rng,N)
    run_lin("kidscore_interaction", X, y, 100.0)
end
let  # sblri-blr: X (N×5)
    rng=MersenneTwister(2); N=200; X=randn(rng,N,5); y=X*[0.5,-0.3,0.2,0.1,-0.15] .+ 0.7 .*randn(rng,N)
    run_lin("sblri_blr", X, y, 10.0)
end
outp=get(ENV,"AS_OUTPUT","")
if outp!=""
    receipt=Dict("schema"=>"fair-posteriordb-linear-v1","generated_at"=>string(now()),
        "methodology"=>"median over $ROUNDS rounds; RK conjugate vs optimized Stan (normal_id_glm) vs optimized Turing; parity vs FD oracle",
        "environment"=>Dict("julia"=>string(VERSION),"arch"=>string(Sys.ARCH),"cpu"=>Sys.cpu_info()[1].model,"stan"=>"2.39.0 (BridgeStan 2.9.0)"),
        "models"=>results)
    open(outp,"w") do io; TOML.print(io,receipt); end; println("\nwrote receipt: $outp")
end
println("\nFAIR_LIN_DONE")
