# Fair benchmark: gp_regr (Gaussian-process regression; structured covariance + cholesky, dim 3).
# q=[log_rho, log_alpha, log_sigma]. RK vs optimized Stan (multi_normal_cholesky) vs optimized Turing.
using Random, LinearAlgebra, Statistics, SpecialFunctions, LogExpFunctions
using BenchmarkTools
using ReactiveKernels
import BridgeStan
import Turing, DynamicPPL, Distributions
import ADTypes, Mooncake
using DifferentiationInterface
import Enzyme
const LOG2PI=log(2π); const ROUNDS=10
nlp(x,m,s)= -0.5*LOG2PI - log(s) - 0.5*((x-m)/s)^2
gammalp(x,a,b)= a*log(b) - loggamma(a) + (a-1)*log(x) - b*x
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

function gen_gp(N;seed=20260906)
    rng=MersenneTwister(seed); x=sort(rand(rng,N).*4 .- 2); D2=[(x[i]-x[j])^2 for i in 1:N, j in 1:N]
    K=1.0.*exp.(-0.5.*D2./0.8^2) .+ 0.05.*Matrix(I,N,N); L=cholesky(Symmetric(K)).L; y=L*randn(rng,N); x,y,D2
end
# covariance from squared-distance matrix D2 (data, precomputed)
gp_cov(D2,alpha,rho,sigma,N)= alpha^2 .* exp.(-0.5.*D2./rho^2) .+ sigma.*Matrix(I,N,N)
function gp_ll(K,y); C=cholesky(Symmetric(K)); z=C.L\y; N=length(y)
    -0.5*N*LOG2PI - sum(log, diag(C.L)) - 0.5*dot(z,z)
end
function oracle_gp(q,D2,y,N); lr=q[1]; la=q[2]; ls=q[3]; rho=exp(lr); alpha=exp(la); sigma=exp(ls)
    prior=gammalp(rho,25,4)+nlp(alpha,0,2)+nlp(sigma,0,1)+lr+la+ls
    K=gp_cov(D2,alpha,rho,sigma,N); prior+gp_ll(K,y)
end
const STAN_GP=raw"""
data{int<lower=1> N;array[N] real x;vector[N] y;}
parameters{real<lower=0> rho;real<lower=0> alpha;real<lower=0> sigma;}
model{matrix[N,N] cov=gp_exp_quad_cov(x,alpha,rho)+diag_matrix(rep_vector(sigma,N));
 matrix[N,N] L=cholesky_decompose(cov);
 rho~gamma(25,4); alpha~normal(0,2); sigma~normal(0,1);
 y~multi_normal_cholesky(rep_vector(0,N),L);}
"""
Turing.@model function turing_gp(x,y,N,D2)
    rho~Distributions.Gamma(25,1/4); alpha~Distributions.truncated(Distributions.Normal(0,2);lower=0)
    sigma~Distributions.truncated(Distributions.Normal(0,1);lower=0)
    K=alpha^2 .* exp.(-0.5.*D2./rho^2) .+ sigma.*Matrix(I,N,N)
    y~Distributions.MvNormal(zeros(N), Symmetric(K))
end
gp_lr(q)=q[1]; gp_la(q)=q[2]; gp_ls(q)=q[3]
gp_rho(lr)=exp(lr); gp_al(la)=exp(la); gp_sig(ls)=exp(ls)
gp_prior(rho,al,sig,lr,la,ls)=gammalp(rho,25,4)+nlp(al,0,2)+nlp(sig,0,1)+lr+la+ls
gp_post(p,l)=p+l
rk_gp=@kernel model(unconstrained,D2,y,N)=begin
    lr=gp_lr(unconstrained); la=gp_la(unconstrained); ls=gp_ls(unconstrained)
    rho=gp_rho(lr); al=gp_al(la); sig=gp_sig(ls)
    K=gp_cov(D2,al,rho,sig,N); prior=gp_prior(rho,al,sig,lr,la,ls)
    likelihood=gp_ll(K,y); posterior=gp_post(prior,likelihood)
    return posterior
end
let
    N=30; x,y,D2=gen_gp(N); q=[log(0.8),log(1.0),log(0.05)]
    ref=oracle_gp(q,D2,y,N); gref=fd_grad(qq->oracle_gp(qq,D2,y,N),q); gsc=maximum(abs,gref)
    dj=jobj(["N"=>N,"x"=>jvec(x),"y"=>jvec(y)])
    sm=stan_model(STAN_GP,dj,"gp")
    ldf=stable_ldf(turing_gp(x,y,N,D2);adtype=ADTypes.AutoMooncake(config=nothing))
    kb=prepare(rk_gp;have=(:unconstrained,:D2,:y,:N),want=:posterior,bound=(;D2=D2,y=y,N=N))
    prep=prepare_ad(kb,AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse),function_annotation=Enzyme.Const),q;active=:unconstrained)
    gbuf=similar(q); gerr(g)=maximum(abs,g.-gref)/gsc
    println("stan grad err ", gerr(BridgeStan.log_density_gradient(sm,q;propto=false,jacobian=true)[2]))
    println("turing grad err ", gerr(DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,q)[2]))
    println("rk grad err ", gerr(ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q)[2]))
    println("\n===== gp_regr (GP, N=$N, dim 3) =====")
    for (nm,f) in (("stan_primal",()->BridgeStan.log_density(sm,q;propto=false,jacobian=true)),
                   ("turing_primal",()->DynamicPPL.LogDensityProblems.logdensity(ldf,q)),
                   ("rk_primal",()->kb(q)),
                   ("stan_grad",()->BridgeStan.log_density_gradient(sm,q;propto=false,jacobian=true)),
                   ("turing_grad",()->DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,q)),
                   ("rk_grad",()->ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q)))
        r=bench(f); println("  $(rpad(nm,14)) $(fmt(r.median_ns))  $(r.median_bytes)B")
    end
end
println("\nGP_DONE")
