# Staging validator body. For each real posteriordb model:
#  (1) oracle-vs-real-Stan: embed the actual .stan + an independent plain-Julia FD
#      oracle (Stan unconstrained param order); assert oracle grad == BridgeStan grad.
#  (2) oracle-vs-Turing: a faithful Turing translation; assert its grad (mapped to
#      canonical order) == the oracle grad.
# Together these validate the parity ANCHOR + two of the three benchmark sides
# before the RK graphs land. propto=false Stan keeps constants the oracle/Turing
# priors also carry or drop consistently; GRADIENTS match (what we gate).
import BridgeStan
import Turing, DynamicPPL, Distributions
import ADTypes, Mooncake
using DifferentiationInterface
using LinearAlgebra
const LOG2PI = log(2π)
logistic(u)=1/(1+exp(-u))
log1pexp(x)= x>0 ? x+log1p(exp(-x)) : log1p(exp(x))
logit(p)=log(p)-log1p(-p)
logaddexp(a,b)=(m=max(a,b); m+log1p(exp(-abs(a-b))))
ijac(u,w)=log(w) - log1pexp(-u) - log1pexp(u)
jnum(x::Integer)=string(x); jnum(x::Real)=(v=Float64(x); isinteger(v) ? string(Int(v))*".0" : repr(v))
jvec(v)="["*join(jnum.(v),",")*"]"; jmat(M)="["*join([jvec(view(M,i,:)) for i in 1:size(M,1)],",")*"]"
jobj(p)="{"*join(["\"$k\":$v" for (k,v) in p],",")*"}"
stan_model(src,dj,tag)=(d=mktempdir();f=joinpath(d,"$tag.stan");write(f,src);BridgeStan.StanModel(f,dj))
fd_grad(f,q)=(g=similar(q);h=1e-6; for i in eachindex(q); qp=copy(q);qp[i]+=h;qm=copy(q);qm[i]-=h; g[i]=(f(qp)-f(qm))/(2h); end; g)
function stable_ldf(model;adtype=nothing,logdensity=DynamicPPL.getlogjoint_internal)
    vi=DynamicPPL.OnlyAccsVarInfo(DynamicPPL.VectorValueAccumulator())
    _,vi=DynamicPPL.init!!(model,vi,DynamicPPL.InitFromUniform(-2.1,-1.9),DynamicPPL.LinkAll())
    fvi=DynamicPPL.OnlyAccsVarInfo(DynamicPPL.FixedTransformAccumulator())
    _,fvi=DynamicPPL.init!!(model,fvi,DynamicPPL.InitFromUniform(-2.1,-1.9),DynamicPPL.LinkAll())
    tr=DynamicPPL.get_fixed_transforms(fvi); vv=DynamicPPL.getacc(vi,Val(:VectorValue)).values
    vv=DynamicPPL.update_transforms!!(vv,tr); DynamicPPL.LogDensityFunction(model,logdensity,vv;adtype,fix_transforms=false)
end
nyears=40; year=collect(range(-1.9,1.9;length=nyears))
schecks=Tuple{String,Float64,Int}[]; tchecks=Tuple{String,Float64,Int}[]
function vstan(name,src,dj,oracle,q)
    try
        sm=stan_model(src,dj,name); gs=BridgeStan.log_density_gradient(sm,q;propto=false,jacobian=true)[2]
        go=fd_grad(oracle,q); re=maximum(abs,gs.-go)/maximum(abs,go); push!(schecks,(name,re,length(q)))
        println("  [stan]   $(rpad(name,16)) dim=$(lpad(length(q),4))  rel-err=$(round(re;sigdigits=3))  $(re<2e-3 ? "OK" : "*** MISMATCH ***")")
    catch e; push!(schecks,(name,NaN,length(q))); println("  [stan]   $(rpad(name,16)) ERROR: $(replace(sprint(showerror,e)[1:min(end,180)],'\n'=>' '))"); end
end
function vturing(name,mkldf,oracle,q;tperm=collect(1:length(q)))
    try
        ldf=mkldf(); tq=q[tperm]; tgt=DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,tq)[2]
        tg=similar(tgt); tg[tperm]=tgt
        go=fd_grad(oracle,q); re=maximum(abs,tg.-go)/maximum(abs,go); push!(tchecks,(name,re,length(q)))
        println("  [turing] $(rpad(name,16)) dim=$(lpad(length(q),4))  rel-err=$(round(re;sigdigits=3))  $(re<2e-3 ? "OK" : "*** MISMATCH ***")")
    catch e; push!(tchecks,(name,NaN,length(q))); println("  [turing] $(rpad(name,16)) ERROR: $(replace(sprint(showerror,e)[1:min(end,180)],'\n'=>' '))"); end
end
mc()=ADTypes.AutoMooncake(config=nothing)

# ---------- GLM_Poisson ----------
const STAN_GLM_POISSON = raw"""
data { int<lower=0> n; array[n] int<lower=0> C; vector[n] year; }
transformed data { vector[n] year_squared = square(year); vector[n] year_cubed = year_squared .* year; }
parameters { real<lower=-20,upper=20> alpha; real<lower=-10,upper=10> beta1; real<lower=-10,upper=10> beta2; real<lower=-10,upper=10> beta3; }
transformed parameters { vector[n] log_lambda = alpha + beta1*year + beta2*year_squared + beta3*year_cubed; }
model { C ~ poisson_log(log_lambda); }
"""
function oracle_glmpois(q,C)
    a=-20+40*logistic(q[1]); b1=-10+20*logistic(q[2]); b2=-10+20*logistic(q[3]); b3=-10+20*logistic(q[4])
    lp=ijac(q[1],40)+ijac(q[2],20)+ijac(q[3],20)+ijac(q[4],20)
    for i in eachindex(year); z=a+b1*year[i]+b2*year[i]^2+b3*year[i]^3; lp+=C[i]*z-exp(z); end; lp
end
Turing.@model function turing_glmpois(year,C)
    alpha~Distributions.Uniform(-20,20); beta1~Distributions.Uniform(-10,10); beta2~Distributions.Uniform(-10,10); beta3~Distributions.Uniform(-10,10)
    z=alpha .+ beta1.*year .+ beta2.*year.^2 .+ beta3.*year.^3
    DynamicPPL.@addlogprob! sum(C.*z .- exp.(z))
end
let
    C=round.(Int,exp.(2 .+0.4 .*year .-0.2 .*year.^2 .+0.05 .*year.^3)); q=zeros(4)
    vstan("GLM_Poisson",STAN_GLM_POISSON,jobj(["n"=>nyears,"C"=>jvec(C),"year"=>jvec(year)]),q->oracle_glmpois(q,C),q)
    vturing("GLM_Poisson",()->stable_ldf(turing_glmpois(year,C);adtype=mc()),q->oracle_glmpois(q,C),q)
end

# ---------- GLM_Binomial ----------
const STAN_GLM_BINOMIAL = raw"""
data { int<lower=0> nyears; array[nyears] int<lower=0> C; array[nyears] int<lower=0> N; vector[nyears] year; }
transformed data { vector[nyears] year_squared = year .* year; }
parameters { real alpha; real beta1; real beta2; }
transformed parameters { vector[nyears] logit_p = alpha + beta1*year + beta2*year_squared; }
model { alpha ~ normal(0,100); beta1 ~ normal(0,100); beta2 ~ normal(0,100); C ~ binomial_logit(N, logit_p); }
"""
function oracle_glmbin(q,C,N)
    a=q[1];b1=q[2];b2=q[3]; lp=sum(p-> -0.5*LOG2PI-log(100.0)-0.5*(p/100)^2,(a,b1,b2))
    for i in eachindex(year); z=a+b1*year[i]+b2*year[i]^2; lp+=C[i]*z-N[i]*log1pexp(z); end; lp
end
Turing.@model function turing_glmbin(year,C,N)
    alpha~Distributions.Normal(0,100); beta1~Distributions.Normal(0,100); beta2~Distributions.Normal(0,100)
    z=alpha .+ beta1.*year .+ beta2.*year.^2
    DynamicPPL.@addlogprob! sum(C.*z .- N.*log1pexp.(z))
end
let
    N=fill(50,nyears); C=round.(Int,N.*logistic.(0.3 .+0.5 .*year .-0.2 .*year.^2)); q=zeros(3)
    vstan("GLM_Binomial",STAN_GLM_BINOMIAL,jobj(["nyears"=>nyears,"C"=>jvec(C),"N"=>jvec(N),"year"=>jvec(year)]),q->oracle_glmbin(q,C,N),q)
    vturing("GLM_Binomial",()->stable_ldf(turing_glmbin(year,C,N);adtype=mc()),q->oracle_glmbin(q,C,N),q)
end

# ---------- GLMM_Poisson (Turing declares sigma before eps -> tperm swap) ----------
const STAN_GLMM_POISSON = raw"""
data { int<lower=0> n; array[n] int<lower=0> C; vector[n] year; }
transformed data { vector[n] year_squared = year .* year; vector[n] year_cubed = year .* year .* year; }
parameters { real<lower=-20,upper=20> alpha; real<lower=-10,upper=10> beta1; real<lower=-10,upper=10> beta2; real<lower=-10,upper=10> beta3; vector[n] eps; real<lower=0,upper=5> sigma; }
transformed parameters { vector[n] log_lambda = alpha + beta1*year + beta2*year_squared + beta3*year_cubed + eps; }
model { alpha~uniform(-20,20); beta1~uniform(-10,10); beta2~uniform(-10,10); beta3~uniform(-10,10); sigma~uniform(0,5); C ~ poisson_log(log_lambda); eps ~ normal(0,sigma); }
"""
function oracle_glmm(q,C,n)
    a=-20+40*logistic(q[1]);b1=-10+20*logistic(q[2]);b2=-10+20*logistic(q[3]);b3=-10+20*logistic(q[4])
    eps=@view q[5:4+n]; sig=5*logistic(q[5+n])
    lp=ijac(q[1],40)+ijac(q[2],20)+ijac(q[3],20)+ijac(q[4],20)+ijac(q[5+n],5)
    for i in 1:n; z=a+b1*year[i]+b2*year[i]^2+b3*year[i]^3+eps[i]; lp+=C[i]*z-exp(z); end
    lp += sum(-0.5*LOG2PI-log(sig)-0.5*(eps[i]/sig)^2 for i in 1:n); lp
end
Turing.@model function turing_glmm(year,C,n)
    alpha~Distributions.Uniform(-20,20); beta1~Distributions.Uniform(-10,10); beta2~Distributions.Uniform(-10,10); beta3~Distributions.Uniform(-10,10)
    sigma~Distributions.Uniform(0,5); eps~Distributions.MvNormal(zeros(n), sigma^2*I)
    z=alpha .+ beta1.*year .+ beta2.*year.^2 .+ beta3.*year.^3 .+ eps
    DynamicPPL.@addlogprob! sum(C.*z .- exp.(z))
end
let
    C=round.(Int,exp.(2 .+0.4 .*year .-0.2 .*year.^2 .+0.05 .*year.^3)); n=nyears; q=vcat(zeros(4),zeros(n),[0.0])
    vstan("GLMM_Poisson",STAN_GLMM_POISSON,jobj(["n"=>n,"C"=>jvec(C),"year"=>jvec(year)]),q->oracle_glmm(q,C,n),q)
    vturing("GLMM_Poisson",()->stable_ldf(turing_glmm(year,C,n);adtype=mc()),q->oracle_glmm(q,C,n),q; tperm=vcat(1:4,5+n,5:4+n))
end

# ---------- Rate_1 ----------
const STAN_RATE1 = raw"""
data { int<lower=1> n; int<lower=0> k; } parameters { real<lower=0,upper=1> theta; } model { theta ~ beta(1,1); k ~ binomial(n, theta); }
"""
oracle_rate1(q,n,k)=(th=logistic(q[1]); ijac(q[1],1)+k*log(th)+(n-k)*log1p(-th))
Turing.@model function turing_rate1(n,k); theta~Distributions.Beta(1,1); k~Distributions.Binomial(n,theta); end
let; q=zeros(1)
    vstan("Rate_1",STAN_RATE1,jobj(["n"=>100,"k"=>40]),q->oracle_rate1(q,100,40),q)
    vturing("Rate_1",()->stable_ldf(turing_rate1(100,40);adtype=mc()),q->oracle_rate1(q,100,40),q)
end

# ---------- blr / sblri ----------
const STAN_BLR = raw"""
data { int<lower=0> N; int<lower=0> D; matrix[N,D] X; vector[N] y; }
parameters { vector[D] beta; real<lower=0> sigma; }
model { target += normal_lpdf(beta | 0, 10); target += normal_lpdf(sigma | 0, 10); target += normal_lpdf(y | X*beta, sigma); }
"""
function oracle_blr(q,X,y,D)
    beta=@view q[1:D]; sig=exp(q[D+1]); mu=X*beta
    lp=sum(b-> -0.5*LOG2PI-log(10.0)-0.5*(b/10)^2,beta) + (-0.5*LOG2PI-log(10.0)-0.5*(sig/10)^2) + q[D+1]
    lp += sum(-0.5*LOG2PI-log(sig)-0.5*((y[i]-mu[i])/sig)^2 for i in eachindex(y)); lp
end
Turing.@model function turing_blr(X,y,D)
    beta~Distributions.MvNormal(zeros(D), Diagonal(fill(100.0,D)))
    sigma~Distributions.truncated(Distributions.Normal(0,10);lower=0)
    mu=X*beta; DynamicPPL.@addlogprob! sum(-0.5*LOG2PI .- log(sigma) .- 0.5.*((y.-mu)./sigma).^2)
end
let
    N=200; D=5; X=[sin(i*0.3+j) for i in 1:N, j in 1:D]; y=X*[0.5,-0.3,0.2,0.1,-0.15] .+0.7 .*[cos(i*0.7) for i in 1:N]; q=vcat(zeros(D),[0.0])
    vstan("blr_sblri",STAN_BLR,jobj(["N"=>N,"D"=>D,"X"=>jmat(X),"y"=>jvec(y)]),q->oracle_blr(q,X,y,D),q)
    vturing("blr_sblri",()->stable_ldf(turing_blr(X,y,D);adtype=mc()),q->oracle_blr(q,X,y,D),q)
end

# ---------- kidscore_interaction (flat beta) ----------
const STAN_KIDSCORE = raw"""
data { int<lower=0> N; vector<lower=0,upper=200>[N] kid_score; vector<lower=0,upper=200>[N] mom_iq; vector<lower=0,upper=1>[N] mom_hs; }
transformed data { vector[N] inter = mom_hs .* mom_iq; }
parameters { vector[4] beta; real<lower=0> sigma; }
model { sigma ~ cauchy(0, 2.5); kid_score ~ normal(beta[1] + beta[2]*mom_hs + beta[3]*mom_iq + beta[4]*inter, sigma); }
"""
function oracle_kid(q,hs,iq,inter,kid)
    b=@view q[1:4]; sig=exp(q[5]); lp = -log(π*2.5)-log1p((sig/2.5)^2) + q[5]
    mu=b[1] .+ b[2].*hs .+ b[3].*iq .+ b[4].*inter
    lp += sum(-0.5*LOG2PI-log(sig)-0.5*((kid[i]-mu[i])/sig)^2 for i in eachindex(kid)); lp
end
Turing.@model function turing_kid(hs,iq,inter,kid)
    beta~Distributions.MvNormal(zeros(4), Diagonal(fill(1e10,4)))
    sigma~Distributions.truncated(Distributions.Cauchy(0,2.5);lower=0)
    mu=beta[1] .+ beta[2].*hs .+ beta[3].*iq .+ beta[4].*inter
    DynamicPPL.@addlogprob! sum(-0.5*LOG2PI .- log(sigma) .- 0.5.*((kid.-mu)./sigma).^2)
end
let
    N=200; hs=Float64.([isodd(i) for i in 1:N]); iq=80 .+30 .*[abs(sin(i*0.4)) for i in 1:N]; inter=hs.*iq
    kid=clamp.(40 .+6 .*hs .+0.5 .*iq .+10 .*[cos(i*0.6) for i in 1:N],1.0,199.0); q=zeros(5)
    vstan("kidscore_inter",STAN_KIDSCORE,jobj(["N"=>N,"kid_score"=>jvec(kid),"mom_iq"=>jvec(iq),"mom_hs"=>jvec(hs)]),q->oracle_kid(q,hs,iq,inter,kid),q)
    vturing("kidscore_inter",()->stable_ldf(turing_kid(hs,iq,inter,kid);adtype=mc()),q->oracle_kid(q,hs,iq,inter,kid),q)
end

# ---------- arK ----------
const STAN_ARK = raw"""
data { int<lower=0> K; int<lower=0> T; array[T] real y; }
parameters { real alpha; array[K] real beta; real<lower=0> sigma; }
model { alpha~normal(0,10); beta~normal(0,10); sigma~cauchy(0,2.5);
  for (t in (K+1):T) { real mu=alpha; for (k in 1:K) mu+=beta[k]*y[t-k]; y[t]~normal(mu,sigma); } }
"""
function oracle_ark(q,y,K,T)
    a=q[1]; beta=@view q[2:K+1]; sig=exp(q[K+2])
    lp = -0.5*LOG2PI-log(10.0)-0.5*(a/10)^2 + sum(b-> -0.5*LOG2PI-log(10.0)-0.5*(b/10)^2,beta) + (-log(π*2.5)-log1p((sig/2.5)^2)) + q[K+2]
    for t in (K+1):T; mu=a; for k in 1:K; mu+=beta[k]*y[t-k]; end; lp += -0.5*LOG2PI-log(sig)-0.5*((y[t]-mu)/sig)^2; end; lp
end
Turing.@model function turing_ark(y,K,T)
    alpha~Distributions.Normal(0,10); beta~Distributions.MvNormal(zeros(K), Diagonal(fill(100.0,K))); sigma~Distributions.truncated(Distributions.Cauchy(0,2.5);lower=0)
    s=0.0; for t in (K+1):T; mu=alpha; for k in 1:K; mu+=beta[k]*y[t-k]; end; s += -0.5*LOG2PI-log(sigma)-0.5*((y[t]-mu)/sigma)^2; end
    DynamicPPL.@addlogprob! s
end
let
    K=5; T=100; phi=[0.3,-0.1,0.05,0.02,-0.03]; y=zeros(T)
    for t in 1:T; m=0.2; for k in 1:K; t-k>=1 && (m+=phi[k]*y[t-k]); end; y[t]=m+0.5*sin(t*0.9); end; q=vcat([0.0],zeros(K),[0.0])
    vstan("arK",STAN_ARK,jobj(["K"=>K,"T"=>T,"y"=>jvec(y)]),q->oracle_ark(q,y,K,T),q)
    vturing("arK",()->stable_ldf(turing_ark(y,K,T);adtype=mc()),q->oracle_ark(q,y,K,T),q)
end

# ---------- gp_regr ----------
const STAN_GP = raw"""
data { int<lower=1> N; array[N] real x; vector[N] y; }
parameters { real<lower=0> rho; real<lower=0> alpha; real<lower=0> sigma; }
model { matrix[N,N] cov = gp_exp_quad_cov(x, alpha, rho) + diag_matrix(rep_vector(sigma, N));
  matrix[N,N] L = cholesky_decompose(cov); rho~gamma(25,4); alpha~normal(0,2); sigma~normal(0,1);
  y ~ multi_normal_cholesky(rep_vector(0,N), L); }
"""
function oracle_gp(q,D2,y,N)
    rho=exp(q[1]);al=exp(q[2]);sig=exp(q[3])
    lp = 24*log(rho)-4*rho + (-0.5*LOG2PI-log(2.0)-0.5*(al/2)^2) + (-0.5*LOG2PI-0.5*sig^2) + q[1]+q[2]+q[3]
    K=al^2 .* exp.(-0.5.*D2./rho^2) .+ sig.*Matrix(I,N,N); C=cholesky(Symmetric(K)); z=C.L\y
    lp - 0.5*N*LOG2PI - sum(log,diag(C.L)) - 0.5*dot(z,z)
end
Turing.@model function turing_gp(D2,y,N)
    rho~Distributions.Gamma(25,1/4); alpha~Distributions.truncated(Distributions.Normal(0,2);lower=0); sigma~Distributions.truncated(Distributions.Normal(0,1);lower=0)
    K=alpha^2 .* exp.(-0.5.*D2./rho^2) .+ sigma.*Matrix(I,N,N); C=cholesky(Symmetric(K)); z=C.L\y
    DynamicPPL.@addlogprob! (-0.5*N*LOG2PI - sum(log,diag(C.L)) - 0.5*dot(z,z))
end
let
    N=30; x=sort([2.0*(i/N)-1.0 for i in 1:N].*2); D2=[(x[i]-x[j])^2 for i in 1:N, j in 1:N]
    y=cholesky(Symmetric(1.0.*exp.(-0.5.*D2).+0.1.*Matrix(I,N,N))).L*[sin(i*1.3) for i in 1:N]; q=zeros(3)
    vstan("gp_regr",STAN_GP,jobj(["N"=>N,"x"=>jvec(x),"y"=>jvec(y)]),q->oracle_gp(q,D2,y,N),q)
    vturing("gp_regr",()->stable_ldf(turing_gp(D2,y,N);adtype=mc()),q->oracle_gp(q,D2,y,N),q)
end

# ---------- Mh (masked binomial-logit) ----------
const STAN_MH = raw"""
data { int<lower=0> M; int<lower=0> T; array[M] int<lower=0,upper=T> y; }
parameters { real<lower=0,upper=1> omega; real<lower=0,upper=1> mean_p; real<lower=0,upper=5> sigma; vector[M] eps_raw; }
transformed parameters { vector[M] eps = logit(mean_p) + sigma * eps_raw; }
model { eps_raw ~ normal(0,1);
  for (i in 1:M) {
    if (y[i] > 0) target += bernoulli_lpmf(1|omega) + binomial_logit_lpmf(y[i]|T, eps[i]);
    else target += log_sum_exp(bernoulli_lpmf(1|omega) + binomial_logit_lpmf(0|T, eps[i]), bernoulli_lpmf(0|omega)); } }
"""
function oracle_mh(q,y,M,T)
    om=logistic(q[1]);mp=logistic(q[2]);sig=5*logistic(q[3]); er=@view q[4:3+M]
    lp=ijac(q[1],1)+ijac(q[2],1)+ijac(q[3],5) + sum(-0.5*LOG2PI-0.5*er[i]^2 for i in 1:M)
    lo=log(om);l1=log1p(-om);lmp=logit(mp)
    for i in 1:M; e=lmp+sig*er[i]; bl=-T*log1pexp(e); lp += y[i]>0 ? (lo+y[i]*e+bl) : logaddexp(lo+bl,l1); end; lp
end
Turing.@model function turing_mh(y,M,T)
    omega~Distributions.Uniform(0,1); mean_p~Distributions.Uniform(0,1); sigma~Distributions.Uniform(0,5)
    eps_raw~Distributions.MvNormal(zeros(M), I)
    lo=log(omega);l1=log1p(-omega);lmp=logit(mean_p); s=0.0
    for i in 1:M; e=lmp+sigma*eps_raw[i]; bl=-T*log1pexp(e); s += y[i]>0 ? (lo+y[i]*e+bl) : logaddexp(lo+bl,l1); end
    DynamicPPL.@addlogprob! s
end
let
    M=100; T=7; y=zeros(Int,M); for i in 1:M; if isodd(i*7%5+i%3); y[i]=min(T,(i*3)%(T+1)); end; end; q=vcat(zeros(3),zeros(M))
    vstan("Mh",STAN_MH,jobj(["M"=>M,"T"=>T,"y"=>jvec(y)]),q->oracle_mh(q,y,M,T),q)
    vturing("Mh",()->stable_ldf(turing_mh(y,M,T);adtype=mc()),q->oracle_mh(q,y,M,T),q)
end

println("\n===== staging parity summary =====")
sbad=[c for c in schecks if isnan(c[2])||c[2]>=2e-3]; tbad=[c for c in tchecks if isnan(c[2])||c[2]>=2e-3]
println("Stan-vs-oracle:   $(isempty(sbad) ? "ALL $(length(schecks)) OK" : "MISMATCH $sbad")")
println("Turing-vs-oracle: $(isempty(tbad) ? "ALL $(length(tchecks)) OK" : "MISMATCH $tbad")")
println("\nFAITHFUL_STAGING_DONE")
