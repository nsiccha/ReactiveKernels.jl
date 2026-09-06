# Per-model native-vs-Reactant HMC-loop table. Each model's density is wired into
# the same transpiler path as prepared_hmc.jl (transpiled_endpoint + prepare_transpiled),
# timed native and Reactant. Reports which of ALL 10 named posteriordb models lower
# to Reactant, and for those that do not, the exact reason. Densities are authored
# Reactant-friendly (1-element reductions instead of scalar q[i]; sum(a.*b) instead of
# dot; vectorized data-masks instead of per-element `if`) — the generic RK
# Reactant-lowering frictions. Models that still fail (dense in-graph cholesky) are
# reported honestly, not worked around.
import Reactant
using ReactiveKernels, LinearAlgebra, Random, Statistics
import Enzyme
using DifferentiationInterface
const STD = joinpath(@__DIR__, "sampler_transpiler")
const BENCH = dirname(STD)
include(joinpath(STD, "eight_schools_density.jl"))          # Potential, Gradient, CallbackHandle, build_density
include(joinpath(BENCH, "nuts_kernel_authoring_fixture_b.jl"))
include(joinpath(STD, "position_multinomial_hmc_kernel.jl"))
using .EightSchoolsDensity: Potential, Gradient, CallbackHandle, build_density
const F = NUTSBMutationAuthoringFixture
const LOG2PI=log(2π)
nlp(x,m,s)= -0.5*LOG2PI - log(s) - 0.5*((x-m)/s)^2
gammalp(x,a,b)= a*log(b) + (a-1)*log(x) - b*x   # loggamma(a) constant dropped (HMC-invariant offset)

# ---- model density builders: each returns (; density, ad, q) like build_density ----
_backend()=AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)
_prep(model, have, q, bound)=(d=prepare(model; have, want=:posterior, bound); (; density=d, ad=prepare_ad(d,_backend(),q;active=:unconstrained), q))

# ---------- Rate_1 (beta-binomial, dim 1) ----------
rate_th(q)=1/(1+exp(-sum(view(q,1:1))))
rate_post(q,y,N,a0,b0)=begin th=rate_th(q); lj=log(th)+log1p(-th)
    (a0-1)*log(th)+(b0-1)*log1p(-th) + y*log(th)+(N-y)*log1p(-th) + lj end
rk_rate=@kernel model(unconstrained,y,N,a0,b0)=begin posterior=rate_post(unconstrained,y,N,a0,b0); return posterior end
build_rate()= _prep(rk_rate,(:unconstrained,:y,:N,:a0,:b0),[0.0],(;y=40.0,N=100.0,a0=1.0,b0=1.0))

# ---------- GLM_Poisson (dim 4) ----------
pois_a(q)=sum(view(q,1:1)); pois_b(q)=@view q[2:end]; pois_eta(a,b,X)=a.+X*b; pois_prior(q)=sum(x->nlp(x,0,5), q)
pois_ll(C,eta,ct)=sum(C.*eta)-sum(exp,eta)-ct; pois_post(p,l)=p+l
rk_pois=@kernel model(unconstrained,X,C,cterm)=begin
    a=pois_a(unconstrained); b=pois_b(unconstrained); eta=pois_eta(a,b,X)
    prior=pois_prior(unconstrained); likelihood=pois_ll(C,eta,cterm); posterior=pois_post(prior,likelihood); return posterior
end
function build_pois()
    rng=MersenneTwister(1); n=100; yr=collect(range(-1.5,1.5;length=n)); X=hcat(yr,yr.^2,yr.^3)
    C=Float64.([rand(rng,0:5) for _ in 1:n]); ct=0.0; q=[0.7,0.3,-0.15,0.08]
    _prep(rk_pois, (:unconstrained,:X,:C,:cterm), q, (;X=X,C=C,cterm=ct))
end

# ---------- GLM_Binomial (dim 3) ----------
bin_a(q)=sum(view(q,1:1)); bin_b(q)=@view q[2:end]; bin_eta(a,b,X)=a.+X*b
bin_ll(y,ntr,eta)=sum(y.*eta) - sum(ntr.*log.(1 .+ exp.(eta)))
rk_bin=@kernel model(unconstrained,X,y,ntr)=begin
    a=bin_a(unconstrained); b=bin_b(unconstrained); eta=bin_eta(a,b,X)
    posterior = sum(x->nlp(x,0,5), unconstrained) + bin_ll(y,ntr,eta); return posterior
end
function build_bin()
    rng=MersenneTwister(2); n=100; yr=collect(range(-1.5,1.5;length=n)); X=hcat(yr,yr.^2)
    th=1 ./(1 .+exp.(-(0.2 .+X*[0.5,-0.3]))); ntr=fill(10.0,n)
    y=Float64.([rand(rng,0:10) < 10*th[i] ? rand(rng,0:10) : round(Int,10*th[i]) for i in 1:n])
    _prep(rk_bin,(:unconstrained,:X,:y,:ntr),[0.2,0.3,-0.2],(;X=X,y=y,ntr=ntr))
end

# ---------- GLMM_Poisson (dim 45) ----------
glmm_a(q)=sum(view(q,1:1)); glmm_bf(q)=@view q[2:4]; glmm_ls(q)=sum(view(q,5:5)); glmm_eps(q,n)=@view q[6:5+n]; glmm_s(ls)=exp(ls)
glmm_eta(a,bf,X,eps)=a.+X*bf.+eps
glmm_prior(q,s,ls)=sum(x->nlp(x,0,10), @view q[1:4]) + nlp(s,0,5) + ls
glmm_lleps(eps,ls,s,n)= -0.5*n*LOG2PI - n*ls - 0.5*sum(abs2,eps)/s^2
glmm_llpois(C,eta,ct)=sum(C.*eta)-sum(exp,eta)-ct; glmm_post(p,le,lp)=p+le+lp
rk_glmm=@kernel model(unconstrained,X,C,cterm,n)=begin
    a=glmm_a(unconstrained); bf=glmm_bf(unconstrained); eps=glmm_eps(unconstrained,n); ls=glmm_ls(unconstrained); s=glmm_s(ls)
    eta=glmm_eta(a,bf,X,eps); prior=glmm_prior(unconstrained,s,ls)
    ll_eps=glmm_lleps(eps,ls,s,n); ll_pois=glmm_llpois(C,eta,cterm); posterior=glmm_post(prior,ll_eps,ll_pois); return posterior
end
function build_glmm()
    rng=MersenneTwister(1); n=40; yr=collect(range(-1.5,1.5;length=n)); X=hcat(yr,yr.^2,yr.^3)
    C=Float64.([rand(rng,0:8) for _ in 1:n]); ct=0.0
    q=vcat([0.5,0.25,-0.1,0.06,log(0.4)], fill(0.0,n))
    _prep(rk_glmm, (:unconstrained,:X,:C,:cterm,:n), q, (;X=X,C=C,cterm=ct,n=n))
end

# ---------- linear conjugate family: kidscore_interaction (d5), sblri-blr (d7), arK (d7) ----------
lin_a(q)=sum(view(q,1:1)); lin_ls(q,D)=sum(view(q,D:D)); lin_b(q,D)=@view q[2:D-1]; lin_s(ls)=exp(ls)
lin_sse(a,b,XtX,Xty,xsum,ysum,yty,N)= yty - 2*a*ysum - 2*sum(b.*Xty) + N*a^2 + 2*a*sum(xsum.*b) + sum(b.*(XtX*b))
lin_like(sse,ls,s,N)= -0.5*N*LOG2PI - N*ls - 0.5*sse/s^2
lin_prior(q,s,ls,ps,D)= sum(x->nlp(x,0,ps), @view q[1:D-1]) + nlp(s,0,10) + ls
rk_lin=@kernel model(unconstrained,XtX,Xty,xsum,ysum,yty,N,ps,D)=begin
    a=lin_a(unconstrained); b=lin_b(unconstrained,D); ls=lin_ls(unconstrained,D); s=lin_s(ls)
    sse=lin_sse(a,b,XtX,Xty,xsum,ysum,yty,N); prior=lin_prior(unconstrained,s,ls,ps,D); like=lin_like(sse,ls,s,N)
    posterior=prior+like; return posterior
end
function build_lin(X,y,ps,q0)
    N=size(X,1); D=length(q0); XtX=X'X; Xty=X'y; xsum=vec(sum(X;dims=1)); ysum=sum(y); yty=dot(y,y)
    _prep(rk_lin,(:unconstrained,:XtX,:Xty,:xsum,:ysum,:yty,:N,:ps,:D),q0,
        (;XtX=XtX,Xty=Xty,xsum=xsum,ysum=ysum,yty=yty,N=Float64(N),ps=ps,D=D))
end
function build_kidscore()
    rng=MersenneTwister(1); N=434; hs=Float64.(rand(rng,0:1,N)); iq=80 .+30 .*randn(rng,N)
    X=hcat(hs,iq,hs.*iq); y=25 .+6 .*hs .+0.5 .*iq .+18 .*randn(rng,N)
    build_lin(X,y,100.0,vcat([0.3],fill(0.1,3),[log(0.6)]))
end
function build_sblri()
    rng=MersenneTwister(2); N=200; X=randn(rng,N,5); y=X*[0.5,-0.3,0.2,0.1,-0.15] .+0.7 .*randn(rng,N)
    build_lin(X,y,10.0,vcat([0.3],fill(0.1,5),[log(0.6)]))
end
function build_ark()
    rng=MersenneTwister(3); T=200; phi=[0.3,-0.1,0.05,0.02,-0.03]; a=0.2; s=0.5; y=zeros(T)
    for t in 1:T; m=a; for k in 1:5; t-k>=1 && (m+=phi[k]*y[t-k]); end; y[t]=m+s*randn(rng); end
    X=zeros(T-5,5); yt=zeros(T-5); for t in 6:T; X[t-5,:].=[y[t-1],y[t-2],y[t-3],y[t-4],y[t-5]]; yt[t-5]=y[t]; end
    build_lin(X,yt,1.0,vcat([0.1],fill(0.05,5),[log(0.5)]))
end

# ---------- gp_regr (dim 3) — dense in-graph cholesky (the hard case) ----------
gp_cov(D2,al,rho,sig,N)= al^2 .* exp.(-0.5.*D2./rho^2) .+ sig.*Matrix(I,N,N)
gp_ll(K,y,N)=begin C=cholesky(Symmetric(K)); z=C.L\y; -0.5*N*LOG2PI - sum(log,diag(C.L)) - 0.5*sum(z.*z) end
gp_prior(rho,al,sig,lr,la,ls)=gammalp(rho,25,4)+nlp(al,0,2)+nlp(sig,0,1)+lr+la+ls
rk_gp=@kernel model(unconstrained,D2,y,N)=begin
    lr=sum(view(unconstrained,1:1)); la=sum(view(unconstrained,2:2)); ls=sum(view(unconstrained,3:3))
    rho=exp(lr); al=exp(la); sig=exp(ls)
    K=gp_cov(D2,al,rho,sig,N); prior=gp_prior(rho,al,sig,lr,la,ls)
    posterior=prior+gp_ll(K,y,N); return posterior
end
function build_gp()
    rng=MersenneTwister(6); N=30; x=sort(rand(rng,N).*4 .-2); D2=[(x[i]-x[j])^2 for i in 1:N, j in 1:N]
    K=1.0.*exp.(-0.5.*D2./0.8^2) .+0.05.*Matrix(I,N,N); L=cholesky(Symmetric(K)).L; y=L*randn(rng,N)
    _prep(rk_gp,(:unconstrained,:D2,:y,:N),[log(0.8),log(1.0),log(0.05)],(;D2=D2,y=y,N=N))
end

# ---------- Mh (capture-recapture, dim 303) — data-branch vectorized to a mask ----------
mh_om(q)=1/(1+exp(-sum(view(q,1:1)))); mh_mp(q)=1/(1+exp(-sum(view(q,2:2)))); mh_sg(q)=5/(1+exp(-sum(view(q,3:3))))
mh_er(q)=@view q[4:end]
mh_eps(mp,sg,er)= (log(mp)-log1p(-mp)) .+ sg.*er
mh_ll(om,eps,y,obs,T,ct)=begin lo=log(om); l1=log1p(-om); lp = -T .* log.(1 .+ exp.(eps))
    term_obs = lo .+ y.*eps .+ lp; term_un = log.(exp.(lo .+ lp) .+ (1-om))
    sum(obs.*term_obs .+ (1 .- obs).*term_un) + ct end
mh_jac(om,mp,sg)=log(om)+log1p(-om)+log(mp)+log1p(-mp)+log(5)+log(sg/5)+log1p(-sg/5)
mh_prioreps(er,M)= -0.5*M*LOG2PI - 0.5*sum(er.*er)
rk_mh=@kernel model(unconstrained,y,obs,T,ct,M)=begin
    om=mh_om(unconstrained); mp=mh_mp(unconstrained); sg=mh_sg(unconstrained); er=mh_er(unconstrained)
    eps=mh_eps(mp,sg,er); jac=mh_jac(om,mp,sg); pe=mh_prioreps(er,M); ll=mh_ll(om,eps,y,obs,T,ct)
    posterior=jac+pe+ll; return posterior
end
function build_mh()
    rng=MersenneTwister(20260906); M=300; T=7; y=zeros(M)
    for i in 1:M; if rand(rng)<0.6; p=1/(1+exp(-(log(0.4/0.6)+0.8*randn(rng)))); y[i]=rand(rng,0:T); end; end
    obs=Float64.(y.>0); ct=0.0   # binomial-coefficient constant dropped (HMC-invariant offset)
    _prep(rk_mh,(:unconstrained,:y,:obs,:T,:ct,:M),vcat([0.0,0.0,0.0],fill(0.0,M)),(;y=y,obs=obs,T=Float64(T),ct=ct,M=M))
end

# ---- HMC loop timer (mirrors prepared_hmc.prepare_hmc for any density) ----
function hmc_program(built, backend, transitions, steps)
    density, ad, q = built.density, built.ad, built.q; D=length(q)
    point = transpiled_endpoint(F.euclidean_phasepoint, F.leapfrog!,
        CallbackHandle(Potential(density)), CallbackHandle(Gradient(ad)),
        Diagonal(ones(D)), copy(q), zeros(D))
    prepare_transpiled(PositionMultinomialHMCAuthoring.multinomial_hmc_state, point;
        backend, method=:step!, argument=(backend===:reactant ? Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91,77])) : Xoshiro(91)),
        iterations=transitions, kernel_kwargs=(n_steps=steps, step_f=F.leapfrog!, stepsize=0.03), outputs=(position=(:init,:pos),))
end
function time_loop(builder, backend; T=1000, steps=16, rounds=6)
    built=builder()
    prog=hmc_program(built, backend, T, steps)
    st=initial_transpiled_state(prog)
    rng = backend===:reactant ? Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91,77])) : Xoshiro(91)
    prog(st, rng)                       # warmup
    ts=[ @elapsed prog(st, rng) for _ in 1:rounds ]
    median(ts)/T*1e6
end

const MODELS = [
    ("Rate_1(d1)", build_rate, 1), ("GLM_Binomial(d3)", build_bin, 3), ("gp_regr(d3)", build_gp, 3),
    ("GLM_Poisson(d4)", build_pois, 4), ("kidscore(d5)", build_kidscore, 5), ("arK(d7)", build_ark, 7),
    ("sblri_blr(d7)", build_sblri, 7), ("eight_schools(d10)", build_density, 10),
    ("GLMM_Poisson(d45)", build_glmm, 45), ("Mh(d303)", build_mh, 303)]
results=Dict{String,Any}()
println("\n===== all-10 native vs Reactant HMC loop (multinomial, L16, 1000 transitions/batch; µs/transition) =====")
for (name, builder, dim) in MODELS
    local n_us, r_us, rstat
    try; n_us = time_loop(builder, :native); catch e; n_us = NaN; end
    try; r_us = time_loop(builder, :reactant); rstat=""; catch e; r_us = NaN; rstat=sprint(showerror,e); end
    key=replace(name, r"[()]"=>"_")
    if isnan(r_us)
        rshort = replace(first(rstat, 150), "\n"=>" ")
        results[key]=Dict("dim"=>dim,"native_us"=>(isnan(n_us) ? "FAILED" : round(n_us;digits=2)),"reactant"=>"FAILED","error"=>rshort)
        println("  $(rpad(name,20)) native $(isnan(n_us) ? "FAILED" : string(round(n_us;digits=2))*" µs") | reactant did NOT transpile: $rshort")
    else
        results[key]=Dict("dim"=>dim,"native_us"=>round(n_us;digits=2),"reactant_us"=>round(r_us;digits=2),"ratio"=>round(r_us/n_us;digits=3))
        println("  $(rpad(name,20)) native $(round(n_us;digits=2)) µs | reactant $(round(r_us;digits=2)) µs | ratio $(round(r_us/n_us;digits=3))×")
    end
end
outp=get(ENV,"AS_OUTPUT","")
if outp!=""
    import TOML, Dates
    open(outp,"w") do io; TOML.print(io, Dict("schema"=>"reactant-hmc-loop-table-v2","generated_at"=>string(Dates.now()),
        "methodology"=>"all 10 named posteriordb models, native vs Reactant HMC loop (multinomial HMC, L16, 1000 transitions/batch) via the merged transpiler; same authored @kernel lowered to both backends; compile+warmup excluded; median of 6 batches",
        "note"=>"Reactant-friendly authoring required to transpile: 1-element reductions sum(view(q,i:i)) vs scalar q[i]; sum(a.*b) vs dot; vectorized data-masks vs per-element `if`. Models that still fail are reported honestly (in-graph dense cholesky).",
        "results"=>results)); end
    println("\nwrote receipt: $outp")
end
println("\nRHMC_TABLE_DONE")
