# Faithful posteriordb comparison body — RK-BUILTIN graphs vs optimized Stan vs
# optimized Turing, all three on the SAME real model. Parity vs an independent
# FD oracle; native primal+gradient at the fair boundary. Reference model:
# eight_schools (consumes ReactiveKernelsPPLExamples.EightSchoolsExample). New
# models are added as `run_*` blocks as the posteriordb lane lands them.
using Random, LinearAlgebra, Statistics, Dates
using BenchmarkTools
import TOML
using ReactiveKernels, ReactiveKernelsPPLExamples
import BridgeStan
import Turing, DynamicPPL, Distributions
import ADTypes, Mooncake
using DifferentiationInterface
import Enzyme
const LOG2PI = log(2π); const ROUNDS = 12
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
results=Any[]

function report(name,q,oracle,stanspecs,ldf,kb,prep,gbuf,optbase)
    gref=fd_grad(oracle,q); gsc=maximum(abs,gref); gerr(g)=maximum(abs,g.-gref)/gsc
    for (nm,sm) in stanspecs; e=gerr(BridgeStan.log_density_gradient(sm,q;propto=false,jacobian=true)[2]); e<2e-3 || error("parity $name/$nm=$e"); end
    et=gerr(DynamicPPL.LogDensityProblems.logdensity_and_gradient(ldf,q)[2]); et<2e-3 || error("parity $name/turing=$et")
    er=gerr(ReactiveKernels.ad_value_and_gradient!(prep,gbuf,q)[2]); er<2e-3 || error("parity $name/rk=$er")
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
    println("\n===== $name  dim=$(length(q))  (parity stan/turing/rk = $(round(maximum(gerr(BridgeStan.log_density_gradient(base,q;propto=false,jacobian=true)[2]));sigdigits=2))/$(round(et;sigdigits=2))/$(round(er;sigdigits=2))) =====")
    for m in meas; b=m["kind"]=="primal" ? bp : bg
        println("  $(rpad(m["kind"]*"/"*m["impl"],14)) $(lpad(fmt(m["median_ns"]),10))  $(lpad(string(round(m["median_ns"]/b;digits=2))*"x",8))  $(m["median_bytes"])B"); end
    push!(results, Dict("model"=>name,"dim"=>length(q),"measurements"=>meas,
        "parity_relerr"=>Dict("turing"=>et,"rk"=>er),"rk_builtin"=>true,"optbase"=>optbase))
end

# ================= eight_schools (RK-BUILTIN graph, faithful) =================
const ES = ReactiveKernelsPPLExamples.EightSchoolsExample
function oracle_es(q,y,sigma)
    mu=q[1]; lt=q[2]; th=@view q[3:end]; tau=exp(lt); J=length(y)
    lp = -0.5*LOG2PI - log(5.0) - 0.5*(mu/5)^2                 # mu ~ Normal(0,5)
    lp += log(2.0) - log(π*5.0) - log1p((tau/5.0)^2)          # tau ~ HalfCauchy(0,5)
    lp += lt                                                   # jacobian d tau / d log_tau
    lp += sum(i-> -0.5*LOG2PI - log(tau) - 0.5*((th[i]-mu)/tau)^2, 1:J)      # theta ~ Normal(mu,tau)
    lp += sum(i-> -0.5*LOG2PI - log(sigma[i]) - 0.5*((y[i]-th[i])/sigma[i])^2, 1:J)  # y ~ Normal(theta,sigma)
    lp
end
const STAN_ES = raw"""
data { int<lower=0> J; vector[J] y; vector<lower=0>[J] sigma; }
parameters { real mu; real<lower=0> tau; vector[J] theta; }
model { mu ~ normal(0, 5); tau ~ cauchy(0, 5); theta ~ normal(mu, tau); y ~ normal(theta, sigma); }
"""
Turing.@model function turing_es(y,sigma,J)
    mu ~ Distributions.Normal(0,5)
    tau ~ Distributions.truncated(Distributions.Cauchy(0,5); lower=0)
    theta ~ Distributions.MvNormal(fill(mu,J), tau^2*I)
    y ~ Distributions.MvNormal(theta, Diagonal(sigma.^2))
end
let
    y=ES.EIGHT_SCHOOLS_Y; sigma=ES.EIGHT_SCHOOLS_SIGMA; J=length(y)
    q=vcat([0.0, log(5.0)], zeros(J))
    g=ES.build_eight_schools_graph()
    kb=prepare(g; have=(:unconstrained,:observations,:observation_scales), want=:posterior,
        bound=(observations=y, observation_scales=sigma))
    prep=prepare_ad(kb, AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const), q; active=:unconstrained)
    dj=jobj(["J"=>J,"y"=>jvec(y),"sigma"=>jvec(sigma)])
    sm=stan_model(STAN_ES,dj,"eight_schools")
    ldf=stable_ldf(turing_es(y,sigma,J); adtype=ADTypes.AutoMooncake(config=nothing))
    report("eight_schools", q, qq->oracle_es(qq,y,sigma), ["stan"=>sm], ldf, kb, prep, similar(q), "stan")
end

outp=get(ENV,"AS_OUTPUT","")
if outp!=""
    receipt=Dict("schema"=>"faithful-posteriordb-v1","generated_at"=>string(now()),
        "methodology"=>"RK BUILTIN graphs (ReactiveKernelsPPLExamples, idiomatic distribution-kernel authoring) vs optimized Stan vs optimized Turing, all three on the SAME faithful model; native primal+gradient; median over $ROUNDS rounds; parity vs FD oracle",
        "environment"=>Dict("julia"=>string(VERSION),"arch"=>string(Sys.ARCH),"cpu"=>Sys.cpu_info()[1].model,"stan"=>"2.39.0 (BridgeStan 2.9.0)"),
        "models"=>results)
    open(outp,"w") do io; TOML.print(io,receipt); end; println("\nwrote receipt: $outp")
end
println("\nFAITHFUL_POSTERIORDB_DONE")
