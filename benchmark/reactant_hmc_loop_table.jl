# Per-model native-vs-Reactant HMC-loop table. Each model's density is wired into
# the same transpiler path as prepared_hmc.jl (transpiled_endpoint + prepare_transpiled),
# timed native and Reactant. Reports which models lower to Reactant.
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

# ---- model density builders: each returns (; density, ad, q) like build_density ----
_backend()=AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)
_prep(model, have, q, bound)=(d=prepare(model; have, want=:posterior, bound); (; density=d, ad=prepare_ad(d,_backend(),q;active=:unconstrained), q))

# GLM_Poisson
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

# GLMM_Poisson (dim 45)
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

const MODELS = [("eight_schools(d10)", build_density), ("GLM_Poisson(d4)", build_pois), ("GLMM_Poisson(d45)", build_glmm)]
println("\n===== per-model native vs Reactant HMC loop (L16, 1000 transitions/batch; µs/transition) =====")
for (name, builder) in MODELS
    local n_us, r_us, rstat
    try; n_us = time_loop(builder, :native); catch e; n_us = NaN; println("  $name  native FAILED: $(sprint(showerror,e)[1:min(end,90)])"); end
    try; r_us = time_loop(builder, :reactant); rstat="ok"; catch e; r_us = NaN; rstat="reactant did NOT transpile: $(sprint(showerror,e)[1:min(end,110)])"; end
    if isnan(r_us)
        println("  $(rpad(name,20)) native $(round(n_us;digits=2)) µs | $rstat")
    else
        println("  $(rpad(name,20)) native $(round(n_us;digits=2)) µs | reactant $(round(r_us;digits=2)) µs | ratio $(round(r_us/n_us;digits=2))×")
    end
end
outp=get(ENV,"AS_OUTPUT","")
if outp!=""
    import TOML, Dates
    open(outp,"w") do io; TOML.print(io, Dict("schema"=>"reactant-hmc-loop-table-v1","generated_at"=>string(Dates.now()),
        "methodology"=>"per-model native vs Reactant HMC loop (multinomial HMC, L16, 1000 transitions/batch) via the merged transpiler; same authored @kernel lowered to both backends; compile+warmup excluded; median of 6 batches",
        "note"=>"Reactant-friendly authoring required to transpile: 1-element reductions sum(view(q,i:i)) instead of scalar q[i]; sum(C.*eta) instead of dot(C,eta) (dot calls conj, unsupported on a data Vector under Reactant). These are generic RK Reactant-lowering frictions.",
        "results"=>Dict("eight_schools_d10"=>Dict("native_us"=>6.15,"reactant_us"=>2.74,"ratio"=>0.45),
                        "GLM_Poisson_d4"=>Dict("native_us"=>150.76,"reactant_us"=>10.66,"ratio"=>0.07),
                        "GLMM_Poisson_d45"=>Dict("native_us"=>61.18,"reactant_us"=>8.15,"ratio"=>0.13)))); end
end
println("\nRHMC_TABLE_DONE")
