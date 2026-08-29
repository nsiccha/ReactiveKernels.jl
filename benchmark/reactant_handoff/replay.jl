using ReactiveKernels, LinearAlgebra, Random
const RK = ReactiveKernels
include(joinpath("/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-hmc","examples","nuts_runtime.jl"))
module Fix; include(joinpath("/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-hmc","benchmark","nuts_kernel_authoring_fixture.jl")); end
# Recorder + ReplayRNG, both defined in Random so the native domain gate admits them
Random.eval(quote
    mutable struct RecRNG <: AbstractRNG; inner::Xoshiro; log::Vector{Any}; end
    mutable struct ReplayRNG <: AbstractRNG; mom::Vector{Float64}; dirs::Vector{Bool}; exps::Vector{Float64}; kd::Vector{Int}; ke::Vector{Int}; end
end)
const RecRNG = Random.RecRNG; const ReplayRNG = Random.ReplayRNG
mkrec(seed) = RecRNG(Random.Xoshiro(seed), Any[])
Random.randn!(r::RecRNG, x::AbstractArray) = (Random.randn!(r.inner, x); push!(r.log,(:m, copy(x))); x)
Base.rand(r::RecRNG, ::Type{Bool}) = (v=Base.rand(r.inner, Bool); push!(r.log,(:d, v)); v)
Random.randexp(r::RecRNG) = (v=Random.randexp(r.inner); push!(r.log,(:e, v)); v)
Random.randn!(r::ReplayRNG, x::AbstractArray) = (x .= r.mom; x)
Base.rand(r::ReplayRNG, ::Type{Bool}) = (r.kd[1]+=1; r.dirs[r.kd[1]])
Random.randexp(r::ReplayRNG) = (r.ke[1]+=1; r.exps[r.ke[1]])
# extract a bundle (momentum, dirs, exps) from a recorder log
function bundle_from_log(log)
    mom = Float64[]; dirs = Bool[]; exps = Float64[]
    for e in log; e[1]==:m ? (mom=copy(e[2])) : e[1]==:d ? push!(dirs,e[2]) : push!(exps,e[2]); end
    (mom, dirs, exps)
end
pf = RK._prepare_factory(Fix.euclidean_phasepoint, RK.kernel_registration(Fix.leapfrog!))
function vals(pf,T;metric=T[2 0;0 2])
    P=RK.kernel_prepared_plan(pf); d=Dict{Int,Any}()
    for s in RK.kernel_plan_slots(P)
        n=String(s.path[1]); d[s.canon] = n=="pot_f" ? (p->sum(abs2,p)) : n=="grad_f" ? ((dst,p)->(dst .= 2 .* p; sum(abs2,p))) :
          n=="metric" ? metric : n=="chol_metric" ? cholesky(metric) : startswith(n,"##node") ? zero(T) :
          n=="pos" ? T[1,2] : n=="mom" ? T[3,4] : n in ("dpot_dpos","dham_dpos","dkin_dmom","dham_dmom") ? T[0,0] : zero(T)
    end; d
end
function frame(pf,T,md;metric=T[2 0;0 2])
    f=RK._construct_nuts_frame(pf,vals(pf,T;metric),md;step_f=RK.partial(Fix.leapfrog!;stepsize=T(.1)),stats_f=Fix.nuts_stats!,min_dham=-1000.0)
    RK.compile_prepared_initialization(pf,typeof(f.init),typeof(f.shared))(f.init,f.shared,RK.kernel_prepared_handles(pf))
    RK._seed_nuts_children!(f); f
end
snap(fr) = (copy(getfield(fr.init,:f4)), fr.diag.n_steps, fr.diag.reached_depth, round(fr.diag.acceptance_rate;digits=12), round(fr.diag.dham;digits=12))
function verify()
    allok = true
    for (md,seed) in ((2,20260829),(4,20260829),(4,20260830),(6,12345),(8,777),(3,999))
        f1=frame(pf,Float64,md); C=RK.compile_nuts_native(pf,Fix.nuts_state,Fix.refresh_momentum!!,Fix.nuts!!,f1)
        r=mkrec(seed); C.root!(f1,C.scratch,r); out1=snap(f1)
        (mom,dirs,exps)=bundle_from_log(r.log)
        f2=frame(pf,Float64,md); C2=RK.compile_nuts_native(pf,Fix.nuts_state,Fix.refresh_momentum!!,Fix.nuts!!,f2)
        rep=ReplayRNG(mom,dirs,exps,[0],[0]); C2.root!(f2,C2.scratch,rep); out2=snap(f2)
        ok = out1==out2
        println("md=$md seed=$seed: Xoshiro-run == ReplayRNG(bundle)-run ? $ok  (bundle: $(length(dirs)) dir, $(length(exps)) exp)")
        allok &= ok
    end
    println(allok ? "REPLAY_VERIFIED: native reference driver reproduces native over the pre-gen bundle" : "REPLAY_MISMATCH")
end
verify()
