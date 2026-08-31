using ReactiveKernels, LinearAlgebra, Random
import ReactiveKernelsNUTSExamples
const RK = ReactiveKernels
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "examples", "nuts_runtime.jl"))
module Fix; include(joinpath(@__DIR__, "..", "nuts_kernel_authoring_fixture.jl")); end
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
    f=ReactiveKernelsNUTSExamples._construct_nuts_frame(pf,vals(pf,T;metric),md;step_f=RK.partial(Fix.leapfrog!;stepsize=T(.1)),stats_f=Fix.nuts_stats!,min_dham=-1000.0)
    RK.compile_prepared_initialization(pf,typeof(f.init),typeof(f.shared))(f.init,f.shared,RK.kernel_prepared_handles(pf))
    ReactiveKernelsNUTSExamples._seed_nuts_children!(f); f
end
# phasepoint owned slots that carry live numeric data (f4=pos f5=mom f7=pot f8=dpot f10=dkin f11=kin f12=ham)
const PP_VEC = (4,5,8,10); const PP_SCA = (7,11,12)
ppget(p) = (Tuple(copy(getfield(p,Symbol("f",i))) for i in PP_VEC)..., Tuple(getfield(p,Symbol("f",i)) for i in PP_SCA)...)
ppset!(p, t) = begin
    for (k,i) in enumerate(PP_VEC); getfield(p,Symbol("f",i)) .= t[k]; end
    for (k,i) in enumerate(PP_SCA); setfield!(p, Symbol("f",i), t[length(PP_VEC)+k]); end
end
treeget(tr) = (copy(tr.log_weight), copy(tr.bwd.mom), copy(tr.bwd.dham_dmom), copy(tr.bwd_fwd.mom), copy(tr.bwd_fwd.dham_dmom), copy(tr.summed_mom.bwd), copy(tr.summed_mom.fwd))
treeset!(tr, t) = (tr.log_weight .= t[1]; tr.bwd.mom .= t[2]; tr.bwd.dham_dmom .= t[3]; tr.bwd_fwd.mom .= t[4]; tr.bwd_fwd.dham_dmom .= t[5]; tr.summed_mom.bwd .= t[6]; tr.summed_mom.fwd .= t[7])
# extract full mutable NUTS state (excludes shared read-only + config)
function frame_to_state(fr)
    (init=ppget(fr.init), fwd=ppget(fr.fwd), bwd=ppget(fr.bwd),
     proposals=[ppget(p) for p in fr.proposals], trees=[treeget(t) for t in fr.trees],
     gofwd=fr.gofwd, may_sample=fr.may_sample, may_continue=fr.may_continue, diverged=fr.diverged,
     n_steps=fr.diag.n_steps, reached_depth=fr.diag.reached_depth, acceptance_rate=fr.diag.acceptance_rate, dham=fr.diag.dham,
     pending=fr.diag.pending, committed=fr.diag.committed)
end
function state_to_frame!(fr, s)
    ppset!(fr.init,s.init); ppset!(fr.fwd,s.fwd); ppset!(fr.bwd,s.bwd)
    for (p,t) in zip(fr.proposals,s.proposals); ppset!(p,t); end
    for (tr,t) in zip(fr.trees,s.trees); treeset!(tr,t); end
    fr.gofwd=s.gofwd; fr.may_sample=s.may_sample; fr.may_continue=s.may_continue; fr.diverged=s.diverged
    ReactiveKernelsNUTSExamples._diag_set_value!(fr.diag, Val(1), s.n_steps); ReactiveKernelsNUTSExamples._diag_set_value!(fr.diag, Val(2), s.reached_depth)
    ReactiveKernelsNUTSExamples._diag_set_value!(fr.diag, Val(3), s.acceptance_rate); ReactiveKernelsNUTSExamples._diag_set_value!(fr.diag, Val(4), s.dham)
    fr
end
snap(fr)=(copy(getfield(fr.init,:f4)), fr.diag.n_steps, fr.diag.reached_depth, round(fr.diag.acceptance_rate;digits=12), round(fr.diag.dham;digits=12))
# ROUND-TRIP TEST: native on fr1 vs native on fr2 seeded via frame_to_state(fr1)->state_to_frame!(fr2)
allok=true
for (md,seed) in ((2,20260829),(4,20260829),(6,12345),(8,777))
    fr1=frame(pf,Float64,md); C1=ReactiveKernelsNUTSExamples.compile_nuts_native(pf,Fix.nuts_state,Fix.refresh_momentum!!,Fix.nuts!!,fr1)
    s = frame_to_state(fr1)                      # capture pre-transition state as tensors
    fr2=frame(pf,Float64,md); C2=ReactiveKernelsNUTSExamples.compile_nuts_native(pf,Fix.nuts_state,Fix.refresh_momentum!!,Fix.nuts!!,fr2)
    state_to_frame!(fr2, s)                        # reconstruct fr2 from the tensor state
    C1.root!(fr1,C1.scratch,Random.Xoshiro(seed)); C2.root!(fr2,C2.scratch,Random.Xoshiro(seed))
    ok = snap(fr1)==snap(fr2)
    global allok &= ok
    println("md=$md seed=$seed: native(fr) == native(roundtrip(state)) ? $ok")
end
println(allok ? "TENSORIZE_OK: frame_to_state/state_to_frame! capture the FULL mutable NUTS state (round-trip preserves the transition)" : "TENSORIZE_MISMATCH")
