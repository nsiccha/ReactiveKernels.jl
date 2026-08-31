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
snap(fr)=(copy(getfield(fr.init,:f4)), fr.diag.n_steps, fr.diag.reached_depth, round(fr.diag.acceptance_rate;digits=12), round(fr.diag.dham;digits=12))
allok=true
for (md,seed) in ((2,20260829),(4,20260829),(6,12345),(8,777))
    fn=frame(pf,Float64,md); Cn=ReactiveKernelsNUTSExamples.compile_nuts_native(pf,Fix.nuts_state,Fix.refresh_momentum!!,Fix.nuts!!,fn)
    fc=frame(pf,Float64,md); Cc=ReactiveKernelsNUTSExamples.compile_nuts(pf,Fix.nuts_state,Fix.refresh_momentum!!,Fix.nuts!!,fc)
    Cn.root!(fn,Cn.scratch,Random.Xoshiro(seed)); Cc.root!(fc,Cc.scratch,Random.Xoshiro(seed))
    ok = snap(fn)==snap(fc); global allok &= ok
    println("md=$md seed=$seed: compile_nuts (SoA imperative) == compile_nuts_native ? $ok")
end
println(allok ? "SOA_REF_OK: compile_nuts is a verified plain-Julia state-machine reference (== native)" : "SOA_REF_MISMATCH")
