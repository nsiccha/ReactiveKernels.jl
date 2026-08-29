using ReactiveKernels, LinearAlgebra, Random
const RK = ReactiveKernels
include(joinpath("/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-hmc","examples","nuts_runtime.jl"))
module Fix; include(joinpath("/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-hmc","benchmark","nuts_kernel_authoring_fixture.jl")); end

pf = RK._prepare_factory(Fix.euclidean_phasepoint, RK.kernel_registration(Fix.leapfrog!))
function vals(pf,T;metric=T[2 0;0 2])
    P=RK.kernel_prepared_plan(pf); d=Dict{Int,Any}()
    for s in RK.kernel_plan_slots(P)
        n=String(s.path[1]); d[s.canon] = n=="pot_f" ? (p->sum(abs2,p)) :
          n=="grad_f" ? ((dst,p)->(dst .= 2 .* p; sum(abs2,p))) :
          n=="metric" ? metric : n=="chol_metric" ? cholesky(metric) : startswith(n,"##node") ? zero(T) :
          n=="pos" ? T[1,2] : n=="mom" ? T[3,4] :
          n in ("dpot_dpos","dham_dpos","dkin_dmom","dham_dmom") ? T[0,0] : zero(T)
    end; d
end
function frame(pf,T,md;metric=T[2 0;0 2])
    f=RK._construct_nuts_frame(pf,vals(pf,T;metric),md;step_f=RK.partial(Fix.leapfrog!;stepsize=T(.1)),stats_f=Fix.nuts_stats!,min_dham=-1000.0)
    RK.compile_prepared_initialization(pf,typeof(f.init),typeof(f.shared))(f.init,f.shared,RK.kernel_prepared_handles(pf))
    RK._seed_nuts_children!(f); f
end
dumpep(ep,label) = begin
    println("-- $label ($(typeof(ep).name.name)) --")
    for i in 1:12; v=getfield(ep, Symbol("f",i)); println("  f$i :: $(typeof(v)) = $v"); end
end
fr = frame(pf, Float64, 4)
println("=== plan slots (canon => path) ==="); for s in RK.kernel_plan_slots(RK.kernel_prepared_plan(pf)); println("  canon=$(s.canon) path=$(s.path) fields=$(fieldnames(typeof(s)))"); end
dumpep(fr.init, "init BEFORE")
println("=== diag :: $(typeof(fr.diag)) fields=$(fieldnames(typeof(fr.diag))) = $(fr.diag)")
C = RK.compile_nuts_native(pf, Fix.nuts_state, Fix.refresh_momentum!!, Fix.nuts!!, fr)
r = C.root!(fr, C.scratch, Random.Xoshiro(20260829))
println("=== r===fr ? $(r===fr)")
dumpep(fr.init, "init AFTER txn1")
println("diag AFTER txn1 = $(fr.diag)")
C.root!(fr, C.scratch, Random.Xoshiro(20260830))
dumpep(fr.init, "init AFTER txn2")
println("ORACLE_OK")
