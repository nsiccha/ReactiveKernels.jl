using ReactiveKernels, LinearAlgebra, Random
const RK = ReactiveKernels
include(joinpath("/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-hmc","examples","nuts_runtime.jl"))
module Fix; include(joinpath("/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-hmc","benchmark","nuts_kernel_authoring_fixture.jl")); end
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
fr = frame(pf, Float64, 4)
println("fr fields: ", fieldnames(typeof(fr)))
println("fr.init type: ", typeof(fr.init).name.name, " fields=", fieldnames(typeof(fr.init)))
println("trees :: ", typeof(fr.trees).name.name, " len=", length(fr.trees))
t1 = fr.trees[1]; println("trees[1] type=", typeof(t1).name.name, " fields=", fieldnames(typeof(t1)))
for f in fieldnames(typeof(t1)); v=getfield(t1,f); println("   trees[1].$f :: $(typeof(v).name.name) = ", v isa AbstractArray ? "arr$(size(v))" : (v isa NamedTuple ? "NT$(keys(v))" : v)); end
# tree subfields (mv/trajectory namedtuples)
for f in fieldnames(typeof(t1)); v=getfield(t1,f); if v isa NamedTuple; for k in keys(v); println("      trees[1].$f.$k :: ", typeof(v[k]).name.name, " ", v[k] isa AbstractArray ? size(v[k]) : v[k]); end; end; end
println("proposals :: len=", length(fr.proposals), " elt=", typeof(fr.proposals[1]).name.name)
println("shared :: ", typeof(fr.shared).name.name, " fields=", fieldnames(typeof(fr.shared)))
for f in fieldnames(typeof(fr.shared)); v=getfield(fr.shared,f); println("   shared.$f :: ", typeof(v).name.name, " ", v isa AbstractArray ? size(v) : (isa(v,Number) ? v : "")); end
println("mask fields: gofwd=$(fr.gofwd) may_sample=$(fr.may_sample) may_continue=$(fr.may_continue) diverged=$(fr.diverged) max_depth=$(fr.max_depth) min_dham=$(fr.min_dham)")
println("FRAME_STRUCT_DONE")
