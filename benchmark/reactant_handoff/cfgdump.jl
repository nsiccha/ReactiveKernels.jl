using ReactiveKernels, LinearAlgebra, Random
const RK = ReactiveKernels
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "examples", "nuts_runtime.jl"))
module Fix; include(joinpath(@__DIR__, "..", "nuts_kernel_authoring_fixture.jl")); end

irs = RK.method_irs(Fix.nuts_state)
rec = RK.defunctionalized_mids(irs)
by_mid = Dict(ir.id.decl => ir for ir in irs)
println("=== nuts_state methods (all) ===")
for ir in irs; println("  mid=$(RK.mid_of(ir.id)) name=$(ir.id.name)  in_rec=$(RK.mid_of(ir.id) in rec)"); end
println("=== defunctionalized (SCC/suspending) methods → the state machine ===")
for ir in irs
    m = RK.mid_of(ir.id); (m in rec) || continue
    cfg = RK.build_method(ir, by_mid, rec)
    stored = RK.live_formals(ir, cfg.blks)
    println("--- mid=$m name=$(ir.id.name): entry=$(cfg.entry), $(length(cfg.blks)) blocks, stored_formals=$stored ---")
    for b in cfg.blks
        t = b.term
        ts = t isa RK.TBranch ? "TBranch(then=$(t.then_pc),else=$(t.else_pc))" :
             t isa RK.TCall ? "TCall(mid=$(t.callee_mid),resume=$(t.resume_pc),nargs=$(length(t.args)))" :
             t isa RK.TGoto ? "TGoto($(t.pc))" : t isa RK.TRet ? "TRet" : string(typeof(t).name.name)
        # effect kinds (first symbol of each)
        effk = [string(typeof(e).name.name) for e in b.effects]
        println("    pc=$(b.pc): term=$ts effects=$effk")
    end
end
println("CFGDUMP_DONE")
