# Prototype: a PLAIN-JULIA control-flow interpreter generated from build_method's CFG for nuts_state,
# stub effects. Goal this step: verify the loop + stack + terminator (TBranch/TCall/TGoto/TRet) machinery
# runs and terminates with a sane block-visit trace — the spine that becomes the traced @trace while.
using ReactiveKernels, LinearAlgebra, Random
const RK = ReactiveKernels
include(joinpath("/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-hmc","examples","nuts_runtime.jl"))
module Fix; include(joinpath("/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-hmc","benchmark","nuts_kernel_authoring_fixture.jl")); end

irs = RK.method_irs(Fix.nuts_state)
rec = RK.defunctionalized_mids(irs)
by_mid = Dict(ir.id.decl => ir for ir in irs)
# per-method CFG (blocks keyed by pc) + entry
cfg = Dict{Int,Any}(); entry = Dict{Int,Int}()
for ir in irs
    m = RK.mid_of(ir.id); (m in rec) || continue
    c = RK.build_method(ir, by_mid, rec)
    cfg[m] = Dict(b.pc => b for b in c.blks); entry[m] = c.entry
end
root_mid = first(RK.mid_of(ir.id) for ir in irs if ir.id.name === :step!)
CAP = 4*(10+2)

# The interpreter spine. ctrl stack = (mid,fidx,pc) parallel Int vectors; csp scalar (plain-Julia proto).
# For now: STUB effects; TBranch conditions are driven by a SCRIPTED oracle of branch decisions so we can
# exercise the control machinery deterministically (real conds come from the tensor state later).
# We just verify: push/pop/branch/goto terminate and visit blocks coherently.
function run_spine(branch_decisions)
    mids = zeros(Int, CAP); fidxs = zeros(Int, CAP); pcs = zeros(Int, CAP)
    fsp = Dict(m => 0 for m in keys(cfg))
    csp = 0; bi = 0; visits = Tuple{Int,Int}[]
    # push root frame
    fsp[root_mid] += 1; csp += 1; mids[csp]=root_mid; fidxs[csp]=fsp[root_mid]; pcs[csp]=entry[root_mid]
    steps = 0
    while csp >= 1 && steps < 100000
        steps += 1
        m = mids[csp]; f = fidxs[csp]; p = pcs[csp]
        push!(visits, (m,p))
        b = cfg[m][p]; t = b.term
        if t isa RK.TRet
            fsp[m] -= 1; csp -= 1
        elseif t isa RK.TGoto
            pcs[csp] = t.pc == 0 ? (fsp[m]-=1; -1) : t.pc      # pc==0 means return→pop
            if t.pc == 0; fsp[m] -= 1; csp -= 1; end
        elseif t isa RK.TBranch
            bi += 1; dec = bi <= length(branch_decisions) ? branch_decisions[bi] : false
            np = dec ? t.then_pc : t.else_pc
            if np == 0; fsp[m] -= 1; csp -= 1; else; pcs[csp] = np; end
        elseif t isa RK.TCall
            c = t.callee_mid
            pcs[csp] = t.resume_pc   # set resume on current frame
            fsp[c] += 1; csp += 1
            csp <= CAP || error("overflow"); fsp[c] <= CAP || error("frame overflow")
            mids[csp]=c; fidxs[csp]=fsp[c]; pcs[csp]=entry[c]
        else
            error("unknown term $(typeof(t))")
        end
    end
    (steps, length(visits), csp)
end
# exercise with a few random branch-decision scripts
for seed in (1,2,3)
    Random.seed!(seed); decs = rand(Bool, 500)
    (steps, nv, endcsp) = run_spine(decs)
    println("seed=$seed: steps=$steps visits=$nv end_csp=$endcsp terminated=$(endcsp==0 && steps<100000)")
end
println("SPINE_DONE")
