# Reactant axes (3 & 4) for seq_batch_gate.jl. Included ONLY under
# SEQ_REACTANT=1, AFTER `import Reactant`, so `Reactant.@compile` is
# macro-expanded only in a process where Reactant is genuinely loaded. A plain
# `if DO_REACTANT` around this def does NOT defer macro expansion (a top-level
# `if` lowers both branches before the runtime condition), so the
# Reactant-unloaded native phase must never even parse it.
function _reactant_axes(name, kb, prep, sm, pts)
    q = pts[1]; rq = Reactant.to_rarray(q); tr = time()
    # axis 3: Reactant primal vs native
    kbc = Reactant.@compile sync = true kb(rq)
    vc = Float64(kbc(rq)); vn = kb(q)
    rp = _relv(vc, vn)
    @assert isfinite(vc) && rp < RPRIMAL_TOL "$name: Reactant primal rel=$rp ≥ $RPRIMAL_TOL"
    println("  [3] Reactant primal rel=$(round(rp; sigdigits=4))  (< $RPRIMAL_TOL) PASS ($(round(time()-tr;digits=1))s)"); flush(stdout)
    # axis 4: Reactant gradient vs Stan (finite)
    tg = time()
    gb = Reactant.to_rarray(similar(q))
    gc = Reactant.@compile sync = true ReactiveKernels.ad_value_and_gradient!(prep, gb, rq)
    _, rg = gc(prep, gb, rq); gh = Array{Float64}(rg)
    gs = sgrad(sm, q)
    rgr = relerr(gh, gs)
    @assert all(isfinite, gh) "$name: Reactant gradient not finite"
    @assert rgr < RGRAD_TOL "$name: Reactant gradient vs Stan rel=$rgr ≥ $RGRAD_TOL"
    println("  [4] Reactant grad  rel=$(round(rgr; sigdigits=4))  (< $RGRAD_TOL) PASS ($(round(time()-tg;digits=1))s)"); flush(stdout)
end
