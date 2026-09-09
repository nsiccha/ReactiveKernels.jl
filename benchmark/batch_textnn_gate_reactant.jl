# Reactant axes for batch_textnn_gate.jl. This file is INCLUDED ONLY from the
# `if DO_REACTANT` branch, AFTER `import Reactant`, so its `Reactant.@compile`
# macro calls are lowered only in a process where Reactant is actually loaded.
# It shares the enclosing script's helpers (_relv, relerr, sgrad, the *_TOL
# constants, ReactiveKernels) by being included into the same module.

# `runtime_vals` are the data ports moved from bound to a traced runtime input
# for the Reactant compile (empty for the fully-bound case; the 376MB MNIST
# matrix for rbmJ100 — see the §7e note in the gate). They are passed as traced
# rarrays after the parameter vector, and are inactive in the AD.
function _reactant_axes(name, kb, prep, sm, pts, runtime_vals = ())
    q = pts[1]; rq = Reactant.to_rarray(q)
    rrt = map(Reactant.to_rarray, runtime_vals)

    # ---- axis 3: Reactant primal vs native ----
    kbc = Reactant.@compile sync = true kb(rq, rrt...)
    vc = Float64(kbc(rq, rrt...)); vn = kb(q, runtime_vals...)
    rp = _relv(vc, vn)
    @assert isfinite(vc) && rp < RPRIMAL_TOL "$name: Reactant primal rel=$rp ≥ $RPRIMAL_TOL"
    println("  [3] Reactant primal rel=$(round(rp; sigdigits = 4)) (< $RPRIMAL_TOL) PASS"); flush(stdout)

    # ---- axis 4: Reactant gradient vs Stan ----
    gb = Reactant.to_rarray(similar(q))
    gc = Reactant.@compile sync = true ReactiveKernels.ad_value_and_gradient!(prep, gb, rq, rrt...)
    _, rg = gc(prep, gb, rq, rrt...)
    gh = Array{Float64}(rg)
    gs = sgrad(sm, q)
    rgr = relerr(gh, gs)
    @assert all(isfinite, gh) "$name: Reactant gradient not finite"
    @assert rgr < RGRAD_TOL "$name: Reactant gradient vs Stan rel=$rgr ≥ $RGRAD_TOL"
    println("  [4] Reactant grad  rel=$(round(rgr; sigdigits = 4)) (< $RGRAD_TOL) PASS"); flush(stdout)
end
