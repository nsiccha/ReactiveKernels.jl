# Reactant axes for structured_gate.jl. This file is INCLUDED ONLY from the
# `if DO_REACTANT` branch, AFTER `import Reactant`, so its `Reactant.@compile`
# macro calls are lowered only in a process where Reactant is actually loaded.
# It shares the enclosing script's helpers (_relv, relerr, sgrad, the *_TOL
# constants, _assert_documented_gap, ReactiveKernels) by being included into
# the same module.

function _reactant_axes(name, kb, prep, sm, pts)
    q = pts[1]; rq = Reactant.to_rarray(q)

    # ---- axis 3: Reactant primal vs native ----
    kbc = Reactant.@compile sync = true kb(rq)
    vc = Float64(kbc(rq)); vn = kb(q)
    rp = _relv(vc, vn)
    @assert isfinite(vc) && rp < RPRIMAL_TOL "$name: Reactant primal rel=$rp ≥ $RPRIMAL_TOL"
    println("  [3] Reactant primal rel=$(round(rp; sigdigits = 4))  (< $RPRIMAL_TOL) PASS"); flush(stdout)

    # ---- axis 4: Reactant gradient vs Stan ----
    # For the hmm_drive_1 grad_gap model, this axis runs AFTER the native
    # runtime-activity diagnostic (axis 2r) in this gate's fixed order. That
    # matters: on a FIRST trace without axis 2r, the compiled gradient dies
    # with the same EnzymeRuntimeActivityError (receipt in snag
    # scan-prior-enzym-d67d4ac1), while exercising the runtime-activity native
    # axis first lets the same compile+execute succeed and match Stan. The gate
    # pins this exact order, asserts the gradient against Stan, and reports the
    # ordering sensitivity rather than hiding either observation.
    gb = Reactant.to_rarray(similar(q))
    gc = Reactant.@compile sync = true ReactiveKernels.ad_value_and_gradient!(prep, gb, rq)
    _, rg = gc(prep, gb, rq)
    gh = Array{Float64}(rg)
    gs = sgrad(sm, q)
    rgr = relerr(gh, gs)
    @assert all(isfinite, gh) "$name: Reactant gradient not finite"
    @assert rgr < RGRAD_TOL "$name: Reactant gradient vs Stan rel=$rgr ≥ $RGRAD_TOL"
    println("  [4] Reactant grad  rel=$(round(rgr; sigdigits = 4))  (< $RGRAD_TOL) PASS"); flush(stdout)
end
