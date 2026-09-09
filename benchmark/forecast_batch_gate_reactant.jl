# Reactant axes for forecast_batch_gate.jl. This file is INCLUDED ONLY from the
# `if DO_REACTANT` branch of the gate, AFTER `import Reactant`, so its
# `Reactant.@compile` macro calls are lowered only in a process where Reactant is
# actually loaded. It shares the enclosing script's helpers (_relv, relerr, sgrad,
# the *_TOL constants, ReactiveKernels) by being included into the same module.

function _reactant_axes(name, kb, prep, sm, pts, stress_q = nothing)
    q = pts[1]; rq = Reactant.to_rarray(q)

    # ---- axis 3: Reactant primal vs native ----
    kbc = Reactant.@compile sync = true kb(rq)
    vc = Float64(kbc(rq)); vn = kb(q)
    rp = _relv(vc, vn)
    @assert isfinite(vc) && rp < RPRIMAL_TOL "$name: Reactant primal rel=$rp ≥ $RPRIMAL_TOL"
    println("  [3] Reactant primal rel=$(round(rp; sigdigits = 4))  (< $RPRIMAL_TOL) PASS"); flush(stdout)

    # ---- axis 4: Reactant gradient vs Stan ----
    gb = Reactant.to_rarray(similar(q))
    gc = Reactant.@compile sync = true ReactiveKernels.ad_value_and_gradient!(prep, gb, rq)
    _, rg = gc(prep, gb, rq)
    gh = Array{Float64}(rg)
    gs = sgrad(sm, q)
    rgr = relerr(gh, gs)
    @assert all(isfinite, gh) "$name: Reactant gradient not finite"
    @assert rgr < RGRAD_TOL "$name: Reactant gradient vs Stan rel=$rgr ≥ $RGRAD_TOL"
    println("  [4] Reactant grad  rel=$(round(rgr; sigdigits = 4))  (< $RGRAD_TOL) PASS"); flush(stdout)

    # ---- stress probe through the COMPILED program (where supplied) ----
    # Same reference-validity-first contract as the native stress probe: assert
    # the Stan oracle value+gradient finite, then the compiled Reactant primal
    # (vs native and Stan) and compiled Reactant gradient (vs Stan). The probe
    # has the same shape as pts[1], so the already-compiled `kbc`/`gc` closures
    # are reused at the stress point.
    if stress_q !== nothing
        vsb = sval(sm, stress_q); gsb = sgrad(sm, stress_q)
        @assert isfinite(vsb) && all(isfinite, gsb) "$name: Reactant stress BridgeStan reference not finite"
        rqb = Reactant.to_rarray(stress_q)
        vcb = Float64(kbc(rqb)); vnb = kb(stress_q)
        rpb = _relv(vcb, vnb)
        @assert isfinite(vcb) && rpb < RPRIMAL_TOL "$name: Reactant stress primal rel=$rpb ≥ $RPRIMAL_TOL"
        @assert _relv(vnb, vsb) < VALUE_TOL "$name: Reactant stress native value ≠ Stan"
        _, rgb = gc(prep, gb, rqb)
        ghb = Array{Float64}(rgb)
        @assert all(isfinite, ghb) "$name: Reactant stress gradient not finite"
        rgberr = relerr(ghb, gsb)
        @assert rgberr < RGRAD_TOL "$name: Reactant stress gradient vs Stan rel=$rgberr ≥ $RGRAD_TOL"
        println("  [b] Reactant stress-point primal+grad max_rel=$(round(max(rpb, rgberr); sigdigits = 4)) PASS"); flush(stdout)
    end
end
