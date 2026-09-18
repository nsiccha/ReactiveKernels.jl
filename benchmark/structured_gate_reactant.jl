# Reactant axes for structured_gate.jl. This file is INCLUDED ONLY from the
# `if DO_REACTANT` branch, AFTER `import Reactant`, so its `Reactant.compile`
# calls are lowered only in a process where Reactant is actually loaded. It
# shares the enclosing script's helpers and constants by being included into
# the same module. The Reactant axes use pts[1] (one probe) — recorded here and
# in the per-axis accounting.

function _reactant_axes(name, build, kb, sm, pts, have, bind)
    q = pts[1]; rq = Reactant.to_rarray(q)

    # ---- [3a] Reactant primal of the public ALL-BOUND query vs native ----
    # The HLO shape of THIS query is measured and reported without claiming a
    # carry loop: with every data port bound, the compiler specializes the
    # recurrence for the query shape and the compiled module contains no
    # stablehlo.while region. The traced-stream query below is the boundary
    # where the carry loop is required and asserted.
    kbc = Reactant.compile(kb, (rq,); sync = true)
    vc = Float64(kbc(rq)); vn = kb(q)
    rp = _relv(vc, vn)
    @assert isfinite(vc) && rp < RPRIMAL_TOL "$name: Reactant primal rel=$rp ≥ $RPRIMAL_TOL"
    bound_while = count("stablehlo.while", kbc.module_string)
    _axis!(name, "PASS Reactant primal (all-bound query) rel=$(round(rp; sigdigits=4)) (1 probe); stablehlo.while regions=$bound_while (compiler-specialized shape, no carry-loop claim)")
    println("  [3a] Reactant primal (all-bound) rel=$(round(rp; sigdigits=4))  (< $RPRIMAL_TOL; 1 probe) PASS; stablehlo.while regions=$bound_while (reported; compiler-specialized shape)"); flush(stdout)

    # ---- [3b] traced-stream query: measure the HLO shape on this boundary ----
    # MEASURED, not assumed: on this model's traced-stream query the compiled
    # module_string also contains 0 stablehlo.while regions (value parity is
    # exact). No carry-loop or no-unrolling claim is made for either query
    # shape; the counts are recorded per axis. See the reactivekernels-use §7c
    # divergence snag for the doc-level mismatch.
    data_bnd = bind(_GATE_DATA[])
    kb_free = prepare(build(); have, want = :posterior)
    traced = (Reactant.to_rarray(q),
              (Reactant.to_rarray(getfield(data_bnd, f)) for f in Base.tail(have))...)
    kbt = Reactant.compile(kb_free, traced; sync = true)
    vt = Float64(kbt(traced...))
    rt = _relv(vt, vn)
    @assert isfinite(vt) && rt < RPRIMAL_TOL "$name: traced-stream Reactant primal rel=$rt ≥ $RPRIMAL_TOL"
    traced_while = count("stablehlo.while", kbt.module_string)
    _axis!(name, "PASS Reactant primal (traced-stream query) rel=$(round(rt; sigdigits=4)) (1 probe); stablehlo.while regions=$traced_while (measured; no carry-loop claim)")
    println("  [3b] Reactant primal (traced streams) rel=$(round(rt; sigdigits=4))  (< $RPRIMAL_TOL; 1 probe) PASS; stablehlo.while regions=$traced_while (measured; no carry-loop claim)"); flush(stdout)

    # ---- [4] Reactant gradient vs Stan (fresh process; no priming here) ----
    grad_result = try
        prep = prepare_ad(kb, AE, q; active = :unconstrained)
        gb = Reactant.to_rarray(similar(q))
        gc = Reactant.compile(ReactiveKernels.ad_value_and_gradient!, (prep, gb, rq);
                              sync = true)
        _, rg = gc(prep, gb, rq)
        gh = Array{Float64}(rg)
        gs = sgrad(sm, q)
        @assert all(isfinite, gh) "$name: Reactant gradient not finite"
        rgr = relerr(gh, gs)
        @assert rgr < RGRAD_TOL "$name: Reactant gradient vs Stan rel=$rgr ≥ $RGRAD_TOL"
        "PASS Reactant gradient rel=$(round(rgr; sigdigits=4)) (1 probe)"
    catch err
        text = _retain(name, 4, err)
        if occursin(_GAP_NEEDLE, text)
            "UNSUPPORTED Reactant gradient: documented $_GAP_SNAG failure; complete diagnostic retained"
        else
            rethrow(err)
        end
    end
    _axis!(name, grad_result)
    println("  [4] Reactant grad  ", startswith(grad_result, "PASS") ?
            "$(split(grad_result)[4]) (< $RGRAD_TOL; 1 probe) PASS" :
            "UNSUPPORTED (documented $_GAP_SNAG; diagnostic retained)"); flush(stdout)
    return
end
