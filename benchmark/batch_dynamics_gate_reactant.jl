# Reactant axes for batch_dynamics_gate.jl. This file is INCLUDED ONLY from the
# `if DO_REACTANT` branch, AFTER `import Reactant`, so its `Reactant.@compile`
# macro calls are lowered only in a process where Reactant is actually loaded.
# It shares the enclosing script's helpers (_relv, relerr, sgrad,
# ReactiveKernels) by being included into the same module.
#
# Unlike the static-graph gate, axes 3-4 here are ATTEMPTED with full
# receipts: synthetic-shape probe evidence (Reactant 0.2.285) shows primal
# tracing of the natural adaptive solve fails (scalar-indexing refusal;
# boolean-context failure in the adaptive stepper), while a concrete-input
# compile constant-folds. The two-point check below exists precisely to catch
# a constant-folded "pass": the compiled artifact must agree with the native
# primal at BOTH pts[1] and pts[2].

function _reactant_axes(name, kb, prep, sm, pts)
    ok3 = false
    ok4 = false
    # ---- axis 3: Reactant primal vs native, at two probe points ----
    try
        q1, q2 = pts[1], pts[2]
        r1, r2 = Reactant.to_rarray(q1), Reactant.to_rarray(q2)
        kbc = Reactant.@compile sync = true kb(r1)
        v1 = Float64(kbc(r1)); n1 = kb(q1)
        v2 = Float64(kbc(r2)); n2 = kb(q2)
        r1r = _relv(v1, n1); r2r = _relv(v2, n2)
        println("  [3-attempt] Reactant primal pt1_rel=$r1r pt2_rel=$r2r"); flush(stdout)
        if isfinite(v1) && isfinite(v2) && r1r < 1e-6 && r2r < 1e-6
            ok3 = true
            println("  [3] Reactant primal two-point PASS"); flush(stdout)
        else
            println("  [3] Reactant primal RECORDED-FAIL (mismatch or nonfinite; " *
                    "a pt1-only match with pt2 mismatch is the constant-fold signature)")
            flush(stdout)
        end
    catch e
        println("  [3] Reactant primal RECORDED-FAIL: $(typeof(e))"); flush(stdout)
        println("      $(first(sprint(showerror, e), 800))"); flush(stdout)
    end

    # ---- axis 4: Reactant gradient vs Stan ----
    try
        q = pts[1]; rq = Reactant.to_rarray(q)
        gb = Reactant.to_rarray(similar(q))
        gc = Reactant.@compile sync = true ReactiveKernels.ad_value_and_gradient!(prep, gb, rq)
        _, rg = gc(prep, gb, rq)
        gh = Array{Float64}(rg)
        gs = sgrad(sm, q)
        rgr = relerr(gh, gs)
        println("  [4-attempt] Reactant grad rel=$rgr"); flush(stdout)
        if all(isfinite, gh) && rgr < 2e-3
            ok4 = true
            println("  [4] Reactant grad PASS"); flush(stdout)
        else
            println("  [4] Reactant grad RECORDED-FAIL"); flush(stdout)
        end
    catch e
        println("  [4] Reactant grad RECORDED-FAIL: $(typeof(e))"); flush(stdout)
        println("      $(first(sprint(showerror, e), 800))"); flush(stdout)
    end
    println("  Reactant axes verdict: primal=$ok3 gradient=$ok4 (attempted, see receipts above)")
    flush(stdout)
end
