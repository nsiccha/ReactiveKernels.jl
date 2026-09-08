# Reactant axes (3, 4) for gp_gate.jl. `include`d ONLY from the
# `GP_GATE_REACTANT=1` branch, AFTER `import Reactant`, because a
# `Reactant.@compile` inside a plain `if` still macro-expands at definition time
# (it would abort the Reactant-unloaded phase). Uses the tolerances/helpers and
# `Reactant` from the enclosing gp_gate.jl scope.

function reactant_axes(name, kb_r, prep_r, sm, pts, reactant_runtime; chol_grad_gap = false)
    q = pts[1]
    rq = Reactant.to_rarray(q)
    rrt = map(Reactant.to_rarray, reactant_runtime)

    # ---- axis 3: Reactant primal vs native ----
    kbc = Reactant.@compile sync = true kb_r(rq, rrt...)
    vc = Float64(kbc(rq, rrt...)); vn = kb_r(q, reactant_runtime...)
    rp = _relv(vc, vn)
    @assert isfinite(vc) && rp < RPRIMAL_TOL "$name: Reactant primal rel=$rp ≥ $RPRIMAL_TOL"
    println("  [3] Reactant primal rel=$(round(rp; sigdigits = 4)) (< $RPRIMAL_TOL) PASS"); flush(stdout)

    # ---- axis 4: Reactant gradient vs Stan ----
    gb = Reactant.to_rarray(similar(q))
    if chol_grad_gap
        # KNOWN GAP (snag reactant-compile-f877fcfd): the compiled reverse of a
        # dense in-graph Cholesky has no EnzymeMLIR adjoint for the Cholesky
        # primitives — `stablehlo.triangular_solve` (marginal-GP solve, gp_regr)
        # and `stablehlo.cholesky` (latent-GP factor, gp_pois_regr). Assert axis 4
        # fails with EXACTLY one of those adjoint diagnostics — a regression to a
        # different failure, or a silent fix, trips this and the gate (and this
        # flag) must be updated.
        ok = false; msg = ""
        try
            Reactant.@compile sync = true ReactiveKernels.ad_value_and_gradient!(prep_r, gb, rq, rrt...)
            ok = true
        catch e
            msg = sprint(showerror, e)
        end
        @assert !ok "$name: Reactant gradient of the Cholesky now COMPILES — the known gap (snag reactant-compile-f877fcfd) appears fixed; hard-assert it against Stan and clear chol_grad_gap."
        known = occursin("adjoint", msg) && (occursin("triangular_solve", msg) || occursin("cholesky", msg))
        @assert known "$name: Reactant gradient failed with an UNEXPECTED error (not the known missing Cholesky-primitive adjoint, snag reactant-compile-f877fcfd): $(first(msg, 400))"
        prim = occursin("triangular_solve", msg) ? "stablehlo.triangular_solve" : "stablehlo.cholesky"
        println("  [4] Reactant grad: KNOWN GAP confirmed — no adjoint for $prim (snag reactant-compile-f877fcfd)"); flush(stdout)
    else
        gc = Reactant.@compile sync = true ReactiveKernels.ad_value_and_gradient!(prep_r, gb, rq, rrt...)
        _, rg = gc(prep_r, gb, rq, rrt...)
        gh = Array{Float64}(rg); gs = sgrad(sm, q); rgr = relerr(gh, gs)
        @assert all(isfinite, gh) "$name: Reactant gradient not finite"
        @assert rgr < RGRAD_TOL "$name: Reactant gradient vs Stan rel=$rgr ≥ $RGRAD_TOL"
        println("  [4] Reactant grad rel=$(round(rgr; sigdigits = 4)) (< $RGRAD_TOL) PASS"); flush(stdout)
    end
    return
end
