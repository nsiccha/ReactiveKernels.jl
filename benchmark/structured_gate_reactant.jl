# Reactant axes for structured_gate.jl. This file is INCLUDED ONLY from the
# `if DO_REACTANT` branch, AFTER `import Reactant`, so its `Reactant.compile`
# calls are lowered only in a process where Reactant is actually loaded. It
# shares the enclosing script's helpers and constants by being included into
# the same module. The Reactant axes use pts[1] (one probe) — recorded here and
# in the per-axis accounting.

function _reactant_axes(name, build, kb, sm, pts, have, bind, data,
                        mod_data = nothing, reactant_grad = :pin_boundary)
    q = pts[1]; rq = Reactant.to_rarray(q)

    # ---- [3a] Reactant primal of the public ALL-BOUND query vs native ----
    # The HLO shape of THIS query is measured and reported without claiming a
    # carry loop: with every data port bound, the compiler specializes the
    # recurrence for the query shape. The while count is read from
    # repr(Reactant.@code_hlo ...) — the same surface test_authored_scan_reactant.jl
    # asserts on — because `module_string` is empty on the pinned Reactant
    # (0.2.285) and a count against it is vacuous; the HLO byte size is recorded
    # next to every count as the positive control (bytes=0 → no shape claim).
    kbc = Reactant.compile(kb, (rq,); sync = true)
    vc = Float64(kbc(rq)); vn = kb(q)
    rp = _relv(vc, vn)
    @assert isfinite(vc) && rp < RPRIMAL_TOL "$name: Reactant primal rel=$rp ≥ $RPRIMAL_TOL"
    bound_hlo = repr(Reactant.@code_hlo optimize = false kb(rq))
    bound_while = count("stablehlo.while", bound_hlo)
    bound_bytes = length(codeunits(bound_hlo))
    _axis!(name, "PASS Reactant primal (all-bound query) rel=$(round(rp; sigdigits=4)) (1 probe); stablehlo.while=$bound_while, hlo_bytes=$bound_bytes (@code_hlo; compiler-specialized shape, no carry-loop claim)")
    println("  [3a] Reactant primal (all-bound) rel=$(round(rp; sigdigits=4))  (< $RPRIMAL_TOL; 1 probe) PASS; stablehlo.while=$bound_while, hlo_bytes=$bound_bytes (@code_hlo; reported)"); flush(stdout)

    # ---- [3b] traced-stream query: measure the HLO shape on this boundary ----
    # MEASURED, not assumed: the while count is recorded per axis from
    # @code_hlo with its byte-size control; no carry-loop or no-unrolling
    # claim is asserted on this shape at this pin. Upstream RK landed the
    # traced eachrow while-lowering (main @ 90acd41c) after this pin; once
    # this tree carries it, the traced-stream count is expected to be 1 and
    # may then be asserted (snag scan-while-claim-5006b5b9 / todo 1wbtwl9).
    #
    # Integer data ports (ii/jj/y/I) CANNOT lower as traced streams at this
    # pin (traced-boolean TypeError in the compiler, retained as [3b]
    # diagnostic); models whose HAVEs include them use the IRT-lane boundary
    # instead: data bound, only the unconstrained vector traced, plus the
    # compiled-gradient axis [4] on that same query.
    int_ports = any(f -> data[String(f)] isa AbstractVector{Int} ||
                           data[String(f)] isa Int, Base.tail(have))
    if !int_ports
        data_bnd = bind(_GATE_DATA[])
        kb_free = prepare(build(); have, want = :posterior)
        traced = (Reactant.to_rarray(q),
                  (Reactant.to_rarray(getfield(data_bnd, f)) for f in Base.tail(have))...)
        kbt = Reactant.compile(kb_free, traced; sync = true)
        vt = Float64(kbt(traced...))
        rt = _relv(vt, vn)
        @assert isfinite(vt) && rt < RPRIMAL_TOL "$name: traced-stream Reactant primal rel=$rt ≥ $RPRIMAL_TOL"
        traced_hlo = repr(Reactant.@code_hlo optimize = false kb_free(traced...))
        traced_while = count("stablehlo.while", traced_hlo)
        traced_bytes = length(codeunits(traced_hlo))
        _axis!(name, "PASS Reactant primal (traced-stream query) rel=$(round(rt; sigdigits=4)) (1 probe); stablehlo.while=$traced_while, hlo_bytes=$traced_bytes (@code_hlo; measured, not asserted)")
        println("  [3b] Reactant primal (traced streams) rel=$(round(rt; sigdigits=4))  (< $RPRIMAL_TOL; 1 probe) PASS; stablehlo.while=$traced_while, hlo_bytes=$traced_bytes (@code_hlo; measured, not asserted)"); flush(stdout)
    else
        _axis!(name, "UNSUPPORTED Reactant traced-stream query at this pin: integer data ports (ii/jj/y/I) do not lower as traced streams (traced-boolean TypeError; retained diagnostic)")
        println("  [3b] traced streams UNSUPPORTED (traced-integer ports at this pin; diagnostic retained)"); flush(stdout)
    end

    # ---- [4] Reactant gradient vs Stan (fresh process; no priming here) ----
    # At THIS pin both queries compile with the scan UNROLLED (measured above:
    # 8.1 MB / 9.6 MB @code_hlo with 0 whiles). Differentiating that module in
    # EnzymeMLIR is infeasible on this host: the attempt was killed by an
    # EXTERNAL SIGTERM inside the MLIR pass pipeline (retained gate log,
    # 2026-09-18 run), so a default attempt cannot complete and is recorded as
    # UNSUPPORTED-AT-THIS-PIN instead of being silently skipped. The attempt
    # itself stays available opt-in (STRUCTURED_REACTANT_GRAD=1) for a run on a
    # host that can carry it; once this tree carries RK main @ 90acd41c (traced
    # eachrow while-lowering, single-while shape), the traced-stream gradient
    # becomes tractable and this axis returns to a hard-asserted attempt.
    # reactant_grad === :pin_boundary records the unrolled-scan boundary and
    # attempts only opt-in; :assert hard-asserts the compiled gradient on the
    # bound-data query (IRT-lane recipe, scan-free graphs).
    if reactant_grad === :pin_boundary
        if get(ENV, "STRUCTURED_REACTANT_GRAD", "0") != "1"
            _axis!(name, "UNSUPPORTED Reactant gradient at this pin: unrolled scan module (see axes 3a/3b byte sizes) killed the EnzymeMLIR reverse compile with an external SIGTERM (retained gate log, kb-run-compact.X7biXF, 2026-09-18); opt-in attempt via STRUCTURED_REACTANT_GRAD=1, hard re-measure after RK 90acd41c lands in-tree")
            println("  [4] Reactant grad  UNSUPPORTED (documented unrolled-scan pin boundary; diagnostic retained)"); flush(stdout)
            return
        end
    end
    begin
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
    end
    return
end
