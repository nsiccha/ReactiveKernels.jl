# Reactant per-OPERATION cells — INCLUDED ONLY IN THE REACTANT PHASE (so Reactant.@compile is
# lowered only where Reactant is loaded). Reactant support for a FAITHFUL graph is a per-model,
# per-operation matrix: the primal @compile, the gradient @compile, and the transpiled :reactant
# HMC loop can each INDEPENDENTLY succeed (a finite number) or fail. Each is caught SEPARATELY so
# one failure never suppresses evidence another path works. On failure a cell holds a STABLE
# diagnostic string = operation + exception TYPE + concise message (docs deduplicate identical
# signatures). No flat / Reactant-friendly substitute densities — an unsupported faithful lowering
# is reported honestly, not routed around. Uses the body's `med`, `@be` (Chairmarks) and `hmc_loop`
# (in Main scope via include).

# op + the exception (showerror already prints the exact type, e.g. "MethodError: …").
_reactant_reason(op, err) = string(op, ": ", first(replace(sprint(showerror, err), "\n" => " "), 200))

function reactant_cells(kb, prep, q)
    row = Dict{String,Any}()
    rq = Reactant.to_rarray(q)
    # op1 — primal @compile + value-check vs native + time
    row["primal_rk_reactant"] = try
        kbc = Reactant.@compile sync = true kb(rq)
        vc = Float64(kbc(rq)); vn = kb(q)
        abs(vc - vn) < 1e-6 || error("primal parity $vc vs native $vn")
        med(@be kbc(rq))
    catch err
        _reactant_reason("primal @compile", err)
    end
    # op2 — gradient @compile + host-converted parity vs native + time (independent of op1)
    row["gradient_rk_reactant"] = try
        gnative = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        gb = Reactant.to_rarray(similar(q))
        gradc = Reactant.@compile sync = true ReactiveKernels.ad_value_and_gradient!(prep, gb, rq)
        _, rgrad = gradc(prep, gb, rq)
        ghost = Array{Float64}(rgrad)
        gerr = maximum(abs, ghost .- gnative) / max(maximum(abs, gnative), eps())
        gerr < 2e-3 || error("gradient parity relerr $gerr vs native")
        all(isfinite, ghost) || error("gradient non-finite")
        med(@be gradc(prep, gb, rq))
    catch err
        _reactant_reason("gradient @compile", err)
    end
    # op3 — transpiled :reactant HMC loop (independent of the @compile single-eval path)
    row["hmc_rk_reactant"] = try
        hmc_loop(kb, prep, q, :reactant, () -> Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91, 77])))
    catch err
        _reactant_reason("transpiled HMC :reactant", err)
    end
    row
end
