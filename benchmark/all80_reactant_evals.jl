# Reactant per-OPERATION cells — INCLUDED ONLY IN THE REACTANT PHASE (so Reactant.@compile is
# lowered only where Reactant is loaded). Reactant support for a FAITHFUL graph is a per-model,
# per-operation matrix: the primal @compile, the gradient @compile, and the transpiled :reactant
# HMC loop can each INDEPENDENTLY succeed (a finite number) or fail. Each is caught SEPARATELY so
# one failure never suppresses evidence another path works. On failure a cell holds a STABLE
# diagnostic string = operation + exception TYPE + concise message (docs deduplicate identical
# signatures). No flat / Reactant-friendly substitute densities — an unsupported faithful lowering
# is reported honestly, not routed around. Uses the body's `med`, `@be` (Chairmarks) and `hmc_loop`
# (in Main scope via include).

# op + the exception (showerror already prints the exact type, e.g. "MethodError: …") + the
# top stack frames (the call boundary the compiler/AD failure surfaced at), so a diagnostic is
# fixable and not merely groupable. Bounded so a TOML cell stays small; docs deduplicate
# identical signatures. `bt` is the caught backtrace where the catch site can supply one (a
# stored preparation exception carries none, so it defaults to a message-only reason).
function _reactant_reason(op, err, bt = nothing)
    msg = first(replace(sprint(showerror, err), "\n" => " "), 400)
    top = ""
    if bt !== nothing
        frames = stacktrace(bt)
        isempty(frames) || (top = " | at " * join(
            (string(f.func, "@", basename(string(f.file)), ":", f.line) for f in first(frames, 3)),
            " ← "))
    end
    string(op, ": ", msg, top)
end

function reactant_cells(kb, prep, q; transitions = 4, grad_oracle = nothing)
    row = Dict{String,Any}()
    rq = Reactant.to_rarray(q)
    # op1 — primal @compile + value-check vs native + time
    row["primal_rk_reactant"] = try
        kbc = Reactant.@compile sync = true kb(rq)
        vc = Float64(kbc(rq)); vn = kb(q)
        # RELATIVE parity: an absolute 1e-6 is unsatisfiable for a large-magnitude density
        # (the float ULP at |logdensity|~1e11 is ~1e-5), which false-rejected earn_height
        # (relerr ~1e-13). XLA sum reassociation only ever perturbs the low bits.
        abs(vc - vn) <= 1e-6 * max(abs(vn), 1.0) ||
            error("primal parity relerr $(abs(vc - vn) / max(abs(vn), 1.0)) ($vc vs native $vn)")
        med(@be kbc(rq))
    catch err
        _reactant_reason("primal @compile", err, catch_backtrace())
    end
    # op2 — gradient @compile + parity vs the REFERENCE STAN gradient + time. The oracle is
    # the real Stan gradient (BridgeStan log_density_gradient, propto=false jacobian=true),
    # mapped to RK unconstrained order via stan_perm and passed in as `grad_oracle` — the
    # SAME oracle the native phase gates against (all80_posteriordb_body.jl). No finite
    # differences. The RK↔Stan density offset is a constant, so its gradient is 0 and the
    # two gradients must agree directly.
    row["gradient_rk_reactant"] =
        prep isa Exception ? _reactant_reason("gradient preparation", prep) :
        grad_oracle === nothing ? "gradient oracle: no reference Stan gradient supplied" :
        grad_oracle isa Exception ? _reactant_reason("gradient Stan-oracle", grad_oracle) : try
        gb = Reactant.to_rarray(similar(q))
        gradc = Reactant.@compile sync = true ReactiveKernels.ad_value_and_gradient!(prep, gb, rq)
        _, rgrad = gradc(prep, gb, rq)
        ghost = Array{Float64}(rgrad)
        all(isfinite, ghost) || error("gradient non-finite")
        gerr = maximum(abs, ghost .- grad_oracle) / max(maximum(abs, grad_oracle), eps())
        gerr < 2e-3 || error("gradient parity relerr $gerr vs reference Stan gradient")
        med(@be gradc(prep, gb, rq))
    catch err
        _reactant_reason("gradient @compile", err, catch_backtrace())
    end
    # op3 — transpiled :reactant HMC loop (independent of the @compile single-eval path)
    row["hmc_rk_reactant"] = prep isa Exception ?
        _reactant_reason("transpiled HMC gradient preparation", prep) : try
        hmc_loop(kb, prep, q, :reactant,
            () -> Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91, 77]));
            T = transitions, steps = All80Axes.HMC_STEPS, rounds = All80Axes.HMC_ROUNDS)
    catch err
        _reactant_reason("transpiled HMC :reactant", err, catch_backtrace())
    end
    row
end
