# Phase-compute CONTRACT for the all-82 benchmark. The measurement body calls exactly one
# phase per subprocess (RK_ALL80_PHASE), so native and Reactant never share compiler state.
#
#   measure_native(ctx)   -> Dict of NATIVE_CELLS + descriptive/parity fields
#       primal_rk, primal_turing, primal_stan            (median ns, via Chairmarks)
#       gradient_rk, gradient_turing, gradient_stan       (median ns)
#       hmc_rk_native                                     (µs/transition, transpiler :native)
#       hmc_ahmc_turing                                   (µs/transition, AdvancedHMC on the Turing LDF)
#       dim, family, note, parity_pass, rk_off, tu_off, off_reason, rk_grad_relerr, tu_grad_relerr
#   measure_reactant(ctx) -> Dict of REACTANT_CELLS
#       primal_rk_reactant, gradient_rk_reactant          (median ns, Reactant-compiled RK kernel)
#       hmc_rk_reactant                                   (µs/transition, transpiler :reactant)
#
# `ctx` is the shared per-model bundle the body builds once (name, model, data, sm, cmap,
# groups, points, kb, prep, tldf, off_rk, off_tu, ...). Timing helpers return MEDIAN
# NANOSECONDS (Chairmarks `median(b).time` is SECONDS — convert with *1e9; the original
# body's missing conversion is why the first smoke printed "0.0 ns").
#
# HMC uses the SAME transpiler path as benchmark/reactant_hmc_loop_table.jl
# (transpiled_endpoint / prepare_transpiled / time_loop, backend :native|:reactant), but
# fed the FAITHFUL registry graphs (ctx.kb / ctx.prep) rather than the historical hand-rolled
# flat densities. That machinery lives under benchmark/sampler_transpiler/ and is included by
# the body only in the phase that needs it.
module All80Axes

using Statistics: median
import Chairmarks

# median NANOSECONDS from a Chairmarks benchmark (`.time` is in seconds).
med_ns(b) = median(b).time * 1e9

# ---- native single-eval timings (VALIDATED path — mirrors the committed body) ----
# stan_primal/stan_grad/turing/rk closures are supplied by the body (it owns the imports).
"""populate the native primal+gradient cells (median ns) from the body-supplied closures."""
function native_single_eval!(row, c)
    row["primal_rk"]       = med_ns(Chairmarks.@be c.rk_primal())
    row["primal_turing"]   = med_ns(Chairmarks.@be c.tu_primal())
    row["primal_stan"]     = med_ns(Chairmarks.@be c.stan_primal())
    row["gradient_rk"]     = med_ns(Chairmarks.@be c.rk_grad())
    row["gradient_turing"] = med_ns(Chairmarks.@be c.tu_grad())
    row["gradient_stan"]   = med_ns(Chairmarks.@be c.stan_grad())
    row
end

# ---- HMC throughput (µs/transition) --------------------------------------------------
# transpiler backend :native / :reactant — the body includes the sampler_transpiler
# machinery and passes `hmc_time_loop` (a closure capturing transpiled_endpoint /
# prepare_transpiled / F.leapfrog! bound to ctx.kb + ctx.prep), so this module stays free of
# the Reactant/transpiler imports until the phase that needs them.
"""native HMC: RK density+grad driven through the transpiler :native backend."""
hmc_rk_native!(row, c) = (row["hmc_rk_native"] = c.hmc_time_loop(:native); row)
"""reactant HMC: SAME transpiled program lowered through Reactant."""
hmc_rk_reactant!(row, c) = (row["hmc_rk_reactant"] = c.hmc_time_loop(:reactant); row)
"""AHMC+Turing HMC: AdvancedHMC over the Turing LDF (value+gradient); µs/transition.
Body supplies `c.ahmc_time_loop` (AdvancedHMC.HMC, fixed L, warmup excluded)."""
hmc_ahmc_turing!(row, c) = (row["hmc_ahmc_turing"] = c.ahmc_time_loop(); row)

# ---- Reactant single-eval (primal+gradient of the compiled RK kernel) ----------------
# Body supplies `c.rk_reactant_primal` / `c.rk_reactant_grad` closures (Reactant-compiled
# kb / prep via `@compile sync=true`), value-checked == native before timing.
function reactant_single_eval!(row, c)
    row["primal_rk_reactant"]   = med_ns(Chairmarks.@be c.rk_reactant_primal())
    row["gradient_rk_reactant"] = med_ns(Chairmarks.@be c.rk_reactant_grad())
    row
end

# ---- phase entrypoints ---------------------------------------------------------------
# native phase (NO `using Reactant`): single-eval rk/turing/stan + RK-native HMC +
# AHMC-Turing HMC + descriptive/parity. Loading Reactant must not perturb these timings.
function measure_native(c)
    row = Dict{String,Any}("dim" => c.dim, "family" => c.family, "note" => c.note,
        "parity_pass" => c.parity_pass, "rk_off" => c.rk_off, "tu_off" => c.tu_off,
        "off_reason" => c.off_reason, "rk_grad_relerr" => c.rk_grad_relerr,
        "tu_grad_relerr" => c.tu_grad_relerr)
    native_single_eval!(row, c)
    hmc_rk_native!(row, c)
    hmc_ahmc_turing!(row, c)
    row
end

# reactant phase (Reactant + transpiler loaded): RK+Reactant single-eval + RK+Reactant HMC.
# Same multinomial HMC + identical q/integrator/steps/warmup as the native RK HMC.
function measure_reactant(c)
    row = Dict{String,Any}()
    reactant_single_eval!(row, c)
    hmc_rk_reactant!(row, c)
    row
end

end # module All80Axes
