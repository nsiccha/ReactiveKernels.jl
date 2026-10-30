# Measurement CONTRACT for the all-82 benchmark. The measurement body calls
# `measure_native` once per model.
#
#   measure_native(ctx)   -> Dict of NATIVE_CELLS + descriptive/parity fields
#       primal_rk, primal_turing, primal_stan            (median ns, via Chairmarks)
#       gradient_rk, gradient_turing, gradient_stan       (median ns)
#       hmc_rk_native                                     (µs/transition, transpiled HMC program)
#       hmc_ahmc_turing                                   (µs/transition, AdvancedHMC on the Turing LDF)
#       dim, family, note, parity_pass, rk_off, tu_off, off_reason, rk_grad_relerr, tu_grad_relerr
#
# `ctx` is the shared per-model bundle the body builds once (name, model, data, sm, cmap,
# groups, points, kb, prep, tldf, off_rk, off_tu, ...). Timing helpers return MEDIAN
# NANOSECONDS (Chairmarks `median(b).time` is SECONDS — convert with *1e9; the original
# body's missing conversion is why the first smoke printed "0.0 ns").
#
# HMC uses the transpiled multinomial-HMC program (transpiled_endpoint / prepare_transpiled)
# fed the FAITHFUL registry graphs (ctx.kb / ctx.prep). That machinery lives under
# benchmark/sampler_transpiler/ and is included by the body.
module All80Axes

using Statistics: median
import Chairmarks

# median NANOSECONDS from a Chairmarks benchmark (`.time` is in seconds).
med_ns(b) = median(b).time * 1e9

# ---- native single-eval timings (VALIDATED path — mirrors the committed body) ----
# stan_primal/stan_grad/turing/rk closures are supplied by the body (it owns the imports).
"""populate the native primal+gradient cells (median ns) from the body-supplied closures.
`gradient_rk` is numeric where the RK reverse works; where the graph hits a KNOWN core
defect the body pre-computes `c.rk_grad_diag` (a nonempty diagnostic string, e.g. the
authored-plate Int-axis/Any-materialization Enzyme failure) and we record THAT instead of
re-attempting the slow failing compile. Every OTHER native cell (primal rk/turing/stan,
gradient turing/stan) is unaffected and stays a hard number for all 82."""
function native_single_eval!(row, c)
    # Fix D per-cell preservation: an RK cell is a NUMBER only where that operation is verified;
    # a failed RK operation carries its exact diagnostic STRING — never a timing a ratio plot would
    # render as a passing verdict. Reference cells (turing/stan) are ALWAYS numeric.
    # PROPAGATION: a defective RK PRIMAL makes the whole RK graph unreliable, so it BLOCKS gradient_rk
    # (and hmc_rk_native, below) even if the reverse happened to be finite/accurate — no RK cell may
    # report a number off a broken graph (performance contract 2026-09-08).
    row["primal_rk"]       = c.rk_primal_diag === nothing ?
        med_ns(Chairmarks.@be c.rk_primal()) : c.rk_primal_diag
    row["primal_turing"]   = med_ns(Chairmarks.@be c.tu_primal())
    row["primal_stan"]     = med_ns(Chairmarks.@be c.stan_primal())
    row["gradient_rk"]     = c.rk_grad_diag !== nothing ? c.rk_grad_diag :
        c.rk_primal_diag !== nothing ?
            "gradient_rk: not reported — RK primal defective, graph unreliable: " * c.rk_primal_diag :
        med_ns(Chairmarks.@be c.rk_grad())
    row["gradient_turing"] = med_ns(Chairmarks.@be c.tu_grad())
    row["gradient_stan"]   = med_ns(Chairmarks.@be c.stan_grad())
    row
end

# ---- HMC throughput (µs/transition) --------------------------------------------------
# The body includes the sampler_transpiler machinery and passes `hmc_time_loop` (a closure
# capturing transpiled_endpoint / prepare_transpiled / F.leapfrog! bound to ctx.kb +
# ctx.prep), so this module stays free of the transpiler imports.
"""native HMC: RK density+grad driven through the transpiled HMC program. The RK-native
HMC loop needs the RK reverse gradient, so a model whose gradient hits the known core defect
(`c.rk_grad_diag` set) records the SAME diagnostic here (HMC is blocked BY the gradient), never
a fabricated number. hmc_ahmc_turing is unaffected (Turing side)."""
hmc_rk_native!(row, c) = (row["hmc_rk_native"] =
    c.rk_grad_diag !== nothing ? "hmc_rk_native blocked by RK gradient: " * c.rk_grad_diag :
    c.rk_primal_diag !== nothing ? "hmc_rk_native blocked by RK primal: " * c.rk_primal_diag :
    c.hmc_time_loop(row["hmc_transitions"]); row)
"""AHMC+Turing HMC: AdvancedHMC over the Turing LDF (value+gradient); µs/transition.
Body supplies `c.ahmc_time_loop` (AdvancedHMC.HMC, fixed L, warmup excluded)."""
hmc_ahmc_turing!(row, c) =
    (row["hmc_ahmc_turing"] = c.ahmc_time_loop(row["hmc_transitions"]); row)

# Keep leapfrog work fixed while adapting repetitions to a bounded timing budget. The
# slower available gradient determines ONE transition count shared by RK and AHMC, so a
# row never compares different HMC workloads. Six median rounds remain fixed; only the
# number of transitions in each round changes. The 1.25 factor leaves headroom for the
# integrator and sampler bookkeeping beyond the 16 gradient evaluations.
const HMC_STEPS = 16
const HMC_ROUNDS = 6
const HMC_MIN_TRANSITIONS = 4
const HMC_MAX_TRANSITIONS = 1000
const HMC_TARGET_ROUND_SECONDS = 0.5

function hmc_transitions(row)
    gradients = Float64[]
    for cell in ("gradient_rk", "gradient_turing")
        v = get(row, cell, nothing)
        v isa Real && isfinite(v) && push!(gradients, Float64(v))
    end
    isempty(gradients) && error("adaptive HMC needs at least one finite gradient timing")
    estimated_transition_ns = 1.25 * HMC_STEPS * maximum(gradients)
    clamp(floor(Int, HMC_TARGET_ROUND_SECONDS * 1e9 / estimated_transition_ns),
        HMC_MIN_TRANSITIONS, HMC_MAX_TRANSITIONS)
end

function hmc_protocol!(row, c)
    row["hmc_steps"] = HMC_STEPS
    row["hmc_rounds"] = HMC_ROUNDS
    row["hmc_target_round_seconds"] = HMC_TARGET_ROUND_SECONDS
    row["hmc_transitions"] = hmc_transitions(row)
    hmc_rk_native!(row, c)
    hmc_ahmc_turing!(row, c)
    row
end

# ---- entrypoint -----------------------------------------------------------------------
# single-eval rk/turing/stan + RK-native HMC + AHMC-Turing HMC + descriptive/parity.
function measure_native(c)
    row = Dict{String,Any}("dim" => c.dim, "family" => c.family, "note" => c.note,
        "parity_pass" => c.parity_pass, "rk_off" => c.rk_off, "tu_off" => c.tu_off,
        "off_reason" => c.off_reason, "rk_grad_relerr" => c.rk_grad_relerr,
        "tu_grad_relerr" => c.tu_grad_relerr, "rk_stab" => c.rk_stab, "tu_stab" => c.tu_stab,
        "mag_rk" => c.mag_rk, "mag_tu" => c.mag_tu,
        "rk_primal_ok" => c.rk_primal_ok, "rk_grad_ok" => c.rk_grad_ok,
        "turing_support_ok" => c.turing_support_ok,
        "turing_support_diag" => (c.turing_support_diag === nothing ? "" : c.turing_support_diag),
        "protocol" => c.protocol)
    native_single_eval!(row, c)
    hmc_protocol!(row, c)
    row
end

end # module All80Axes
