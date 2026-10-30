# Measurement CONTRACT for the all-82 benchmark. The measurement body calls
# `measure_native` once per model.
#
#   measure_native(ctx)   -> Dict of NATIVE_CELLS + descriptive/parity fields
#       primal_rk, primal_turing, primal_stan            (median ns, via Chairmarks)
#       gradient_rk, gradient_turing, gradient_stan       (median ns)
#       dim, family, note, parity_pass, rk_off, tu_off, off_reason, rk_grad_relerr, tu_grad_relerr
#
# `ctx` is the shared per-model bundle the body builds once (name, model, data, sm, cmap,
# groups, points, kb, prep, tldf, off_rk, off_tu, ...). Timing helpers return MEDIAN
# NANOSECONDS (Chairmarks `median(b).time` is SECONDS — convert with *1e9; the original
# body's missing conversion is why the first smoke printed "0.0 ns").
#
# Historical receipts also carry hmc_rk_native and hmc_ahmc_turing (µs/transition): the producer
# measured HMC throughput until ReactiveKernels' sampler code was removed.
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
    # even if the reverse happened to be finite/accurate — no RK cell may report a number off a
    # broken graph (performance contract 2026-09-08).
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

# ---- entrypoint -----------------------------------------------------------------------
# single-eval rk/turing/stan + descriptive/parity.
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
    row
end

end # module All80Axes
