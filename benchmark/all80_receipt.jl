# Versioned receipt schema + deterministic aggregator for the all-82 posteriordb
# benchmark. Process-isolation contract (parent-mandated): native and Reactant timings
# are produced in SEPARATE Julia subprocesses (loading Reactant must not perturb native
# compiler state), each writing a PHASE receipt; the single entrypoint aggregates them
# by model key into ONE final receipt. Pure Julia + stdlib TOML (no heavy deps), so it
# is usable from the measurement body, the orchestrator, and a standalone check.
module All80Receipt

import TOML, Dates

const SCHEMA = "all80-benchmark-v1"

# The MANDATORY numeric cells — real numbers required for ALL 82 rows (publication gate).
# primal_*/gradient_* are MEDIAN NANOSECONDS; hmc_* are MEDIAN MICROSECONDS/transition
# (lower = faster). Which PHASE (isolated subprocess) produces which cell (LOCKED split —
# the native phase must NOT `using Reactant`, so loading Reactant cannot perturb native
# timings; cross-process fairness comes from identical q/integrator/steps/warmup/procedure):
#   native (NO Reactant loaded): primal_{rk,turing,stan}, gradient_{rk,turing,stan},
#       hmc_rk_native (RK-native multinomial-HMC loop), hmc_ahmc_turing (AdvancedHMC)
#   reactant (Reactant + transpiler loaded): primal_rk_reactant, gradient_rk_reactant,
#       hmc_rk_reactant (SAME multinomial HMC, Reactant-compiled)
# MANDATORY (a finite number for ALL 82): the 8 NATIVE cells. RK+Reactant is NOT mandatory
# for all 82 — the user (2026-09-07) explicitly does not expect every faithful graph to lower
# through Reactant; the honest deliverable is a per-model transpile/no-transpile breakdown.
const NATIVE_CELLS = ("primal_rk", "primal_turing", "primal_stan",
    "gradient_rk", "gradient_turing", "gradient_stan",
    "hmc_rk_native", "hmc_ahmc_turing")
# STRICTLY-MANDATORY: a finite real number for ALL 82 — the primal on every side, the
# Turing/Stan reverse gradient, and the AdvancedHMC-Turing throughput. None of these depend
# on the RK reverse, so none are affected by the authored-plate core defect.
const MANDATORY_CELLS = ("primal_rk", "primal_turing", "primal_stan",
    "gradient_turing", "gradient_stan", "hmc_ahmc_turing")
# RK-REVERSE cells: numeric where the RK graph differentiates (the ~75 unaffected models),
# else a NONEMPTY diagnostic string carrying the exact Enzyme failure for a model that hits
# the known authored-plate core defect (snag authored-plate-i-4556ee01). Present-and-typed,
# NEVER absent; republished as finite numbers once the core fix lands (user decision 0fmcsb6).
const RK_REVERSE_CELLS = ("gradient_rk", "hmc_rk_native")
# CONDITIONAL cells: a finite NUMBER where it applies, else a NONEMPTY string carrying the
# exact reason. REACTANT cells = numeric where the model lowers through Reactant, else the
# exact Reactant-lowering error. OPTIONAL optimized-Stan/further-Turing = number if a distinct
# verified-faster impl exists, else the user-directive deferral provenance. Never merely absent.
const REACTANT_CELLS = ("primal_rk_reactant", "gradient_rk_reactant", "hmc_rk_reactant")
const OPTIONAL_CELLS = ("primal_opt_stan", "primal_further_turing",
    "gradient_opt_stan", "gradient_further_turing")
const CONDITIONAL_CELLS = (REACTANT_CELLS..., OPTIONAL_CELLS...)
# Descriptive per-model fields carried alongside the cells (native phase authors them).
const DESCRIPTIVE = ("dim", "family", "note", "parity_pass", "rk_off", "tu_off",
    "off_reason", "rk_grad_relerr", "tu_grad_relerr")

"""Write ONE phase's receipt. `rows` maps model-key => Dict{String,Any} of that phase's cells."""
function write_phase(path::AbstractString, phase, rows::AbstractDict)
    doc = Dict("schema" => SCHEMA, "phase" => String(phase),
               "generated_at" => string(Dates.now()),
               "models" => Dict(String(k) => v for (k, v) in rows))
    mkpath(dirname(path))
    open(path, "w") do io; TOML.print(io, doc; sorted = true); end
    path
end

"""Merge phase receipts (native ∪ reactant, per model key) into the final receipt.
Phases carry disjoint cell sets by construction; a cell that appears in two phases must
AGREE — a conflicting duplicate is a HARD ERROR, never a silent overwrite. Model ordering
and cell content are deterministic (TOML-sorted); the only non-reproducible field is the
`generated_at` timestamp, so the doc is content-stable, not byte-stable."""
function aggregate(phase_paths, out_path::AbstractString; meta = Dict{String,Any}())
    merged = Dict{String,Dict{String,Any}}()
    for p in phase_paths
        isfile(p) || error("All80Receipt.aggregate: missing phase receipt $p")
        d = TOML.parsefile(p)
        get(d, "schema", "") == SCHEMA || error("schema mismatch in $p: $(get(d,"schema",""))")
        for (k, cells) in get(d, "models", Dict())
            dst = get!(merged, k, Dict{String,Any}())
            for (ck, cv) in cells
                if haskey(dst, ck) && dst[ck] != cv
                    error("All80Receipt.aggregate: conflicting duplicate cell $k.$ck across phases: $(repr(dst[ck])) vs $(repr(cv)) (from $p)")
                end
                dst[ck] = cv
            end
        end
    end
    doc = Dict("schema" => SCHEMA, "generated_at" => string(Dates.now()),
               "meta" => meta, "models" => merged)
    mkpath(dirname(out_path))
    open(out_path, "w") do io; TOML.print(io, doc; sorted = true); end
    out_path
end

"""Publication-gate check: every model carries all MANDATORY cells as finite REAL
NUMBERS, AND every OPTIONAL cell is PRESENT as either a finite number or a NONEMPTY
provenance string (N/A-with-provenance is required — a merely-absent optional cell is a
gate failure, not an implicit N/A). Returns a list of issues (empty == passes)."""
function validate(path::AbstractString; expected_models = nothing)
    d = TOML.parsefile(path)
    models = get(d, "models", Dict())
    issues = String[]
    if expected_models !== nothing && length(models) != expected_models
        push!(issues, "row count $(length(models)) != expected $expected_models")
    end
    for k in sort(collect(keys(models)))
        cells = models[k]
        for c in MANDATORY_CELLS
            v = get(cells, c, nothing)
            (v isa Real && isfinite(v)) ||
                push!(issues, "$k: mandatory cell $c is not a finite number ($(repr(v)))")
        end
        for c in RK_REVERSE_CELLS
            v = get(cells, c, nothing)
            ok = (v isa Real && isfinite(v)) || (v isa AbstractString && !isempty(v))
            ok || push!(issues, "$k: RK-reverse cell $c must be a finite number or a nonempty core-defect diagnostic string ($(repr(v)))")
        end
        for c in CONDITIONAL_CELLS
            v = get(cells, c, nothing)
            ok = (v isa Real && isfinite(v)) || (v isa AbstractString && !isempty(v))
            ok || push!(issues, "$k: conditional cell $c must be a finite number or a nonempty reason/provenance string ($(repr(v)))")
        end
    end
    issues
end

end # module All80Receipt
