# Versioned receipt schema + deterministic aggregator for the all-82 posteriordb
# benchmark. Process-isolation contract (parent-mandated): native and Reactant timings
# are produced in SEPARATE Julia subprocesses (loading Reactant must not perturb native
# compiler state), each writing a PHASE receipt; the single entrypoint aggregates them
# by model key into ONE final receipt. Pure Julia + stdlib TOML (no heavy deps), so it
# is usable from the measurement body, the orchestrator, and a standalone check.
module All80Receipt

import TOML, Dates, SHA

const SCHEMA = "all80-benchmark-v1"

# SOURCE/PROTOCOL IDENTITY — WIP, NOT YET WIRED (kept separate from the live incident per
# performance review 2026-09-08). Purpose: a resume REJECTS a leftover receipt from a DIFFERENT
# run/source (schema+phase alone cannot; an old checkpoint shares them) — what stops a stale
# cross-run receipt silently seeding rows (incident 2026-09-08). Before wiring into
# write_phase/resume, the reviewed design must: FREEZE the id ONCE at process start (recomputing
# per write_phase from disk would falsely stamp already-loaded OLD code with NEW file hashes);
# cover dirty MODEL/core source (git diff HEAD, not just harness files); fail-closed on missing
# files; treat no-git as UNCERTIFIED (never certify production); + a fixture simulating a
# mid-run file mutation. Current definition below is the pre-review sketch, intentionally unused.
const _HARNESS_FILES = ("all80_posteriordb_body.jl", "all80_registry.jl", "all80_axes.jl",
    "all80_reactant_body.jl", "all80_reactant_evals.jl", "all80_receipt.jl", "all80_parity.jl")
function source_identity()
    root = normpath(joinpath(@__DIR__, ".."))
    head = try
        strip(read(setenv(`git -C $root rev-parse HEAD`; dir = root), String))
    catch
        "no-git"
    end
    ctx = IOBuffer(); write(ctx, "head:", head, "\n")
    for f in _HARNESS_FILES
        p = joinpath(@__DIR__, f)
        isfile(p) && write(ctx, f, ":", read(p), "\n")
    end
    bytes2hex(SHA.sha256(take!(ctx)))
end

"""Verify a resumed prior receipt was produced by the CURRENT harness/source; throw otherwise.
Rejects a receipt with a missing/mismatched `source_id` (a leftover cross-run receipt)."""
function assert_resume_source!(prior, receipt_path)
    want = source_identity()
    got = get(prior, "source_id", "")
    got == want || error("all80 resume REFUSED: source_id " *
        (isempty(got) ? "ABSENT (pre-identity or foreign receipt)" : "MISMATCH (prior=$got)") *
        " ≠ current $want at $receipt_path — will not seed rows from a different run/source. " *
        "Delete that receipt and re-run fresh, or RK_ALL80_RETRY the exact keys.")
    nothing
end

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
# STRICTLY-MANDATORY (report-completeness): a finite real number for ALL 82 — the REFERENCE-side
# cells only (Turing/Stan primal, Turing/Stan reverse gradient, AdvancedHMC-Turing throughput).
# NONE depend on the RK graph, so no RK-side defect can make them absent. `primal_rk` moved to
# RK_CELLS (Fix D): a genuine RK primal defect (e.g. wells logistic saturation → -Inf where Stan
# is finite) yields a primal_rk DIAGNOSTIC + parity_pass=false, not a missing cell — the row stays
# report-complete, never a passing correctness verdict.
const MANDATORY_CELLS = ("primal_turing", "primal_stan",
    "gradient_turing", "gradient_stan", "hmc_ahmc_turing")
# RK CELLS (Fix D independent per-cell preservation): numeric where THAT RK operation is verified
# vs reference Stan (the unaffected models), else a NONEMPTY diagnostic string carrying the exact
# defect — RK primal (wells logistic saturation), RK reverse (authored-plate/scan Enzyme failure;
# dogs_hier direct-p boundary NaN), or RK-native HMC blocked by either. Present-and-typed, NEVER
# absent; a diagnostic cell forces parity_pass=false and prints NO benchmark ratio; republished as
# finite numbers once the per-model fix lands (canonicals bound per follow-on).
const RK_CELLS = ("primal_rk", "gradient_rk", "hmc_rk_native")
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
    "off_reason", "rk_grad_relerr", "tu_grad_relerr",
    "rk_stab", "tu_stab", "mag_rk", "mag_tu",   # scale-aware parity evidence (roundoff floor = STAB_ATOL + STAB_ULP_C·mag·eps)
    "rk_primal_ok", "rk_grad_ok",               # per-axis RK correctness flags (Fix D); parity_pass = false on any false
    "turing_support_ok", "turing_support_diag", # Fix E: false ⇒ non-equivalent Turing support (no RK/Turing ratio)
    "protocol")                                 # protocol stamp; absent ⇒ measured under the old (pre-scale-aware) protocol

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

"""REPORT-COMPLETENESS gate (DISTINCT from model correctness — see `correctness_failures`):
every model carries all MANDATORY reference cells as finite REAL NUMBERS, every RK cell as a
finite number OR a nonempty defect diagnostic, and every CONDITIONAL/OPTIONAL cell as a finite
number or a NONEMPTY provenance/reason string (merely-absent is a gate failure, not an implicit
N/A). A DOCUMENTED RK defect (a diagnostic RK cell) PASSES this gate — the row is complete — while
`parity_pass=false` records it is NOT a passing correctness verdict. Returns issues (empty == passes)."""
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
        for c in RK_CELLS
            v = get(cells, c, nothing)
            ok = (v isa Real && isfinite(v)) || (v isa AbstractString && !isempty(v))
            ok || push!(issues, "$k: RK cell $c must be a finite number or a nonempty defect diagnostic string ($(repr(v)))")
        end
        for c in CONDITIONAL_CELLS
            v = get(cells, c, nothing)
            ok = (v isa Real && isfinite(v)) || (v isa AbstractString && !isempty(v))
            ok || push!(issues, "$k: conditional cell $c must be a finite number or a nonempty reason/provenance string ($(repr(v)))")
        end
    end
    issues
end

"""MODEL-CORRECTNESS gate (DISTINCT from report-completeness): the rows that are NOT a verified
faithful comparison — a row is correct only when every RK cell is a finite number AND
`parity_pass` is true. Returns `key => reason` for each failing row (empty == all verified).
Report-completeness may PASS while this is non-empty: a documented RK defect is complete but not
correct. Use this for any published claim of repaired correctness — never `validate` alone."""
function correctness_failures(path::AbstractString)
    d = TOML.parsefile(path)
    fails = Dict{String,String}()
    for (k, cells) in get(d, "models", Dict())
        reasons = String[]
        get(cells, "parity_pass", false) == true || push!(reasons, "parity_pass=false")
        for c in RK_CELLS
            v = get(cells, c, nothing)
            (v isa Real && isfinite(v)) || push!(reasons, "$c not numeric")
        end
        isempty(reasons) || (fails[k] = join(reasons, "; "))
    end
    fails
end

end # module All80Receipt
