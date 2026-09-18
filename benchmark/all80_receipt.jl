# Versioned receipt schema + deterministic aggregator for the all-82 posteriordb
# benchmark. Process-isolation contract (parent-mandated): native and Reactant timings
# are produced in SEPARATE Julia subprocesses (loading Reactant must not perturb native
# compiler state), each writing a PHASE receipt; the single entrypoint aggregates them
# by model key into ONE final receipt. Pure Julia + stdlib TOML (no heavy deps), so it
# is usable from the measurement body, the orchestrator, and a standalone check.
module All80Receipt

import TOML, Dates, SHA

const SCHEMA = "all80-benchmark-v1"

const _HARNESS_FILES = (
    "all80_posteriordb.jl", "all80_posteriordb_body.jl", "all80_reactant_body.jl",
    "all80_reactant_evals.jl", "all80_registry.jl", "all80_receipt.jl",
    "all80_axes.jl", "all80_metadata.jl", "all80_parity.jl")
const _HMC_HARNESS_FILES = (
    "sampler_transpiler/eight_schools_density.jl",
    "nuts_kernel_authoring_fixture_b.jl",
    "sampler_transpiler/position_multinomial_hmc_kernel.jl")
const _ENV_FILES = ("Project.toml", "Manifest.toml")
const ORDINARY_AD_BACKEND = "AutoEnzyme(mode=Enzyme.Reverse; no function_annotation)"

# ---- Incremental process-start provenance -------------------------------------------------
# A batch publication must distinguish THREE facts: (1) the bytes selected by the parent before
# spawning Julia; (2) the modules actually loaded by that subprocess; and (3) the on-disk bytes at
# every incremental receipt write. The parent freezes (1) before the phase command, the child loads
# and verifies it before heavy imports, certifies (2) through `pathof(module)` after imports, and
# `verify_provenance()` rechecks (3) before every write. A post-import hash alone cannot make this
# claim, and an uncertified/no-git snapshot is never publication-certified.
const _PROV_START = Ref{Dict{String,Any}}()

_git_bytes(root, args...) = try
    read(pipeline(setenv(Cmd(String["git", "-C", root, args...]); dir = root); stderr = devnull))
catch
    nothing
end
_git_at(root, args...) = begin
    bytes = _git_bytes(root, args...)
    bytes === nothing ? "?" : strip(String(bytes))
end
_is_commit(s) = occursin(r"^[0-9a-f]{40}$", s)
_file_sha256(path) = bytes2hex(SHA.sha256(read(path)))

function _source_tree_hash(pkgroot)
    isdir(joinpath(pkgroot, "src")) || error("provenance package has no src tree: $pkgroot")
    files = String[]
    for subdir in ("src", "ext")
        rootdir = joinpath(pkgroot, subdir)
        isdir(rootdir) || continue
        for (root, _, names) in walkdir(rootdir)
            for name in names
                endswith(name, ".jl") && push!(files, joinpath(root, name))
            end
        end
    end
    isfile(joinpath(pkgroot, "Project.toml")) && push!(files, joinpath(pkgroot, "Project.toml"))
    ctx = IOBuffer()
    for path in sort(files)
        write(ctx, relpath(path, pkgroot), ":", _file_sha256(path), "\n")
    end
    bytes2hex(SHA.sha256(take!(ctx)))
end

function _git_snapshot(root)
    head = _git_at(root, "rev-parse", "HEAD")
    status = _git_at(root, "status", "--porcelain")
    diff = _git_bytes(root, "diff", "HEAD", "--")
    diff_digest = diff === nothing ? "?" : bytes2hex(SHA.sha256(diff))
    (; head = head, status = status,
       diff_sha256 = diff_digest,
       certified = _is_commit(head) && status != "?" && diff_digest != "?")
end

function _package_snapshot(root)
    git = _git_snapshot(root)
    Dict{String,Any}(
        "root" => normpath(root), "git_head" => git.head,
        "git_dirty" => !isempty(git.status), "git_status" => git.status,
        "git_diff_head_sha256" => git.diff_sha256,
        "git_certified" => git.certified,
        "src_ext_project_sha256" => _source_tree_hash(root))
end

function _snapshot_source_id(s)
    stable = Dict{String,Any}()
    for (key, value) in s
        key in ("captured_at", "run_id", "loaded_modules_certified",
                "loaded_module_paths", "module_versions") && continue
        stable[key] = value
    end
    ctx = IOBuffer(); TOML.print(ctx, stable; sorted = true)
    bytes2hex(SHA.sha256(take!(ctx)))
end

"""Freeze the parent-side source/environment snapshot before spawning a phase subprocess.
`harness_files` are absolute, or relative to this benchmark directory, and every path must exist.
`packages` maps a package name to its developed root. `upstream_files` maps each fetched upstream
file to its byte digest. `extra` carries phase/run/query configuration (including `run_id`)."""
function freeze_provenance!(; packages = Dict{String,String}(), upstream_hash = "",
        upstream_files = Dict{String,String}(), harness_files = _HARNESS_FILES,
        extra = Dict{String,Any}())
    isempty(harness_files) && error("provenance freeze requires the phase's harness files")
    harness = Dict{String,String}()
    for name in harness_files
        path = isabspath(name) ? name : joinpath(@__DIR__, name)
        isfile(path) || error("provenance harness file missing: $path")
        harness[name] = _file_sha256(path)
    end
    isempty(packages) && error("provenance freeze requires the developed-package roots")

    pkg = Dict{String,Any}()
    for (name, root) in sort(collect(packages); by = first)
        pkg[String(name)] = _package_snapshot(String(root))
    end

    upstream = Dict{String,Any}()
    for (path, digest) in sort(collect(upstream_files); by = first)
        upstream[String(path)] = Dict{String,Any}("sha256" => String(digest))
    end
    environment = Dict{String,Any}()
    for name in _ENV_FILES
        path = joinpath(@__DIR__, "all80-env", name)
        environment[name] = _file_sha256(path)
    end

    snap = merge(Dict{String,Any}(
        "captured_at" => string(Dates.now()),
        "root" => normpath(joinpath(@__DIR__, "..")),
        "harness_file_sha256" => harness,
        "packages" => pkg,
        "upstream_posteriordb_models_sha256" => String(upstream_hash),
        "upstream_file_sha256" => upstream,
        "environment_file_sha256" => environment,
        "julia_parent" => string(VERSION),
        "loaded_modules_certified" => false), extra)
    snap["source_id"] = _snapshot_source_id(snap)
    _PROV_START[] = snap
    _PROV_START[]
end

"""Write a parent-side lock. The child loads this before importing the measured packages."""
function write_provenance_lock(path::AbstractString,
        snapshot = _PROV_START[])
    open(path, "w") do io
        TOML.print(io, snapshot; sorted = true)
    end
    path
end

"""Load and immediately verify a parent-side lock in the child process."""
function load_provenance!(path::AbstractString)
    isfile(path) || error("provenance lock missing: $path")
    _PROV_START[] = TOML.parsefile(path)
    verify_provenance()
end

"""Certify that the loaded Julia modules resolve inside the exact package roots frozen by the
parent. This closes the “post-import filesystem hash” gap: source bytes and loaded roots are both
recorded. No-git or unavailable module paths remain uncertified and throw for a batch publication."""
function certify_loaded_modules!(modules::AbstractDict)
    isassigned(_PROV_START) || error("certify_loaded_modules!: no frozen provenance")
    snap = _PROV_START[]
    expected_modules = sort(String.(collect(keys(snap["packages"]))))
    Set(String.(keys(modules))) == Set(expected_modules) ||
        error("certify_loaded_modules!: module map must exactly cover $expected_modules")
    paths = Dict{String,String}()
    versions = Dict{String,String}()
    for (name, module_or_name) in modules
        key = String(name)
        info = get(snap["packages"], key, nothing)
        info === nothing && error("loaded module not represented in provenance: $key")
        value = module_or_name
        if value isa Symbol || value isa AbstractString
            matches = [m for m in values(Base.loaded_modules)
                       if String(nameof(m)) == String(value)]
            length(matches) == 1 || error("loaded module name is absent or ambiguous: $value")
            value = only(matches)
        end
        path = pathof(value)
        if path === nothing
            conventional = joinpath(String(info["root"]), "src",
                String(nameof(value)) * ".jl")
            isfile(conventional) && (path = conventional)
        end
        path === nothing && error("loaded module has no source path: $key")
        # normpath keeps a trailing separator when the input resolves through ".."
        # (normpath("…/benchmark/..") == "…/" on Julia 1.10) — the frozen roots ARE that
        # shape (normpath(joinpath(@__DIR__, ".."))). Strip it or the join below doubles
        # the separator and rejects every in-root path (live DISCOVER failure: the printed
        # path visibly inside the printed root still failed).
        root = rstrip(normpath(String(info["root"])), '/')
        startswith(normpath(path), root * Base.Filesystem.path_separator) ||
            error("loaded module root mismatch for $key: $path is outside $root")
        paths[key] = normpath(path)
        versions[key] = try string(Base.pkgversion(value)) catch; "unavailable" end
    end
    all(info -> info["git_certified"], values(snap["packages"])) ||
        error("provenance is not Git-certified (missing/no-git package root)")
    snap["loaded_module_paths"] = paths
    snap["module_versions"] = versions
    snap["loaded_modules_certified"] = true
    verify_provenance()
end

"""Re-hash every frozen byte source and return the immutable snapshot, or refuse the receipt."""
function verify_provenance()
    isassigned(_PROV_START) || error("verify_provenance: freeze_provenance! was not called at process start")
    s = _PROV_START[]
    for (name, digest) in s["harness_file_sha256"]
        path = joinpath(@__DIR__, name)
        h = isfile(path) ? _file_sha256(path) : "MISSING"
        h == digest || error("provenance DRIFT: harness `$name` changed since freeze ($digest → $h)")
    end
    for (name, digest) in s["environment_file_sha256"]
        path = joinpath(@__DIR__, "all80-env", name)
        h = isfile(path) ? _file_sha256(path) : "MISSING"
        h == digest || error("provenance DRIFT: environment `$name` changed since freeze ($digest → $h)")
    end
    for (path, info) in s["upstream_file_sha256"]
        h = isfile(path) ? _file_sha256(path) : "MISSING"
        h == info["sha256"] ||
            error("provenance DRIFT: upstream file `$path` changed since freeze")
    end
    for (name, info) in s["packages"]
        h = _source_tree_hash(String(info["root"]))
        h == info["src_ext_project_sha256"] ||
            error("provenance DRIFT: package `$name` src/ext/Project bytes changed since freeze")
    end
    s
end

"""Exact-key batch resume gate. Old/foreign checkpoints cannot seed a new batch run."""
function assert_batch_resume!(prior, receipt_path; phase, targets, provenance)
    want_keys = sort(String.(collect(targets)))
    got_phase = String(get(prior, "phase", ""))
    got_batch = String(get(get(prior, "provenance", Dict()), "batch", ""))
    got_keys = sort(String.(get(get(prior, "provenance", Dict()), "requested_keys", String[])))
    got_source = String(get(prior, "source_id", ""))
    got_run = String(get(prior, "run_id", ""))
    want_source = String(get(provenance, "source_id", ""))
    want_run = String(get(provenance, "run_id", ""))
    got_phase == String(phase) || error("batch resume REFUSED: phase $got_phase != $phase in $receipt_path")
    got_batch == String(get(provenance, "batch", "")) ||
        error("batch resume REFUSED: batch identity mismatch in $receipt_path")
    got_keys == want_keys ||
        error("batch resume REFUSED: requested keys $(repr(got_keys)) != $(repr(want_keys)) in $receipt_path")
    got_source == want_source && !isempty(got_source) ||
        error("batch resume REFUSED: source_id absent/foreign in $receipt_path")
    got_run == want_run && !isempty(got_run) ||
        error("batch resume REFUSED: run_id absent/foreign in $receipt_path")
    prior_keys = sort(collect(String, keys(get(prior, "models", Dict()))))
    isempty(setdiff(Set(prior_keys), Set(want_keys))) ||
        error("batch resume REFUSED: prior rows are outside the exact requested key set in $receipt_path")
    nothing
end

"""Normalize the per-model input/query identity recorded by each phase. The caller supplies
already-measured paths/digests and query fields; this helper only gives both phases one schema."""
function input_identity(; phase, posterior, model, stan_path, stan_sha256,
        library, library_sha256, dataset_path, dataset_json_sha256,
        dataset_json_bytes, query, stan_perm)
    Dict{String,Any}(
        "phase" => String(phase), "posterior" => String(posterior),
        "model" => String(model), "stan_path" => String(stan_path),
        "stan_sha256" => String(stan_sha256), "stan_compiled_library" => String(library),
        "stan_compiled_library_sha256" => String(library_sha256),
        "dataset_path" => String(dataset_path),
        "dataset_json_sha256" => String(dataset_json_sha256),
        "dataset_json_bytes" => Int(dataset_json_bytes),
        "query" => query,
        "stan_parameter_permutation" =>
            stan_perm === nothing ? "identity" : collect(stan_perm))
end

function _provenance_schema_issues(snapshot, phase)
    issues = String[]
    for field in ("source_id", "run_id", "batch", "requested_keys",
                  "harness_file_sha256", "upstream_file_sha256",
                  "environment_file_sha256", "packages", "loaded_module_paths",
                  "module_versions")
        haskey(snapshot, field) || push!(issues, "$phase provenance lacks $field")
    end
    String(get(snapshot, "phase", "")) == String(phase) ||
        push!(issues, "$phase provenance phase mismatch")
    get(snapshot, "loaded_modules_certified", false) === true ||
        push!(issues, "$phase loaded modules are uncertified")
    packages = get(snapshot, "packages", Dict())
    packages isa AbstractDict && !isempty(packages) ||
        push!(issues, "$phase package identity map is absent/empty")
    loaded = get(snapshot, "loaded_module_paths", Dict())
    loaded isa AbstractDict && !isempty(loaded) &&
        Set(String.(keys(loaded))) == Set(String.(keys(packages))) ||
        push!(issues, "$phase loaded-module map does not exactly cover package identities")
    issues
end

function selected_point_stan_parity_ok(value, stan_value, declared_offset;
        rtol = 1e-6, atol = 1e-6)
    expected = stan_value - declared_offset
    isfinite(value) && isfinite(expected) &&
        abs(value - expected) <= max(atol, rtol * max(abs(expected), 1.0))
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

"""Write ONE phase's receipt. `rows` maps model-key => Dict{String,Any} of that phase's cells.
`provenance` (OPT-IN, batch-1 only) writes a process-start `provenance` block; `nothing` (the
default, used by the frozen-82 flow) omits it, keeping that receipt byte-for-byte as before."""
function write_phase(path::AbstractString, phase, rows::AbstractDict; provenance = nothing)
    doc = Dict("schema" => SCHEMA, "phase" => String(phase),
               "generated_at" => string(Dates.now()),
               "models" => Dict(String(k) => v for (k, v) in rows))
    if provenance !== nothing
        doc["provenance"] = provenance
        haskey(provenance, "source_id") && (doc["source_id"] = provenance["source_id"])
        haskey(provenance, "run_id") && (doc["run_id"] = provenance["run_id"])
    end
    mkpath(dirname(path))
    open(path, "w") do io; TOML.print(io, doc; sorted = true); end
    path
end

"""Merge phase receipts (native ∪ reactant, per model key) into the final receipt.
Phases carry disjoint cell sets by construction; a cell that appears in two phases must
AGREE — a conflicting duplicate is a HARD ERROR, never a silent overwrite. Model ordering
and cell content are deterministic (TOML-sorted); the only non-reproducible field is the
`generated_at` timestamp, so the doc is content-stable, not byte-stable."""
function aggregate(phase_paths, out_path::AbstractString; meta = Dict{String,Any}(),
                   preserve_provenance = false, require_phases = String[],
                   expected_keys = String[], batch = "")
    merged = Dict{String,Dict{String,Any}}()
    prov = Dict{String,Any}()   # per-phase process-start provenance (batch-1); native + reactant kept BOTH
    seen_phases = String[]
    run_ids = String[]
    strict = !isempty(require_phases)
    strict && Set(String.(require_phases)) == Set(["native", "reactant"]) ||
        isempty(require_phases) || error("All80Receipt.aggregate: required phases must be exactly native+reactant")
    strict && length(phase_paths) == length(require_phases) ||
        isempty(require_phases) ||
        error("All80Receipt.aggregate: expected $(length(require_phases)) phase receipts, got $(length(phase_paths))")
    expected = sort(String.(expected_keys))
    for p in phase_paths
        isfile(p) || error("All80Receipt.aggregate: missing phase receipt $p")
        d = TOML.parsefile(p)
        get(d, "schema", "") == SCHEMA || error("schema mismatch in $p: $(get(d,"schema",""))")
        ph = String(get(d, "phase", "?"))
        ph in seen_phases && error("All80Receipt.aggregate: duplicate phase $ph (from $p)")
        push!(seen_phases, ph)
        if strict
            ph in require_phases || error("All80Receipt.aggregate: unexpected phase $ph in $p")
            haskey(d, "provenance") || error("All80Receipt.aggregate: phase $ph lacks provenance ($p)")
            snapshot = d["provenance"]
            schema_issues = _provenance_schema_issues(snapshot, ph)
            isempty(schema_issues) ||
                error("All80Receipt.aggregate: $(join(schema_issues, "; ")) (from $p)")
            String(get(d, "source_id", "")) == String(get(snapshot, "source_id", "")) ||
                error("All80Receipt.aggregate: phase $ph top-level source_id mismatch in $p")
            String(get(d, "run_id", "")) == String(get(snapshot, "run_id", "")) ||
                error("All80Receipt.aggregate: phase $ph top-level run_id mismatch in $p")
            isempty(batch) || String(get(snapshot, "batch", "")) == batch ||
                error("All80Receipt.aggregate: phase $ph batch mismatch in $p")
            String(get(snapshot, "phase", "")) == ph ||
                error("All80Receipt.aggregate: phase $ph provenance phase mismatch in $p")
            sort(String.(get(snapshot, "requested_keys", String[]))) == expected ||
                error("All80Receipt.aggregate: phase $ph requested-key mismatch in $p")
            get(snapshot, "loaded_modules_certified", false) === true ||
                error("All80Receipt.aggregate: phase $ph loaded modules are uncertified in $p")
            String(get(snapshot, "ad_backend", "")) != ORDINARY_AD_BACKEND &&
                error("All80Receipt.aggregate: phase $ph AD backend is not ordinary reverse in $p")
            push!(run_ids, String(get(snapshot, "run_id", "")))
            phase_keys = sort(collect(String, keys(get(d, "models", Dict()))))
            phase_keys == expected ||
                error("All80Receipt.aggregate: phase $ph model keys differ from the exact requested batch in $p")
        end
        if preserve_provenance && haskey(d, "provenance")
            haskey(prov, ph) && error("All80Receipt.aggregate: duplicate provenance for phase $ph (from $p)")
            prov[ph] = d["provenance"]   # the LAST phase never overwrites the first — keyed by phase
        end
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
    strict && !(length(Set(run_ids)) == 1 && all(!isempty, run_ids)) &&
        error("All80Receipt.aggregate: phase run identities differ")
    meta_out = preserve_provenance && !isempty(prov) ?
        merge(Dict{String,Any}(meta), Dict("provenance" => prov)) : meta
    doc = Dict("schema" => SCHEMA, "generated_at" => string(Dates.now()),
               "meta" => meta_out, "models" => merged)
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

"""Strict incremental-batch publication contract: exact phases/keys/multiplicity, matching
per-phase provenance and run id, certified loaded roots, recorded configuration, per-model input
identities, plus the ordinary report-completeness gate. Missing Reactant provenance in the saved
batch is therefore an explicit certification failure, not a silently accepted aggregate."""
function validate_batch(path::AbstractString; expected_keys, phases = ("native", "reactant"),
        batch = "")
    d = TOML.parsefile(path)
    issues = String[]
    push!(issues, validate(path; expected_models = length(expected_keys))...)
    get(d, "schema", "") == SCHEMA || push!(issues, "schema must be $SCHEMA")
    meta = get(d, "meta", Dict())
    isempty(batch) || String(get(meta, "batch", "")) == batch ||
        push!(issues, "meta.batch must be $batch")
    args = String.(get(meta, "args", String[]))
    args == String.(collect(expected_keys)) ||
        push!(issues, "meta.args must equal the exact requested key sequence")

    expected = sort(String.(collect(expected_keys)))
    models = get(d, "models", Dict())
    sort(collect(String, keys(models))) == expected ||
        push!(issues, "model key set differs from the exact requested batch")
    provenance = get(meta, "provenance", Dict())
    Set(String.(keys(provenance))) == Set(String.(collect(phases))) ||
        push!(issues, "provenance phase set differs from the exact requested phases")
    run_ids = String[]
    for phase in phases
        snapshot = get(provenance, String(phase), nothing)
        snapshot === nothing && (push!(issues, "missing $phase process-start provenance"); continue)
        append!(issues, _provenance_schema_issues(snapshot, phase))
        String(get(snapshot, "phase", "")) == String(phase) ||
            push!(issues, "$phase provenance phase mismatch")
        isempty(batch) || String(get(snapshot, "batch", "")) == batch ||
            push!(issues, "$phase provenance batch mismatch")
        sort(String.(get(snapshot, "requested_keys", String[]))) == expected ||
            push!(issues, "$phase provenance requested-key mismatch")
        get(snapshot, "loaded_modules_certified", false) === true ||
            push!(issues, "$phase loaded-module roots are uncertified")
        ad = String(get(snapshot, "ad_backend", ""))
        ad == ORDINARY_AD_BACKEND ||
            push!(issues, "$phase AD backend is not the ordinary reverse configuration")
        push!(run_ids, String(get(snapshot, "run_id", "")))
    end
    isempty(run_ids) || (length(Set(run_ids)) == 1 && all(!isempty, run_ids)) ||
        push!(issues, "phase run identities are absent or differ")
    for key in expected
        haskey(models, key) || continue
        _input_query_identity_ok(get(models[key], "input_identity_native", nothing),
            models[key], "native", key) ||
            push!(issues, "$key lacks a complete native input/query identity")
        _input_query_identity_ok(get(models[key], "input_identity_reactant", nothing),
            models[key], "reactant", key) ||
            push!(issues, "$key lacks a complete Reactant input/query identity")
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

function _input_query_identity_ok(identity, row, phase, key)
    identity isa AbstractDict || return false
    String(get(identity, "phase", "")) == String(phase) || return false
    String(get(identity, "posterior", "")) == String(key) || return false
    !isempty(String(get(identity, "model", ""))) || return false
    for field in ("stan_path", "stan_compiled_library", "dataset_path")
        !isempty(String(get(identity, field, ""))) || return false
    end
    for field in ("stan_sha256", "dataset_json_sha256")
        occursin(r"^[0-9a-f]{64}$", String(get(identity, field, ""))) || return false
    end
    library_digest = String(get(identity, "stan_compiled_library_sha256", ""))
    (library_digest == "not-present" ||
        occursin(r"^[0-9a-f]{64}$", library_digest)) || return false
    query = get(identity, "query", nothing)
    query isa AbstractDict || return false
    points = Vector{Vector{Float64}}(get(query, "points",
        Vector{Vector{Float64}}(Any[get(query, "selected_point", Float64[])])))
    dim = Int(get(row, "dim", -1))
    !isempty(points) || return false
    all(point -> point isa AbstractVector && length(point) == dim &&
        all(isfinite, point), points) || return false
    if haskey(query, "selected_point_index")
        Int(get(query, "selected_point_index", 0)) in eachindex(points) || return false
    end
    String(get(query, "ad_backend", "")) == ORDINARY_AD_BACKEND
end

end # module All80Receipt
