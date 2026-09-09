#!/usr/bin/env julia
#
# All-implemented-model posteriordb benchmark — idiomatic RK-builtin graphs vs
# UPSTREAM optimized Turing vs REFERENCE (posteriordb) Stan, on FULL real data,
# every side parity-gated against reference Stan (BridgeStan) before timing.
#
# CONSUME-NOT-COPY: the upstream Turing models + benchmark driver are consumed as
# an upstream input at a PINNED, DIGEST-VERIFIED DynamicPPL revision (fetched, not
# vendored):
#   DynamicPPL.jl @ 6378673b0029518c8a698096eb4f8ab662333083
#     benchmarks/posteriordb_models.jl  sha256 a7ef985b93df973dd0598f0c772f3f002afa25be80829fc38a9e8cb7232413ad
#     benchmarks/posteriordb.jl         sha256 12a814438d69501b6f8a14e3dd3f8ae1e1cb278f7fbf14c43abe7485338fc1a5
# Reference Stan + data come through the PosteriorDB.jl dependency. RK graphs come
# from the developed ReactiveKernelsPPLExamples package.
#
# REPRODUCIBLE ENV: the comparator environment is PINNED in a checked-in
# benchmark/all80-env/{Project,Manifest}.toml. When the Manifest exists it is only
# `instantiate`d (never re-resolved), so external versions cannot float. First-time
# resolution (RK_ALL80_RESOLVE=1, or absent Manifest) adds+develops and writes the
# Manifest to be committed.

import Pkg, SHA
using Random: RandomDevice

const _INNER = "RK_ALL80_INNER"
const _BODY = joinpath(@__DIR__, "all80_posteriordb_body.jl")
const _REACTANT_BODY = joinpath(@__DIR__, "all80_reactant_body.jl")
const DPPL_SHA = "6378673b0029518c8a698096eb4f8ab662333083"
const UPSTREAM_DIGESTS = Dict(
    "posteriordb_models.jl" => "a7ef985b93df973dd0598f0c772f3f002afa25be80829fc38a9e8cb7232413ad",
    "posteriordb.jl"        => "12a814438d69501b6f8a14e3dd3f8ae1e1cb278f7fbf14c43abe7485338fc1a5",
)
const ENV_DIR = joinpath(@__DIR__, "all80-env")
include(joinpath(@__DIR__, "all80_receipt.jl"))

function _ensure_env()
    root = normpath(joinpath(@__DIR__, ".."))
    Pkg.activate(ENV_DIR)
    resolve = get(ENV, "RK_ALL80_RESOLVE", "") == "1" || !isfile(joinpath(ENV_DIR, "Manifest.toml"))
    if resolve
        Pkg.add([Pkg.PackageSpec(name = n) for n in (
            "Chairmarks","ADTypes","Enzyme","Mooncake","ForwardDiff","BridgeStan",
            "DynamicPPL","Distributions","Bijectors","FillArrays","StatsFuns",
            "DifferentiationInterface","LogExpFunctions","SpecialFunctions","PosteriorDB",
            "OrdinaryDiffEqBDF","OrdinaryDiffEqLowOrderRK","OrdinaryDiffEqTsit5",
            "SciMLBase","SciMLSensitivity",
            # coexistence superset for the RK+Reactant single-eval + HMC transpiler
            # (Reactant) and the AHMC-Turing HMC throughput axis (AdvancedHMC).
            "Reactant","AdvancedHMC")])
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(root, "packages", "ReactiveKernelsDistributionKernels")),
            Pkg.PackageSpec(path = joinpath(root, "packages", "ReactiveKernelsPPLExamples")),
        ])
        @info "RK_ALL80: env RESOLVED into $ENV_DIR — commit benchmark/all80-env/{Project,Manifest}.toml to pin"
    else
        Pkg.instantiate()
        @info "RK_ALL80: env INSTANTIATED from committed Manifest (pinned, no re-resolve)"
    end
end

function _fetch_upstream()
    up = joinpath(ENV_DIR, "upstream"); mkpath(up)
    base = "https://raw.githubusercontent.com/TuringLang/DynamicPPL.jl/$(DPPL_SHA)/benchmarks"
    for (f, want) in UPSTREAM_DIGESTS
        dst = joinpath(up, f)
        run(`curl -sS --fail-with-body -o $dst "$base/$f"`)
        got = bytes2hex(SHA.sha256(read(dst)))
        got == want || error("upstream digest mismatch for $f: got $got want $want")
    end
    up
end

const RECEIPT_DIR = joinpath(@__DIR__, "receipts")

# One isolated Julia subprocess per PHASE over the SAME pinned coexistence env. Native and
# Reactant timings must never share a process (loading Reactant perturbs native compiler
# state), so each phase re-execs this body with RK_ALL80_PHASE set and writes its own phase
# receipt; the reactant phase additionally pins JULIA_NUM_PRECOMPILE_TASKS=1 (the primer's
# ReactiveKernelsReactantExt precompile-wedge mitigation).
function _phase_cmd(phase, up, receipt)
    body = phase == "reactant" ? _REACTANT_BODY : _BODY
    cmd = addenv(
        `$(Base.julia_cmd()) --startup-file=no --project=$ENV_DIR $(body) $ARGS`,
        _INNER => "1", "RK_ALL80_UPSTREAM" => up, "RK_ALL80_DPPL_SHA" => DPPL_SHA,
        "RK_ALL80_PHASE" => phase, "RK_ALL80_RECEIPT" => receipt,
        "RK_ALL80_BATCH" => get(ENV, "RK_ALL80_BATCH", ""),
        "RK_ALL80_RUN_ID" => get(ENV, "RK_ALL80_RUN_ID", ""),
        "RK_ALL80_SOURCE_LOCK" => get(ENV, "RK_ALL80_SOURCE_LOCK", ""),
    )
    phase == "reactant" ? addenv(cmd, "JULIA_NUM_PRECOMPILE_TASKS" => "1") : cmd
end

function _phase_harness_files(phase)
    common = ("all80_posteriordb.jl", "all80_registry.jl", "all80_receipt.jl",
              "all80_axes.jl")
    phase == "native" ? (common..., "all80_posteriordb_body.jl",
        "all80_metadata.jl", "all80_parity.jl", _HMC_HARNESS_FILES...) :
        (common..., "all80_reactant_body.jl", "all80_reactant_evals.jl",
         _HMC_HARNESS_FILES...)
end

function _write_phase_source_lock(phase, up, requested_keys, run_id)
    upstream_files = Dict(
        joinpath(up, name) => bytes2hex(SHA.sha256(read(joinpath(up, name))))
        for name in keys(UPSTREAM_DIGESTS))
    root = normpath(joinpath(@__DIR__, ".."))
    snapshot = All80Receipt.freeze_provenance!(;
        packages = Dict(
            "ReactiveKernels" => root,
            "ReactiveKernelsDistributionKernels" =>
                joinpath(root, "packages", "ReactiveKernelsDistributionKernels"),
            "ReactiveKernelsPPLExamples" =>
                joinpath(root, "packages", "ReactiveKernelsPPLExamples")),
        upstream_hash = UPSTREAM_DIGESTS["posteriordb_models.jl"],
        upstream_files = upstream_files,
        harness_files = _phase_harness_files(phase),
        extra = Dict{String,Any}(
            "batch" => get(ENV, "RK_ALL80_BATCH", ""),
            "phase" => phase,
            "discover" => get(ENV, "RK_ALL80_DISCOVER", "") == "1",
            "requested_keys" => collect(requested_keys),
            "run_id" => run_id,
            "ad_backend" => All80Receipt.ORDINARY_AD_BACKEND))
    path = tempname() * ".toml"
    All80Receipt.write_provenance_lock(path, snapshot)
    path
end

function _run()
    _ensure_env()
    up = _fetch_upstream()
    mkpath(RECEIPT_DIR)
    # OPT-IN incremental batch (todo 1x4pytu/1q387t4): a SEPARATE receipt namespace + PRESERVED
    # process-start provenance, so the frozen-82 flow (batch unset) is byte-for-byte unchanged.
    batch = get(ENV, "RK_ALL80_BATCH", "")
    prefix = isempty(batch) ? "all80" : "all80-$(batch)"
    non_flag = [a for a in ARGS if !startswith(a, "-")]
    isempty(batch) || !isempty(non_flag) ||
        error("RK_ALL80_BATCH=$batch requires explicit model keys (refusing to run the default sweep into a batch receipt)")
    duplicated = unique(filter(key -> count(==(key), non_flag) > 1, non_flag))
    isempty(duplicated) ||
        error("RK_ALL80 duplicate model key(s) requested: $(join(duplicated, ", "))")
    phases = String.(split(get(ENV, "RK_ALL80_PHASES", "native,reactant"), ','))
    phase_receipts = String[]
    locks = String[]
    run_id = bytes2hex(rand(RandomDevice, UInt8, 16))
    for ph in phases
        rec = joinpath(RECEIPT_DIR, "$(prefix)-$(ph).toml")
        lock = isempty(batch) ? "" : _write_phase_source_lock(ph, up, non_flag, run_id)
        isempty(lock) || push!(locks, lock)
        withenv("RK_ALL80_RUN_ID" => run_id, "RK_ALL80_SOURCE_LOCK" => lock) do
            @info "RK_ALL80: phase=$ph in an isolated subprocess -> $rec"
            run(_phase_cmd(ph, up, rec))
        end
        push!(phase_receipts, rec)
    end
    # Deterministic aggregation of the phase receipts into the final receipt.
    out = joinpath(RECEIPT_DIR, "$(prefix)-v1.toml")
    meta = isempty(batch) ? Dict("dppl_sha" => DPPL_SHA, "args" => collect(ARGS)) :
                            Dict("dppl_sha" => DPPL_SHA, "args" => collect(ARGS), "batch" => batch)
    Base.invokelatest(All80Receipt.aggregate, phase_receipts, out;
        meta = meta, preserve_provenance = !isempty(batch),
        require_phases = isempty(batch) ? String[] : phases,
        expected_keys = isempty(batch) ? String[] : non_flag,
        batch = batch)
    @info "RK_ALL80: aggregated final receipt -> $out"
    complete = isempty(batch) && isempty(non_flag) && Set(phases) == Set(["native", "reactant"])
    issues = if isempty(batch)
        # A complete default run also gates the ROW COUNT (all 82 present), not just cell fill.
        Base.invokelatest(All80Receipt.validate, out;
            expected_models = complete ? 82 : nothing)
    else
        Base.invokelatest(All80Receipt.validate_batch, out;
            expected_keys = non_flag, phases = Tuple(phases), batch = batch)
    end
    if !isempty(issues)
        if complete || !isempty(batch)
            @error "RK_ALL80: publication gate FAILED — $(length(issues)) certification/completeness issue(s)"
            foreach(i -> println("  ", i), first(issues, 30))
            exit(1)
        else
            @warn "RK_ALL80: PARTIAL run — $(length(issues)) mandatory cell(s) unpopulated (expected for a key-subset or single-phase run)"
            foreach(i -> println("  ", i), first(issues, 20))
        end
    end
    foreach(lock -> rm(lock; force = true), locks)
end

get(ENV, _INNER, "") == "1" ? include(_BODY) : _run()
