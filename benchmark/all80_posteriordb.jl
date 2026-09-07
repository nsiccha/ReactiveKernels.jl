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

const _INNER = "RK_ALL80_INNER"
const _BODY = joinpath(@__DIR__, "all80_posteriordb_body.jl")
const _REACTANT_BODY = joinpath(@__DIR__, "all80_reactant_body.jl")
const DPPL_SHA = "6378673b0029518c8a698096eb4f8ab662333083"
const UPSTREAM_DIGESTS = Dict(
    "posteriordb_models.jl" => "a7ef985b93df973dd0598f0c772f3f002afa25be80829fc38a9e8cb7232413ad",
    "posteriordb.jl"        => "12a814438d69501b6f8a14e3dd3f8ae1e1cb278f7fbf14c43abe7485338fc1a5",
)
const ENV_DIR = joinpath(@__DIR__, "all80-env")

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
        `$(Base.julia_cmd()) --startup-file=no --project=$ENV_DIR $(body) $(ARGS...)`,
        _INNER => "1", "RK_ALL80_UPSTREAM" => up, "RK_ALL80_DPPL_SHA" => DPPL_SHA,
        "RK_ALL80_PHASE" => phase, "RK_ALL80_RECEIPT" => receipt,
    )
    phase == "reactant" ? addenv(cmd, "JULIA_NUM_PRECOMPILE_TASKS" => "1") : cmd
end

function _run()
    _ensure_env()
    up = _fetch_upstream()
    mkpath(RECEIPT_DIR)
    phases = String.(split(get(ENV, "RK_ALL80_PHASES", "native,reactant"), ','))
    phase_receipts = String[]
    for ph in phases
        rec = joinpath(RECEIPT_DIR, "all80-$(ph).toml")
        @info "RK_ALL80: phase=$ph in an isolated subprocess -> $rec"
        run(_phase_cmd(ph, up, rec))
        push!(phase_receipts, rec)
    end
    # Deterministic aggregation of the phase receipts into the final receipt.
    include(joinpath(@__DIR__, "all80_receipt.jl"))
    out = joinpath(RECEIPT_DIR, "all80-benchmark-v1.toml")
    Base.invokelatest(All80Receipt.aggregate, phase_receipts, out;
        meta = Dict("dppl_sha" => DPPL_SHA, "args" => collect(ARGS)))
    @info "RK_ALL80: aggregated final receipt -> $out"
    non_flag = [a for a in ARGS if !startswith(a, "-")]
    complete = isempty(non_flag) && Set(phases) == Set(["native", "reactant"])
    # A complete default run also gates the ROW COUNT (all 82 present), not just cell fill.
    issues = Base.invokelatest(All80Receipt.validate, out; expected_models = complete ? 82 : nothing)
    if !isempty(issues)
        if complete
            @error "RK_ALL80: publication gate FAILED — $(length(issues)) unpopulated mandatory cell(s)"
            foreach(i -> println("  ", i), first(issues, 30))
            exit(1)
        else
            @warn "RK_ALL80: PARTIAL run — $(length(issues)) mandatory cell(s) unpopulated (expected for a key-subset or single-phase run)"
            foreach(i -> println("  ", i), first(issues, 20))
        end
    end
end

get(ENV, _INNER, "") == "1" ? include(_BODY) : _run()
