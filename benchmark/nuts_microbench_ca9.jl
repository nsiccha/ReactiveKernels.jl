#!/usr/bin/env julia

# Reproducible THREE-WAY decomposed microbenchmark that adds the actual public
# ReactiveHMC ca9 sampler as a distinct arm alongside ReactiveKernels' compiled
# CompiledNUTSState and its internal handwritten _OracleNUTSState.
#
#   julia --startup-file=no benchmark/nuts_microbench_ca9.jl
#
# The upstream packages live in a fresh temporary environment at EXACT pinned
# revisions (no moving branch); ReactiveKernels is developed from this tree. The
# internal _OracleNUTSState is a faithful port of ca9 and is reported as a distinct
# arm from the actual upstream package — never relabelled as the public baseline.
#
# Pinned provenance (validated to resolve + load on Julia 1.10):
#   ReactiveObjects  https://github.com/nsiccha/ReactiveObjects.jl @ 419881dc…
#   ReactiveHMC      https://github.com/nsiccha/ReactiveHMC.jl     @ ca9ea4ca…
#   (registered transitive deps: ElasticArrays, LambertW, LogExpFunctions, …)

import Pkg

const _CA9_INNER = "RK_CA9_MICROBENCH_INNER"
const _REACTIVE_OBJECTS_REV = "419881dcfe93fbf0c679837c5421322fbd2c6888"
const _REACTIVE_HMC_REV = "ca9ea4ca41924bb0e1fadc01c717e1333916aba6"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned()
    root = normpath(joinpath(@__DIR__, ".."))
    candidate_sha = _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-ca9-microbench-") do environment
        Pkg.activate(environment)
        Pkg.add(Pkg.PackageSpec(
            url = "https://github.com/nsiccha/ReactiveObjects.jl",
            rev = _REACTIVE_OBJECTS_REV))
        Pkg.add(Pkg.PackageSpec(
            url = "https://github.com/nsiccha/ReactiveHMC.jl",
            rev = _REACTIVE_HMC_REV))
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(root, "packages", "ReactiveKernelsNUTSExamples")),
        ])
        Pkg.precompile()
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(@__FILE__)`,
            _CA9_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => candidate_sha)
        run(command)
    end
end

function _inner()
    @eval begin
        using LinearAlgebra
        using Random
        using Pkg
        using ReactiveKernels
        using ReactiveKernelsNUTSExamples
        import ReactiveHMC
        import ReactiveHMC.ReactiveObjects: @invalidatedependants!
    end
    Base.include(@__MODULE__, joinpath(@__DIR__, "_ca9_microbench_body.jl"))
end

if get(ENV, _CA9_INNER, "") == "1"
    _inner()
else
    _run_pinned()
end
