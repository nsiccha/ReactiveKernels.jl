#!/usr/bin/env julia

# Reproducible native-RK versus Reactant comparison for the exact centered
# Eight Schools model and primal HAVE/WANT matrix published by
# eight_schools_primal_comparison.jl. The authored model source is imported from
# ReactiveKernelsPPLExamples; it is never copied into this benchmark.

import Pkg

const _EIGHT_SCHOOLS_REACTANT_INNER = "RK_EIGHT_SCHOOLS_REACTANT_INNER"
const _EIGHT_SCHOOLS_REACTANT_VERSION = v"0.2.278"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-eight-schools-reactant-") do environment
        setup_seconds = precompile_seconds = 0.0
        withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
            setup_seconds = @elapsed begin
                Pkg.activate(environment)
                Pkg.add(Pkg.PackageSpec(
                    name = "Reactant", version = _EIGHT_SCHOOLS_REACTANT_VERSION))
                Pkg.pin(Pkg.PackageSpec(name = "Reactant"))
                Pkg.develop([
                    Pkg.PackageSpec(path = root),
                    Pkg.PackageSpec(path = joinpath(
                        root, "packages", "ReactiveKernelsDistributionKernels")),
                    Pkg.PackageSpec(path = joinpath(
                        root, "packages", "ReactiveKernelsPPLExamples")),
                ])
                Pkg.instantiate()
            end
            precompile_seconds = @elapsed Pkg.precompile()
        end
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(joinpath(@__DIR__, "eight_schools_reactant_comparison_body.jl")) $(ARGS...)`,
            _EIGHT_SCHOOLS_REACTANT_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "RK_EIGHT_SCHOOLS_ENV_SETUP_SECONDS" => string(setup_seconds),
            "RK_EIGHT_SCHOOLS_PRECOMPILE_SECONDS" => string(precompile_seconds),
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
        )
        run(command)
    end
end

get(ENV, _EIGHT_SCHOOLS_REACTANT_INNER, "") == "1" ?
    include(joinpath(@__DIR__, "eight_schools_reactant_comparison_body.jl")) :
    _run_pinned_comparison()
