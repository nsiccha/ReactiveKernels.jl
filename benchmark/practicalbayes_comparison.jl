#!/usr/bin/env julia

# Reproducible PracticalBayes comparator for the PPL model and evaluation-
# throughput matrices. PracticalBayes is installed at an exact Git
# revision in a fresh environment; it never enters ReactiveKernels' dependency
# graph. Its Bijectors pin is incompatible with the Turing/DynamicPPL benchmark
# environment, so the results live in a separate receipt by construction.

import Pkg

const _PRACTICALBAYES_INNER = "RK_PRACTICALBAYES_INNER"
const PRACTICALBAYES_REPOSITORY =
    "https://github.com/EvoArt/PracticalBayes.git"
const PRACTICALBAYES_REVISION =
    "c6b340baef4f4a9e3d26cd0ea5082a2baf26dcf9"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-practicalbayes-") do environment
        setup_seconds = precompile_seconds = 0.0
        withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
            setup_seconds = @elapsed begin
                Pkg.activate(environment)
                Pkg.add(Pkg.PackageSpec(
                    url = PRACTICALBAYES_REPOSITORY,
                    rev = PRACTICALBAYES_REVISION))
                Pkg.add([
                    Pkg.PackageSpec(name = "ADTypes", version = v"1.24.0"),
                    Pkg.PackageSpec(name = "BenchmarkTools", version = v"1.6.3"),
                    Pkg.PackageSpec(name = "Distributions", version = v"0.25.131"),
                    Pkg.PackageSpec(name = "Enzyme", version = v"0.13.199"),
                    Pkg.PackageSpec(name = "LogDensityProblems", version = v"2.2.0"),
                    Pkg.PackageSpec(name = "MLDatasets", version = v"0.7.21"),
                    Pkg.PackageSpec(name = "NNlib", version = v"0.9.45"),
                ])
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
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(joinpath(@__DIR__, "practicalbayes_comparison_body.jl")) $(ARGS...)`,
            _PRACTICALBAYES_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "RK_PRACTICALBAYES_ENV_SETUP_SECONDS" => string(setup_seconds),
            "RK_PRACTICALBAYES_PRECOMPILE_SECONDS" => string(precompile_seconds),
            "RK_PRACTICALBAYES_REPOSITORY" => PRACTICALBAYES_REPOSITORY,
            "RK_PRACTICALBAYES_REVISION" => PRACTICALBAYES_REVISION,
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            "DATADEPS_ALWAYS_ACCEPT" => "true",
        )
        run(command)
    end
end

get(ENV, _PRACTICALBAYES_INNER, "") == "1" ?
    include(joinpath(@__DIR__, "practicalbayes_comparison_body.jl")) :
    _run_pinned_comparison()
