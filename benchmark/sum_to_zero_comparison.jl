#!/usr/bin/env julia

# Reproducible native primal/AD comparison for the sum-to-zero sampler hot
# loop. Reactant compilation has a separate receipt because it requires a much
# heavier pinned environment and has different timing rules.

import Pkg

const _SUM_TO_ZERO_COMPARISON_INNER = "RK_SUM_TO_ZERO_COMPARISON_INNER"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_sum_to_zero_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-sum-to-zero-comparison-") do environment
        Pkg.activate(environment)
        Pkg.add([
            Pkg.PackageSpec(name = "BenchmarkTools", version = v"1.6.3"),
            Pkg.PackageSpec(name = "Turing", version = v"0.47.1"),
            Pkg.PackageSpec(name = "DynamicPPL", version = v"0.42.6"),
            Pkg.PackageSpec(name = "Distributions", version = v"0.25.131"),
            Pkg.PackageSpec(
                name = "DifferentiationInterface", version = v"0.7.21"),
            Pkg.PackageSpec(name = "Enzyme", version = v"0.13.199"),
        ])
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsDistributionKernels")),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsPPLExamples")),
        ])
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(joinpath(@__DIR__, "sum_to_zero_comparison_body.jl")) $(ARGS...)`,
            _SUM_TO_ZERO_COMPARISON_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
        )
        run(command)
    end
end

get(ENV, _SUM_TO_ZERO_COMPARISON_INNER, "") == "1" ?
    include(joinpath(@__DIR__, "sum_to_zero_comparison_body.jl")) :
    _run_pinned_sum_to_zero_comparison()
