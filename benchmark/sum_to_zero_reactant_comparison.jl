#!/usr/bin/env julia

# Reproducible Reactant primal/AD comparison for the exact sum-to-zero sampler
# hot loop and the exact optimized manual control from the native receipt.

import Pkg

const _SUM_TO_ZERO_REACTANT_INNER = "RK_SUM_TO_ZERO_REACTANT_INNER"
const _SUM_TO_ZERO_REACTANT_VERSION = v"0.2.284"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_sum_to_zero_reactant_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-sum-to-zero-reactant-") do environment
        setup_seconds = precompile_seconds = 0.0
        withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
            setup_seconds = @elapsed begin
                Pkg.activate(environment)
                Pkg.add([
                    Pkg.PackageSpec(
                        name = "Reactant", version = _SUM_TO_ZERO_REACTANT_VERSION),
                    Pkg.PackageSpec(name = "BenchmarkTools", version = v"1.6.3"),
                    Pkg.PackageSpec(name = "Turing", version = v"0.47.1"),
                    Pkg.PackageSpec(name = "DynamicPPL", version = v"0.42.6"),
                    Pkg.PackageSpec(name = "Distributions", version = v"0.25.131"),
                    Pkg.PackageSpec(
                        name = "DifferentiationInterface", version = v"0.7.21"),
                    Pkg.PackageSpec(name = "Enzyme", version = v"0.13.199"),
                ])
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
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(joinpath(@__DIR__, "sum_to_zero_reactant_comparison_body.jl")) $(ARGS...)`,
            _SUM_TO_ZERO_REACTANT_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "RK_SUM_TO_ZERO_REACTANT_ENV_SETUP_SECONDS" => string(setup_seconds),
            "RK_SUM_TO_ZERO_REACTANT_PRECOMPILE_SECONDS" => string(precompile_seconds),
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
        )
        run(command)
    end
end

get(ENV, _SUM_TO_ZERO_REACTANT_INNER, "") == "1" ?
    include(joinpath(@__DIR__, "sum_to_zero_reactant_comparison_body.jl")) :
    _run_pinned_sum_to_zero_reactant_comparison()
