#!/usr/bin/env julia

# Reproducible primal-only comparison of the centered Eight Schools model.
# The timed operations are model views that DynamicPPL exposes directly and
# ReactiveKernels exposes as HAVE/WANT cuts of one authored graph. Setup,
# preparation, transform discovery, and package loading are outside timing.
# Gradient performance belongs to the top-level AD benchmark and docs page.

import Pkg

const _EIGHT_SCHOOLS_COMPARISON_INNER = "RK_EIGHT_SCHOOLS_COMPARISON_INNER"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-eight-schools-comparison-") do environment
        Pkg.activate(environment)
        Pkg.add([
            Pkg.PackageSpec(name = "BenchmarkTools", version = v"1.6.3"),
            Pkg.PackageSpec(name = "Turing", version = v"0.47.1"),
            Pkg.PackageSpec(name = "DynamicPPL", version = v"0.42.6"),
            Pkg.PackageSpec(name = "Distributions", version = v"0.25.131"),
        ])
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsDistributionKernels")),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsPPLExamples")),
        ])
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(joinpath(@__DIR__, "eight_schools_primal_comparison_body.jl")) $(ARGS...)`,
            _EIGHT_SCHOOLS_COMPARISON_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
        )
        run(command)
    end
end

get(ENV, _EIGHT_SCHOOLS_COMPARISON_INNER, "") == "1" ?
    include(joinpath(@__DIR__, "eight_schools_primal_comparison_body.jl")) :
    _run_pinned_comparison()
