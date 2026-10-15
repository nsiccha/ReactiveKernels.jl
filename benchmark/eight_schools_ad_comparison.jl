#!/usr/bin/env julia

# Reproducible AD-only comparison for the exact published Eight Schools graph
# and the exact optimized Turing/manual definitions used by the primal matrix.
# Planning, AD preparation, and first execution are recorded separately from
# steady-state value-and-gradient timing and allocation measurements.

import Pkg

const _EIGHT_SCHOOLS_AD_INNER = "RK_EIGHT_SCHOOLS_AD_INNER"

include(joinpath(@__DIR__, "_repro_guard.jl"))
include(joinpath(@__DIR__, "_enzyme_variants.jl"))
using .EnzymeVariants

# One run per Enzyme variant (registered, then local dogfood); see
# `_enzyme_variants.jl` for the variant receipts and matrix file.
function _run_pinned_eight_schools_ad()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    run_enzyme_variants(; suite = "eight_schools_ad_comparison", root,
            output = requested_output(ARGS)) do environment, variant
        Pkg.add([
            Pkg.PackageSpec(name = "BenchmarkTools", version = v"1.6.3"),
            Pkg.PackageSpec(name = "Turing", version = v"0.47.1"),
            Pkg.PackageSpec(name = "DynamicPPL", version = v"0.42.6"),
            Pkg.PackageSpec(name = "Distributions", version = v"0.25.131"),
            Pkg.PackageSpec(name = "DifferentiationInterface", version = v"0.7.21"),
            Pkg.PackageSpec(name = "Enzyme", version = v"0.13.199"),
        ])
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsDistributionKernels")),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsPPLExamples"));
            rk_source_dependencies()])
        select_enzyme!(variant)
        addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(joinpath(@__DIR__, "eight_schools_ad_comparison_body.jl")) $(variant_arguments(ARGS, variant))`,
            _EIGHT_SCHOOLS_AD_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
        )
    end
end

get(ENV, _EIGHT_SCHOOLS_AD_INNER, "") == "1" ?
    include(joinpath(@__DIR__, "eight_schools_ad_comparison_body.jl")) :
    _run_pinned_eight_schools_ad()
