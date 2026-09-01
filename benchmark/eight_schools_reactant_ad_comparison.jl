#!/usr/bin/env julia

# Reproducible native-RK-AD versus Reactant-compiled-AD comparison for the exact
# centered Eight Schools model, reusing the SAME derivative outcome/boundary
# protocol published by the AD-only receipt (eight-schools-ad-v1.toml). This is
# the AD analog of eight_schools_reactant_comparison.jl. The authored model
# source is imported from ReactiveKernelsPPLExamples; no density, transform, or
# AD evaluator is ever copied into this benchmark, and the Reactant-compiled
# gradient is obtained from the first-class RK verb `compile_ad_value_and_gradient`.
# Native structured gradients and pointwise pullbacks are measured through their
# public prepared RK verbs, but are never presented as Reactant-compiled surfaces.

import Pkg

const _EIGHT_SCHOOLS_REACTANT_AD_INNER = "RK_EIGHT_SCHOOLS_REACTANT_AD_INNER"
const _EIGHT_SCHOOLS_REACTANT_AD_VERSION = v"0.2.278"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-eight-schools-reactant-ad-") do environment
        setup_seconds = precompile_seconds = 0.0
        withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
            setup_seconds = @elapsed begin
                Pkg.activate(environment)
                Pkg.add(Pkg.PackageSpec(
                    name = "Reactant",
                    version = _EIGHT_SCHOOLS_REACTANT_AD_VERSION))
                Pkg.pin(Pkg.PackageSpec(name = "Reactant"))
                Pkg.add("Enzyme")
                Pkg.add("DifferentiationInterface")
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
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(joinpath(@__DIR__, "eight_schools_reactant_ad_comparison_body.jl")) $(ARGS...)`,
            _EIGHT_SCHOOLS_REACTANT_AD_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "RK_EIGHT_SCHOOLS_AD_ENV_SETUP_SECONDS" => string(setup_seconds),
            "RK_EIGHT_SCHOOLS_AD_PRECOMPILE_SECONDS" => string(precompile_seconds),
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
        )
        run(command)
    end
end

get(ENV, _EIGHT_SCHOOLS_REACTANT_AD_INNER, "") == "1" ?
    include(joinpath(@__DIR__, "eight_schools_reactant_ad_comparison_body.jl")) :
    _run_pinned_comparison()
