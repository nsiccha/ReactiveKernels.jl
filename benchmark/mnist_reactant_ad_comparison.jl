#!/usr/bin/env julia

# Reproducible native-RK-AD versus Reactant-compiled-AD comparison for the exact
# MNIST multinomial-logistic model, reusing the SAME derivative outcome/boundary
# protocol published by the AD-only receipt (mnist-logistic-ad-v2.toml). This is
# the AD analog of mnist_reactant_comparison.jl. The authored model source is
# imported from ReactiveKernelsPPLExamples; no density or AD evaluator is ever
# copied into this benchmark, and the Reactant-compiled gradient is obtained
# from the first-class RK verb `compile_ad_value_and_gradient`.

import Pkg

const _MNIST_REACTANT_AD_INNER = "RK_MNIST_REACTANT_AD_INNER"
const _MNIST_REACTANT_AD_VERSION = v"0.2.278"

_mnist_reactant_ad_body() = any(
    ==("--dataset=wren-pca40"), ARGS) ?
    "mnist_reactant_ad_wren_pca40_comparison_body.jl" :
    "mnist_reactant_ad_comparison_body.jl"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    body = joinpath(@__DIR__, _mnist_reactant_ad_body())
    mktempdir(prefix = "reactivekernels-mnist-reactant-ad-") do environment
        setup_seconds = precompile_seconds = 0.0
        withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
            setup_seconds = @elapsed begin
                Pkg.activate(environment)
                Pkg.add([
                    Pkg.PackageSpec(
                        name = "Reactant", version = _MNIST_REACTANT_AD_VERSION),
                    Pkg.PackageSpec(name = "BenchmarkTools", version = v"1.6.3"),
                    Pkg.PackageSpec(name = "MLDatasets", version = v"0.7.21"),
                ])
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
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $body $ARGS`,
            _MNIST_REACTANT_AD_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "RK_MNIST_REACTANT_AD_ENV_SETUP_SECONDS" => string(setup_seconds),
            "RK_MNIST_REACTANT_AD_PRECOMPILE_SECONDS" => string(precompile_seconds),
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            "DATADEPS_ALWAYS_ACCEPT" => "true",
        )
        run(command)
    end
end

get(ENV, _MNIST_REACTANT_AD_INNER, "") == "1" ?
    include(joinpath(@__DIR__, _mnist_reactant_ad_body())) :
    _run_pinned_comparison()
