#!/usr/bin/env julia

# Reproducible AD-only comparison for the exact published MNIST logistic graph
# and the exact Turing/manual comparator definitions used by its primal matrix.
# Data loading, AD preparation, and first execution are retained separately from
# steady-state value-and-gradient timing and allocation measurements.

import Pkg

const _MNIST_LOGISTIC_AD_INNER = "RK_MNIST_LOGISTIC_AD_INNER"

_mnist_logistic_ad_body() = any(==("--dataset=wren-pca40"), ARGS) ?
    "mnist_logistic_ad_wren_pca40_comparison_body.jl" :
    "mnist_logistic_ad_comparison_body.jl"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_mnist_logistic_ad()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-mnist-logistic-ad-") do environment
        Pkg.activate(environment)
        Pkg.add([
            Pkg.PackageSpec(name = "BenchmarkTools", version = v"1.6.3"),
            Pkg.PackageSpec(name = "Turing", version = v"0.47.1"),
            Pkg.PackageSpec(name = "DynamicPPL", version = v"0.42.6"),
            Pkg.PackageSpec(name = "Distributions", version = v"0.25.131"),
            Pkg.PackageSpec(name = "NNlib", version = v"0.9.45"),
            Pkg.PackageSpec(name = "MLDatasets", version = v"0.7.21"),
            Pkg.PackageSpec(name = "DifferentiationInterface", version = v"0.7.21"),
            Pkg.PackageSpec(name = "Enzyme", version = v"0.13.199"),
        ])
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsDistributionKernels")),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsPPLExamples")),
        ])
        body = joinpath(@__DIR__, _mnist_logistic_ad_body())
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $body $ARGS`,
            _MNIST_LOGISTIC_AD_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            "DATADEPS_ALWAYS_ACCEPT" => "true",
        )
        run(command)
    end
end

get(ENV, _MNIST_LOGISTIC_AD_INNER, "") == "1" ?
    include(joinpath(@__DIR__, _mnist_logistic_ad_body())) :
    _run_pinned_mnist_logistic_ad()
