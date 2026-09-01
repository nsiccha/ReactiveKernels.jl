#!/usr/bin/env julia

# Reproducible primal-only comparison of the multinomial-logistic (softmax) MNIST
# classifier. The timed operations are HAVE/WANT cuts of one authored RK graph, a
# handwritten Julia control, and Turing's native public interfaces. Full MNIST is
# loaded via MLDatasets; data loading, kernel/LogDensityFunction preparation,
# transform discovery, and package loading are all outside timing. Gradient
# performance belongs to the top-level AD benchmark and docs page.

import Pkg

const _MNIST_LOGISTIC_COMPARISON_INNER = "RK_MNIST_LOGISTIC_COMPARISON_INNER"

_mnist_logistic_body() = any(==("--dataset=wren-pca40"), ARGS) ?
    "mnist_logistic_wren_pca40_comparison_body.jl" :
    "mnist_logistic_comparison_body.jl"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-mnist-logistic-comparison-") do environment
        Pkg.activate(environment)
        Pkg.add([
            Pkg.PackageSpec(name = "BenchmarkTools", version = v"1.6.3"),
            Pkg.PackageSpec(name = "Turing", version = v"0.47.1"),
            Pkg.PackageSpec(name = "DynamicPPL", version = v"0.42.6"),
            Pkg.PackageSpec(name = "Distributions", version = v"0.25.131"),
            Pkg.PackageSpec(name = "NNlib", version = v"0.9.45"),
            Pkg.PackageSpec(name = "MLDatasets", version = v"0.7.21"),
        ])
        # The non-allocating RK column needs the optional MutatingFunctions
        # extension, at the same reviewed pin the integration suite uses
        # (test/run_nonallocating_integration.jl).
        Pkg.add(Pkg.PackageSpec(
            url = "https://github.com/nsiccha/MutatingFunctions.jl",
            rev = "b353559ef3e391ae2e2d98256b6967903fdfa410"))
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsDistributionKernels")),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsPPLExamples")),
        ])
        body = joinpath(@__DIR__, _mnist_logistic_body())
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $body $ARGS`,
            _MNIST_LOGISTIC_COMPARISON_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            "DATADEPS_ALWAYS_ACCEPT" => "true",
        )
        run(command)
    end
end

get(ENV, _MNIST_LOGISTIC_COMPARISON_INNER, "") == "1" ?
    include(joinpath(@__DIR__, _mnist_logistic_body())) :
    _run_pinned_comparison()
