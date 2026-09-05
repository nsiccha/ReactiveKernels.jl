#!/usr/bin/env julia

# Reproducible native-RK versus Reactant comparison for the exact MNIST
# multinomial-logistic model and primal HAVE/WANT matrix published by
# mnist_logistic_comparison.jl. The authored model source is imported from
# ReactiveKernelsPPLExamples; it is never copied into this benchmark.

import Pkg

const _MNIST_REACTANT_INNER = "RK_MNIST_REACTANT_INNER"
const _MNIST_REACTANT_VERSION = v"0.2.284"

_mnist_reactant_body() = any(
    ==("--dataset=wren-pca40"), ARGS) ?
    "mnist_reactant_wren_pca40_comparison_body.jl" :
    "mnist_reactant_comparison_body.jl"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    body = joinpath(@__DIR__, _mnist_reactant_body())
    mktempdir(prefix = "reactivekernels-mnist-reactant-") do environment
        setup_seconds = precompile_seconds = 0.0
        withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
            setup_seconds = @elapsed begin
                Pkg.activate(environment)
                Pkg.add([
                    Pkg.PackageSpec(
                        name = "Reactant", version = _MNIST_REACTANT_VERSION),
                    Pkg.PackageSpec(name = "BenchmarkTools", version = v"1.6.3"),
                    Pkg.PackageSpec(name = "MLDatasets", version = v"0.7.21"),
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
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $body $ARGS`,
            _MNIST_REACTANT_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "RK_MNIST_REACTANT_ENV_SETUP_SECONDS" => string(setup_seconds),
            "RK_MNIST_REACTANT_PRECOMPILE_SECONDS" => string(precompile_seconds),
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            "DATADEPS_ALWAYS_ACCEPT" => "true",
        )
        run(command)
    end
end

get(ENV, _MNIST_REACTANT_INNER, "") == "1" ?
    include(joinpath(@__DIR__, _mnist_reactant_body())) :
    _run_pinned_comparison()
