#!/usr/bin/env julia

# Reproducible primal comparison of the opt-in partial-evaluation pass on the
# multinomial-logistic (softmax) MNIST classifier. One authored RK graph whose
# data-only work (the reference-logits row, observation/feature scaffolding) is
# reachable from the data ports alone is prepared twice — without and with
# `bound = (; X, y, num_classes)` — and both are compared, at parity, against
# the landed docs model. Full MNIST is loaded via MLDatasets; data loading and
# kernel preparation (including the bind-time prefix execution) are outside
# timing.

import Pkg

const _PARTIAL_EVALUATION_COMPARISON_INNER = "RK_PARTIAL_EVALUATION_COMPARISON_INNER"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-partial-evaluation-comparison-") do environment
        Pkg.activate(environment)
        Pkg.add([
            Pkg.PackageSpec(name = "BenchmarkTools", version = v"1.6.3"),
            Pkg.PackageSpec(name = "MLDatasets", version = v"0.7.21"),
        ])
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsDistributionKernels")),
            Pkg.PackageSpec(path = joinpath(
                root, "packages", "ReactiveKernelsPPLExamples")),
        ])
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(joinpath(@__DIR__, "partial_evaluation_comparison_body.jl")) $(ARGS...)`,
            _PARTIAL_EVALUATION_COMPARISON_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            "DATADEPS_ALWAYS_ACCEPT" => "true",
        )
        run(command)
    end
end

get(ENV, _PARTIAL_EVALUATION_COMPARISON_INNER, "") == "1" ?
    include(joinpath(@__DIR__, "partial_evaluation_comparison_body.jl")) :
    _run_pinned_comparison()
