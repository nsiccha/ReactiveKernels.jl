#!/usr/bin/env julia

# Reproducible, opt-in comparison of ReactiveKernels' graph-backed NUTS with
# established AdvancedHMC and DynamicHMC samplers. Comparator packages live in
# a fresh temporary environment and never enter ReactiveKernels' root deps.
#
# Run from any directory with:
#
#   julia --startup-file=no benchmark/nuts_comparison.jl
#
# The default workload is deliberately the publication receipt: four chains,
# 900 warmup transitions, 1,000 retained draws, and maximum tree depth 8.

import Pkg

const _COMPARISON_INNER = "RK_NUTS_COMPARISON_INNER"
const _MUTATING_FUNCTIONS_REVISION =
    "b353559ef3e391ae2e2d98256b6967903fdfa410"

function _checked_candidate(root)
    sha = readchomp(`git -C $root rev-parse HEAD`)
    dirty = readchomp(
        `git -C $root status --porcelain --untracked-files=no`,
    )
    isempty(dirty) || error(
        "comparison requires a tracked-clean ReactiveKernels candidate; " *
        "commit or revert these changes first:\n$dirty",
    )
    sha
end

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    candidate_sha = _checked_candidate(root)
    mktempdir(prefix = "reactivekernels-nuts-comparison-") do environment
        Pkg.activate(environment)

        # ReactiveKernels carries an unregistered weak dependency. Developing
        # it first preserves that weakdep/extension relationship on Julia 1.10
        # while keeping it outside the package's ordinary hard dependencies.
        Pkg.add(Pkg.PackageSpec(
            url = "https://github.com/nsiccha/MutatingFunctions.jl",
            rev = _MUTATING_FUNCTIONS_REVISION,
        ))
        Pkg.develop(path = root)
        Pkg.add([
            Pkg.PackageSpec(name = "AdvancedHMC", version = v"0.8.6"),
            Pkg.PackageSpec(name = "DynamicHMC", version = v"3.6.1"),
            Pkg.PackageSpec(name = "LogDensityProblems", version = v"2.2.0"),
            Pkg.PackageSpec(name = "MCMCDiagnosticTools", version = v"0.3.19"),
            # Every sampler's runtime gradient is DI+Enzyme reverse mode, pinned
            # to the exact versions the DI+Enzyme test suite validated against.
            Pkg.PackageSpec(name = "DifferentiationInterface", version = v"0.7.21"),
            Pkg.PackageSpec(name = "Enzyme", version = v"0.13.199"),
        ])
        Pkg.precompile()

        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(@__FILE__)`,
            _COMPARISON_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => candidate_sha,
        )
        run(command)
    end
end

if get(ENV, _COMPARISON_INNER, "") == "1"
    include(joinpath(@__DIR__, "nuts_comparison_body.jl"))
else
    _run_pinned_comparison()
end
