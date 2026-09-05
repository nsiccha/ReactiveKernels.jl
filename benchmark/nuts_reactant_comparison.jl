#!/usr/bin/env julia

# Reproducible, opt-in benchmark of the accepted full-depth adaptive NUTS
# transition through Reactant versus the source-faithful native compiler path.
# Both arms execute the same authored fixture, target, metric, independently
# matched start state, maximum depth, and pre-generated random bundles. Compilation, host/device
# transfers, RNG generation, and result readback are outside steady-state timing.
#
# Run from any directory with:
#
#   julia --startup-file=no benchmark/nuts_reactant_comparison.jl
#
# A publication receipt must be produced from a tracked-clean detached worktree:
#
#   julia --startup-file=no benchmark/nuts_reactant_comparison.jl \
#     --output=benchmark/receipts/nuts-reactant-v1.toml
#
# For exploratory work on an attached/dirty branch, pass --allow-dirty and write
# outside the repository. Environment overrides RK_NUTS_REACTANT_ROUNDS and
# RK_NUTS_REACTANT_TRANSITIONS shorten or lengthen the measured corpus.

import Pkg

const _COMPARISON_INNER = "RK_NUTS_REACTANT_COMPARISON_INNER"
const _REACTANT_VERSION = v"0.2.284"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    allow_dirty = "--allow-dirty" in ARGS
    candidate_sha = allow_dirty ? readchomp(`git -C $root rev-parse HEAD`) :
        _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-nuts-reactant-comparison-") do environment
        Pkg.activate(environment)
        Pkg.add(Pkg.PackageSpec(name = "Reactant", version = _REACTANT_VERSION))
        Pkg.pin(Pkg.PackageSpec(name = "Reactant"))
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(root, "packages", "ReactiveKernelsNUTSExamples")),
        ])
        Pkg.instantiate()
        Pkg.precompile()
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(@__FILE__) $(ARGS...)`,
            _COMPARISON_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => candidate_sha,
        )
        run(command)
    end
end

if get(ENV, _COMPARISON_INNER, "") == "1"
    include(joinpath(@__DIR__, "nuts_reactant_comparison_body.jl"))
    NUTSReactantComparison.run_benchmark()
else
    _run_pinned_comparison()
end
