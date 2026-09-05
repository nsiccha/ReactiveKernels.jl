#!/usr/bin/env julia

# Reproducible, opt-in comparison of ReactiveKernels vs Turing.jl EVALUATION
# throughput — primal (log density), gradient (w.r.t. the position, reverse-mode
# Enzyme via DifferentiationInterface), and generated quantities — each WITH and
# WITHOUT Reactant compilation.
#
# ReactiveKernels lowers a log density to a straight-line kernel that Reactant
# can @compile for the primal and the generated quantities, and that Enzyme can
# differentiate under Reactant for the gradient. Turing's DynamicPPL evaluations
# are measured natively; DynamicPPL re-evaluates the model rather than being a
# straight-line array program, so it is not Reactant-traceable and those cells
# are reported unsupported instead of faked.
#
# Comparator packages live in a fresh temporary environment and never enter
# ReactiveKernels' root deps. BenchmarkTools is deliberately NOT used — it pins
# JSON below the version Reactant requires — so timing is a hand-rolled
# minimum-of-batched-runs.
#
# Run from any directory with:
#
#   julia --startup-file=no benchmark/eval_throughput_comparison.jl
#
# Write the static receipt with --output=benchmark/receipts/eval-throughput-v1.toml
# Quick smoke: prefix RK_EVAL_SIZES=16,64, RK_EVAL_ROUNDS=20, and/or
# RK_EVAL_REPLICAS=8. Published receipts use 256 independent replicas per call.

import Pkg

const _COMPARISON_INNER = "RK_EVAL_COMPARISON_INNER"
const _REACTANT_VERSION = v"0.2.284"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = readchomp(`git -C $root rev-parse HEAD`)
    mktempdir(prefix = "reactivekernels-eval-comparison-") do environment
        _with_serial_pkg_precompile() do
            Pkg.activate(environment)
            Pkg.add(Pkg.PackageSpec(name = "Reactant", version = _REACTANT_VERSION))
            Pkg.pin(Pkg.PackageSpec(name = "Reactant"))
            Pkg.develop(path = root)
            Pkg.add([
                Pkg.PackageSpec(name = "Turing", version = v"0.47.1"),
                Pkg.PackageSpec(name = "DynamicPPL", version = v"0.42.6"),
                Pkg.PackageSpec(name = "Distributions", version = v"0.25.131"),
                Pkg.PackageSpec(name = "DifferentiationInterface", version = v"0.7.21"),
                Pkg.PackageSpec(name = "Enzyme", version = v"0.13.199"),
            ])
            Pkg.precompile()
        end
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(@__FILE__) $(ARGS...)`,
            _COMPARISON_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
        )
        run(command)
    end
end

if get(ENV, _COMPARISON_INNER, "") == "1"
    include(joinpath(@__DIR__, "eval_throughput_comparison_body.jl"))
else
    _run_pinned_comparison()
end
