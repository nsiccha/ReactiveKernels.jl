#!/usr/bin/env julia

# Faithful all-sides posteriordb benchmark — RK measured USING RK BUILTINS.
#
# The RK side consumes the rich, idiomatic RK-builtin model graphs authored by
# ReactiveKernels:ppl:posteriordb in packages/ReactiveKernelsPPLExamples
# (distribution kernels + plate + named have/want nodes — the eight-schools
# pattern), NOT hand-rolled flat densities. All three sides — RK-builtin,
# optimized Stan, optimized Turing — run the SAME faithful model (real priors,
# real data). Native primal + gradient at the fair boundary; parity gated
# against an independent finite-difference oracle.
#
# Reference model: eight_schools (ready now). Extended per-model as the
# posteriordb lane lands each rich translation.
#
# Self-contained: builds a pinned temporary environment and re-execs the body.
# Optional AS_OUTPUT=<path.toml> writes a machine-readable receipt.

import Pkg

const _INNER = "RK_FAITHFUL_POSTERIORDB_INNER"
const _BODY = joinpath(@__DIR__, "faithful_posteriordb_body.jl")

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    mktempdir(prefix = "reactivekernels-faithful-posteriordb-") do environment
        Pkg.activate(environment)
        Pkg.add([
            Pkg.PackageSpec(name = "BenchmarkTools", version = v"1.6.3"),
            Pkg.PackageSpec(name = "Turing", version = v"0.47.1"),
            Pkg.PackageSpec(name = "DynamicPPL", version = v"0.42.6"),
            Pkg.PackageSpec(name = "Distributions", version = v"0.25.131"),
            Pkg.PackageSpec(name = "DifferentiationInterface"),
            Pkg.PackageSpec(name = "Enzyme"),
            Pkg.PackageSpec(name = "Mooncake"),
            Pkg.PackageSpec(name = "ADTypes"),
            Pkg.PackageSpec(name = "BridgeStan"),
            Pkg.PackageSpec(name = "SpecialFunctions"),
            Pkg.PackageSpec(name = "LogExpFunctions"),
            Pkg.PackageSpec(name = "JSON"),
        ])
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(root, "packages", "ReactiveKernelsDistributionKernels")),
            Pkg.PackageSpec(path = joinpath(root, "packages", "ReactiveKernelsPPLExamples")),
        ])
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(_BODY) $(ARGS...)`,
            _INNER => "1",
        )
        run(command)
    end
end

get(ENV, _INNER, "") == "1" ? include(_BODY) : _run_pinned_comparison()
