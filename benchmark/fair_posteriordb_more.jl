#!/usr/bin/env julia

# Fair all-sides benchmark on posteriordb models, batch 2 (arK AR, GLMM_Poisson hierarchical).
#
# The DynamicPPL posteriordb benchmark reports Turing/Stan ratios against the
# NAIVE posteriordb reference Stan. This runs the same model families with EVERY
# side optimized and adds ReactiveKernels: RK @kernel (one authored graph that
# lowers to native + Reactant; pure-function poisson/binomial densities) vs Stan
# (naive loop / vectorized / fused-GLM where it exists) vs the posteriordb-style
# optimized Turing translation. Native primal + gradient at the fair boundary;
# parity gated against an independent finite-difference oracle.
#
# Self-contained: builds a pinned temporary environment and re-execs the body.
# Optional AS_OUTPUT=<path.toml> writes a machine-readable receipt.

import Pkg

const _INNER = "RK_FAIR_POSTERIORDB_MORE_INNER"
const _BODY = joinpath(@__DIR__, "fair_posteriordb_more_body.jl")

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    mktempdir(prefix = "reactivekernels-fair-posteriordb-more-") do environment
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
        ])
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(root, "packages", "ReactiveKernelsDistributionKernels")),
        ])
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(_BODY) $(ARGS...)`,
            _INNER => "1",
        )
        run(command)
    end
end

get(ENV, _INNER, "") == "1" ? include(_BODY) : _run_pinned_comparison()
