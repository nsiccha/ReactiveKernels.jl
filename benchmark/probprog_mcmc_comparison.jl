#!/usr/bin/env julia

# Reproducible NUTS sampling comparison over the exact RK-authored Eight
# Schools and MNIST (Wren PCA-40) densities: Reactant ProbProg's compiled
# `mcmc_logpdf`, native AdvancedHMC over the RK density + prepared gradient,
# and Turing NUTS on the source-attested Turing twins. The authored model
# sources are imported from ReactiveKernelsPPLExamples; nothing is copied.

import Pkg

const _PROBPROG_MCMC_INNER = "RK_PROBPROG_MCMC_INNER"
const _PROBPROG_MCMC_REACTANT_VERSION = v"0.2.284"

include(joinpath(@__DIR__, "_repro_guard.jl"))

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    sha = _require_clean_detached_candidate(root)
    mktempdir(prefix = "reactivekernels-probprog-mcmc-") do environment
        setup_seconds = precompile_seconds = 0.0
        withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
            setup_seconds = @elapsed begin
                Pkg.activate(environment)
                Pkg.add(Pkg.PackageSpec(
                    name = "Reactant", version = _PROBPROG_MCMC_REACTANT_VERSION))
                Pkg.pin(Pkg.PackageSpec(name = "Reactant"))
                Pkg.add(["AdvancedHMC", "Turing", "DynamicPPL",
                         "MCMCDiagnosticTools", "Enzyme", "ADTypes", "NNlib",
                         "Distributions", "MLDatasets"])
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
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(joinpath(@__DIR__, "probprog_mcmc_comparison_body.jl")) $(ARGS...)`,
            _PROBPROG_MCMC_INNER => "1",
            "REACTIVEKERNELS_CANDIDATE_SHA" => sha,
            "RK_PROBPROG_MCMC_ENV_SETUP_SECONDS" => string(setup_seconds),
            "RK_PROBPROG_MCMC_PRECOMPILE_SECONDS" => string(precompile_seconds),
            "JULIA_PKG_PRECOMPILE_AUTO" => "0",
        )
        run(command)
    end
end

get(ENV, _PROBPROG_MCMC_INNER, "") == "1" ?
    include(joinpath(@__DIR__, "probprog_mcmc_comparison_body.jl")) :
    _run_pinned_comparison()
