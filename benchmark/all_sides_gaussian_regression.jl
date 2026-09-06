#!/usr/bin/env julia

# All-sides-optimized Bayesian Gaussian linear-regression comparison.
#
# Answers "by how much can ReactiveKernels beat Turing AND Stan when EVERY side
# is optimized?" on the diamonds / `sblrc-blr` conjugate archetype. It runs
# three fully optimized implementations of ONE posterior, plus naive/vectorized
# Stan rows that quantify the naive-vs-optimized-Stan gap the DynamicPPL
# posteriordb benchmark leaves on the table:
#
#   * Stan (BridgeStan 2.9): naive per-element loop, fused `normal_id_glm`, and
#     the conjugate O(K^2) sufficient-statistic form (transformed-data).
#   * Turing (@model): the diamonds-style conjugate sufficient-statistic form
#     (XtX/Xty precomputed in make_model), Mooncake gradient.
#   * ReactiveKernels (@kernel): the same conjugate form as one authored graph,
#     with the sufficient statistics as a data-only HAVE prefix; density-only
#     WANT cut (prediction pruned); DI+Enzyme gradient; `bound` data-hoisting.
#
# Fair boundary: setup / compile / first call excluded; the sufficient
# statistics are precomputed ONCE for every conjugate side (as Stan
# transformed-data, Turing make_model, and the RK data prefix), so the timed
# region is the same O(K^2) work for all three. Parity is gated against an
# independent finite-difference oracle (gradient) before any timing.
#
# Self-contained: it builds a pinned temporary environment and re-execs the
# body. Optional `AS_OUTPUT=<path.toml>` writes a machine-readable receipt.

import Pkg

const _AS_INNER = "RK_ALL_SIDES_GAUSSIAN_INNER"
const _AS_BODY = joinpath(@__DIR__, "all_sides_gaussian_regression_body.jl")

function _run_pinned_comparison()
    root = normpath(joinpath(@__DIR__, ".."))
    mktempdir(prefix = "reactivekernels-all-sides-gaussian-") do environment
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
        ])
        Pkg.develop([
            Pkg.PackageSpec(path = root),
            Pkg.PackageSpec(path = joinpath(root, "packages", "ReactiveKernelsDistributionKernels")),
        ])
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$environment $(_AS_BODY) $(ARGS...)`,
            _AS_INNER => "1",
        )
        run(command)
    end
end

get(ENV, _AS_INNER, "") == "1" ? include(_AS_BODY) : _run_pinned_comparison()
