#!/usr/bin/env julia

# Staging validator for the faithful posteriordb redo. For each model it embeds
# the REAL posteriordb `.stan` source (ground truth the ReactiveKernels:ppl:posteriordb
# lane translates from) and an independent plain-Julia FD oracle of the same
# model in Stan's unconstrained parameter order, then checks that the oracle's
# gradient matches BridgeStan's on the real model. This validates the parity
# ANCHOR each benchmark side (RK-builtin / optimized Stan / optimized Turing)
# will later be gated against — before the RK graphs land. Realistic-shaped data
# (real values bound per-model when the posteriordb lane sends them); parity is
# data-independent so it validates the translation regardless.
#
# Minimal env (BridgeStan only) so it builds fast. Re-execs the body.

import Pkg
const _INNER = "RK_FAITHFUL_STAGING_INNER"
const _BODY = joinpath(@__DIR__, "faithful_staging_body.jl")
function _run()
    mktempdir(prefix = "reactivekernels-faithful-staging-") do env
        Pkg.activate(env)
        Pkg.add([
            Pkg.PackageSpec(name = "BridgeStan"),
            Pkg.PackageSpec(name = "Turing", version = v"0.47.1"),
            Pkg.PackageSpec(name = "DynamicPPL", version = v"0.42.6"),
            Pkg.PackageSpec(name = "Distributions", version = v"0.25.131"),
            Pkg.PackageSpec(name = "DifferentiationInterface"),
            Pkg.PackageSpec(name = "Mooncake"),
            Pkg.PackageSpec(name = "ADTypes"),
        ])
        run(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$env $(_BODY) $(ARGS...)`, _INNER => "1"))
    end
end
get(ENV, _INNER, "") == "1" ? include(_BODY) : _run()
