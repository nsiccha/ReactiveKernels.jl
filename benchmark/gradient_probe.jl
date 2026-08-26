#!/usr/bin/env julia

# Diagnostic probe: isolate the RK DI+Enzyme gradient invocation cost at concrete
# barriers to localize the comparator's per-gradient overhead. Reproducible temp
# env with the SAME pinned DI+Enzyme the comparator uses.
#
#   julia --startup-file=no benchmark/gradient_probe.jl
#
# Measures @allocated (named-barrier per call) + min ns + @inferred for identical
# prepared DI work reached five ways:
#   direct   value_and_gradient!(logp, grad, prep, backend, x)
#   callback potential_gradient!(grad, x)                       (the RK wrapper)
#   bundle   _nuts_cache_apply(vg, _grad_bundle, pg!, x)        (in-place bundle)
#   getter   get!(state, fwd_dpot_dpos) after invalidating pos  (reactive RGF getter)
#   group    set!(fwd_pos); get!(fwd_dpot_dpos)                 (full invalidate+recompute)

import Pkg

const _PROBE_INNER = "RK_GRADIENT_PROBE_INNER"
const _MF_REV = "b353559ef3e391ae2e2d98256b6967903fdfa410"

function _run()
    root = normpath(joinpath(@__DIR__, ".."))
    mktempdir(prefix = "rk-gradient-probe-") do env
        Pkg.activate(env)
        Pkg.add(Pkg.PackageSpec(url = "https://github.com/nsiccha/MutatingFunctions.jl", rev = _MF_REV))
        Pkg.develop(path = root)
        Pkg.add([
            Pkg.PackageSpec(name = "DifferentiationInterface", version = v"0.7.21"),
            Pkg.PackageSpec(name = "Enzyme", version = v"0.13.199")])
        Pkg.precompile()
        run(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$env $(@__FILE__)`,
                   _PROBE_INNER => "1"))
    end
end

function _inner()
    @eval begin
        using LinearAlgebra, Random, InteractiveUtils
        using DifferentiationInterface
        import Enzyme
        using ReactiveKernels
    end
    Base.include(@__MODULE__, joinpath(@__DIR__, "_gradient_probe_body.jl"))
end

get(ENV, _PROBE_INNER, "") == "1" ? _inner() : _run()
