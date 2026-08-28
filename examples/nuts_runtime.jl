"""
    ReactiveKernelsNUTSExample

Example-owned NUTS/HMC runtime and native compiler-acceptance backend.

Loading this file opts into the external exemplar. The ReactiveKernels package itself
ships only generic authoring, planning, reactive-state, and compiler machinery; it does
not load or export a sampler API.
"""
module ReactiveKernelsNUTSExample

import ReactiveKernels

const _RUNTIME_DIR = joinpath(@__DIR__, "nuts_runtime")
const _RUNTIME_FILES = (
    "kernel_factory.jl",
    "kernel_codegen.jl",
    "kernel_nuts.jl",
    "kernel_nuts_native.jl",
    "hmc.jl",
    "reactive_nuts.jl",
)

# The exemplar extends private compiler seams deliberately, but only when explicitly
# loaded by an example, benchmark, docs build, or test. A second include is harmless.
if !isdefined(ReactiveKernels, :_NutsFrame)
    for file in _RUNTIME_FILES
        Base.include(ReactiveKernels, joinpath(_RUNTIME_DIR, file))
    end
end

const _EXAMPLE_API = (
    :ReactivePhasePoint, :euclidean_phasepoint, :riemannian_phasepoint,
    :reactive_nuts_group, :compiled_nuts_state, :CompiledNUTSState,
    :leapfrog!, :generalized_leapfrog!, :implicit_midpoint!, :multistep,
    :NUTSDiagnostics, :nuts_state, :step!, :refresh_momentum!, :diagnostics,
    :sample!, :find_initial_stepsize!, :warmup!,
    :DualAveragingState, :dual_averaging_state, :fit!,
    :WelfordVariance, :welford_var,
    :TrajectoryStats, :SamplingStats, :trajectory_stats, :sampling_stats, :reset!,
)

for name in _EXAMPLE_API
    @eval const $name = getproperty(ReactiveKernels, $(QuoteNode(name)))
    @eval export $name
end

end # module ReactiveKernelsNUTSExample
