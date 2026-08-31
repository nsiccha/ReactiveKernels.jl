"""
    ReactiveKernelsNUTSExample

Compatibility launcher for the package-owned external NUTS/HMC exemplar.

Loading this file opts into the sampler runtime; it never adds sampler names to
the `ReactiveKernels` module.
"""
module ReactiveKernelsNUTSExample

using ReactiveKernelsNUTSExamples

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
    @eval const $name = getproperty(ReactiveKernelsNUTSExamples, $(QuoteNode(name)))
    @eval export $name
end

end # module ReactiveKernelsNUTSExample
