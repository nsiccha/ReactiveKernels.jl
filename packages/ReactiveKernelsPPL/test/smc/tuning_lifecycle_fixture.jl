module SMCTuningLifecycleFixture
include("prepared_fixture.jl")
using .PreparedSMCFixture, ReactiveKernelsPPL, Random, Serialization
const F = PreparedSMCFixture
const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)

target(; kwargs...) = F.target(; kwargs...)
function tune(t=target(); kwargs...)
    tune_smc(t; rng=Xoshiro(1731), init=[0.1, -0.2], pilot=:adaptive,
        reference=:pilot, temperatures=(0.0, 0.5, 1.0), pilot_steps=4,
        recording_target=16, pilot_evaluations=64, max_reflections=1,
        options=SMCOptions(moves=1, max_tree_depth=4, history_limit=2), kwargs...)
end
function signature(s)
    p = s.pilot
    (; source=s.source, target=s.target, reference=s.reference,
        temperatures=s.temperatures, kernels=s.kernels,
        pilot=merge(p, (; initial_rng=rand(copy(p.initial_rng), UInt64, 16),
            final_rng=rand(copy(p.final_rng), UInt64, 16))))
end
function bytes(value)
    io = IOBuffer()
    serialize(io, value)
    take!(io)
end
end
