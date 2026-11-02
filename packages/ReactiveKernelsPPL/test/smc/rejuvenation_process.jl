using ReactiveKernelsPPL, Serialization, Test
include("prepared_fixture.jl")
using .PreparedSMCFixture
const F = PreparedSMCFixture
t = F.target(F.GaussianFactory(location=[1.0], scale=[0.1]))
saved = deserialize(only(ARGS))
restored = resume_smc(t, saved.checkpoint)
@test restored.history[end].kind == :rejuvenate
@test F.snapshot(smc(restored)) == saved.expected
