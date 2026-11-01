include("window_fixture.jl")
using .SMCWindowFixture
using ReactiveKernelsPPL, Random, Serialization, Test
const W = SMCWindowFixture
mode,path = ARGS
target = W.window()
if mode == "write"
    schedule = tune_smc(target;rng=Xoshiro(927),init=[0.1,-0.1],
        temperatures=(0.0,0.1,1.0),pilot_steps=12)
    state = smc(target;rng=Xoshiro(71),particles=64,
        options=SMCOptions(tuning=schedule),stages=1)
    serialize(path,(; checkpoint=smc_checkpoint(state),expected=W.signature(smc(state))))
elseif mode == "read"
    payload = deserialize(path)
    @test W.signature(smc(resume_smc(target,payload.checkpoint))) == payload.expected
else
    error("expected write or read")
end
