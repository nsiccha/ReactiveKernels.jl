include("filter_learning_fixture.jl")
using .FilterLearningFixture
using Test, Random, ReactiveKernelsPPL, Serialization
const F=FilterLearningFixture
mode,path=ARGS
target=F.target()
if mode=="write"
    schedule=tune_filter_smc(target;rng=Xoshiro(401),init=[0.0],
        temperatures=(0.0,1.0),pilot_steps=12,
        options=SMCOptions(max_tree_depth=3))
    s=initialize_filter_smc(target;rng=Xoshiro(293),members=24,particles=24,
        window_length=3,ess_fraction=0.95,options=SMCOptions(tuning=schedule))
    s=filter_stream(s,F.records(F.Y[1:2]))
    recovery=tune_filter_smc(s;rng=Xoshiro(402),init=[0.0],
        temperatures=(0.0,1.0),pilot_steps=12,
        options=SMCOptions(max_tree_depth=3))
    s=rejuvenate_smc(s;tuning=recovery,moves=4)
    checkpoint=filter_checkpoint(s)
    expected=F.signature(filter_stream(s,F.records(F.Y[3:end])))
    serialize(path,(;checkpoint,expected))
elseif mode=="read"
    payload=deserialize(path)
    actual=filter_stream(resume_filter(target,payload.checkpoint),F.records(F.Y[3:end]))
    @test F.equal_signatures(F.signature(actual),payload.expected)
else
    error("expected write or read")
end
