include("projection_fixture.jl")
using .SMCProjectionFixture
using ReactiveKernelsPPL, Random, Test, Serialization
const P=SMCProjectionFixture
mode,path=ARGS
if mode=="write"
    carried=carry_smc(P.completed();parameters=filter_parameter_names(P.model()),calibration_id="portable")
    target=P.learning(carried)
    state=initialize_filter_smc(target;rng=Xoshiro(33),members=24,particles=16,
        window_length=2,ess_fraction=0.9,
        options=SMCOptions(stepsize=0.15,moves=1,max_tree_depth=3))
    state=filter_step(state,(;y=0.9))
    checkpoint=filter_checkpoint(state)
    expected=P.F.signature(filter_stream(state,P.F.records([0.5,0.6,0.8])))
    serialize(path,(;carried,checkpoint,expected))
elseif mode=="read"
    payload=deserialize(path)
    # Rebuild the unbound filter only. No calibration observations, old
    # sampler target, latent layout or old prepared queries are reconstructed.
    target=P.learning(payload.carried)
    state=resume_filter(target,payload.checkpoint)
    @test P.F.equal_signatures(P.F.signature(filter_stream(state,P.F.records([0.5,0.6,0.8]))),payload.expected)
else
    error("expected write or read")
end
