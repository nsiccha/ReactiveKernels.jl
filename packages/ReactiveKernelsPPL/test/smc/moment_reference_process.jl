include("window_fixture.jl")
using .SMCWindowFixture
using ReactiveKernelsPPL, Random, Serialization, Test
const W=SMCWindowFixture
options=SMCOptions(metric=W.SCALE,stepsize=0.3,moves=1,max_tree_depth=4)
mode,path=ARGS
if mode=="write"
    fit=smc(W.window();rng=Xoshiro(42),particles=128,options)
    carried=carry_smc(fit;calibration_id="moments",basis=W.SCALE)
    target=W.window([1.1,0.8];previous=carried,binding_id="next")
    state=initialize_smc(target;rng=Xoshiro(43),particles=128,options)
    cp=smc_checkpoint(smc_step(state))
    expected=W.signature(smc(resume_smc(target,cp)))
    serialize(path,(;carried,cp,expected))
elseif mode=="read"
    data=deserialize(path)
    target=W.window([1.1,0.8];previous=data.carried,binding_id="next")
    @test W.signature(smc(resume_smc(target,data.cp)))==data.expected
else
    error("mode")
end
