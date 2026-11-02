include("selective_fixture.jl")
using .SMCSelectiveFixture
using ReactiveKernelsPPL, Random, Serialization, Test
const F = SMCSelectiveFixture

function replay_target(growing; changed=false)
    if growing
        source = quote
            m ~ Normal(0,4)
            z[1:nb] .~ Normal.(m,2)
            y .~ Normal.(z[block],1)
        end
        old = F.calibration(F.build(2,Int[],Float64[];source),
            SMCReference([1.0,-0.3,0.4],[0.5,0.7,0.9]))
        F.window(F.build(3,[3],[changed ? 4.0 : 3.0];source),old)
    else
        scale=(;diagonal=[sqrt(2.0),sqrt(0.02),0.8],
            indices=[[1,2]],reflections=[[cos(pi/8),sin(pi/8)]])
        old=F.calibration(F.build(3,[2],[1.4]),
            SMCReference([0.2,-0.1,0.7],scale))
        F.window(F.build(3,[2],[changed ? 2.0 : 1.4]),old)
    end
end

mode,path=ARGS[1:2]
cases = length(ARGS)==3 && ARGS[3]=="old-only" ? (false,) : (false,true)
if mode == "write"
    payload=map(cases) do growing
        target=replay_target(growing)
        partial=smc(target;rng=Xoshiro(71),particles=64,
            options=SMCOptions(stepsize=0.35,moves=2,max_tree_depth=4),stages=1)
        completed=smc(partial)
        (;checkpoint=smc_checkpoint(partial),expected=F.signature(completed),
            recovered=smc_recover(completed,completed.positions[:,1];rng=Xoshiro(29)),
            carried=carry_smc(completed;calibration_id="replay").reference.location)
    end
    serialize(path,payload)
elseif mode == "read"
    payload=deserialize(path)
    for (growing,record) in zip(cases,payload)
        target=replay_target(growing)
        completed=smc(resume_smc(target,record.checkpoint))
        @test F.signature(completed)==record.expected
        @test smc_recover(completed,completed.positions[:,1];rng=Xoshiro(29))==record.recovered
        @test carry_smc(completed;calibration_id="replay").reference.location==record.carried
        @test_throws ArgumentError resume_smc(replay_target(growing;changed=true),record.checkpoint)
    end
else
    error("expected write or read")
end
