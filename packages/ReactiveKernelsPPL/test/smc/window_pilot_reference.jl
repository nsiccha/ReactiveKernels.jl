include("window_fixture.jl")
using .SMCWindowFixture
using ReactiveKernelsPPL, Random, Test
const W = SMCWindowFixture
@testset "Learned pilot proposals for a carried window" begin
    target = W.window()
    prior = deepcopy(target.prior.reference)
    options = SMCOptions(moves=1,max_tree_depth=4)
    schedule = tune_smc(target;rng=Xoshiro(552),init=zeros(2),pilot=:adaptive,
        reference=:pilot,reference_inflation=2,temperatures=(0.0,0.1,1.0),
        pilot_steps=32,pilot_evaluations=100,recording_target=32,max_reflections=2,
        options)
    @test schedule.pilot.reference_work.reference_inflation == 2
    @test target.prior.reference.location == prior.location
    @test target.prior.reference.scale == prior.scale
    state = initialize_smc(target;rng=Xoshiro(553),particles=128,
        options=SMCOptions(tuning=schedule))
    cp = smc_checkpoint(smc_step(state))
    fit = smc(resume_smc(target,cp))
    @test W.signature(fit) == W.signature(smc(resume_smc(target,cp)))
    carried = carry_smc(fit;basis=:fit,calibration_id="pilot-window")
    @test carried.fit_diagnostics.reflections == 1
    @test length(last(fit.history).reference_sd) == 2
    @test length(last(fit.history).coordinate_sd) == 2
end
