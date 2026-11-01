using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using WarmupHMC, SequentialMonteCarlo, LogDensityProblems, Random, Test

const EXT = Base.get_extension(ReactiveKernelsPPL,:ReactiveKernelsPPLSMCExt)
function prior_target(d)
    source=quote
        b[1:d] .~ Normal.(0,1)
        y .~ Normal.(sum(b),1)
    end
    data=(;d,y=Float64[])
    plan=bind_data(lower_rkppl(source,data;conditioned=(:y,)),data)
    built=build_kernel(plan)
    prepare_smc(built,plan,zeros(d);backend=AutoEnzyme(mode=Enzyme.Reverse),
        model_id="tuning-dimension-$d",binding_id="empty")
end

@testset "Near-zero temperature selection and linear schedule retention" begin
    target=prior_target(1)
    grid=(0.0,1e-6,5e-6,2e-5,1e-4,5e-4,0.001,0.005,0.02,0.1,0.5,1.0)
    tuning=tune_smc(target;rng=Xoshiro(17),init=[0.2],temperatures=grid,pilot_steps=2)
    options=SMCOptions(tuning=tuning)
    @test EXT._stage_kernel(options,1,5.08e-6)[2] == 3
    @test EXT._stage_kernel(options,1,3.77e-5)[2] == 4
    @test EXT._stage_kernel(options,1,2.02e-4)[2] == 5
    @test EXT._stage_kernel(options,1,0.005165785)[2] == 8
    @test EXT._stage_kernel(options,1,5.08e-6)[1] == tuning.kernels[3]
    default=tune_smc(target;rng=Xoshiro(16),init=[0.2],pilot_steps=1)
    @test any(t -> 0 < t <= 1e-6,default.temperatures)
    function footprint(d)
        t=prior_target(d)
        schedule=tune_smc(t;rng=Xoshiro(13),init=fill(0.1,d),
            temperatures=(0.0,1e-6,1e-4,1.0),pilot_steps=2)
        @test all(k -> size(k.metric) == (d,),schedule.kernels)
        state=initialize_smc(t;rng=Xoshiro(71),particles=8,options=SMCOptions(tuning=schedule))
        return Base.summarysize(schedule),Base.summarysize(smc_checkpoint(state))
    end
    small,large=footprint(128),footprint(256)
    @test large[1]/small[1] < 2.2
    @test large[2]/small[2] < 2.2
    @info "retained tuning/checkpoint bytes" small large
end
