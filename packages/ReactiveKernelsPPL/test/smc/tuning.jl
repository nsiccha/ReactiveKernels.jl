module SMCTuningTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random
using Statistics, Test, Serialization

const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)
const BACKEND = AutoEnzyme(mode=Enzyme.Reverse)

function target(y; id=string(y))
    source = quote
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1)
    end
    data = (; y)
    plan = bind_data(lower_rkppl(source, data; conditioned=(:y,)), data)
    built = build_kernel(plan)
    return prepare_smc(built, plan, zeros(built.layout.total);
        backend=BACKEND, model_id="tuned-gaussian", binding_id=id)
end

function equal_continuation(a,b)
    @test a.positions == b.positions
    @test a.weights == b.weights
    @test a.history == b.history
    @test a.lineage == b.lineage
    @test a.log_normalizer == b.log_normalizer
    @test rand(copy(a.rng), UInt64, 16) == rand(copy(b.rng), UInt64, 16)
end

@testset "Independent pilot, fixed stage kernels and exact continuation" begin
    t = target([0.8, 1.2, 0.3, 0.7])
    rng, init = Xoshiro(1928), [0.2]
    rng_before, init_before = copy(rng), copy(init)
    reference = SMCReference([0.3], [1.4])
    schedule = tune_smc(t; rng, init, reference,
        temperatures=[0.0, 0.25, 1.0], pilot_steps=24,
        options=SMCOptions(moves=3, max_tree_depth=5))
    @test init == init_before
    @test rand(copy(rng), UInt64, 16) == rand(rng_before, UInt64, 16)
    @test schedule.temperatures == [0.0, 0.25, 1.0]
    @test all(k -> k.moves == 3 && k.max_tree_depth == 5, schedule.kernels)
    @test all(k -> isfinite(k.stepsize) && k.stepsize > 0, schedule.kernels)
    @test sum(x -> x.transitions, schedule.pilot.work) == 72
    @test smc_reference(schedule).location == reference.location
    @test smc_reference(schedule).scale == reference.scale
    options = SMCOptions(tuning=schedule, history_limit=3)
    @test EXT._stage_kernel(options,1,0.125)[2] == 2
    @test EXT._stage_kernel(options,1,0.1)[2] == 1
    start = initialize_smc(t; rng=Xoshiro(278), particles=1200, options)
    @test smc_reference(start).location == reference.location
    saved_positions, saved_rng = copy(start.positions), copy(start.rng)
    partial = smc_step(start)
    @test start.positions == saved_positions
    @test rand(copy(start.rng), UInt64, 16) == rand(saved_rng, UInt64, 16)
    @test partial.history[end].kernel == schedule.kernels[partial.history[end].tuning_index]
    @test partial.history[end].moves == 3600
    @test partial.history[end].evaluations >= partial.history[end].moves
    @test partial.history[end].mean_squared_movement > 0
    cp = smc_checkpoint(partial)
    io = IOBuffer(); serialize(io,cp); seekstart(io)
    resumed = resume_smc(t, deserialize(io))
    final = smc(partial)
    equal_continuation(final, smc(resumed))
    @test mean(final.positions) ≈ 3/5 atol=0.06
    @test var(final.positions) ≈ 1/5 atol=0.04
    @test length(final.history) <= 3
    @test_throws ArgumentError initialize_smc(t; rng=Xoshiro(1), options,
        reference=SMCReference([0.3], [1.5]))
    @test_throws ArgumentError initialize_smc(target([0.9]); rng=Xoshiro(1), options)
    broken = deepcopy(options)
    broken.tuning.kernels[1].metric[1] = NaN
    @test_throws ArgumentError initialize_smc(t; rng=Xoshiro(1), options=broken)
    @test_throws ArgumentError resume_smc(t, merge(cp,(;options=broken)))
    @test_throws ArgumentError tune_smc(t; rng, init, temperatures=[0.0,0.0,1.0])
    @test_throws ArgumentError tune_smc(t; rng, init, options)
    @test_throws ArgumentError tune_smc(t; rng, init, pilot_steps=0)
    @test_throws DimensionMismatch tune_smc(t; rng, init=[0.0,0.0])

    next = target([0.8, 1.2, 0.3, 0.7, 1.1])
    @test_throws ArgumentError update_smc(final, next; stages=0)
    update_schedule = tune_smc(next; previous=t, rng=Xoshiro(9021), init=[0.0],
        temperatures=(0.0,1.0), pilot_steps=20)
    @test smc_reference(update_schedule) === nothing
    @test_throws ArgumentError tune_smc(next; previous=final, rng, init)
    @test_throws ArgumentError tune_smc(next; previous=t, reference, rng, init)
    updating = update_smc(final, next; options=SMCOptions(tuning=update_schedule), stages=0)
    @test smc_reference(updating) === nothing
    @test_throws ArgumentError initialize_smc(next; rng=Xoshiro(3),
        options=SMCOptions(tuning=update_schedule))
    update_cp = smc_checkpoint(updating)
    equal_continuation(smc(updating),smc(resume_smc(next,update_cp; previous=t)))
    @test rand(copy(schedule.pilot.final_rng),UInt64,8) ==
        rand(copy(cp.options.tuning.pilot.final_rng),UInt64,8)
end

@testset "Fresh-process tuned continuation" begin
    mktempdir() do directory
        path = joinpath(directory,"tuned.bin")
        fixture = joinpath(@__DIR__,"tuning_process.jl")
        cmd = `$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --project=$(dirname(Base.active_project())) $fixture`
        @test success(pipeline(`$cmd write $path`;stdout=stdout,stderr=stderr))
        @test success(pipeline(`$cmd read $path`;stdout=stdout,stderr=stderr))
    end
end
end
