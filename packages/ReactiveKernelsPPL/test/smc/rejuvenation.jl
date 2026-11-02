module SMCRejuvenationTests
using ReactiveKernelsPPL, Random, Statistics, Test, Serialization
include("prepared_fixture.jl")
using .PreparedSMCFixture
const F = PreparedSMCFixture
const E = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)

# Construct an already-collapsed checkpoint-era state, and a deterministic
# pre-collapse cloud. These exercise the public engine rather than just its ESS.
function cloud(state, positions; temperature=state.temperature, lineage=state.lineage,
        options=state.options)
    E.SMCState(state.source, state.target, positions, copy(state.weights), temperature,
        state.log_normalizer, copy(state.rng), options, copy(state.ancestors),
        copy(lineage), copy(state.history), state.update, state.stage)
end
function caught(f)
    try
        f()
        nothing
    catch e
        e
    end
end

@testset "Full spatial collapse is transactional, including final temperature" begin
    t = F.target(F.GaussianFactory(location=[1.0], scale=[0.1]))
    start = initialize_smc(t; rng=Xoshiro(30), particles=2,
        options=SMCOptions(ess_fraction=0.01, stepsize=1e-200, moves=1, nthreads=1))
    start = cloud(start, reshape([1.0, 2.0], 1, 2))
    before = deepcopy(F.snapshot(start))
    # The final stage resamples the first point; the deliberately immobile
    # kernel leaves two copies. It must never be returned as a completed fit.
    err = caught(() -> smc_step(start))
    @test err isa SMCDegeneracyError
    @test err.state === start
    @test err.phase == :stage
    @test err.diagnostics.temperature == 1
    @test err.diagnostics.mean_squared_movement == 0
    @test err.diagnostics.kind == :temper
    @test F.snapshot(start) == before
    @test length(sprint(showerror, err)) < 512
    @test length(sprint(showerror, err, backtrace())) < 512

    for beta in (0.054010084807354336, 1.0)
        collapsed = cloud(start, ones(1, 2); temperature=beta)
        for advance in (smc_step, smc, s -> smc(s; stages=0))
            err = caught(() -> advance(collapsed))
            @test err isa SMCDegeneracyError
            @test err.phase == :input
            @test err.state === collapsed
            @test err.diagnostics === nothing
        end
        @test_throws SMCDegeneracyError update_smc(collapsed, t; stages=0)
        @test F.snapshot(resume_smc(t, smc_checkpoint(collapsed))) == F.snapshot(collapsed)
    end
end

@testset "Independent fixed-temperature rejuvenation and exact continuation" begin
    t = F.target(F.GaussianFactory(location=[1.0], scale=[0.1]))
    tuning = tune_smc(t; rng=Xoshiro(81), init=[0.0],
        temperatures=(0.0, 0.25, 0.5, 0.75, 1.0), pilot_steps=64)
    initial = initialize_smc(t; rng=Xoshiro(10), particles=1500,
        options=SMCOptions(nthreads=1, history_limit=2))
    input = cloud(initial, ones(1, 1500); temperature=0.5, lineage=ones(Int, 1500))
    before = deepcopy(F.snapshot(input))
    pilot_before = deepcopy(tuning)
    default_before = copy(Random.default_rng())
    recovered = rejuvenate_smc(input; tuning, moves=32)
    @test F.snapshot(input) == before
    @test E._same(tuning.kernels, pilot_before.kernels)
    @test rand(copy(Random.default_rng()), UInt64, 8) == rand(default_before, UInt64, 8)
    @test recovered.temperature == input.temperature
    @test recovered.log_normalizer == input.log_normalizer
    @test recovered.weights == input.weights
    @test recovered.lineage == input.lineage
    @test recovered.ancestors == collect(1:1500)
    @test recovered.stage == input.stage
    @test recovered.positions !== input.positions
    @test recovered.weights !== input.weights
    @test recovered.lineage !== input.lineage
    @test recovered.options.tuning !== tuning
    @test recovered.history[end].kind == :rejuvenate
    @test recovered.history[end].kernel.moves == 32
    @test recovered.history[end].moves == 32*1500
    @test recovered.history[end].log_normalizer_increment == 0
    @test mean(recovered.positions) ≈ 100/101 atol=0.015
    @test var(recovered.positions) ≈ 2/101 atol=0.003
    @test F.snapshot(rejuvenate_smc(input; tuning, moves=32)) == F.snapshot(recovered)
    threaded_input = cloud(input, input.positions;
        options=SMCOptions(nthreads=2, history_limit=2))
    threaded = rejuvenate_smc(threaded_input; tuning, moves=32)
    @test threaded.positions == recovered.positions
    @test threaded.rng == recovered.rng
    @test threaded.history[end].nthreads == min(2, Threads.nthreads())
    resumed = resume_smc(t, smc_checkpoint(recovered))
    @test F.snapshot(resumed) == F.snapshot(recovered)
    final = smc(recovered)
    @test F.snapshot(smc(resumed)) == F.snapshot(final)
    @test final.temperature == 1
    @test mean(final.positions) ≈ 1.0 atol=0.015
    @test var(final.positions) ≈ 0.01 atol=0.002
    @test length(final.history) <= 2
    @test final.history[end].kind == :temper
    @test final.history[end].kernel.moves == tuning.kernels[end].moves
    @test_throws ArgumentError rejuvenate_smc(input; tuning, moves=0)
    other = F.target(F.GaussianFactory(location=[1.0], scale=[0.1]); binding_id="other")
    wrong = tune_smc(other; rng=Xoshiro(81), init=[0.0], pilot_steps=2)
    @test_throws ArgumentError rejuvenate_smc(input; tuning=wrong, moves=1)

    # A recovery attempt has a fixed budget; an immobile result throws and
    # does not consume its input RNG or replace the successful schedule.
    bad = deepcopy(tuning)
    for i in eachindex(bad.kernels)
        bad.kernels[i] = merge(bad.kernels[i], (; stepsize=1e-200))
    end
    err = caught(() -> rejuvenate_smc(input; tuning=bad, moves=4))
    @test err isa SMCDegeneracyError
    @test err.state === input
    @test err.phase == :rejuvenation
    @test err.diagnostics.moves == 4*1500
    @test F.snapshot(input) == before

    # Fresh process restores the recovery schedule/history and resumes exactly.
    mktempdir() do dir
        cp = joinpath(dir, "recovered.jls")
        serialize(cp, (; checkpoint=smc_checkpoint(recovered), expected=F.snapshot(final)))
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) $(joinpath(@__DIR__, "rejuvenation_process.jl")) $cp`
        @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
    end
end

@testset "Equal ratios, unchanged moves and partial constancy remain valid" begin
    matched = F.target(F.GaussianFactory(location=[0.0], scale=[1.0], logmass=0.0))
    start = initialize_smc(matched; rng=Xoshiro(19), particles=128,
        options=SMCOptions(stepsize=1e-200, nthreads=1))
    result = smc(start)
    @test result.temperature == 1
    @test result.history[end].ess_before_resampling ≈ 128
    @test result.history[end].mean_squared_movement == 0
    @test !E._collapsed(result.positions)
    t = F.target()
    start = initialize_smc(t; rng=Xoshiro(19), particles=128)
    positions = copy(start.positions)
    positions[1,:] .= 1
    partial = cloud(start, positions; lineage=ones(Int, 128))
    @test smc(partial; stages=0) === partial
    # Variance underflow must not turn tiny but distinct values into collapse.
    @test !E._collapsed(reshape([0.0, nextfloat(0.0)], 1, 2))

    empty_target = F.target(F.GaussianFactory(location=Float64[], scale=Float64[]))
    empty_start = initialize_smc(empty_target; rng=Xoshiro(19), particles=8)
    @test smc(empty_start).temperature == 1
    # Tuning a zero-dimensional target is deliberately unsupported; a valid
    # data-only empty schedule exercises its no-work rejuvenation semantics.
    kernel = (; stepsize=0.1, metric=Float64[], moves=1, max_tree_depth=2)
    tuning = SMCTuningSchedule(E._source_id(empty_start.source), empty_target.signature,
        smc_reference(empty_start), [0.0,1.0], [kernel,kernel], (;))
    empty_recovered = rejuvenate_smc(empty_start; tuning, moves=3)
    @test empty_recovered.positions == empty_start.positions
    @test rand(copy(empty_recovered.rng), UInt64, 8) == rand(copy(empty_start.rng), UInt64, 8)
    @test empty_recovered.history[end].moves == 0
    @test empty_recovered.history[end].mean_acceptance === nothing
end
end
