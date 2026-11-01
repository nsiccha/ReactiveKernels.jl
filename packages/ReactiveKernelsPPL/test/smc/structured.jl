module SMCStructuredTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random
using LinearAlgebra, Statistics, Test, Serialization
const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)
const BACKEND = AutoEnzyme(mode=Enzyme.Reverse)

function target()
    source = quote
        a ~ Normal(0, 1)
        b ~ Normal(a, 0.2)
        y .~ Normal.(b, 1)
    end
    data = (; y=Float64[])
    plan = bind_data(lower_rkppl(source, data; conditioned=(:y,)), data)
    built = build_kernel(plan)
    prepare_smc(built, plan, zeros(2); backend=BACKEND,
        model_id="structured-normal", binding_id="empty")
end
function scale_data(d=2)
    (; diagonal=[sqrt(2.0), sqrt(0.02), ones(d-2)...],
        indices=[[1,2]], reflections=[[cos(pi/8), sin(pi/8)]])
end

@testset "Portable correlated reference and invariant fixed mutation" begin
    t = target()
    data = scale_data()
    reference = SMCReference([0.1,-0.1], data)
    data.diagonal[1] = 7.0
    @test reference.scale.diagonal[1] == sqrt(2.0)
    scale = WarmupHMC.scale_from_data(reference.scale)
    # A two-dimensional dense oracle is confined to this test.
    L = hcat(scale*[1.0,0.0], scale*[0.0,1.0])
    covariance = L*L'
    @test covariance ≈ [1.01 0.99; 0.99 1.01] atol=1e-12
    prepared = EXT._reference(t, reference)
    u, gradient = [0.5,0.3], zeros(2)
    delta = u-reference.location
    expected = -log(2pi)-logdet(covariance)/2-dot(delta,covariance\delta)/2
    @test EXT._density(prepared,u) ≈ expected atol=1e-12
    @test EXT._gradient(prepared,gradient,u) ≈ expected atol=1e-12
    @test gradient ≈ -(covariance\delta) atol=1e-11
    initial = initialize_smc(t; rng=Xoshiro(43), particles=8000, reference,
        options=SMCOptions(metric=reference.scale, stepsize=0.5, moves=2, max_tree_depth=5))
    @test vec(mean(initial.positions;dims=2)) ≈ reference.location atol=0.04
    @test cov(initial.positions;dims=2) ≈ covariance atol=0.06
    start = initialize_smc(t; rng=Xoshiro(541), particles=1000, reference,
        options=SMCOptions(metric=reference.scale, stepsize=0.5, moves=2, max_tree_depth=5))
    cp = smc_checkpoint(start)
    io=IOBuffer(); serialize(io,cp); seekstart(io)
    final = smc(start)
    resumed = smc(resume_smc(t,deserialize(io)))
    @test final.positions == resumed.positions
    @test final.history == resumed.history
    @test final.log_normalizer == resumed.log_normalizer
    @test rand(copy(final.rng), UInt64, 8) == rand(copy(resumed.rng), UInt64, 8)
    @test abs(final.log_normalizer) < 0.08
    @test maximum(abs, vec(mean(final.positions;dims=2))) < 0.12
    @test cov(final.positions;dims=2) ≈ [1.0 1.0;1.0 1.04] atol=0.16
    @test all(s -> 0 <= s.mean_acceptance <= 1, final.history)
    @test all(s -> s.kernel.metric == reference.scale, final.history)
    broken = deepcopy(cp)
    broken.options.metric.diagonal[1] = NaN
    @test_throws ArgumentError resume_smc(t,broken)
    @test start.positions == cp.positions
    @test rand(copy(start.rng), UInt64, 8) == rand(copy(cp.rng), UInt64, 8)
    small, large = SMCReference(zeros(128),scale_data(128)), SMCReference(zeros(256),scale_data(256))
    @test Base.summarysize(large) < 2.1Base.summarysize(small)
end

@testset "Independent adaptive scale and learned normalized reference" begin
    t = target()
    rng, init = Xoshiro(987), [0.1,-0.1]
    rng0, init0 = copy(rng), copy(init)
    schedule = tune_smc(t; rng, init, pilot=:adaptive, reference=:pilot,
        temperatures=(0.0,1e-5,1.0), pilot_steps=64, pilot_evaluations=100,
        recording_target=64, max_reflections=4,
        options=SMCOptions(moves=2,max_tree_depth=6))
    @test init == init0
    @test rand(copy(rng),UInt64,8) == rand(rng0,UInt64,8)
    @test schedule.pilot.method == :adaptive
    @test schedule.pilot.reference_work.retained_draws >= 64
    @test all(k -> k.metric isa NamedTuple, schedule.kernels)
    @test all(k -> length(k.metric.reflections) <= 4, schedule.kernels)
    @test smc_reference(schedule).scale isa NamedTuple
    start = initialize_smc(t; rng=Xoshiro(438), particles=1000,
        options=SMCOptions(tuning=schedule))
    cp = smc_checkpoint(start)
    final = smc(start)
    restored = smc(resume_smc(t,cp))
    @test final.positions == restored.positions
    @test final.history == restored.history
    @test rand(copy(final.rng),UInt64,8) == rand(copy(restored.rng),UInt64,8)
    @test maximum(abs, vec(mean(final.positions;dims=2))) < 0.15
    @test cov(final.positions;dims=2) ≈ [1.0 1.0;1.0 1.04] atol=0.2
    @test abs(final.log_normalizer) < 0.12
    @test_throws ArgumentError tune_smc(t; rng,init,reference=:pilot)
    @test_throws ArgumentError tune_smc(t; rng,init,pilot=:unknown)
    @info "Adaptive schedule" families=[w.family for w in schedule.pilot.work] reflections=[length(k.metric.reflections) for k in schedule.kernels]
end
end
