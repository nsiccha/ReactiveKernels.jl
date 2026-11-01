module SMCPilotReferenceTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random
using LinearAlgebra, Statistics, Test, Serialization

const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)
const BACKEND = AutoEnzyme(mode=Enzyme.Reverse)

function target()
    data = (; y=Float64[])
    source = quote
        a ~ Normal(0, 1)
        b ~ Normal(a, 0.2)
        y .~ Normal.(b, 1)
    end
    plan = bind_data(lower_rkppl(source, data; conditioned=(:y,)), data)
    built = build_kernel(plan)
    prepare_smc(built, plan, zeros(2); backend=BACKEND,
        model_id="pilot-reference-moments", binding_id="empty")
end

@testset "Reference fits retained pilot moments independently of mutation scales" begin
    t = target()
    rng, init = Xoshiro(987), [0.1, -0.1]
    rng_before, init_before = copy(rng), copy(init)
    default_rng_before = copy(Random.default_rng())
    options = SMCOptions(moves=2, max_tree_depth=6)
    kwargs = (; rng, init, pilot=:adaptive, reference=:pilot,
        temperatures=(0.0, 1.0), pilot_steps=64, pilot_evaluations=100,
        recording_target=64, max_reflections=4, options)
    schedule = tune_smc(t; kwargs...)
    uninflated = tune_smc(t; kwargs..., reference_inflation=1)
    @test init == init_before
    @test rand(copy(rng), UInt64, 8) == rand(rng_before, UInt64, 8)
    @test rand(copy(Random.default_rng()), UInt64, 8) == rand(default_rng_before, UInt64, 8)
    @test schedule.pilot.reference_work.reference_inflation == 2
    @test schedule.pilot.reference_work.seed == uninflated.pilot.reference_work.seed
    reference = smc_reference(schedule)
    @test reference.location == smc_reference(uninflated).location
    @test reference.scale.diagonal ≈ 2 .* smc_reference(uninflated).scale.diagonal
    @test reference.scale.reflections == smc_reference(uninflated).scale.reflections

    # Replay ONLY the separate reference pilot through its public upstream
    # API, and compare with a small dense moment oracle in its learned basis.
    initial_reference = EXT._reference(t, SMCReference(zeros(2), ones(2)))
    result = adaptive_warmup_mcmc(Xoshiro(schedule.pilot.reference_work.seed),
        EXT.BridgeTarget(initial_reference, t, 1.0, 2);
        init=(; position=copy(init), squared_scale=ones(2)), n_draws=64,
        n_evaluations=100, recording_target=64, max_reflections=4,
        max_tree_depth=6, nonlinear_adapt=false, monitor_ess=false)
    basis = scale_data(result.scale_options[result.active_transformation])
    operator = scale_from_data(basis)
    L = hcat(operator * [1.0, 0.0], operator * [0.0, 1.0])
    Q = L / Diagonal(basis.diagonal)
    draws = Matrix(result.posterior_position)
    location = vec(mean(draws; dims=2))
    rotated = Q \ (draws .- location)
    expected = Q * Diagonal(2 .* vec(std(rotated; dims=2, corrected=false)))
    reference_operator = scale_from_data(reference.scale)
    actual = hcat(reference_operator * [1.0, 0.0], reference_operator * [0.0, 1.0])
    @test reference.location ≈ location
    @test actual ≈ expected rtol=1e-10
    @test reference.scale.diagonal != basis.diagonal
    @test schedule.pilot.reference_work.draw_sd ≈ vec(std(draws; dims=2, corrected=false))

    prepared = EXT._reference(t, reference)
    point, gradient = [0.5, 0.3], zeros(2)
    covariance = actual * transpose(actual)
    delta = point - reference.location
    expected_density = -log(2pi) - logdet(covariance)/2 - dot(delta, covariance \ delta)/2
    @test EXT._density(prepared, point) ≈ expected_density atol=1e-11
    @test EXT._gradient(prepared, gradient, point) ≈ expected_density atol=1e-11
    @test gradient ≈ -(covariance \ delta) atol=1e-10

    # Independently replay the first knot: proposal inflation must not be
    # applied to the mutation scale or separated from its paired step size.
    knot = adaptive_warmup_mcmc(Xoshiro(first(schedule.pilot.work).seed),
        EXT.BridgeTarget(prepared, t, 0.0, 2);
        init=(; position=copy(init), squared_scale=ones(2)), n_draws=64,
        n_evaluations=100, recording_target=64, max_reflections=4,
        max_tree_depth=6, nonlinear_adapt=false, monitor_ess=false)
    @test first(schedule.kernels).metric == scale_data(knot.scale_options[knot.active_transformation])
    @test first(schedule.kernels).stepsize == knot.stepsize

    start = initialize_smc(t; rng=Xoshiro(438), particles=1000,
        options=SMCOptions(tuning=schedule, history_limit=2))
    cp = smc_checkpoint(start)
    final = smc(start)
    replay = smc(resume_smc(t, cp))
    @test final.positions == replay.positions
    @test final.history == replay.history
    @test last(final.history).coordinate_sd ≈ vec(std(final.positions; dims=2, corrected=false))
    @test last(final.history).reference_sd ≈ sqrt.(diag(actual * transpose(actual)))
    @test last(final.history).reference_sd !== start.source.standard_deviation
    @test length(final.history) <= 2
    @test all(s -> s.kernel == schedule.kernels[s.tuning_index], final.history)
    @test start.positions == cp.positions
    @test cov(final.positions; dims=2) ≈ [1.0 1.0; 1.0 1.04] atol=0.2
    for bad in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError tune_smc(t; kwargs..., reference_inflation=bad)
    end
end

@testset "A matched reference legitimately takes the full temperature step" begin
    # The actual temperature selector should accept constant log ratios.
    @test EXT._temperature(zeros(256), 0.0, SMCOptions()) == 1.0
    @test EXT._ess(zeros(256), 1.0) == 256
end
end
