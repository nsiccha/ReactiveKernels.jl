using DifferentiationInterface
import Enzyme
using LinearAlgebra
using Random
using ReactiveKernels
using ReactiveKernelsNUTSExamples
using Statistics
using Test

# Reverse-mode Enzyme through DifferentiationInterface. The density-only PPL
# plan has no active temporary container, so ordinary static activity suffices.
const _HMC_ENZYME_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

_gaussian_potential(position) = sum(abs2, position) / 2
# Analytic (value, gradient) callback — oracle/parity fixtures only.
_gaussian_gradient(position) =
    (_gaussian_potential(position), copy(position))
# Public compiled-group boundary: DI+Enzyme gradient prepared ONCE, filled into
# the passed owned buffer in place. This is the sampler's runtime gradient path.
const _HMC_GAUSSIAN_PREP =
    prepare_gradient(_gaussian_potential, _HMC_ENZYME_BACKEND, zeros(2))
_gaussian_gradient!(gradient, position) = first(value_and_gradient!(
    _gaussian_potential, gradient, _HMC_GAUSSIAN_PREP,
    _HMC_ENZYME_BACKEND, position))
_phase_hamiltonian(point) = point.ham

function _phasepoint_receipt(point)
    _phase_hamiltonian(point)
    leapfrog!(point; stepsize = 0.1)
    _phase_hamiltonian(point)
    cached_allocations = @allocated _phase_hamiltonian(point)
    leapfrog_allocations = @allocated leapfrog!(point; stepsize = 0.1)
    inferred_hamiltonian = @inferred _phase_hamiltonian(point)
    (; cached_allocations, leapfrog_allocations, inferred_hamiltonian)
end

function _criterion_receipt(left, right, backward_velocity, forward_velocity)
    criterion = ReactiveKernelsNUTSExamples._compute_criterion_sum(
        left, right, backward_velocity, forward_velocity,
    )
    allocations = @allocated ReactiveKernelsNUTSExamples._compute_criterion_sum(
        left, right, backward_velocity, forward_velocity,
    )
    (; criterion, allocations)
end

function _trace_product_receipt(left, right)
    value = ReactiveKernelsNUTSExamples._tr_prod(left, right)
    allocations = @allocated ReactiveKernelsNUTSExamples._tr_prod(left, right)
    (; value, allocations)
end

@testset "compiled ReactiveHMC phase points" begin
    gradient_calls = Ref(0)
    counted_gradient(position) = begin
        gradient_calls[] += 1
        _gaussian_gradient(position)
    end
    position = [0.1, -0.2]
    momentum = [0.3, -0.4]
    point = euclidean_phasepoint(
        _gaussian_potential,
        counted_gradient,
        Diagonal(ones(2)),
        copy(position),
        copy(momentum),
    )

    @test point.pos == position
    @test point.mom == momentum
    @test point.pot == 0.025000000000000005
    @test point.dham_dpos == position
    @test point.dham_dmom == momentum
    @test point.ham == 0.15
    @test gradient_calls[] == 2 # one type probe, one first lazy graph demand
    point.ham
    @test gradient_calls[] == 2

    # Exact upstream `@.` syntax mutates a HAVE slot. The phase-point property
    # transaction invalidates dependent compiled slots without an HMC refresh.
    @. point.pos += 0.5
    @test point.dham_dpos == [0.6, 0.3]
    @test gradient_calls[] == 3
    @test occursin("__valid__", string(code_expr(point, :ham)))
    @test plan(point) === reactive_program(point).plan

    receipt_point = euclidean_phasepoint(
        _gaussian_potential,
        _gaussian_gradient,
        Diagonal(ones(2)),
        copy(position),
        copy(momentum),
    )
    receipt = _phasepoint_receipt(receipt_point)
    @test receipt.cached_allocations == 0
    # The allocating gradient/linear-solve operations, rather than reactive
    # orchestration, account for the small leapfrog allocation.
    @test receipt.leapfrog_allocations <= 512
    @test isfinite(receipt.inferred_hamiltonian)

    criterion_receipt = _criterion_receipt(
        [0.3, 0.4], [0.1, 0.2], [1.0, 1.0], [2.0, 1.0],
    )
    @test criterion_receipt.criterion
    @test criterion_receipt.allocations == 0

    trace_receipt = _trace_product_receipt(
        [1.0 2.0; 3.0 4.0], [0.5 1.0; 1.5 2.0],
    )
    @test trace_receipt.value == 14.5
    @test trace_receipt.allocations == 0

    metric(position) = begin
        potential, gradient = _gaussian_gradient(position)
        (potential, gradient, Diagonal(1 .+ abs2.(position)))
    end
    metric_gradient(position) = begin
        potential, gradient, value = metric(position)
        derivative = zeros(
            eltype(position), length(position), length(position), length(position),
        )
        for index in eachindex(position)
            derivative[index, index, index] = 2position[index]
        end
        (potential, gradient, value, derivative)
    end
    riemannian = riemannian_phasepoint(
        _gaussian_potential,
        _gaussian_gradient,
        metric,
        metric_gradient,
        [0.25, -0.5],
        [0.4, 0.1],
    )
    @test riemannian.dham_dmom ≈ [0.4 / 1.0625, 0.1 / 1.25]
    @test riemannian.dham_dpos ≈ [0.4498615916955017, -0.8968]
    initial_hamiltonian = riemannian.ham
    generalized_leapfrog!(riemannian; stepsize = 0.02, n_fi_steps = 2)
    @test isfinite(riemannian.ham)
    @test abs(riemannian.ham - initial_hamiltonian) < 1e-5

    direct = euclidean_phasepoint(
        _gaussian_potential, _gaussian_gradient, Diagonal(ones(2)),
        copy(position), copy(momentum),
    )
    wrapped = copy(direct)
    for _ in 1:3
        leapfrog!(direct; stepsize = 0.02)
    end
    multistep(leapfrog!, wrapped; n_steps = 3, stepsize = 0.06)
    @test wrapped.pos == direct.pos
    @test wrapped.mom == direct.mom
end

@testset "ReactiveHMC dev multinomial NUTS parity" begin
    point = euclidean_phasepoint(
        _gaussian_potential,
        _gaussian_gradient,
        Diagonal(ones(2)),
        [0.1, -0.2],
        [0.3, -0.4],
    )
    stepper = partial(leapfrog!; stepsize = 0.25)
    @test stepper.stepsize == 0.25
    state = ReactiveKernelsNUTSExamples._oracle_nuts_state(
        point;
        rng = Xoshiro(42),
        step_f = stepper,
        max_depth = 3,
    )
    transition = @inferred step!(state)

    # Exact deterministic output from the executable upstream fixture at
    # `test/fixtures/reactivehmc_ca9_reference.jl`, run in a checkout of
    # ReactiveHMC.jl main@ca9ea4ca41924bb0e1fadc01c717e1333916aba6.
    # That revision's `src/nuts.jl` is byte-identical on
    # dev@a8a33f958ab0dffb5696ce7da7fcdcdd6983c208.
    @test state.init.pos == [0.27957763671875, -0.4214599609375]
    @test state.init.mom == [0.15133209228515626, -0.15659484863281245]
    @test state.init.ham == 0.15160775120020845
    @test transition.depth == 3
    @test transition.n_steps == 7
    @test transition.energy_error == -0.0012290068185673575
    @test transition.acceptance_rate == 0.9985695900582436
    @test !transition.diverged
end

function _gaussian_chain(seed; draws = 600, warmup = 200)
    group = reactive_nuts_group(
        _gaussian_gradient!, Diagonal(ones(2)), zeros(2), zeros(2))
    state = nuts_state(
        group;
        rng = Xoshiro(seed),
        step_f = partial(leapfrog!; stepsize = 0.35),
        max_depth = 7,
    )
    sample!(state, draws; discard_initial = warmup)
end

@testset "usable Gaussian NUTS chain" begin
    first_chain = _gaussian_chain(20260825)
    second_chain = _gaussian_chain(20260825)
    @test first_chain.samples == second_chain.samples
    @test first_chain.diagnostics == second_chain.diagnostics
    @test all(isfinite, first_chain.samples)
    @test all(!diagnostic.diverged for diagnostic in first_chain.diagnostics)
    means = vec(mean(first_chain.samples; dims = 2))
    variances = vec(var(first_chain.samples; dims = 2))
    @test all(abs.(means) .< 0.15)
    @test all(abs.(variances .- 1) .< 0.2)
    @test mean(d.acceptance_rate for d in first_chain.diagnostics) > 0.9
    @test sum(d.n_steps for d in first_chain.diagnostics) >
        length(first_chain.diagnostics)
end

@testset "eight-schools graph and DI+Enzyme inside NUTS" begin
    if !isdefined(Main, :ReactiveKernelsPPLExamples)
        if samefile(Base.active_project(), joinpath(@__DIR__, "..", "Project.toml"))
            package_directory = normpath(joinpath(@__DIR__, "..", "packages"))
            package_directory in LOAD_PATH || pushfirst!(LOAD_PATH, package_directory)
        end
        Base.eval(Main, :(using ReactiveKernelsPPLExamples))
    end
    eight_schools = Main.ReactiveKernelsPPLExamples.EightSchoolsExample
    model = eight_schools.build_eight_schools_graph()
    observations = eight_schools.EIGHT_SCHOOLS_Y
    scales = eight_schools.EIGHT_SCHOOLS_SIGMA
    density_kernel = prepare(
        model.graph;
        have = (
            model.unconstrained, model.observations, model.observation_scales,
        ),
        want = (model.posterior,),
    )
    density_calls = Ref(0)
    potential_calls = Ref(0)
    gradient_calls = Ref(0)
    logdensity(position) = begin
        density_calls[] += 1
        density_kernel(collect(position), observations, scales)
    end
    potential(position) = begin
        potential_calls[] += 1
        -logdensity(position)
    end
    # Public compiled-group boundary: differentiate the scalar potential
    # (= -posterior) with RK's shared DI+Enzyme wrapper prepared ONCE. The
    # posterior value and derivative come from one generated-kernel evaluation;
    # only the sampler-facing sign convention is applied here. Model data are
    # rebound as DI Constants on every call, with plain reverse mode and no
    # runtime-activity or function annotation.
    initial = zeros(10)
    initial[2] = log(5.0)
    posterior_ad = prepare_ad(
        density_kernel, _HMC_ENZYME_BACKEND,
        copy(initial), observations, scales; active = :unconstrained,
    )
    potential_gradient!(gradient, position) = begin
        gradient_calls[] += 1
        density_calls[] += 1
        posterior = first(ad_value_and_gradient!(
            posterior_ad, gradient, position, observations, scales))
        gradient .*= -1
        -posterior
    end
    group = reactive_nuts_group(
        potential_gradient!, Diagonal(ones(10)), initial, zeros(10))
    state = nuts_state(
        group;
        rng = Xoshiro(8008),
        step_f = partial(leapfrog!; stepsize = 0.03),
        max_depth = 6,
    )
    # A 300-transition adaptation window keeps this difficult centered-funnel
    # quality gate stable across Julia 1.10–1.12 without weakening its divergence
    # or acceptance thresholds (1.12 evidence: 0 divergences, mean acceptance 0.908).
    warmup = warmup!(state, 300; target_accept = 0.9)
    chain = sample!(state, 100)
    densities = [logdensity(view(chain.samples, :, draw))
                 for draw in axes(chain.samples, 2)]

    @test size(chain.samples) == (10, 100)
    @test all(isfinite, chain.samples)
    @test all(isfinite, densities)
    divergences = count(diagnostic -> diagnostic.diverged, chain.diagnostics)
    @test divergences <= 5
    # The centered funnel is intentionally difficult; require a usable,
    # non-degenerate post-warmup chain without pretending it reaches the
    # nominal target acceptance in only 150 warmup transitions.
    @test mean(diagnostic.acceptance_rate for diagnostic in chain.diagnostics) > 0.7
    @test warmup.final_stepsize > 0
    @test all(>(0), diag(warmup.metric))
    @test maximum(vec(std(chain.samples; dims = 2))) > 0.01
    # The prepared RK/DI boundary owns potential+gradient; the separate
    # potential-only alternative must never be called in NUTS.
    @test potential_calls[] == 0
    @test gradient_calls[] > 1
    @test density_calls[] > gradient_calls[]
end

@testset "ReactiveHMC adaptation utilities" begin
    adaptation = dual_averaging_state(0.1)
    @test adaptation.current ≈ 1.0
    fit!(adaptation, 0.9)
    @test isfinite(adaptation.current)
    @test isfinite(adaptation.final)

    variance = welford_var(2)
    step!(variance, [1.0, 2.0])
    step!(variance, [2.0, 4.0])
    step!(variance, [3.0, 6.0])
    @test variance.mean == [2.0, 4.0]
    @test variance.var ≈ [2 / 3, 8 / 3]

    group = reactive_nuts_group(
        _gaussian_gradient!, Diagonal(ones(2)), [0.1, -0.2], [0.3, -0.4])
    trajectory = trajectory_stats(2)
    state = nuts_state(
        group;
        rng = Xoshiro(42),
        step_f = partial(leapfrog!; stepsize = 0.25),
        stats_f = trajectory,
        max_depth = 3,
    )
    transition = sample!(state)
    @test size(trajectory.positions) == (2, transition.n_steps + 1)
    @test size(trajectory.gradients) == size(trajectory.positions)
    @test length(trajectory.dhams) == transition.n_steps + 1
    @test length(trajectory.pots) == transition.n_steps + 1
    @test sort(trajectory.idxs) == 0:transition.n_steps

    run_stats = sampling_stats(trajectory)
    run_stats(state, adaptation)
    @test run_stats.draws[:, 1] == state.init.pos
    @test run_stats.n_steps == [transition.n_steps]
    @test run_stats.stepsizes == [0.25]
    @test run_stats.acc_rate == [transition.acceptance_rate]
    @test run_stats.diverged == [transition.diverged]
    @test run_stats.full_history[1] == trajectory.positions
    @test run_stats.full_idxs[1] == trajectory.idxs

    # Spell out the fixed two-dimensional metric instead of capturing a mutable
    # Vector as temporary function state; static activity can then prove the
    # complete boundary without runtime checks.
    scales = (0.25, 4.0)
    anisotropic_potential(position) =
        (abs2(position[1]) / 0.25 + abs2(position[2]) / 4.0) / 2
    anisotropic_preparation =
        prepare_gradient(anisotropic_potential, _HMC_ENZYME_BACKEND, zeros(2))
    anisotropic_gradient!(gradient, position) = first(value_and_gradient!(
        anisotropic_potential, gradient, anisotropic_preparation,
        _HMC_ENZYME_BACKEND, position))
    adapted_group = reactive_nuts_group(
        anisotropic_gradient!, Diagonal(ones(2)), zeros(2), zeros(2))
    adapted_state = nuts_state(
        adapted_group;
        rng = Xoshiro(20260825),
        step_f = partial(leapfrog!; stepsize = 1.0),
        max_depth = 7,
    )
    warmup = warmup!(
        adapted_state, 120;
        initial_buffer = 20,
        terminal_buffer = 20,
        first_window = 20,
    )
    @test isfinite(warmup.initial_stepsize)
    @test warmup.initial_stepsize > 0
    @test isfinite(warmup.final_stepsize)
    @test warmup.final_stepsize > 0
    @test warmup.metric isa Diagonal
    @test all(isfinite, diag(warmup.metric))
    @test all(>(0), diag(warmup.metric))
    @test diag(warmup.metric) != ones(2)
    @test diag(warmup.metric)[1] > diag(warmup.metric)[2]
    @test warmup.metric_window_ends == [40, 100]
    @test length(warmup.diagnostics) == 120

    adapted_chain = sample!(adapted_state, 400)
    adapted_means = vec(mean(adapted_chain.samples; dims = 2))
    adapted_variances = vec(var(adapted_chain.samples; dims = 2))
    @test all(isfinite, adapted_chain.samples)
    @test all(!diagnostic.diverged for diagnostic in adapted_chain.diagnostics)
    @test all(abs.(adapted_means) .< [0.2, 0.8])
    @test all(abs.(adapted_variances ./ scales .- 1) .< 0.4)

    short_state = nuts_state(
        reactive_nuts_group(
            _gaussian_gradient!, Diagonal(ones(2)), zeros(2), zeros(2));
        rng = Xoshiro(7),
        step_f = partial(leapfrog!; stepsize = 0.5),
        max_depth = 4,
    )
    short_warmup = warmup!(short_state, 19)
    @test isempty(short_warmup.metric_window_ends)
    @test short_warmup.metric == Diagonal(ones(2))
end
