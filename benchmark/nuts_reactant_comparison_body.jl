module NUTSReactantComparison

using Dates
using LinearAlgebra
using Pkg
using Printf
using Random
using Reactant
using ReactiveKernels
using Statistics
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "examples", "nuts_runtime.jl"))

module Fixture
include(joinpath(@__DIR__, "nuts_kernel_authoring_fixture.jl"))
end

Random.eval(quote
    mutable struct RKReactantBenchmarkReplayRNG <: AbstractRNG
        momentum::Vector{Float64}
        directions::Vector{Bool}
        exponentials::Vector{Float64}
        direction_index::Int
        exponential_index::Int
    end
end)
const ReplayRNG = Random.RKReactantBenchmarkReplayRNG
Random.randn!(rng::ReplayRNG, destination::AbstractArray) =
    (copyto!(destination, rng.momentum); destination)
Base.rand(rng::ReplayRNG, ::Type{Bool}) =
    (rng.direction_index += 1; rng.directions[rng.direction_index])
Random.randexp(rng::ReplayRNG) =
    (rng.exponential_index += 1; rng.exponentials[rng.exponential_index])

const PF = ReactiveKernels._prepare_factory(
    Fixture.euclidean_phasepoint,
    ReactiveKernels.kernel_registration(Fixture.leapfrog!),
)

function _potential(position)
    quadratic = sum(abs2, position)
    oftype(quadratic, 0.5) * quadratic
end
function _gradient!(gradient, position)
    gradient .= position
    _potential(position)
end

function _values(dimension)
    metric = Matrix{Float64}(I, dimension, dimension)
    position = collect(range(-0.8, 0.8; length = dimension))
    values = Dict{Int,Any}()
    for slot in ReactiveKernels.kernel_plan_slots(
            ReactiveKernels.kernel_prepared_plan(PF))
        name = String(slot.path[1])
        values[slot.canon] = name == "pot_f" ? _potential :
            name == "grad_f" ? _gradient! :
            name == "metric" ? metric :
            name == "chol_metric" ? cholesky(metric) :
            startswith(name, "##node") ? 0.0 :
            name == "pos" ? copy(position) :
            name == "mom" ? zeros(dimension) :
            name in ("dpot_dpos", "dham_dpos", "dkin_dmom", "dham_dmom") ?
                zeros(dimension) : 0.0
    end
    values
end

function _frame(dimension, max_depth, stepsize)
    frame = ReactiveKernels._construct_nuts_frame(
        PF, _values(dimension), max_depth;
        step_f=ReactiveKernels.partial(Fixture.leapfrog!; stepsize),
        stats_f=Fixture.nuts_stats!, min_dham=-1000.0,
    )
    ReactiveKernels.compile_prepared_initialization(
        PF, typeof(frame.init), typeof(frame.shared))(
            frame.init, frame.shared,
            ReactiveKernels.kernel_prepared_handles(PF))
    ReactiveKernels._seed_nuts_children!(frame)
    frame
end

function _bundle(rng, dimension, max_depth)
    momentum = randn(rng, dimension)
    directions = rand(rng, Bool, max_depth)
    exponentials = randexp(rng, 1 << max_depth)
    ReactiveKernels.nuts_reactant_bundle(
        momentum, directions, exponentials, max_depth)
end

function _replay(bundle)
    ReplayRNG(
        copy(bundle.momentum), Bool.(bundle.dirs), copy(bundle.exps), 0, 0)
end

_close(a, b) = all(isapprox.(a, b; rtol=0, atol=128eps(Float64)))

function _parity(native_frame, output, replay)
    position = Array(output.pp_pos)[:, 1]
    momentum = Array(output.pp_mom)[:, 1]
    dpotential = Array(output.pp_dpot)[:, 1]
    dkinetic = Array(output.pp_dkin)[:, 1]
    scalars = (Array(output.pp_pot)[1], Array(output.pp_kin)[1],
               Array(output.pp_ham)[1])
    phase_ok =
        _close(getfield(native_frame.init, :f4), position) &&
        _close(getfield(native_frame.init, :f5), momentum) &&
        _close(getfield(native_frame.init, :f8), dpotential) &&
        _close(getfield(native_frame.init, :f10), dkinetic) &&
        _close((getfield(native_frame.init, :f7), getfield(native_frame.init, :f11),
                getfield(native_frame.init, :f12)), scalars)
    diagnostics_ok =
        native_frame.diag.n_steps == Array(output.n_steps)[1] &&
        native_frame.diag.reached_depth == Array(output.reached_depth)[1] &&
        _close((native_frame.diag.acceptance_rate, native_frame.diag.dham),
               (Array(output.acc)[1], Array(output.dham)[1])) &&
        Int(native_frame.diverged) == Array(output.diverged)[1]
    controls_ok =
        replay.direction_index == Array(output.kd)[1] &&
        replay.exponential_index == Array(output.ke)[1] &&
        all(iszero, Array(output.overflow)) &&
        Array(output.csp)[1] == 0
    phase_ok && diagnostics_ok && controls_ok
end

function _parity_diagnostic(native_frame, output, replay)
    native_position = getfield(native_frame.init, :f4)
    reactant_position = Array(output.pp_pos)[:, 1]
    (;
        native_steps = native_frame.diag.n_steps,
        reactant_steps = Array(output.n_steps)[1],
        native_depth = native_frame.diag.reached_depth,
        reactant_depth = Array(output.reached_depth)[1],
        native_diverged = native_frame.diverged,
        reactant_diverged = Bool(Array(output.diverged)[1]),
        native_directions = replay.direction_index,
        reactant_directions = Array(output.kd)[1],
        native_exponentials = replay.exponential_index,
        reactant_exponentials = Array(output.ke)[1],
        max_position_delta = maximum(abs.(native_position .- reactant_position)),
        overflow = Array(output.overflow),
    )
end

_median(values) = median(Float64.(values))

function _package_version(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(info.version)
    end
    error("package $name is absent from the benchmark environment")
end

_git(args...) = readchomp(Cmd(["git", "-C", ROOT, string.(args)...]))

function _output_path()
    for argument in ARGS
        startswith(argument, "--output=") &&
            return split(argument, '='; limit=2)[2]
    end
    nothing
end

function run_benchmark()
    dimension = parse(Int, get(ENV, "RK_NUTS_REACTANT_DIMENSION", "5"))
    max_depth = parse(Int, get(ENV, "RK_NUTS_REACTANT_MAX_DEPTH", "10"))
    stepsize = parse(Float64, get(ENV, "RK_NUTS_REACTANT_STEPSIZE", "0.3"))
    rounds = parse(Int, get(ENV, "RK_NUTS_REACTANT_ROUNDS", "9"))
    transitions_per_round =
        parse(Int, get(ENV, "RK_NUTS_REACTANT_TRANSITIONS", "32"))
    dimension == 5 || throw(ArgumentError(
        "publication protocol requires dimension=5; use 5 for exploratory runs too"))
    max_depth == 10 || throw(ArgumentError(
        "publication protocol requires max_depth=10; use 10 for exploratory runs too"))
    rounds > 0 || throw(ArgumentError("rounds must be positive"))
    transitions_per_round > 0 ||
        throw(ArgumentError("transitions per round must be positive"))

    compile_frame = _frame(dimension, max_depth, stepsize)
    reactant_frame = _frame(dimension, max_depth, stepsize)
    native_compile_seconds = @elapsed native = ReactiveKernels.compile_nuts(
        PF, Fixture.nuts_state, Fixture.refresh_momentum!!,
        Fixture.nuts!!, compile_frame)
    reactant_lower_seconds = @elapsed compiled = ReactiveKernels.compile_nuts_reactant(
        PF, Fixture.nuts_state, Fixture.refresh_momentum!!,
        Fixture.nuts!!, reactant_frame)

    rng = Xoshiro(0x726b5f6e757473)
    transition_count = rounds * transitions_per_round
    warm_bundle = _bundle(rng, dimension, max_depth)
    candidate_bundles = [
        _bundle(rng, dimension, max_depth)
        for _ in 1:max(4transition_count, transition_count + 64)
    ]
    warm_host_state = ReactiveKernels.nuts_reactant_state(
        compiled, reactant_frame, warm_bundle)
    warm_transfer_seconds = @elapsed warm_state =
        map(Reactant.to_rarray, warm_host_state)
    reactant_compile_seconds = @elapsed executable =
        ReactiveKernels.nuts_reactant_compile(compiled, warm_state; sync=true)
    stablehlo_while_count =
        count(_ -> true, eachmatch(r"stablehlo\.while", executable.module_string))

    # One warm-up transition removes first-call runtime effects from the
    # steady-state corpus. Both arms consume the exact same full-capacity bundle.
    warm_replay = _replay(warm_bundle)
    warm_native_frame = _frame(dimension, max_depth, stepsize)
    native_first_seconds = @elapsed native.root!(
        warm_native_frame, native.scratch, warm_replay)
    reactant_first_seconds = @elapsed warm_state = executable(warm_state)
    _parity(warm_native_frame, warm_state, warm_replay) ||
        error("native/Reactant parity failed during warm-up")

    # Adaptive NUTS is numerically chaotic at a U-turn boundary: an ulp-scale
    # backend difference can validly change the branch and therefore the amount
    # of work. Screen the deterministic candidate stream outside timing and
    # publish how many candidates were excluded. The timed corpus consequently
    # compares identical control flow/work; it is not an end-to-end sampler run.
    selected_bundles = Vector{typeof(warm_bundle)}()
    candidates_examined = 0
    candidates_rejected = 0
    screening_seconds = @elapsed for bundle in candidate_bundles
        candidates_examined += 1
        frame = _frame(dimension, max_depth, stepsize)
        replay = _replay(bundle)
        host_state = ReactiveKernels.nuts_reactant_state(compiled, frame, bundle)
        input_state = map(Reactant.to_rarray, host_state)
        native.root!(frame, native.scratch, replay)
        output = executable(input_state)
        if _parity(frame, output, replay)
            push!(selected_bundles, bundle)
            length(selected_bundles) == transition_count && break
        else
            candidates_rejected += 1
        end
    end
    length(selected_bundles) == transition_count || error(
        "only $(length(selected_bundles)) of $transition_count matched-control " *
        "transitions found in $(length(candidate_bundles)) deterministic candidates")

    transition_frames = [
        _frame(dimension, max_depth, stepsize) for _ in 1:transition_count
    ]
    transition_host_states = [
        ReactiveKernels.nuts_reactant_state(
            compiled, transition_frames[index], selected_bundles[index])
        for index in eachindex(transition_frames)
    ]
    state_transfer_seconds = @elapsed transition_states = [
        map(Reactant.to_rarray, state) for state in transition_host_states
    ]

    native_round_seconds = Float64[]
    reactant_round_seconds = Float64[]
    round_steps = Int[]
    round_directions = Int[]
    round_exponentials = Int[]
    round_max_depth = Int[]
    round_divergences = Int[]
    cursor = 1
    for round in 1:rounds
        native_elapsed = 0.0
        reactant_elapsed = 0.0
        steps = 0
        directions = 0
        exponentials = 0
        reached_depth = 0
        divergences = 0
        for _ in 1:transitions_per_round
            bundle = selected_bundles[cursor]
            replay = _replay(bundle)
            native_frame = transition_frames[cursor]
            input_state = transition_states[cursor]
            if isodd(round)
                native_elapsed += @elapsed native.root!(
                    native_frame, native.scratch, replay)
                reactant_elapsed += @elapsed output = executable(input_state)
            else
                reactant_elapsed += @elapsed output = executable(input_state)
                native_elapsed += @elapsed native.root!(
                    native_frame, native.scratch, replay)
            end
            _parity(native_frame, output, replay) || error(
                "native/Reactant parity failed at round $round transition $cursor: " *
                string(_parity_diagnostic(native_frame, output, replay)))
            steps += native_frame.diag.n_steps
            directions += replay.direction_index
            exponentials += replay.exponential_index
            reached_depth = max(reached_depth, native_frame.diag.reached_depth)
            divergences += Int(native_frame.diverged)
            cursor += 1
        end
        push!(native_round_seconds, native_elapsed)
        push!(reactant_round_seconds, reactant_elapsed)
        push!(round_steps, steps)
        push!(round_directions, directions)
        push!(round_exponentials, exponentials)
        push!(round_max_depth, reached_depth)
        push!(round_divergences, divergences)
        @printf(
            "round=%d steps=%d native_ms=%.3f reactant_ms=%.3f\n",
            round, steps, 1e3native_elapsed / transitions_per_round,
            1e3reactant_elapsed / transitions_per_round,
        )
    end

    native_transition_ms =
        1e3 .* native_round_seconds ./ transitions_per_round
    reactant_transition_ms =
        1e3 .* reactant_round_seconds ./ transitions_per_round
    native_steps_per_second = round_steps ./ native_round_seconds
    reactant_steps_per_second = round_steps ./ reactant_round_seconds
    native_median_ms = _median(native_transition_ms)
    reactant_median_ms = _median(reactant_transition_ms)
    native_median_steps = _median(native_steps_per_second)
    reactant_median_steps = _median(reactant_steps_per_second)

    dirty = !isempty(_git("status", "--porcelain", "--untracked-files=no"))
    receipt = Dict{String,Any}(
        "schema" => "nuts-reactant-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(
                ENV, "REACTIVEKERNELS_CANDIDATE_SHA", _git("rev-parse", "HEAD")),
            "reactivekernels_dirty" => dirty,
            "accepted_implementation_sha" =>
                "0c96826957b9eee177be5818239e30e587109d8c",
            "authored_fixture_blob" =>
                _git("rev-parse", "HEAD:benchmark/nuts_kernel_authoring_fixture.jl"),
            "reactant_version" => string(Base.pkgversion(Reactant)),
            "reactant_jll_version" => _package_version("Reactant_jll"),
            "julia_version" => string(VERSION),
        ),
        "environment" => Dict(
            "os" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(),
            "reactant_backend" => "default CPU",
        ),
        "protocol" => Dict(
            "target" => "5D standard normal potential with analytic in-place gradient",
            "dimension" => dimension,
            "metric" => "unit diagonal Euclidean",
            "stepsize" => stepsize,
            "max_depth" => max_depth,
            "rounds" => rounds,
            "transitions_per_round" => transitions_per_round,
            "state" =>
                "independent matched starting state for every full adaptive transition",
            "randomness" =>
                "identical pre-generated full-capacity momentum/direction/exponential bundles selected from a deterministic candidate stream",
            "bundle_seed_hex" => "0x726b5f6e757473",
            "candidate_selection" =>
                "first N candidates with native/Reactant observable and random-consumption parity; screening is outside timing",
            "candidate_bundles_examined" => candidates_examined,
            "candidate_bundles_rejected" => candidates_rejected,
            "timing" =>
                "per-call @elapsed; median of per-round aggregate synchronous execution",
            "reactant_sync" => true,
            "compile_time_in_steady_state" => false,
            "host_device_transfers_in_steady_state" => false,
            "rng_generation_in_steady_state" => false,
            "result_readback_in_steady_state" => false,
            "input_rebundle_in_steady_state" => false,
            "per_transition_state_setup_in_steady_state" => false,
            "parity_screening_in_steady_state" => false,
            "compile_order" =>
                ["native source compiler", "Reactant lowering", "Reactant XLA compile"],
            "compile_times_include_first_julia_jit" => true,
            "adaptation_measured" => false,
            "ess_measured" => false,
            "end_to_end_sampling_measured" => false,
        ),
        "compilation" => Dict(
            "native_seconds" => native_compile_seconds,
            "reactant_lower_seconds" => reactant_lower_seconds,
            "reactant_xla_seconds" => reactant_compile_seconds,
            "reactant_total_seconds" =>
                reactant_lower_seconds + reactant_compile_seconds,
            "native_first_execution_seconds" => native_first_seconds,
            "reactant_first_execution_seconds" => reactant_first_seconds,
            "warm_input_transfer_seconds" => warm_transfer_seconds,
            "all_input_state_transfers_seconds" => state_transfer_seconds,
            "matched_corpus_screening_seconds" => screening_seconds,
        ),
        "raw" => Dict(
            "round_steps" => round_steps,
            "round_directions" => round_directions,
            "round_exponentials" => round_exponentials,
            "round_max_reached_depth" => round_max_depth,
            "round_divergences" => round_divergences,
            "native_round_seconds" => native_round_seconds,
            "reactant_round_seconds" => reactant_round_seconds,
            "native_transition_ms" => native_transition_ms,
            "reactant_transition_ms" => reactant_transition_ms,
            "native_steps_per_second" => native_steps_per_second,
            "reactant_steps_per_second" => reactant_steps_per_second,
        ),
        "medians" => Dict(
            "native_transition_ms" => native_median_ms,
            "reactant_transition_ms" => reactant_median_ms,
            "native_steps_per_second" => native_median_steps,
            "reactant_steps_per_second" => reactant_median_steps,
            "reactant_over_native_transition_time" =>
                reactant_median_ms / native_median_ms,
            "reactant_over_native_steps_per_second" =>
                reactant_median_steps / native_median_steps,
        ),
        "acceptance" => Dict(
            "same_authored_transition" => true,
            "same_target_metric_state_depth_randomness" => true,
            "matched_independent_start_states" => true,
            "matched_control_flow_corpus" => true,
            "parity_screening_reported" => true,
            "per_transition_observable_parity" => true,
            "random_consumption_parity" => true,
            "all_overflow_flags_zero" => true,
            "stablehlo_while_count" => stablehlo_while_count,
        ),
    )

    output = _output_path()
    if output === nothing
        TOML.print(stdout, receipt; sorted=true)
    else
        mkpath(dirname(abspath(output)))
        open(output, "w") do io
            TOML.print(io, receipt; sorted=true)
        end
        println("receipt=", abspath(output))
    end
    receipt
end

end # module NUTSReactantComparison
