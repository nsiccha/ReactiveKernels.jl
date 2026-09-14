module HMCBenchmark
using ReactiveKernels, LinearAlgebra, Statistics, Dates
include("density_callbacks.jl")
using .DensityCallbacks: Potential, Gradient, CallbackHandle
include("../nuts_kernel_authoring_fixture_b.jl")
include("position_multinomial_hmc_kernel.jl")
const F = NUTSBMutationAuthoringFixture

"""Prepare the existing authored multinomial HMC loop for any RK density/AD pair."""
function prepare_hmc(density, ad, position, rng; backend=:native,
        transitions=1000, steps=16, stepsize=0.03,
        metric=Diagonal(ones(length(position))), context=())
    point = transpiled_endpoint(F.euclidean_phasepoint, F.leapfrog!,
        CallbackHandle(Potential(density, context)),
        CallbackHandle(Gradient(ad, context)), metric, position, zeros(length(position)))
    prepare_transpiled(PositionMultinomialHMCAuthoring.multinomial_hmc_state, point;
        backend, method=:step!, argument=rng, iterations=transitions,
        kernel_kwargs=(n_steps=steps, step_f=F.leapfrog!, stepsize),
        outputs=(position=(:init, :pos),))
end

"""
Measure synchronized batches through the public transpiled-program interface.
RNG construction and result inspection are outside timing; wrapper-owned state
copies/output snapshots remain inside. This measures throughput, not ESS.
"""
function benchmark_hmc(density, ad, position, make_rng; backend=:native,
        transitions=1000, steps=16, stepsize=0.03, rounds=9,
        metric=Diagonal(ones(length(position))), context=())
    transitions > 0 && steps > 0 && rounds > 0 || error("positive work counts required")
    rng = make_rng(0)
    prepare_seconds = @elapsed program = prepare_hmc(density, ad, position, rng;
        backend, transitions, steps, stepsize, metric, context)
    state = initial_transpiled_state(program)
    first_seconds = @elapsed first_result = program(state, rng)
    inspect(result) = begin
        q = Array(result.outputs.position)
        all(isfinite, q) || error("nonfinite HMC position")
        value = density(q, context...)
        isfinite(value) || error("nonfinite HMC density")
        q, value
    end
    inspect(first_result)
    # Let runtime caches settle after first execution, especially for tiny
    # batches. Keep this work separate from both cold and warmed measurements.
    for _ in 1:3
        inspect(program(state, rng))
    end
    # Distinct seeds are constructed before the timer. Every replicate starts
    # at the same position; continuation is checked separately below.
    rngs = [make_rng(i) for i in 1:rounds]
    times, bytes, gc_seconds = Float64[], Int[], Float64[]
    final_values, displacements = Float64[], Float64[]
    started = string(now(UTC))
    for rng_i in rngs
        measured = @timed program(state, rng_i)
        q, value = inspect(measured.value)
        push!(times, measured.time)
        push!(bytes, measured.bytes)
        push!(gc_seconds, measured.gctime)
        push!(final_values, value)
        push!(displacements, norm(q - position))
    end
    continued = first_result
    previous = Array(continued.outputs.position)
    moved = 0
    for _ in 1:4
        continued = program(continued.state, continued.argument)
        q, _ = inspect(continued)
        moved += q != previous
        previous = q
    end
    moved > 0 || error("continuation never moved")
    any(>(0), displacements) || error("timed batches never moved from their initial position")
    Dict("backend"=>string(backend), "transitions"=>transitions,
        "leapfrog_steps"=>steps, "leapfrog_steps_per_batch"=>transitions*steps,
        "stepsize"=>stepsize, "rounds"=>rounds, "extra_warmup_batches"=>3,
        "started_utc"=>started,
        "finished_utc"=>string(now(UTC)), "prepare_seconds"=>prepare_seconds,
        "first_execution_seconds"=>first_seconds, "raw_batch_seconds"=>times,
        "raw_julia_bytes"=>bytes, "raw_gc_seconds"=>gc_seconds,
        "median_batch_seconds"=>median(times),
        "median_us_per_transition"=>1e6*median(times)/transitions,
        "final_logdensities"=>final_values, "final_displacement_norms"=>displacements,
        "timed_moved_batches"=>count(>(0), displacements),
        "continuation_batches"=>4, "continuation_moved_batches"=>moved)
end
end
