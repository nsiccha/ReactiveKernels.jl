module OnlineStatsExample

using ReactiveKernels
using Statistics

# NUTS diagnostics are part of the external sampler exemplar, not RK's package API.
isdefined(Main, :ReactiveKernelsNUTSExample) ||
    Base.include(Main, joinpath(@__DIR__, "nuts_runtime.jl"))
using Main.ReactiveKernelsNUTSExample: NUTSDiagnostics, welford_var, step!

export MomentsAccumulator, update, fit
export HMCDiagnosticsAccumulator, record_transition, fit_diagnostics
export sample_count, max_tree_depth, divergence_rate, divergence_percent
export mean_tree_depth, mean_leapfrog_steps, mean_acceptance_rate
export mean_energy_error, mean_stepsize
export metric_adaptation_report
export build_online_stats_graph
export kernel_performance_report, diagnostics_performance_report
export reactive_performance_report
export demo

"""
    MomentsAccumulator(T=Float64)
    MomentsAccumulator{T}(n, mean, m2)

Immutable sufficient statistics for the mean and variance of a scalar stream.
`T` must be a floating-point type so the empty and corrected-singleton results
can be represented as `NaN` without changing the return type.

States maintain `m2 >= 0`, except that `NaN` and `+Inf` propagate from
non-finite observations. Negative values, including `-Inf`, are rejected. The
update and merge algorithms clamp only a finite negative value within a small
floating-point roundoff tolerance back to zero; a larger negative value is an
error.
"""
struct MomentsAccumulator{T<:AbstractFloat}
    n::Int
    mean::T
    m2::T

    function MomentsAccumulator{T}(n::Integer, mean::Real,
                                   m2::Real) where {T<:AbstractFloat}
        count = Int(n)
        count >= 0 || throw(ArgumentError("observation count must be non-negative"))
        μ = convert(T, mean)
        second = convert(T, m2)
        second < zero(T) &&
            throw(ArgumentError("m2 must not be negative"))
        count == 0 && (!iszero(μ) || !iszero(second)) &&
            throw(ArgumentError("an empty accumulator must have zero mean and m2"))
        count == 1 && !iszero(second) &&
            throw(ArgumentError("a singleton accumulator must have zero m2"))
        new{T}(count, μ, second)
    end
end

MomentsAccumulator(::Type{T}=Float64) where {T<:AbstractFloat} =
    MomentsAccumulator{T}(0, zero(T), zero(T))

function _nonnegative_m2(value::T, scale::T) where {T<:AbstractFloat}
    (isnan(value) || value >= zero(T)) && return value
    isinf(value) &&
        throw(DomainError(value, "m2 became negative infinity"))
    tolerance = T(16) * eps(T) * max(one(T), abs(scale))
    value >= -tolerance && return zero(T)
    throw(DomainError(value, "m2 became negative beyond floating-point roundoff"))
end

"""
    update(accumulator, observation)

Return a new accumulator containing `observation`. This is Welford's stable
single-pass update expressed as a pure function, so it is safe to use as a
`ReactiveKernels` recipe.
"""
function update(accumulator::MomentsAccumulator{T},
                observation::Real) where {T<:AbstractFloat}
    x = convert(T, observation)
    accumulator.n == 0 && return MomentsAccumulator{T}(1, x, zero(T))

    n = Base.checked_add(accumulator.n, 1)
    W = promote_type(T, Float64)
    wide_x = W(x)
    wide_mean = W(accumulator.mean)
    delta = wide_x - wide_mean
    next_mean = wide_mean + delta / W(n)
    increment = delta * (wide_x - next_mean)
    wide_m2 = W(accumulator.m2)
    next_m2 = _nonnegative_m2(wide_m2 + increment,
                              abs(wide_m2) + abs(increment))
    MomentsAccumulator{T}(n, next_mean, next_m2)
end

"""
    fit(accumulator, observations)
    fit(T, observations)
    fit(observations)

Fold observations into an immutable [`MomentsAccumulator`](@ref). The default
storage type is `Float64`.
"""
function fit(accumulator::MomentsAccumulator, observations)
    for observation in observations
        accumulator = update(accumulator, observation)
    end
    accumulator
end

fit(::Type{T}, observations) where {T<:AbstractFloat} =
    fit(MomentsAccumulator(T), observations)
fit(observations) = fit(Float64, observations)

"""
    merge(a::MomentsAccumulator, b::MomentsAccumulator)

Combine two partitions with Chan's parallel-variance formula. Empty
accumulators are exact identities. Both arguments must use the same floating
storage type, keeping the return type statically known. Count ratios and the
cross term use at least `Float64` arithmetic so narrow storage types never
convert a valid large `Int` count to floating-point infinity.
"""
function Base.merge(a::MomentsAccumulator{T},
                    b::MomentsAccumulator{T}) where {T<:AbstractFloat}
    a.n == 0 && return b
    b.n == 0 && return a

    n = Base.checked_add(a.n, b.n)
    W = promote_type(T, Float64)
    wide_a_mean = W(a.mean)
    wide_b_mean = W(b.mean)
    delta = wide_b_mean - wide_a_mean
    b_weight = W(b.n) / W(n)
    next_mean = wide_a_mean + delta * b_weight
    cross = delta * delta * (W(a.n) * W(b.n) / W(n))
    wide_a_m2 = W(a.m2)
    wide_b_m2 = W(b.m2)
    next_m2 = _nonnegative_m2(wide_a_m2 + wide_b_m2 + cross,
                              abs(wide_a_m2) + abs(wide_b_m2) + abs(cross))
    MomentsAccumulator{T}(n, next_mean, next_m2)
end

"Return the stream mean, or `NaN` for an empty accumulator."
Statistics.mean(accumulator::MomentsAccumulator{T}) where {T} =
    accumulator.n == 0 ? T(NaN) : accumulator.mean

"""
    var(accumulator; corrected=true)

Return the sample variance by default. Empty and corrected-singleton variance
are `NaN`; uncorrected singleton variance is zero.
"""
function Statistics.var(accumulator::MomentsAccumulator{T};
                        corrected::Bool=true) where {T}
    denominator = accumulator.n - Int(corrected)
    denominator > 0 || return T(NaN)
    W = promote_type(T, Float64)
    T(W(accumulator.m2) / W(denominator))
end

"""
    HMCDiagnosticsAccumulator(T=Float64)

Fixed-size, mergeable summaries of the per-transition diagnostics the
compiled-reactive NUTS sampler reports as a [`NUTSDiagnostics`](@ref): tree
depth, leapfrog-step count, mean acceptance rate, and completed-transition
energy error, together with an exact divergence count and the exact maximum
observed tree depth. The four continuous summaries use
[`MomentsAccumulator`](@ref); divergences and the maximum depth are exact.

Step size is **not** part of `NUTSDiagnostics` — it is an adaptation quantity
owned by the sampler's dual-averaging state — so it is ingested separately when
a value is available and represented as unavailable (`NaN`/`missing`) summary
otherwise. The energy-error and step-size summaries therefore fold only their
finite (and, for step size, positive) observations, so their counts may be
below the transition count while depth, leapfrog, and acceptance are recorded
for every transition.

This state deliberately does not claim to represent rank-normalized R-hat or
ESS. Those diagnostics require retained, ordered draws from multiple chains and
cannot be recovered exactly from a constant-size mergeable accumulator. For the
reactive, in-place diagonal-metric variance the sampler adapts, reuse the
canonical [`WelfordVariance`](@ref)/[`welford_var`](@ref) (see
[`metric_adaptation_report`](@ref)) rather than this mergeable scalar surface.
"""
struct HMCDiagnosticsAccumulator{T<:AbstractFloat}
    n_divergent::Int
    max_depth::Int
    depth::MomentsAccumulator{T}
    leapfrog_steps::MomentsAccumulator{T}
    acceptance_rate::MomentsAccumulator{T}
    energy_error::MomentsAccumulator{T}
    stepsize::MomentsAccumulator{T}

    function HMCDiagnosticsAccumulator{T}(
        n_divergent::Integer,
        max_depth::Integer,
        depth::MomentsAccumulator{T},
        leapfrog_steps::MomentsAccumulator{T},
        acceptance_rate::MomentsAccumulator{T},
        energy_error::MomentsAccumulator{T},
        stepsize::MomentsAccumulator{T},
    ) where {T<:AbstractFloat}
        divergences = Int(n_divergent)
        divergences >= 0 ||
            throw(ArgumentError("divergence count must be non-negative"))
        deepest = Int(max_depth)
        deepest >= 0 ||
            throw(ArgumentError("maximum tree depth must be non-negative"))
        n = depth.n
        leapfrog_steps.n == n && acceptance_rate.n == n ||
            throw(ArgumentError("per-transition diagnostic counts must agree"))
        energy_error.n <= n && stepsize.n <= n ||
            throw(ArgumentError("optional diagnostic counts cannot exceed the transition count"))
        divergences <= n ||
            throw(ArgumentError("divergence count cannot exceed the transition count"))
        n == 0 && deepest != 0 &&
            throw(ArgumentError("an empty accumulator must have zero maximum tree depth"))
        new{T}(divergences, deepest, depth, leapfrog_steps, acceptance_rate,
               energy_error, stepsize)
    end
end

function HMCDiagnosticsAccumulator(::Type{T}=Float64) where {T<:AbstractFloat}
    empty = MomentsAccumulator(T)
    HMCDiagnosticsAccumulator{T}(0, 0, empty, empty, empty, empty, empty)
end

"Number of transitions summarized by an HMC diagnostics accumulator."
sample_count(accumulator::HMCDiagnosticsAccumulator) = accumulator.depth.n

"Exact maximum tree depth observed, or `0` for an empty state."
max_tree_depth(accumulator::HMCDiagnosticsAccumulator) = accumulator.max_depth

"Fraction of divergent transitions, or `NaN` when no transitions were seen."
function divergence_rate(accumulator::HMCDiagnosticsAccumulator{T}) where {T}
    n = sample_count(accumulator)
    n == 0 && return T(NaN)
    W = promote_type(T, Float64)
    T(W(accumulator.n_divergent) / W(n))
end

"Divergent transitions as a percentage, or `NaN` for an empty state."
function divergence_percent(accumulator::HMCDiagnosticsAccumulator{T}) where {T}
    rate = divergence_rate(accumulator)
    T(T(100) * rate)
end

"Mean tree depth, or `NaN` for an empty state."
mean_tree_depth(accumulator::HMCDiagnosticsAccumulator) = mean(accumulator.depth)

"Mean leapfrog-step count, or `NaN` for an empty state."
mean_leapfrog_steps(accumulator::HMCDiagnosticsAccumulator) =
    mean(accumulator.leapfrog_steps)

"Mean transition acceptance rate, or `NaN` for an empty state."
mean_acceptance_rate(accumulator::HMCDiagnosticsAccumulator) =
    mean(accumulator.acceptance_rate)

"Mean completed-transition energy error over finite observations, or `NaN`."
mean_energy_error(accumulator::HMCDiagnosticsAccumulator) =
    mean(accumulator.energy_error)

"""
Mean sampler step size over the transitions that reported an available (finite,
positive) step size, or `NaN` when no step size was available.
"""
mean_stepsize(accumulator::HMCDiagnosticsAccumulator) = mean(accumulator.stepsize)

"""
    record_transition(accumulator, diagnostics::NUTSDiagnostics, stepsize=NaN)

Fold one canonical [`NUTSDiagnostics`](@ref) transition record into the
mergeable summary. Tree depth, leapfrog count, and acceptance rate are recorded
for every transition, and the maximum tree depth is tracked exactly.

The completed-transition `energy_error` is folded only when finite. Because the
canonical sampler normalizes any non-finite energy error to `-Inf` and marks
that transition divergent, a non-finite error is accepted only when `diverged`
is true (captured by the divergence count, not folded); a non-finite error with
`diverged == false` is invalid telemetry and is rejected.

Step size is not part of `NUTSDiagnostics`; pass the sampler's active step size
to summarize it, or leave it `NaN` (the default) or `missing` to record the
transition with an unavailable step size (never folded). A finite non-positive
or infinite step size is rejected as invalid telemetry rather than treated as
unavailable.
"""
function record_transition(
    accumulator::HMCDiagnosticsAccumulator{T},
    diagnostics::NUTSDiagnostics,
    stepsize::Union{Real,Missing} = T(NaN),
) where {T<:AbstractFloat}
    diagnostics.depth >= 0 ||
        throw(DomainError(diagnostics.depth, "tree depth must be non-negative"))
    diagnostics.n_steps >= 0 ||
        throw(DomainError(diagnostics.n_steps,
                          "leapfrog-step count must be non-negative"))

    acceptance = convert(T, diagnostics.acceptance_rate)
    isfinite(acceptance) && zero(T) <= acceptance <= one(T) ||
        throw(DomainError(diagnostics.acceptance_rate,
                          "acceptance rate must be finite and in [0, 1]"))

    # Energy error: the canonical sampler normalizes any non-finite error to
    # `-Inf` AND marks that transition divergent. A divergent transition's
    # non-finite error is therefore captured by the divergence count and not
    # folded; a non-finite error on a NON-divergent transition is invalid
    # telemetry and is rejected.
    energy = convert(T, diagnostics.energy_error)
    next_energy_error = if isfinite(energy)
        update(accumulator.energy_error, energy)
    else
        diagnostics.diverged ||
            throw(DomainError(diagnostics.energy_error,
                "a non-finite energy error must be marked divergent"))
        accumulator.energy_error
    end

    # Step size is optional and lives outside `NUTSDiagnostics`. `NaN` or
    # `missing` records the transition with an UNAVAILABLE step size (never
    # folded); any other invalid value (finite non-positive, or infinite) is
    # rejected rather than silently treated as unavailable.
    next_stepsize = if ismissing(stepsize)
        accumulator.stepsize
    else
        epsilon = convert(T, stepsize)
        if isnan(epsilon)
            accumulator.stepsize
        elseif isfinite(epsilon) && epsilon > zero(T)
            update(accumulator.stepsize, epsilon)
        else
            throw(DomainError(stepsize,
                "step size must be NaN/missing (unavailable) or finite and positive"))
        end
    end

    divergences = diagnostics.diverged ?
        Base.checked_add(accumulator.n_divergent, 1) : accumulator.n_divergent

    HMCDiagnosticsAccumulator{T}(
        divergences,
        max(accumulator.max_depth, Int(diagnostics.depth)),
        update(accumulator.depth, convert(T, diagnostics.depth)),
        update(accumulator.leapfrog_steps, convert(T, diagnostics.n_steps)),
        update(accumulator.acceptance_rate, convert(T, diagnostics.acceptance_rate)),
        next_energy_error,
        next_stepsize,
    )
end

_transition_diagnostics(diagnostics::NUTSDiagnostics) = diagnostics
_transition_diagnostics(transition) = transition.diagnostics
_transition_stepsize(::NUTSDiagnostics, ::Type{T}) where {T} = T(NaN)
function _transition_stepsize(transition, ::Type{T}) where {T}
    hasproperty(transition, :stepsize) || return T(NaN)
    stepsize = transition.stepsize
    # `missing` is a first-class unavailable form; only a present value converts.
    stepsize === missing ? missing : convert(T, stepsize)
end

"""
    fit_diagnostics(accumulator, transitions)
    fit_diagnostics(T, transitions)
    fit_diagnostics(transitions)

Fold an iterable of transitions into a mergeable diagnostics state. Each element
is either a [`NUTSDiagnostics`](@ref) (recorded with an unavailable step size) or
a named tuple carrying a `diagnostics` field and an optional `stepsize` (a
value, `missing`, or omitted — the latter two record an unavailable step size).
"""
function fit_diagnostics(accumulator::HMCDiagnosticsAccumulator{T},
                         transitions) where {T}
    for transition in transitions
        accumulator = record_transition(
            accumulator,
            _transition_diagnostics(transition),
            _transition_stepsize(transition, T),
        )
    end
    accumulator
end

fit_diagnostics(::Type{T}, transitions) where {T<:AbstractFloat} =
    fit_diagnostics(HMCDiagnosticsAccumulator(T), transitions)
fit_diagnostics(transitions) = fit_diagnostics(Float64, transitions)

"Combine independently summarized HMC transition partitions."
function Base.merge(a::HMCDiagnosticsAccumulator{T},
                    b::HMCDiagnosticsAccumulator{T}) where {T<:AbstractFloat}
    sample_count(a) == 0 && return b
    sample_count(b) == 0 && return a
    HMCDiagnosticsAccumulator{T}(
        Base.checked_add(a.n_divergent, b.n_divergent),
        max(a.max_depth, b.max_depth),
        merge(a.depth, b.depth),
        merge(a.leapfrog_steps, b.leapfrog_steps),
        merge(a.acceptance_rate, b.acceptance_rate),
        merge(a.energy_error, b.energy_error),
        merge(a.stepsize, b.stepsize),
    )
end

"""
    build_online_stats_graph(T=Float64)

Build declarative scalar-moment, partition-merge, and HMC-diagnostics kernels
over immutable accumulator states, then compose them by named ports. The returned
[`KernelSpec`](@ref) keeps the update path as its default boundary while making
every port available for explicit `have`/`want` queries.
"""
function build_online_stats_graph(::Type{T}=Float64) where {T<:AbstractFloat}
    @kernel updates(state::MomentsAccumulator{T}, observation::T) = begin
        updated::MomentsAccumulator{T} = update(state, observation)
        average::T = mean(updated)
        sample_variance::T = var(updated)
        return updated, average, sample_variance
    end

    @kernel partitions(left_partition::MomentsAccumulator{T},
                       right_partition::MomentsAccumulator{T}) = begin
        merged::MomentsAccumulator{T} = merge(left_partition, right_partition)
        merged_average::T = mean(merged)
        merged_variance::T = var(merged)
        return merged, merged_average, merged_variance
    end

    diagnostics = @kernel begin
        diagnostics_state::HMCDiagnosticsAccumulator{T}
        transition::NUTSDiagnostics{T}
        stepsize_observation::T
        updated_diagnostics::HMCDiagnosticsAccumulator{T} = record_transition(
            diagnostics_state,
            transition,
            stepsize_observation,
        )
        diagnostic_sample_count::Int = sample_count(updated_diagnostics)
        diagnostic_max_tree_depth::Int = max_tree_depth(updated_diagnostics)
        diagnostic_divergence_percent::T = divergence_percent(updated_diagnostics)
        diagnostic_mean_acceptance_rate::T = mean_acceptance_rate(updated_diagnostics)
        diagnostic_mean_tree_depth::T = mean_tree_depth(updated_diagnostics)
        diagnostic_mean_leapfrog_steps::T = mean_leapfrog_steps(updated_diagnostics)
        diagnostic_mean_energy_error::T = mean_energy_error(updated_diagnostics)
        diagnostic_mean_stepsize::T = mean_stepsize(updated_diagnostics)
        return updated_diagnostics, diagnostic_sample_count,
               diagnostic_max_tree_depth, diagnostic_divergence_percent,
               diagnostic_mean_acceptance_rate, diagnostic_mean_tree_depth,
               diagnostic_mean_leapfrog_steps, diagnostic_mean_energy_error,
               diagnostic_mean_stepsize
    end

    merge(merge(updates, partitions), diagnostics)
end

@noinline function _run_kernel_updates(kernel, accumulator, iterations::Int)
    one_value = one(typeof(accumulator.mean))
    for i in 1:iterations
        observation = isodd(i) ? -one_value : one_value
        accumulator = kernel(accumulator, observation)
    end
    accumulator
end

"""
    kernel_performance_report(kernel, seed=MomentsAccumulator(); iterations=100_000)

Warm the generated update kernel, then measure steady-state allocated bytes and
elapsed time in separate runs.
"""
function kernel_performance_report(kernel,
                                   seed::MomentsAccumulator=MomentsAccumulator();
                                   iterations::Int=100_000)
    iterations > 0 || throw(ArgumentError("iterations must be positive"))
    _run_kernel_updates(kernel, seed, 1)
    allocated_bytes = @allocated _run_kernel_updates(kernel, seed, iterations)
    started = time_ns()
    result = _run_kernel_updates(kernel, seed, iterations)
    elapsed_ns = Int(time_ns() - started)
    (; iterations, allocated_bytes, elapsed_ns,
       nanoseconds_per_update=elapsed_ns / iterations, result)
end

@noinline function _run_diagnostics_updates(kernel, accumulator,
                                            iterations::Int)
    T = typeof(accumulator.acceptance_rate.mean)
    epsilon = T(0.2)
    for i in 1:iterations
        transition = NUTSDiagnostics(3, 7, T(0.9), iszero(i % 31), T(-0.05))
        accumulator = kernel(accumulator, transition, epsilon)
    end
    accumulator
end

"""
    diagnostics_performance_report(kernel,
        seed=HMCDiagnosticsAccumulator(); iterations=100_000)

Warm the generated HMC diagnostics kernel, then measure steady-state allocated
bytes and elapsed time in separate runs.
"""
function diagnostics_performance_report(
    kernel,
    seed::HMCDiagnosticsAccumulator=HMCDiagnosticsAccumulator();
    iterations::Int=100_000,
)
    iterations > 0 || throw(ArgumentError("iterations must be positive"))
    _run_diagnostics_updates(kernel, seed, 1)
    allocated_bytes = @allocated _run_diagnostics_updates(
        kernel, seed, iterations)
    started = time_ns()
    result = _run_diagnostics_updates(kernel, seed, iterations)
    elapsed_ns = Int(time_ns() - started)
    (; iterations, allocated_bytes, elapsed_ns,
       nanoseconds_per_update=elapsed_ns / iterations, result)
end

@noinline function _run_reactive_updates!(state, model, accumulator,
                                          iterations::Int)
    one_value = one(typeof(accumulator.mean))
    for i in 1:iterations
        observation = isodd(i) ? -one_value : one_value
        set!(state, model.state, accumulator)
        set!(state, model.observation, observation)
        accumulator = get!(state, model.updated)
    end
    accumulator
end

"""
    reactive_performance_report(model; iterations=1_000)

Warm a `ReactiveState`, then report steady-state orchestration allocations and
elapsed time separately. Unlike the generated kernel, this path intentionally
includes source versioning, validity checks, planning-cache lookup, and result
materialization.
"""
function reactive_performance_report(model; iterations::Int=1_000)
    iterations > 0 || throw(ArgumentError("iterations must be positive"))
    seed = MomentsAccumulator(ReactiveKernels.valtype(model.observation))
    state = ReactiveState(model; materialize=(model.updated,))
    _run_reactive_updates!(state, model, seed, 1)
    allocated_bytes = @allocated _run_reactive_updates!(
        state, model, seed, iterations)
    started = time_ns()
    result = _run_reactive_updates!(state, model, seed, iterations)
    elapsed_ns = Int(time_ns() - started)
    (; iterations, allocated_bytes, elapsed_ns,
       nanoseconds_per_update=elapsed_ns / iterations, result)
end

# Deterministic parameter-vector draws for the metric-adaptation showcase, so the
# reactive Welford estimate is reproducible in the docs build and tests.
const METRIC_ADAPTATION_DRAWS = [
    [1.0, -2.0, 0.5],
    [1.5, -1.0, 0.0],
    [0.5, -3.0, 1.0],
    [2.0, -2.5, -0.5],
]

"""
    metric_adaptation_report(observations=METRIC_ADAPTATION_DRAWS)

Showcase the canonical reactive [`welford_var`](@ref)/[`WelfordVariance`](@ref)
estimator — the sampler's own in-place diagonal-metric adaptation statistic —
by folding parameter-vector `observations` and reading back its online
componentwise mean and variance. This deliberately reuses the exported canonical
estimator rather than re-implementing a vector Welford: it is the reactive,
in-place, history-dependent counterpart to the immutable, mergeable
[`MomentsAccumulator`](@ref) used for the scalar summaries above, and is *not*
itself mergeable across partitions.
"""
function metric_adaptation_report(observations=METRIC_ADAPTATION_DRAWS)
    isempty(observations) &&
        throw(ArgumentError("need at least one observation vector"))
    dimension = length(first(observations))
    dimension > 0 ||
        throw(ArgumentError("observation vectors must have at least one component"))
    all(observation -> length(observation) == dimension, observations) ||
        throw(DimensionMismatch(
            "all observation vectors must share the same dimension"))
    estimate = welford_var(dimension)
    for value in observations
        step!(estimate, value)
    end
    (; dimension, count=estimate.n,
       mean=copy(estimate.mean), variance=copy(estimate.var))
end

function demo()
    model = build_online_stats_graph()
    update_plan = plan(model;
        have=(:state, :observation),
        want=(:updated, :average, :sample_variance))
    update_kernel = prepare(update_plan)

    data = [1.0, 2.0, 4.0, 8.0]
    streaming = fit(data)
    partitions = (fit(data[1:1]), fit(data[2:3]), fit(data[4:4]))
    merged = reduce(merge, partitions)
    # Canonical per-transition diagnostics: NUTSDiagnostics(depth, n_steps,
    # acceptance_rate, diverged, energy_error). Step size is adapted separately,
    # so it is paired in explicitly and left NaN when unavailable.
    transitions = [
        (diagnostics=NUTSDiagnostics(3, 7, 0.92, false, -0.05), stepsize=0.25),
        (diagnostics=NUTSDiagnostics(5, 15, 0.81, true, -6.0), stepsize=0.20),
        (diagnostics=NUTSDiagnostics(2, 5, 0.96, false, -0.01),),  # step size unavailable
    ]
    diagnostics = fit_diagnostics(transitions)
    adaptation = metric_adaptation_report()

    println("streaming state = ", streaming)
    println("merged state    = ", merged)
    println("mean = ", mean(streaming), ", sample variance = ", var(streaming))
    println("generated update = ", update_kernel(MomentsAccumulator(), 3.0))
    println("generated source:\n", code_expr(update_kernel))
    println("colored DAG (DOT interchange):\n", dot_source(update_plan))
    println("HMC diagnostics = ", diagnostics)
    println("divergences = ", diagnostics.n_divergent, "/",
            sample_count(diagnostics), " (", divergence_percent(diagnostics), "%)")
    println("max tree depth = ", max_tree_depth(diagnostics),
            ", mean energy error = ", mean_energy_error(diagnostics),
            ", mean step size = ", mean_stepsize(diagnostics))
    println("metric adaptation (canonical welford_var) variance = ",
            adaptation.variance)
    println("kernel performance = ", kernel_performance_report(
        prepare(model; have=(:state, :observation), want=:updated)))
    println("reactive performance = ", reactive_performance_report(model))
    nothing
end

end # module OnlineStatsExample

if abspath(PROGRAM_FILE) == @__FILE__
    OnlineStatsExample.demo()
end
