module OnlineStatsExample

using ReactiveKernels
using Statistics

# NUTS diagnostics are part of the external sampler exemplar, not RK's package API.
isdefined(Main, :ReactiveKernelsNUTSExample) ||
    Base.include(Main, joinpath(@__DIR__, "nuts_runtime.jl"))
using Main.ReactiveKernelsNUTSExample: NUTSDiagnostics, welford_var, step!

export MomentsAccumulator, update, fit
export OnlineMoments, online_moments, update!, fit!, snapshot, reset!
export HMCDiagnosticsAccumulator, record_transition, fit_diagnostics
export OnlineDiagnostics, online_diagnostics, record!
export sample_count, max_tree_depth, divergence_rate, divergence_percent
export mean_tree_depth, mean_leapfrog_steps, mean_acceptance_rate
export mean_energy_error, mean_stepsize
export metric_adaptation_report
export build_partition_graph
export partition_performance_report, diagnostics_performance_report
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

_moment_average(n::Int, mean::T) where {T<:AbstractFloat} =
    n == 0 ? T(NaN) : mean

function _moment_variance(n::Int, m2::T, corrected::Bool) where {T<:AbstractFloat}
    denominator = n - Int(corrected)
    denominator > 0 || return T(NaN)
    W = promote_type(T, Float64)
    T(W(m2) / W(denominator))
end

# -- BEGIN DOCS: stateful online moments --
@reactive specialize=true _online_moments_state(n, mean, m2) = begin
    average::typeof(mean) = _moment_average(n, mean)
    sample_variance::typeof(m2) = _moment_variance(n, m2, true)
    population_variance::typeof(m2) = _moment_variance(n, m2, false)

    update!(observation::Real) = begin
        T = typeof(mean)
        x = convert(T, observation)
        old_n = n
        old_mean = mean
        old_m2 = m2

        if old_n == 0
            n = 1
            mean = x
            m2 = zero(T)
        else
            next_n = Base.checked_add(old_n, 1)
            W = promote_type(T, Float64)
            wide_x = W(x)
            wide_mean = W(old_mean)
            delta = wide_x - wide_mean
            next_mean = wide_mean + delta / W(next_n)
            increment = delta * (wide_x - next_mean)
            wide_m2 = W(old_m2)
            next_m2 = _nonnegative_m2(
                wide_m2 + increment,
                abs(wide_m2) + abs(increment),
            )
            n = next_n
            mean = T(next_mean)
            m2 = T(next_m2)
        end
        __self__
    end

    fit!(observations) = begin
        for observation in observations
            update!(observation)
        end
        __self__
    end

    snapshot() = MomentsAccumulator{typeof(mean)}(n, mean, m2)

    reset!() = begin
        n = zero(n)
        mean = zero(mean)
        m2 = zero(m2)
        __self__
    end
end
# -- END DOCS: stateful online moments --

"A stateful scalar-moments object authored through ReactiveKernels' `@reactive`."
const OnlineMoments = ReactiveObject{:_online_moments_state}

"""
    online_moments(T=Float64)

Construct a stateful reactive Welford accumulator. [`update!`](@ref) and
[`fit!`](@ref) mutate its graph-owned `n`, `mean`, and `m2` sources in place;
`average`, `sample_variance`, and `population_variance` are lazily invalidated
derived values. Call [`snapshot`](@ref) only when an immutable, mergeable
[`MomentsAccumulator`](@ref) value is needed at a partition boundary.
"""
online_moments(::Type{T}=Float64) where {T<:AbstractFloat} =
    _online_moments_state(0, zero(T), zero(T))

Statistics.mean(statistics::OnlineMoments) = statistics.average
function Statistics.var(statistics::OnlineMoments; corrected::Bool=true)
    corrected ? statistics.sample_variance : statistics.population_variance
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

# -- BEGIN DOCS: stateful HMC diagnostics --
@reactive specialize=true _online_diagnostics_state(
        n_divergent, max_depth, depth, leapfrog_steps,
        acceptance_rate, energy_error, stepsize) = begin
    diagnostic_sample_count::Int = depth.n
    diagnostic_max_tree_depth::Int = max_depth
    diagnostic_divergence_percent::typeof(acceptance_rate.mean) =
        _diagnostic_divergence_percent(n_divergent, depth.n, acceptance_rate.mean)
    diagnostic_mean_tree_depth::typeof(depth.mean) = mean(depth)
    diagnostic_mean_leapfrog_steps::typeof(leapfrog_steps.mean) =
        mean(leapfrog_steps)
    diagnostic_mean_acceptance_rate::typeof(acceptance_rate.mean) =
        mean(acceptance_rate)
    diagnostic_mean_energy_error::typeof(energy_error.mean) = mean(energy_error)
    diagnostic_mean_stepsize::typeof(stepsize.mean) = mean(stepsize)

    record!(diagnostics::NUTSDiagnostics,
            stepsize_observation::Union{Real,Missing}=NaN) = begin
        diagnostics.depth >= 0 ||
            throw(DomainError(diagnostics.depth,
                              "tree depth must be non-negative"))
        diagnostics.n_steps >= 0 ||
            throw(DomainError(diagnostics.n_steps,
                              "leapfrog-step count must be non-negative"))

        T = typeof(acceptance_rate.mean)
        acceptance = convert(T, diagnostics.acceptance_rate)
        isfinite(acceptance) && zero(T) <= acceptance <= one(T) ||
            throw(DomainError(diagnostics.acceptance_rate,
                              "acceptance rate must be finite and in [0, 1]"))

        energy = convert(T, diagnostics.energy_error)
        next_energy_error = if isfinite(energy)
            update(energy_error, energy)
        else
            diagnostics.diverged ||
                throw(DomainError(diagnostics.energy_error,
                    "a non-finite energy error must be marked divergent"))
            energy_error
        end

        next_stepsize = if ismissing(stepsize_observation)
            stepsize
        else
            epsilon = convert(T, stepsize_observation)
            if isnan(epsilon)
                stepsize
            elseif isfinite(epsilon) && epsilon > zero(T)
                update(stepsize, epsilon)
            else
                throw(DomainError(stepsize_observation,
                    "step size must be NaN/missing (unavailable) or finite and positive"))
            end
        end

        next_divergences = diagnostics.diverged ?
            Base.checked_add(n_divergent, 1) : n_divergent
        next_max_depth = max(max_depth, Int(diagnostics.depth))
        next_depth = update(depth, convert(T, diagnostics.depth))
        next_leapfrog_steps = update(
            leapfrog_steps, convert(T, diagnostics.n_steps))
        next_acceptance_rate = update(acceptance_rate, acceptance)

        n_divergent = next_divergences
        max_depth = next_max_depth
        depth = next_depth
        leapfrog_steps = next_leapfrog_steps
        acceptance_rate = next_acceptance_rate
        energy_error = next_energy_error
        stepsize = next_stepsize
        __self__
    end

    fit!(transitions) = begin
        for transition in transitions
            record!(
                _transition_diagnostics(transition),
                _transition_stepsize(transition, typeof(acceptance_rate.mean)),
            )
        end
        __self__
    end

    snapshot() = HMCDiagnosticsAccumulator{typeof(acceptance_rate.mean)}(
        n_divergent,
        max_depth,
        depth,
        leapfrog_steps,
        acceptance_rate,
        energy_error,
        stepsize,
    )

    reset!() = begin
        T = typeof(acceptance_rate.mean)
        empty = MomentsAccumulator(T)
        n_divergent = 0
        max_depth = 0
        depth = empty
        leapfrog_steps = empty
        acceptance_rate = empty
        energy_error = empty
        stepsize = empty
        __self__
    end
end
# -- END DOCS: stateful HMC diagnostics --

function _diagnostic_divergence_percent(n_divergent::Int, n::Int, prototype::T) where {T}
    n == 0 && return T(NaN)
    W = promote_type(T, Float64)
    T(W(100) * W(n_divergent) / W(n))
end

"A stateful HMC-diagnostics object authored through ReactiveKernels' `@reactive`."
const OnlineDiagnostics = ReactiveObject{:_online_diagnostics_state}

"""
    online_diagnostics(T=Float64)

Construct a stateful diagnostics recorder. [`record!`](@ref) mutates its
graph-owned counters and sufficient-statistic sources, and the `diagnostic_*`
summary fields recompute lazily. [`snapshot`](@ref) materializes an immutable
[`HMCDiagnosticsAccumulator`](@ref) only for partition merging or persistence.
"""
function online_diagnostics(::Type{T}=Float64) where {T<:AbstractFloat}
    empty = MomentsAccumulator(T)
    _online_diagnostics_state(0, 0, empty, empty, empty, empty, empty)
end

sample_count(statistics::OnlineDiagnostics) = statistics.diagnostic_sample_count
max_tree_depth(statistics::OnlineDiagnostics) =
    statistics.diagnostic_max_tree_depth
divergence_percent(statistics::OnlineDiagnostics) =
    statistics.diagnostic_divergence_percent
function divergence_rate(statistics::OnlineDiagnostics)
    T = typeof(statistics.acceptance_rate.mean)
    statistics.diagnostic_divergence_percent / T(100)
end
mean_tree_depth(statistics::OnlineDiagnostics) =
    statistics.diagnostic_mean_tree_depth
mean_leapfrog_steps(statistics::OnlineDiagnostics) =
    statistics.diagnostic_mean_leapfrog_steps
mean_acceptance_rate(statistics::OnlineDiagnostics) =
    statistics.diagnostic_mean_acceptance_rate
mean_energy_error(statistics::OnlineDiagnostics) =
    statistics.diagnostic_mean_energy_error
mean_stepsize(statistics::OnlineDiagnostics) = statistics.diagnostic_mean_stepsize

"""
    build_partition_graph(T=Float64)

Build the declarative immutable-snapshot merge kernel. Streaming updates belong
to [`online_moments`](@ref) and [`online_diagnostics`](@ref); full-state return
is kept only at this explicit partition boundary, where it is the natural value
semantics for Chan's associative merge.
"""
function build_partition_graph(::Type{T}=Float64) where {T<:AbstractFloat}
    @kernel partitions(left_partition::MomentsAccumulator{T},
                       right_partition::MomentsAccumulator{T}) = begin
        merged::MomentsAccumulator{T} = merge(left_partition, right_partition)
        merged_average::T = mean(merged)
        merged_variance::T = var(merged)
        return merged, merged_average, merged_variance
    end
    partitions
end

@noinline function _run_partition_merges(kernel, left, right, iterations::Int)
    result = left
    for _ in 1:iterations
        result = kernel(result, right)
    end
    result
end

"""
    partition_performance_report(kernel; iterations=100_000)

Warm the generated immutable-snapshot merge kernel, then measure steady-state
allocated bytes and elapsed time in separate runs. This is deliberately the
value-returning partition lane, not the streaming interface.
"""
function partition_performance_report(kernel; iterations::Int=100_000)
    iterations > 0 || throw(ArgumentError("iterations must be positive"))
    left = MomentsAccumulator()
    right = update(MomentsAccumulator(), 1.0)
    _run_partition_merges(kernel, left, right, 1)
    allocated_bytes = @allocated _run_partition_merges(
        kernel, left, right, iterations)
    started = time_ns()
    result = _run_partition_merges(kernel, left, right, iterations)
    elapsed_ns = Int(time_ns() - started)
    (; iterations, allocated_bytes, elapsed_ns,
       nanoseconds_per_merge=elapsed_ns / iterations, result)
end

@noinline function _run_diagnostics_updates!(statistics, iterations::Int)
    T = typeof(statistics.acceptance_rate.mean)
    epsilon = T(0.2)
    for i in 1:iterations
        transition = NUTSDiagnostics(3, 7, T(0.9), iszero(i % 31), T(-0.05))
        record!(statistics, transition, epsilon)
    end
    statistics
end

"""
    diagnostics_performance_report(seed=online_diagnostics(); iterations=100_000)

Warm a stateful diagnostics object, then measure its steady-state in-place
`record!` path. Construction and graph preparation are outside the measurement.
"""
function diagnostics_performance_report(
    seed::OnlineDiagnostics=online_diagnostics();
    iterations::Int=100_000,
)
    iterations > 0 || throw(ArgumentError("iterations must be positive"))
    _run_diagnostics_updates!(copy(seed), 1)
    measured = copy(seed)
    allocated_bytes = @allocated _run_diagnostics_updates!(measured, iterations)
    timed = copy(seed)
    started = time_ns()
    _run_diagnostics_updates!(timed, iterations)
    elapsed_ns = Int(time_ns() - started)
    (; iterations, allocated_bytes, elapsed_ns,
       nanoseconds_per_update=elapsed_ns / iterations, result=snapshot(timed))
end

@noinline function _run_reactive_updates!(statistics, iterations::Int)
    one_value = one(typeof(statistics.mean))
    for i in 1:iterations
        observation = isodd(i) ? -one_value : one_value
        update!(statistics, observation)
    end
    statistics
end

"""
    reactive_performance_report(seed=online_moments(); iterations=1_000)

Warm a stateful moments object, then report steady-state `update!` allocations
and elapsed time separately. Object construction and graph preparation are
outside the measurement; source writes and reactive invalidation are included.
"""
function reactive_performance_report(
    seed::OnlineMoments=online_moments();
    iterations::Int=1_000,
)
    iterations > 0 || throw(ArgumentError("iterations must be positive"))
    _run_reactive_updates!(copy(seed), 1)
    measured = copy(seed)
    allocated_bytes = @allocated _run_reactive_updates!(measured, iterations)
    timed = copy(seed)
    started = time_ns()
    _run_reactive_updates!(timed, iterations)
    elapsed_ns = Int(time_ns() - started)
    (; iterations, allocated_bytes, elapsed_ns,
       nanoseconds_per_update=elapsed_ns / iterations, result=snapshot(timed))
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
    data = [1.0, 2.0, 4.0, 8.0]
    statistics = online_moments()
    fit!(statistics, data)
    streaming = snapshot(statistics)
    partitions = (fit(data[1:1]), fit(data[2:3]), fit(data[4:4]))
    merged = reduce(merge, partitions)
    partition_model = build_partition_graph()
    partition_plan = plan(partition_model;
        have=(:left_partition, :right_partition),
        want=(:merged, :merged_average, :merged_variance))
    partition_kernel = prepare(partition_plan)

    # Canonical per-transition diagnostics: NUTSDiagnostics(depth, n_steps,
    # acceptance_rate, diverged, energy_error). Step size is adapted separately,
    # so it is paired in explicitly and left NaN when unavailable.
    transitions = [
        (diagnostics=NUTSDiagnostics(3, 7, 0.92, false, -0.05), stepsize=0.25),
        (diagnostics=NUTSDiagnostics(5, 15, 0.81, true, -6.0), stepsize=0.20),
        (diagnostics=NUTSDiagnostics(2, 5, 0.96, false, -0.01),),  # step size unavailable
    ]
    diagnostics = online_diagnostics()
    fit!(diagnostics, transitions)
    diagnostics_snapshot = snapshot(diagnostics)
    adaptation = metric_adaptation_report()

    println("streaming state = ", streaming)
    println("merged state    = ", merged)
    println("mean = ", mean(statistics),
            ", sample variance = ", var(statistics))
    println("stateful moments plan:\n",
            explain(reactive_program(statistics).plan))
    println("generated average getter:\n", code_expr(
        reactive_program(statistics), statistics.handles.average))
    println("partition merge = ", partition_kernel(partitions[1], partitions[2]))
    println("partition DAG (DOT interchange):\n", dot_source(partition_plan))
    println("HMC diagnostics = ", diagnostics_snapshot)
    println("divergences = ", diagnostics.n_divergent, "/",
            sample_count(diagnostics), " (", divergence_percent(diagnostics), "%)")
    println("max tree depth = ", max_tree_depth(diagnostics),
            ", mean energy error = ", mean_energy_error(diagnostics),
            ", mean step size = ", mean_stepsize(diagnostics))
    println("metric adaptation (canonical welford_var) variance = ",
            adaptation.variance)
    println("partition performance = ", partition_performance_report(
        prepare(partition_model;
            have=(:left_partition, :right_partition), want=:merged)))
    println("reactive performance = ", reactive_performance_report())
    println("diagnostics performance = ", diagnostics_performance_report())
    nothing
end

end # module OnlineStatsExample

if abspath(PROGRAM_FILE) == @__FILE__
    OnlineStatsExample.demo()
end
