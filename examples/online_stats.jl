module OnlineStatsExample

using ReactiveKernels
using Statistics

# NUTS diagnostics are part of the external sampler exemplar, not RK's package API.
isdefined(Main, :ReactiveKernelsNUTSExample) ||
    Base.include(Main, joinpath(@__DIR__, "nuts_runtime.jl"))
using Main.ReactiveKernelsNUTSExample: NUTSDiagnostics

export MomentsAccumulator, update, fit
export OnlineMoments, online_moments, update!, fit!, snapshot, reset!
export welford_var, step!
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

# -- BEGIN DOCS: ReactiveHMC Welford @kernel --
# Algorithm-structure authority: ReactiveHMC.jl
# ca9ea4ca41924bb0e1fadc01c717e1333916aba6/src/adaptation.jl:47-59.
# `smooth(old, new, w)` is expanded as `(1-w)*old + w*new` so the captured
# MethodIR contains only the compiler's supported Base arithmetic primitives.
@kernel welford_var(template::AbstractVector) = begin
    n = zero(eltype(template))
    mean = zero(template)
    var = zero(template)
    step!(x::AbstractVector; dn = one(n)) = begin
        n += dn
        w = dn / n
        @. var = (one(w) - w) * var +
            w * (x - ((one(w) - w) * mean + w * x)) * (x - mean)
        @. mean = (one(w) - w) * mean + w * x
    end
    step!(x::AbstractMatrix; kwargs...) = for xi in eachcol(x)
        step!(__self__, xi; kwargs...)
    end
end
# -- END DOCS: ReactiveHMC Welford @kernel --

"Compiled state for the ReactiveHMC-shaped method-bearing `welford_var` kernel."
mutable struct OnlineMoments{K,S}
    kernel::K
    state::S
end

"""
    online_moments(template)
    online_moments(dimension, T=Float64)

Compile and instantiate the method-bearing [`welford_var`](@ref) kernel. The
running `n`, componentwise `mean`, and population `var` are compiler-owned state;
[`step!`](@ref) executes the authored vector or matrix overload in place.
"""
function online_moments(template::AbstractVector)
    seed = zero(template)
    kernel = compile_stateful(welford_var, seed)
    OnlineMoments(kernel, kernel(seed))
end

online_moments(dimension::Integer=1, ::Type{T}=Float64) where {T<:AbstractFloat} =
    (dimension > 0 || throw(ArgumentError("dimension must be positive"));
     online_moments(zeros(T, dimension)))

@inline function Base.getproperty(statistics::OnlineMoments, name::Symbol)
    name in (:kernel, :state) && return getfield(statistics, name)
    ReactiveKernels.stateful_get(getfield(statistics, :state), Val(name))
end

Base.propertynames(::OnlineMoments, private::Bool=false) =
    private ? (:kernel, :state, :n, :mean, :var) : (:n, :mean, :var)

function Base.copy(statistics::OnlineMoments)
    OnlineMoments(getfield(statistics, :kernel), deepcopy(getfield(statistics, :state)))
end

function step!(statistics::OnlineMoments, x::AbstractVector; dn=one(statistics.n))
    ReactiveKernels.stateful_call!(statistics.state, Val(:step!), x; dn)
    statistics
end

function step!(statistics::OnlineMoments, x::AbstractMatrix; kwargs...)
    ReactiveKernels.stateful_call!(statistics.state, Val(:step!), x; kwargs...)
    statistics
end

update!(statistics::OnlineMoments, observation::Real) =
    step!(statistics, [convert(eltype(statistics.mean), observation)])

function fit!(statistics::OnlineMoments, observations)
    for observation in observations
        observation isa AbstractVector ? step!(statistics, observation) :
            update!(statistics, observation)
    end
    statistics
end

snapshot(statistics::OnlineMoments) =
    (n=statistics.n, mean=copy(statistics.mean), var=copy(statistics.var))

function reset!(statistics::OnlineMoments)
    seed = zero(statistics.mean)
    statistics.state = statistics.kernel(seed)
    statistics
end

Statistics.mean(statistics::OnlineMoments) = statistics.mean
function Statistics.var(statistics::OnlineMoments; corrected::Bool=true)
    corrected || return statistics.var
    statistics.n > one(statistics.n) || return fill(eltype(statistics.var)(NaN), length(statistics.var))
    statistics.var .* (statistics.n / (statistics.n - one(statistics.n)))
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
in-place diagonal-metric variance the sampler adapts, use this page's compiled,
method-bearing [`welford_var`](@ref) kernel (see
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

"Mutable convenience wrapper around a mergeable diagnostics accumulator."
mutable struct OnlineDiagnostics{T<:AbstractFloat}
    accumulator::HMCDiagnosticsAccumulator{T}
end

@inline function Base.getproperty(statistics::OnlineDiagnostics, name::Symbol)
    name === :accumulator && return getfield(statistics, :accumulator)
    getproperty(getfield(statistics, :accumulator), name)
end

Base.propertynames(statistics::OnlineDiagnostics, private::Bool=false) =
    private ? (:accumulator, propertynames(statistics.accumulator)... ) :
              propertynames(statistics.accumulator)

"""
    online_diagnostics(T=Float64)

Construct a mutable convenience recorder around the immutable, mergeable
[`HMCDiagnosticsAccumulator`](@ref). This is deliberately separate from the
ReactiveHMC-shaped method-bearing Welford kernel above.
"""
online_diagnostics(::Type{T}=Float64) where {T<:AbstractFloat} =
    OnlineDiagnostics(HMCDiagnosticsAccumulator(T))

function record!(statistics::OnlineDiagnostics, diagnostics::NUTSDiagnostics,
                 stepsize::Union{Real,Missing}=NaN)
    statistics.accumulator = record_transition(statistics.accumulator, diagnostics, stepsize)
    statistics
end

function fit!(statistics::OnlineDiagnostics, transitions)
    for transition in transitions
        record!(statistics, _transition_diagnostics(transition),
                _transition_stepsize(transition, typeof(statistics.acceptance_rate.mean)))
    end
    statistics
end

snapshot(statistics::OnlineDiagnostics) = statistics.accumulator
Base.copy(statistics::OnlineDiagnostics) = OnlineDiagnostics(statistics.accumulator)
function reset!(statistics::OnlineDiagnostics{T}) where {T}
    statistics.accumulator = HMCDiagnosticsAccumulator(T)
    statistics
end

sample_count(statistics::OnlineDiagnostics) = sample_count(statistics.accumulator)
max_tree_depth(statistics::OnlineDiagnostics) = max_tree_depth(statistics.accumulator)
divergence_percent(statistics::OnlineDiagnostics) = divergence_percent(statistics.accumulator)
divergence_rate(statistics::OnlineDiagnostics) = divergence_rate(statistics.accumulator)
mean_tree_depth(statistics::OnlineDiagnostics) = mean_tree_depth(statistics.accumulator)
mean_leapfrog_steps(statistics::OnlineDiagnostics) =
    mean_leapfrog_steps(statistics.accumulator)
mean_acceptance_rate(statistics::OnlineDiagnostics) =
    mean_acceptance_rate(statistics.accumulator)
mean_energy_error(statistics::OnlineDiagnostics) = mean_energy_error(statistics.accumulator)
mean_stepsize(statistics::OnlineDiagnostics) = mean_stepsize(statistics.accumulator)

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
    one_value = one(eltype(statistics.mean))
    observation = similar(statistics.mean)
    for i in 1:iterations
        fill!(observation, isodd(i) ? -one_value : one_value)
        step!(statistics, observation)
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

Showcase the exact ReactiveHMC-shaped method-bearing [`welford_var`](@ref)
kernel by folding parameter-vector `observations` and reading back its online
componentwise mean and variance. It is the compiler-backed, in-place,
history-dependent counterpart to the immutable, mergeable
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
    T = float(eltype(first(observations)))
    estimate = online_moments(dimension, T)
    for value in observations
        step!(estimate, T.(value))
    end
    (; dimension, count=estimate.n,
       mean=copy(estimate.mean), variance=copy(estimate.var))
end

function demo()
    data = [1.0, 2.0, 4.0, 8.0]
    statistics = online_moments(1)
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
    println("mean = ", only(mean(statistics)),
            ", sample variance = ", only(var(statistics)))
    println("stateful moments plan:\n", explain(statistics.kernel.prepared.plan))
    println("partition merge = ", partition_kernel(partitions[1], partitions[2]))
    println("partition DAG (DOT interchange):\n", dot_source(partition_plan))
    println("HMC diagnostics = ", diagnostics_snapshot)
    println("divergences = ", diagnostics.n_divergent, "/",
            sample_count(diagnostics), " (", divergence_percent(diagnostics), "%)")
    println("max tree depth = ", max_tree_depth(diagnostics),
            ", mean energy error = ", mean_energy_error(diagnostics),
            ", mean step size = ", mean_stepsize(diagnostics))
    println("metric adaptation (compiled @kernel welford_var) variance = ",
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
