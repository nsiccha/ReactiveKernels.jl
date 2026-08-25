module OnlineStatsExample

using ReactiveKernels
using Statistics

export MomentsAccumulator, update, fit
export build_online_stats_graph
export kernel_performance_report, reactive_performance_report
export demo

"""
    MomentsAccumulator(T=Float64)
    MomentsAccumulator{T}(n, mean, m2)

Immutable sufficient statistics for the mean and variance of a scalar stream.
`T` must be a floating-point type so the empty and corrected-singleton results
can be represented as `NaN` without changing the return type.

Finite states maintain `m2 >= 0`. The update and merge algorithms clamp only a
negative value within a small floating-point roundoff tolerance back to zero;
a larger negative value is an error.
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
        isfinite(second) && second < zero(T) &&
            throw(ArgumentError("finite m2 must be non-negative"))
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
    (!isfinite(value) || value >= zero(T)) && return value
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
    delta = x - accumulator.mean
    next_mean = accumulator.mean + delta / T(n)
    increment = delta * (x - next_mean)
    next_m2 = _nonnegative_m2(accumulator.m2 + increment,
                              abs(accumulator.m2) + abs(increment))
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
storage type, keeping the return type statically known.
"""
function Base.merge(a::MomentsAccumulator{T},
                    b::MomentsAccumulator{T}) where {T<:AbstractFloat}
    a.n == 0 && return b
    b.n == 0 && return a

    n = Base.checked_add(a.n, b.n)
    delta = b.mean - a.mean
    b_weight = T(b.n) / T(n)
    next_mean = a.mean + delta * b_weight
    cross = delta * delta * (T(a.n) * T(b.n) / T(n))
    next_m2 = _nonnegative_m2(a.m2 + b.m2 + cross,
                              abs(a.m2) + abs(b.m2) + abs(cross))
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
    accumulator.m2 / T(denominator)
end

"""
    build_online_stats_graph(T=Float64)

Build update, partition-merge, and summary recipes over immutable accumulator
states. The returned named tuple exposes the graph and every port so consumers
can prepare only the subkernel they need.
"""
function build_online_stats_graph(::Type{T}=Float64) where {T<:AbstractFloat}
    graph = Graph()

    state = value!(graph, :state, MomentsAccumulator{T})
    observation = value!(graph, :observation, T)
    updated = value!(graph, :updated, MomentsAccumulator{T})
    average = value!(graph, :mean, T)
    sample_variance = value!(graph, :sample_variance, T)

    left_partition = value!(graph, :left_partition, MomentsAccumulator{T})
    right_partition = value!(graph, :right_partition, MomentsAccumulator{T})
    merged = value!(graph, :merged, MomentsAccumulator{T})
    merged_average = value!(graph, :merged_mean, T)
    merged_variance = value!(graph, :merged_variance, T)

    add!(graph, (state, observation) => updated, update)
    add!(graph, updated => average, mean)
    add!(graph, updated => sample_variance, var)

    add!(graph, (left_partition, right_partition) => merged, merge)
    add!(graph, merged => merged_average, mean)
    add!(graph, merged => merged_variance, var)

    (; graph, state, observation, updated, average, sample_variance,
       left_partition, right_partition, merged, merged_average,
       merged_variance)
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
    state = ReactiveState(model.graph; materialize=(model.updated,))
    _run_reactive_updates!(state, model, seed, 1)
    allocated_bytes = @allocated _run_reactive_updates!(
        state, model, seed, iterations)
    started = time_ns()
    result = _run_reactive_updates!(state, model, seed, iterations)
    elapsed_ns = Int(time_ns() - started)
    (; iterations, allocated_bytes, elapsed_ns,
       nanoseconds_per_update=elapsed_ns / iterations, result)
end

function demo()
    model = build_online_stats_graph()
    update_plan = plan(model.graph;
        have=(model.state, model.observation),
        want=(model.updated, model.average, model.sample_variance))
    update_kernel = prepare(update_plan)

    data = [1.0, 2.0, 4.0, 8.0]
    streaming = fit(data)
    partitions = (fit(data[1:1]), fit(data[2:3]), fit(data[4:4]))
    merged = reduce(merge, partitions)

    println("streaming state = ", streaming)
    println("merged state    = ", merged)
    println("mean = ", mean(streaming), ", sample variance = ", var(streaming))
    println("generated update = ", update_kernel(MomentsAccumulator(), 3.0))
    println("generated source:\n", code_expr(update_kernel))
    println("colored DAG (DOT interchange):\n", dot_source(update_plan))
    println("kernel performance = ", kernel_performance_report(
        prepare(model.graph; have=(model.state, model.observation),
                want=(model.updated,))))
    println("reactive performance = ", reactive_performance_report(model))
    nothing
end

end # module OnlineStatsExample

if abspath(PROGRAM_FILE) == @__FILE__
    OnlineStatsExample.demo()
end
