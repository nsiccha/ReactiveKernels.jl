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
    build_online_stats_graph(T=Float64)

Build declarative update/summary and partition-merge kernels over immutable
accumulator states, then compose them by named ports. The returned
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

    merge(updates, partitions)
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

    println("streaming state = ", streaming)
    println("merged state    = ", merged)
    println("mean = ", mean(streaming), ", sample variance = ", var(streaming))
    println("generated update = ", update_kernel(MomentsAccumulator(), 3.0))
    println("generated source:\n", code_expr(update_kernel))
    println("colored DAG (DOT interchange):\n", dot_source(update_plan))
    println("kernel performance = ", kernel_performance_report(
        prepare(model; have=(:state, :observation), want=:updated)))
    println("reactive performance = ", reactive_performance_report(model))
    nothing
end

end # module OnlineStatsExample

if abspath(PROGRAM_FILE) == @__FILE__
    OnlineStatsExample.demo()
end
