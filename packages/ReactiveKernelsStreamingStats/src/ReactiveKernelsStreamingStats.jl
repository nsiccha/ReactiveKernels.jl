module ReactiveKernelsStreamingStats

using ReactiveKernels
using Statistics

export MomentsAccumulator, update, fit
export OnlineMoments, online_moments, update!, fit!, snapshot, reset!
export welford_var, step!

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

end # module ReactiveKernelsStreamingStats
