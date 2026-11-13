# Integer observations need no conversion to the machine integer range.
@traceable _integer_observation(x::Integer) = true
@traceable _integer_observation(x::Real) = isfinite(x) && x == floor(x)

# The scalar isapprox check used by Distributions.isprobvec, with its default
# relative tolerance and zero absolute tolerance. Retain the ordinary fallback
# for other Real types; floating sums have an authored body for JAX emission.
_categorical_sum_is_one(total::Real, unit::Real) = isapprox(total, unit)
@traceable _categorical_sum_is_one(total::Integer, unit::Integer) = total == unit
@traceable function _categorical_sum_is_one(total::T, unit::U) where {T<:AbstractFloat,U<:AbstractFloat}
    rtol = max(sqrt(eps(T)), sqrt(eps(U)))
    return total == unit || (isfinite(total) &&
        abs(total - unit) <= rtol * max(abs(total), abs(unit)))
end

# Constructor checks shared by discrete graph adapters and PPL sampling.
@traceable function _checked_probability(p::Real)
    (0 <= p) && (p <= 1) || throw(DomainError(p, "probability must be between zero and one"))
    return Float64(p)
end

@traceable function _checked_trials(n::Integer)
    n >= 0 || throw(DomainError(n, "number of trials must be nonnegative"))
    return Int(n)
end

@traceable function _checked_categorical_probabilities(probabilities::AbstractVector)
    all(p >= zero(p) for p in probabilities) &&
        _categorical_sum_is_one(sum(probabilities), one(eltype(probabilities))) ||
        throw(DomainError(probabilities,
            "categorical probabilities must be nonnegative and sum to one"))
    return probabilities
end

@traceable function _checked_categorical_probabilities(n::Integer)
    n >= 1 || throw(DomainError(n, "at least one category is required"))
    return fill(1.0 / n, n)
end

# A scalar Real argument is the one-element probability-vector constructor;
# the Integer method above retains Categorical(number_of_categories).
@traceable _checked_categorical_probabilities(p::Real) = _checked_categorical_probabilities([p])

