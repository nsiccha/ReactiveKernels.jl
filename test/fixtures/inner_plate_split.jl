module InnerPlateSplit

using ReactiveKernels

# Instrumentation is a semantic control only.
const calls = Ref(0)
counted_term(y) = (calls[] += 1; log(y + 1.0))
counted_root(y) = (calls[] += 1; sqrt(y))
counted_sum(y) = (calls[] += 1; sum(log1p, y))

# A data-only call beside a live term in one top-level recipe.
@kernel top_level(q::Vector{Float64}, y) = begin
    total::Float64 = sum(y .* q) - counted_sum(y)
end

# A data-only subexpression inside a cell recipe that also reads a live value.
@kernel mixed(q, y) = begin
    pointwise = plate(y, q) do d, parameter
        d * parameter - exp(parameter) - counted_term(d)
    end
    total::Float64 = sum(pointwise)
end

# The same cell with its live argument's rank declared.
@kernel mixed_typed(q::AbstractVector, y) = begin
    pointwise = plate(y, q) do d, parameter
        d * parameter - exp(parameter) - counted_term(d)
    end
    total::Float64 = sum(pointwise)
end

# A data-bound branch whose taken arm holds the data-only subexpression.
@kernel guarded(q, y) = begin
    pointwise = plate(y, q) do d, parameter
        d > 0 ? counted_term(d) + d * parameter : parameter
    end
    total::Float64 = sum(pointwise)
end

# A branch on a live value: its arm's data-only subexpression is computed for
# every lane at preparation, and the arm reads it where it is taken.
@kernel live_branch(q, y) = begin
    pointwise = plate(y, q) do d, parameter
        parameter > 0 ? counted_term(d) * parameter : parameter
    end
    total::Float64 = sum(pointwise)
end

# A short-circuit operand is not entered either.
@kernel short_circuit(q, y) = begin
    pointwise = plate(y, q) do d, parameter
        parameter * ((parameter > 0) && (counted_term(d) > 1.0))
    end
    total::Float64 = sum(pointwise)
end

# A validity check that reads the live value guards the data-only term, as in
# a count likelihood.
@kernel live_guard(q, y) = begin
    pointwise = plate(y, q) do d, parameter
        (d >= 0) & (parameter > -5) ? d * parameter - counted_term(d) : -Inf
    end
    total::Float64 = sum(pointwise)
end

# A guarded term that throws on a lane whose arm is never taken.
@kernel live_guard_domain(q, y) = begin
    pointwise = plate(y, q) do d, parameter
        (d >= 0) & (parameter > -5) ? parameter * counted_root(d) : 0.0
    end
    total::Float64 = sum(pointwise)
end

# A data-only subexpression that throws on a bound cell an empty live input
# leaves unvisited.
@kernel guarded_domain(q, y) = begin
    pointwise = plate(y, q) do d, parameter
        parameter * counted_root(d)
    end
    total::Float64 = sum(pointwise)
end

# The shape of a count likelihood's normalizer: a data-only difference of
# data-only terms beside the live terms.
@kernel nested(q, y, n) = begin
    pointwise = plate(y, n, q) do d, m, parameter
        counted_term(m) - counted_term(d) - d * log1p(exp(parameter)) +
            (m - d) * parameter
    end
    total::Float64 = sum(pointwise)
end

end
