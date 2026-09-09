module AuthoredScanFixtures

using ReactiveKernels

@kernel authored_scan_arma(q::Vector{Float64}, series::Vector{Float64}) = begin
    mu::Float64 = sum(view(q, 1:1))
    phi::Float64 = sum(view(q, 2:2))
    theta::Float64 = sum(view(q, 3:3))
    errors::Vector{Float64} = scan(series, Ref(mu), Ref(phi), Ref(theta);
            init = (; previous = mu, error = 0.0)) do carry, y, m, f, t
        e = y - (m + f * carry.previous + t * carry.error)
        ((; previous = y, error = e), e)
    end
    pointwise = plate(errors) do e
        -0.5 * e^2
    end
    total::Float64 = sum(pointwise)
    energy::Float64 = sum(abs2, errors)
    joint::Float64 = total + energy
    return total
end

function _authored_scan_reference(q, series)
    previous, error = q[1], 0.0
    errors = similar(series)
    for i in eachindex(series)
        error = series[i] - (q[1] + q[2] * previous + q[3] * error)
        previous = series[i]
        errors[i] = error
    end
    errors
end

# A first-order linear recurrence carry_i = a[i] * carry_{i-1} + b[i] over TWO
# co-varying per-step sequences advanced in lockstep — the shape the single-`xs`
# scan could not author (each per-step operand had to be a broadcast-invariant
# `Ref`).  It feeds a plate + sum exactly like `authored_scan_arma`, so the same
# fused/materialized/Reactant lowering paths exercise the multi-sequence step.
@kernel authored_scan_lockstep(a::Vector{Float64}, b::Vector{Float64}) = begin
    seq::Vector{Float64} = scan(a, b; init = 0.0) do carry, ai, bi
        next = ai * carry + bi
        (next, next)
    end
    pointwise = plate(seq) do s
        -0.5 * s^2
    end
    total::Float64 = sum(pointwise)
    return total
end

function _authored_scan_lockstep_reference(a, b)
    seq = similar(a)
    carry = 0.0
    for i in eachindex(a, b)
        carry = a[i] * carry + b[i]
        seq[i] = carry
    end
    seq
end

end
