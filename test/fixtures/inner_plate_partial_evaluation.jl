module InnerPlatePartialEvaluation

using ReactiveKernels

# Instrumentation is a semantic control only; timing uses the pure fixtures.
const calls = Ref(0)
counted_log(x) = (calls[] += 1; log(x))
counted_findall(f, x) = (calls[] += 1; findall(f, x))

@kernel unbound_plate(q::Vector{Float64}, data::Vector{Float64}) = begin
    data_sum::Float64 = sum(data)
    pointwise = plate(q) do parameter
        offset::Float64 = 2.0
        result::Float64 = parameter + offset
        result
    end
    total::Float64 = sum(pointwise) + data_sum
end

@kernel counted(q::Float64, data) = begin
    pointwise = plate(data, q) do d, parameter
        transformed::Float64 = counted_log(d)
        result::Float64 = transformed * parameter
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel affine(q::Float64, data, a::Float64, b::Float64) = begin
    pointwise = plate(data, a, b, q) do d, scale, offset, parameter
        transformed::Float64 = scale * d + offset
        result::Float64 = transformed * parameter
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel projected(q, data, shift) = begin
    pointwise = plate(data, shift, q) do d, s, parameter
        transformed::Float64 = counted_log(d)
        shifted::Float64 = transformed + s
        result::Float64 = shifted * parameter
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel projected_scalar(q::Float64, data, shift) = begin
    pointwise = plate(data, shift, q) do d, s, parameter
        transformed::Float64 = counted_log(d)
        result::Float64 = (transformed + s) * parameter
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel projected_matrix(q::AbstractMatrix{Float64}, data, shift) = begin
    pointwise = plate(data, shift, q) do d, s, parameter
        transformed::Float64 = counted_log(d)
        shifted::Float64 = transformed + s
        result::Float64 = shifted * parameter
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel vector_live(q::AbstractVector{Float64}, data) = begin
    pointwise = plate(data, q) do d, parameter
        transformed::Float64 = counted_log(d)
        result::Float64 = transformed * parameter
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel atomic(q::Float64, data, coefficients) = begin
    pointwise = plate(data, Ref(coefficients), q) do d, coefs, parameter
        transformed::Float64 = counted_log(d) + sum(coefs)
        result::Float64 = transformed * parameter
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel chain(q::Float64, data, observations) = begin
    means = plate(data, q) do d, parameter
        transformed::Float64 = counted_log(d)
        result::Float64 = transformed * parameter
        result
    end
    pointwise = plate(means, observations) do mu, y
        (mu - y)^2
    end
    total::Float64 = sum(pointwise)
    extra::Float64 = sum(means)
end

@kernel inline(q::Float64, data) = begin
    pointwise = plate(data, q) do d, parameter
        parameter * counted_log(d)
    end
    total::Float64 = sum(pointwise)
end

@kernel pure(q::Vector{Float64}, data::Vector{Float64}) = begin
    parameter::Float64 = sum(q)
    pointwise = plate(data, parameter) do d, theta
        transformed::Float64 = log(d)
        result::Float64 = transformed * theta
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel ref_live(q::Vector{Float64}, data::Vector{Float64}) = begin
    pointwise = plate(data, Ref(q)) do d, whole
        transformed::Float64 = log(d)
        result::Float64 = transformed * sum(whole)
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel boolean(q::Vector{Float64}, data::Vector{Int}) = begin
    parameter::Float64 = sum(q)
    pointwise = plate(data, parameter) do d, theta
        valid::Bool = d >= 0
        result::Float64 = ifelse(valid, theta * d, 0.0)
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel mutable_result(q::Float64, data) = begin
    pointwise = plate(data, q) do d, parameter
        transformed::Vector{Float64} = fill(d, 2)
        result::Float64 = sum(transformed) * parameter
        result
    end
    total::Float64 = sum(pointwise)
end

@kernel tuple_result(q::Float64, data) = begin
    pointwise = plate(data, q) do d, parameter
        transformed::Tuple{Float64,Float64} = (d, d + 1.0)
        result::Float64 = (transformed[1] + transformed[2]) * parameter
        result
    end
    total::Float64 = sum(pointwise)
end


# Per-cell ragged index lists over a bound subject domain with `Ref` operands:
# the selection is data-only, the final gather reads a live vector.
@kernel ragged_reads(live, kinds_by_subject, read_idx, subjects) = begin
    out = plate(subjects, Ref(kinds_by_subject), Ref(read_idx), Ref(live)) do s, kinds_all, idx_all, live_all
        kinds = kinds_all[s]
        read_positions = counted_findall(isone, kinds)
        observation_operations = read_positions[idx_all[s]]
        observed = live_all[observation_operations]
        sum(observed)
    end
    total = sum(out)
end

@kernel position_reader(kinds) = begin
    read_positions = counted_findall(isone, kinds)
    positions() = read_positions
end

# The same selection authored as a composed child kernel inside the cell.
@kernel composed_reads(live, kinds_by_subject, read_idx, subjects) = begin
    out = plate(subjects, Ref(kinds_by_subject), Ref(read_idx), Ref(live)) do s, kinds_all, idx_all, live_all
        kinds = kinds_all[s]
        read_positions = position_reader(kinds).positions()
        observation_operations = read_positions[idx_all[s]]
        observed = live_all[observation_operations]
        sum(observed)
    end
    total = sum(out)
end

# A lazy branch whose condition reads a cached per-cell array.
@kernel guarded_reads(q::Vector{Float64}, kinds_by_subject, subjects) = begin
    theta::Float64 = sum(q)
    out = plate(subjects, Ref(kinds_by_subject), theta) do s, kinds_all, t
        read_positions = counted_findall(isone, kinds_all[s])
        result = isempty(read_positions) ? 0.0 : t * sum(read_positions)
        result
    end
    total = sum(out)
end

# A floating-point array per cell crosses the cache boundary unreduced.
@kernel array_weights(q::Vector{Float64}, xs) = begin
    parameter::Float64 = sum(q)
    pointwise = plate(xs, parameter) do x, theta
        weights = log.(x)
        result::Float64 = theta * sum(weights)
        result
    end
    total::Float64 = sum(pointwise)
end

end
