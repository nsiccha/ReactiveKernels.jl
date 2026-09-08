module InnerPlatePartialEvaluation

using ReactiveKernels

# Instrumentation is a semantic control only; timing uses the pure fixtures.
const calls = Ref(0)
counted_log(x) = (calls[] += 1; log(x))

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

end
