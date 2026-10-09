module NestedPlates
using ReactiveKernels

# One public source authority, also replayed by acceptance and shown in docs.
const NORMAL_SOURCE = raw"""
@kernel visible_observations(observation_groups, scale::Float64) = begin
    group_logdensity = plate(observation_groups) do observations
        pointwise = plate(observations) do observation
            -0.5 * log(2 * pi) - log(scale) - 0.5 * (observation / scale)^2
        end
        sum(pointwise)
    end
    logdensity = sum(group_logdensity)
    return logdensity
end
"""
Core.eval(@__MODULE__, Meta.parse(NORMAL_SOURCE))

# Complete public source also exercises replay without producer-side bindings.
const ENDPOINT_SOURCE = raw"""
@kernel scalar_normal(y, mu, sigma) = begin
    standardized = (y - mu) / sigma
    density = -0.5 * log(2 * pi) - log(sigma) - 0.5 * standardized^2
    logpdf()::Float64 = density
end
@kernel normal_method(mu, sigma) = begin
    logpdf(y::Float64)::Float64 = begin
        z = (y - mu) / sigma
        -0.5 * log(2 * pi) - log(sigma) - 0.5 * z^2
    end
end
@kernel object_observations(groups, mu::Float64, sigma::Float64) = begin
    grouped = plate(groups, Ref(mu), Ref(sigma)) do observations, location, scale
        pointwise = plate(observations) do value
            scalar_normal(value, location, scale).logpdf()
        end
        sum(pointwise)
    end
    total = sum(grouped)
    return total
end
@kernel method_observations(groups, mu::Float64, sigma::Float64) = begin
    grouped = plate(groups, Ref(mu), Ref(sigma)) do observations, location, scale
        sum(plate(observations, Ref(location), Ref(scale)) do value, location, scale
            normal_method(; mu=location, sigma=scale).logpdf(value)
        end)
    end
    total = sum(grouped)
    return total
end
@kernel computed_observations(groups, mu::Float64, sigma::Float64) = begin
    grouped = plate(groups, Ref(mu), Ref(sigma)) do observations, location, scale
        shifted = location + 0.25
        pointwise = plate(observations, Ref(shifted)) do value, shifted
            centered = value - shifted
            normal_method(0.0, scale).logpdf(centered + 0.25) + 0.0
        end
        sum(pointwise)
    end
    total = sum(grouped)
    return total
end
@kernel guarded_observations(groups, mu::Float64, sigma::Float64) = begin
    grouped = plate(groups, Ref(mu), Ref(sigma)) do observations, location, scale
        pointwise = plate(observations) do value
            value >= 0 ? normal_method(location, scale).logpdf(value) :
                         normal_method(location, -scale).logpdf(value)
        end
        sum(pointwise)
    end
    total = sum(grouped)
    return total
end
@kernel deep_object_observations(groups, mu::Float64, sigma::Float64) = begin
    grouped = plate(groups, Ref(mu), Ref(sigma)) do group, location, scale
        middle = plate(group) do observations
            pointwise = plate(observations) do value
                scalar_normal(value, location, scale).logpdf()
            end
            sum(pointwise)
        end
        sum(middle)
    end
    total = sum(grouped)
    return total
end
@kernel scanned_object_observations(groups, mu::Float64, sigma::Float64) = begin
    grouped = plate(groups, Ref(mu), Ref(sigma)) do observations, location, scale
        terms = scan(observations, Ref(location), Ref(scale); init=0.0) do carry, value, m, s
            density = normal_method(m, s).logpdf(value)
            (carry + density, density)
        end
        sum(terms)
    end
    total = sum(grouped)
    return total
end
"""
Core.eval(@__MODULE__, Meta.parseall(ENDPOINT_SOURCE))

const DOCS_SOURCE = NORMAL_SOURCE * raw"""

inputs = (observation_groups = [[0.2, 0.7], Float64[], [-1.2]], scale = 1.3)
reader = prepare(visible_observations)
output = reader(values(inputs)...)
expected = -1.5 * log(2 * pi) - 3 * log(inputs.scale) -
           (0.2^2 + 0.7^2 + 1.2^2) / (2 * inputs.scale^2)
@assert output ≈ expected
docs_example = (; name = :nested_observation_plate,
                 origin = "test/fixtures/nested_plates.jl",
                 inputs, kernel = reader, output)
"""

@kernel cross_product(xs, ys) = begin
    out = plate(xs) do x
        inner = plate(ys) do y
            x * y
        end
        sum(inner)
    end
    return out
end

# Computed axes and several shared ports require concrete type propagation
# through a materialized inner plate followed by its sum consumer.
@kernel indexed_projection(matrix, rates, groups, parameters::Vector{Float64}) = begin
    scale = parameters[1]
    decay = parameters[2]
    coefficients = reshape(parameters[4:end], 2, size(matrix, 2))
    projected = plate(axes(matrix, 1), groups) do i, g
        terms = plate(axes(matrix, 2)) do j
            weight = scale * exp(-decay * rates[j])
            matrix[i, j] * weight * coefficients[g, j]
        end
        sum(terms)
    end
    values() = projected
end

@kernel indexed_loss(matrix, rates, groups, targets, parameters::Vector{Float64}) = begin
    predictions = indexed_projection(matrix, rates, groups, parameters).values()
    shifted = parameters[3] .* ones(length(targets)) .+ predictions
    errors = plate(shifted, targets) do prediction, target
        (prediction - target)^2
    end
    total = sum(errors)
    return total
end

@kernel inline_product(xs, ys) = begin
    out = plate(xs) do x
        sum(plate(ys) do y
            x * y
        end)
    end
    return out
end

@kernel inner_product(ys, x) = begin
    cells = plate(ys) do y
        x * y
    end
    total = sum(cells)
    return total
end

@kernel composed_product(xs, ys) = begin
    out = plate(xs) do x
        inner_product(ys, x)
    end
    return out
end

@kernel three_levels(groups, scale) = begin
    values = plate(groups) do group
        middle = plate(group) do observations
            inner = plate(observations) do x
                scale * x
            end
            sum(inner)
        end
        sum(middle)
    end
    total = sum(values)
    return total
end

@kernel rectangular(X, scale) = begin
    values = plate(eachcol(X)) do observations
        pointwise = plate(observations) do x
            scale * x
        end
        sum(pointwise)
    end
    total = sum(values)
    return total
end

@kernel nested_axes(groups, scales) = begin
    values = plate(groups) do observations
        cells = plate(observations, scales) do x, s
            x * s
        end
        sum(cells)
    end
    return values
end

@kernel guarded(groups, scale) = begin
    values = plate(groups) do observations
        pointwise = plate(observations) do x
            x >= 0 ? log(x + scale) : -scale * x
        end
        sum(pointwise)
    end
    total = sum(values)
    return total
end

head_count(ex, head) = ex isa Expr ?
    Int(ex.head === head) + sum(x -> head_count(x, head), ex.args; init=0) : 0
allocated(k::K, args::Vararg{Any,N}) where {K,N} = @allocated k(args...)
end
