module NestedPlates
using ReactiveKernels

# One public source authority, also replayed by acceptance and shown in docs.
const NORMAL_SOURCE = raw"""
@kernel visible_observations(observation_groups, scale::Float64) = begin
    group_logdensity = plate(observation_groups, Ref(scale)) do observations, sigma
        pointwise = plate(observations, Ref(sigma)) do observation, s
            -0.5 * log(2 * pi) - log(s) - 0.5 * (observation / s)^2
        end
        sum(pointwise)
    end
    logdensity = sum(group_logdensity)
    return logdensity
end
"""
Core.eval(@__MODULE__, Meta.parse(NORMAL_SOURCE))

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
    out = plate(xs, Ref(ys)) do x, shared
        inner = plate(shared, Ref(x)) do y, xi
            xi * y
        end
        sum(inner)
    end
    return out
end

@kernel inline_product(xs, ys) = begin
    out = plate(xs, Ref(ys)) do x, shared
        sum(plate(shared, Ref(x)) do y, xi
            xi * y
        end)
    end
    return out
end

@kernel inner_product(ys, x) = begin
    cells = plate(ys, Ref(x)) do y, xi
        xi * y
    end
    total = sum(cells)
    return total
end

@kernel composed_product(xs, ys) = begin
    out = plate(xs, Ref(ys)) do x, shared
        inner_product(shared, x)
    end
    return out
end

@kernel three_levels(groups, scale) = begin
    values = plate(groups, Ref(scale)) do group, s
        middle = plate(group, Ref(s)) do observations, sigma
            inner = plate(observations, Ref(sigma)) do x, t
                t * x
            end
            sum(inner)
        end
        sum(middle)
    end
    total = sum(values)
    return total
end

@kernel rectangular(X, scale) = begin
    values = plate(eachcol(X), Ref(scale)) do observations, s
        pointwise = plate(observations, Ref(s)) do x, sigma
            sigma * x
        end
        sum(pointwise)
    end
    total = sum(values)
    return total
end

@kernel nested_axes(groups, scales) = begin
    values = plate(groups, Ref(scales)) do observations, sigma
        cells = plate(observations, sigma) do x, s
            x * s
        end
        sum(cells)
    end
    return values
end

@kernel guarded(groups, scale) = begin
    values = plate(groups, Ref(scale)) do observations, sigma
        pointwise = plate(observations, Ref(sigma)) do x, s
            x >= 0 ? log(x + s) : -s * x
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
