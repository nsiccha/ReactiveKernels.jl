module ScanEndpointScopeFixtures
using ReactiveKernels

@kernel step_object(dt, value) = begin
    next_value = value + dt
    state()::Float64 = next_value
    shifted(offset::Float64)::Float64 = next_value + offset
end

@kernel step_function(dt, value) = begin
    next_value = value + dt
    return next_value
end

@kernel object_scan(times) = begin
    trajectory = scan(times; init = 0.0) do previous, time
        dt = time - previous
        next_value = step_object(dt, previous).state()
        (next_value, next_value)
    end
    total = sum(trajectory)
    return trajectory
end

@kernel function_scan(times) = begin
    trajectory = scan(times; init = 0.0) do previous, time
        dt = time - previous
        next_value = step_function(dt, previous)
        (next_value, next_value)
    end
    total = sum(trajectory)
    return trajectory
end

@kernel shifted_scan(xs) = begin
    trajectory = scan(xs; init = 0.0) do previous, x
        dt = 2x
        offset = x / 2
        next_value = step_object(; dt, value = previous).shifted(offset)
        (next_value, next_value)
    end
    return trajectory
end

@kernel computed_scan(xs) = begin
    trajectory = scan(xs; init = 0.0) do previous, x
        dt = 2x
        next_value = 2 * step_object(dt + 1, previous).shifted(x / 2)
        (next_value, next_value)
    end
    return trajectory
end

# A do-block formal passed directly to a typed endpoint argument takes that
# argument's declared type, as a plate formal does.
@kernel formal_argument_scan(xs, offset::Float64) = begin
    trajectory = scan(xs; init = 0.0) do previous, x
        next_value = step_object(offset, previous).shifted(x)
        (next_value, next_value)
    end
    return trajectory
end

@kernel lazy_scan(xs) = begin
    trajectory = scan(xs; init = 0.0) do previous, x
        next_value = x > 0 ? step_object(log(x), previous).state() : previous
        (next_value, next_value)
    end
    return trajectory
end

@kernel scaled_scan(q::Vector{Float64}, xs::Vector{Float64}) = begin
    trajectory = scan(xs; init = 0.0) do previous, x
        dt = q[1] * x
        next_value = step_object(dt, previous).state()
        (next_value, next_value)
    end
    total = sum(trajectory)
    return total
end

@kernel scaled_function_scan(q::Vector{Float64}, xs::Vector{Float64}) = begin
    trajectory = scan(xs; init = 0.0) do previous, x
        dt = q[1] * x
        next_value = step_function(dt, previous)
        (next_value, next_value)
    end
    total = sum(trajectory)
    return total
end
end
