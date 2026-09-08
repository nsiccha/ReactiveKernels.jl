module AuthoredPlateChains
using ReactiveKernels

@kernel chain(q::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}) = begin
    beta::Float64 = sum(view(q, 1:1))
    mu = plate(x, beta) do xi, bi
        xi * bi
    end
    pointwise = plate(mu, y) do mi, yi
        -0.5 * (mi - yi)^2
    end
    total::Float64 = sum(pointwise)
    extra::Float64 = sum(mu)
    return total
end

@kernel flat(q::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}) = begin
    beta::Float64 = sum(view(q, 1:1))
    pointwise = plate(x, beta, y) do xi, bi, yi
        -0.5 * (xi * bi - yi)^2
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel multidimensional(x, z, y, scale) = begin
    mu = plate(x, z) do a, b
        a + b
    end
    pointwise = plate(mu, y, scale) do a, b, s
        (a - b) * log(s)
    end
    total = sum(pointwise)
    return total
end

@kernel repeated(x, scale) = begin
    first = plate(x, scale) do a, b
        a * b
    end
    second = plate(first) do a
        a
    end
    third = plate(second, second) do a, b
        a + b
    end
    total = sum(third)
    return total
end

@kernel atomic(x, y) = begin
    mu = plate(x) do a
        a^2
    end
    pointwise = plate(y, Ref(mu)) do a, b
        a + sum(b)
    end
    total = sum(pointwise)
    return total
end

@kernel unused_axis(x, y) = begin
    mu = plate(x) do a
        1.0
    end
    pointwise = plate(mu, y) do a, b
        a + b
    end
    total = sum(pointwise)
    return total
end

# A chain whose parameter vector `q` reaches BOTH plates only through an atomic
# `Ref(q)` whole-vector capture (indexed `q[1]`/`q[2]` inside the cell), mixed
# with an ordinary elementwise use of the axis operand. When `x`/`y` are bound,
# `q` is the only live HAVE yet is atomic-for-broadcast — the shape that used to
# make `prepare(...; bound = (; x, y))` throw "an embedded plate requires an
# array-valued HAVE port in the outer kernel".
@kernel ref_atomic_chain(q::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}) = begin
    middle = plate(x, Ref(q)) do xi, qq
        2 * xi + qq[1]
    end
    pointwise = plate(y, middle, Ref(q)) do yi, mi, qq
        yi + mi^2 + qq[2] * mi
    end
    total::Float64 = sum(pointwise)
    return total
end

allocated(kernel::K, args::Vararg{Any,N}) where {K,N} = @allocated kernel(args...)
end
