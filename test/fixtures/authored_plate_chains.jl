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

allocated(kernel::K, args::Vararg{Any,N}) where {K,N} = @allocated kernel(args...)
end
