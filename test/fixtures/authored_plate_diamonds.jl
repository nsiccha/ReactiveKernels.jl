module AuthoredPlateDiamonds
using ReactiveKernels
opaque_sum(x) = sum(abs2, x)

# Graph depth and source order vary independently of runtime array dimensions.
# Every level has a shared predecessor, two branches, and a reconvergence.
for depth in (1, 3, 6), swapped in (false, true)
    name = Symbol(:diamond_, depth, swapped ? :_swapped : :_ordered)
    statements = Any[]
    previous = :base
    for level in 1:depth
        left, right, joined = Symbol(:left_, level), Symbol(:right_, level), Symbol(:joined_, level)
        branches = [
            :($left = plate($previous) do a; sin(a); end),
            :($right = plate($previous) do a; cos(a); end)]
        append!(statements, swapped ? reverse(branches) : branches)
        push!(statements, :($joined = plate($previous, $left, $right) do a, b, c
            a + b * c
        end))
        previous = joined
    end
    @eval @kernel $name(q::Vector{Float64}, x) = begin
        beta::Float64 = sum(q)
        base = plate(x, beta) do a, b; a * b; end
        $(statements...)
        total::Float64 = sum($previous)
        extra::Float64 = sum(base)
        opaque::Float64 = opaque_sum(base)
        return total
    end
end
spec(depth, swapped = false) = getfield(@__MODULE__,
    Symbol(:diamond_, depth, swapped ? :_swapped : :_ordered))

function reference(q, x, depth)
    values = sum(q) .* x
    slopes = copy(x)
    for _ in 1:depth
        slopes .*= 1 .+ cos.(2 .* values)
        values .+= sin.(values) .* cos.(values)
    end
    sum(values), fill(sum(slopes), length(q))
end

@kernel multidimensional(q::Vector{Float64}, x, z) = begin
    beta::Float64 = sum(q)
    base = plate(x, z, beta) do a, b, c; (a + b) * c; end
    left = plate(base) do a; sin(a); end
    right = plate(base, left) do a, b; a + b; end
    total::Float64 = sum(right)
    return total
end

@kernel atomic(q::Vector{Float64}, x) = begin
    base = plate(x, Ref(q)) do a, b; a * sum(b); end
    left = plate(base) do a; sin(a); end
    right = plate(base, left, Ref(base)) do a, b, whole
        a + b + sum(whole)
    end
    total::Float64 = sum(right)
    return total
end

# Two reconvergences of one producer: neither branch alone is its sole reader.
@kernel independent(q::Vector{Float64}, x) = begin
    beta::Float64 = sum(q)
    base = plate(x, beta) do a, b; a * b; end
    left = plate(base) do a; sin(a); end
    right = plate(base) do a; cos(a); end
    total::Float64 = sum(left) + sum(right)
    return total
end

# Opaque scalar calls retain ordinary Julia branches and errors.
checked(x) = x < 0 ? throw(DomainError(x)) : (x == 0 ? 0.0 : log(x))
@kernel checked_diamond(x) = begin
    base = plate(x) do a; checked(a); end
    left = plate(base) do a; sin(a); end
    right = plate(base, left) do a, b; a + b; end
    total::Float64 = sum(right)
    return total
end

@kernel expanding_diamond(x, y) = begin
    base = plate(x) do a; checked(a); end
    left = plate(base) do a; sin(a); end
    right = plate(base, left, y) do a, b, c; a + b + c; end
    total::Float64 = sum(right)
    return total
end

nodes(x) = x isa Expr ? 1 + sum(nodes, x.args; init = 0) : 1
allocated(k::K, args::Vararg{Any,N}) where {K,N} = @allocated k(args...)
end
