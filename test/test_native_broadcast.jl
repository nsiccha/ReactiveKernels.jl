module NativeBroadcastTests
using ReactiveKernels, Test

@kernel fused(x, y) = begin
    result = sin.(x .* y) .+ 2 .* x
end
@kernel barrier(x, y) = begin
    result = cos.(x .+ sum(y .* y))
end
@kernel callable(f, x) = begin
    result = f.(x)
end
@kernel noargs(f) = begin
    result = f.()
end
@kernel qualified(x) = begin
    result = Base.sin.(x)
end
@kernel splatted(xs) = begin
    result = .+(xs...)
end
@kernel owned_update(x) = begin
    result = let dst = copy(x)
        dst .= x .* 2
        dst .+= sin.(x) .* 3
        dst
    end
end
@kernel lazy(x, take) = begin
    result = take ? sin.(x) : sqrt.(-x)
end
@kernel dotted_and(x) = begin
    result = x .> 0 .&& log.(x) .> 0
end
@kernel dotted_or(x) = begin
    result = x .< 0 .|| sqrt.(x) .> 0
end
@kernel nested_dotted_and(x) = begin
    result = identity.(x .> 0 .&& log.(x) .> 0)
end

# A custom style must retain its own copy/instantiate protocol.
struct StyledVector{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(x::StyledVector) = size(x.data)
Base.getindex(x::StyledVector, i::Int) = x.data[i]
Base.BroadcastStyle(::Type{<:StyledVector}) = Base.Broadcast.ArrayStyle{StyledVector}()
function Base.copy(bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{StyledVector}})
    plain = Base.Broadcast.Broadcasted(
        Base.Broadcast.DefaultArrayStyle{1}(), bc.f, bc.args, bc.axes)
    StyledVector(Base.materialize(plain))
end

# Specialized broadcasted methods can supply axes, which Base validates.
struct FixedAxesVector{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(x::FixedAxesVector) = size(x.data)
Base.getindex(x::FixedAxesVector, i::Int) = x.data[i]
Base.Broadcast.broadcasted(::typeof(sin), x::FixedAxesVector) =
    Base.Broadcast.Broadcasted(Base.Broadcast.DefaultArrayStyle{1}(), sin,
                               (x,), (Base.OneTo(1),))

@testset "native dotted calls preserve Julia broadcast semantics" begin
    f = prepare(fused)
    for (x, y) in ((2.0, 3.0), (fill(2.0), fill(3.0)),
                   ((1.0, 2.0), 3.0), (1:3, 2),
                   ([1.0, 2.0], Ref(3.0)),
                   (reshape(1.0:6.0, 3, 2), [2.0, 3.0, 4.0]),
                   (view([1.0, 2.0, 3.0], 1:2), [3.0, 4.0]),
                   (Float64[], 2.0))
        expected = sin.(x .* y) .+ 2 .* x
        actual = f(x, y)
        @test actual == expected
        @test typeof(actual) == typeof(expected)
    end
    @test_throws DimensionMismatch f(ones(2), ones(3))
    x = [0.2, 0.4, 0.7]
    saved = copy(x)
    @test prepare(barrier)(x, x) == cos.(x .+ sum(x .* x))
    @test prepare(callable)(sin, x) == sin.(x)
    @test prepare(noargs)(() -> 2.0) == 2.0
    @test prepare(qualified)(x) == sin.(x)
    @test prepare(splatted)((x, x, x)) == .+(x, x, x)
    @test prepare(owned_update)(x) == 2 .* x .+ 3 .* sin.(x)
    @test prepare(lazy)(x, true) == sin.(x)
    @test prepare(lazy)(-x, false) == sqrt.(x)
    mixed = [-2.0, -1.0, 0.2, 2.0]
    @test prepare(dotted_and)(mixed) == (mixed .> 0 .&& log.(mixed) .> 0)
    @test prepare(dotted_or)(mixed) == (mixed .< 0 .|| sqrt.(mixed) .> 0)
    @test prepare(nested_dotted_and)(mixed) ==
        identity.(mixed .> 0 .&& log.(mixed) .> 0)
    @test x == saved
    styled = StyledVector(x)
    result = prepare(qualified)(styled)
    @test result isa StyledVector
    @test result.data == sin.(x)
    @test_throws DimensionMismatch prepare(qualified)(FixedAxesVector(x))
end
end
