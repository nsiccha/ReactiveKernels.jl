module NativeBroadcastTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

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
@kernel identity_values(x) = begin
    result = identity.(x)
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

struct SpecializedBroadcast end
(::SpecializedBroadcast)(x, rest...) = x + sum(rest)
Base.Broadcast.broadcasted(::SpecializedBroadcast, x, rest...) =
    Base.Broadcast.broadcasted(+, x .+ 7, rest...)

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
    # Nonconcrete inferred elements retain Base's widening copy behavior.
    values = Any[1, "two", nothing]
    actual_values = prepare(identity_values)(values)
    @test actual_values == identity.(values)
    @test typeof(actual_values) == typeof(identity.(values))
    @test values == Any[1, "two", nothing]
    # Boolean results keep Base's BitArray container, including its large-copy path.
    signs = collect(range(-2.0, 2.0; length=257))
    mask = prepare(dotted_and)(signs)
    @test mask == (signs .> 0 .&& log.(signs) .> 0)
    @test mask isa BitVector
    @test x == saved
    styled = StyledVector(x)
    result = prepare(qualified)(styled)
    @test result isa StyledVector
    @test result.data == sin.(x)
    @test_throws DimensionMismatch prepare(qualified)(FixedAxesVector(x))
end

# Expand the authored expression's fixed arity, never its data-sized row loop.
macro wide_broadcast(name, width)
    terms = [:(q[$i] .* xs[$i]) for i in 1:width]
    predictor = Expr(:call, :.+, terms...)
    esc(:(@kernel $name(q::Vector{Float64}, xs) = begin
        result = sum(exp.($predictor))
    end))
end
@wide_broadcast wide30 30
@wide_broadcast wide40 40
@wide_broadcast wide60 60

@kernel factor_broadcast(x::Vector{Float64}, rows::Vector{Int}) = begin
    result = sum(exp.(x[rows]))
end

function broadcast_bytes(f, args...)
    f(args...)
    minimum(@allocated(begin f(args...) end) for _ in 1:3)
end

@testset "wide native broadcasts retain every operand and runtime row loop" begin
    # Changing the native constructor must retain broadcast recognition in
    # the compiler's row analysis, including narrow nested expressions.
    factors = ReactiveKernels._cell_factor_graph(factor_broadcast, :result;
        bound=(; rows=[1, 3]), locality=:error)
    @test length(factors.sums) == 1
    @test ReactiveKernels._cell_factor_rows(factors.spec,
        only(factors.sums).argument) !== nothing
    for (p, spec) in ((30, wide30), (40, wide40), (60, wide60))
        q = collect(range(-0.01, 0.02; length=p))
        kernel = prepare(spec; type_stability=:error)
        for n in (0, 1, 7)
            xs = ntuple(i -> i % 3 == 0 ? fill(i % 5, n) :
                i % 3 == 1 ? fill(Float32(i / p), n) : fill(-i / p, n), p)
            saved = deepcopy(xs)
            eta = zeros(n)
            for j in 1:p
                eta .+= q[j] .* xs[j]
            end
            expected = sum(exp.(eta))
            @test kernel(q, xs) ≈ expected
            bound = prepare(spec; bound=(;xs), type_stability=:error)
            @test bound(q) ≈ expected
            ad = prepare_ad(bound, AutoEnzyme(; mode=Enzyme.Reverse), q; active=:q)
            value, gradient = ad_value_and_gradient(ad, q)
            @test value ≈ expected
            @test gradient ≈ [sum(xs[j] .* exp.(eta)) for j in 1:p]
            @test xs == saved
            @test isempty(ReactiveKernels._code_findings(
                bound, Tuple{typeof(q)}))
        end
    end
    # The regression was heap tuple reconstruction at one element, not row
    # storage: previously 77/343 KB at 40/60 operands on Julia 1.10.
    for p in (40, 60), n in (1, 128)
        args = ntuple(i -> fill(i / p, n), p)
        bc = Base.Broadcast.broadcasted(+, args...)
        got = ReactiveKernels._native_broadcast_materialize(bc)
        @test got == Base.materialize(bc)
        @test broadcast_bytes(ReactiveKernels._native_broadcast_materialize, bc) <
            4096 + 32p * n
        @test all(x -> !Base.mightalias(got, x), args)
    end
end

@testset "wide broadcasts preserve shape, type and custom protocols" begin
    materialize = ReactiveKernels._native_broadcast_materialize
    # Singleton expansion, views, scalar Ref values, and zero-dimensional
    # inputs keep the same indexing and element promotion as ordinary Julia.
    data = reshape(collect(1.0:12.0), 3, 4)
    args = ntuple(i -> i % 4 == 0 ? view(data, :, :) :
        i % 4 == 1 ? [1, 2, 3] : i % 4 == 2 ? Ref(0.5f0) : fill(0.25), 40)
    bc = Base.Broadcast.broadcasted(+, args...)
    actual = materialize(bc)
    @test actual == Base.materialize(bc)
    @test typeof(actual) == typeof(Base.materialize(bc))
    nested = Base.Broadcast.broadcasted(sin, bc)
    @test materialize(nested) == Base.materialize(nested)
    bad = Base.Broadcast.broadcasted(+, args..., ones(2))
    @test_throws DimensionMismatch materialize(bad)
    bools = Base.Broadcast.broadcasted(>, bc, 0)
    @test materialize(bools) == Base.materialize(bools)
    @test materialize(bools) isa BitMatrix
    # A custom scalar function receives all arguments in their original order.
    order_sensitive(first, rest...) = first - sum(rest)
    custom = Base.Broadcast.broadcasted(order_sensitive, args...)
    @test materialize(custom) == Base.materialize(custom)
    # Fused short-circuit calls must not evaluate the skipped scalar arm.
    negative = Base.Broadcast.broadcasted(+, ntuple(_ -> [-1.0, -2.0], 40)...)
    unsafe = Base.Broadcast.broadcasted(>, Base.Broadcast.broadcasted(log, negative), 0)
    short_circuit = Base.Broadcast.broadcasted(Base.Broadcast.andand,
        [false, false], unsafe)
    @test materialize(short_circuit) == Base.materialize(short_circuit) == [false, false]
    # Inbounds operand reads must not suppress checks in the user's function.
    checked(x, rest...) = [x][2]
    @test_throws BoundsError materialize(Base.Broadcast.broadcasted(checked, args...))
    # Type-valued and nonconcrete elements keep Base's inference/widening.
    typed(::Type{T}, x, rest...) where {T} = T(x + sum(rest))
    typed_bc = Base.Broadcast.broadcasted(typed, Ref(Float32), args...)
    @test materialize(typed_bc) == Base.materialize(typed_bc)
    @test typeof(materialize(typed_bc)) == typeof(Base.materialize(typed_bc))
    heterogeneous = Base.Broadcast.broadcasted((x, rest...) -> x,
        Any[1, "two"], ntuple(_ -> [0, 0], 40)...)
    @test materialize(heterogeneous) == Base.materialize(heterogeneous)
    @test typeof(materialize(heterogeneous)) == typeof(Base.materialize(heterogeneous))
    styled = Base.Broadcast.broadcasted(+, StyledVector([1.0, 2.0]),
        ntuple(_ -> [1.0, 2.0], 40)...)
    @test materialize(styled) isa StyledVector
    @test materialize(styled).data == Base.materialize(styled).data
    styled_args = (StyledVector([1.0, 2.0]), ntuple(_ -> [1.0, 2.0], 40)...)
    @test prepare(splatted)(styled_args) isa StyledVector
    @test prepare(splatted)(styled_args).data == Base.materialize(styled).data
    custom_args = ntuple(i -> [Float64(i)], 40)
    custom_constructor = ReactiveKernels._native_broadcasted(
        SpecializedBroadcast(), custom_args...)
    @test materialize(custom_constructor) ==
        Base.materialize(Base.broadcasted(SpecializedBroadcast(), custom_args...))
    for f in (+, *, min, max)
        constructed = ReactiveKernels._native_broadcasted(f, custom_args...)
        @test materialize(constructed) == Base.materialize(Base.broadcasted(f, custom_args...))
    end
    # The arithmetic expansion retains Base's left association, including
    # rounding before a later operand promotes the scalar result.
    cancellation = ([1e16], [-1e16], [1.0], ntuple(_ -> [0.0], 37)...)
    @test prepare(splatted)(cancellation) == .+(cancellation...) == [1.0]
    promotion = ([16777216], [1.0f0], [-16777216.0], ntuple(_ -> [0.0], 37)...)
    @test prepare(splatted)(promotion) == .+(promotion...)
    # Axes supplied by a specialized broadcasted implementation stay on its
    # original validation/materialization path.
    supplied = Base.Broadcast.Broadcasted(Base.Broadcast.DefaultArrayStyle{1}(),
        +, ntuple(_ -> [1.0, 2.0], 40), (Base.OneTo(1),))
    @test_throws DimensionMismatch materialize(supplied)
end
end
