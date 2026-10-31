module ValueMetadataTests
using Test, ReactiveKernels, DifferentiationInterface, Enzyme
const OS = ReactiveKernels.OutputSignatures

const helper_calls = Ref(0)
opaque_shape_change(x) = (helper_calls[] += 1; length(x))
opaque_value(v) = (helper_calls[] += 1; v)
const sum_alias = sum
const pow_alias = ^

@kernel metadata_total(b, alpha, beta, x, group) = begin
    total = sum(alpha .+ beta .* x .+ b[group])
end

info(ex, bindings; kwargs...) =
    ReactiveKernels._expression_value_info(ex, bindings, @__MODULE__; kwargs...)

@testset "visible calls use numerical metadata without executing helpers" begin
    bindings = Dict(:a=>OS.scalar_info(Float64), :x=>OS.vector_info(Float64, 3),
        :unknown=>OS.ValueInfo(), :flags=>OS.vector_info(Bool, 3))
    scalar = info(:(x[2]), bindings)
    @test OS.value_rank(scalar) === 0
    @test OS.type_bound(scalar) === Float64
    vector = info(:(a .+ x), bindings)
    @test OS.value_axes(vector) == (Base.OneTo(3),)
    @test OS.type_bound(vector) === Vector{Float64}
    @test OS.type_bound(info(:(sum(flags)), bindings)) === Int
    @test OS.type_bound(info(:(flags .+ flags), bindings)) === Vector{Int}
    @test OS.value_rank(info(:(opaque_shape_change(x)), bindings)) === nothing
    @test helper_calls[] == 0
    @test OS.value_rank(info(:(a .+ unknown), bindings)) === nothing
    @test OS.value_rank(info(:(Base.sum(x)), bindings)) === 0
    @test OS.value_rank(info(:(sum_alias(x)), bindings)) === 0
    @test OS.value_rank(info(:(ValueMetadataTests.opaque_shape_change(x)), bindings)) === nothing
    @test helper_calls[] == 0
    @test OS.value_rank(info(:(sum(x)), bindings;
        shadowed=union(keys(bindings), (:sum,)))) === nothing
    @test OS.value_rank(info(:(exp.(x)), bindings;
        shadowed=union(keys(bindings), (:exp,)))) === nothing
    # A keyword is never mistaken for a full reduction: `dims` keeps the
    # reduced axis, and a keyword without a rule leaves the result unknown.
    dims_sum = info(:(sum(x; dims=1)), bindings)
    @test OS.value_rank(dims_sum) === 1
    @test OS.value_length(dims_sum) === 1
    @test OS.value_rank(info(:(sum(x; init=0.0)), bindings)) === nothing
    @test OS.value_rank(info(:(x .+ y), merge(bindings,
        Dict(:y=>OS.vector_info(Float64, 2))))) === nothing
    partial = info(:(a .+ x), merge(bindings,
        Dict(:x=>OS.vector_info(Float64))))
    @test OS.value_rank(partial) === 1
    @test OS.value_length(partial) === nothing
end

@testset "literal powers preserve Julia's source semantics" begin
    bindings = Dict(:x=>OS.vector_info(Int8, 2), :a=>OS.scalar_info(Int8),
        :n=>OS.scalar_info(Int))
    for exponent in (-1, 0, 2, 3)
        scalar = info(Expr(:call, :^, :a, exponent), bindings)
        vector = info(Expr(:call, Symbol(".^"), :x, exponent), bindings)
        expected = Base.literal_pow(^, Int8(2), Val(exponent))
        @test typeof(expected) <: OS.type_bound(scalar)
        @test OS.type_bound(vector) === Vector{typeof(expected)}
    end
    # A normal call through an alias is not Julia's literal-power syntax.
    @test OS.type_bound(info(:(pow_alias(a, -1)), bindings)) === Int8
    @test OS.type_bound(info(:(Base.:^(a, -1)), bindings)) === Int8
    @test OS.type_bound(info(:( (^).(x, -1) ), bindings)) === Vector{Float64}
    @test OS.type_bound(info(Expr(:call, GlobalRef(Base, :^), :a, -1), bindings)) === Float64
    @test OS.type_bound(info(:(a ^ n), bindings)) === Int8
end

@testset "concatenation keeps the axes of an opaque broadcast" begin
    # An opaque scalar broadcast knows its axes but not its element type.
    # Concatenation takes its axes from operand shapes alone, so the result
    # is `:unknown` with sound partial axes (RKPPL keeps them to size
    # `beta[axes(X, 2)]`). Partial facts do not authorize fusion:
    # `_call_value_info` is the supported-only view.
    calls = helper_calls[]
    meta(ex, args) = ReactiveKernels._call_metadata(ex, OS.ValueInfo[args...],
        @__MODULE__, ())
    for n in (0, 1, 7)
        column = OS.vector_info(Float64, n)
        scaled = info(:(a .* x), Dict(:a=>OS.scalar_info(Float64), :x=>column))
        opaque = meta(:(opaque_value.(z)), [scaled])
        @test opaque.status === :unknown
        @test OS.value_axes(opaque.value) == (Base.OneTo(n),)
        X = meta(:(hcat(c, z)), [column, opaque.value])
        @test X.status === :unknown
        @test OS.value_axes(X.value) == (Base.OneTo(n), Base.OneTo(2))
        @test OS.value_rank(ReactiveKernels._call_value_info(:(hcat(c, z)),
            OS.ValueInfo[column, opaque.value], @__MODULE__, ())) === nothing
        stacked = meta(:(vcat(c, z)), [column, opaque.value])
        @test stacked.status === :unknown
        @test OS.value_axes(stacked.value) == (Base.OneTo(2n),)
    end
    # Known unequal row counts are a proved conflict, not an unknown.
    @test meta(:(hcat(c, z)), [OS.vector_info(Float64, 2),
        OS.vector_info(Float64, 3)]).status === :incompatible
    @test helper_calls[] == calls
end

@testset "comparisons and ifelse have metadata" begin
    # OutputSignatures 0.1.6 describes the comparisons and `ifelse` over
    # machine-real scalars, called or broadcast: RKPPL re-proves a location
    # summand `b .* ifelse.(a .> 0, a, b)` scalar through these facts.
    bindings = Dict(:a=>OS.scalar_info(Float64), :b=>OS.scalar_info(Float64),
        :x=>OS.vector_info(Float64, 3))
    for op in (:<, :<=, :>, :>=, :(==), :!=)
        called = info(Expr(:call, op, :a, :b), bindings)
        @test OS.value_rank(called) === 0
        @test OS.type_bound(called) === Bool
        dotted = info(Expr(:call, Symbol(".", op), :a, 0), bindings)
        @test OS.value_rank(dotted) === 0
        @test OS.type_bound(dotted) === Bool
    end
    for ex in (:(ifelse(a > 0, a, b)), :(ifelse.(a .> 0, a, b)),
            :(b .* ifelse.(a .<= b, exp.(a), b)))
        s = info(ex, bindings)
        @test OS.value_rank(s) === 0
        @test OS.type_bound(s) === Float64
    end
    masked = info(:(ifelse.(x .> a, x, b)), bindings)
    @test OS.value_axes(masked) == (Base.OneTo(3),)
    @test OS.type_bound(masked) === Vector{Float64}
end

@testset "kernel chain agrees with Julia and ordinary Reverse" begin
    f = prepare(metadata_total)
    for group in (Int[], [2], [1, 3, 2, 3, 1])
        b, alpha, beta = [-0.1, 0.2, 0.4], 0.3, -0.2
        x = [sin(i) for i in eachindex(group)]
        bindings = Dict(:b=>OS.value_info(b), :alpha=>OS.value_info(alpha),
            :beta=>OS.value_info(beta), :x=>OS.value_info(x),
            :group=>OS.value_info(group))
        result = info(:(sum(alpha .+ beta .* x .+ b[group])), bindings)
        expected = sum(alpha .+ beta .* x .+ b[group])
        @test typeof(expected) <: OS.type_bound(result)
        @test OS.value_rank(result) === 0
        @test f(b, alpha, beta, x, group) ≈ expected
        @test ad_gradient(metadata_total, AutoEnzyme(; mode=Enzyme.Reverse),
            b, alpha, beta, x, group; active=:b, want=:total) ≈
            [count(==(i), group) for i in eachindex(b)]
    end
    @test_throws BoundsError f([0.1], 0.2, 0.3, [0.4], [2])
    @test_throws DimensionMismatch f([0.1], 0.2, 0.3, [0.4, 0.5], [1, 1, 1])
end

@kernel metadata_plate(x, rows) = begin
    values = plate(rows) do i
        v::Float64 = opaque_value(x[i])
        v
    end
    return values
end

@kernel metadata_design(x, rows, one, beta) = begin
    values = metadata_plate(x, rows)
    X = hcat(one, log.(values))
    result = X * beta
    return result
end

struct ConvertedVector <: AbstractVector{Float64}
    data::Vector{Float64}
end
Base.size(x::ConvertedVector) = size(x.data)
Base.getindex(x::ConvertedVector, i::Int) = x.data[i]
Base.convert(::Type{ConvertedVector}, x::Vector{Float64}) =
    (helper_calls[] += 1; ConvertedVector(x[1:1]))

@kernel metadata_conversion(x::Vector{Float64}) = begin
    result::ConvertedVector = x
    return result
end

@kernel metadata_opaque(x) = begin
    result = opaque_shape_change(x)
    return result
end

@testset "visible kernel graphs retain plate axes without running cells" begin
    calls = helper_calls[]
    for n in (0, 1, 7, OS.SymbolicAxis(:rows), nothing)
        axes = (n isa Int ? Base.OneTo(n) : n,)
        args = OS.ValueInfo[OS.vector_info(Float64, n), OS.vector_info(Int, n)]
        result = ReactiveKernels._call_metadata(:(metadata_plate(x, rows)), args,
            @__MODULE__, ())
        @test result.status === :unknown
        @test OS.value_axes(result.value) == axes
        @test OS.element_type(result.value) === Float64
        design = ReactiveKernels._call_metadata(:(metadata_design(x, rows, one, beta)),
            OS.ValueInfo[args; OS.vector_info(Float64, n); OS.vector_info(Float64, 2)],
            @__MODULE__, ())
        @test OS.value_axes(design.value) == axes
        # These are graph facts, not permission to substitute an elementwise
        # numerical rule for the kernel (whose cell can contain an opaque call).
        @test OS.value_rank(ReactiveKernels._call_value_info(:(metadata_plate(x, rows)),
            args, @__MODULE__, ())) === nothing
    end
    for kernel in (metadata_conversion, metadata_opaque)
        result = ReactiveKernels._metadata_call(kernel,
            OS.ValueInfo[OS.vector_info(Float64, 7)];
            kwargs = Pair{Symbol,OS.ValueInfo}[])
        @test result.status === :unknown
        @test OS.value_axes(result.value) === nothing
    end
    @test helper_calls[] == calls
end
end
