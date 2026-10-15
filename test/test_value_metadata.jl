module ValueMetadataTests
using Test, ReactiveKernels, DifferentiationInterface, Enzyme
const OS = ReactiveKernels.OutputSignatures

const helper_calls = Ref(0)
opaque_shape_change(x) = (helper_calls[] += 1; length(x))
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
    # Unsupported keyword semantics must not be mistaken for a full reduction.
    @test OS.value_rank(info(:(sum(x; dims=1)), bindings)) === nothing
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
    @test OS.type_bound(info(:(a ^ n), bindings)) === Int8
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
end
