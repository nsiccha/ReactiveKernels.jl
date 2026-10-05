module NativeConcatTests
using ReactiveKernels, Test
const RK = ReactiveKernels

# Deterministic operands of element type `T`, offset so that operands differ.
_values(T, n, offset) = T === Bool ? Bool[isodd(i + offset) for i in 1:n] :
    T[T(i + offset) for i in 1:n]
_operand(T, offset, n) = _values(T, n, offset)
_operand(T, offset, m, n) = reshape(_values(T, m * n, offset), m, n)

# Same value and type as Base, or the same exception type and message.
function _same_as_base(native, base, args...)
    expected = try
        base(args...)
    catch err
        err
    end
    actual = try
        native(args...)
    catch err
        err
    end
    expected isa Exception &&
        return typeof(actual) == typeof(expected) &&
               sprint(showerror, actual) == sprint(showerror, expected)
    typeof(actual) == typeof(expected) && size(actual) == size(expected) &&
        isequal(actual, expected)
end

@testset "native concatenation companions follow Base" begin
    for T in (Float64, Float32, Int, Bool, ComplexF64)
        v(n, o = 0) = _operand(T, o, n)
        m(r, c, o = 0) = _operand(T, o, r, c)
        hcats = ((v(3), v(3, 7)), (v(3), v(3, 1), v(3, 2)), (m(3, 2), v(3)),
                 (v(3), m(3, 2)), (m(3, 2), m(3, 1, 4), v(3, 9)), (v(0), v(0)),
                 (m(0, 2), v(0)), (m(2, 0), v(2)), (v(4),), (m(2, 3),),
                 (v(3), v(2)), (m(3, 2), v(2)), (v(3), m(2, 2)))
        for args in hcats
            @test _same_as_base(RK._native_hcat, hcat, args...)
        end
        vcats = ((v(3), v(2, 5)), (v(0), v(4)), (v(2),), (m(3, 2), m(1, 2, 6)),
                 (m(3, 1), v(2)), (v(2), m(3, 1, 4)), (m(2, 2),),
                 (m(0, 2), m(2, 2)), (m(3, 2), v(2)), (m(3, 2), m(2, 3)))
        for args in vcats
            @test _same_as_base(RK._native_vcat, vcat, args...)
        end
        hvcats = (((2, 2), m(2, 2), m(2, 1, 4), m(1, 2, 6), m(1, 1, 8)),
                  ((2,), v(2), v(2, 3)), ((1, 2), m(1, 3), m(1, 1, 3), m(1, 2, 4)),
                  ((2, 2), v(2), v(2, 2), v(2, 4), v(2, 6)),
                  ((1, 1), v(3), m(2, 1)), ((2, 1), m(0, 2), m(0, 1), m(2, 3)),
                  ((2,), m(2, 2), m(3, 1)), ((2, 1), m(1, 1), m(1, 1), m(1, 3)),
                  ((2, 2), m(1, 1), m(1, 1), m(1, 1)), ((3,), v(2), v(2)))
        for (rows, args...) in hvcats
            @test _same_as_base(RK._native_hvcat, hvcat, rows, args...)
        end
    end
    # Operand combinations the companions do not specialize keep Base's call.
    x, y = [1.0, 2.0], [3, 4]
    for args in ((x, y), (x, 5.0), (5.0, x), (["a", "b"], ["c", "d"]),
                 (Any[1, 2], Any[3, 4]), (view(x, 1:2), x), (x', x'), (1, 2.0), ())
        @test _same_as_base(RK._native_hcat, hcat, args...)
        @test _same_as_base(RK._native_vcat, vcat, args...)
    end
    @test _same_as_base(RK._native_hvcat, hvcat, (2, 2), 1.0, 2.0, 3.0, 4.0)
    @test _same_as_base(RK._native_hvcat, hvcat, 2, [1.0 2.0], [3.0 4.0])
    @test _same_as_base(RK._native_hvcat, hvcat, (1, 1), x', [3.0 4.0])
    # A fresh output: writing it never reaches an operand.
    a, b, c = [1.0, 2.0], [3.0, 4.0], [5.0 6.0; 7.0 8.0]
    for out in (RK._native_hcat(a, b), RK._native_vcat(a, b),
                RK._native_vcat(c, c), RK._native_hvcat((2, 1), a, b, c))
        out .= -1
    end
    @test a == [1.0, 2.0] && b == [3.0, 4.0] && c == [5.0 6.0; 7.0 8.0]
end

module OtherConcat
hcat(args...) = :other
end

@kernel calls(a, b) = begin
    result = (hcat(a, b), vcat(a, b), Base.hcat(a, b), hvcat((2, 2), a, b, b, a))
end
@kernel brackets(a, b) = begin
    result = ([a b], [a; b], [a b; b a], [[a b]; b a], [a b b])
end
@kernel recipe(a, b) = begin
    result = hcat(a, b)
end
@kernel shadowed_local(a, b) = begin
    result = let hcat = (x, y) -> x .+ y
        hcat(a, b)
    end
end
@kernel shadowed_port(hcat, a, b) = begin
    result = hcat(a, b)
end
@kernel shadowed_module_port(Base, a, b) = begin
    result = (Base.hcat(a, b), Base.vcat(a, b), Base.hvcat((2,), a, b))
end
@kernel shadowed_module_local(a, b) = begin
    result = let Base = (; hcat = (x, y) -> x .+ y)
        Base.hcat(a, b)
    end
end
@kernel other_function(a, b) = begin
    result = OtherConcat.hcat(a, b)
end
@kernel cell_vcat(x, c) = begin
    pointwise = plate(x, Ref(c)) do xi, cc
        sum(vcat(cc, [2xi]))
    end
    total = sum(pointwise)
end

@testset "native kernel bodies concatenate as Base" begin
    a, b = [1.0, 2.0], [3.0, 4.0]
    @test prepare(calls)(a, b) ==
          (hcat(a, b), vcat(a, b), hcat(a, b), hvcat((2, 2), a, b, b, a))
    @test prepare(brackets)(a, b) == ([a b], [a; b], [a b; b a], [[a b]; b a], [a b b])
    @test prepare(recipe)(a, b) == hcat(a, b)
    @test prepare(recipe; bound = (; a))(b) == hcat(a, b)
    @test prepare(shadowed_local)(a, b) == a .+ b
    @test prepare(shadowed_port)((x, y) -> x .- y, a, b) == a .- b
    alternate = (; hcat = (x, y) -> x .- y, vcat = (x, y) -> x .* y,
                   hvcat = (rows, x, y) -> x .+ y)
    @test prepare(shadowed_module_port)(alternate, a, b) ==
          (a .- b, a .* b, a .+ b)
    @test prepare(shadowed_module_local)(a, b) == a .+ b
    @test prepare(other_function)(a, b) === :other
    @test prepare(cell_vcat)([1.0, 2.0, 3.0], [0.5, 0.25]) ≈ 3 * 0.75 + 12.0
    @test_throws DimensionMismatch prepare(recipe)([1.0, 2.0], [3.0])
    @test a == [1.0, 2.0] && b == [3.0, 4.0]

    # The native body names the companions; the tensorized body is unchanged.
    native(ex, names...) = RK._kernel_native_body(ex, @__MODULE__, Set{Symbol}(names))
    has(ex, name) = occursin(string(name), string(ex))
    @test has(native(:(hcat(a, b)), :a, :b), :_native_hcat)
    @test has(native(:([a; b]), :a, :b), :_native_vcat)
    @test has(native(:([a b; b a]), :a, :b), :_native_hvcat)
    @test !has(native(:(hcat(a, b)), :a, :b, :hcat), :_native_hcat)
    @test !has(native(:(OtherConcat.hcat(a, b)), :a, :b), :_native_hcat)
    @test !has(native(:(cat(a, b; dims = 2)), :a, :b), :_native_)
end
end
