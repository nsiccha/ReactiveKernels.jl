module NativeConcatTests
using ReactiveKernels, Test
const RK = ReactiveKernels

# Deterministic operands of element type `T`, offset so that operands differ.
_values(T, n, offset) = T === Bool ? Bool[isodd(i + offset) for i in 1:n] :
    T[T(i + offset) for i in 1:n]
_operand(T, offset, n) = _values(T, n, offset)
_operand(T, offset, m, n) = reshape(_values(T, m * n, offset), m, n)

# Same value and type as Base.  Where Base throws, the same exception when
# the companion calls Base, or the companion's own `DimensionMismatch` or
# `ArgumentError` for a layout `hvcat` rejects itself (core.jl).
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
               sprint(showerror, actual) == sprint(showerror, expected) ||
               actual isa Union{DimensionMismatch,ArgumentError}
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
        # Scalars are 1×1 blocks; numbers and vectors alone stack to a vector.
        s(o) = only(_values(T, 1, o))
        hcats = ((s(1), s(2)), (s(1),), (m(1, 2), s(3)), (v(1), s(2), m(1, 2)),
                 (v(3), s(1)), (s(1), m(2, 2)))
        for args in hcats
            @test _same_as_base(RK._native_hcat, hcat, args...)
        end
        vcats = ((s(1), s(2), s(3)), (s(1),), (v(2), s(1)), (s(1), v(2), s(4)),
                 (v(0), s(1)), (m(2, 1), s(3)), (s(1), m(2, 1, 2)),
                 (m(2, 2), s(3)), (v(2), s(1), m(1, 1)),
                 # Base fills leading numbers across a wider matrix.
                 (s(1), m(2, 2)), (s(1), s(2), m(0, 3)), (s(1), m(1, 2), s(2)))
        for args in vcats
            @test _same_as_base(RK._native_vcat, vcat, args...)
        end
        hvcats = (((3, 3, 3), ntuple(k -> s(k), 9)...), ((1,), s(1)),
                  ((2, 4), m(3, 3), v(3, 9), s(1), s(2), s(3), s(4)),
                  ((2, 3), m(2, 2), v(2), s(1), s(2), s(3)),
                  ((3, 1), s(1), s(2), m(1, 1), m(2, 3)),
                  ((2, 2), m(2, 2), v(2), s(1), s(2)),
                  ((2, 2), s(1), s(2), s(3)), ((2, 2), ntuple(k -> s(k), 5)...),
                  ((3, 2), ntuple(k -> s(k), 5)...),
                  ((1, 2), m(0, 2), s(1), s(2)))
        for (rows, args...) in hvcats
            @test _same_as_base(RK._native_hvcat, hvcat, rows, args...)
        end
    end
    # Mixed element types promote as in Base.
    x, y = [1.0, 2.0], [3, 4]
    for args in ((x, y), (x, 5.0), (5.0, x), (1, 2.0), (true, 2), (Int8(1), [2.0]),
                 (π, 1.0), (π,), (Float32(1), [2.0f0], 3), (x, [true, false]))
        @test _same_as_base(RK._native_hcat, hcat, args...)
        @test _same_as_base(RK._native_vcat, vcat, args...)
    end
    @test _same_as_base(RK._native_hvcat, hvcat, (2, 2), 1, 2.5, true, 0)
    @test _same_as_base(RK._native_hvcat, hvcat, (2, 3), [1 2; 3 4], x, 0, 0.5f0, 1)
    # Operand combinations the companions do not specialize keep Base's call.
    for args in ((["a", "b"], ["c", "d"]), (Any[1, 2], Any[3, 4]),
                 (view(x, 1:2), x), (x', x'), (big(1.0), 2.0), (x, Real[3]), ())
        @test _same_as_base(RK._native_hcat, hcat, args...)
        @test _same_as_base(RK._native_vcat, vcat, args...)
    end
    @test _same_as_base(RK._native_hvcat, hvcat, (2, 2), 1.0, 2.0, 3.0, 4.0)
    @test _same_as_base(RK._native_hvcat, hvcat, (2, 2), big(1.0), 2.0, 3.0, 4.0)
    @test _same_as_base(RK._native_hvcat, hvcat, 2, [1.0 2.0], [3.0 4.0])
    @test _same_as_base(RK._native_hvcat, hvcat, (1, 1), x', [3.0 4.0])
    # Layouts `hvcat` rejects throw the companion's own errors, worded as Base's.
    M = [1.0 2.0; 3.0 4.0]
    @test_throws DimensionMismatch("mismatched height in block row 1 (expected 2, got 1)") RK._native_hvcat((2, 2), M, 1.0, 2.0, 3.0)
    @test_throws DimensionMismatch("block row 2 has mismatched number of columns (expected 3, got 2)") RK._native_hvcat((2, 2), M, x, 0.0, 1.0)
    # refused: block-row counts that do not describe the operands are a
    # malformed layout; Julia 1.12 rejects them, while Julia 1.10's `hvcat`
    # silently drops operands or returns an empty vector (dev §1: no silent
    # errors).
    for (rows, args) in (((), (x, 2.0)), ((0,), (x, 2.0)), ((0, 2), (x, 2.0)),
                         ((2, -1), (1.0, 2.0)), ((1,), ([1.0 2.0], [3.0 4.0])),
                         ((2,), ([1.0], 2.0, 3.0)), ((2, 2), (1.0, 2.0, 3.0)))
        @test_throws ArgumentError RK._native_hvcat(rows, args...)
    end
    # A fresh output: writing it never reaches an operand.
    a, b, c = [1.0, 2.0], [3.0, 4.0], [5.0 6.0; 7.0 8.0]
    for out in (RK._native_hcat(a, b), RK._native_vcat(a, b),
                RK._native_vcat(c, c), RK._native_hvcat((2, 1), a, b, c),
                RK._native_vcat(a, 1.0), RK._native_hvcat((2, 3), c, a, 0.0, 0.0, 1),
                RK._native_vcat(1.0, a, 2))
        out .= -1
    end
    @test a == [1.0, 2.0] && b == [3.0, 4.0] && c == [5.0 6.0; 7.0 8.0]
end

# Scalar and mixed literals as the native body lowers them: an inferred dense
# result, whichever generic method (SparseArrays, which Enzyme loads, claims
# Base's) would otherwise take the call.  Base's own mixed `hvcat` infers no
# concrete type.
_system(a, b) = RK._native_hvcat((2, 2), -a, 0, a, -b)
_affine(M, e) = RK._native_hvcat((2, 3), M, RK._native_vcat(e, 0.0), 0.0, 0.0, 1.0)
_row(a, M) = RK._native_hcat(a, M[:, 1]', 1)
_column(a, x) = RK._native_vcat(a, x, 1)

@testset "native scalar and mixed literals are inferred" begin
    M = [1.0 2.0; 3.0 4.0]
    @test @inferred(_system(0.5, 0.25)) == [-0.5 0; 0.5 -0.25]
    @test @inferred(_system(0.5f0, 0.25f0)) isa Matrix{Float32}
    @test @inferred(_affine(M, 0.7)) == [M [0.7, 0.0]; 0.0 0.0 1.0]
    @test @inferred(_column(0.5, [1.0, 2.0])) == [0.5, 1.0, 2.0, 1.0]
    @test _row(0.5, M) == hcat(0.5, M[:, 1]', 1)
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
@kernel literals(a, b, M) = begin
    result = ([-a 0; a -b], [M [a, b]; 0.0 0.0 1.0], [a; b; 1], [a b 1.0], [a M[1, :]'])
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
    pointwise = plate(x) do xi
        sum(vcat(c, [2xi]))
    end
    total = sum(pointwise)
end

@testset "native kernel bodies concatenate as Base" begin
    a, b = [1.0, 2.0], [3.0, 4.0]
    @test prepare(calls)(a, b) ==
          (hcat(a, b), vcat(a, b), hcat(a, b), hvcat((2, 2), a, b, b, a))
    @test prepare(brackets)(a, b) == ([a b], [a; b], [a b; b a], [[a b]; b a], [a b b])
    M = [5.0 6.0; 7.0 8.0]
    literal_values = prepare(literals)(0.5, 0.25, M)
    expected = ([-0.5 0; 0.5 -0.25], [M [0.5, 0.25]; 0.0 0.0 1.0], [0.5; 0.25; 1],
                [0.5 0.25 1.0], [0.5 M[1, :]'])
    @test literal_values == expected
    @test map(typeof, literal_values) == map(typeof, expected)
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
