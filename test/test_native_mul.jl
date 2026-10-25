module NativeMulTests
using ReactiveKernels, LinearAlgebra, Test
const RK = ReactiveKernels

# A lazily indexed vector, as RKPPL's compact-cell overlay is: its views and
# their reshapes are not strided, so Base's `*` of them runs LinearAlgebra's
# generic `mul!`. It opts in to the companion's loop.
struct LazyVector{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(v::LazyVector) = size(v.data)
Base.IndexStyle(::Type{<:LazyVector}) = IndexLinear()
Base.getindex(v::LazyVector, i::Int) = v.data[i]
RK._native_mul_operand(::LazyVector) = true
struct Unopted <: AbstractVector{Float64} end
Base.size(::Unopted) = (2,)
Base.getindex(::Unopted, i::Int) = Float64(i)

_values(T, n, offset) = T === Bool ? Bool[isodd(i + offset) for i in 1:n] :
    T[T(i + offset) for i in 1:n]
_matrix(T, m, n, offset = 0) = reshape(_values(T, m * n, offset), m, n)

# Same value and type as Base, or the same exception.
function _same_as_base(args...)
    expected = try
        *(args...)
    catch err
        err
    end
    actual = try
        RK._native_mul(args...)
    catch err
        err
    end
    expected isa Exception &&
        return typeof(actual) == typeof(expected) &&
               sprint(showerror, actual) == sprint(showerror, expected)
    typeof(actual) == typeof(expected) && size(actual) == size(expected) &&
        isequal(actual, expected)
end

_bytes(args...) = (RK._native_mul(args...); @allocated RK._native_mul(args...))

@testset "native matrix product follows Base" begin
    for T in (Float64, Float32, Int, Bool)
        u = LazyVector(_values(T, 24, 0))
        Z = reshape(view(u, 1:12), 6, 2)
        row = view(Z, 3:3, :)
        products = ((Z, _matrix(T, 2, 3)), (row, transpose(_matrix(T, 3, 2, 1))),
                    (row, _matrix(T, 2, 2)'), (Z, _values(T, 2, 5)),
                    (_matrix(T, 3, 6), Z), (transpose(Z), _matrix(T, 6, 2)),
                    (Z', _values(T, 6, 2)), (_matrix(T, 2, 2), view(u, 13:14)),
                    (Z, view(u, 13:14)), (Z, reshape(view(u, 13:16), 2, 2)),
                    (reshape(view(u, 1:0), 0, 2), _matrix(T, 2, 3)),
                    (reshape(view(u, 1:0), 3, 0), _matrix(T, 0, 2)),
                    (reshape(view(u, 1:0), 3, 0), view(u, 1:0)))
        for args in products
            @test _same_as_base(args...)
            # The companion's own loop allocates only its result, where
            # Julia 1.10's generic `mul!` allocates two 10 KB tile buffers.
            @test _bytes(args...) < 1024
        end
        # A mismatched shared dimension throws Base's error.
        @test _same_as_base(Z, _matrix(T, 3, 3))
        @test _same_as_base(Z, _values(T, 3, 0))
        @test u.data == _values(T, 24, 0)
    end
    # Mixed element types promote as Base's product does.
    u = LazyVector(_values(Float64, 12, 0))
    Z = reshape(view(u, 1:6), 3, 2)
    for args in ((Z, [1 2; 3 4]), (Z, [true, false]), ([1 2; 3 4; 5 6], view(u, 1:2)),
                 (Z, Float32[1 2; 3 4]), (Int8[1 2; 3 4], reshape(view(u, 1:4), 2, 2)))
        @test _same_as_base(args...)
    end
    # Operand lists the companion does not compute keep Base's call: two
    # strided operands (BLAS, or Base's own methods for them), structured,
    # row-vector, non-isbits, complex and offset-axes operands, three factors.
    lazy = LazyVector(ComplexF64[1 + 2im, 3 - im, 2im, 4])
    for args in (([1.0 2.0; 3.0 4.0], [5.0, 6.0]), ([1 2; 3 4], [5 6; 7 8]),
                 (view([1.0 2.0; 3.0 4.0], 1:1, :), [5.0 6.0; 7.0 8.0]),
                 (Diagonal([1.0, 2.0]), reshape(view(u, 1:4), 2, 2)),
                 (reshape(view(u, 1:4), 2, 2), Diagonal([1.0, 2.0])),
                 (LowerTriangular([1.0 0.0; 2.0 3.0]), view(u, 1:2)),
                 (transpose(view(u, 1:3)), Z),
                 (reshape(view(u, 1:3), 3, 1), transpose(view(u, 4:6))),
                 (reshape(view(lazy, 1:4), 2, 2), [1.0, 2.0]),
                 (reshape(view(LazyVector(big.(collect(1:4))), 1:4), 2, 2), [1, 2]),
                 (view(Z, Base.IdentityUnitRange(2:3), :), [1.0, 2.0]),
                 (Z, [1.0 2.0; 3.0 4.0], [1.0, 2.0]), (2.0, Z), (Z, 2.0), (2.0, 3.0))
        @test _same_as_base(args...)
    end
    # A lazy array that does not opt in keeps Base's call.
    @test !RK._native_mul_operand(view(Unopted(), 1:2))
    @test _same_as_base(reshape(view(Unopted(), 1:2), 2, 1), [1.0 2.0])
    # A fresh result: writing it never reaches an operand.
    out = RK._native_mul(Z, [1.0 2.0; 3.0 4.0])
    out .= 0
    @test u.data == _values(Float64, 12, 0)
end

# The native body calls the companion for Base's `*`, by any resolved
# spelling; a local named `*` keeps its own meaning.
@testset "native bodies call the product companion" begin
    body = RK._kernel_native_body(:(A * B), @__MODULE__, Set([:A, :B]))
    @test body == Expr(:call, GlobalRef(RK, :_native_mul), :A, :B)
    qualified = RK._kernel_native_body(:(Base.:*(A, B)), @__MODULE__, Set([:A, :B]))
    @test qualified == Expr(:call, GlobalRef(RK, :_native_mul), :A, :B)
    shadowed = RK._kernel_native_body(:(A * B), @__MODULE__, Set([:A, :B, :*]))
    @test shadowed == :(A * B)
end
end
