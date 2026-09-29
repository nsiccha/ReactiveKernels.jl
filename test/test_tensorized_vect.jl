module TensorizedVectTests

using ReactiveKernels, Test
const RK = ReactiveKernels

# Scalar-vector literals lower to the `_tensorized_vect` companion, which
# reduces to `Base.vect` natively; typed `Float64[...]` literals keep their
# `:ref` shape and route through `_tensorized_getindex`, which reduces to
# `getindex` natively. Evaluating the rewrite must equal evaluating the
# original (the `test_authoring.jl` tensorized-companion idiom).
tensorized(body) = eval(Expr(:->, :(a, b, c, i),
    RK._kernel_tensorized_rhs(body, Set([:a, :b, :c, :i]))))
native(body) = eval(Expr(:->, :(a, b, c, i), body))

@testset "literal rewrites evaluate like the original natively" begin
    a, b, c = 0.5, 5.0, 3.0
    i = [1, 3, 2]
    bodies = Any[
        :([a, b, c]),
        :([a, b, c][i]),
        :([a, b, c][2]),
        :([a]),
        :([a][1]),
        Expr(:vect),
        :([a, 1, c]),
        :([a, b]),
        :([[a, b], [c]]),
        :(Float64[a, b, c]),
        :(Float64[a, b, c][i]),
        :(Float32[a, b, c]),
        :(Any[a, b]),
        :(Base.vect(a, b, c)),
        :(Base.vect(a, b, c)[i]),
        # A computed array evaluates once; endpoints read from the value.
        :(([a, b, c])[end]),
        :(([a, b, c])[1:end]),
    ]
    for body in bodies
        @test Base.invokelatest(tensorized(body), a, b, c, i) ==
            Base.invokelatest(native(body), a, b, c, i)
    end
    # Exact integer conversion still throws on both sides.
    for f in (tensorized, native)
        @test_throws InexactError Base.invokelatest(
            f(:(Int[a, b, c])), a, b, c, i)
    end
end

@testset "vect rewrite targets the companion" begin
    known = Set([:a, :b, :c])
    vect_call = RK._kernel_tensorized_rhs(:([a, b, c]), known)
    @test vect_call isa Expr && vect_call.head === :call
    @test vect_call.args[1] == GlobalRef(RK, :_tensorized_vect)
    @test vect_call.args[2:end] == [:a, :b, :c]
    explicit = RK._kernel_tensorized_rhs(:(vect(a, b, c)), known)
    @test explicit.args[1] == GlobalRef(RK, :_tensorized_vect)
    # A dotted `Base.vect` callee passes through untouched, like every
    # other companion's dotted spelling.
    dotted = :(Base.vect)
    qualified = RK._kernel_tensorized_rhs(Expr(:call, dotted, :a, :b, :c), known)
    @test qualified.args[1] == dotted
    @test qualified.args[2:end] == [:a, :b, :c]
    typed = RK._kernel_tensorized_rhs(:(Float64[a, b, c]), known)
    @test typed.args[1] == GlobalRef(RK, :_tensorized_getindex)
    @test typed.args[2] === :Float64
    # A graph port named `vect` keeps its port meaning.
    shadowed = RK._kernel_tensorized_rhs(
        :(vect(a, b, c)), Set([:a, :b, :c, :vect]))
    @test shadowed.args[1] === :vect
end

@testset "the vect companion reduces to Base.vect natively" begin
    @test RK._tensorized_vect(0.5, 5.0, 3.0) == [0.5, 5.0, 3.0]
    @test eltype(RK._tensorized_vect(0.5, 5.0, 3.0)) === Float64
    @test isempty(RK._tensorized_vect())
    @test eltype(RK._tensorized_vect()) === Any
    @test RK._tensorized_vect(0.5) == [0.5]
    @test RK._tensorized_vect(0.5, 5) == [0.5, 5.0]
    @test eltype(RK._tensorized_vect(0.5, 5)) === Float64
    @test RK._tensorized_vect([1.0], [2.0]) == [[1.0], [2.0]]
    @test RK._tensorized_vect_eltype(2.5) === Float64
    @test RK._tensorized_vect_eltype(2) === Int
    @test RK._tensorized_vect_construct(Float64, (0.5, 5)) == [0.5, 5.0]
    @test eltype(RK._tensorized_vect_construct(Float64, (0.5, 5))) === Float64
    @test RK._tensorized_vect_construct(Float32, (0.5, 1.25)) ==
        Float32[0.5, 1.25]
end

@kernel _gather_fused(s1::Float64, s2::Float64, s3::Float64, i) = begin
    out = [s1, s2, s3][i]
    return out
end

@kernel _gather_steps(s1::Float64, s2::Float64, s3::Float64, i) = begin
    v = [s1, s2, s3]
    out = v[i]
    return out
end

@kernel _gather_typed(s1::Float64, s2::Float64, s3::Float64, i) = begin
    out = Float64[s1, s2, s3][i]
    return out
end

@kernel _construct_typed(s1::Float64, s2::Float64, s3::Float64) = begin
    out = Float64[s1, s2, s3]
    return out
end

@testset "literal-gather kernels run natively" begin
    s1, s2, s3 = 0.5, 5.0, 3.0
    i = [1, 3, 2, 1, 2, 3]
    expect = [s1, s2, s3][i]
    @test prepare(_gather_fused; want = :out)(s1, s2, s3, i) == expect
    @test prepare(_gather_steps; want = :out)(s1, s2, s3, i) == expect
    @test prepare(_gather_typed; want = :out)(s1, s2, s3, i) == expect
    @test prepare(_gather_fused; want = :out)(s1, s2, s3, 2) == s2
    @test prepare(_construct_typed; want = :out)(s1, s2, s3) == [s1, s2, s3]
    @test eltype(prepare(_construct_typed; want = :out)(s1, s2, s3)) ===
        Float64
end

end
