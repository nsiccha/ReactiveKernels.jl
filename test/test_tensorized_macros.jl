module TensorizedMacroTests

using ReactiveKernels, Test
const RK = ReactiveKernels

# A macro in a recipe reads the syntax of its arguments, so the tensorized
# companion lowers the macro's EXPANSION rather than rewriting its arguments
# first. Evaluating the companion on host values must equal evaluating the
# authored expression (the `test_tensorized_vect.jl` idiom).
tensorized(body) = eval(Expr(:->, :(v, m, s),
    RK._kernel_tensorized_rhs(body, Set([:v, :m, :s]), @__MODULE__)))
native(body) = eval(Expr(:->, :(v, m, s), body))

const V = [2.0, 3.0, 5.0]
const M = [1.0 2.0; 3.0 4.0]
const S = 2.0

@testset "macro recipes evaluate like the original natively" begin
    bodies = Any[
        :(@.(100 * (v - v[1]) / v[1])),
        :(@.(2 * v + 1)),
        :(@.(100 * (v - s) / s)),
        :(v .+ @.(2 * v)),
        :(@. sqrt(abs(v - @inbounds(v[1])) + 1)),
        # Elementwise, never the matrix product or the matrix exponential.
        :(@.(m * m)),
        :(@.(exp(m))),
        :(sum(@view v[1:2])),
        :(sum(@view v[2:end])),
        :(@views sum(v[1:2]) + v[3]),
        :(@inbounds v[1] + v[2]),
        :(@fastmath v[1] * v[2] + v[3]),
        :(Base.@evalpoly(v[1], 1.0, 2.0, 3.0)),
    ]
    for body in bodies
        @test Base.invokelatest(tensorized(body), V, M, S) ==
            Base.invokelatest(native(body), V, M, S)
    end
    @test Base.invokelatest(tensorized(:(@.(m * m))), V, M, S) == M .* M
    @test Base.invokelatest(tensorized(:(@.(m * m))), V, M, S) != M * M
end

@testset "the broadcast macro lowers to the broadcast companion" begin
    known = Set([:v])
    lowered = RK._kernel_tensorized_rhs(
        :(@.(100 * (v - v[1]) / v[1])), known, @__MODULE__)
    explicit = RK._kernel_tensorized_rhs(
        :(100 .* (v .- v[1]) ./ v[1]), known, @__MODULE__)
    @test lowered isa Expr && lowered.head === :call
    # Only the outermost dotted call materializes; nested ones stay lazy.
    @test lowered.args[1] == GlobalRef(RK, :_tensorized_broadcast)
    inner = lowered.args[3]
    @test inner.args[1] == GlobalRef(RK, :_tensorized_lazy_broadcast)
    # The indexed read stays a scalar gather, as in the explicit spelling.
    @test lowered.args[end] ==
        Expr(:call, GlobalRef(RK, :_tensorized_getindex), :v, 1)
    @test explicit.args[1] == GlobalRef(RK, :_tensorized_broadcast)
    @test explicit.args[end] == lowered.args[end]
    # No macro call survives the lowering.
    has_macro(ex) = ex isa Expr &&
        (ex.head === :macrocall || any(has_macro, ex.args))
    @test !has_macro(lowered)
end

@kernel change_from_baseline(v::Vector{Float64}) = begin
    out::Vector{Float64} = @.(100 * (v - v[1]) / v[1])
    return out
end

@kernel viewed_head(v::Vector{Float64}) = begin
    out::Float64 = sum(@view v[1:2])
    return out
end

@kernel macro_cell(v::Vector{Float64}, s::Float64) = begin
    pointwise = plate(v, s) do x, scale
        @.(scale * (x - 1))
    end
    total::Float64 = sum(pointwise)
    return total
end

source_ops(spec) =
    [r.op for r in spec.graph.recipes if r.op isa RK._KernelSourceOp]

@testset "authored kernels carry matching bodies" begin
    expected = 100 .* (V .- V[1]) ./ V[1]
    @test prepare(change_from_baseline)(V) == expected
    op = only(source_ops(change_from_baseline))
    @test op.f(V) == expected
    @test op.tensor_f(V) == expected

    # `@view` in a recipe defines, and both bodies read the same window.
    @test prepare(viewed_head)(V) == 5.0
    viewed = only(source_ops(viewed_head))
    @test viewed.f(V) == viewed.tensor_f(V) == 5.0

    @test prepare(macro_cell)(V, S) == sum(S .* (V .- 1))
end

end # module
