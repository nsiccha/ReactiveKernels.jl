using ReactiveKernels
import Enzyme
using Test

struct _LowerWithOpsCoeffs{T}
    a21::T
    b1::T
    b2::T
end

# Straight-line Function-port stage kernel: the gap-repro shape (a kernel
# graph over an explicit RHS port plus a coefficient struct) that downstream
# lanes evaluate functionally inside their own differentiated loops.
@kernel _lower_with_ops_stage(f::Function, uprev::AbstractVector,
        k1::AbstractVector, p, t::Number, dt::Number,
        c::_LowerWithOpsCoeffs) = begin
    k2::AbstractVector = f(uprev .+ dt .* (c.a21 .* k1), p, t + dt)
    u::AbstractVector = uprev .+ dt .* (c.b1 .* k1 .+ c.b2 .* k2)
    return u
end

@kernel _lower_with_ops_plate(x::Vector{Float64}, s::Float64) = begin
    pointwise = plate(x, s) do xi, si
        y::Float64 = -0.5 * (xi / si)^2
        return y
    end
    return sum(pointwise)
end

const _LOWER_WITH_OPS_HAVE = (:f, :uprev, :k1, :p, :t, :dt, :c)

_bitwise_equal(a::AbstractVector{Float64}, b::AbstractVector{Float64}) =
    axes(a) == axes(b) && bitstring.(a) == bitstring.(b)

# Top-level functional product plus a top-level calling loop, exactly the
# downstream consumption shape: plain reverse-mode differentiation through
# repeated functional calls with a fresh per-call closure RHS.
const _LWO_F = (v, pp, tt) -> -0.5 .* v
const _LWO_C = _LowerWithOpsCoeffs(0.2, 0.6, 0.4)
const _LWO_TRIPLE = lower_with_ops(
    plan(_lower_with_ops_stage; have = _LOWER_WITH_OPS_HAVE, want = :u))
const _LWO_FUNCTIONAL = compile(first(_LWO_TRIPLE))
const _LWO_OPS = _LWO_TRIPLE[2]

function _lwo_looploss(x)
    u = copy(x)
    kk = _LWO_F(u, nothing, 0.0)
    for j in 1:3
        u = _LWO_FUNCTIONAL(_LWO_OPS, _LWO_F, u, kk, nothing, 0.1 * j, 0.1,
            _LWO_C)
        kk = _LWO_F(u, nothing, 0.1 * j)
    end
    return sum(u)
end

@testset "lower_with_ops" begin
    @testset "public triple contract" begin
        @test :lower_with_ops in names(ReactiveKernels)
        p = plan(_lower_with_ops_stage; have = _LOWER_WITH_OPS_HAVE, want = :u)
        ast, ops, recipes = lower_with_ops(p)
        @test ast == lower(p)
        @test ast.head === :function
        sig = ast.args[1]
        @test sig.args[1] === :__ops__
        portnames = map(sig.args[2:end]) do arg
            arg isa Symbol ? arg : arg.args[1]
        end
        @test Tuple(portnames) == _LOWER_WITH_OPS_HAVE
        @test ops isa Tuple
        @test recipes isa Tuple
        @test length(ops) == length(recipes)
        @test all(recipe -> recipe isa Recipe, recipes)
    end

    @testset "functional evaluation matches prepare bit-for-bit" begin
        p = plan(_lower_with_ops_stage; have = _LOWER_WITH_OPS_HAVE, want = :u)
        ast, ops, _ = lower_with_ops(p)
        prepared = prepare(p)
        values = (_LWO_F, [1.0, 2.0], [-0.5, -1.0], nothing, 0.0, 0.1, _LWO_C)
        reference = prepared(values...)
        @test _bitwise_equal(compile(ast)(ops, values...), reference)
        @test _bitwise_equal(
            Core.eval(@__MODULE__, ast)(ops, values...), reference)
    end

    @testset "keyword forms on an authored plate" begin
        p = plan(_lower_with_ops_plate)
        x = [1.0, 2.0, 3.0]
        s = 2.0
        reference = prepare(p)(x, s)
        ast, ops, _ = lower_with_ops(p)
        @test compile(ast)(ops, x, s) == reference
        inlined_ast, inlined_ops, _ =
            lower_with_ops(p; inline_embedded = true)
        @test compile(inlined_ast)(inlined_ops, x, s) == reference
        _, tensorized_ops, _ = lower_with_ops(p; tensorized = true)
        @test tensorized_ops == ops
        _, tensorized_inlined_ops, _ =
            lower_with_ops(p; tensorized = true, inline_embedded = true)
        @test tensorized_inlined_ops == inlined_ops
    end

    @testset "reverse-mode gradient through the functional form" begin
        uprev = [1.0, 2.0]
        grad = Enzyme.gradient(Enzyme.Reverse, _lwo_looploss, copy(uprev))[1]
        e = 1e-8
        fd = [(_lwo_looploss(uprev .+ e .* b) -
               _lwo_looploss(uprev .- e .* b)) / 2e
              for b in ([1.0, 0.0], [0.0, 1.0])]
        @test grad ≈ fd atol = 1e-6
    end
end
