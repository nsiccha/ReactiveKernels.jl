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
const _LWO_CONSTANTS = _LWO_TRIPLE[2]
const _LWO_CALLABLES = _LWO_TRIPLE[3]

# Public generated breadth for the construction-cost regression: a model with
# many independent authored plates and selected sum consumers, without relying
# on any application model or retained corpus. This is deliberately generated
# here so increasing the breadth does not leave a hand-written wall of recipes.
let
    body = Expr(:block)
    totals = Symbol[]
    for index in 1:64
        pointwise = Symbol(:pointwise_, index)
        transformed = Symbol(:transformed_, index)
        cell = Symbol(:cell_, index)
        total = Symbol(:total_, index)
        push!(body.args, :($pointwise = plate(data, q) do value, parameter
            $transformed::Float64 = log(value) + $index
            $cell::Float64 = $transformed * parameter
            $cell
        end))
        push!(body.args, :($total::Float64 = sum($pointwise)))
        push!(totals, total)
    end
    push!(body.args, :(return $(Expr(:call, :+, totals...))))
    definition = :(_lower_with_ops_many_plates(
        q::Float64, data::Vector{Float64}) = $body)
    Core.eval(@__MODULE__, Expr(:macrocall, Symbol("@kernel"),
                                LineNumberNode(@__LINE__, @__FILE__), definition))
end

function _lwo_looploss(x)
    u = copy(x)
    kk = _LWO_F(u, nothing, 0.0)
    for j in 1:3
        u = _LWO_FUNCTIONAL(_LWO_CONSTANTS, _LWO_CALLABLES, _LWO_F, u, kk, nothing,
            0.1 * j, 0.1, _LWO_C)
        kk = _LWO_F(u, nothing, 0.1 * j)
    end
    return sum(u)
end

@kernel _lower_with_ops_literals(x::Float64) = begin
    shift::Float64 = -1.5
    scale::Float64 = 0.75
    label::Symbol = :shifted
    y::Float64 = scale * (x + shift)
    return (y, label)
end

@testset "lower_with_ops" begin
    @testset "literal statements run as their literal" begin
        kernel = prepare(_lower_with_ops_literals)
        @test kernel(2.0) == (0.75 * (2.0 - 1.5), :shifted)
        statements = filter(s -> s isa Expr && s.head === :(=), code_expr(kernel).args[2].args)
        # Each literal recipe's statement is its literal, and its tables hold
        # nothing for it: no generated function carries `-1.5`.
        for (name, literal) in ((:shift, -1.5), (:scale, 0.75), (:label, QuoteNode(:shifted)))
            statement = only(filter(s -> s.args[1] === name, statements))
            @test statement.args[2] == literal
            slot = findfirst(r -> only(r.outputs).name === name, kernel.lowered_recipes)
            @test kernel.callables[slot] === nothing
            @test kernel.constants[slot] === nothing
        end
        y_statement = only(filter(s -> s.args[1] === :y, statements))
        slot, args = ReactiveKernels._inline_statement_call(y_statement.args[2])
        @test kernel.lowered_recipes[slot] === kernel.lowered_recipes[findfirst(
            r -> only(r.outputs).name === :y, kernel.lowered_recipes)]
        @test Set(args) == Set(Any[:scale, :x, :shift])
    end

    @testset "public tables contract" begin
        @test :lower_with_ops in names(ReactiveKernels)
        p = plan(_lower_with_ops_stage; have = _LOWER_WITH_OPS_HAVE, want = :u)
        ast, constants, callables, recipes = lower_with_ops(p)
        @test ast == lower(p)
        @test ast.head === :function
        sig = ast.args[1]
        @test sig.args[1] === :__constants__
        @test sig.args[2] === :__callables__
        portnames = map(sig.args[3:end]) do arg
            arg isa Symbol ? arg : arg.args[1]
        end
        @test Tuple(portnames) == _LOWER_WITH_OPS_HAVE
        @test constants isa Tuple
        @test callables isa Tuple
        @test recipes isa Tuple
        @test length(constants) == length(callables) == length(recipes)
        @test all(recipe -> recipe isa Recipe, recipes)
        # The plan binds nothing, so the constants table is empty of values.
        @test all(isnothing, constants)
    end

    @testset "functional evaluation matches prepare bit-for-bit" begin
        p = plan(_lower_with_ops_stage; have = _LOWER_WITH_OPS_HAVE, want = :u)
        ast, constants, callables, _ = lower_with_ops(p)
        prepared = prepare(p)
        values = (_LWO_F, [1.0, 2.0], [-0.5, -1.0], nothing, 0.0, 0.1, _LWO_C)
        reference = prepared(values...)
        @test _bitwise_equal(compile(ast)(constants, callables, values...), reference)
        # Julia 1.12+ enforces the world age of bindings created by `Core.eval`;
        # call the freshly eval'd function in the latest world.
        @test _bitwise_equal(
            Base.invokelatest(Core.eval(@__MODULE__, ast), constants, callables, values...),
            reference)
    end

    @testset "keyword forms on an authored plate" begin
        p = plan(_lower_with_ops_plate)
        x = [1.0, 2.0, 3.0]
        s = 2.0
        reference = prepare(p)(x, s)
        ast, constants, callables, _ = lower_with_ops(p)
        @test compile(ast)(constants, callables, x, s) == reference
        inlined_ast, inlined_constants, inlined_callables, _ =
            lower_with_ops(p; inline_embedded = true)
        @test compile(inlined_ast)(inlined_constants, inlined_callables, x, s) == reference
    end

    @testset "generated breadth of authored plate consumers" begin
        p = plan(_lower_with_ops_many_plates)
        @test count(recipe -> recipe.op isa ReactiveKernels._AuthoredPlateOp,
                    p.recipes) == 64
        q, data = 2.0, [1.0, 2.0, 4.0]
        reference = sum(sum((log.(data) .+ index) .* q) for index in 1:64)
        prepared = prepare(p; bound = p.have[2] => data)
        @test prepared(q) == reference
        ast, constants, callables, recipes = lower_with_ops(p)
        @test compile(ast)(constants, callables, q, data) == reference
        @test length(constants) == length(callables) == length(recipes)
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
