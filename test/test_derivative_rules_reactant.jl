# Derivative rules under Reactant (src/derivative_rules.jl,
# ext/ReactiveKernelsReactantExt.jl). Every cut of a generated rule is plain
# graph mathematics, so the callable (primal cut), the scalar partial cuts and
# the vector forward/reverse cuts trace into a compiled program like any other
# kernel call, and a rule passes through tracing as a compile-time constant.
# RK emits no custom rule into EnzymeMLIR (that waits on the upstream
# custom-rule bridge, docs/src/manual-derivative-rules.md § Backend boundary):
# Enzyme under Reactant differentiates the primal cut's traced operations.
using LinearAlgebra: dot
using Reactant
import Enzyme
using ReactiveKernels
using ReactiveKernels: derivative_cut
using Test

module DerivativeRuleReactantGraphs
using ReactiveKernels

@kernel two_input_graph(a::Float64, b::Float64) = begin
    s::Float64 = a + b
    e::Float64 = exp(a * b)
    y::Float64 = e + log(s)
    dy_da::Float64 = b * e + 1 / s
    dy_db::Float64 = a * e + 1 / s
    return y, dy_da, dy_db
end

@kernel matvec_rule(
        A::Matrix{Float64}, x::Vector{Float64},
        A_dot::Matrix{Float64}, x_dot::Vector{Float64},
        y_bar::Vector{Float64}) = begin
    y::Vector{Float64} = A * x
    A_direction::Vector{Float64} = A_dot * x
    x_direction::Vector{Float64} = A * x_dot
    y_dot::Vector{Float64} = A_direction + x_direction
    A_bar::Matrix{Float64} = y_bar * transpose(x)
    x_bar::Vector{Float64} = transpose(A) * y_bar
    return y, y_dot, A_bar, x_bar
end

const two_input = scalar_derivative_rule(
    two_input_graph; primal = :y, partials = (a = :dy_da, b = :dy_db),
    name = :two_input)
const matvec = derivative_rule(matvec_rule; primal = :y,
    directions = (A = :A_dot, x = :x_dot), tangent = :y_dot,
    covector = :y_bar, cotangents = (A = :A_bar, x = :x_bar), name = :matvec)
end # module

const _DRR = DerivativeRuleReactantGraphs
_rnum(x) = Reactant.ConcreteRNumber(x)
_rarr(x) = Reactant.to_rarray(x)

@testset "scalar rule cuts trace under Reactant" begin
    rule = _DRR.two_input
    a, b = 0.3, 1.7
    ra, rb = _rnum(a), _rnum(b)

    # The rule as an argument: a compile-time constant, not a traced value.
    call(r, a, b) = r(a, b)
    primal = Reactant.@compile call(rule, ra, rb)
    @test Float64(primal(rule, ra, rb)) ≈ rule(a, b)

    cut_both(r, a, b) = derivative_cut(r, (true, true), a, b)
    cut_a(r, a, b) = derivative_cut(r, (true, false), a, b)
    both = Reactant.@compile cut_both(rule, ra, rb)
    only_a = Reactant.@compile cut_a(rule, ra, rb)
    @test all(map(Float64, both(rule, ra, rb)) .≈
        derivative_cut(rule, (true, true), a, b))
    @test all(map(Float64, only_a(rule, ra, rb)) .≈
        derivative_cut(rule, (true, false), a, b))

    # A rule captured by a closure (a published global constant) traces the
    # same way.
    captured = (a, b) -> _DRR.two_input(a, b)
    @test Float64((Reactant.@compile captured(ra, rb))(ra, rb)) ≈ rule(a, b)
end

@testset "vector rule cuts trace under Reactant" begin
    rule = _DRR.matvec
    A = [1.0 2.0; 3.0 4.0]; x = [0.5, -1.0]
    A_dot = [0.1 -0.2; 0.3 0.4]; x_dot = [-0.7, 0.2]; y_bar = [1.2, -0.4]
    rA, rx, rA_dot, rx_dot, ry_bar = map(_rarr, (A, x, A_dot, x_dot, y_bar))

    call(r, A, x) = r(A, x)
    @test Array((Reactant.@compile call(rule, rA, rx))(rule, rA, rx)) ≈ A * x

    fwd(r, A, x, A_dot, x_dot) = forward_cut(r, A, x, A_dot, x_dot)
    y, y_dot = (Reactant.@compile fwd(rule, rA, rx, rA_dot, rx_dot))(
        rule, rA, rx, rA_dot, rx_dot)
    @test Array(y) ≈ A * x
    @test Array(y_dot) ≈ A_dot * x + A * x_dot

    rev3(r, A, x, y_bar) = reverse_cut(r, Val(3), A, x, y_bar)
    A_bar, x_bar = (Reactant.@compile rev3(rule, rA, rx, ry_bar))(
        rule, rA, rx, ry_bar)
    @test Array(A_bar) ≈ y_bar * transpose(x)
    @test Array(x_bar) ≈ transpose(A) * y_bar
    @test dot(y_bar, Array(y_dot)) ≈
        dot(Array(A_bar), A_dot) + dot(Array(x_bar), x_dot)

    # An x-only reverse cut reads A alone; the unread input may be `nothing`.
    rev2(r, A, y_bar) = reverse_cut(r, Val(2), A, nothing, y_bar)
    (x_only,) = (Reactant.@compile rev2(rule, rA, ry_bar))(rule, rA, ry_bar)
    @test Array(x_only) ≈ transpose(A) * y_bar
end

@testset "Enzyme under Reactant differentiates the primal cut" begin
    # No custom rule reaches EnzymeMLIR; the gradient comes from the traced
    # primal operations and agrees with the graph's authored partials.
    rule = _DRR.two_input
    v = [0.3, 1.7]
    loss(r, v) = Reactant.@allowscalar r(v[1], v[2])
    grad(r, v) = Enzyme.gradient(Enzyme.Reverse, loss, Enzyme.Const(r), v)[2]
    rv = _rarr(v)
    g = Array((Reactant.@compile grad(rule, rv))(rule, rv))
    _, dy_da, dy_db = derivative_cut(rule, (true, true), v[1], v[2])
    @test g ≈ [dy_da, dy_db]
end
