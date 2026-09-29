using ReactiveKernels
using ReactiveKernels: derivative_cut, forward_cut, reverse_cut,
    stage_primal, stage_reverse, stage_residuals
using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme
using Test

module MultiOutputDerivativeRuleGraphs
using ReactiveKernels
value(a, b) = exp(a * b)
function joint(a, b)
    y = exp(a * b)
    return y, b * y, a * y
end
pullback(da, db, seed) = (seed * da, seed * db)

function make_graph(value_op = value, joint_op = joint, pullback_op = pullback)
    @kernel graph(a::Float64, b::Float64, adot::Float64, bdot::Float64,
            seed::Float64) = begin
        y::Float64 = value_op(a, b)
        (y, da::Float64, db::Float64) = joint_op(a, b)
        ydot::Float64 = da * adot + db * bdot
        (abar::Float64, bbar::Float64) = pullback_op(da, db, seed)
        return y, ydot, abar, bbar
    end
    return graph
end
make_rule(graph) = derivative_rule(graph; primal = :y,
    directions = (a = :adot, b = :bdot), tangent = :ydot,
    covector = :seed, cotangents = (a = :abar, b = :bbar))
function make_scalar_graph(value_op, joint_op)
    @kernel graph(a::Float64, b::Float64) = begin
        y::Float64 = value_op(a, b)
        (y, da::Float64, db::Float64) = joint_op(a, b)
        return y, da, db
    end
    return graph
end
const rule = make_rule(make_graph())
objective(x) = rule(x[1], x[2])
end

@testset "joint mathematical recipes in generated derivative rules" begin
    G = MultiOutputDerivativeRuleGraphs
    calls = zeros(Int, 3)
    value_op(a, b) = (calls[1] += 1; G.value(a, b))
    joint_op(a, b) = (calls[2] += 1; G.joint(a, b))
    pullback_op(a, b, seed) = (calls[3] += 1; G.pullback(a, b, seed))
    spec = G.make_graph(value_op, joint_op, pullback_op)
    rule = G.make_rule(spec)
    @test calls == [0, 0, 0]  # construction/planning never runs the math

    a, b, seed = .3, .7, 2.
    y, da, db = G.joint(a, b)
    @test rule(a, b) == y
    @test calls == [1, 0, 0]  # no partial-producing recipe on the primal path

    for mask in (1, 2, 3)
        fill!(calls, 0)
        primal, tape = stage_primal(rule, Val(mask), a, b)
        @test primal == y
        @test calls == [0, 1, 0]  # joint recipe once; no separate value pass
        @test stage_residuals(rule, Val(mask)) == (:da, :db)
        selected = Tuple((seed * da, seed * db)[i] for i in 1:2
            if (mask >> (i - 1)) & 1 == 1)
        @test stage_reverse(rule, Val(mask), tape, seed) == selected
        @test calls == [0, 1, 1]  # reverse contracts the saved partials
        @test reverse_cut(rule, Val(mask), a, b, seed) == selected
        @test calls == [0, 2, 2]
    end

    fill!(calls, 0)
    @test forward_cut(rule, a, b, 1., 2.) == (y, da + 2db)
    @test calls == [0, 1, 0]

    # The same graph still uses ordinary bound preparation: frozen work runs
    # once, and a new binding recomputes from the changed mathematical inputs.
    fill!(calls, 0)
    frozen = prepare(spec; have = (:a, :b), want = :y, bound = (; a, b))
    @test calls == [1, 0, 0]
    for _ in 1:3
        @test Base.invokelatest(frozen) == y
    end
    @test calls == [1, 0, 0]
    rebound = prepare(spec; have = (:a, :b), want = :y, bound = (; a = a + 1, b))
    @test Base.invokelatest(rebound) == G.value(a + 1, b)
    @test calls == [2, 0, 0]

    # The scalar generator shares the same encoding/lowering boundary.
    scalar = scalar_derivative_rule(G.make_scalar_graph(value_op, joint_op);
        primal = :y, partials = (a = :da, b = :db))
    @test scalar(a, b) == y
    @test derivative_cut(scalar, (true, true), a, b) == (y, da, db)

    # Ordinary reverse AD reaches the generated adapter. The instrumented
    # call counters above are not part of this pure mathematical graph.
    @test gradient(G.objective, AutoEnzyme(; mode = Enzyme.Reverse), [a, b]) ≈ [da, db]
    @test @inferred(G.rule(a, b)) == y
    @test @inferred(stage_primal(G.rule, Val(3), a, b))[1] == y
end
