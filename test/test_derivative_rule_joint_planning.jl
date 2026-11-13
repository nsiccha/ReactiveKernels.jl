module JointDerivativePlanningTests
using ReactiveKernels, Test
using ReactiveKernels: stage_primal, stage_reverse, stage_residuals, reverse_cut
import Enzyme

value(x) = x*x
joint(x) = (x*x, 2x)
direct(x, seed) = 2x*seed
contract(partial, seed) = partial*seed

function graph(value_op=value, joint_op=joint, direct_op=direct, contract_op=contract;
               joint_cost=5)
    @kernel spec(x::Float64, seed::Float64) = begin
        @recipe (cost=4) y::Float64 = value_op(x)
        @recipe (cost=joint_cost) (y, partial::Float64) = joint_op(x)
        @recipe (cost=4) dx::Float64 = direct_op(x, seed)
        @recipe (cost=1) dx = contract_op(partial, seed)
        return y, dx
    end
    spec
end
make_rule(spec) = derivative_rule(spec; primal=:y, covector=:seed,
    cotangents=(x=:dx,), name=:joint_planning)
const pure_rule = make_rule(graph())
const pure_primal = prepare(graph(); have=(:x,), want=:y)

@testset "reverse staging selects the joint recipe first" begin
    calls = zeros(Int, 4)
    value_op(x) = (calls[1] += 1; value(x))
    joint_op(x) = (calls[2] += 1; joint(x))
    direct_op(x, seed) = (calls[3] += 1; direct(x, seed))
    contract_op(partial, seed) = (calls[4] += 1; contract(partial, seed))
    for (joint_cost, expected, residuals) in
            ((5, [0,1,0,1], (:partial,)), (10, [1,0,1,0], (:x,)))
        fill!(calls, 0)
        spec = graph(value_op, joint_op, direct_op, contract_op; joint_cost)
        rule = make_rule(spec)
        @test calls == [0,0,0,0] # planning does not execute the recipes
        @test plan(spec; have=(:x,:seed), want=:dx).cost == 4
        joint_plan = plan(spec; have=(:x,:seed), want=(:y,:dx))
        @test joint_plan.cost == min(joint_cost+1, 8)
        @test stage_residuals(rule, Val(1)) == residuals
        for x in (-3.0, 0.0, 0.5, 4.0), seed in (-2.0, 0.0, 1.0, 3.0)
            fill!(calls, 0)
            y, tape = stage_primal(rule, Val(1), x)
            @test y == x*x
            @test stage_reverse(rule, Val(1), tape, seed) == (2x*seed,)
            @test calls == expected
            @test sum(calls .* [4,joint_cost,4,1]) == joint_plan.cost
        end
        # A primal-only call and a self-contained reverse keep their own cuts.
        fill!(calls, 0)
        @test rule(3.0) == 9.0
        @test calls == [1,0,0,0]
        fill!(calls, 0)
        @test reverse_cut(rule, Val(1), 3.0, 2.0) == (12.0,)
        @test calls == [0,0,1,0]
    end
end

@testset "joint staging keeps independent tapes and ordinary AD" begin
    first_value, first_tape = @inferred stage_primal(pure_rule, Val(1), 2.0)
    second_value, second_tape = @inferred stage_primal(pure_rule, Val(1), -3.0)
    @test (first_value,second_value) == (4.0,9.0)
    @test @inferred(stage_reverse(pure_rule, Val(1), second_tape, 2.0)) == (-12.0,)
    @test @inferred(stage_reverse(pure_rule, Val(1), first_tape, 3.0)) == (12.0,)
    for x in (-3.0, 0.0, 0.5, 4.0)
        result = Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(pure_rule),
                                Enzyme.Active, Enzyme.Active(x))
        @test result[2] == x*x
        @test result[1][1] == 2x
        ordinary = Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(pure_primal),
                                  Enzyme.Active, Enzyme.Active(x))
        @test ordinary == result
    end
end

# This equivalent primal recipe mentions the seed, which is unavailable in
# augmented_primal. Joint planning must not require that early seed or reject
# the existing graph: its ordinary seed-independent primal remains valid.
seeded_value(x, seed) = x*x
@kernel seeded_alternative(x::Float64, seed::Float64) = begin
    @recipe (cost=4) y::Float64 = value(x)
    @recipe (cost=0) y = seeded_value(x, seed)
    @recipe (cost=4) dx::Float64 = direct(x, seed)
    return y, dx
end
@testset "joint alternatives preserve stage availability" begin
    rule = make_rule(seeded_alternative)
    for x in (-3.0, 0.0, 2.0)
        y, tape = stage_primal(rule, Val(1), x)
        @test y == x*x
        @test stage_reverse(rule, Val(1), tape, 2.0) == (4x,)
    end
end
end
