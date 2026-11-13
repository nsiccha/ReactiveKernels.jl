using ReactiveKernels, Test, LinearAlgebra
import Enzyme
include(joinpath(@__DIR__, "derivative_snapshot_borrowing.jl"))
ResidualBorrowPrototype.install!()

@testset "borrowed snapshots retain ownership and concrete audits" begin
    include(joinpath(@__DIR__, "..", "test", "test_derivative_rule_snapshots.jl"))
    r = ResidualBorrowPrototype.retain
    a = [1.0, 2.0]
    tape = (a, a, 1.5)
    borrowed = @inferred r((1, -1, -1), tape, 1.0, (false, false))
    @test borrowed[1] === a
    @test borrowed[2] === a
    @test borrowed[3] === 1.5
    for (primal, flags) in ((1.0, (false,true)), (a, (false,false)))
        saved = @inferred r((1, -1, -1), tape, primal, flags)
        @test saved[1] !== a
        @test saved[2] === saved[1]
        @test saved[3] === 1.5
    end
    nested = [a]
    saved = @inferred r((1,), (nested,), 1.0, (false,false))
    @test saved[1] !== nested
    @test only(saved[1]) !== a
    # Audit the owned specialization directly as well as its caller. A Base
    # boundary must not hide nonconcrete code introduced by this prototype.
    env = ResidualBorrowPrototype.Environment(tape, (1,-1,-1),
        (false,false), true)
    findings = ReactiveKernels._level_code_findings(:all, Base.deepcopy_internal,
        (typeof(env), IdDict{Any,Any}))
    println("OWNED_SNAPSHOT_HOOK_AUDIT ", findings)
    @test isempty(findings)
    record = DerivativeRuleSnapshotTests.Record(a)
    for value in (view(a,:), (a,), (;data=a), record,
            DerivativeRuleSnapshotTests.MutableRecord(a), [a])
        @test !ResidualBorrowPrototype.flat_environment((a,value))
        saved = r((1,-1), (a,value), 1.0, (false,false))
        @test saved[1] !== a
        if value isa SubArray
            @test parent(saved[2]) === saved[1]
        else
            @test DerivativeRuleSnapshotTests.unwrap(saved[2]) === saved[1]
        end
        # Unsupported borrowing forms retain the original snapshot path. Audit
        # that reachable path, rather than a hook it never calls.
        @test isempty(ReactiveKernels._level_code_findings(:all,
            r, (Tuple{Int,Int}, typeof((a,value)), Float64, Tuple{Bool,Bool})))
    end
end

module BorrowedMutationChecks
using ReactiveKernels, Test, LinearAlgebra
import Enzyme
const global_matrix = zeros(4, 2)
native(A,b,x) = sum(abs2, A*x-b; init=0.0)
@kernel spec(A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64},
        seed::Float64) = begin
    residual::Vector{Float64} = A*x-b
    y::Float64 = sum(abs2, residual; init=0.0)
    dx::Vector{Float64} = (2seed) .* (transpose(A)*residual)
    return y, dx
end
const rule = derivative_rule(spec; primal=:y, covector=:seed,
    cotangents=(A=nothing, b=nothing, x=:dx))
function direct_mutation(A,b,x)
    y = rule(A,b,x)
    fill!(A, 0.0)
    y
end
function view_mutation(A,b,x)
    alias = view(A, :, :)
    y = rule(A,b,x)
    fill!(alias, 0.0)
    y
end
function global_mutation(A,b,x)
    y = rule(A,b,x)
    fill!(global_matrix, 0.0)
    y
end
function gradient!(dx, f::F, A,b,x) where {F}
    fill!(dx,0.0)
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(f), Enzyme.Active,
        Enzyme.Const(A), Enzyme.Const(b), Enzyme.Duplicated(x,dx))[2]
end
@testset "backend write flags through direct, view and global mutation" begin
    original = [sin(i+0.4j) for i in 1:4, j in 1:2]
    b = [cos(i) for i in 1:4]; x = [0.2,-0.4]; dx = zeros(2)
    expected = 2 .* (transpose(original)*(original*x-b))
    expected_value = native(original,b,x)
    for f in (rule, direct_mutation, view_mutation, global_mutation)
        global_matrix .= original
        y = gradient!(dx,f,global_matrix,b,x)
        @test y ≈ expected_value
        @test dx ≈ expected
        @test f === rule ? global_matrix == original : all(iszero,global_matrix)
    end
end
end
println("BORROWED_SNAPSHOT_CHECKS_COMPLETE")
