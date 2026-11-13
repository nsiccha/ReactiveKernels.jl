module DerivativeRuleSnapshotTests
using ReactiveKernels, Test
using ReactiveKernels: stage_primal, stage_residuals
import Enzyme

struct Record{T}
    data::T
end
mutable struct MutableRecord{T}
    data::T
end
unwrap(x::AbstractVector{Float64}) = x
unwrap(x::Tuple) = only(x)
unwrap(x::NamedTuple) = x.data
unwrap(x::Union{Record,MutableRecord}) = x.data
unwrap(x::Vector{Vector{Float64}}) = only(x)
wrap_view(x) = view(x, :)
wrap_tuple(x) = (x,)
wrap_named(x) = (; data=x)
wrap_nested(x) = [x]

square_value(x) = sum(abs2, x; init=0.0)
direct(x, seed) = (2seed) .* x
contract(partial, seed) = (2seed) .* unwrap(partial)
struct Joint{F}
    wrap::F
end
(f::Joint)(x) = (square_value(x), f.wrap(x))

function make_rule(wrap::F, cost) where {F}
    joint = Joint(wrap)
    @kernel spec(x::Vector{Float64}, seed::Float64) = begin
        @recipe (cost=4) y::Float64 = square_value(x)
        @recipe (cost=cost) (y, partial) = joint(x)
        @recipe (cost=4) dx::Vector{Float64} = direct(x, seed)
        @recipe (cost=1) dx = contract(partial, seed)
        return y, dx
    end
    derivative_rule(spec; primal=:y, covector=:seed, cotangents=(x=:dx,))
end
function overwrite_input(f::F, x) where {F}
    y = f(x)
    fill!(x, 0.0)
    y
end
function gradient!(dx, f::F, x) where {F}
    fill!(dx, 0.0)
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(f), Enzyme.Active,
        Enzyme.Duplicated(x, dx))[2]
end

function exp_joint(x)
    y = exp.(x)
    y, view(y, :)
end
exp_contract(partial, seed) = seed .* partial
function exp_rule()
    @kernel spec(x::Vector{Float64}, seed::Vector{Float64}) = begin
        (y, partial) = exp_joint(x)
        dx::Vector{Float64} = exp_contract(partial, seed)
        return y, dx
    end
    derivative_rule(spec; primal=:y, covector=:seed, cotangents=(x=:dx,))
end
function overwrite_primal(f::F, x) where {F}
    y = f(x)
    result = sum(y; init=0.0)
    fill!(y, 0.0)
    result
end

@testset "DerivativeRule residual mutation snapshots" begin
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt)
    wrappers = (identity, copy, wrap_view, wrap_tuple, wrap_named, Record,
        MutableRecord, wrap_nested)
    for wrap in wrappers
        # Cost 2 selects a partial in both old and joint staging. Cost 10
        # exercises retention of the original input, without a joint producer.
        for cost in (2, 10)
            rule = make_rule(wrap, cost)
            @test stage_residuals(rule, Val(1)) == (cost == 2 ? (:partial,) : (:x,))
            for n in (0, 1, 3, 17)
                original = [0.7sin(i) for i in 1:n]
                x = copy(original); dx = zeros(n)
                caller = x -> overwrite_input(rule, x)
                @test gradient!(dx, caller, x) ≈ square_value(original)
                @test dx ≈ 2 .* original
                @test all(iszero, x)
            end
            y, tape = stage_primal(rule, Val(1), [0.2, -0.4, 0.7])
            findings = ReactiveKernels._level_code_findings(:all,
                ext._retain_residuals,
                (Tuple{Int}, typeof(tape), typeof(y), Tuple{Bool,Bool}))
            @test isempty(findings)
        end
    end
    array_rule = exp_rule()
    for n in (0, 1, 3, 17)
        x = [0.7sin(i) for i in 1:n]; dx = zeros(n)
        caller = x -> overwrite_primal(array_rule, x)
        @test gradient!(dx, caller, x) ≈ sum(exp, x; init=0.0)
        @test dx ≈ exp.(x)
    end
    # Preserve shared references across distinct tuple entries, including a
    # view's parent; independent element copies would lose this relationship.
    x = [0.2, -0.4]
    tape = (Record(x), (x,), view(x, :))
    retained = @inferred ext._retain_residuals((-1,-1,-1), tape, 0.0, (false,true))
    @test typeof(retained) === typeof(tape)
    @test retained[1].data === retained[2][1]
    @test parent(retained[3]) === retained[1].data
    @test retained[1].data !== x
    fill!(x, 0.0)
    @test retained[1].data == [0.2, -0.4]
    scalar = (1.5, (2, 3.0))
    @test (@inferred ext._retain_residuals((-1,-1), scalar, 0.0, (false,))) === scalar
    @test (@inferred ext._retain_residuals((), (), 0.0, (false,))) === ()
end

@testset "borrowed snapshots retain ownership and concrete audits" begin
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt)
    r = ext._retain_residuals
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
    env = ext._DerivativeResidualEnvironment(tape, (1,-1,-1),
        (false,false), true)
    findings = ReactiveKernels._level_code_findings(:all, Base.deepcopy_internal,
        (typeof(env), IdDict{Any,Any}))
    println("OWNED_SNAPSHOT_HOOK_AUDIT ", findings)
    @test isempty(findings)
    record = Record(a)
    for value in (view(a,:), (a,), (;data=a), record,
            MutableRecord(a), [a])
        @test !ext._flat_residual_environment((a,value))
        saved = r((1,-1), (a,value), 1.0, (false,false))
        @test saved[1] !== a
        if value isa SubArray
            @test parent(saved[2]) === saved[1]
        else
            @test unwrap(saved[2]) === saved[1]
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
end
