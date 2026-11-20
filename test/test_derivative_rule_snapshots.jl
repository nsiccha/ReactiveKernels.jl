module DerivativeRuleSnapshotTests
using ReactiveKernels, Test
using InteractiveUtils: code_llvm
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

reverse_from_inputs(rule, inputs::Tuple, seed) = reverse_cut(rule, Val(1), inputs..., seed)

@testset "Reverse cuts with unknown tuple length" begin
    rule = make_rule(Record, 2)
    x = [1.5, -2.0]
    _, residuals = stage_primal(rule, Val(1), x)
    for tape_type in (Any, Tuple)
        @test only(Base.return_types(stage_reverse,
            Tuple{typeof(rule), Val{1}, tape_type, Float64})) !== Union{}
        @test only(Base.return_types(reverse_from_inputs,
            Tuple{typeof(rule), tape_type, Float64})) !== Union{}
    end
    @test stage_reverse(rule, Val(1), residuals, 3.0) == (6 .* x,)
    @test reverse_from_inputs(rule, (x,), 3.0) == (6 .* x,)
    @test_throws ArgumentError stage_reverse(rule, Val(1), (), 3.0)
    @test_throws ArgumentError reverse_from_inputs(rule, (x, x), 3.0)
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
    # Without Enzyme's repair `overwritten_reachable_data` (`Val(false)`), an
    # input's flag proves nothing about the arrays it holds, so they are
    # copied. With it (`Val(true)`), an unchanged input is kept whole.
    nested = [a]
    saved = @inferred r(Val(false), (1,), (nested,), 1.0, (false,false))
    @test saved[1] !== nested
    @test only(saved[1]) !== a
    @test (@inferred r(Val(true), (1,), (nested,), 1.0, (false,false)))[1] === nested
    # Audit the generated snapshot directly as well as its caller.
    for deep in (false, true)
        signature = (typeof(tape), Tuple{Int,Int,Int}, Tuple{Bool,Bool}, Val{true}, Val{deep})
        @test hasmethod(ext._snapshot_residuals, signature)
        findings = ReactiveKernels._level_code_findings(:all, ext._snapshot_residuals, signature)
        println("OWNED_SNAPSHOT_AUDIT ", findings)
        @test isempty(findings)
    end
    record = Record(a)
    for value in (view(a,:), (a,), (;data=a), record,
            MutableRecord(a), [a])
        @test !ext._flat_residual_environment((a,value))
        for deep in (false, true)
            saved = r(Val(deep), (1,-1), (a,value), 1.0, (false,false))
            # With the repair, the unchanged input `a` is kept, and so is the
            # intermediate that holds it, unless that is snapshot by `deepcopy`.
            kept = deep && !(value isa Union{MutableRecord,Vector{Vector{Float64}}})
            @test (saved[1] === a) == kept
            if value isa SubArray
                @test parent(saved[2]) === saved[1]
            else
                @test unwrap(saved[2]) === saved[1]
            end
        end
        # Unsupported borrowing forms retain the original snapshot path. Audit
        # that reachable path, rather than a hook it never calls.
        @test isempty(ReactiveKernels._level_code_findings(:all,
            r, (Tuple{Int,Int}, typeof((a,value)), Float64, Tuple{Bool,Bool})))
    end
end

# Measure the prepared, typed call and consume its result. A top-level call
# through a locally loaded extension function boxes its argument/result tuples
# on Julia 1.10, even when retention itself allocates nothing.
function retention_allocations(f::F, sources, residuals, primal, flags) where {F}
    @allocated(f(sources, residuals, primal, flags) === residuals)
end

@testset "dense primal alias protection" begin
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt)
    retain = ext._retain_residuals
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    readonly = (a, b, 1.5)
    sources, flags = (1, 2, -1), (false, false, false)
    primal = similar(a)
    @test (@inferred retain(sources, readonly, primal, flags)) === readonly
    retention_allocations(retain, sources, readonly, primal, flags)
    @test retention_allocations(retain, sources, readonly, primal, flags) == 0
    protected = @inferred retain(sources, readonly, b, flags)
    @test protected[1] === a
    @test protected[2] !== b
    @test protected[2] == b
    borrowed = @inferred retain((1, -1), (a, a), b, (false, false))
    @test borrowed[1] === a
    @test borrowed[2] === a
    for (primal, flags) in ((a, (false, false)), (b, (false, true)))
        saved = @inferred retain((1, -1), (a, a), primal, flags)
        @test saved[1] !== a
        @test saved[2] === saved[1]
        @test saved[1] == a
    end
    @test retain((1,), ([a],), b, (false, false))[1][1] !== a
    @test retain((1,), (a,), view(b, :), (false, false))[1] !== a
    # Distinct wrappers can overlap without sharing an array header or their
    # first element's address. Keep the owner alive throughout this check.
    owner = collect(1.0:8.0)
    GC.@preserve owner begin
        firstpart = unsafe_wrap(Array, pointer(owner), 4; own=false)
        overlap = unsafe_wrap(Array, pointer(owner, 3), 4; own=false)
        separate = unsafe_wrap(Array, pointer(owner, 5), 4; own=false)
        @test ext._residual_storage_overlaps(firstpart, overlap)
        @test ext._residual_storage_overlaps(overlap, firstpart)
        @test !ext._residual_storage_overlaps(firstpart, separate)
        @test !ext._residual_storage_overlaps(separate, firstpart)
        saved = retain((1,), (firstpart,), overlap, (false, false))
        @test saved[1] !== firstpart
        @test saved[1] == firstpart
    end
    @test !ext._residual_storage_overlaps(Float64[], a)
    @test !ext._residual_storage_overlaps(a, Float64[])
    @test isempty(ReactiveKernels._level_code_findings(:all, retain,
        (Tuple{Int,Int}, Tuple{Vector{Float64},Vector{Float64}},
         Vector{Float64}, Tuple{Bool,Bool})))
end

# The snapshot is concretely typed code (snag `rkppl-ark-dogfoo-72035cd9`).
# Base's `deepcopy(@nospecialize x)` compiled to a boxed call, a `jl_invoke`
# in the derivative Enzyme compiles where it emits the rule into it, which the
# check of that derivative refuses at `type_stability_level = :all`, also on a
# path the program never takes. Mutable containers keep `deepcopy`, whose memo
# follows their identity. (Whether Enzyme emits a rule into the derivative or
# calls it natively depends on Julia's inlining of the rule, so these tests
# read the compiled code itself; RKPPL's `test_glm_object_values.jl` checks a
# derivative the report sees.)
function retention_llvm(f, types)
    io = IOBuffer()
    code_llvm(io, f, types; debuginfo = :none)
    String(take!(io))
end
function no_dispatch(code)
    !occursin("deepcopy", code) && !occursin("jl_invoke", code) &&
        !occursin("jl_apply_generic", code)
end

@testset "residual snapshots compile without deepcopy" begin
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt)
    retain = ext._retain_residuals
    a, b = [1.0, 2.0], [3.0, 4.0, 5.0]
    # Flat with a borrowed and a copied input; nested containers; a mutable
    # result other than a dense array.
    for (tape, sources, primal, flags, borrow) in (
            ((a, b, 1.5), (1, 2, -1), 0.0, (false, false, true), true),
            ((a, view(a, :), (b,)), (1, -1, -1), 0.0, (false, false), false),
            (((; data = a), Record(b)), (-1, -1), view(b, :), (false,), false))
        types = Tuple{typeof(sources), typeof(tape), typeof(primal), typeof(flags)}
        @test no_dispatch(retention_llvm(retain, types))
        for deep in (false, true)
            @test no_dispatch(retention_llvm(ext._snapshot_residuals,
                Tuple{typeof(tape), typeof(sources), typeof(flags), Val{borrow}, Val{deep}}))
        end
        @test only(Base.return_types(retain, types)) === typeof(tape)
    end
    # The overlap check of a dense result lists storage from the types alone,
    # through mutable structs such as a `Dict` too; an array of arrays cannot
    # be listed without reading it and counts as overlapping.
    for value in (Record(a), MutableRecord(a), Dict(1 => 2.0), (a, view(b, :)))
        @test no_dispatch(retention_llvm(ext._residual_overlaps,
            Tuple{typeof(value), Vector{Float64}}))
        @test ext._residual_overlaps(value, a) == !(value isa Dict)
        @test !ext._residual_overlaps(value, [1.0])
    end
    @test ext._residual_overlaps([a], [1.0])
    # Residuals the rule's graph does not infer (a type as its callable's
    # field, as `Record` in the first testset) reach the snapshot through a
    # run-time call. Retention preserves the known tuple shape for the reverse.
    for (tape, sources) in ((Tuple{Any}, Tuple{Int}), (Tuple{Any,Float64}, Tuple{Int,Int}))
        for (primal, flags) in ((Float64, Tuple{Bool,Bool}), (SubArray, Tuple{Bool}))
            @test only(Base.return_types(retain, Tuple{sources, tape, primal, flags})) <: tape
        end
    end
    # Positive control: a mutable container keeps Base's `deepcopy`.
    tape = (MutableRecord(a), b)
    @test occursin("deepcopy", retention_llvm(ext._snapshot_residuals,
        Tuple{typeof(tape), Tuple{Int,Int}, Tuple{Bool}, Val{false}, Val{false}}))
    saved = retain((-1, -1), tape, 0.0, (false,))
    @test saved[1] !== tape[1] && saved[1].data == a && saved[1].data !== a
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

# The caller writes, after the rule's call, to an array that a constant
# composite input holds: through the input's field, through another argument or
# a view that is the same array, and through a global. Before Enzyme's dogfood
# repair `overwritten_reachable_data`, the write flag of the composite input
# stays unset in each case (benchmark/repro_enzyme_nested_overwrite.jl), so the
# reverse has to read the weight the adapter retained, not the caller's array.
# With the repair the flag is set, and when nothing is written the adapter
# borrows the input's array instead of copying it.
module CompositeInputChecks
using ReactiveKernels, Test
import Enzyme

const ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt)
const repaired = isdefined(Enzyme, :has_dogfood_repair) &&
    Enzyme.has_dogfood_repair(:overwritten_reachable_data)

struct Weights
    values::Vector{Float64}
end
@kernel function weighted(g, x::Float64, seed::Float64)
    y::Float64 = g.values[1] * x
    dx::Float64 = g.values[1] * seed
    return y, dx
end
const rule = derivative_rule(weighted; primal=:y, covector=:seed,
    cotangents=(g=nothing, x=:dx))
const shared = [3.0]

function through_field(g, alias, x)
    y = rule(g, x)
    g.values[1] = 42.0
    y
end
function through_alias(g, alias, x)
    y = rule(g, x)
    alias[1] = 42.0
    y
end
function through_global(g, alias, x)
    y = rule(g, x)
    shared[1] = 42.0
    y
end

@testset "caller writes to storage inside a constant rule input" begin
    for (f, alias) in ((through_field, shared), (through_alias, shared),
            (through_alias, view(shared, :)), (through_global, shared)), w in (3.0, -2.0)
        shared[1] = w
        g = Weights(shared)
        cotangents, y = Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(f), Enzyme.Active,
            Enzyme.Const(g), Enzyme.Const(alias), Enzyme.Active(2.0))
        @test y == 2w
        @test last(cotangents) == w
        @test shared[1] == 42.0
    end
end

steady(g, x) = rule(g, x)
function reverse_bytes(g)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(steady), Enzyme.Active,
        Enzyme.Const(g), Enzyme.Active(2.0))
    @allocated Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(steady), Enzyme.Active,
        Enzyme.Const(g), Enzyme.Active(2.0))
end

# A geometry that also holds a `Dict`, read by a rule with an array result:
# the result's storage is checked against every array the input holds.
struct Indexed
    index::Dict{Int,Int32}
    values::Vector{Float64}
end
row_value(g, x) = [g.values[1] * x]
row_slope(g, bar) = g.values[1] * only(bar)
@kernel function indexed_row(g, x::Float64, bar::Vector{Float64})
    y::Vector{Float64} = row_value(g, x)
    dx::Float64 = row_slope(g, bar)
    return y, dx
end
const row_rule = derivative_rule(indexed_row; primal=:y, covector=:bar,
    cotangents=(g=nothing, x=:dx))
row_loss(g, x) = only(row_rule(g, x))
function row_bytes(g)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(row_loss), Enzyme.Active,
        Enzyme.Const(g), Enzyme.Active(2.0))
    @allocated Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(row_loss), Enzyme.Active,
        Enzyme.Const(g), Enzyme.Active(2.0))
end

# Borrowing an array inside an input is sound only where Enzyme's flag covers
# it, so the adapter copies the array exactly when the repair is missing.
@testset "an unchanged composite input is borrowed only with the repair" begin
    @test ext._OVERWRITTEN_REACHABLE_DATA[] == repaired
    g = Weights(fill(3.0, 1 << 16))
    @test (reverse_bytes(g) < sizeof(g.values)) == repaired
    indexed = Indexed(Dict(1 => Int32(1)), fill(3.0, 1 << 16))
    @test Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(row_loss), Enzyme.Active,
        Enzyme.Const(indexed), Enzyme.Active(2.0))[1][2] == 3.0
    @test (row_bytes(indexed) < sizeof(indexed.values)) == repaired
end

# Retention given the flags, for either build: `Val(true)` is the repaired one.
@testset "retention of arrays inside an input" begin
    v = [3.0, 1.0, 2.0]
    g = Weights(v)
    residuals = (g, view(v, 1:2))
    unchanged = (false, false, false)
    written = (false, true, false)
    # Kept inside an unchanged input, also through an intermediate view of it.
    kept = ext._retain_residuals(Val(true), (1, -1), residuals, 0.0, unchanged)
    @test kept[1].values === v && parent(kept[2]) === v
    # A fresh dense result keeps it too.
    @test ext._retain_residuals(Val(true), (1,), (g,), [1.0], unchanged)[1].values === v
    # Copied once and shared by the view: the input may be written, the flag
    # proves nothing without the repair, or a dense result shares its storage
    # (itself, or another header over the same memory).
    for (deep, primal, flags) in ((true, 0.0, written), (false, 0.0, unchanged),
            (true, v, unchanged), (true, reshape(v, 3, 1), unchanged))
        saved = ext._retain_residuals(Val(deep), (1, -1), residuals, primal, flags)
        @test saved[1].values !== v && saved[1].values == v
        @test parent(saved[2]) === saved[1].values
    end
    # The arrays of a mutable struct, such as a `Dict`, are listed for that
    # check without reading its elements.
    indexed = Indexed(Dict(1 => Int32(2)), v)
    @test ext._retain_residuals(Val(true), (1,), (indexed,), [1.0], unchanged)[1] === indexed
    saved = ext._retain_residuals(Val(true), (1,), (indexed,), v, unchanged)[1]
    @test saved.values !== v && saved.values == v && saved.index == indexed.index
    # An array of arrays cannot be listed without reading its elements, so it
    # counts as sharing the result's storage.
    nested = [v]
    @test only(ext._retain_residuals(Val(true), (1,), (nested,), [1.0], unchanged)[1]) !== v
end
end
end
