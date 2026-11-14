# Diagnostic only: whether a retained custom-reverse call makes its scalar
# Active seed escape. No ReactiveKernels or DifferentiationInterface import.
using Enzyme, Test, Profile, InteractiveUtils
const ER = Enzyme.EnzymeRules
mutable struct Slot
    tape::Vector{Float64}
    empty::Vector{Float64}
end
Slot() = (empty = Float64[]; Slot(empty, empty))
function take!(slot, n)
    tape = slot.tape
    length(tape) == n || return zeros(n)
    slot.tape = slot.empty
    fill!(tape, 0.0)
end
struct Reduction{Retained} end
function (::Reduction)(q, idx, slot)
    total = 0.0
    for j in idx
        total += sin(q[j])
    end
    total
end
function ER.augmented_primal(config::ER.RevConfig, f::Const{<:Reduction},
        ::Type{Active{Float64}}, q::Duplicated, idx::Const, slot::Enzyme.Annotation)
    tape = take!(slot.val, length(q.val))
    value = 0.0
    for j in idx.val
        value += sin(q.val[j])
        tape[j] += cos(q.val[j])
    end
    ER.AugmentedReturn(ER.needs_primal(config) ? value : nothing, nothing, tape)
end
function reverse_body(dy, tape, q, slot)
    for j in eachindex(q.dval)
        q.dval[j] += dy.val * tape[j]
    end
    slot.val.tape = tape
    (nothing, nothing, nothing)
end
@inline function ER.reverse(config::ER.RevConfig, f::Const{Reduction{false}},
        dy::Active, tape, q::Duplicated, idx::Const, slot::Enzyme.Annotation)
    reverse_body(dy, tape, q, slot)
end
Base.@noinline function ER.reverse(config::ER.RevConfig, f::Const{Reduction{true}},
        dy::Active, tape, q::Duplicated, idx::Const, slot::Enzyme.Annotation)
    reverse_body(dy, tape, q, slot)
end
scaled(q, f, idx, slot, scale) = scale * f(q, idx, slot)
function gradient!(dq, q, f, idx, slot, shadow, scale)
    fill!(dq, 0.0)
    result = autodiff(ReverseWithPrimal, Const(scaled), Active,
        Duplicated(q, dq), Const(f), Const(idx), Duplicated(slot, shadow), Const(scale))
    last(result)
end
@noinline function triple(f::F) where {F}
    total = 0.0
    for _ in 1:3
        total += f()
    end
    total
end
function probe(::Val{retained}) where {retained}
    q, dq = [.1,-.4,.3], zeros(3)
    idx = [mod1(i,3) for i in 1:128]
    slot, shadow = Slot(), Slot()
    f = Reduction{retained}()
    call = () -> gradient!(dq, q, f, idx, slot, shadow, .7)
    for point in ([.1,-.4,.3],[-.8,.6,1.2])
        q .= point
        expected = zeros(3)
        for j in idx
            expected[j] += .7cos(q[j])
        end
        @test call() ≈ .7sum(sin(q[j]) for j in idx)
        @test dq ≈ expected
    end
    triple(call); triple(call)
    println("SEED_BOX retained=", retained, " calls=3 bytes=", @allocated(triple(call)))
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1.0 begin
        triple(call)
        nothing
    end
    for a in Profile.Allocs.fetch().allocs
        println("SEED_BOX_ALLOC retained=", retained, " bytes=", a.size, " type=", a.type)
        for frame in Iterators.take(a.stacktrace, 8)
            println("  ", frame.func, " at ", frame.file, ":", frame.line)
        end
    end
    if "--dump-llvm" in ARGS
        directory = get(ENV, "SEED_BOX_IR_DIR", tempdir())
        mkpath(directory)
        path = joinpath(directory, "seed-box-ad-$(retained).ll")
        open(path, "w") do io
            Enzyme.Compiler.enzyme_code_llvm(io, scaled, Active,
                Tuple{Duplicated{Vector{Float64}},Const{typeof(f)},Const{Vector{Int}},Duplicated{Slot},Const{Float64}};
                returnPrimal=true, raw=true, debuginfo=:source, dump_module=true)
        end
        println("SEED_BOX_LLVM retained=", retained, " path=", path)
    end
    flush(stdout)
end
@testset "retained scalar seed boundary" begin
    probe(Val(false))
    probe(Val(true))
end
println("SEED_BOX_COMPLETE"); flush(stdout)
