module CompilerAuditTests
using ReactiveKernels, Test

module ConstantReturns
using ReactiveKernels

integer(x) = 1
nothing_value(x) = nothing
tuple_value(x) = (:constant, 1.0)
folded(x) = one(x)

const counter = Ref(0)
function with_effect(x)
    counter[] += 1
    1
end

function unstable_effect(x)
    counter[] += Base.inferencebarrier(x)
    1
end

@kernel constant_kernel(x::Float64) = begin
    y = one(x)
    return y
end
@kernel literal_kernel() = 42
end

@testset "type-stability audit of constant returns" begin
    K = ConstantReturns
    for f in (K.integer, K.nothing_value, K.tuple_value, K.folded, K.with_effect)
        @test isempty(ReactiveKernels._code_findings(f, (Float64,)))
    end
    # Returning a constant must not hide dynamic work before the return.
    found = ReactiveKernels._code_findings(K.unstable_effect, (Int,))
    @test any(f -> f.kind === :dynamic, found)
    @test K.counter[] == 0 # Auditing only infers; it must not execute effects.
    kernel = @test_logs min_level = Base.CoreLogging.Warn prepare(
        K.constant_kernel; type_stability = :error)
    @test kernel(2.0) === 1.0
    literal = @test_logs min_level = Base.CoreLogging.Warn prepare(
        K.literal_kernel; type_stability = :error)
    @test literal() === 42
end

module Inlining
const scale = Ref{Any}(2.0)
@inline leaf(x) = (scale[] * x)::Float64
@inline middle(x) = leaf(x) + 1.0
outer(x) = middle(x) - 1.0
untyped = 2.0
read_global(x) = untyped * x

function logged(x)
    x < 0 && @warn "negative input" shown = string(Base.inferencebarrier(x))
    2.0 * x
end
function logged_dynamic(x)
    x < 0 && @warn "negative input" x
    leaf(x)
end
end

@testset "type-stability audit follows compiler debug info" begin
    ci = first(only(code_typed(Inlining.outer, (Float64,);
                               optimize = true, debuginfo = :source)))
    chains = [ReactiveKernels._code_inlining_chain(ci, i) for i in eachindex(ci.code)]
    @test any(chain -> :leaf in chain && :middle in chain, chains)
    @test any(f -> f.kind === :dynamic,
              ReactiveKernels._code_findings(Inlining.outer, (Float64,)))
    # Walking debug info must preserve the logging exemption without hiding
    # a dynamic call in the same function's own code.
    @test isempty(ReactiveKernels._code_findings(Inlining.logged, (Float64,)))
    found = ReactiveKernels._code_findings(Inlining.logged_dynamic, (Float64,))
    @test any(f -> f.kind === :dynamic, found)
    @test all(f -> !occursin("shouldlog", f.statement), found)
    found = ReactiveKernels._code_findings(Inlining.read_global, (Float64,))
    @test any(f -> f.kind === :nonconcrete && occursin("untyped", f.statement), found)
    @test any(f -> f.kind === :dynamic, found)
end

end # module CompilerAuditTests
