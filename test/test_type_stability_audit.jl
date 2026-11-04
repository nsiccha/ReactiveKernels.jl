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
    chains = [[f.name for f in ReactiveKernels._code_frames(ci, i)]
              for i in eachindex(ci.code)]
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

module Located
using ReactiveKernels
const scale = Ref{Any}(2.0)

@inline leaf(x) = (scale[] * x)::Float64
@inline middle(x) = leaf(x) + 1.0
outer(x) = middle(x) - 1.0

# A splat into a typed array literal, inlined into the program that calls it:
# the method a finding names is then the prepared program's entry call.
@inline harmonics(K) = Float64[repeat(1:K, 2)...]
@kernel splat_kernel(K::Int) = begin
    h = harmonics(K)
    total = sum(h)
    return total
end

# Lowering records a macro expansion as a frame when the macro is written in
# another file than its call.
include_string(@__MODULE__, """
    macro scaled(x)
        quote
            scale[] * \$(esc(x))
        end
    end
    """, "scaled_macro.jl")
@inline expanded(x) = @scaled(x)::Float64
outer_expanded(x) = expanded(x) - 1.0
end

@testset "a code finding names its statement's source, innermost frame first" begin
    file = Symbol(@__FILE__)
    frame(f) = let m = only(methods(f))
        ReactiveKernels._CodeFrame(m.name, file, m.line)
    end
    found = ReactiveKernels._code_findings(Located.outer, (Float64,))
    finding = only(filter(f -> f.kind === :dynamic, found))
    # `leaf` and `middle` are inlined, so the method names only the entry call.
    @test finding.method == "call"
    # `outer`'s frame is the line of its body, which is its definition's line.
    @test finding.location ==
          [frame(Located.leaf), frame(Located.middle), frame(Located.outer)]
    @test sprint(show, finding) ==
          "dynamic in call %$(finding.index) :: Any  $(finding.statement)" *
          "  at leaf @ $file:$(only(methods(Located.leaf)).line)"

    # A macro expansion is a frame between the functions it lies in: its line is
    # the one in the macro's `quote`, then the call's.
    found = ReactiveKernels._code_findings(Located.outer_expanded, (Float64,))
    location = only(filter(f -> f.kind === :dynamic, found)).location
    @test location == [
        ReactiveKernels._CodeFrame(Symbol("macro expansion"), Symbol("scaled_macro.jl"), 3),
        frame(Located.expanded), frame(Located.outer_expanded)]

    # The reported shape: a typed array splat, inlined into the entry call.
    e = try
        prepare(Located.splat_kernel; type_stability = :error,
                type_stability_level = :all)
        nothing
    catch err
        err
    end
    @test e isa TypeInstabilityError
    message = only(e.messages)
    @test occursin("The prepared program's entry call is not type stable", message)
    @test occursin("is a call dispatched at run time: `Core._apply_iterate", message)
    # The helper's frame comes first, directly under the statement.
    line = only(methods(Located.harmonics)).line
    @test occursin("`\n      at harmonics @ $file:$line\n", message)

    # The warning is the refusal's text.
    @test_logs (:warn, message) prepare(Located.splat_kernel;
                                        type_stability_level = :all)

    # A warning is once per finding: the same statement written elsewhere is
    # another finding, and repeats of one location stay one warning.
    report(found) = ReactiveKernels._report_type_stability((), found, :warn)
    statement(line) = ReactiveKernels._CodeFinding(
        :dynamic, "call", 1, Any, "f(_2)",
        [ReactiveKernels._CodeFrame(:f, :client, line)])
    @test_logs (:warn, r"at f @ client:1") (:warn, r"at f @ client:2") begin
        report([statement(1)])
        report([statement(2)])
        report([statement(1)])
    end
end

end # module CompilerAuditTests
