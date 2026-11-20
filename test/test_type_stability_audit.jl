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

module Reached
using ReactiveKernels
const scale = Ref{Any}(2.0)

# A helper the prepared program calls, not inlined: its finding is in its own
# method, which alone does not say which statement of the program calls it.
@noinline helper(x) = (scale[] * x)::Float64
@kernel helper_kernel(a::Float64, n::Int) = begin
    mu = helper(a)
    nu = helper(n)
    total = mu + nu
    return total
end

# A keyword method that passes its varargs on, as `MutatingFunctions.apply!!`
# does: Julia compiles it for `Vararg{Any}` whatever the call passes, so the
# splat in it dispatches at run time (MutatingFunctions snag
# `reactivekernels-fa93ad90`).
# The `@noinline` keeps that compiled body a call of its own here, as it is in
# the reported program, where it was not inlined.
apply(::Nothing, f, args...; kwargs...) = (@noinline; f(args...; kwargs...))
struct Gather end
(::Gather)(x, i) = x[i]
@inline gathered(x, i) = apply(nothing, Gather(), x, i)
@kernel gather_kernel(x::Vector{Float64}, i::Vector{Int}) = begin
    g = gathered(x, i)
    total = sum(g)
    return total
end
end

# Snag `located-the-caus-a708f08d`: a finding in a callee names the calls that
# reach it from the prepared program, with the argument types each passes.
@testset "a code finding in a callee names the calls that reach it" begin
    refusal(kernel) = try
        prepare(kernel; type_stability = :error, type_stability_level = :all)
        nothing
    catch err
        err
    end
    file = @__FILE__
    helper_line = only(methods(Reached.helper)).line

    # One message per signature the helper is compiled for, each with its call.
    e = refusal(Reached.helper_kernel)
    @test e isa TypeInstabilityError
    @test length(e.messages) == 2
    for (T, value) in ((Float64, "mu"), (Int, "nu"))
        message = only(filter(m -> occursin("`invoke helper(::$T)`", m), e.messages))
        @test occursin("Reached.helper @ $file:$helper_line` is not type stable", message)
        @test occursin("The prepared program reaches it through these calls, " *
                       "outermost first:", message)
        # The call names the value whose statement makes it, then its frames,
        # among them that statement's scope.
        @test occursin(Regex("\\n  - in [^\\n]*: `invoke helper\\(::$T\\)`\\n" *
                             "      computing `$value`\\n(      at .*\\n)*" *
                             "      at macro expansion @ rk:$value:"), message)
        # The call passes exactly the signature it calls.
        @test !occursin("called with", message)
    end
    kernel = ReactiveKernels._with_type_stability(() -> prepare(Reached.helper_kernel),
                                                  nothing, :values)
    found = ReactiveKernels._code_findings(kernel, (Float64, Int);
        recurse = ReactiveKernels._code_recursion(Val(:all)))
    helper_findings = filter(f -> occursin("Reached.helper @", f.method), found)
    @test !isempty(helper_findings)
    for f in helper_findings
        # Outermost first: the entry call's, down to the call of the helper.
        @test first(f.calls).method == "call"
        @test startswith(last(f.calls).statement, "invoke helper(")
    end
    @test all(f -> isempty(f.calls), filter(f -> f.method == "call", found))

    # A method compiled for wider types than its call passes: the call's own
    # argument types are written out beside the signature it calls.
    e = refusal(Reached.gather_kernel)
    @test e isa TypeInstabilityError
    message = only(e.messages)
    @test occursin("is a call dispatched at run time: `Core._apply_iterate", message)
    @test occursin(r"`invoke [^`]*apply[^`]*\(::Nothing, ::[^`]*Gather, " *
                   r"::Vector\{Float64\}, ::Vararg\{Any\}\)`\n      computing `g`\n" *
                   r"      called with `[^`]*apply[^`]*\(::Nothing, ::[^`]*Gather, " *
                   r"::Vector\{Float64\}, ::Vector\{Int64\}\)`\n", message)
    @test occursin(Regex("\\n      at gathered @ $file:" *
                         "$(only(methods(Reached.gathered)).line)\\n"), message)
    # The keyword method still runs.
    gather = ReactiveKernels._with_type_stability(() -> prepare(Reached.gather_kernel),
                                                  nothing, :values)
    @test gather([1.0, 2.0, 3.0], [1, 3]) == 4.0

    # The warning is once per call path: the same finding reached through
    # another call warns again.
    report(found) = ReactiveKernels._report_type_stability((), found, :warn)
    frame = ReactiveKernels._CodeFrame(:f, :client, 1)
    reached(line) = ReactiveKernels._CodeFinding(
        :dynamic, "f", 1, Any, "g(_2)", [frame],
        [ReactiveKernels._CodeCall("call", "invoke f(::Int64)", "",
                                   [ReactiveKernels._CodeFrame(:caller, :client, line)])])
    @test_logs (:warn, r"at caller @ client:7") (:warn, r"at caller @ client:8") begin
        report([reached(7)])
        report([reached(8)])
        report([reached(7)])
    end
end

end # module CompilerAuditTests
