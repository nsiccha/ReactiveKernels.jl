# Attribute snapshot audit findings without changing the audit or its policy.
# The paired functions invoke the same Base copying implementation. A separate
# user-code control verifies that preserving a call boundary does not exempt it.
using ReactiveKernels, Test

copy_plain(x) = deepcopy(x)
copy_leaf(x) = Base.@noinline deepcopy(x)
function nonconcrete_user(x)
    items = Any[x[1]]
    only(items)::Float64
end
user_leaf(x) = Base.@noinline nonconcrete_user(x)

@testset "snapshot audit attribution" begin
    x = [0.2, -0.4, 0.7]
    tape = (x, view(x, :))
    for (label, f) in ((:plain, copy_plain), (:base_leaf, copy_leaf))
        result = f(tape)
        @test typeof(result) === typeof(tape)
        @test result[1] == x
        @test result[1] !== x
        @test parent(result[2]) === result[1]
        findings = ReactiveKernels._level_code_findings(:all, f, (typeof(tape),))
        println((; label, findings))
        for finding in findings
            println((; label, kind=finding.kind, method=finding.method,
                statement=finding.statement, location=finding.location))
        end
        # Actual typed IR, with Julia's ordinary display; no view rewriting.
        println("ACTUAL_TYPED_IR ", label)
        show(stdout, MIME"text/plain"(), Base.code_typed(f, (typeof(tape),);
            optimize=true, debuginfo=:source))
        println()
    end
    findings = ReactiveKernels._level_code_findings(:all, user_leaf, (typeof(x),))
    println((; label=:nonconcrete_user_leaf, findings))
    @test !isempty(findings)
end
println("SNAPSHOT_AUDIT_ATTRIBUTION_COMPLETE")
