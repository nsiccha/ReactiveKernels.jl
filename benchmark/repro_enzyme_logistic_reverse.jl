# Backend-only native sigmoid reverse reproducer; no ReactiveKernels or PPL.
# Run in a consumer environment with Enzyme and LogExpFunctions:
#   julia --startup-file=no --project=<env> benchmark/repro_enzyme_logistic_reverse.jl
# Reproduced on Julia 1.10.12, Enzyme 0.13.209, LogExpFunctions 0.3.29 and
# 1.0.1. Upstream tracking: EnzymeAD/Enzyme.jl#3583 and PR #3595.
# The broken checks expose the upstream boundary and must be revisited when
# that boundary is lifted. The passing control uses ordinary reverse AD on
# the sigmoid's log-domain primal, with no derivative rule or annotation.
using Enzyme, LogExpFunctions, Test

println("Julia=", VERSION, " Enzyme=", pkgversion(Enzyme),
        " LogExpFunctions=", pkgversion(LogExpFunctions))
ordinary_logistic(x) = logistic(x)
log_sigmoid(x) = exp(-log1pexp(-x))
# Independent high-precision value/derivative oracle, used only for checks.
function reference(x)
    setprecision(512) do
        z = BigFloat(x)
        e = exp(-abs(z))
        p = z >= 0 ? inv(1 + e) : e / (1 + e)
        (typeof(x)(p), typeof(x)(e / (1 + e)^2))
    end
end
# The original reporter's six calls, reproduced without package code.
for f in (ordinary_logistic, log_sigmoid), x in (-1000.0, 0.0, 1000.0)
    derivative = autodiff(Reverse, f, Active, Active(x))[1][1]
    println((; function_name = nameof(f), x, value = f(x), derivative))
end

@testset "upstream logistic reverse boundary (#3583)" begin
    for x in (1000.0, 1000.0f0)
        derivative = autodiff(Reverse, ordinary_logistic, Active, Active(x))[1][1]
        @test_broken isfinite(derivative)
    end
    # A saturated primal can still have a representable sigmoid derivative.
    derivative = autodiff(Reverse, ordinary_logistic, Active, Active(40.0))[1][1]
    _, expected = reference(40.0)
    @test_broken isapprox(derivative, expected; rtol=2e-14, atol=0)
end

@testset "log-sigmoid ordinary reverse controls" begin
    for T in (Float64, Float32)
        for x in T.((-1000, -800, -745, -720, -710, -709, -700, -100, -40, -20, -1, 0, 1, 20, 40, 100, 700, 709, 710, 720, 745, 800, 1000))
            value, expected = reference(x)
            original = autodiff(Reverse, ordinary_logistic, Active, Active(x))[1][1]
            stable = autodiff(Reverse, log_sigmoid, Active, Active(x))[1][1]
            rtol = T === Float64 ? 2e-14 : 2e-6
            atol = 8 * nextfloat(zero(T))
            @test isapprox(log_sigmoid(x), value; rtol, atol)
            @test isfinite(stable)
            @test isapprox(stable, expected; rtol, atol)
            println((; type=T, x, original, stable, expected))
        end
    end
end
