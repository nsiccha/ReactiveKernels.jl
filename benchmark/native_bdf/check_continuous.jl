using Enzyme, Test
include("prototype.jl")
using .NativeBDFPrototype

decay(t, y, rate) = [-rate * y[1]]
const TIMES = [0.4, 1.6]
loss(u, rate) = sum(continuous_bdf(decay, u, 0.0, TIMES,
    2e-6, 3e-6, 4096, rate))
budget_loss(u, rate) = sum(continuous_bdf(decay, u, 0.0, TIMES,
    2e-6, 3e-6, 4096, rate; budget = true))
time_loss(u, ts) = sum(continuous_bdf(decay, u, 0.0, ts,
    2e-6, 3e-6, 4096, 0.7))

@testset "continuous BDF without output-budget callback" begin
    u, du = [1.0], zeros(1)
    values = continuous_bdf(decay, u, 0.0, TIMES, 2e-6, 3e-6, 4096, 0.7)
    expected = exp.(-0.7 .* TIMES)
    @test values[:, 1] ≈ expected rtol = 4e-5
    result = Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active,
        Enzyme.Duplicated(u, du), Enzyme.Active(0.7))
    @test du[1] ≈ sum(expected) rtol = 4e-5
    @test result[1][2] ≈ -sum(TIMES .* expected) rtol = 4e-5
    @test u == [1.0] && TIMES == [0.4, 1.6]
    println("CONTINUOUS_BASE_REVERSE_PASS ", result, " du=", du)
end

# Diagnostics below report unresolved cases; the script's success is not
# acceptance of these cases or of the original application.
for (label, run) in (
        ("budget_callback", () -> begin
            u, du = [1.0], zeros(1)
            Enzyme.autodiff(Enzyme.Reverse, budget_loss, Enzyme.Active,
                Enzyme.Duplicated(u, du), Enzyme.Active(0.7))
            println("BUDGET_REVERSE_RESULT du=", du)
        end),
        ("active_output_times", () -> begin
            u, du, ts, dts = [1.0], zeros(1), copy(TIMES), zeros(2)
            Enzyme.autodiff(Enzyme.Reverse, time_loss, Enzyme.Active,
                Enzyme.Duplicated(u, du), Enzyme.Duplicated(ts, dts))
            expected = -0.7 .* exp.(-0.7 .* ts)
            println("ACTIVE_TIME_RESULT ", dts, " expected=", expected)
            isapprox(dts, expected; rtol = 4e-5) || error("incorrect active output-time gradient")
        end))
    println("DIAGNOSTIC_BEGIN ", label)
    flush(stdout)
    try
        run()
        println("DIAGNOSTIC_PASS ", label)
    catch e
        println("DIAGNOSTIC_FAIL ", label, " ", typeof(e))
        println(first(split(sprint(showerror, e), '\n'), 3))
    end
    flush(stdout)
end
