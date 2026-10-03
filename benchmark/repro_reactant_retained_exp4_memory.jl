# Public generic compiler stress reproducer; no RK/model code.
# The scalar-expanded implementation exceeded 63 GiB RSS in ordinary reverse.
# Run the numerical checks with lazy branches and default compiler settings.
using Reactant, StaticArrays, Enzyme, LinearAlgebra, Test

function retained_exp4_term(q, t)
    @allowscalar A = SMatrix{4,4}(q)
    @trace result = if t > 0
        sum(exp(A * t))
    else
        0.0
    end
    return result
end

function retained_exp4_cost(q, times)
    result = 0.0
    @trace for i in eachindex(times)
        @allowscalar t = times[i]
        contribution = retained_exp4_term(q, t)
        result = result + contribution
    end
    return result
end

retained_exp4_gradient(q, times) =
    only(Enzyme.gradient(Enzyme.Reverse, x -> retained_exp4_cost(x, times), q))

function native_retained_exp4(q, times)
    A = SMatrix{4,4}(q)
    return sum(t > 0 ? sum(exp(A * t)) : 0.0 for t in times)
end

@testset "Four-dimensional exponentials in a retained lazy loop" begin
    host_q = vec(Matrix(-0.1I, 4, 4) + 0.03reshape(sin.(1:16), 4, 4))
    host_times = [0.1, 0.0, 20.0]
    q = Reactant.to_rarray(host_q)
    times = Reactant.to_rarray(host_times)
    println("Compiling full-active 4D retained exponential primal"); flush(stdout)
    value_exe = @compile retained_exp4_cost(q, times)
    println("Compiling full-active 4D retained exponential ordinary reverse"); flush(stdout)
    gradient_exe = @compile retained_exp4_gradient(q, times)
    println("Evaluating full-active 4D retained exponential parity"); flush(stdout)
    @test Float64(value_exe(q, times)) ≈ native_retained_exp4(host_q, host_times)
    reference = only(Enzyme.gradient(
        Enzyme.Reverse, x -> native_retained_exp4(x, host_times), host_q))
    @test Array(gradient_exe(q, times)) ≈ reference rtol=1e-9
    @test Array(q) == host_q
    @test Array(times) == host_times
end
