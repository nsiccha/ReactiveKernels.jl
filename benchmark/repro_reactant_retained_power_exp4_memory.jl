# Public generic compiler stress reproducer; no RK/model code.
# Full-active4D retained exponential and binary power ordinary reverse was terminated under host memory pressure (TEsBFY).
# Not an accepted numerical regression: retain lazy branches and default compiler settings.
using Reactant, StaticArrays, Enzyme, LinearAlgebra, Test

function retained_binary_matrix_power(B, count)
    remaining = copy(count)
    result = one(B)
    @trace while remaining > 0
        @trace result = if isodd(remaining)
            result * B
        else
            result
        end
        remaining = div(remaining, 2)
        @trace B = if remaining > 0
            B * B
        else
            B
        end
    end
    return result
end

function retained_power_exp_term(q, t, count)
    @allowscalar A = SMatrix{4,4}(q)
    @trace term = if t > 0
        sum(retained_binary_matrix_power(exp(A * t), count))
    else
        0.0
    end
    return term
end

function retained_power_exp_cost(q, times, counts)
    result = 0.0
    @trace for i in eachindex(times)
        @allowscalar begin
            t = times[i]
            exponent_read = counts[i]
        end
        term = retained_power_exp_term(q, t, exponent_read)
        result = result + term
    end
    return result
end

retained_power_exp_gradient(q, times, counts) = only(Enzyme.gradient(
    Enzyme.Reverse, x -> retained_power_exp_cost(x, times, counts), q))

function native_binary_matrix_power(B, count)
    result = one(B)
    while count > 0
        if isodd(count)
            result *= B
        end
        count = div(count, 2)
        if count > 0
            B *= B
        end
    end
    return result
end

function native_power_exp_cost(q, times, counts)
    A = SMatrix{4,4}(q)
    return sum(t > 0 ? sum(native_binary_matrix_power(exp(A * t), count)) : 0.0
               for (t, count) in zip(times, counts))
end

@testset "Static exponentials and retained runtime binary powers" begin
    host_q = vec(Matrix(-0.1I, 4, 4) + 0.03reshape(sin.(1:16), 4, 4))
    host_times = [0.1, 0.0, 20.0]
    q = Reactant.to_rarray(host_q)
    times = Reactant.to_rarray(host_times)
    counts = Reactant.to_rarray(Int64[3, 0, 9])
    println("Compiling full-active 4D exponential/binary-power primal"); flush(stdout)
    value_exe = @compile retained_power_exp_cost(q, times, counts)
    println("Compiling full-active 4D exponential/binary-power ordinary reverse"); flush(stdout)
    gradient_exe = @compile retained_power_exp_gradient(q, times, counts)
    println("Evaluating full-active 4D exponential/binary-power parity"); flush(stdout)
    for host_counts in (Int64[3, 0, 9], Int64[17, 1, 0])
        runtime_counts = Reactant.to_rarray(host_counts)
        @test Float64(value_exe(q, times, runtime_counts)) ≈
            native_power_exp_cost(host_q, host_times, host_counts)
        reference = only(Enzyme.gradient(Enzyme.Reverse,
            x -> native_power_exp_cost(x, host_times, host_counts), host_q))
        @test Array(gradient_exe(q, times, runtime_counts)) ≈ reference rtol=1e-8 atol=1e-10
        @test Array(q) == host_q
        @test Array(times) == host_times
        @test Array(runtime_counts) == host_counts
    end
end
