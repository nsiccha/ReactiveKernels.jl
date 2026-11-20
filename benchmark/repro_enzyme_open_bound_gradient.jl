# Run with julia --project=<environment with Enzyme> benchmark/repro_enzyme_open_bound_gradient.jl
# Reduced from an RKPPL truncated Normal with a scalar lower bound.
# Enzyme.jl 987a9463 and libEnzyme 2528afe8 return two NaNs when lo == -Inf.
using Enzyme
using Test

function endpoint(b, m, s)
    b == -Inf && return -Inf
    z = (b - m) / s
    return exp(-z^2)
end

function objective(x, y, lo)
    m, s = x[1], exp(x[2])
    total = 0.0
    @simd for i in eachindex(y)
        v = y[i]
        total += if v < lo || v > 2.0
            -Inf
        else
            mass = lo == -Inf ? endpoint(2.0, m, s) :
                endpoint(2.0, m, s) - endpoint(lo, m, s)
            -((v - m) / s)^2 / 2 - log(s) - mass
        end
    end
    return total
end

x, y = [0.7, log(0.6)], [1.2, 0.9, -0.5]
@testset "scalar lower bound" begin
    @testset "lo = $lo" for lo in (-Inf, -1.0)
        g = zero(x)
        Enzyme.autodiff(Reverse, objective, Active, Duplicated(x, g), Const(y), Const(lo))
        h = 1e-6
        fd = map(eachindex(x)) do j
            step = zero(x)
            step[j] = h
            (objective(x + step, y, lo) - objective(x - step, y, lo)) / (2h)
        end
        @test g ≈ fd atol = 1e-7
    end
end
