# Standalone CPU reproducer: only Enzyme is required.
# Run with julia --project=<environment with Enzyme> benchmark/repro_enzyme_open_bound_gradient.jl
#
# Reduced from an RKPPL truncated Normal whose scalar bound datum is infinite:
# `truncated.(Normal.(a, s), lo, 2.0)` with `lo = -Inf`, or
# `truncated.(Normal.(a, s), 0.0, hi)` with `hi = Inf`
# (`packages/ReactiveKernelsPPL/test/test_evidence_edgecases.jl`). The loop
# guards the omitted endpoint, so no arm that runs reads it. LLVM still moves
# the endpoint's standardization `(lo - m) / s` out of the loop, where it is
# infinite, and ordinary Enzyme Reverse multiplies its zero adjoint by an
# infinite partial. The value is right in every case.
#
# "scalar lower bound": the standardized endpoint is used only in the skipped
# arm. Enzyme.jl 987a9463 with libEnzyme 2528afe8 returns [NaN, NaN]. Enzyme
# dogfood 4ff5e0e4 (libEnzyme 593c7937, repair `conditional_use_adjoints`,
# Enzyme snag `reverse-simd-laz-ff510820`) returns the gradient.
#
# "compared interval width": the skipped arm first compares the interval width
# `b - a`, as `_normal_direct_interval` does before the central log mass. The
# width is computed from the infinite endpoint outside the loop and is used
# only by that comparison. Enzyme dogfood 4ff5e0e4 still returns [finite, NaN]
# for either open bound, as the RKPPL sampler did with it. The same arm without
# the width comparison ("plain" below) gives the gradient. Dogfood 9770912c
# (libEnzyme 078431a8, repair `compared_operand_adjoints`, Enzyme snag
# `reverse-nan-hois-9f3292ff`) passes all 10 checks here, and the sampler's
# infinite-bound testset in `test_evidence_edgecases.jl` passes 49 of 49.
# Recorded on a Linux fleet host, Julia 1.10.12, 2026-11-20.
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

# The finite-bound arm compares the interval width before it uses the lower
# endpoint. `width = false` keeps the arm and drops the width from the test.
function lower_objective(x, y, lo, ::Val{width}) where {width}
    m, s = x[1], exp(x[2])
    total = 0.0
    @simd for i in eachindex(y)
        v = y[i]
        total += if v < lo || v > 2.0
            -Inf
        else
            b = (2.0 - m) / s
            mass = if lo == -Inf
                exp(-b * b)
            else
                a = (lo - m) / s
                (width ? b - a >= 0.125 : a >= -8) ? exp(-a * a) : 0.0
            end
            -((v - m) / s)^2 / 2 - log(s) - mass
        end
    end
    return total
end

# The same shape with the upper endpoint open.
function upper_objective(x, y, hi, ::Val{width}) where {width}
    m, s = x[1], exp(x[2])
    total = 0.0
    @simd for i in eachindex(y)
        v = y[i]
        total += if v < 0.0 || v > hi
            -Inf
        else
            a = (0.0 - m) / s
            mass = if hi == Inf
                exp(-a * a)
            else
                b = (hi - m) / s
                (width ? b - a >= 0.125 : b <= 8) ? exp(-b * b) : 0.0
            end
            -((v - m) / s)^2 / 2 - log(s) - mass
        end
    end
    return total
end

function gradient_and_difference(f, x, arguments...)
    g = zero(x)
    Enzyme.autodiff(Reverse, f, Active, Duplicated(x, g), map(Const, arguments)...)
    h = 1e-6
    fd = map(eachindex(x)) do j
        step = zero(x)
        step[j] = h
        (f(x + step, arguments...) - f(x - step, arguments...)) / (2h)
    end
    return g, fd
end

x = [0.7, log(0.6)]
@testset "scalar lower bound" begin
    @testset "lo = $lo" for lo in (-Inf, -1.0)
        g, fd = gradient_and_difference(objective, x, [1.2, 0.9, -0.5], lo)
        @test g ≈ fd atol = 1e-7
    end
end
@testset "compared interval width" begin
    @testset "$name, lo = $lo" for (name, width) in (("width", Val(true)), ("plain", Val(false))),
            lo in (-Inf, -1.0)
        g, fd = gradient_and_difference(lower_objective, x, [1.2, 0.9, -0.5], lo, width)
        @test g ≈ fd atol = 1e-7
    end
    @testset "$name, hi = $hi" for (name, width) in (("width", Val(true)), ("plain", Val(false))),
            hi in (Inf, 2.0)
        g, fd = gradient_and_difference(upper_objective, x, [1.2, 0.9, 0.5], hi, width)
        @test g ≈ fd atol = 1e-7
    end
end
