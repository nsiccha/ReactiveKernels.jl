module PlateReverseIndexRunADTests
using ReactiveKernels, DifferentiationInterface, MutatingFunctions, Test
import Enzyme
const RK = ReactiveKernels

@kernel nonlinear_runs(q::Vector{Float64}, s::Float64,
        idx::Vector{Int}, x::Vector{Float64}) = begin
    gathered = ReactiveKernels._readonly_gather(q, idx)
    terms = plate(gathered, x) do v, y
        sin(s * v + y)
    end
    total::Float64 = sum(terms)
    return total
end

# Inspect the callable held by the AD preparation, not a regenerated report.
actual_program(x) = RK._program_expression(x)
actual_program(x::RK._ADNonAllocatingKernelCall) = actual_program(x.f)
actual_program(x::RK._ADNativeKernelCall) = actual_program(x.native)
actual_program(x::RK._ADPackedOperandsCall) = actual_program(x.call)
actual_program(x::RK._PackedOperandCall) = actual_program(x.call)
actual_program(x::RK._ExternalizedBoundArrayCall) = actual_program(x.f)
rules!(out, x) = out
rules!(out, x::RK._PlateSumRule) = push!(out, x)
function rules!(out, x::Expr)
    foreach(a -> rules!(out, a), x.args)
    out
end
rules!(out, x::RK._PlateProgram) = rules!(out, x.ast)
rules!(out, x::RK._PlateCall) = rules!(out, RK._program_expression(x.program))
function calls_run_guard(x)
    x isa Expr || return false
    (x.head === :call && first(x.args) === GlobalRef(RK, :_pr_run_disjoint)) ||
        any(calls_run_guard, x.args)
end

function derivatives(q, s, idx, x, direction)
    gradient, hvp = zeros(length(q)), zeros(length(q))
    for i in eachindex(idx)
        j = idx[i]
        argument = s * q[j] + x[i]
        gradient[j] += s * cos(argument)
        hvp[j] -= s^2 * sin(argument) * direction[j]
    end
    gradient, hvp
end

@testset "index-run higher derivatives preserve repeated-call semantics" begin
    forward = AutoEnzyme(; mode=Enzyme.Forward)
    reverse = AutoEnzyme(; mode=Enzyme.Reverse)
    for backend in (SecondOrder(forward, reverse), SecondOrder(reverse, forward)),
            enabled in (false, true), n in (128, 257)
        idx = [mod1(div(i-1, 32)+1, 3) for i in 1:n]
        x = [sin(i/3)/2 for i in 1:n]
        q, direction = [.1, -.4, .3], [.2, -.7, 1.1]
        kernel = RK.with_bound_statistics(false) do
            with_plate_reverse(enabled) do
                prepare_nonallocating(prepare(nonlinear_runs; bound=(; idx, x)), q, 1.2)
            end
        end
        prepared = with_plate_reverse(enabled) do
            prepare_ad_hvp(kernel, backend, (direction,), q, 1.2; active=:q)
        end
        rules = rules!(Any[], actual_program(prepared.call))
        grouped = count(rule -> calls_run_guard(RK._program_expression(rule.reverse.program)), rules)
        @test (grouped > 0) == enabled
        gradient_buffer, hvp_buffers = fill(71., 3), (fill(72., 3),)
        # The same preparation and destinations must survive changing points,
        # constants and directions, including returning to an earlier point.
        for rep in 1:2, (point, scale, tangent) in (
                (q, 1.2, direction), ([-.8, .6, 1.2], .7, [-.4, .1, .9]),
                (q, 1.4, -direction))
            pristine = (copy(point), copy(tangent))
            expected_g, expected_h = derivatives(point, scale, idx, x, tangent)
            @test only(ad_hvp(prepared, (tangent,), point, scale)) ≈ expected_h rtol=3e-10 atol=3e-10
            g, (h,) = ad_gradient_and_hvp(prepared, (tangent,), point, scale)
            @test g ≈ expected_g rtol=3e-10 atol=3e-10
            @test h ≈ expected_h rtol=3e-10 atol=3e-10
            fill!(gradient_buffer, 71.)
            fill!(only(hvp_buffers), 72.)
            returned = ad_gradient_and_hvp!(prepared, gradient_buffer, hvp_buffers,
                                            (tangent,), point, scale)
            @test returned[1] === gradient_buffer && returned[2] === hvp_buffers
            @test gradient_buffer ≈ expected_g rtol=3e-10 atol=3e-10
            @test only(hvp_buffers) ≈ expected_h rtol=3e-10 atol=3e-10
            @test point == pristine[1] && tangent == pristine[2]
        end
    end
end
end
