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

# Whether the loaded backend lets a rule read run metadata (the Enzyme
# extension refuses without dogfood repair `unchanged_custom_rule_roots`). A
# refused plate keeps ordinary reverse, and every numerical check still applies.
runs_scheduled() = RK._PR_INDEX_RUN_REFUSAL[] === nothing

function first_order(q, s, idx, x)
    value, gradient, scale = 0.0, zeros(length(q)), 0.0
    for i in eachindex(idx)
        j = idx[i]
        argument = s * q[j] + x[i]
        value += sin(argument)
        gradient[j] += s * cos(argument)
        scale += q[j] * cos(argument)
    end
    value, gradient, scale
end

@testset "index-run gradients on both adapters, and a backend's refusal" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    refusal = "this backend's rule cannot read run metadata"
    saved = RK._PR_INDEX_RUN_REFUSAL[]
    try
        for (n, run) in ((24, 8), (128, 32)), enabled in (false, true),
                refused in (enabled ? (false, true) : (false,))
            RK._PR_INDEX_RUN_REFUSAL[] = refused ? refusal : saved
            idx = [mod1(div(i-1, run)+1, 3) for i in 1:n]
            x = [sin(i/3)/2 for i in 1:n]
            q = [.1, -.4, .3]
            kernel = RK.with_bound_statistics(false) do
                with_plate_reverse(enabled) do
                    prepare_nonallocating(prepare(nonlinear_runs; bound=(; idx, x)), q, 1.2)
                end
            end
            refused && @test any(r -> last(r) == refusal, plate_reverse_report(kernel, (1, 2)))
            prepared = with_plate_reverse(enabled) do
                prepare_ad(kernel, backend, q, 1.2; active=(:q, :s))
            end
            rules = rules!(Any[], actual_program(prepared.call))
            grouped = count(rule -> calls_run_guard(RK._program_expression(rule.reverse.program)), rules)
            @test (grouped > 0) == (enabled && !refused && runs_scheduled())
            refused && @test isempty(rules)
            destination = (fill(71., 3), Ref(81.))
            for rep in 1:2, (point, scale) in ((q, 1.2), ([-.8, .6, 1.2], .7), (q, 1.4))
                pristine = copy(point)
                value, gradient, dscale = first_order(point, scale, idx, x)
                v, returned = ad_value_and_gradient!(prepared, destination, point, scale)
                @test v ≈ value rtol=3e-10 atol=3e-10
                @test destination[1] ≈ gradient rtol=3e-10 atol=3e-10
                @test destination[2][] ≈ dscale rtol=3e-10 atol=3e-10
                @test returned === destination
                @test point == pristine
            end
        end
    finally
        RK._PR_INDEX_RUN_REFUSAL[] = saved
    end
end

@testset "index-run higher derivatives preserve repeated-call semantics" begin
    forward = AutoEnzyme(; mode=Enzyme.Forward)
    reverse = AutoEnzyme(; mode=Enzyme.Reverse)
    # 24 cells take the two-pass adapter, whose reverse pass reads the run
    # metadata from the rule's arguments; 128 and 257 take one pass.
    for backend in (SecondOrder(forward, reverse), SecondOrder(reverse, forward)),
            enabled in (false, true), (n, run) in ((24, 8), (128, 32), (257, 32))
        @test (n < RK._PR_ONE_PASS_MIN_CELLS) == (n == 24)
        idx = [mod1(div(i-1, run)+1, 3) for i in 1:n]
        @test RK._index_run_metadata(idx)[4]
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
        @test (grouped > 0) == (enabled && runs_scheduled())
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
