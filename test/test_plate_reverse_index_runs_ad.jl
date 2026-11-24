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
actual_program(x::RK._ADKernelCall) = actual_program(x.kernel)
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

@kernel optional_gather(q::Vector{Float64}, s::Float64, rows::Vector{Int}, optional::Vector{Int}, take::Bool) = begin
    a = ReactiveKernels._readonly_gather(q, rows)
    n = take ? length(optional) : 0
    chosen = view(optional, 1:n)
    b = ReactiveKernels._readonly_gather(q, chosen)
    left = plate(a) do v
        sin(s*v)
    end
    right = plate(b) do v
        cos(s*v)
    end
    result::Float64 = sum(left) + sum(right) + log(s)
    return result
end

function site_count(prepared)
    guards = visits = 0
    RK._index_run_visit(actual_program(prepared.call)) do ex
        if Meta.isexpr(ex, :call)
            visits += first(ex.args) === GlobalRef(RK, :_PRObserveSchedule)
            guards += first(ex.args) === GlobalRef(RK, :_pr_rule_applies)
        end
    end
    @test visits == 0
    guards
end

@testset "independent plate rules across live branch and size changes" begin
    q = [0.3sin(i) for i in 1:32]
    s = 1.2
    rows = [mod1(3i, length(q)) for i in 1:192]
    optional = [mod1(5i, length(q)) for i in 1:257]
    full = (q, s, rows, optional, true)
    kernel = prepare(optional_gather; have=(:q, :s, :rows, :optional, :take),
                     want=:result, exemplars=full)
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for base in (kernel, prepare_nonallocating(kernel, full...))
        prepare_at = (args...) -> with_plate_reverse(() ->
            prepare_ad(base, backend, args...; active=(:q, :s)), true)
        one = prepare_at(q, s, rows, optional, false)
        both = prepare_at(full...)
        short = prepare_at(q, s, rows, optional[1:2], true)
        empty_rows = prepare_at(q, s, Int[], Int[], false)
        @test site_count(one) == 1
        @test site_count(both) == 2
        @test site_count(short) == 1
        @test site_count(empty_rows) == 0
        larger = prepare_at(q, s, repeat(rows, 4), repeat(optional, 4), true)
        @test typeof(larger.call) === typeof(both.call)

        for prepared in (one, both, short, empty_rows),
                (active_rows, ids, take) in ((rows, optional, true),
                    (Int[], optional, true), (rows, [length(q)+1], false),
                    (Int[], Int[], false), (rows, optional[1:2], true))
            value = log(s)
            dq, ds = zero(q), inv(s)
            for i in active_rows
                value += sin(s*q[i])
                dq[i] += s*cos(s*q[i])
                ds += q[i]*cos(s*q[i])
            end
            if take
                for i in ids
                    value += cos(s*q[i])
                    dq[i] -= s*sin(s*q[i])
                    ds -= q[i]*sin(s*q[i])
                end
            end
            v, g = ad_value_and_gradient(prepared, q, s, active_rows, ids, take)
            @test v ≈ value && g[1] ≈ dq && g[2] ≈ ds
        end
        failed = prepare_at(q, -s, rows, optional, true)
        @test site_count(failed) == 0
        @test_throws DomainError ad_value_and_gradient(failed, q, -s, rows, optional, true)
        expected = ad_value_and_gradient(both, full...)
        v, (dq, ds) = ad_value_and_gradient(failed, full...)
        @test v ≈ expected[1] && dq ≈ expected[2][1] && ds ≈ expected[2][2]
    end
end

# Both gathers depend on q in the source. When Enzyme proves the entries of
# the second root constant, its grouped rule declines the call.
@kernel constant_entries(q::Vector{Float64}, s::Float64, idx::Vector{Int}) = begin
    live = ReactiveKernels._readonly_gather(q, idx)
    fixed = fill(0.25, length(q))
    inactive = ReactiveKernels._readonly_gather(fixed, idx)
    left = plate(live) do v
        sin(s * v)
    end
    right = plate(inactive) do v
        cos(s * v)
    end
    total::Float64 = sum(left) + sum(right)
    return total
end

@testset "preparation retains the other gather when Enzyme declines a rule" begin
    q, s = [0.15, -0.4, 0.3], 1.2
    idx = [mod1(i, length(q)) for i in 1:192]
    kernel = prepare(constant_entries; have=(:q, :s, :idx), want=:total, exemplars=(q, s, idx))
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    @test count(r -> last(r) == "rule", plate_reverse_report(kernel, (1, 2))) == 2
    # Ask the loaded Enzyme about the original two-site derivative. Its
    # activity analysis of fill(0.25, length(q)) differs between LLVM versions.
    args = (q, s, idx)
    trial, sites = with_plate_reverse() do
        RK._pr_prepared_call(nothing, true) do
            RK._ad_call(kernel, args, (:q, :s))
        end
    end
    @test [mark.ordinal for mark in sites.marks] == [1, 2]
    call, point, contexts = trial
    native = RK._ad_prepared_backend(Val(:gradient), backend)
    accepted = RK._pr_checking_sites(sites) do
        prep = RK._ad_operator_preparation(Val(:gradient), call, native, point, (), contexts)
        DifferentiationInterface.value_and_gradient(call, prep, native, point, contexts...)
        :both
    end
    @test accepted === :both || collect(keys(accepted)) == [1]
    expected_count = accepted === :both ? 2 : 1
    for warm in (false, true)
        prepared = with_plate_reverse() do
            prepare_ad(kernel, backend, q, s, idx; active=(:q, :s), warm)
        end
        @test site_count(prepared) == expected_count
        for ids in (idx, idx[1:7], reverse(idx))
            value, (dq, ds) = ad_value_and_gradient(prepared, q, s, ids)
            @test value ≈ sum(sin(s * q[i]) + cos(0.25s) for i in ids)
            @test dq ≈ [count(==(j), ids) * s * cos(s * q[j]) for j in eachindex(q)]
            @test ds ≈ sum(q[i] * cos(s * q[i]) - 0.25sin(0.25s) for i in ids)
        end
    end
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
            @test (grouped > 0) == (enabled && !refused && runs_scheduled() && n >= RK._PR_ONE_PASS_MIN_CELLS)
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
    # With these bound indices the 24-row plate runs inline. The two larger
    # plates can use the one-pass schedule at the exemplar.
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
        @test (grouped > 0) == (enabled && runs_scheduled() && n >= RK._PR_ONE_PASS_MIN_CELLS)
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
