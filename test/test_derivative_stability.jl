module DerivativeStabilityTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# The check of the derivative native Enzyme compiles for a prepared program
# (snag `rk-ad-pass-type-b179e693`, user decision
# `ReactiveKernels:snag.rk-ad-pass-type-b179e693/decisions/2026-11-17T22-00-23-356-1auv5ma`):
# `type_stability` at `:all` covers the derivative's run-time dispatch and boxes,
# `ad_allocations` its allocations, and `ad_derivative_report` lists every
# finding. Reading the derivative needs Enzyme's dogfood repair
# `derivative_report`; without it the derivative is not checked.

const HOOK = isdefined(Enzyme, :derivative_report) &&
    isdefined(Enzyme, :has_dogfood_repair) && Enzyme.has_dogfood_repair(:derivative_report)

@kernel linear_gaussian(a::Float64, b::Float64, s::Float64, x::Vector{Float64},
                        y::Vector{Float64}) = begin
    mu = a .+ b .* x
    total::Float64 = sum(-0.5 .* ((y .- mu) ./ s) .^ 2) - length(y) * log(s)
    return total
end

@kernel plate_location(a::Float64, s::Float64, t::Vector{Float64},
                       y::Vector{Float64}) = begin
    mu = t .* exp(a)
    pointwise = plate(y, mu) do yi, mi
        -0.5 * ((yi - mi) / s)^2 - log(s)
    end
    total::Float64 = sum(pointwise)
    return total
end

# A plate over RK's scalar rule: the rule's tape over the plate is a native
# allocation whose size is the plate's length.
@kernel logistic_rule(beta::Vector{Float64}, X::Matrix{Float64},
                      y::Vector{Float64}) = begin
    eta = X * beta
    pointwise = plate(y, eta) do yi, ei
        yi * ei - rk_log1pexp(ei)
    end
    total::Float64 = sum(pointwise)
    return total
end

# A helper that reads a global of abstract type: its call dispatches at run
# time, in the primal (found at `:all`) and in the derivative.
const SCALE = Ref{Any}(2.0)
hidden(x) = (SCALE[] * x)::Float64
@kernel hidden_scale(a::Float64, y::Vector{Float64}) = begin
    mu = hidden(a)
    total::Float64 = -sum(abs2, y .- mu)
    return total
end

@kernel quadratic(q::Vector{Float64}, scale::Float64) = begin
    total::Float64 = scale * sum(abs2, q) / 2
    return total
end

const n = 16
const x = collect(range(-1, 1; length = n))
const y = 0.3 .+ 0.7 .* x .+ 0.1 .* sin.(1:n)
const X = hcat(ones(n), x)
const yb = Float64.(sin.(1:n) .> 0)
const backend = AutoEnzyme(; mode = Enzyme.Reverse)

const quiet = Base.CoreLogging.Warn
derivative_warning(pattern) =
    (:warn, Regex("ReactiveKernels: the reverse-mode derivative .*" * pattern, "s"))

function refusal(f)
    try
        f()
    catch err
        return err
    end
    nothing
end

@testset "ad_allocations: choices" begin
    @test_throws ArgumentError prepare_ad(plate_location, backend, 0.1, 1.2, x, y;
        want = :total, active = :a, ad_allocations = :sometimes)
end

@testset "unstable: code run undifferentiated counts with Enzyme's reachability repair" begin
    # Enzyme's repair `derivative_report_reachability` marks the code of a
    # function only called on an error path `throw_only`; with it, `unstable`
    # also counts calls in code the derivative runs undifferentiated.
    finding(sweep; throw_only = false) = (; kind = :dynamic_dispatch,
        callee = "jl_invoke", in_function = "f", sweep,
        frames = Base.StackTraces.StackFrame[], in_loop = false, throw_only,
        bytes = nothing, count = 1)
    derivative = (; func = Nothing, mode = :reverse, note = "",
        findings = [finding(:primal), finding(:forward), finding(:primal; throw_only = true)])
    flag = ReactiveKernels._DERIVATIVE_REPORT_REACHABILITY
    saved = flag[]
    try
        flag[] = false
        @test [last(p).sweep for p in ADDerivativeReport(nothing, [derivative]).unstable] ==
            [:forward]
        flag[] = true
        @test [last(p).sweep for p in ADDerivativeReport(nothing, [derivative]).unstable] ==
            [:primal, :forward]
    finally
        flag[] = saved
    end
end

if !HOOK
    @testset "without Enzyme's derivative report the derivative is not checked" begin
        # Never refused, warned once where a check was asked for.
        kernel = prepare(linear_gaussian)
        prepared = prepare_ad(kernel, backend, 0.1, 0.5, 1.2, x, y; active = :a,
                              ad_allocations = :error)
        @test ad_value_and_gradient(prepared, 0.1, 0.5, 1.2, x, y)[1] ≈ kernel(0.1, 0.5, 1.2, x, y)
        @test_throws ArgumentError ad_derivative_report(prepared, 0.1, 0.5, 1.2, x, y)
    end
else

@testset "the derivative's allocations: report, warn, refuse" begin
    args = (0.1, 0.5, 1.2, x, y)
    kernel = prepare(linear_gaussian)
    # The broadcast temporaries the primal allocates are allocated by its
    # derivative too.
    prepared = @test_logs min_level = quiet prepare_ad(kernel, backend, args...; active = :a,
                                                       type_stability = :error)
    report = ad_derivative_report(prepared, args...)
    @test report isa ADDerivativeReport
    @test report.value[1] ≈ kernel(args...)
    @test any(d -> d.func == typeof(prepared.call), report.derivatives)
    @test isempty(report.unstable)
    @test !isempty(report.allocations)
    @test all(p -> last(p).kind in (:gc_allocation, :native_allocation), report.allocations)
    @test all(p -> !last(p).throw_only, report.allocations)

    @test_logs derivative_warning("allocates:\n  - an allocation on Julia's heap") match_mode = :any prepare_ad(
        kernel, backend, args...; active = :a, ad_allocations = :warn)
    e = refusal(() -> prepare_ad(kernel, backend, args...; active = :a, ad_allocations = :error))
    @test e isa DerivativeAllocationError
    @test occursin("an allocation on Julia's heap", only(e.messages))
    @test occursin("ad_allocations = :error", sprint(showerror, e))
    # A setting other than `:ignore` checks the derivative at every level.
    @test_throws DerivativeAllocationError prepare_ad(kernel, backend, args...; active = :a,
        ad_allocations = :error, type_stability_level = :values)
    # The preference, scoped like `type_stability`.
    @test_throws DerivativeAllocationError ReactiveKernels._with_type_stability(
        () -> prepare_ad(kernel, backend, args...; active = :a), nothing, nothing, :error)

    # A plate cell whose derivative allocates nothing passes `:error`.
    plate_args = (0.1, 1.2, x, y)
    plate_kernel = prepare(plate_location)
    prepared = @test_logs min_level = quiet prepare_ad(plate_kernel, backend, plate_args...;
        active = :a, type_stability = :error, ad_allocations = :error)
    @test isempty(ad_derivative_report(prepared, plate_args...).allocations)

    # A custom rule's tape over a plate: native, its size known only at run time,
    # which `@allocated` does not count.
    rule_args = ([0.2, -0.3], X, yb)
    prepared = prepare_ad(prepare(logistic_rule), backend, rule_args...; active = :beta)
    report = ad_derivative_report(() -> ad_value_and_gradient!(prepared, zeros(2), rule_args...))
    @test any(p -> last(p).kind === :native_allocation && last(p).bytes === nothing,
              report.allocations)
end

@testset "the derivative's run-time dispatch at type_stability_level = :all" begin
    args = (0.1, y)
    kernel = prepare(hidden_scale; type_stability_level = :rk)
    # `:rk` checks the primal only: the helper is a leaf, its call statically
    # dispatched.
    prepared = @test_logs min_level = quiet prepare_ad(kernel, backend, args...; active = :a,
        type_stability = :error, type_stability_level = :rk)
    report = ad_derivative_report(prepared, args...)
    @test !isempty(report.unstable)
    @test any(p -> last(p).kind in (:dynamic_dispatch, :enzyme_runtime_fallback),
              report.unstable)
    @test any(p -> any(frame -> frame.func === :hidden, last(p).frames), report.unstable)
    @test ad_gradient(prepared, args...) ≈ -2 * sum(2.0 .* 0.1 .- y) * 2.0

    # At `:all` both the primal and its derivative are reported.
    @test_logs derivative_warning("is not type stable:\n  - a call .* at run time") match_mode = :any prepare_ad(
        kernel, backend, args...; active = :a)
    # Refused with `type_stability = :error`: the primal audit refuses first, so
    # the derivative's refusal is checked on the report itself.
    e = refusal(() -> ReactiveKernels._report_derivative(
        report, typeof(prepared.call), :error, :ignore))
    @test e isa TypeInstabilityError
    @test occursin("derivative native Enzyme compiles", join(e.messages))
    @test occursin("hidden", join(e.messages))
end

@testset "every prepared operator: the derivative check runs at preparation" begin
    args = (0.1, 1.2, x, y)
    pullback = @test_logs min_level = quiet prepare_ad_pullback(plate_location, backend,
        1.0, args...; want = :total, active = :a, type_stability = :error)
    @test !isempty(ad_derivative_report(pullback, 1.0, args...).derivatives)
    # Forward over Reverse: the inner derivative is part of the outer one's code.
    second = SecondOrder(AutoEnzyme(; mode = Enzyme.Forward), backend)
    q, tangents = [0.3, -1.2, 2.0], ([1.0, 0.5, -0.25],)
    hvp = @test_logs min_level = quiet prepare_ad_hvp(quadratic, second, tangents,
        q, 1.7; want = :total, active = :q, type_stability = :error)
    report = ad_derivative_report(hvp, tangents, q, 1.7)
    @test !isempty(report.derivatives)
    @test isempty(report.unstable)
    @test only(ad_hvp(hvp, tangents, q, 1.7)) ≈ 1.7 .* only(tangents)
end

end # HOOK

end # module
