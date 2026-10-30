module TypeStabilityTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

include(joinpath(@__DIR__, "type_stability_audit.jl"))
using .TypeStabilityAudit: findings, ad_call_findings, rk_or_generated

# RK tries to guarantee that the programs it prepares are type stable and
# concretely typed (principle review
# `ReactiveKernels:snag.joint-nlme-enzym-f8d3cb5f/decisions/2026-10-12T20-09-11-062-114x8kz`;
# audit and warning-only client check, user decision `…/2026-10-12T20-13-15-811-1hui2qi`).
# The corpus covers the lowering's constructs with concretely typed client
# code: whatever non-concrete value or dynamic call appears in a prepared
# native body or a prepared gradient call is RK's own.
module Corpus
using ReactiveKernels

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

# Per-subject cells over ragged data, each running an inner plate.
@kernel subject_cells(a::Float64, s::Float64, ts::Vector{Vector{Float64}},
                      ys::Vector{Vector{Float64}}) = begin
    subtotals = plate(ts, ys) do ti, yi
        mi = ti .* exp(a)
        inner = plate(yi, mi) do y, m
            -0.5 * ((y - m) / s)^2 - log(s)
        end
        sum(inner)
    end
    total::Float64 = sum(subtotals)
    return total
end

# A plate whose array-valued cells feed a second plate.
@kernel two_stage(a::Float64, s::Float64, ts::Vector{Vector{Float64}},
                  ys::Vector{Vector{Float64}}) = begin
    mus = plate(ts) do ti
        ti .* exp(a)
    end
    subtotals = plate(ys, mus) do yi, mi
        inner = plate(yi, mi) do y, m
            -0.5 * ((y - m) / s)^2 - log(s)
        end
        sum(inner)
    end
    total::Float64 = sum(subtotals)
    return total
end

@kernel scan_plate(gain::Float64, xs::Vector{Float64}, ys::Vector{Float64}) = begin
    states = scan(xs; init = 0.0) do carry, x
        next = 0.9 * carry + gain * x
        (next, next)
    end
    pointwise = plate(ys, states) do y, m
        -0.5 * (y - m)^2
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel scan_with_init(gain::Float64, xs::Vector{Float64}) = begin
    path = scan(xs; init = 1.0, include_init = true) do carry, x
        next = carry * exp(gain * x)
        (next, next)
    end
    total::Float64 = sum(abs2, path)
    return total
end

# The coefficient-outer `evalpoly` cell path.
@kernel evalpoly_plate(c::Vector{Float64}, xs::Vector{Float64}) = begin
    ys = plate(xs) do x
        evalpoly(x, c)
    end
    total::Float64 = sum(ys)
    return total
end

@kernel shifted(location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    logpdf(x::Float64)::Float64 = begin
        z::Float64 = (x - location) / scale
        -0.5 * z^2 - log_scale
    end
end

# A plate cell calling another kernel's endpoint.
@kernel embedded_endpoint(m::Vector{Float64}, xs::Vector{Float64}) = begin
    cells = plate(xs, m) do x, mi
        (shifted(mi, 0.7)).logpdf(x)
    end
    total::Float64 = sum(cells)
    return total
end

@kernel running_total(xs, gain) = begin
    updates = scan(xs; init = 0.0) do carry, x
        next = carry + x * gain
        (next, next)
    end
    total = sum(updates)
    return total
end

# A plate cell calling a scan kernel in a lazy arm.
@kernel group_totals(gain::Float64, groups::Vector{Vector{Float64}}) = begin
    cells = plate(groups) do xs
        gain > 0.0 ? running_total(xs, gain) : 0.0
    end
    total::Float64 = sum(cells)
    return total
end

@kernel logistic_design(beta::Vector{Float64}, X::Matrix{Float64},
                        y::Vector{Float64}) = begin
    eta = X * beta
    pointwise = plate(y, eta) do yi, ei
        yi * ei - log1p(exp(ei))
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel concatenated(a::Float64, b::Float64) = begin
    M = [-a 0.0; a -b]
    total::Float64 = sum(M * [1.0, 2.0])
    return total
end

# A data-bound branch in a plate cell (split per arm when `g` is bound).
@kernel branch_cell(mu::Vector{Float64}, y::Vector{Float64}, g::Vector{Int}) = begin
    pointwise = plate(y, mu, g) do yi, mi, gi
        cell::Float64 = gi == 0 ? -0.5 * (yi - mi)^2 : 0.0
        cell
    end
    total::Float64 = sum(pointwise)
    return total
end

# The same branch with its pointwise values read in lane order: the arms'
# values are reassembled (`_LaneAssemble`).
@kernel branch_pointwise(mu::Vector{Float64}, y::Vector{Float64},
                         g::Vector{Int}) = begin
    pointwise = plate(y, mu, g) do yi, mi, gi
        cell::Float64 = gi == 0 ? -0.5 * (yi - mi)^2 : mi
        cell
    end
    total::Float64 = sum(pointwise .* y)
    return total
end

@kernel destructured(a::Float64, x::Vector{Float64}) = begin
    (lo, hi) = extrema(x .* a)
    total::Float64 = hi - lo + a^2
    return total
end
end # module Corpus

module Uninferred
using ReactiveKernels
# Julia infers this helper's result as `Any`.
uninferred_curve(t, a) = Base.inferencebarrier(t .* exp(a))

@kernel location(a::Float64, s::Float64, t::Vector{Float64},
                 y::Vector{Float64}) = begin
    mu = uninferred_curve(t, a)
    pointwise = plate(y, mu) do yi, mi
        -0.5 * ((yi - mi) / s)^2 - log(s)
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel declared(a::Float64, s::Float64, t::Vector{Float64},
                 y::Vector{Float64}) = begin
    mu::Vector{Float64} = uninferred_curve(t, a)
    pointwise = plate(y, mu) do yi, mi
        -0.5 * ((yi - mi) / s)^2 - log(s)
    end
    total::Float64 = sum(pointwise)
    return total
end

# A helper that dispatches at run time inside but asserts its result: every
# value of the program infers concretely, its code does not.
const SCALE = Ref{Any}(2.0)
hidden_scale(x) = (SCALE[] * x)::Float64

@kernel hidden(a::Float64, y::Vector{Float64}) = begin
    mu = hidden_scale(a)
    total::Float64 = sum((y .- mu) .^ 2)
    return total
end

# A helper that logs on cold branches, as SpecialFunctions' `beta_inc` does on
# underflow (snag `type-stability-a-a6941de1`). The expansion of Julia's
# logging macros dispatches at run time by design, the record's fields
# included: it is Julia's code, not the helper's.
function logged_scale(x)
    if x == 0.0
        @debug "zero scale" x
        return 1.0
    end
    x < -10.0 && @warn "large negative scale" shown = string(Base.inferencebarrier(x))
    return 2.0 * x
end

@kernel logged(a::Float64, y::Vector{Float64}) = begin
    mu = logged_scale(a)
    total::Float64 = sum((y .- mu) .^ 2)
    return total
end

# The same logging beside a run-time dispatch of the helper's own.
function logged_hidden_scale(x)
    x == 0.0 && @debug "zero scale" x
    return (SCALE[] * x)::Float64
end

@kernel logged_hidden(a::Float64, y::Vector{Float64}) = begin
    mu = logged_hidden_scale(a)
    total::Float64 = sum((y .- mu) .^ 2)
    return total
end

# Two independent uninferred values: the location and the scale.
uninferred_scale(s) = Base.inferencebarrier(abs(s))

@kernel two_roots(a::Float64, s::Float64, t::Vector{Float64},
                  y::Vector{Float64}) = begin
    mu = uninferred_curve(t, a)
    sd = uninferred_scale(s)
    pointwise = plate(y, mu) do yi, mi
        -0.5 * ((yi - mi) / sd)^2 - log(sd)
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel untyped(a, s, t, y) = begin
    mu = uninferred_curve(t, a)
    total = sum(-0.5 .* ((y .- mu) ./ s) .^ 2 .- log(s))
    return total
end

@kernel subject_cells(a::Float64, s::Float64, ts::Vector{Vector{Float64}},
                      ys::Vector{Vector{Float64}}) = begin
    mus = plate(ts) do ti
        uninferred_curve(ti, a)
    end
    subtotals = plate(ys, mus) do yi, mi
        inner = plate(yi, mi) do y, m
            -0.5 * ((y - m) / s)^2 - log(s)
        end
        sum(inner)
    end
    total::Float64 = sum(subtotals)
    return total
end

# A scan step whose lazy arm calls the helper.
@kernel step_arm(a::Float64, kinds::Vector{Int}, xs::Vector{Float64}) = begin
    path = scan(kinds, xs; init = 0.0) do carry, kind, x
        next = kind == 2 ? uninferred_curve(x, a) : carry + x
        (next, next)
    end
    total::Float64 = sum(path)
    return total
end
end # module Uninferred

const C = Corpus
const t = collect(range(0.5, 4.0; length = 7))
const y = 1.3 .* t .+ 0.1 .* sin.(1:7)
const ts = [t[1:3], t[4:7], t[2:5]]
const ys = [y[1:3], y[4:7], y[2:5]]
const X = [1.0 0.5; 1.0 -0.25; 1.0 2.0; 1.0 0.0]
const yb = [1.0, 0.0, 1.0, 1.0]
const g = [0, 1, 0, 0, 1, 0, 0]

# (name, spec, bound data, remaining arguments, active port)
const CASES = (
    ("linear gaussian", C.linear_gaussian, (;), (0.1, 0.7, 0.8, t, y), :b),
    ("linear gaussian, bound data", C.linear_gaussian, (; x = t, y), (0.1, 0.7, 0.8), :b),
    ("plate", C.plate_location, (;), (0.1, 0.8, t, y), :a),
    ("plate, bound data", C.plate_location, (; t, y), (0.1, 0.8), :a),
    ("subject cells", C.subject_cells, (;), (0.1, 0.8, ts, ys), :a),
    ("subject cells, bound data", C.subject_cells, (; ts, ys), (0.1, 0.8), :s),
    ("two stages", C.two_stage, (;), (0.1, 0.8, ts, ys), :a),
    ("two stages, bound data", C.two_stage, (; ts, ys), (0.1, 0.8), :a),
    ("scan into plate", C.scan_plate, (;), (0.3, t, y), :gain),
    ("scan including its seed", C.scan_with_init, (;), (0.1, t), :gain),
    ("evalpoly cells", C.evalpoly_plate, (;), ([0.5, -1.0, 0.25], t), :c),
    ("embedded endpoint", C.embedded_endpoint, (;), (y, t), :m),
    ("scan child in a lazy arm", C.group_totals, (;), (0.4, ts), :gain),
    ("design matrix", C.logistic_design, (;), ([0.2, -0.3], X, yb), :beta),
    ("design matrix, bound data", C.logistic_design, (; X, y = yb), ([0.2, -0.3],), :beta),
    ("dense concatenation", C.concatenated, (;), (0.3, 0.7), :a),
    ("data-bound branch", C.branch_cell, (; y, g), (y .+ 0.1,), :mu),
    ("data-bound branch, lane order", C.branch_pointwise, (; y, g), (y .+ 0.1,), :mu),
    ("destructured outputs", C.destructured, (;), (0.5, t), :a),
)

prepared_kernel(spec, bound) = isempty(bound) ? prepare(spec) : prepare(spec; bound)

function report(name, found)
    isempty(found) && return true
    println(stderr, "type-stability findings for ", name, ":")
    foreach(f -> println(stderr, "  ", f), found)
    false
end

@testset "prepared programs are concretely typed" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    recurse = rk_or_generated(Corpus)
    for (name, spec, bound, args, active) in CASES
        @testset "$name" begin
            kernel = @test_logs min_level = Base.CoreLogging.Warn prepared_kernel(spec, bound)
            spec === C.branch_pointwise &&
                @test any(op -> op isa ReactiveKernels._LaneAssemble, kernel.callables)
            @test report("$name (native body)",
                         findings(kernel, map(Core.Typeof, args); recurse))
            prepared = @test_logs min_level = Base.CoreLogging.Warn prepare_ad(
                kernel, backend, args...; active)
            @test report("$name (gradient call)",
                         ad_call_findings(prepared, args...; recurse))
        end
    end
end

uninferred_warning(name) = (:warn, Regex("`$name` is not concretely typed"))

# The value check alone, `type_stability_level = :values`, for the preparations
# `f` makes.
values_level(f) = ReactiveKernels._with_type_stability(f, nothing, :values)

@testset "uninferred client values: findings and preparation warning" begin
values_level() do
    U = Uninferred
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    args = (0.1, 0.8, t, y)

    # The audit detects the client's uninferred result in both programs.
    kernel = @test_logs uninferred_warning("mu") prepare(U.location)
    found = findings(kernel, map(Core.Typeof, args); recurse = rk_or_generated(U))
    @test any(f -> f.kind === :root && occursin("uninferred_curve", f.statement), found)
    @test any(f -> f.kind === :dynamic, found)
    prepared = @test_logs uninferred_warning("mu") prepare_ad(
        kernel, backend, args...; active = :a)
    @test !isempty(ad_call_findings(prepared, args...; recurse = rk_or_generated(U)))
    # Preparation warns and continues: the primal still evaluates.
    @test kernel(args...) ≈ prepare(Corpus.plate_location)(args...)

    # Bound data types the recipe from the bound values.
    @test_logs uninferred_warning("mu") prepare(U.location; bound = (; t, y))

    # A declared, concretely typed local asserts the value: no warning.
    @test_logs min_level = Base.CoreLogging.Warn prepare(U.declared)

    # Unknown HAVE types leave the walk to the AD exemplars.
    untyped = @test_logs min_level = Base.CoreLogging.Warn prepare(U.untyped)
    @test_logs uninferred_warning("mu") prepare_ad(untyped, backend, args...; active = :a)

    # A cell's uninferred result is named with the plate that encloses it.
    @test_logs (:warn, r"The cell value of `mus` is not concretely typed.*uninferred_curve\(ti, a\)") prepare(U.subject_cells)

    # A scan step's uninferred value is named with the scan that encloses it.
    @test_logs (:warn, r"`next` in the body of `path` is not concretely typed.*uninferred_curve\(x, a\)") prepare(U.step_arm)
end end

# `type_stability = :error` refuses the program instead of warning (snag
# `type-stability-c-7eeae56d`, user decision
# `ReactiveKernels:snag.type-stability-c-7eeae56d/decisions/2026-10-27T20-42-40-959-1jkb0is`).
refusal(f) = try
    f()
    nothing
catch e
    e
end

# A project environment, ahead of the active one on the load path, with the
# given ReactiveKernels preferences (`"type_stability" => "error"`). A session
# reads the preferences at its first preparation; forgetting that read stands in
# for a new session.
forget_type_stability_preferences() =
    foreach(preference -> preference[] = nothing,
            ReactiveKernels._TYPE_STABILITY_PREFERENCES)

function with_type_stability_preferences(f, preferences::Pair...)
    dir = mktempdir()
    uuid = string(Base.PkgId(ReactiveKernels).uuid)
    entries = join(("$name = $(repr(value))" for (name, value) in preferences), "\n")
    write(joinpath(dir, "Project.toml"), """
        [deps]
        ReactiveKernels = "$uuid"

        [preferences.ReactiveKernels]
        $entries
        """)
    pushfirst!(LOAD_PATH, dir)
    forget_type_stability_preferences()
    try
        f()
    finally
        popfirst!(LOAD_PATH)
        forget_type_stability_preferences()
        rm(dir; recursive = true)
    end
end

@testset "uninferred client values: refusal with type_stability = :error" begin
values_level() do
    U = Uninferred
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    args = (0.1, 0.8, t, y)

    # The error names every uninferred value of the preparation, and nothing is
    # logged.
    e = @test_logs refusal(() -> prepare(U.two_roots; type_stability = :error))
    @test e isa TypeInstabilityError
    @test length(e.messages) == 2
    @test any(m -> occursin(r"`mu` is not concretely typed.*uninferred_curve\(t, a\)", m),
              e.messages)
    @test any(m -> occursin(r"`sd` is not concretely typed.*uninferred_scale\(s\)", m),
              e.messages)
    shown = sprint(showerror, e)
    @test startswith(shown,
        "TypeInstabilityError: the prepared program is not type stable (2 findings)")
    @test all(m -> occursin(m, shown), e.messages)

    # Every preparation is refused, not only the first in the process.
    @test_throws TypeInstabilityError prepare(U.two_roots; type_stability = :error)
    @test_throws TypeInstabilityError prepare(U.location; type_stability = :error)
    @test_throws TypeInstabilityError prepare(U.location; type_stability = :error,
                                               bound = (; t, y))
    @test_throws TypeInstabilityError prepare(U.subject_cells; type_stability = :error)
    @test_throws TypeInstabilityError prepare(U.step_arm; type_stability = :error)

    # A concretely typed program prepares, and a declared local still asserts.
    @test_logs min_level = Base.CoreLogging.Warn prepare(
        Corpus.plate_location; type_stability = :error)
    @test_logs min_level = Base.CoreLogging.Warn prepare(U.declared; type_stability = :error)

    # The AD preparation checks its exemplars in the same mode.
    untyped = @test_logs min_level = Base.CoreLogging.Warn prepare(U.untyped)
    @test_throws TypeInstabilityError prepare_ad(untyped, backend, args...; active = :a,
                                                  type_stability = :error)
    @test_throws TypeInstabilityError prepare_ad(U.location, backend, args...;
        active = :a, want = :total, type_stability = :error)

    # A preparation cached under `:warn` is not handed to an `:error` call.
    cache = PreparationCache()
    @test_logs uninferred_warning("mu") prepare!(cache, U.location)
    @test_throws TypeInstabilityError prepare!(cache, U.location; type_stability = :error)
    # The graph holds this binding under `:warn`; its rebindings are not
    # checked again, so an `:error` binding must not reuse it.
    Base.CoreLogging.with_logger(() -> prepare(U.location; bound = (; t, y)),
                                 Base.CoreLogging.NullLogger())
    @test_throws TypeInstabilityError prepare(U.location; bound = (; t, y),
                                               type_stability = :error)
    @test_throws TypeInstabilityError prepare!(cache, U.location; bound = (; t, y),
                                                type_stability = :error)

    # The keyword is checked.
    @test_throws ArgumentError prepare(Corpus.plate_location; type_stability = :strict)
    @test_throws ArgumentError prepare(Corpus.plate_location; type_stability = "error")
end end

@testset "uninferred client values: the type_stability preference" begin
    U = Uninferred
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    args = (0.1, 0.8, t, y)
    with_type_stability_preferences("type_stability" => "error") do
        @test_throws TypeInstabilityError prepare(U.location)
        @test_throws TypeInstabilityError prepare(U.location; bound = (; t, y))
        @test_throws TypeInstabilityError prepare_ad(prepare(U.untyped), backend, args...;
                                                      active = :a)
        # A call's keyword takes precedence, nested preparations included.
        @test_logs uninferred_warning("mu") match_mode = :any prepare(
            U.location; type_stability = :warn)
        @test_logs uninferred_warning("mu") match_mode = :any prepare_ad(
            U.location, backend, args...; active = :a, want = :total,
            type_stability = :warn)
        @test_logs min_level = Base.CoreLogging.Warn prepare(Corpus.plate_location)
    end
    with_type_stability_preferences("type_stability" => "warn") do
        @test_logs uninferred_warning("mu") match_mode = :any prepare(U.location)
    end
    with_type_stability_preferences("type_stability" => "strict") do
        @test_throws ArgumentError prepare(Corpus.plate_location)
    end
    # Without a preference, preparation warns.
    @test_logs uninferred_warning("mu") match_mode = :any prepare(U.location)
end

# `type_stability_level` sets how far the check reaches (user decision
# `ReactiveKernels:snag.type-stability-c-7eeae56d/decisions/2026-10-27T21-12-20-388-04ks7hd`:
# "everything", in levels "either everything rk, or everything everything").
code_warning(pattern) = (:warn, Regex("ReactiveKernels: .*" * pattern, "s"))

@testset "type-stability levels: values, the code RK prepares, everything" begin
    U = Uninferred
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    hidden_args = (0.1, y)

    # Every value of `hidden` infers concretely: only `:all` audits the body of
    # `hidden_scale`, a client function, and finds its run-time dispatch.
    @test_logs min_level = Base.CoreLogging.Warn prepare(U.hidden;
                                                          type_stability_level = :values)
    @test_logs min_level = Base.CoreLogging.Warn prepare(U.hidden;
                                                          type_stability_level = :rk)
    hidden_finding = code_warning("`[^`]*hidden_scale @ [^`]*` is not type stable:" *
                                  ".*is a call dispatched at run time")
    @test_logs hidden_finding prepare(U.hidden; type_stability_level = :all)
    # `:all` is the default.
    @test_logs hidden_finding prepare(U.hidden)
    e = refusal(() -> prepare(U.hidden; type_stability = :error))
    @test e isa TypeInstabilityError
    @test length(e.messages) == 1
    @test occursin("hidden_scale", only(e.messages))
    # The AD preparation audits the callable its backend differentiates.
    kernel = values_level(() -> prepare(U.hidden))
    @test_throws TypeInstabilityError prepare_ad(kernel, backend, hidden_args...;
                                                 active = :a, type_stability = :error)
    @test_logs min_level = Base.CoreLogging.Warn prepare_ad(
        kernel, backend, hidden_args...; active = :a, type_stability_level = :rk)
    # From a spec, the kernel's own preparation checks values only; the
    # differentiated callable's audit still finds the helper.
    @test_throws TypeInstabilityError prepare_ad(U.hidden, backend, hidden_args...;
        active = :a, want = :total, type_stability = :error)

    # A declared local asserts the value, so the value check is silent, but the
    # body RK generates still receives the helper's `Any` result: `:rk` reports
    # that leaf.
    @test_logs min_level = Base.CoreLogging.Warn prepare(U.declared;
                                                          type_stability_level = :values)
    leaf = "uninferred_curve\\(::Vector\\{Float64\\}, ::Float64\\)"
    @test_logs code_warning("is not type stable:\\n  - %\\d+ infers `Any`: " *
                            "`invoke [^`]*$leaf`") prepare(
        U.declared; type_stability_level = :rk)

    # A concretely typed program passes every level, refusing nothing.
    for level in (:values, :rk, :all)
        @test_logs min_level = Base.CoreLogging.Warn prepare(
            Corpus.plate_location; type_stability = :error, type_stability_level = level)
    end

    # Preparations are cached per level.
    cache = PreparationCache()
    @test_logs min_level = Base.CoreLogging.Warn prepare!(
        cache, U.hidden; type_stability_level = :values)
    @test_throws TypeInstabilityError prepare!(cache, U.hidden; type_stability = :error)

    # The level preference, and its validation.
    with_type_stability_preferences("type_stability" => "error",
                                    "type_stability_level" => "rk") do
        @test_logs min_level = Base.CoreLogging.Warn prepare(U.hidden)
        @test_throws TypeInstabilityError prepare(U.hidden; type_stability_level = :all)
        @test_throws TypeInstabilityError prepare(U.declared)
    end
    with_type_stability_preferences("type_stability_level" => "values") do
        @test_logs min_level = Base.CoreLogging.Warn prepare(U.declared)
    end
    with_type_stability_preferences("type_stability_level" => "deep") do
        @test_throws ArgumentError prepare(Corpus.plate_location)
    end
    @test_throws ArgumentError prepare(Corpus.plate_location; type_stability_level = :deep)
end

@testset "type-stability levels: Julia's logging macros are Julia's own code" begin
    U = Uninferred
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    # Control: the helper's own typed code holds the logging dispatch the
    # audit exempts, so the refusals below would fire without the exemption.
    code = first(only(code_typed(U.logged_scale, (Float64,); optimize = true))).code
    @test any(s -> occursin("current_logger_for_env", sprint(show, s)), code)
    @test any(s -> occursin("shouldlog", sprint(show, s)), code)

    for level in (:values, :rk, :all)
        @test_logs min_level = Base.CoreLogging.Warn prepare(
            U.logged; type_stability = :error, type_stability_level = level)
    end
    kernel = prepare(U.logged; type_stability = :error)
    @test kernel(0.1, y) ≈ sum((y .- 0.2) .^ 2)
    @test_logs min_level = Base.CoreLogging.Warn prepare_ad(
        kernel, backend, 0.1, y; active = :a, type_stability = :error)

    # The audit still reports the helper's own run-time dispatch beside its
    # logging, and names none of the logging.
    e = refusal(() -> prepare(U.logged_hidden; type_stability = :error))
    @test e isa TypeInstabilityError
    message = only(e.messages)
    @test occursin("logged_hidden_scale", message)
    @test occursin("is a call dispatched at run time", message)
    @test !occursin("CoreLogging", message)
    @test !occursin("shouldlog", message)
end

end # module
