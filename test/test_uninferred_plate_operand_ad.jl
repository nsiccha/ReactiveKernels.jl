module UninferredPlateOperandADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# A consumer function whose result Julia infers as `Any`, as for a helper whose
# body dispatches every call at run time. Downstream of it a plate's axes and
# Cartesian index are not inferred, so its per-cell operand reads and its
# pointwise-buffer marker are dynamically dispatched calls carrying bound data.
# Native Enzyme Reverse rejects those calls with `EnzymeRuntimeActivityError`
# ("Constant memory is stored (or returned) to a differentiable variable";
# snag `joint-nlme-enzym-f8d3cb5f`). That is an Enzyme gap, repaired in the
# Enzyme dogfood stack rather than worked around in RK (Enzyme snag
# `native-enzyme-au-1ae111b9`; user principle on RK decision `114x8kz`), so
# the gradients are pinned broken: values are exact, and an "Unexpected Pass"
# signals the repair. Kernels should make such a value inferable instead.
uninferred_curve(t, a) = Base.inferencebarrier(t .* exp(a))

@kernel uninferred_location(a::Float64, s::Float64, t::Vector{Float64},
                            y::Vector{Float64}) = begin
    mu = uninferred_curve(t, a)
    pointwise = plate(y, mu) do yi, mi
        -0.5 * ((yi - mi) / s)^2 - log(s)
    end
    total::Float64 = sum(pointwise)
    return total
end

# The same location per subject, in a plate cell whose inner plate reads the
# subject's observations (the snag's per-subject kernel cell shape).
@kernel uninferred_subject_cells(a::Float64, s::Float64,
                                 ts::Vector{Vector{Float64}},
                                 ys::Vector{Vector{Float64}}) = begin
    subtotals = plate(ts, ys) do ti, yi
        mi = uninferred_curve(ti, a)
        inner = plate(yi, mi) do y, m
            -0.5 * ((y - m) / s)^2 - log(s)
        end
        sum(inner)
    end
    total::Float64 = sum(subtotals)
    return total
end

# The per-subject locations come out of one plate (its output is uninferred),
# and a second plate zips them with the bound per-subject observations: the
# second plate's index is uninferred, and its read of each subject's bound
# observation vector is itself a dynamic call.
@kernel uninferred_two_stage(a::Float64, s::Float64,
                             ts::Vector{Vector{Float64}},
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

normal_total(a, s, t, y) = sum(-0.5 * ((y[i] - t[i] * exp(a)) / s)^2 - log(s)
                               for i in eachindex(y))
normal_gradient(a, s, t, y) = (
    sum((y[i] - t[i] * exp(a)) * t[i] * exp(a) / s^2 for i in eachindex(y)),
    sum((y[i] - t[i] * exp(a))^2 / s^3 - 1 / s for i in eachindex(y)))

@testset "native Reverse through plates over an uninferred operand (Enzyme gap)" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    t = collect(range(0.5, 4.0; length = 7))
    y = 1.3 .* t .+ 0.1 .* sin.(1:7)
    ts = [t[1:3], t[4:7], t[2:5]]
    ys = [y[1:3], y[4:7], y[2:5]]
    a, s = 0.1, 0.8
    @test Base.return_types(uninferred_curve, (Vector{Float64}, Float64)) == [Any]

    cases = (
        ("flat", uninferred_location, (; t, y),
         normal_total(a, s, t, y), normal_gradient(a, s, t, y)),
        ("subject cells", uninferred_subject_cells, (; ts, ys),
         sum(normal_total(a, s, ts[k], ys[k]) for k in eachindex(ts)),
         map(+, (normal_gradient(a, s, ts[k], ys[k]) for k in eachindex(ts))...)),
        ("two stages", uninferred_two_stage, (; ts, ys),
         sum(normal_total(a, s, ts[k], ys[k]) for k in eachindex(ts)),
         map(+, (normal_gradient(a, s, ts[k], ys[k]) for k in eachindex(ts))...)),
    )
    for (name, spec, data, expected, (da, ds)) in cases, bound in (true, false)
        @testset "$name, bound data = $bound" begin
            kernel = bound ? prepare(spec; bound = data) : prepare(spec)
            args = bound ? (a, s) : (a, s, values(data)...)
            @test kernel(args...) ≈ expected
            for (active, derivative) in ((:a, da), (:s, ds))
                prepared = prepare_ad(kernel, backend, args...; active)
                @test_broken ad_value_and_gradient(prepared, args...)[2] ≈ derivative
            end
        end
    end
end

# A scan step whose lazy arm calls a helper that allocates and mutates local
# arrays: it copies the carry and the captured transition into arrays and
# raises a matrix to a power (snag `scan-step-callin-3d8097ae`). Typed, the
# helper differentiates under plain native Reverse: allocation in a step is
# not the trigger. Behind an inference barrier the step's `next` infers
# `Tuple{Any, Any, Any}` (preparation warns) and the carry dispatches
# dynamically. Plain `AutoEnzyme(Reverse)` then raises
# `EnzymeMutabilityException` ("cannot be proven readonly") at a store of a
# pointer into `innerparm`, Enzyme's own by-value copy of the prepared
# callable; Enzyme dogfood `23b72c9` still raises it. With
# `function_annotation = Enzyme.Const`, released Enzyme raises the
# static-activity error of the plate shapes above, which dogfood repairs.
# Both are Enzyme gaps (principle `114x8kz`): each leg accepts the exact
# gradient or its named exception, recorded as broken.
function repeated_transition(A::Matrix{Float64}, x::Vector{Float64}, amount::Float64,
                             count::Int)
    y = copy(x)
    y[1] += amount
    return (A^count) * y
end
transition_apply(T, s) = (T[1] * s[1] + T[4] * s[2] + T[7] * s[3],
                          T[2] * s[1] + T[5] * s[2] + T[8] * s[3],
                          T[3] * s[1] + T[6] * s[2] + T[9] * s[3])
step_typed(T, s, amount, count) =
    (v = repeated_transition(reshape(collect(T), 3, 3), collect(s), amount, count);
     (v[1], v[2], v[3]))
step_uninferred(T, s, amount, count) =
    (v = Base.inferencebarrier(repeated_transition)(reshape(collect(T), 3, 3), collect(s),
                                                    amount, count);
     (v[1], v[2], v[3]))

@kernel scan_step_typed(rate::Float64, dts::Vector{Float64}, kinds::Vector{Int},
                        steps::Vector{Int}, amounts::Vector{Float64},
                        counts::Vector{Int}) = begin
    transitions = plate(dts) do dt
        a = exp(-rate * dt)
        b = exp(-0.5 * rate * dt)
        (a, rate * dt * a, 0.0, 0.0, b, 0.5 * rate * dt * b, 0.0, 0.0, 1.0)
    end
    path = scan(kinds, steps, amounts, counts; init = (0.0, 0.0, 0.0)) do s, kind, step, amount, count
        m = transition_apply(transitions[step], s)
        next = kind == 2 ? step_typed(transitions[step], m, amount, count) : m
        (next, m[2])
    end
    total::Float64 = sum(path)
    return total
end

@kernel scan_step_uninferred(rate::Float64, dts::Vector{Float64}, kinds::Vector{Int},
                             steps::Vector{Int}, amounts::Vector{Float64},
                             counts::Vector{Int}) = begin
    transitions = plate(dts) do dt
        a = exp(-rate * dt)
        b = exp(-0.5 * rate * dt)
        (a, rate * dt * a, 0.0, 0.0, b, 0.5 * rate * dt * b, 0.0, 0.0, 1.0)
    end
    path = scan(kinds, steps, amounts, counts; init = (0.0, 0.0, 0.0)) do s, kind, step, amount, count
        m = transition_apply(transitions[step], s)
        next = kind == 2 ? step_uninferred(transitions[step], m, amount, count) : m
        (next, m[2])
    end
    total::Float64 = sum(path)
    return total
end

function exact_or_enzyme_gap(gradient, expected, gap::Type, message)
    outcome = try
        gradient()
    catch err
        err
    end
    if outcome isa gap
        @test occursin(message, sprint(showerror, outcome))
        @test_broken false
    elseif outcome isa Exception
        throw(outcome)
    else
        @test outcome ≈ expected rtol = 1e-7
    end
end

@testset "native Reverse through a scan step calling an allocating helper" begin
    plain = AutoEnzyme(; mode = Enzyme.Reverse)
    annotated = AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
    data = (; dts = [0.5, 1.0], kinds = [2, 1, 1, 2, 1, 1], steps = [1, 1, 2, 2, 1, 2],
            amounts = [10.0, 0.0, 0.0, 5.0, 0.0, 0.0], counts = [3, 0, 0, 2, 0, 0])
    rate = 0.7
    @test Base.return_types(step_uninferred,
                            (NTuple{9,Float64}, NTuple{3,Float64}, Float64, Int)) ==
          [Tuple{Any,Any,Any}]
    for bound in (true, false)
        @testset "bound data = $bound" begin
            typed = bound ? prepare(scan_step_typed; bound = data) : prepare(scan_step_typed)
            uninferred = bound ? prepare(scan_step_uninferred; bound = data) :
                prepare(scan_step_uninferred)
            args = bound ? (rate,) : (rate, values(data)...)
            h = 1e-6
            central = (typed(rate + h, args[2:end]...) - typed(rate - h, args[2:end]...)) / 2h
            @test uninferred(args...) == typed(args...)
            for backend in (plain, annotated)
                prepared = prepare_ad(typed, backend, args...; active = :rate)
                @test ad_gradient(prepared, args...) ≈ central rtol = 1e-7
            end
            gaps = ((plain, Enzyme.Compiler.EnzymeMutabilityException,
                     "Function argument passed to autodiff cannot be proven readonly"),
                    (annotated, Enzyme.Compiler.EnzymeRuntimeActivityError,
                     "Detected potential need for runtime activity"))
            for (backend, gap, message) in gaps
                prepared = prepare_ad(uninferred, backend, args...; active = :rate)
                exact_or_enzyme_gap(() -> ad_gradient(prepared, args...), central, gap, message)
            end
        end
    end
end

end # module
