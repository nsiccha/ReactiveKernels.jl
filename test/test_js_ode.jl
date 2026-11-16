# `js_source` translates `rk_ode_rk45_tol` into the vendored `@reside-ic/dopri`
# Dormand–Prince 5(4) integrator (`src/js_ode.jl`). Each emitted module is run
# in Node.js through `js_run` and checked against the native bridge, which runs
# OrdinaryDiffEq's DP5, and the same exception, with the same `ODESolveFailure`
# return code, where native throws. The two implementations take different step
# sequences, so a value agrees with native within the requested tolerance on a
# smooth problem; where the global error exceeds the tolerance (a long solve
# across a kink in the forcing), the JavaScript value is checked against a
# tight native reference instead: no further from it than 5 times native's
# own value is, or one tolerance unit.
module TestJSODE

using ReactiveKernels, OrdinaryDiffEqLowOrderRK, SciMLBase, SciMLSensitivity
using Test

const NODE = ReactiveKernels._js_default_node() !== nothing
const V = Vector{Float64}

js_ode_decay(t, y, rate) = -rate .* y
js_ode_oscillator(t, y, w) = [y[2], -w^2 * y[1]]
# A five-state transit chain on the log scale with feedback from the last
# state, driven by an input that switches off at `stop`: a generic toy, written
# as a vector literal with a branch, as hand-written right-hand sides are.
function js_ode_chain(t, z, ktr, feedback, dose, stop, decay)
    effect = t < stop ? dose * (1 - exp(-decay * t)) :
        dose * (1 - exp(-decay * stop)) * exp(-decay * (t - stop))
    return [ktr * ((1 - effect) * exp(-feedback * z[5]) - 1),
            ktr * (exp(z[1] - z[2]) - 1), ktr * (exp(z[2] - z[3]) - 1),
            ktr * (exp(z[3] - z[4]) - 1), ktr * (exp(z[4] - z[5]) - 1)]
end
js_ode_blowup(t, y) = y .^ 2
js_ode_nan_after(t, y, at) = t > at ? [NaN] : -y
js_ode_wrong_length(t, y) = [y[1], y[1]]

# The reported probe: the initial state, the times and the rate are inputs.
@kernel js_ode_probe(initial, times, rate, rt, at, mx) = begin
    values = rk_ode_rk45_tol(js_ode_decay, initial, 0.0, times, rt, at, mx, rate)
    return values
end
@kernel js_ode_literal(times, rate) = begin
    values = rk_ode_rk45_tol(js_ode_decay, [1.0], 0.0, times, 1e-8, 1e-8, 10000, rate)
    final = values[size(values, 1), 1]
    return values
end
@kernel js_ode_oscillation(times, w, tol) = begin
    values = rk_ode_rk45_tol(js_ode_oscillator, [1.0, 0.0], 0.0, times, tol, tol, 100000, w)
    position = values[:, 1]
    return position
end
@kernel js_ode_transit(times, ktr, feedback, dose, stop, decay, tol) = begin
    values = rk_ode_rk45_tol(js_ode_chain, zeros(5), 0.0, times, tol, tol, 100000,
                             ktr, feedback, dose, stop, decay)
    last_state = values[:, 5]
    return last_state
end
@kernel js_ode_closure(times, rate, start) = begin
    values = rk_ode_rk45_tol((t, y) -> -rate .* y .+ sin(t), [2.0], start, times, 1e-7, 1e-9, 10000)
    return values
end
@kernel js_ode_failing(times, mx) = begin
    values = rk_ode_rk45_tol(js_ode_blowup, [1.0], 0.0, times, 1e-8, 1e-8, mx)
    return values
end
@kernel js_ode_nan(times, at) = begin
    values = rk_ode_rk45_tol(js_ode_nan_after, [1.0], 0.0, times, 1e-8, 1e-8, 10000, at)
    return values
end
@kernel js_ode_wrong(times) = begin
    values = rk_ode_rk45_tol(js_ode_wrong_length, [1.0], 0.0, times, 1e-8, 1e-8, 10000)
    return values
end
@kernel js_ode_none(times) = begin
    total = sum(times)
    return total
end
@kernel js_ode_stiff(times) = begin
    values = rk_ode_bdf_tol(js_ode_decay, [1.0], 0.0, times, 1e-8, 1e-8, 10000, 0.5)
    return values
end

const T5 = (; times = V, ktr = Float64, feedback = Float64, dose = Float64, stop = Float64,
            decay = Float64, tol = Float64)
grid(a, b, n) = collect(range(a, b; length = n))
# (name, kernel, have, want, types, cases, (reltol, abstol) of a case)
const CASES = [
    (:probe, js_ode_probe, (:initial, :times, :rate, :rt, :at, :mx), :values,
     (; initial = V, times = V, rate = Float64, rt = Float64, at = Float64, mx = Int),
     [([1.0], [0.5, 1.0, 2.0], 0.7, 1e-8, 1e-8, 10000),
      ([1.0, 2.0, -3.0], grid(0.1, 5, 40), 1.3, 1e-6, 1e-9, 10000),
      ([1.0], [0.5, 1.0, 2.0], 0.7, 1e-8, 0.0, 10000),     # one tolerance zero
      ([1.0], [0.5, 1.0, 2.0], 0.7, 0.0, 1e-10, 10000),
      ([1.0], grid(0.5, 50, 100), 0.7, 1e-8, 1e-8, 2),     # the step budget between outputs
      (Float64[], [1.0], 0.7, 1e-8, 1e-8, 10),             # invalid inputs
      ([1.0], Float64[], 0.7, 1e-8, 1e-8, 10),
      ([NaN], [1.0], 0.7, 1e-8, 1e-8, 10),
      ([1.0], [1.0, 0.5], 0.7, 1e-8, 1e-8, 10),
      ([1.0], [0.0, 1.0], 0.7, 1e-8, 1e-8, 10),
      ([1.0], [1.0], 0.7, -1e-8, 1e-8, 10),
      ([1.0], [1.0], 0.7, 0.0, 0.0, 10),
      ([1.0], [1.0], 0.7, 1e-8, 1e-8, 0)],
     c -> (c[4], c[5])),
    (:literal, js_ode_literal, (:times, :rate), (:values, :final), (; times = V, rate = Float64),
     [([0.5, 1.0, 2.0], 0.7), (grid(0.01, 3, 300), 0.2)], c -> (1e-8, 1e-8)),
    (:oscillation, js_ode_oscillation, (:times, :w, :tol), :position, (; times = V, w = Float64, tol = Float64),
     [(grid(0.2, 10, 50), 1.3, 1e-6), (grid(0.2, 10, 50), 1.3, 1e-8), ([20.0], 3.0, 1e-7)], c -> (c[3], c[3])),
    (:transit, js_ode_transit, (:times, :ktr, :feedback, :dose, :stop, :decay, :tol), :last_state, T5,
     [(grid(1, 300, 120), 4 / 90, 0.16, 0.6, 24.0, 0.05, 1e-6),
      (grid(1, 300, 120), 4 / 90, 0.16, 0.6, 24.0, 0.05, 1e-8),
      (grid(0.5, 100, 400), 0.2, 0.4, 0.9, 2.5, 0.8, 1e-7)], c -> (c[7], c[7])),
    (:closure, js_ode_closure, (:times, :rate, :start), :values, (; times = V, rate = Float64, start = Float64),
     [(grid(1, 9, 17), 0.4, 0.5), ([0.75], 2.0, 0.5)], c -> (1e-7, 1e-9)),
    (:failing, js_ode_failing, (:times, :mx), :values, (; times = V, mx = Int),
     [([0.5, 0.9], 10000), ([0.5, 2.0], 10000), ([0.5, 2.0], 50)], c -> (1e-8, 1e-8)),
    (:nan, js_ode_nan, (:times, :at), :values, (; times = V, at = Float64), [([0.2, 1.0], 0.5), ([0.2, 1.0], 2.0)],
     c -> (1e-8, 1e-8)),
    (:wrong, js_ode_wrong, (:times,), :values, (; times = V), [([1.0],)], c -> (1e-8, 1e-8)),
]

# Kernels checked against a tight reference: the case with the tolerance 1e-12.
const REFERENCE = Dict(:transit => c -> (c[1:end-1]..., 1e-12))

# Within the requested tolerance of native, entry by entry.
within(a::Real, b::Real, rt, at) = abs(a - b) <= at + rt * abs(a)
within(a::AbstractArray, b::AbstractArray, rt, at) = size(a) == size(b) && all(within.(a, b, rt, at))
# The largest difference in tolerance units, for the record.
units(a::Real, b::Real, rt, at) = abs(a - b) / (at + rt * abs(a))
units(a::AbstractArray, b::AbstractArray, rt, at) = isempty(a) ? 0.0 : maximum(units.(a, b, rt, at))
retcode(e::ReactiveKernels.ODESolveFailure) = e.retcode
retcode(message::AbstractString) = occursin("exceeded maxsteps", message) ? :MaxIters :
    (m = match(r"failed: (\w+) at", message); m === nothing ? nothing : Symbol(m[1]))

const AGREEMENT = Dict{Symbol,Float64}()
# (native, JavaScript) distance from the reference, in tolerance units
const ERRORS = Dict{Symbol,Vector{Tuple{Float64,Float64}}}()

@testset "js_source ODE" begin
    @testset "Node agrees with native: $name ($vectors)" for (name, k, have, want, types, cases, tols) in CASES,
                                                             vectors in (:array, :float64array)
        source = ReactiveKernels.js_source(k; have, want, types, name)
        @test occursin("var _rk_dopri=", source) && occursin("Imperial College", source)
        if !NODE
            @test_skip "no node on the PATH"
            continue
        end
        wants = want isa Symbol ? (want,) : want
        native = prepare(k; have, want = wants)
        outcomes = ReactiveKernels.js_run(source, name, cases; vectors)
        for (c, r) in zip(cases, outcomes)
            expected = try
                native(c...)
            catch e
                e
            end
            @test !r.inputs_mutated
            if expected isa Exception
                @test r.julia_error && r.error == string(nameof(typeof(expected)))
                expected isa ReactiveKernels.ODESolveFailure && @test retcode(r.message) === retcode(expected)
            else
                @test r.error === nothing
                r.error === nothing || (@info "unexpected JavaScript error" name c r.error r.message; continue)
                @test typeof(r.value) == typeof(expected)
                rt, at = tols(c)
                values = want isa Symbol ? ((expected, r.value),) : zip(expected, r.value)
                references = haskey(REFERENCE, name) ?
                    (r = native(REFERENCE[name](c)...); want isa Symbol ? (r,) : r) : nothing
                for (k, (a, b)) in enumerate(values)
                    if references === nothing
                        @test within(a, b, rt, at)
                        AGREEMENT[name] = max(get(AGREEMENT, name, 0.0), units(a, b, rt, at))
                    else
                        ref = references[k]
                        e_native, e_js = units(ref, a, rt, at), units(ref, b, rt, at)
                        @test e_js <= 5 * max(e_native, 1.0)
                        push!(get!(Vector{Tuple{Float64,Float64}}, ERRORS, name), (e_native, e_js))
                    end
                end
            end
        end
    end
    @info "largest difference from native, in units of the requested tolerance" AGREEMENT
    @info "(native, JavaScript) distance from the tight reference, in units of the requested tolerance" ERRORS

    @testset "the module" begin
        source = ReactiveKernels.js_source(js_ode_probe; have = (:initial, :times, :rate, :rt, :at, :mx),
            want = :values, types = CASES[1][5], name = :probe)
        @test count("var _rk_dopri=", source) == 1
        @test occursin("Permission is hereby granted", source)
        @test occursin("{ name: \"values\", type: \"Matrix{Float64}\" }", source)
        # a kernel without an ODE carries no integrator
        plain = ReactiveKernels.js_source(js_ode_none; have = (:times,), want = :total, types = (; times = V))
        @test !occursin("_rk_dopri", plain) && !occursin("Imperial College", plain)
        e = try
            ReactiveKernels.js_source(js_ode_stiff; have = (:times,), want = :values, types = (; times = V))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError && occursin("`rk_ode_bdf_tol` (FBDF)", e.message)
    end
end

end
