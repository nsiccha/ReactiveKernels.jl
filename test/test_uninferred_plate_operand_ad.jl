module UninferredPlateOperandADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# A consumer function whose result Julia infers as `Any`, as for a helper whose
# body dispatches every call at run time. Downstream of it a plate's axes and
# Cartesian index are not inferred, so its per-cell operand reads and its
# pointwise-buffer marker are dynamically dispatched calls. Native Reverse used
# to reject the bound data those calls carried in a fresh `Extruded` box or
# returned as the selected marker: `EnzymeRuntimeActivityError` "Constant
# memory is stored (or returned) to a differentiable variable" (snag
# `joint-nlme-enzym-f8d3cb5f`).
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

@testset "native Reverse through plates over an uninferred operand" begin
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
                value, gradient = ad_value_and_gradient(prepared, args...)
                @test value ≈ expected
                @test gradient ≈ derivative
            end
        end
    end
end

end # module
