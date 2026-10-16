module ScanSliceDemandADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# Demanded-step scans (`test_scan_slice_demand.jl`) under ordinary reverse AD.
# The references spell the slice `getindex(trajectory, lo:hi)` and keep the
# materialized trajectory.
@kernel free_objective(q::Vector{Float64}, xs::Vector{Float64}, lo::Int, hi::Int) = begin
    gain::Float64 = q[1]
    trajectory::Vector{Float64} = scan(xs, Ref(gain); init = q[2],
                                       include_init = true) do carry, x, gain
        next = gain * x + sin(gain * x)
        (next, next)
    end
    window = trajectory[lo:hi]
    objective::Float64 = sum(abs2, window)
    return objective
end
@kernel free_reference(q::Vector{Float64}, xs::Vector{Float64}, lo::Int, hi::Int) = begin
    gain::Float64 = q[1]
    trajectory::Vector{Float64} = scan(xs, Ref(gain); init = q[2],
                                       include_init = true) do carry, x, gain
        next = gain * x + sin(gain * x)
        (next, next)
    end
    window = getindex(trajectory, lo:hi)
    objective::Float64 = sum(abs2, window)
    return objective
end
@kernel prefix_objective(q::Vector{Float64}, xs::Vector{Float64}, lo::Int, hi::Int) = begin
    gain::Float64 = q[1]
    trajectory::Vector{Float64} = scan(xs, Ref(gain); init = q[2],
                                       include_init = true) do carry, x, gain
        next = carry + gain * x
        (next, next)
    end
    window = trajectory[lo:hi]
    objective::Float64 = sum(abs2, window)
    return objective
end
@kernel prefix_reference(q::Vector{Float64}, xs::Vector{Float64}, lo::Int, hi::Int) = begin
    gain::Float64 = q[1]
    trajectory::Vector{Float64} = scan(xs, Ref(gain); init = q[2],
                                       include_init = true) do carry, x, gain
        next = carry + gain * x
        (next, next)
    end
    window = getindex(trajectory, lo:hi)
    objective::Float64 = sum(abs2, window)
    return objective
end
@kernel cell_objective(q::Vector{Float64}, xs::Vector{Float64},
                       starts::Vector{Int}, stops::Vector{Int}) = begin
    totals = plate(q, starts, stops) do s, lo, hi
        trajectory = scan(xs; init = s, include_init = true) do carry, x
            next = s * x + carry * 0.0 + s
            (next, next)
        end
        window = trajectory[lo:hi]
        sum(abs2, window)
    end
    objective::Float64 = sum(totals)
    return objective
end
@kernel cell_reference(q::Vector{Float64}, xs::Vector{Float64},
                       starts::Vector{Int}, stops::Vector{Int}) = begin
    totals = plate(q, starts, stops) do s, lo, hi
        trajectory = scan(xs; init = s, include_init = true) do carry, x
            next = s * x + carry * 0.0 + s
            (next, next)
        end
        window = getindex(trajectory, lo:hi)
        sum(abs2, window)
    end
    objective::Float64 = sum(totals)
    return objective
end

@testset "demanded-step scans under ordinary reverse AD" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    xs = collect(range(-1.0, 2.0; length = 9))
    for (fused, reference) in ((free_objective, free_reference),
                               (prefix_objective, prefix_reference))
        f, r = prepare(fused), prepare(reference)
        @test occursin("scan_slice", string(code_expr(f)))
        for (lo, hi) in ((1, 1), (1, 10), (3, 7), (10, 10), (5, 4))
            q = [0.7, 0.3]
            saved = copy(q), copy(xs)
            f_ad = prepare_ad(f, backend, q, xs, lo, hi; active = :q)
            r_ad = prepare_ad(r, backend, q, xs, lo, hi; active = :q)
            fv, fg = ad_value_and_gradient(f_ad, q, xs, lo, hi)
            rv, rg = ad_value_and_gradient(r_ad, q, xs, lo, hi)
            @test fv ≈ rv
            @test fg ≈ rg
            f_xs = prepare_ad(f, backend, q, xs, lo, hi; active = :xs)
            r_xs = prepare_ad(r, backend, q, xs, lo, hi; active = :xs)
            @test ad_gradient(f_xs, q, xs, lo, hi) ≈ ad_gradient(r_xs, q, xs, lo, hi)
            @test (q, xs) == saved
        end
    end
    f, r = prepare(cell_objective), prepare(cell_reference)
    @test occursin("scan_slice", string(code_expr(f)))
    q = [0.5, -1.0, 2.0]
    starts, stops = [1, 3, 5], [4, 2, 10]
    f_ad = prepare_ad(f, backend, q, xs, starts, stops; active = :q)
    r_ad = prepare_ad(r, backend, q, xs, starts, stops; active = :q)
    fv, fg = ad_value_and_gradient(f_ad, q, xs, starts, stops)
    rv, rg = ad_value_and_gradient(r_ad, q, xs, starts, stops)
    @test fv ≈ rv
    @test fg ≈ rg
end

end # module
