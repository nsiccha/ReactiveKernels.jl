module ScanSliceDemandTests
using ReactiveKernels, Test

# A scan whose trajectory is read only through `trajectory[lo:hi]` runs only
# the steps that slice reads. The references spell the same slice as
# `getindex(trajectory, lo:hi)`, which keeps the materialized trajectory, so
# every comparison below is against the ordinary lowering of the same values.

allocated(f::F, args::Vararg{Any,N}) where {F,N} = @allocated f(args...)
demanded(kernel) = occursin("scan_slice", string(code_expr(kernel)))
thrown(f, args...) = try
    f(args...)
    nothing
catch err
    err
end

# Carry-free: each output depends on its own step only.
@kernel free_window(xs::Vector{Float64}, lo::Int, hi::Int) = begin
    trajectory::Vector{Float64} = scan(xs; init = 0.5, include_init = true) do carry, x
        next = 2x + sin(x)
        (next, next)
    end
    window = trajectory[lo:hi]
    return window
end
@kernel free_window_reference(xs::Vector{Float64}, lo::Int, hi::Int) = begin
    trajectory::Vector{Float64} = scan(xs; init = 0.5, include_init = true) do carry, x
        next = 2x + sin(x)
        (next, next)
    end
    window = getindex(trajectory, lo:hi)
    return window
end

# A recurrence: every step up to the last demanded one runs.
@kernel prefix_window(xs::Vector{Float64}, lo::Int, hi::Int) = begin
    trajectory::Vector{Float64} = scan(xs; init = 1.0, include_init = true) do carry, x
        next = carry + x
        (next, next)
    end
    window = trajectory[lo:hi]
    return window
end
@kernel prefix_window_reference(xs::Vector{Float64}, lo::Int, hi::Int) = begin
    trajectory::Vector{Float64} = scan(xs; init = 1.0, include_init = true) do carry, x
        next = carry + x
        (next, next)
    end
    window = getindex(trajectory, lo:hi)
    return window
end

# Without the seed, slice positions are steps; the bound may be `end`.
@kernel tail_window(xs::Vector{Float64}, lo::Int) = begin
    trajectory::Vector{Float64} = scan(xs; init = 0.0) do carry, x
        next = 0.5carry + x
        (next, next)
    end
    window = trajectory[lo:end]
    return window
end
@kernel tail_window_reference(xs::Vector{Float64}, lo::Int) = begin
    trajectory::Vector{Float64} = scan(xs; init = 0.0) do carry, x
        next = 0.5carry + x
        (next, next)
    end
    window = getindex(trajectory, lo:length(trajectory))
    return window
end

# Per-cell bounds from data, as an observation window per group.
@kernel cell_windows(xs::Vector{Float64}, scale::Vector{Float64},
                     starts::Vector{Int}, stops::Vector{Int}) = begin
    totals = plate(scale, starts, stops) do s, lo, hi
        trajectory = scan(xs; init = 0.0, include_init = true) do carry, x
            next = s * x
            (next, next)
        end
        window = trajectory[lo:hi]
        sum(window)
    end
    return totals
end
@kernel cell_windows_reference(xs::Vector{Float64}, scale::Vector{Float64},
                               starts::Vector{Int}, stops::Vector{Int}) = begin
    totals = plate(scale, starts, stops) do s, lo, hi
        trajectory = scan(xs; init = 0.0, include_init = true) do carry, x
            next = s * x
            (next, next)
        end
        window = getindex(trajectory, lo:hi)
        sum(window)
    end
    return totals
end

# Another reader of the trajectory keeps every step.
@kernel window_and_total(xs::Vector{Float64}, lo::Int, hi::Int) = begin
    trajectory::Vector{Float64} = scan(xs; init = 0.5, include_init = true) do carry, x
        next = 2x + sin(x)
        (next, next)
    end
    window = trajectory[lo:hi]
    total = sum(trajectory)
    return window, total
end

@testset "slice-only scan readers run the demanded steps" begin
    free, free_ref = prepare(free_window), prepare(free_window_reference)
    prefix, prefix_ref = prepare(prefix_window), prepare(prefix_window_reference)
    @test demanded(free) && demanded(prefix)
    @test !demanded(free_ref) && !demanded(prefix_ref)
    @test !demanded(prepare(window_and_total))
    xs = collect(range(-1.0, 2.0; length = 9))
    saved = copy(xs)
    # In-bounds, empty, seed-only and full-trajectory slices.
    for (lo, hi) in ((1, 1), (1, 10), (2, 2), (3, 7), (10, 10), (4, 3), (11, 10),
                     (0, -1), (12, 3))
        @test isequal(free(xs, lo, hi), free_ref(xs, lo, hi))
        @test isequal(prefix(xs, lo, hi), prefix_ref(xs, lo, hi))
        @test free(xs, lo, hi) isa Vector{Float64}
    end
    @test isequal(free(Float64[], 1, 1), [0.5])
    @test isequal(prefix(Float64[], 1, 0), Float64[])
    # An out-of-bounds slice throws the error slicing the whole trajectory throws.
    for (lo, hi) in ((0, 3), (2, 11), (-5, 20), (11, 11))
        for (kernel, reference) in ((free, free_ref), (prefix, prefix_ref))
            err, expected = thrown(kernel, xs, lo, hi), thrown(reference, xs, lo, hi)
            @test err isa BoundsError
            @test sprint(showerror, err) == sprint(showerror, expected)
        end
    end
    @test xs == saved

    tail, tail_ref = prepare(tail_window), prepare(tail_window_reference)
    @test demanded(tail)
    for lo in (1, 4, 9, 10)
        @test isequal(tail(xs, lo), tail_ref(xs, lo))
    end
    @test sprint(showerror, thrown(tail, xs, 0)) == sprint(showerror, thrown(tail_ref, xs, 0))

    # The skipped steps and the trajectory are never allocated.
    long = collect(range(0.0, 1.0; length = 10_000))
    for kernel in (free, prefix, free_ref)
        kernel(long, 3, 5)
    end
    @test allocated(free, long, 3, 5) < 1024
    @test allocated(prefix, long, 3, 5) < 1024
    @test allocated(free_ref, long, 3, 5) > 10_000 * sizeof(Float64)
end

@testset "plate cells read their own scan windows" begin
    cells, cells_ref = prepare(cell_windows), prepare(cell_windows_reference)
    @test demanded(cells)
    @test !demanded(cells_ref)
    xs = collect(range(-1.0, 2.0; length = 12))
    scale = [0.5, -1.0, 2.0, 3.0]
    starts, stops = [1, 3, 6, 13], [4, 3, 13, 13]
    @test isequal(cells(xs, scale, starts, stops), cells_ref(xs, scale, starts, stops))
    # An empty window in one cell, and a cell window out of bounds.
    @test isequal(cells(xs, scale, [1, 5, 2, 2], [0, 4, 2, 1]),
                  cells_ref(xs, scale, [1, 5, 2, 2], [0, 4, 2, 1]))
    bad = ([1, 1, 1, 1], [2, 2, 14, 2])
    @test sprint(showerror, thrown(cells, xs, scale, bad...)) ==
          sprint(showerror, thrown(cells_ref, xs, scale, bad...))
end

end # module
