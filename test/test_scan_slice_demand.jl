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

# Element-only readers: `trajectory[k]`, `last(trajectory)` and
# `first(trajectory)` keep one output and allocate no trajectory. The
# references read the same element through `getindex(trajectory, k)`, which
# keeps the materialized trajectory (`last(a)` is `a[lastindex(a)]` in Base).
element_demanded(kernel) = occursin("scan_element_step", string(code_expr(kernel)))
element_reads(kernel) = length(unique(m.match for m in
    eachmatch(r"scan_element_step#\d+", string(code_expr(kernel)))))

@kernel last_free(xs::Vector{Float64}) = begin
    trajectory::Vector{Float64} = scan(xs; init = 0.5, include_init = true) do carry, x
        next = 2x + sin(x)
        (next, next)
    end
    value::Float64 = last(trajectory)
    return value
end
@kernel last_free_reference(xs::Vector{Float64}) = begin
    trajectory::Vector{Float64} = scan(xs; init = 0.5, include_init = true) do carry, x
        next = 2x + sin(x)
        (next, next)
    end
    value::Float64 = getindex(trajectory, lastindex(trajectory))
    return value
end
@kernel last_recurrence(xs::Vector{Float64}) = begin
    trajectory = scan(xs; init = 1.0) do carry, x
        next = 0.5carry + x
        (next, next)
    end
    value::Float64 = last(trajectory)
    return value
end
@kernel last_recurrence_reference(xs::Vector{Float64}) = begin
    trajectory = scan(xs; init = 1.0) do carry, x
        next = 0.5carry + x
        (next, next)
    end
    value::Float64 = getindex(trajectory, lastindex(trajectory))
    return value
end
# The seed is an `Int`, the steps return `Float64`: the first step is peeled
# and the trajectory's element type is their promotion.
@kernel element_at(xs::Vector{Float64}, k::Int) = begin
    trajectory = scan(xs; init = 1, include_init = true) do carry, x
        next = carry * x + 1
        (next, next)
    end
    value = trajectory[k]
    return value
end
@kernel element_at_reference(xs::Vector{Float64}, k::Int) = begin
    trajectory = scan(xs; init = 1, include_init = true) do carry, x
        next = carry * x + 1
        (next, next)
    end
    value = getindex(trajectory, k)
    return value
end
@kernel end_and_first(xs::Vector{Float64}) = begin
    tail = scan(xs; init = 0.0) do carry, x
        next = carry - x^2
        (next, next)
    end
    head = scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    a = tail[end]
    b = first(head)
    return a + b
end
@kernel end_and_first_reference(xs::Vector{Float64}) = begin
    tail = scan(xs; init = 0.0) do carry, x
        next = carry - x^2
        (next, next)
    end
    head = scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    a = getindex(tail, lastindex(tail))
    b = getindex(head, firstindex(head))
    return a + b
end
# The continued fraction of a truncated normal's variance: a backward
# recurrence over a reversed range whose length follows the input, read only
# through its last output.
@kernel backward_fraction(z::Float64) = begin
    n::Int = ceil(Int, 6 + 120 / z + 100 / z^2)
    seed::Float64 = 2 * (n + 1) / (hypot(z, 2 * sqrt(n + 1)) + z)
    path = scan(n:-1:2, Ref(z); init = seed) do t, k, z
        next = k / (z + t)
        (next, next)
    end
    t::Float64 = last(path)
    t1::Float64 = inv(z + t)
    return t1 * (t - t1)
end
@kernel backward_fraction_reference(z::Float64) = begin
    n::Int = ceil(Int, 6 + 120 / z + 100 / z^2)
    seed::Float64 = 2 * (n + 1) / (hypot(z, 2 * sqrt(n + 1)) + z)
    path = scan(n:-1:2, Ref(z); init = seed) do t, k, z
        next = k / (z + t)
        (next, next)
    end
    t::Float64 = getindex(path, lastindex(path))
    t1::Float64 = inv(z + t)
    return t1 * (t - t1)
end
@kernel cell_lasts(xs::Vector{Float64}, scale::Vector{Float64}) = begin
    finals = plate(scale) do s
        trajectory = scan(xs; init = s) do carry, x
            next = s * carry + x
            (next, next)
        end
        last(trajectory)
    end
    return finals
end
@kernel cell_lasts_reference(xs::Vector{Float64}, scale::Vector{Float64}) = begin
    finals = plate(scale) do s
        trajectory = scan(xs; init = s) do carry, x
            next = s * carry + x
            (next, next)
        end
        getindex(trajectory, lastindex(trajectory))
    end
    return finals
end

@testset "element-only scan readers keep one output" begin
    xs = collect(range(-1.0, 2.0; length = 9))
    pairs = ((prepare(last_free), prepare(last_free_reference)),
             (prepare(last_recurrence), prepare(last_recurrence_reference)))
    for (kernel, reference) in pairs
        @test element_demanded(kernel)
        @test !demanded(reference)
        for n in (1, 2, 9)
            @test kernel(xs[1:n]) === reference(xs[1:n])
        end
    end
    # A seed-only trajectory's last element is the seed; with no seed and no
    # step, `last` throws Base's own error for an empty vector.
    @test prepare(last_free)(Float64[]) === 0.5
    err = thrown(prepare(last_recurrence), Float64[])
    @test err isa BoundsError
    @test sprint(showerror, err) == sprint(showerror, thrown(last, Float64[])) ==
          sprint(showerror, thrown(prepare(last_recurrence_reference), Float64[]))

    at, at_ref = prepare(element_at), prepare(element_at_reference)
    @test element_demanded(at) && !demanded(at_ref)
    for k in 1:10
        @test at(xs, k) === at_ref(xs, k)
    end
    @test at(xs, 1) === 1.0
    @test at(Float64[], 1) === at_ref(Float64[], 1)
    for (args, k) in (((xs,), 0), ((xs,), 11), ((Float64[],), 2))
        err, expected = thrown(at, args..., k), thrown(at_ref, args..., k)
        @test err isa BoundsError
        @test sprint(showerror, err) == sprint(showerror, expected)
    end

    both, both_ref = prepare(end_and_first), prepare(end_and_first_reference)
    @test element_reads(both) == 2 && !demanded(both_ref)
    @test both(xs) === both_ref(xs)

    fraction, fraction_ref = prepare(backward_fraction), prepare(backward_fraction_reference)
    @test element_demanded(fraction) && !demanded(fraction_ref)
    for z in (2.0, 3.7, 14.8, 1e3)
        @test fraction(z) === fraction_ref(z)
    end

    # No trajectory is allocated, however long the sequence.
    long = collect(range(0.0, 1.0; length = 10_000))
    (lf, _), (lr, _) = pairs
    lf_ref = last(first(pairs))
    for kernel in (lf, lr, both, lf_ref)
        kernel(long)
    end
    at(long, 5_000)
    fraction(3.7)
    @test allocated(lf, long) == 0
    @test allocated(lr, long) == 0
    @test allocated(at, long, 5_000) == 0
    @test allocated(both, long) == 0
    @test allocated(fraction, 3.7) == 0
    @test allocated(lf_ref, long) > 10_000 * sizeof(Float64)

    cells, cells_ref = prepare(cell_lasts), prepare(cell_lasts_reference)
    @test element_demanded(cells) && !demanded(cells_ref)
    scale = [0.5, -1.0, 0.25]
    @test isequal(cells(xs, scale), cells_ref(xs, scale))
    @test sprint(showerror, thrown(cells, Float64[], scale)) ==
          sprint(showerror, thrown(cells_ref, Float64[], scale))
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
