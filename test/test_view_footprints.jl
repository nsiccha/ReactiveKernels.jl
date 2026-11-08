module ViewFootprintTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# Constructing a view reads no entries of its parent: a footprint reports the
# parent positions the view's readers read. A typed local that converts the
# view copies it, which reads every viewed position.
struct ReadVector{T} <: AbstractVector{T}
    data::Vector{T}
    reads::Vector{Int}
end
ReadVector(data::Vector) = ReadVector(data, Int[])
Base.size(x::ReadVector) = size(x.data)
Base.IndexStyle(::Type{<:ReadVector}) = IndexLinear()
Base.getindex(x::ReadVector, i::Int) = (push!(x.reads, i); x.data[i])

# Each cell reads selected entries of its own block of a packed vector.
@kernel selected(u::AbstractVector{Float64}, ranges::Vector{UnitRange{Int}},
                 ks::Vector{Int}, picks::Vector{Vector{Int}},
                 y::Vector{Vector{Float64}}) = begin
    packed = view(u, 1:length(u))
    terms = plate(ks, picks, y) do k, p, yk
        block = view(packed, ranges[k])
        m = block[p]
        return -sum(abs2, yk .- m) / 2
    end
    return terms
end
# The same cells through an opaque reader of the whole block.
@kernel coupled(u::AbstractVector{Float64}, ranges::Vector{UnitRange{Int}},
                ks::Vector{Int}, picks::Vector{Vector{Int}},
                y::Vector{Vector{Float64}}) = begin
    packed = view(u, 1:length(u))
    terms = plate(ks, picks, y) do k, p, yk
        block = view(packed, ranges[k])
        m = cumsum(block)[p]
        return -sum(abs2, yk .- m) / 2
    end
    return terms
end

# Weighted gathers inside one compound expression read the gathered entries.
@kernel weighted(u::AbstractVector{Float64}, ranges::Vector{UnitRange{Int}},
                 ks::Vector{Int}, picks::Vector{Vector{Int}},
                 y::Vector{Vector{Float64}}) = begin
    packed = view(u, 1:length(u))
    terms = plate(ks, picks, y) do k, p, yk
        block = view(packed, ranges[k])
        m = 0.5 .* block[p] .+ block[p] .^ 2
        return -sum(abs2, yk .- m) / 2
    end
    return terms
end

const HAVE = (:u, :ranges, :ks, :picks, :y, :t)

# Scalar generator sums keep their visible parameter gathers, including a
# data-only domain/index helper and a captured source without an inner view.
read_slots(a, r) = eachindex(a[r])
read_coordinate(a, r, j) = a[r][j]
@kernel generator_capture(u::AbstractVector{Float64}, a::Vector{Vector{Int}}) = begin
    rows = eachindex(a)
    terms = plate(rows) do r
        sum(u[read_coordinate(a, r, j)]^2 for j in read_slots(a, r); init = 0.0)
    end
end
@kernel generator_view(u::AbstractVector{Float64}, a::Vector{Vector{Int}}) = begin
    rows = eachindex(a)
    terms = plate(rows) do r
        field = view(u, 2:(length(u) - 1))
        sum(field[a[r][j]]^2 for j in 1:length(a[r]); init = 0.0)
    end
end
@kernel generator_nested(u::AbstractVector{Float64}, a::Vector{Vector{Int}}) = begin
    rows = eachindex(a)
    terms = plate(rows) do r
        sum(sum(sum(u[[a[r][j], a[r][j]]].^2) for k in 1:2; init = 0.0)
            for j in read_slots(a, r) if a[r][j] > 0; init = 0.0)
    end
end
@kernel generator_producer(u::AbstractVector{Float64}, a::Vector{Vector{Int}},
                           picks::Vector{Int}) = begin
    rows = eachindex(a)
    field = view(u, 2:(length(u) - 1))
    values = plate(rows) do r
        sum(field[read_coordinate(a, r, j)]^2 for j in read_slots(a, r); init = 0.0)
    end
    terms = plate(picks) do p
        exp(values[p])
    end
end
@kernel generator_copy(u::AbstractVector{Float64}, a::Vector{Vector{Int}},
                       picks::Vector{Int}) = begin
    rows = eachindex(a)
    field = u[2:9]
    values = plate(rows) do r
        sum(field[read_coordinate(a, r, j)]^2 for j in read_slots(a, r); init = 0.0)
    end
    terms = plate(picks) do p
        exp(values[p])
    end
end
@kernel generator_copy_lanes(u::AbstractVector{Float64}, a::Vector{Vector{Int}},
                             picks::Vector{Int}) = begin
    rows = eachindex(a)
    field = u[2:9]
    scales = exp.(u[1:3])
    values = plate(rows, scales) do r, scale
        scale * sum(field[read_coordinate(a, r, j)]^2 for j in read_slots(a, r); init = 0.0)
    end
    terms = plate(picks) do p
        exp(values[p])
    end
end
# Short-circuit consumers and parameter-dependent indices keep a conservative
# footprint. Evaluating the generator's unconsumed tail would throw.
@kernel generator_short(u::AbstractVector{Float64}, a::Vector{Vector{Int}}) = begin
    rows = eachindex(a)
    terms = plate(rows) do r
        first(u[j] for j in (a[r][1], 100))
    end
end
@kernel generator_dependent(u::AbstractVector{Float64}, a::Vector{Vector{Int}}) = begin
    rows = eachindex(a)
    terms = plate(rows) do r
        sum(u[clamp(round(Int, u[1]), 1, length(u))] for j in a[r]; init = 0.0)
    end
end
@kernel generator_lazy(u::AbstractVector{Float64}, a::Vector{Vector{Int}}) = begin
    rows = eachindex(a)
    terms = plate(rows) do r
        sum(j > 0 ? u[j] : u[100] for j in a[r]; init = 0.0)
    end
end
@kernel capture_lazy(u::AbstractVector{Float64}, a::Vector{Vector{Int}}) = begin
    rows = eachindex(a)
    terms = plate(rows) do r
        exp(r > 0 ? u[a[r][1]] : u[100])
    end
end
@kernel generator_shadow(u::AbstractVector{Float64}, Int::Vector{Vector{Int64}}) = begin
    rows = eachindex(Int)
    terms = plate(rows) do r
        sum(u[Int[r][j]]^2 for j in eachindex(Int[r]); init = 0.0)
    end
end

@testset "scalar generator footprints" begin
    u = collect(0.1:0.1:1.0)
    a = [[3, 1, 3], Int[], [2, 4]]
    saved = deepcopy((u, a))
    for (k, offset, scale) in ((generator_capture, 0, 1),
                              (generator_view, 1, 1), (generator_nested, 0, 4))
        spec = plate_cell(k, :terms; index = :t, locality = :error)
        query = prepare(spec; have = (:u, :a, :rows, :t), want = :terms_cell)
        bound = prepare(spec; have = (:u, :a, :t), want = :terms_cell, bound = (; a))
        ad = prepare_ad(bound, AutoEnzyme(; mode = Enzyme.Reverse), u, 1; active = :u)
        for t in eachindex(a)
            positions = a[t] .+ offset
            expected = scale * sum(abs2, u[positions]; init = 0.0)
            tracked = ReadVector(copy(u))
            @test query(tracked, a, eachindex(a), t) ≈ expected
            reads = sort!(unique(positions))
            @test sort!(unique(tracked.reads)) == reads
            @test partition_footprint(query, u, a, eachindex(a), t).u == reads
            @test partition_footprint(bound, u, t).u == reads
            value, gradient = ad_value_and_gradient(ad, u, t)
            expected_gradient = zeros(length(u))
            for i in positions
                expected_gradient[i] += 2scale * u[i]
            end
            @test value ≈ expected
            @test gradient ≈ expected_gradient
        end
    end
    for k in (generator_short, generator_dependent, generator_lazy, capture_lazy)
        spec = plate_cell(k, :terms; index = :t, locality = :error)
        query = prepare(spec; have = (:u, :a, :t), want = :terms_cell, bound = (; a))
        tracked = ReadVector(copy(u))
        @test isfinite(query(tracked, 1))
        @test partition_footprint(query, u, 1).u === :whole
    end
    filtered = prepare(plate_cell(generator_nested, :terms; index = :t, locality = :error);
        have = (:u, :a, :t), want = :terms_cell, bound = (; a = [[0, 3, 0, 1]]))
    @test filtered(u, 1) ≈ 4sum(abs2, u[[3, 1]])
    @test partition_footprint(filtered, u, 1).u == [1, 3]
    shadow = prepare(plate_cell(generator_shadow, :terms; index = :t, locality = :error);
        have = (:u, :Int, :t), want = :terms_cell, bound = (; Int = a))
    @test shadow(u, 1) ≈ sum(abs2, u[a[1]])
    @test partition_footprint(shadow, u, 1).u == [1, 3]
    @test (u, a) == saved
end

@testset "generator reads through a producer plate" begin
    u = collect(0.1:0.1:1.0)
    a, picks = [[3, 1, 3], Int[], [2, 4]], [3, 1, 3, 2]
    saved = deepcopy((u, a, picks))
    for kernel in (generator_producer, generator_copy)
        spec = plate_cell(kernel, :terms; index = :t, locality = :error)
        query = prepare(spec; have = (:u, :a, :picks, :t), want = :terms_cell,
            bound = (; a, picks))
        ad = prepare_ad(query, AutoEnzyme(; mode = Enzyme.Reverse), u, 1; active = :u)
        for t in eachindex(picks)
            positions = a[picks[t]] .+ 1
            expected = exp(sum(abs2, u[positions]; init = 0.0))
            tracked = ReadVector(copy(u))
            @test query(tracked, t) ≈ expected
            reads = sort!(unique(positions))
            @test sort!(unique(tracked.reads)) == reads
            @test partition_footprint(query, u, t).u == reads
            value, gradient = ad_value_and_gradient(ad, u, t)
            expected_gradient = zeros(length(u))
            for i in positions
                expected_gradient[i] += 2u[i] * expected
            end
            @test value ≈ expected
            @test gradient ≈ expected_gradient
        end
    end
    query = prepare(plate_cell(generator_copy_lanes, :terms; index = :t, locality = :error);
        have = (:u, :a, :picks, :t), want = :terms_cell, bound = (; a, picks))
    ad = prepare_ad(query, AutoEnzyme(; mode = Enzyme.Reverse), u, 1; active = :u)
    for t in eachindex(picks)
        r = picks[t]
        positions = a[r] .+ 1
        square = sum(abs2, u[positions]; init = 0.0)
        expected = exp(exp(u[r]) * square)
        tracked = ReadVector(copy(u))
        @test query(tracked, t) ≈ expected
        reads = sort!(unique([positions; r]))
        @test sort!(unique(tracked.reads)) == reads
        @test partition_footprint(query, u, t).u == reads
        value, gradient = ad_value_and_gradient(ad, u, t)
        expected_gradient = zeros(length(u))
        expected_gradient[r] += expected * exp(u[r]) * square
        for i in positions
            expected_gradient[i] += 2u[i] * expected * exp(u[r])
        end
        @test value ≈ expected
        @test gradient ≈ expected_gradient
    end
    @test (u, a, picks) == saved
end

function cases()
    ((; ranges = [1:3, 4:5], picks = [[1], Int[]]),
     (; ranges = [1:3, 4:5, 6:5], picks = [[3, 1], [2], Int[]]),
     (; ranges = [1:4, 5:5], picks = [[2, 2, 4], [1]]))
end

reference(kind, u, d, t) = begin
    block = u[d.ranges[t]]
    values = kind === :coupled ? cumsum(block) : block
    m = values[d.picks[t]]
    kind === :weighted && (m = 0.5 .* m .+ m .^ 2)
    -sum(abs2, fill(0.1, length(d.picks[t])) .- m; init = 0.0) / 2
end

@testset "cell footprints follow reads through views" begin
    for (kind, k) in ((:selected, selected), (:coupled, coupled), (:weighted, weighted)),
            d in cases()
        u = [0.1 + 0.07i for i in 1:last(last(d.ranges))]
        y = [fill(0.1, length(p)) for p in d.picks]
        ks = collect(eachindex(d.ranges))
        saved = deepcopy((u, d, y))
        spec = plate_cell(k, :terms; index = :t, locality = :error)
        query = prepare(spec; have = HAVE, want = :terms_cell)
        footprint = partition_footprint_kernel(query)
        bound = prepare(spec; have = HAVE, want = :terms_cell, bound = (; d.ranges, ks, d.picks, y))
        ad = prepare_ad(bound, AutoEnzyme(; mode = Enzyme.Reverse), u, 1; active = :u)
        for t in eachindex(d.ranges)
            expected = reference(kind, u, d, t)
            tracked = ReadVector(copy(u))
            @test query(u, d.ranges, ks, d.picks, y, t) ≈ expected
            @test query(tracked, d.ranges, ks, d.picks, y, t) ≈ expected
            read = Base.invokelatest(footprint, u, d.ranges, ks, d.picks, y, t).u
            if kind in (:selected, :weighted)
                @test read == sort!(unique(d.ranges[t][d.picks[t]]))
                @test sort!(unique(tracked.reads)) == read
            else
                @test read == collect(d.ranges[t])
                @test issubset(tracked.reads, read)
            end
            @test partition_footprint(bound, u, t).u == read
            value, gradient = ad_value_and_gradient(ad, u, t)
            @test value ≈ expected
            @test issubset(findall(!iszero, gradient), read)
        end
        @test (u, d, y) == saved
    end
end

# A typed local converts a view whose type is not the declared one. A kernel
# that owns a plate declares its typed outputs (`_declare_typed_output!`).
@kernel typed_copy(u::AbstractVector{Float64}, c::Vector{Float64}) = begin
    head::Vector{Float64} = view(u, 2:4)
    w = plate(c) do x
        return 2x
    end
    second = head[2]
    value = second + sum(w)
    return value
end
@kernel untyped_view(u::AbstractVector{Float64}, c::Vector{Float64}) = begin
    head = view(u, 2:4)
    w = plate(c) do x
        return 2x
    end
    second = head[2]
    value = second + sum(w)
    return value
end
# An abstract declared type the view already has converts it to itself, though
# inference over the abstract port types the view `Any` (RKPPL's packed
# `x::AbstractVector{Float64} = view(unconstrained, …)` beside a plate, snag
# `dotted-cell-late-3789b8ea`). A wider declared type that a copy could also
# satisfy still converts the view to itself.
@kernel typed_view(u::AbstractVector{Float64}, c::Vector{Float64}) = begin
    head::AbstractVector{Float64} = view(u, 2:4)
    w = plate(c) do x
        return 2x
    end
    second = head[2]
    value = second + sum(w)
    return value
end
@kernel wide_typed_view(u::AbstractVector{Float64}, c::Vector{Float64}) = begin
    head::AbstractArray = view(u, 2:4)
    w = plate(c) do x
        return 2x
    end
    second = head[2]
    value = second + sum(w)
    return value
end

# A field of existing storage is existing storage: a cell reads a named
# tuple's vector field at its rows. A property that `getproperty` computes
# would be computed whole, which strict locality refuses.
@kernel fields(u::Vector{Float64}, data, y::Vector{Float64}) = begin
    w = data.weight
    mu = w .* u
    terms = plate(mu, y) do m, yi
        return -(m - yi)^2 / 2
    end
    return terms
end
struct ComputedWeight
    raw::Vector{Float64}
end
Base.getproperty(c::ComputedWeight, s::Symbol) =
    s === :weight ? 2 .* getfield(c, :raw) : getfield(c, s)

@testset "cell reads of storage fields" begin
    u, y = [0.1, 0.4, -0.2], [0.0, 0.3, 0.1]
    data = (; weight = [0.5, 2.0, 1.5])
    spec = plate_cell(fields, :terms; index = :t, locality = :error)
    query = prepare(spec; have = (:u, :data, :y, :t), want = :terms_cell,
        exemplars = (u, data, y, 1))
    tracked = (; weight = ReadVector(copy(data.weight)))
    traced = prepare(spec; have = (:u, :data, :y, :t), want = :terms_cell,
        exemplars = (u, tracked, y, 1))
    for t in 1:3
        empty!(tracked.weight.reads)
        expected = -(data.weight[t] * u[t] - y[t])^2 / 2
        @test query(u, data, y, t) ≈ expected
        @test traced(u, tracked, y, t) ≈ expected
        @test unique(tracked.weight.reads) == [t]
        @test partition_footprint(query, u, data, y, t).u == [t]
    end
    computed = ComputedWeight([0.25, 1.0, 0.75])
    @test_throws PartitionLocalityError prepare(spec; have = (:u, :data, :y, :t),
        want = :terms_cell, exemplars = (u, computed, y, 1))
end

# A view or a read inside a larger expression reads its literal positions.
@kernel stacked_edge(u::AbstractVector{Float64}, c::Float64) = begin
    edge = tanh.(view(u, 2:3:8))
    value = sum(edge) + c
    return value
end
@kernel reversed_edge(u::AbstractVector{Float64}, c::Float64) = begin
    edge = tanh.(view(u, 8:-3:2))
    value = sum(edge) + c
    return value
end
@kernel nested_reads(u::AbstractVector{Float64}, c::Float64) = begin
    value = exp(u[3]) + sum(exp.(view(u, 5:6))) + c
    return value
end
# Any other read of the port in that expression reads it whole.
@kernel bare_beside(u::AbstractVector{Float64}, c::Float64) = begin
    value = sum(view(u, 2:3)) + sum(u) + c
    return value
end
@kernel computed_range(u::AbstractVector{Float64}, c::Float64, k::Int) = begin
    value = sum(exp.(view(u, 1:k))) + c
    return value
end
# An inactive arm may hold an invalid literal, which must not be evaluated.
@kernel inactive_arm(u::AbstractVector{Float64}, c::Float64) = begin
    value = c > 0 ? exp(u[1]) : exp(u[100])
    return value
end

@testset "reads nested in a larger expression" begin
    u, c = collect(0.1:0.1:0.9), 1.5
    for (k, expected) in ((stacked_edge, [2, 5, 8]), (reversed_edge, [2, 5, 8]),
                          (nested_reads, [3, 5, 6]))
        kernel = prepare(k; have = (:u, :c), want = :value)
        tracked = ReadVector(copy(u))
        @test kernel(tracked, c) ≈ kernel(u, c)
        @test sort!(unique(tracked.reads)) == expected
        @test partition_footprint(kernel, u, c).u == expected
    end
    bare = prepare(bare_beside; have = (:u, :c), want = :value)
    @test partition_footprint(bare, u, c).u === :whole
    computed = prepare(computed_range; have = (:u, :c, :k), want = :value)
    @test computed(u, c, 2) ≈ exp(u[1]) + exp(u[2]) + c
    @test partition_footprint(computed, u, c, 2).u === :whole
    arm = prepare(inactive_arm; have = (:u, :c), want = :value)
    @test arm(u, c) ≈ exp(u[1])
    @test partition_footprint(arm, u, c).u === :whole
end

@testset "footprints through typed and untyped views" begin
    u, c = collect(0.1:0.1:0.6), [1.0, 2.0]
    for (k, expected) in ((typed_copy, [2, 3, 4]), (untyped_view, [3]),
                          (typed_view, [3]), (wide_typed_view, [3]))
        kernel = prepare(k; have = (:u, :c), want = :value)
        tracked = ReadVector(copy(u))
        @test kernel(tracked, c) ≈ u[3] + 6.0
        @test sort!(unique(tracked.reads)) == expected
        @test partition_footprint(kernel, u, c).u == expected
    end
end
end
