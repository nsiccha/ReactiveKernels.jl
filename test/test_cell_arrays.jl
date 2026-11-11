module CellArrayTests
using Test, ReactiveKernels, DifferentiationInterface, Enzyme

function native_structure(ex)
    ex isa Expr || return (0, 0)
    nodes, loops = 1, Int(ex.head in (:for, :while))
    for arg in ex.args
        child_nodes, child_loops = native_structure(arg)
        nodes += child_nodes
        loops += child_loops
    end
    (nodes, loops)
end

# Embedding must keep Julia's call boundary even when an argument is a
# projection or another expression used repeatedly by the nested body.
argument_once = ReactiveKernels._embedded_statements(:(function (__constants__, __callables__, x)
    return x + x
end), [:(popfirst!(xs))], :result, 0)
@eval function embedded_argument_once(xs)
    $(argument_once...)
    result
end
@testset "embedded arguments evaluate once" begin
    values = [2.0, 3.0]
    @test embedded_argument_once(values) == 4.0
    @test values == [3.0]
end

@kernel column_squares(x::Matrix{Float64}) = begin
    columns = plate(axes(x, 2)) do j
        cells = plate(axes(x, 1)) do i
            abs2(x[i, j])
        end
        sum(cells)
    end
    total = sum(columns)
    return total
end
@kernel array_columns(x, scale::Float64) = begin
    multiplied = scale .* x
    shifted = multiplied .+ 0
    table = hcat(shifted, x)
    total = column_squares(table)
    return total
end
@kernel event_columns(groups, scale::Float64) = begin
    values = plate(eachindex(groups)) do event
        array_columns(groups[event], scale)
    end
    total = sum(values)
    return total
end
@kernel literal_columns(groups, scale::Float64) = begin
    values = plate(eachindex(groups)) do event
        shifted = @. scale + groups[event]
        table = [groups[event] shifted groups[event]]
        column_squares(table)
    end
    total = sum(values)
    return total
end
@kernel inline_columns(groups, scale::Float64) = begin
    values = plate(eachindex(groups)) do event
        column_squares(Base.hcat(scale .* groups[event], groups[event]))
    end
    total = sum(values)
    return total
end
@kernel affine_columns(groups, scale::Float64) = begin
    values = plate(eachindex(groups)) do event
        shifted = scale .+ (scale - 1) .* groups[event]
        table = hcat(shifted, groups[event])
        column_squares(table)
    end
    total = sum(values)
    return total
end
@kernel paired_broadcasts(x, c, scale::Float64) = begin
    left = x .* scale
    right = c .* scale
    table = hcat(left, right)
    total = column_squares(table)
    return total
end
@kernel promoted_entries(x) = begin
    shifted = x .* 2
    table = hcat(shifted, x)
    value = table[1, 1] + table[1, 2]
    return value
end
@kernel broadcast_matrix(x, y) = begin
    shifted = x .+ y
    table = hcat(shifted, shifted)
    total = column_squares(table)
    return total
end
@kernel composed_runs(q::Vector{Float64}, idx::Vector{Int}, x::Vector{Float64}) = begin
    gathered = ReactiveKernels._readonly_gather(q, idx)
    terms = plate(gathered, x) do a, z
        table = hcat(a, z)
        abs2(table[1, 1] - table[1, 2])
    end
    total = sum(terms)
    return total
end
@kernel eager_error(x) = begin
    transformed = log.(x)
    value = transformed[1]
    return value
end
@kernel incompatible_shapes(x, y) = begin
    table = hcat(x, y)
    total = column_squares(table)
    return total
end
@kernel converted_array(x) = begin
    table::Matrix{Float64} = hcat(x, x)
    total = column_squares(table)
    return total
end
@kernel partial_broadcast(x, y) = begin
    transformed = x .+ y
    value = transformed[1]
    return value
end
@kernel checked_array_empty_reader(x, y) = begin
    transformed = log.(x)
    table = hcat(transformed, transformed)
    values = plate(y) do v
        table[1, 1] + v
    end
    total::Float64 = sum(values)
    return total
end
@kernel shaped_array_empty_reader(x, y, z) = begin
    table = hcat(x, y)
    values = plate(z) do v
        table[1, 1] + v
    end
    total::Float64 = sum(values)
    return total
end
@kernel earlier_array_error(x) = begin
    transformed = log.(x)
    table = hcat(transformed, transformed)
    values = plate(axes(table, 1)) do i
        a = table[i, 1]
        a >= 0 ? throw(ArgumentError("consumer reached")) : a
    end
    total::Float64 = sum(values)
    return total
end
opaque_sum(x) = sum(x)
@kernel opaque_reader(x) = begin
    table = hcat(x, x)
    value = opaque_sum(table)
    return value
end
@kernel array_index(x, i) = begin
    table = hcat(x, x)
    value = table[i, 2]
    return value
end
@kernel scalar_broadcast(x::Float64) = begin
    shifted::Float64 = x .+ 1.0
    table = hcat(shifted, x)
    value = table[1, 1] + table[1, 2]
    return value
end
struct MutatingReadVector <: AbstractVector{Float64}
    data::Vector{Float64}
    victim::Vector{Float64}
end
Base.size(x::MutatingReadVector) = size(x.data)
function Base.getindex(x::MutatingReadVector, i::Int)
    x.victim[1] = 99.0
    x.data[i]
end
@kernel custom_concat(x, y) = begin
    shifted = x .* 2
    table = hcat(y, shifted)
    value = table[1, 2]
    return value
end
@kernel custom_broadcast(x, y) = begin
    shifted = x .* 2
    values = y .+ shifted
    value = values[1]
    return value
end
function change!(x, dependency)
    x[1] = 99.0
    zero(dependency)
end
@kernel snapshot_before_effect(x) = begin
    table = hcat(x, x)
    rows = size(table, 1)
    @recipe (effectful = true) token = change!(x, rows)
    value = table[1, 1] + token
    return value
end
# A per-row value of a live scalar and a data column, gathered by each group's
# rows in a cell composed into the group plate.
@kernel gathered_rows(rows, y, c::Vector{Float64}, s::Float64) = begin
    scale = exp.(s .+ 0.5 .* c)
    picked = plate(eachindex(rows)) do g
        scale[rows[g]]
    end
    values = plate(picked, y) do p, yg
        terms = plate(p, yg) do v, w
            v * w
        end
        sum(terms)
    end
    total = sum(values)
    return total
end
gathered_rows_data(groups, n) = (rows = [collect(g:groups:n) for g in 1:groups],
    y = [[0.01 * r for r in g:groups:n] for g in 1:groups], c = [0.1 * sin(i) for i in 1:n])

@testset "array recipes compose through nested scalar readers" begin
    groups = [[0.2, -0.4], Float64[], [0.7, 1.2, -0.1]]
    original = deepcopy(groups)
    for (spec, reference, derivative) in (
        (event_columns, (xs, s) -> sum(x -> sum(abs2, x; init=0.0) * (s^2 + 1), xs; init=0.0),
            (xs, s) -> 2s * sum(x -> sum(abs2, x; init=0.0), xs; init=0.0)),
        (inline_columns, (xs, s) -> sum(x -> sum(abs2, x; init=0.0) * (s^2 + 1), xs; init=0.0),
            (xs, s) -> 2s * sum(x -> sum(abs2, x; init=0.0), xs; init=0.0)),
        (affine_columns, (xs, s) -> sum(x -> sum(v -> abs2(s + (s - 1) * v) + abs2(v), x; init=0.0), xs; init=0.0),
            (xs, s) -> sum(x -> sum(v -> 2(s + (s - 1) * v) * (1 + v), x; init=0.0), xs; init=0.0)),
        (literal_columns, (xs, s) -> sum(x -> 2sum(abs2, x; init=0.0) + sum(v -> abs2(s + v), x; init=0.0), xs; init=0.0),
            (xs, s) -> sum(x -> sum(v -> 2(s + v), x; init=0.0), xs; init=0.0)))
        for data in (groups, [zeros(0)], Vector{Float64}[], [[1.0], [2.0, 3.0]],
                     [reshape([1.0, 2.0, 3.0, 4.0], 2, 2), zeros(0, 2), ones(1, 3)])
            for (k, bound) in ((prepare(spec), false), (prepare(spec; bound=(; groups=data)), true))
                # Both live and bound data retain the same runtime loop body.
                inputs = bound ? (0.6,) : (data, 0.6)
                @test k(inputs...) ≈ reference(data, 0.6)
                ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), inputs...; active=:scale)
                @test ad_gradient(ad, inputs...) ≈ derivative(data, 0.6)
            end
        end
        @test occursin("_cell_array_hcat", string(code_expr(prepare(spec))))
        cache = PreparationCache()
        small = prepare!(cache, spec; bound=(; groups=[ones(2) for _ in 1:3]))
        large = prepare!(cache, spec; bound=(; groups=[ones(5) for _ in 1:60]))
        # Changing both event count and cell width retains runtime loops
        # without growing the emitted body (fresh gensyms may differ).
        @test native_structure(small.ast) == native_structure(large.ast)
        @test last(native_structure(small.ast)) > 0
    end
    @test groups == original
    # Queries still return owned dense authored intermediates.
    x = [0.2, 0.4]
    table = prepare(array_columns; want=:table)(x, 0.6)
    @test table isa Matrix{Float64}
    @test table == hcat(0.6 .* x, x)
    table[1, 2] = 99
    @test x == [0.2, 0.4]
    @test prepare(array_columns; want=:shifted)(x, 0.6) == 0.6 .* x
    @test_throws DomainError prepare(eager_error)([1.0, -1.0])
    @test_throws DimensionMismatch prepare(incompatible_shapes)(ones(2), ones(3))
    @test_throws DimensionMismatch prepare(partial_broadcast)(ones(2), ones(3))
    # A later empty domain cannot suppress an earlier producer's exception
    # or shape validation, including when the empty domain is bound data.
    for bound in (false, true)
        reader = bound ? prepare(checked_array_empty_reader; bound=(; y=Float64[])) :
            prepare(checked_array_empty_reader)
        args(x) = bound ? (x,) : (x, Float64[])
        @test_throws DomainError reader(args([-1.0])...)
        @test reader(args([1.0])...) == 0.0
    end
    shapes = prepare(shaped_array_empty_reader)
    @test_throws DimensionMismatch shapes(ones(2), ones(3), Float64[])
    @test shapes(ones(2), ones(2), Float64[]) == 0.0
    @test_throws DimensionMismatch prepare(shaped_array_empty_reader;
        bound=(; y=ones(3), z=Float64[]))(ones(2))
    @test prepare(shaped_array_empty_reader;
        bound=(; y=ones(2), z=Float64[]))(ones(2)) == 0.0
    # The producer's exception precedes a consumer's competing exception.
    competing = prepare(earlier_array_error)
    err = try competing([1.0, -1.0]); nothing catch e; e end
    @test err isa DomainError && err.val == -1.0
    @test_throws ArgumentError competing([1.0, 1.0])
    @test competing([0.5, 0.25]) ≈ sum(log.([0.5, 0.25]))
    @test prepare(converted_array)(Float32[0.2, 0.4]) ≈ 2sum(abs2, Float64.(Float32[0.2, 0.4]))
    @test prepare(opaque_reader)(x) == 2sum(x)
    @test prepare(array_index)(x, 2) == x[2]
    @test prepare(array_index)(x, 1:2) == x
    @test_throws BoundsError prepare(array_index)(x, 0)
    @test_throws BoundsError prepare(array_index)(x, 3)
    @test prepare(scalar_broadcast)(0.3) == 1.6
    for (spec, expected) in ((custom_concat, 0.4), (custom_broadcast, 0.7))
        source = [0.2, 0.4]
        custom = MutatingReadVector([0.3, 0.5], source)
        @test prepare(spec)(source, custom) ≈ expected
        @test source[1] == 99.0
    end
    @test_throws ReactiveKernels.PlanningError prepare(snapshot_before_effect)
    paired = prepare(paired_broadcasts; bound=(; c=[0.3, 0.7]))
    ad = prepare_ad(paired, AutoEnzyme(; mode=Enzyme.Reverse), x, 0.6; active=:x)
    @test ad_gradient(ad, x, 0.6) ≈ 2 * 0.6^2 .* x
    single = prepare(promoted_entries)
    x32 = Float32[0.2, 0.4]
    @test single(x32) === 3x32[1]
    ad32 = prepare_ad(single, AutoEnzyme(; mode=Enzyme.Reverse), x32; active=:x)
    @test ad_gradient(ad32, x32) == Float32[3, 0]
    column, row = reshape([0.2, 0.4], 2, 1), reshape([0.3, 0.7, -0.2], 1, 3)
    broadcasted = prepare(broadcast_matrix; bound=(; y=row))
    @test broadcasted(column) ≈ 2sum(abs2, column .+ row)
    ad_matrix = prepare_ad(broadcasted, AutoEnzyme(; mode=Enzyme.Reverse), column; active=:x)
    @test ad_gradient(ad_matrix, column) ≈ 4sum(column .+ row; dims=2)
end

function event_allocations(groups, spec=event_columns)
    k = prepare(spec; bound=(; groups))
    ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), 0.6; active=:scale)
    k(0.6), ad_gradient(ad, 0.6)
    ((@allocations k(0.6)), (@allocations ad_gradient(ad, 0.6)))
end
@testset "event-dependent recipes allocate independently of event count" begin
    vectors = map(n -> event_allocations([fill(0.1i, 3) for i in 1:n]), (3, 60))
    @test vectors[1] == vectors[2]
    # The scalar subtraction is lifted out of the event loop. Its generated
    # input name must still resolve when composing the remaining broadcast.
    affine = map(n -> event_allocations([fill(0.1i, 3) for i in 1:n], affine_columns), (3, 60))
    @test affine[1] == affine[2]
    # Matrix broadcasts still materialize per event (`_cell_array_broadcast`).
    matrices = map(n -> event_allocations([fill(0.1i, 2, 2) for i in 1:n]), (3, 60))
    @test_broken matrices[1] == matrices[2]
end

@testset "cell composition preserves index-run grouping" begin
    q = [0.2, -0.4, 0.7]
    for (idx, grouped) in (([1, 3, 2], false),
            (repeat([1, 3, 2]; inner=16), true),
            ([fill(2, 32); fill(1, 64)], true))
        x = [0.1sin(i) for i in eachindex(idx)]
        k = prepare(composed_runs; bound=(; idx, x))
        code = string(code_expr(k))
        @test occursin("run_partial", code) == grouped
        @test occursin("_cell_array_hcat", code)
        @test k(q) ≈ sum(abs2, q[idx] .- x)
        gradient = zero(q)
        for i in eachindex(idx)
            gradient[idx[i]] += 2 * (q[idx[i]] - x[i])
        end
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), q; active=:q)
        @test ad_gradient(ad, q) ≈ gradient
    end
end

@testset "composed arrays read nonscalar indices entry by entry" begin
    x = [0.2, -0.4, 0.7, 1.1]
    dense = 2.0 .* x
    composed = ReactiveKernels._cell_array_broadcast(Base.broadcasted(*, 2.0, x))
    @test composed isa ReactiveKernels._CellBroadcast
    for I in ([3, 1, 3], 2:3, :, Bool[1, 0, 1, 1], Int[], [CartesianIndex(2)],
              reshape([4, 1], 1, 2))
        @test composed[I] == dense[I]
        @test typeof(composed[I]) == typeof(dense[I])
    end
    @test composed[Int32(2)] === dense[2]
    @test_throws BoundsError composed[[1, 5]]
    table = ReactiveKernels._cell_array_hcat(x, composed)
    full = hcat(x, dense)
    for I in ((:, 2), ([4, 2], 1), (2, :), ([1, 3], [2, 1]), (3,), ([5, 8],), (2, 1, 1))
        @test table[I...] == full[I...]
    end
    @test_throws BoundsError table[[1, 5], 1]
end

@testset "a composed value gathered by each group's rows computes those rows" begin
    for groups in (3, 40)
        d = gathered_rows_data(groups, 400)
        reference(s) = sum(sum(exp.(s .+ 0.5 .* d.c[r]) .* yg)
                           for (r, yg) in zip(d.rows, d.y))
        for (k, inputs) in ((prepare(gathered_rows), (d.rows, d.y, d.c, 0.3)),
                            (prepare(gathered_rows; bound=d), (0.3,)))
            @test occursin("_cell_array_broadcast", string(code_expr(k)))
            @test k(inputs...) ≈ reference(0.3)
            # The derivative in `s` of a sum of `exp(s + ...)` terms is the sum.
            ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), inputs...; active=:s)
            @test ad_gradient(ad, inputs...) ≈ reference(0.3)
        end
    end
    # Each row is computed once per call. Before, every group computed the
    # whole per-row value (40 groups: 40 vectors of 1200 entries).
    bytes = map((3, 40)) do groups
        k = prepare(gathered_rows; bound=gathered_rows_data(groups, 1200))
        k(0.3)
        @allocated k(0.3)
    end
    @test bytes[2] < 2bytes[1]
end

# A gather of a computed vector reads what `view` reads and has its type, without
# `view`'s per-call alias check: nothing writes either side of a read-only gather.
@testset "read-only gather of a Vector" begin
    source = [10.0, 20.0, 30.0, 40.0]
    rows = [4, 1, 1]
    gathered = ReactiveKernels._readonly_gather(source, rows)
    @test typeof(gathered) === typeof(view(source, rows))
    @test gathered == [40.0, 10.0, 10.0]
    @test parent(gathered) === source
    @test parentindices(gathered)[1] === rows
    @test isempty(ReactiveKernels._readonly_gather(source, Int[]))
    @test_throws BoundsError ReactiveKernels._readonly_gather(source, [1, 5])
    @test_throws BoundsError ReactiveKernels._readonly_gather(source, [0])
    # Indices that are the parent's own storage still read the right entries.
    self = [2, 1, 3]
    @test ReactiveKernels._readonly_gather(self, self) == [1, 2, 3]
    # Everything else keeps Base's view: a boolean mask and a range index.
    @test ReactiveKernels._readonly_gather(source, [true, false, true, false]) == [10.0, 30.0]
    @test ReactiveKernels._readonly_gather(source, 2:3) == [20.0, 30.0]
end

# A broadcast that a nested plate reads as one of its lanes composes into that
# plate's cells: each cell evaluates its own entry, so a group stores no array.
@kernel lane_series(b::Vector{Vector{Float64}}, y::Vector{Vector{Float64}},
                    off::Vector{Float64}, s::Float64) = begin
    values = plate(eachindex(y)) do g
        m = off[g] .+ s .* b[g]
        terms = plate(m, y[g]) do mi, yi
            abs2(yi - mi)
        end
        sum(terms)
    end
    total = sum(values)
    return total
end
# A function that can throw keeps its eager materialization: its error comes
# before the plate, also when that plate reads no entry.
@kernel lane_eager(b::Vector{Vector{Float64}}, y::Vector{Vector{Float64}}) = begin
    values = plate(eachindex(y)) do g
        m = log.(b[g])
        terms = plate(m, y[g]) do mi, yi
            yi * mi
        end
        sum(terms)
    end
    total = sum(values)
    return total
end
# A reader that is not a plate lane keeps the value stored.
@kernel lane_escapes(b::Vector{Vector{Float64}}, y::Vector{Vector{Float64}}, s::Float64) = begin
    values = plate(eachindex(y)) do g
        m = s .* b[g]
        terms = plate(m, y[g]) do mi, yi
            yi * mi
        end
        sum(terms) + opaque_sum(m)
    end
    total = sum(values)
    return total
end
lane_series_value(b, y, off, s) =
    sum((sum(abs2, y[g] .- (off[g] .+ s .* b[g]); init=0.0) for g in eachindex(y)); init=0.0)
lane_series_off(b, y, off, s) = [-2sum(y[g] .- (off[g] .+ s .* b[g]); init=0.0) for g in eachindex(y)]
lane_series_s(b, y, off, s) =
    sum((-2sum(b[g] .* (y[g] .- (off[g] .+ s .* b[g])); init=0.0) for g in eachindex(y)); init=0.0)
function lane_series_bytes(width)
    b = [fill(0.1g, width) for g in 1:3]; y = [fill(0.2, width) for _ in 1:3]
    off = fill(0.3, 3)
    k = prepare(lane_series)
    ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), b, y, off, 0.7; active=:off)
    k(b, y, off, 0.7), ad_gradient(ad, b, y, off, 0.7)
    ((@allocated k(b, y, off, 0.7)), (@allocated ad_gradient(ad, b, y, off, 0.7)))
end

@testset "a broadcast read as a nested plate's lane composes into its cells" begin
    k = prepare(lane_series)
    @test occursin("_cell_array_broadcast", string(code_expr(k)))
    s = 0.7
    for (b, y) in (([[0.2, -0.4, 0.9], [0.5], Float64[]], [[0.1, 0.3, -0.2], [1.5], Float64[]]),
                   # A one-entry lane is extended over the other lane, as broadcasting does.
                   ([[0.4], [0.2, 0.6]], [[0.1, 0.3, -0.2], [0.5, 0.25]]),
                   (Vector{Float64}[], Vector{Float64}[]))
        off = collect(0.1 .* eachindex(y))
        @test k(b, y, off, s) ≈ lane_series_value(b, y, off, s)
        ad_off = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), b, y, off, s; active=:off)
        @test ad_gradient(ad_off, b, y, off, s) ≈ lane_series_off(b, y, off, s)
        ad_s = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), b, y, off, s; active=:s)
        @test ad_gradient(ad_s, b, y, off, s) ≈ lane_series_s(b, y, off, s)
    end
    # Unequal lanes still fail as broadcasting them fails.
    @test_throws DimensionMismatch k([[0.2, 0.4]], [[0.1, 0.2, 0.3]], [0.0], s)
    # No array is stored per group: neither the value nor its gradient
    # allocates more for wider groups.
    @test lane_series_bytes(4) == lane_series_bytes(64)
    # A function that can throw is materialized at its own statement, so its
    # error comes first (`_cell_array_broadcast` admits total functions only).
    eager = prepare(lane_eager)
    @test eager([[1.0, 2.0]], [[0.5, 0.25]]) ≈ 0.25log(2.0)
    @test_throws DomainError eager([[1.0], [-1.0]], [[0.5], Float64[]])
    # A value that another reader receives whole stays stored.
    escapes = prepare(lane_escapes)
    @test !occursin("_cell_array_broadcast", string(code_expr(escapes)))
    @test escapes([[0.2, 0.4]], [[0.5, 0.25]], 0.5) ≈ (0.5 * 0.1 + 0.25 * 0.2) + (0.1 + 0.2)
end

# A composed broadcast of 32 or more operands, one n-ary `.+` of many terms
# (snag plate-cell-n-ary-3391765b). Base's tuple `map`, `instantiate`, element
# typing and entry read stop specializing there; the cell keeps a concrete,
# type-stable program at every width, so ordinary Reverse differentiates it.
macro wide_lane(name, width)
    terms = [:(q[$i] .* xs[$i][g]) for i in 1:width]
    m = Expr(:call, :.+, :(off[g]), terms...)
    esc(:(@kernel $name(xs::Vector{Vector{Vector{Float64}}}, y::Vector{Vector{Float64}},
                        off::Vector{Float64}, q::Vector{Float64}) = begin
        values = plate(eachindex(y)) do g
            m = $m
            terms = plate(m, y[g]) do mi, yi
                abs2(yi - mi)
            end
            sum(terms)
        end
        total = sum(values)
        return total
    end))
end
@wide_lane wide_lane31 31
@wide_lane wide_lane32 32
@wide_lane wide_lane40 40
wide_lane_mean(xs, off, q, g) = off[g] .+ sum(q[k] .* xs[k][g] for k in eachindex(q))
wide_lane_value(xs, y, off, q) = sum((sum(abs2, y[g] .- wide_lane_mean(xs, off, q, g); init=0.0)
                                      for g in eachindex(y)); init=0.0)
wide_lane_q(xs, y, off, q) = [sum((-2sum(xs[k][g] .* (y[g] .- wide_lane_mean(xs, off, q, g)); init=0.0)
                                   for g in eachindex(y)); init=0.0) for k in eachindex(q)]

@testset "a composed broadcast of 32 or more operands stays type stable" begin
    lens = [3, 1, 0, 5]
    for (width, spec) in ((31, wide_lane31), (32, wide_lane32), (40, wide_lane40))
        xs = [[[0.01k * g + 0.1j for j in 1:lens[g]] for g in eachindex(lens)] for k in 1:width]
        y = [[0.2j - 0.1g for j in 1:lens[g]] for g in eachindex(lens)]
        off = 0.1 .* eachindex(lens)
        q = 0.05 .* sin.(1:width)
        k = prepare(spec; type_stability=:error)
        @test occursin("_cell_array_broadcast", string(code_expr(k)))
        @test k(xs, y, off, q) ≈ wide_lane_value(xs, y, off, q)
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), xs, y, off, q;
                        active=:q, type_stability=:error)
        @test ad_gradient(ad, xs, y, off, q) ≈ wide_lane_q(xs, y, off, q)
    end
    # Unequal operands still fail as broadcasting them fails.
    xs = [i == 40 ? [[0.1, 0.2, 0.3]] : [[0.1, 0.2]] for i in 1:40]
    @test_throws DimensionMismatch prepare(wide_lane40)(
        xs, [[0.1, 0.2]], [0.0], fill(0.1, 40))
end

# A group's gather of a computed vector, read only through a broadcast that a
# nested plate reads as its lane, borrows the vector instead of copying it.
@kernel lane_gather(w::Vector{Vector{Float64}}, traj::Vector{Int}, rows,
                    y::Vector{Vector{Float64}}, k::Float64, off::Vector{Float64}) = begin
    wv = plate(w) do wt
        exp(k) .* wt
    end
    values = plate(eachindex(y)) do g
        c = wv[traj[g]][rows[g]]
        m = off[g] .+ c
        terms = plate(m, y[g]) do mi, yi
            abs2(yi - mi)
        end
        sum(terms)
    end
    total = sum(values)
    return total
end
# The same gather, also returned whole from each cell, keeps its copy.
@kernel lane_gather_escapes(w::Vector{Vector{Float64}}, traj::Vector{Int},
                            rows::Vector{Vector{Int}}, k::Float64) = begin
    wv = plate(w) do wt
        exp(k) .* wt
    end
    picked = plate(eachindex(rows)) do g
        wv[traj[g]][rows[g]]
    end
    return picked
end
lane_gather_value(w, traj, rows, y, k, off) = sum((sum(abs2,
    y[g] .- (off[g] .+ exp(k) .* w[traj[g]][rows[g]]); init=0.0) for g in eachindex(y)); init=0.0)
lane_gather_k(w, traj, rows, y, k, off) = sum((-2sum(exp(k) .* w[traj[g]][rows[g]] .*
    (y[g] .- (off[g] .+ exp(k) .* w[traj[g]][rows[g]])); init=0.0) for g in eachindex(y)); init=0.0)
function lane_gather_bytes(width)
    w = [collect(range(0.1, 1.0; length=80)), collect(range(-0.5, 0.4; length=80))]
    traj = [1, 2, 1]; rows = [collect(1:width), collect(2:width+1), collect(3:width+2)]
    y = [fill(0.2, width) for _ in 1:3]; off = [0.1, -0.2, 0.3]
    k = prepare(lane_gather)
    ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), w, traj, rows, y, 0.3, off; active=:k)
    k(w, traj, rows, y, 0.3, off), ad_gradient(ad, w, traj, rows, y, 0.3, off)
    ((@allocated k(w, traj, rows, y, 0.3, off)), (@allocated ad_gradient(ad, w, traj, rows, y, 0.3, off)))
end

@testset "a gather read only entry by entry borrows its source" begin
    k = prepare(lane_gather)
    @test occursin("_cell_array_gather", string(code_expr(k)))
    w = [collect(0.1:0.1:1.0), collect(-0.5:0.1:0.4)]
    traj = [1, 2, 1]; off = [0.1, -0.2, 0.3]
    for (rows, y) in (([[1, 3, 3, 7], [10, 2], Int[]], [[0.1, 0.4, -0.3, 0.2], [0.5, -0.1], Float64[]]),
                      # A range index keeps Julia's copying gather; the values agree.
                      ([2:4, 1:2, 5:5], [[0.1, 0.4, -0.3], [0.5, -0.1], [0.2]]))
        @test k(w, traj, rows, y, 0.3, off) ≈ lane_gather_value(w, traj, rows, y, 0.3, off)
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), w, traj, rows, y, 0.3, off; active=:k)
        @test ad_gradient(ad, w, traj, rows, y, 0.3, off) ≈ lane_gather_k(w, traj, rows, y, 0.3, off)
    end
    # Every index is checked at the gather, as Julia's indexing checks it.
    @test_throws BoundsError k(w, traj, [[1, 11], [1], Int[]],
        [[0.1, 0.2], [0.3], Float64[]], 0.3, off)
    # Neither the value nor its gradient allocates more for wider groups.
    @test lane_gather_bytes(4) == lane_gather_bytes(64)
    # A gather returned whole is an owned copy, as authored.
    escapes = prepare(lane_gather_escapes)
    @test !occursin("_cell_array_gather", string(code_expr(escapes)))
    picked = escapes(w, traj, [[1, 2], [3], Int[]], 0.0)
    @test picked == [w[1][[1, 2]], w[2][[3]], Float64[]]
    @test all(p -> p isa Vector{Float64}, picked)
end

# A summed plate whose cell is its lane's entry reads the composed value in the
# running total itself, a name the plate assigns twice (seed, then each cell).
@kernel lane_entry_total(b::Vector{Vector{Float64}}, s::Float64) = begin
    values = plate(eachindex(b)) do g
        m = s .* b[g]
        terms = plate(m) do mi
            mi
        end
        sum(terms)
    end
    total = sum(values)
    return total
end
@kernel gather_entry_total(w::Vector{Float64}, rows::Vector{Vector{Int}}, s::Float64) = begin
    scaled = s .* w
    values = plate(eachindex(rows)) do g
        c = scaled[rows[g]]
        terms = plate(c) do ci
            ci
        end
        sum(terms)
    end
    total = sum(values)
    return total
end

@testset "a running total that reads a composed lane keeps its seed" begin
    b = [[0.2, 0.4], Float64[], [1.0]]
    k = prepare(lane_entry_total)
    @test k(b, 0.5) ≈ 0.5 * 1.6
    ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), b, 0.5; active=:s)
    @test ad_gradient(ad, b, 0.5) ≈ 1.6
    w = [0.1, 0.2, 0.3, 0.4]; rows = [[1, 4], Int[], [2, 2, 3]]
    g = prepare(gather_entry_total)
    @test g(w, rows, 2.0) ≈ 2.0 * (0.5 + 0.7)
    adg = prepare_ad(g, AutoEnzyme(; mode=Enzyme.Reverse), w, rows, 2.0; active=:s)
    @test ad_gradient(adg, w, rows, 2.0) ≈ 1.2
end
end
