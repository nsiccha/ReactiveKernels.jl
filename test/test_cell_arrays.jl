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

function event_allocations(groups)
    k = prepare(event_columns; bound=(; groups))
    ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), 0.6; active=:scale)
    k(0.6), ad_gradient(ad, 0.6)
    ((@allocations k(0.6)), (@allocations ad_gradient(ad, 0.6)))
end
@testset "event-dependent recipes allocate independently of event count" begin
    vectors = map(n -> event_allocations([fill(0.1i, 3) for i in 1:n]), (3, 60))
    @test vectors[1] == vectors[2]
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
end
