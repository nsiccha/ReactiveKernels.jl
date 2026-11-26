module JAXLayoutTests

using ReactiveKernels, Test
include("jax_python.jl")

# A plate over units with ragged sequences: each unit walks its own events
# through a shared record table, beside a second ragged argument of its own
# lengths.
@kernel jax_layout_walk(a::Float64, keys::Vector{Float64}, orders::Vector{Vector{Int}},
                        weights::Vector{Vector{Float64}}) = begin
    entries = plate(keys) do key
        (key, a * key)
    end
    totals = plate(orders, weights, Ref(entries)) do order, w, table
        path = scan(order, Ref(table); init = 0.0) do state, index, rows
            row = rows[index]
            next = 0.5 * state + expm1(row[2]) * row[1]
            (next, next)
        end
        sum(path) + a * sum(abs2, w)
    end
    total = sum(totals)
    total
end

# A plate whose cells return rows of their units' lengths, read by a second
# plate over the same units.
@kernel jax_layout_rows(a::Float64, xs::Vector{Vector{Float64}}) = begin
    paths = plate(xs) do x
        scan(x; init = 0.0) do state, value
            next = a * state + value
            (next, next)
        end
    end
    sums = plate(paths, xs) do path, x
        sum(path .* x)
    end
    total = sum(sums)
    total
end

# A plate over unit indices whose cell reads its unit's row from a shared
# ragged array: the row is cut to its group's width where it is read.
@kernel jax_layout_index(a::Float64, units::Vector{Int}, xs::Vector{Vector{Float64}}) = begin
    sums = plate(units, Ref(xs)) do i, rows
        x = rows[i]
        path = scan(x; init = 0.0) do state, value
            next = a * state + value
            (next, next)
        end
        sum(path .* x)
    end
    total = sum(sums)
    total
end

# A step whose record read only one arm makes (passed to the scan as a
# scanned input) in a plate that takes a layout: the scanned rows are cut to
# each group's width with the events they belong to.
@kernel jax_layout_guarded(a::Float64, keys::Vector{Float64}, orders::Vector{Vector{Int}}) = begin
    entries = plate(keys) do key
        (key, a * key)
    end
    totals = plate(orders, Ref(entries)) do order, table
        path = scan(order, Ref(table); init = 0.0) do state, index, rows
            next = index > 1 ? 0.5 * state + rows[index][2] : state + a
            (next, next)
        end
        sum(path)
    end
    total = sum(totals)
    total
end

# The data of `jax_layout.py`, by the same formulas.
const LENGTHS = [3, 1, 0, 7, 40, 2, 2, 0, 11, 5]
# Units that repeat rows, skip some, and are not in the rows' order.
const LAYOUT_UNITS = [5, 1, 1, 9, 3, 10, 4, 4, 2, 8, 5, 6]
layout_orders(scale) = [[1 + (i + t) % 2 for t in 1:scale*n] for (i, n) in enumerate(LENGTHS)]
layout_weights(scale) = [[0.1 * t - 0.05 * i for t in 1:scale*((n + i) % 4)]
                         for (i, n) in enumerate(LENGTHS)]
layout_rows(scale) = [[0.3 * sin(i + t) for t in 1:scale*n] for (i, n) in enumerate(LENGTHS)]

@kernel jax_dense_layout(a::Float64, x::Vector{Float64}) = begin
    total = a * sum(x)
    total
end

@testset "jax_source: layouts of a plate over prepared ragged rows" begin
    have = (:a, :keys, :orders, :weights)
    data = (:keys, :orders, :weights)
    padded = jax_source(jax_layout_walk; have, want = :total, data, plate_layout = :padded)
    @test !occursin("_rk_grouped_plate", string(padded))
    @test !occursin("prepare_plate_layout", string(padded))
    walk = jax_source(jax_layout_walk; have, want = :total, data, plate_layout = :groups)
    default = get(task_local_storage(), :reactivekernels_jax_plate_layout_default, :padded)
    @test string(jax_source(jax_layout_walk; have, want = :total, data)) ==
          string(default === :groups ? walk : padded)
    @test occursin("GROUPED_PLATES = {\"totals_cell\": (\"orders_lengths\", \"weights_lengths\")}",
                   string(walk))
    @test [f.name for f in walk.functions] ==
        ["prepare_data", "kernel", "prepare_plate_layout", "select_plate_layout", "bind_gradient"]
    rows = jax_source(jax_layout_rows; have = (:a, :xs), want = :total, data = (:xs,),
                      plate_layout = :groups)
    # The second plate maps computed rows beside the prepared ones: both
    # plates take a layout from the prepared lengths.
    @test occursin("\"paths_cell\": (\"xs_lengths\",)", string(rows))
    index = jax_source(jax_layout_index; have = (:a, :units, :xs), want = :total,
                       data = (:units, :xs), plate_layout = :groups)
    # A row read at the plate's own index is grouped by the lengths of the
    # rows the units select.
    @test occursin("GROUPED_PLATES = {\"sums_cell\": ((\"xs_lengths\", \"units\"),)}",
                   string(index))
    @test occursin(r"rows\w*\[i - 1\]\[:row_widths\[0\]\]", string(index))
    @test !occursin("row_widths", string(jax_source(jax_layout_index; have = (:a, :units, :xs),
                                                    want = :total, data = (:units, :xs), plate_layout = :padded)))
    @test_throws ArgumentError jax_source(jax_layout_walk; have, want = :total, plate_layout = :groups)
    @test_throws ArgumentError jax_source(jax_layout_walk; have, want = :total, data,
                                          plate_layout = :unknown)
    if rk_jax_available()
        native_walk = prepare(jax_layout_walk; have, want = :total)
        native_rows = prepare(jax_layout_rows; have = (:a, :xs), want = :total)
        native_index = prepare(jax_layout_index; have = (:a, :units, :xs), want = :total)
        native_guarded = prepare(jax_layout_guarded; have = (:a, :keys, :orders), want = :total)
        guarded = jax_source(jax_layout_guarded; have = (:a, :keys, :orders), want = :total,
                             data = (:keys, :orders), plate_layout = :groups)
        keys = [0.25, -0.5]
        mktempdir() do dir
            dense = jax_source(jax_dense_layout; have=(:a, :x), want=:total, data=(:x,), plate_layout=:groups)
            write(joinpath(dir, "dense.py"), dense)
            write(joinpath(dir, "walk.py"), walk)
            write(joinpath(dir, "rows.py"), rows)
            write(joinpath(dir, "index.py"), index)
            write(joinpath(dir, "guarded.py"), guarded)
            binding_test = joinpath(@__DIR__, "jax_gradient_binding.py")
            @test success(pipeline(`$RK_JAX_PYTHON $binding_test $dir`; stderr))
            script = joinpath(@__DIR__, "jax_layout.py")
            expected = (native_walk(0.6, keys, layout_orders(1), layout_weights(1)),
                        native_rows(0.6, layout_rows(1)),
                        native_index(0.6, LAYOUT_UNITS, layout_rows(1)),
                        native_guarded(0.6, keys, layout_orders(1)))
            @test success(pipeline(`$RK_JAX_PYTHON $script $dir $(expected[1]) $(expected[2]) $(expected[3]) $(expected[4])`; stderr))
        end
    else
        @test_skip false
    end
end

end
