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

# The data of `jax_layout.py`, by the same formulas.
const LENGTHS = [3, 1, 0, 7, 40, 2, 2, 0, 11, 5]
layout_orders(scale) = [[1 + (i + t) % 2 for t in 1:scale*n] for (i, n) in enumerate(LENGTHS)]
layout_weights(scale) = [[0.1 * t - 0.05 * i for t in 1:scale*((n + i) % 4)]
                         for (i, n) in enumerate(LENGTHS)]
layout_rows(scale) = [[0.3 * sin(i + t) for t in 1:scale*n] for (i, n) in enumerate(LENGTHS)]

@testset "jax_source: layouts of a plate over prepared ragged rows" begin
    have = (:a, :keys, :orders, :weights)
    data = (:keys, :orders, :weights)
    padded = jax_source(jax_layout_walk; have, want = :total, data)
    @test !occursin("_rk_grouped_plate", string(padded))
    @test !occursin("prepare_plate_layout", string(padded))
    @test string(padded) ==
        string(jax_source(jax_layout_walk; have, want = :total, data, plate_layout = :padded))
    walk = jax_source(jax_layout_walk; have, want = :total, data, plate_layout = :groups)
    @test occursin("GROUPED_PLATES = {\"totals_cell\": (\"orders_lengths\", \"weights_lengths\")}",
                   string(walk))
    @test [f.name for f in walk.functions] == ["prepare_data", "kernel", "prepare_plate_layout"]
    rows = jax_source(jax_layout_rows; have = (:a, :xs), want = :total, data = (:xs,),
                      plate_layout = :groups)
    # The second plate maps computed rows beside the prepared ones: both
    # plates take a layout from the prepared lengths.
    @test occursin("\"paths_cell\": (\"xs_lengths\",)", string(rows))
    @test_throws ArgumentError jax_source(jax_layout_walk; have, want = :total, plate_layout = :groups)
    @test_throws ArgumentError jax_source(jax_layout_walk; have, want = :total, data,
                                          plate_layout = :chains)
    if rk_jax_available()
        native_walk = prepare(jax_layout_walk; have, want = :total)
        native_rows = prepare(jax_layout_rows; have = (:a, :xs), want = :total)
        keys = [0.25, -0.5]
        mktempdir() do dir
            write(joinpath(dir, "walk.py"), walk)
            write(joinpath(dir, "rows.py"), rows)
            script = joinpath(@__DIR__, "jax_layout.py")
            expected = (native_walk(0.6, keys, layout_orders(1), layout_weights(1)),
                        native_rows(0.6, layout_rows(1)))
            @test success(`$RK_JAX_PYTHON $script $dir $(expected[1]) $(expected[2])`)
        end
    else
        @test_skip false
    end
end

end
