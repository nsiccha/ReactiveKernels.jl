module JAXMixedCellDataTests

using ReactiveKernels, Test
include("jax_python.jl")

@traceable function ordered_rows(rows, keys)
    rows[sortperm(keys)]
end

@kernel function mixed_cell_data(scale::Float64, rows::Vector{Vector{Float64}},
                                 keys::Vector{Int}, order::Vector{Int})
    cells = plate(order, Ref(rows), Ref(keys), Ref(scale)) do i, data, keys, x
        table = ordered_rows(data, keys)
        positions = sortperm(data[i])
        j = x > 0 ? i : keys[i]
        x * (sum(table[j]) + sum(positions))
    end
    total = sum(cells)
    return total
end

@testset "Shared ragged tables and per-cell rows in one JAX preparation" begin
    have = (:scale, :rows, :keys, :order)
    native = prepare(mixed_cell_data; have, want = :total)
    data = (rows = [[2., 3.], Float64[], [7.]], keys = [2, 3, 1], order = [2, 1, 3])
    larger = (; rows = [data.rows; [[1., 3., 4., 6.]]], keys = [4, 3, 1, 2],
                order = [3, 4, 2, 1, 3])
    for layout in (:padded, :groups, :chains)
        source(d) = jax_source(mixed_cell_data; have, want = :total, data = d,
                               plate_layout = layout)
        src = source(data)
        @test all(k -> haskey(src.prepared, k), (:table, :positions))
        @test string(src) == string(source(larger))
        rk_jax_available() || continue
        for (x, d) in ((0.5, data), (-0.5, larger),
                       (1., (; rows = [Float64[]], keys = [1], order = [1])),
                       (1., merge(data, (; order = Int[]))))
            result = rk_jax_evaluate(src, :kernel, (x, d.rows, d.keys, d.order);
                                     prepare = :arguments, grad = true)
            value = native(x, d.rows, d.keys, d.order)
            @test result.value ≈ value
            # Each open half-line is linear in x.
            @test result.grad ≈ value / x
        end
    end
end

@kernel function scalar_row_selector(x::Float64, rows::Vector{Vector{Int}}, pick,
                                     order::Vector{Int})
    cells = plate(order, Ref(rows), pick, Ref(x)) do i, data, s, scale
        shared = sortperm(data[s])
        mapped = sortperm(data[i])
        j = scale > 0 ? 1 : length(shared)
        scale * (shared[j] + sum(mapped))
    end
    total = sum(cells)
    return total
end

@testset "A bound scalar selects one shared permutation in a mixed prefix" begin
    have = (:x, :rows, :pick, :order)
    native = prepare(scalar_row_selector; have, want = :total)
    rows = [[8, 3, 1], Int[], [1, 4]]
    data = (; rows, pick = 1, order = [2, 3, 1, 3])
    for layout in (:padded, :groups, :chains)
        src = jax_source(scalar_row_selector; have, want = :total, data, plate_layout = layout)
        @test all(k -> haskey(src.prepared, k), (:shared, :mapped))
        inputs = Dict(only(f for f in src.functions if f.name == "kernel").inputs)
        @test inputs[src.prepared[:shared]] <: AbstractVector{<:Integer}
        rk_jax_available() || continue
        for (x, pick, order) in ((0.5, 1, data.order), (-0.5, 3, data.order),
                                 (1., 0, Int[]))
            args = (x, rows, pick, order)
            result = rk_jax_evaluate(src, :kernel, args; prepare = :arguments, grad = true)
            @test result.value ≈ native(args...)
            @test result.grad ≈ native(args...) / x
        end
    end
end

end
