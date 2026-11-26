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
    source(d) = jax_source(mixed_cell_data; have, want = :total, data = d)
    src = source(data)
    @test all(k -> haskey(src.prepared, k), (:table, :positions))
    larger = (; rows = [data.rows; [[1., 3., 4., 6.]]], keys = [4, 3, 1, 2],
                order = [3, 4, 2, 1, 3])
    @test string(src) == string(source(larger))
    if rk_jax_available()
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

end
