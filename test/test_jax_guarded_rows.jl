module JAXGuardedRowTests
using ReactiveKernels, Test
include("jax_python.jl")

# Each cell of a padded plate reads the row of a shared table that its index
# selects. The cell runs under its activity guard. Once vmap turns that guard
# into a select, a read inside it gives every cell a cotangent of the whole
# table (snag jax-if-nonempty-91c59ce4).
@kernel guarded_rows(table::Vector{Vector{Float64}}, weights::Vector{Float64},
                     groups::Vector{Vector{Vector{Int}}}) = begin
    parts = plate(groups, Ref(table), Ref(weights)) do group, t, u
        cells = plate(eachindex(group), Ref(group), Ref(t), Ref(u)) do i, g, shared, w
            w[i] .* log.(shared[i][g[i]])
        end
        again = plate(eachindex(group), Ref(cells)) do i, selected
            sum(selected[i])
        end
        sum(again)
    end
    total = sum(parts)
end

@testset "guarded plate cells gather their shared rows before the guard" begin
    have = (:table, :weights, :groups)
    native = prepare(guarded_rows; have, want = :total)
    direct = jax_source(guarded_rows; have, want = :total)
    bound = jax_source(guarded_rows; have, want = :total, data = (:groups,))
    for src in (direct, bound)
        # A weight, a table row and a row of `cells` are gathered outside
        # the two cells' guards and passed into them.
        @test count(r"_rk_if_nonempty\(active_cell, lambda: \w+\([^)]*selected_row", string(src)) == 2
    end
    table = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    weights = [0.5, 2.0]
    @test native(table, weights, [[[1, 2, 2], [3]], Vector{Int}[], [[2]]]) ≈
          0.5 * (log(1.0) + 2log(2.0)) + 2log(6.0) + 0.5log(2.0)
    @test native([Float64[] for _ in 1:4], ones(4), [[Int[] for _ in 1:4], [Int[]]]) == 0
    @test_throws BoundsError native(table, weights, [[[1], [2], [3]]])
    @test_throws BoundsError native([Float64[] for _ in 1:4], ones(4), [[Int[], [1]]])
    if rk_jax_available()
        mktempdir() do dir
            paths = [joinpath(dir, "direct.py"), joinpath(dir, "bound.py")]
            write(paths[1], direct)
            write(paths[2], bound)
            driver = joinpath(@__DIR__, "jax_guarded_rows.py")
            @test success(pipeline(`$RK_JAX_PYTHON $driver $paths`; stdout, stderr))
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
end
