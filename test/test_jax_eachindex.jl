module JAXEachindexTests

using ReactiveKernels, Test
include("jax_python.jl")

pick_entries(table, groups, rows) =
    [table[groups[i]][rows[i]] for i in eachindex(groups, rows)]

@kernel function observed_total(values::Vector{Vector{Float64}}, groups::Vector{Int},
                                rows::Vector{Int})
    reads = pick_entries(values, groups, rows)
    total = sum(abs2, reads)
    return total
end

@kernel function indexed_gather(x::Vector{Float64}, table::Vector{Vector{Int}},
                                groups::Vector{Int}, rows::Vector{Int})
    indices = pick_entries(table, groups, rows)
    total = sum(abs2, x[indices])
    return total
end

@kernel function guarded_read(take::Bool, a::Vector{Int}, b::Vector{Float64})
    value = take ? sum(b[i] for i in eachindex(a, b); init = 0.0) : 7.0
    return value
end

@kernel index_count(a, b) = begin
    n = length(eachindex(a, b))
end

@kernel function counted_rows(scale::Float64, A::Vector{Vector{Int}}, B::Vector{Vector{Int}},
                              C::Vector{Vector{Int}})
    cells = plate(A, B, C) do a, b, c
        if scale > 0
            indices = eachindex(a, b, c)
            total = 0.0
            for i in eachindex(a, b)
                total += scale * b[i]
            end
            # At the first position past a row, the denominator would be zero.
            terms = [scale / (length(a) - i + 1) for i in eachindex(a, b)]
            total + length(collect(indices)) + sum(terms)
        else
            scale
        end
    end
    result = sum(cells)
    return result
end

@testset "JAX eachindex checks every vector's logical axis" begin
    # Equal lengths do not establish equal axes for shifted vectors. The
    # array arguments of a JAX module carry no Julia axis origins.
    left, right = Base.Slice(0:1), Base.Slice(1:2)
    @test_throws DimensionMismatch eachindex(left, right)
    @test_throws "one-based linear index domain" jax_source(index_count;
        have = (:a, :b), want = :n, types = (a = typeof(left), b = typeof(right)))
    ports = (:x, :table, :groups, :rows)
    direct = jax_source(indexed_gather; have = ports, want = :total)
    native = prepare(indexed_gather; have = ports, want = :total)
    table = [[2, 1], Int[], [4]]
    bound(groups, rows) = jax_source(indexed_gather; data = (; table, groups, rows),
                                     functions = (value = :total,))
    prepared = bound([1], [2])
    @test string(bound(Int[], Int[])) == string(bound(fill(1, 37), fill(1, 37))) == string(prepared)
    @test occursin("jax.vmap(", string(direct))
    @test occursin("range(1,", string(prepared))

    x = [1.5, 2.0, -3.0, -0.5]
    @test_throws DimensionMismatch native(x, table, [1], [1, 2])
    if !rk_jax_available()
        @test_skip false
    else
        for (src, name, prep) in ((direct, :kernel, false), (prepared, :value, :arguments))
            for (groups, rows) in ((Int[], Int[]), ([1, 3, 1], [1, 1, 1]),
                                   ([3, 1, 1], [1, 2, 1]), (fill(1, 37), fill(2, 37)))
                got = rk_jax_evaluate(src, name, (x, table, groups, rows); grad = :joint, prepare = prep)
                @test got.value == got.joint_value == native(x, table, groups, rows)
                indices = pick_entries(table, groups, rows)
                @test got.grad == [2x[i] * count(==(i), indices) for i in eachindex(x)]
            end
            @test_throws "all inputs to eachindex must have the same indices" rk_jax_evaluate(
                src, name, (x, table, [1], [1, 2]); prepare = prep)
        end
        # Differentiate the captured ragged values themselves. Repeated
        # positions accumulate and empty rows retain zero derivatives.
        values = [[1.5, -0.5], Float64[], [2.0]]
        ports = (:values, :groups, :rows)
        observed = jax_source(observed_total; have = ports, want = :total)
        native_total = prepare(observed_total; have = ports, want = :total)
        stored, _ = rk_jax_pad(values)
        for (groups, rows) in (([1, 3, 1], [2, 1, 2]), ([1, 3], [1, 1]), (Int[], Int[]))
            got = rk_jax_evaluate(observed, :kernel, (values, groups, rows); grad = :joint)
            @test got.value == got.joint_value == native_total(values, groups, rows)
            expected = [[2 * stored[g, r] * count(==((g, r)), zip(groups, rows))
                         for r in axes(stored, 2)] for g in axes(stored, 1)]
            @test got.grad == expected
        end
    end

    ports = (:scale, :A, :B, :C)
    src = jax_source(counted_rows; have = ports, want = :result)
    native_rows = prepare(counted_rows; have = ports, want = :result)
    a = [[1, 2], Int[], [3]]
    b = [[4, 5], Int[], [-2]]
    # The same storage dimensions and different per-row lengths.
    bad = [[4], Int[], [-2, 5]]
    if rk_jax_available()
        for (s, b_rows, c_rows) in ((1.5, b, a), (-1.5, bad, bad))
            got = rk_jax_evaluate(src, :kernel, (s, a, b_rows, c_rows); grad = :joint)
            @test got.value == got.joint_value == native_rows(s, a, b_rows, c_rows)
            @test got.grad == (s > 0 ? sum(sum, b) + 2.5 : length(a))
        end
        stored_a, lengths = rk_jax_pad(a)
        stored_b, _ = rk_jax_pad(b)
        wider_a = hcat(stored_a, fill(99, length(a), 2))
        got = rk_jax_evaluate(src, :kernel,
            (1.5, wider_a, lengths, stored_b, lengths, stored_a, lengths); grad = :joint)
        @test got.value == native_rows(1.5, a, b, a)
        @test got.grad == sum(sum, b) + 2.5
        for (b_rows, c_rows) in ((bad, a), (b, bad))
            @test_throws DimensionMismatch native_rows(1.5, a, b_rows, c_rows)
            @test_throws "all inputs to eachindex must have the same indices" rk_jax_evaluate(
                src, :kernel, (1.5, a, b_rows, c_rows))
        end
        guarded = jax_source(guarded_read; have = (:take, :a, :b), want = :value)
        @test rk_jax_evaluate(guarded, :kernel, (false, [1], Float64[])).value == 7.0
        @test_throws "all inputs to eachindex must have the same indices" rk_jax_evaluate(
            guarded, :kernel, (true, [1], Float64[]))
    end
end

end
