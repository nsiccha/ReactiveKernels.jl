using ReactiveKernels, Test, Enzyme

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# Column assembly (snag jax-reduce-hcat-c647a1a8): an ordinary helper turns a
# vector of columns, such as a plate's per-cell vectors, into Julia's matrix.
jax_columns_matrix(columns) = reduce(hcat, columns)
jax_columns_weighted(columns, W) = sum(jax_columns_matrix(columns) .* W)

@kernel jax_columns_assembled(columns::Vector{Vector{Float64}}) = begin
    result = jax_columns_matrix(columns)
    return result
end
@kernel jax_columns_total(columns::Vector{Vector{Float64}}, W::Matrix{Float64}) = begin
    matrix = jax_columns_matrix(columns)
    total = sum(matrix .* W)
    return total
end
# Plate cells become the columns; a row gather reads the matrix afterwards.
@kernel jax_columns_cells(a::Float64, xs::Vector{Vector{Float64}}, idx::Vector{Int},
                          W::Matrix{Float64}) = begin
    cells = plate(xs) do x
        a .* x .+ 1.0
    end
    matrix = jax_columns_matrix(cells)
    expanded = matrix[idx, :]
    total = sum(expanded .* W)
    return total
end
# Matrix cells contribute their columns in turn.
@kernel jax_columns_blocks(s::Vector{Float64}) = begin
    cells = plate(s) do v
        hcat([v, 3.0 * v], [2.0 * v, v^2])
    end
    matrix = jax_columns_matrix(cells)
    return matrix
end
@kernel jax_columns_blocks_total(s::Vector{Float64}, W::Matrix{Float64}) = begin
    cells = plate(s) do v
        hcat([v, 3.0 * v], [2.0 * v, v^2])
    end
    matrix = jax_columns_matrix(cells)
    total = sum(matrix .* W)
    return total
end
# Data-only assembly runs in host `prepare_data`.
@kernel jax_columns_prepared(a::Float64, columns::Vector{Vector{Float64}}) = begin
    matrix = jax_columns_matrix(columns)
    total = a * sum(matrix .* matrix) + matrix[2, 1]
    return total
end
# The same reducer value reaches `reduce(vcat, ·)` from inside a helper.
jax_columns_flat(cells) = reduce(vcat, cells)
@kernel jax_columns_flat_total(R::Vector{Vector{Float64}}, a::Float64) = begin
    scaled = plate(R) do r
        a .* r
    end
    flat = jax_columns_flat(scaled)
    total = sum(abs2, flat)
    return total
end
@kernel jax_columns_numbers(x::Vector{Float64}) = begin
    result = jax_columns_matrix(x)
    return result
end

@testset "jax_source: reduce(hcat, columns)" begin
    columns = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
    saved = deepcopy(columns)
    @test jax_columns_matrix(columns) == [1.0 2.0; 3.0 4.0; 5.0 6.0]
    @test prepare(jax_columns_assembled)(columns) == [1.0 2.0; 3.0 4.0; 5.0 6.0]
    @test isequal(columns, saved)
    @test_throws DimensionMismatch jax_columns_matrix([[1.0, 2.0], [3.0]])

    assembled = jax_source(jax_columns_assembled; have = (:columns,), want = :result)
    @test occursin("jnp.transpose(", string(assembled))
    @test occursin("each must fill the padded width", string(assembled))
    @test !occursin("IndexStyle", string(assembled))
    blocks = jax_source(jax_columns_blocks; have = (:s,), want = :matrix)
    @test occursin("(1, 0, 2)", string(blocks))
    numbers = try
        jax_source(jax_columns_numbers; have = (:x,), want = :result)
        nothing
    catch e
        e
    end
    @test numbers isa JAXEmissionError
    @test occursin("permutedims", sprint(showerror, numbers))

    # Native ordinary Reverse is the derivative reference.
    W = [0.5 -1.0; 2.0 0.25; -0.75 1.5]
    dcolumns = [zeros(3), zeros(3)]
    Enzyme.autodiff(Enzyme.Reverse, jax_columns_weighted, Enzyme.Active,
                    Enzyme.Duplicated(deepcopy(columns), dcolumns), Enzyme.Const(W))
    @test dcolumns == [W[:, 1], W[:, 2]]

    if !rk_jax_available()
        @test_skip false
    else
        # Julia's orientation: column k of the matrix is columns[k].
        for cols in (columns, [[0.5], [1.5], [2.5]], [[1.0, 2.0, 3.0, 4.0]])
            r = rk_jax_evaluate(assembled, :kernel, (cols,))
            @test r.value == collect(eachrow(jax_columns_matrix(cols)))
        end

        total = jax_source(jax_columns_total; have = (:columns, :W), want = :total)
        r = rk_jax_evaluate(total, :kernel, (columns, W); grad = true)
        @test r.value ≈ prepare(jax_columns_total)(columns, W)
        @test r.grad ≈ dcolumns

        cells = jax_source(jax_columns_cells; have = (:a, :xs, :idx, :W), want = :total)
        xs = [[0.2, -0.4, 0.9], [1.1, 0.3, -0.6]]
        idx = [1, 1, 2, 3, 3, 2]
        Wc = reshape(collect(range(-1.0, 1.0; length = 12)), 6, 2)
        for a in (0.3, -1.2)
            r = rk_jax_evaluate(cells, :kernel, (a, xs, idx, Wc); grad = true)
            @test r.value ≈ prepare(jax_columns_cells)(a, xs, idx, Wc) rtol = 1e-12
            @test r.grad ≈ sum(reduce(hcat, xs)[idx, :] .* Wc) rtol = 1e-12
        end

        s = [0.4, -1.3, 2.0]
        r = rk_jax_evaluate(blocks, :kernel, (s,))
        @test r.value ≈ collect(eachrow(prepare(jax_columns_blocks)(s)))
        @test size(prepare(jax_columns_blocks)(s)) == (2, 6)
        Wb = reshape(collect(range(-2.0, 2.0; length = 12)), 2, 6)
        blocks_total = jax_source(jax_columns_blocks_total; have = (:s, :W), want = :total)
        r = rk_jax_evaluate(blocks_total, :kernel, (s, Wb); grad = true)
        @test r.value ≈ prepare(jax_columns_blocks_total)(s, Wb)
        ds = zeros(length(s))
        Enzyme.autodiff(Enzyme.Reverse,
                        (s, W) -> sum(reduce(hcat, [hcat([v, 3.0 * v], [2.0 * v, v^2]) for v in s]) .* W),
                        Enzyme.Active, Enzyme.Duplicated(copy(s), ds), Enzyme.Const(Wb))
        @test r.grad ≈ ds

        # No columns: Julia's reduce over an empty collection raises, and the
        # module raises while JAX traces it.
        @test_throws MethodError prepare(jax_columns_blocks)(Float64[])
        err = try
            rk_jax_evaluate(blocks, :kernel, (Float64[],))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("reducing over an empty collection", sprint(showerror, err))

        prep = jax_source(jax_columns_prepared; have = (:a, :columns), want = :total,
                          data = (columns = columns,))
        @test occursin("np.column_stack(", string(prep))
        for cols in (columns, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            r = rk_jax_evaluate(prep, :kernel, (0.7, cols); grad = true, prepare = true)
            @test r.value ≈ prepare(jax_columns_prepared)(0.7, cols)
            @test r.grad ≈ sum(abs2, jax_columns_matrix(cols))
        end

        R = [[0.5, 1.0], [2.0], [0.1, 0.2, 0.3]]
        flat = jax_source(jax_columns_flat_total; have = (:a, :R), want = :total, data = (; R))
        @test occursin("jnp.nonzero(", string(flat))
        r = rk_jax_evaluate(flat, :kernel, (0.7, R); grad = true, prepare = true)
        @test r.value ≈ prepare(jax_columns_flat_total)(R, 0.7) rtol = 1e-12
        @test r.grad ≈ 2 * 0.7 * sum(abs2, reduce(vcat, R)) rtol = 1e-10

        # Columns of different lengths are Julia's DimensionMismatch: the module
        # raises when it runs on them, on the device and on the host.
        for (source, args, preparation) in ((assembled, ([[1.0, 2.0], [3.0]],), false),
                                            (prep, (0.7, [[1.0, 2.0], [3.0]]), true))
            err = try
                rk_jax_evaluate(source, :kernel, args; prepare = preparation)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin(preparation === false ?
                           "reduce(hcat, columns): the columns differ in length" :
                           "ValueError", sprint(showerror, err))
        end
    end
end
