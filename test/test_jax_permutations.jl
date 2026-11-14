using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# `sortperm` and `invperm` of data run in the host `prepare_data`, as `sort`
# and `unique` do: Julia's own methods have loops and branches, so neither
# the functions nor `prepare_data` could compute them (snag
# jax-host-transla-e885b100).
@kernel jax_permutation_values(theta::Float64, x::Vector{Float64}, p::Vector{Int},
                               k::Vector{Int}) = begin
    order = sortperm(x)
    inverse = invperm(p)
    counts = sortperm(k)
    total = theta .* vcat(order, inverse, counts)
    return total
end
# The position of every row in the concatenation of two row lists.
@kernel jax_row_positions(theta::Vector{Float64}, rows_a::Vector{Int},
                          rows_b::Vector{Int}) = begin
    pos = invperm(vcat(rows_a, rows_b))
    order = sortperm(pos)
    total = sum(theta[pos] .* theta) + sum(theta[order] .* theta)
    return total
end
# `sortperm` in a recipe that reads a parameter runs in the JAX functions,
# which have no translation of it: the refusal names it.
@kernel jax_traced_order(theta::Vector{Float64}, rows::Vector{Int}) = begin
    total = sum(theta[sortperm(rows)] .* theta)
    return total
end
@traceable function jax_ranks(x)
    order = sortperm(x)
    ranks = invperm(order)
    ranks
end
@kernel jax_rank_weights(theta::Vector{Float64}, x::Vector{Float64}) = begin
    ranks = jax_ranks(x)
    total = sum(theta .* ranks)
    return total
end
@kernel jax_reverse_order(theta::Float64, x::Vector{Float64}) = begin
    order = sortperm(x; rev = true)
    total = theta * sum(order)
    return total
end

_jax_host_part(src) = split(split(string(src), "def prepare_data")[2], "\ndef kernel")[1]

@testset "jax_source: sortperm and invperm in prepare_data" begin
    have = (:theta, :x, :p, :k)
    # Julia's `isless` order: -0.0 before 0.0, every NaN last, equal values
    # in ascending position (`sortperm` is stable).
    x = [0.5, NaN, -0.0, 0.0, 0.5, -1.0, NaN, 0.0]
    p = [3, 1, 4, 2, 6, 5]
    k = [2, 1, 2, 1, 3]
    src = jax_source(jax_permutation_values; have, want = :total, data = (; x, p, k))
    host = _jax_host_part(src)
    @test !occursin("Computed by the functions", host)
    @test !occursin("jnp.", host)
    @test all(h -> occursin(h, host), ("_rk_sortperm_np(", "_rk_invperm_np("))
    # One module text across data sizes: nothing is unrolled per entry.
    longer = jax_source(jax_permutation_values; have, want = :total,
                        data = (; x = [x; x], p = [p; 7:12], k = [k; k]))
    @test longer.text == src.text

    rows = (; rows_a = [1, 3, 4], rows_b = [2, 5])
    pos = jax_source(jax_row_positions; have = (:theta, :rows_a, :rows_b), want = :total,
                     data = rows)
    @test occursin("_rk_invperm_np(", _jax_host_part(pos))
    # Through a `@traceable` helper's recorded body as well: its host
    # function, which `prepare_data` calls, sorts and inverts.
    ranked = jax_source(jax_rank_weights; have = (:theta, :x), want = :total, data = (; x))
    @test occursin("_jax_ranks_np(x)", _jax_host_part(ranked))
    @test !occursin("Computed by the functions", _jax_host_part(ranked))
    helper = split(split(string(ranked), "def _jax_ranks_np")[2], "\ndef ")[1]
    @test all(h -> occursin(h, helper), ("= _rk_sortperm_np(", "= _rk_invperm_np("))

    # Keyword orders have no host translation; the error names the keyword.
    err = try
        jax_source(jax_reverse_order; have = (:theta, :x), want = :total, data = (; x))
        nothing
    catch e
        e
    end
    @test err isa JAXEmissionError
    @test occursin("keyword arguments to `sortperm`", err.message)
    err = try
        jax_source(jax_traced_order; have = (:theta, :rows), want = :total,
                   data = (; rows = [2, 1, 3]))
        nothing
    catch e
        e
    end
    @test err isa JAXEmissionError
    @test occursin("`sortperm` has no translation in the JAX functions", err.message)

    if !rk_jax_available()
        @test_skip "RK_JAX_PYTHON not set"
    else
        for (x, p, k) in ((x, p, k), (Float64[], Int[], Int[]), ([2.0], [1], [7]),
                          (collect(10.0:-1.0:1.0), collect(10:-1:1), [5, 5, 5, 1, 1]))
            src_x = jax_source(jax_permutation_values; have, want = :total,
                               data = (; x, p, k))
            @test src_x.text == src.text
            r = rk_jax_evaluate(src_x, "kernel", (1.0, x, p, k); prepare = true)
            @test r.value == Float64[sortperm(x); invperm(p); sortperm(k)]
        end
        theta = collect(1.0:5.0) ./ 7
        native = prepare(jax_row_positions; have = (:theta, :rows_a, :rows_b), want = :total)
        r = rk_jax_evaluate(pos, "kernel", (theta, rows.rows_a, rows.rows_b); grad = true,
                            prepare = true)
        @test r.value ≈ native(theta, rows.rows_a, rows.rows_b) rtol = 1e-12
        @test r.grad ≈ rk_fd_gradient(t -> native(t, rows.rows_a, rows.rows_b), theta) rtol = 1e-6
        theta = collect(1.0:length(x)) ./ 3
        native = prepare(jax_rank_weights; have = (:theta, :x), want = :total)
        @test rk_jax_evaluate(ranked, "kernel", (theta, x); prepare = true).value ≈
              native(theta, x) rtol = 1e-12
        # Not a permutation: Python raises as Julia's `invperm` throws.
        for bad in ([1, 1, 2], [0, 1, 2], [1, 2, 4])
            @test_throws ArgumentError invperm(bad)
            src_bad = jax_source(jax_permutation_values; have, want = :total,
                                 data = (; x = [1.0, 2.0, 3.0], p = bad, k = [1, 2, 3]))
            err = try
                rk_jax_evaluate(src_bad, "kernel", (1.0, [1.0, 2.0, 3.0], bad, [1, 2, 3]);
                                prepare = true)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("argument is not a permutation", err.msg)
        end
    end
end
