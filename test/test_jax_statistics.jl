using ReactiveKernels, Statistics, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# Statistics' `mean`, `var` and `std` (`ReactiveKernelsStatisticsExt`). Their
# Julia methods branch, so `jax_source` refused them as "the function `_mean`
# has loops or branches" (snag brm-mean-reducti-ab8534c4: a centred effect
# `s * (x - mean(x))` and `A .- mean(A; dims = 1)` of a BRM model).

# Centred values, whole-array and function forms, both `corrected` values.
@kernel jax_stat_vector(x::Vector{Float64}, s::Float64) = begin
    centred = s .* (x .- mean(x))
    total = sum(abs2, centred) + mean(abs2, x) + var(x) + std(x; corrected = false) +
            var(x; mean = 0.5)
    return total
end
# Reductions over `dims`, the result kept as Julia keeps it (a reduced axis of
# length one), and `dims = :`.
@kernel jax_stat_dims(A::Matrix{Float64}) = begin
    columns = A .- mean(A; dims = 1)
    rows = A .- mean(exp, A; dims = 2)
    spread = var(A; dims = 1) .* std(A; dims = 2, corrected = false)
    total = sum(abs2, columns) + sum(rows) + sum(spread) + mean(A; dims = :)
    return total
end
# Padded rows of a plate: each cell reduces its own entries only.
@kernel jax_stat_ragged(theta::Float64, xs::Vector{Vector{Float64}}) = begin
    ys = plate(xs) do x
        m = mean(theta .* x)
        m + var(theta .* x; corrected = false) + mean(x; dims = 1)[1]
    end
    total = sum(ys)
    return total
end
# A data-only centred column is `prepare_data` host code.
@kernel jax_stat_host(theta::Float64, x::Vector{Float64}) = begin
    centred = (x .- mean(x)) ./ std(x)
    total = sum(abs2, theta .* centred)
    return total
end
# `corrected` must be known while JAX traces.
@kernel jax_stat_computed_corrected(x::Vector{Float64}) = begin
    y = var(x; corrected = x[1] > 0)
    return y
end
# Empty and single-entry arguments keep Julia's NaN.
@kernel jax_stat_edge(x::Vector{Float64}) = begin
    values = [mean(x), var(x), std(x), var(x; corrected = false)]
    return values
end

@testset "jax_source: Statistics mean, var and std" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsStatisticsExt) !== nothing
    v = jax_source(jax_stat_vector; have = (:x, :s), want = :total)
    @test !occursin("_mean", string(v))
    @test occursin("jnp.sum(x)", string(v))
    d = jax_source(jax_stat_dims; have = (:A,), want = :total)
    @test occursin("axis=", string(d))
    # The module is the same text for every data length: the reductions are
    # whole-array calls, never unrolled over the entries.
    host(n) = string(jax_source(jax_stat_host; have = (:theta, :x), want = :total,
                                data = (x = collect(range(0.1, 2.0; length = n)),)))
    @test host(3) == host(40)
    @test occursin("def prepare_data", host(3))
    @test_throws "corrected" jax_source(jax_stat_computed_corrected; have = (:x,), want = :y)
    if !rk_jax_available()
        @test_skip false
    else
        for x in ([0.3, -1.2, 2.5, 0.8], [0.7, 0.1])
            native = prepare(jax_stat_vector; have = (:x, :s), want = :total)
            r = rk_jax_evaluate(v, :kernel, (x, 1.7); grad = true)
            @test r.value ≈ native(x, 1.7) rtol = 1e-13
            @test r.grad ≈ rk_fd_gradient(u -> native(u, 1.7), x) rtol = 1e-6 atol = 1e-8
        end
        for A in ([0.3 -1.2 2.5; 0.8 0.1 -0.4], [0.2 1.1; 0.9 -0.3; -0.5 0.4])
            native = prepare(jax_stat_dims; have = (:A,), want = :total)
            r = rk_jax_evaluate(d, :kernel, (A,); grad = true)
            @test r.value ≈ native(A) rtol = 1e-13
            # The Python driver returns a matrix gradient as a vector of rows.
            g = rk_fd_gradient(u -> native(reshape(u, size(A))), vec(A))
            @test vec(reduce(hcat, r.grad)') ≈ g rtol = 1e-6 atol = 1e-8
        end
        rows = [[0.2, -0.1, 0.8], [0.4], [1.5, -0.3, 0.6, 0.9, 0.2]]
        ragged = jax_source(jax_stat_ragged; have = (:theta, :xs), want = :total)
        native = prepare(jax_stat_ragged; have = (:theta, :xs), want = :total)
        r = rk_jax_evaluate(ragged, :kernel, (0.8, rows); grad = true)
        @test r.value ≈ native(0.8, rows) rtol = 1e-13
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), rows), [0.8])) rtol = 1e-6
        for n in (3, 40)
            x = collect(range(0.1, 2.0; length = n))
            src = jax_source(jax_stat_host; have = (:theta, :x), want = :total, data = (; x))
            native = prepare(jax_stat_host; have = (:theta, :x), want = :total)
            r = rk_jax_evaluate(src, :kernel, (0.6, x); grad = true, prepare = :arguments)
            @test r.value ≈ native(0.6, x) rtol = 1e-13
            @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), x), [0.6])) rtol = 1e-6
        end
        edge = jax_source(jax_stat_edge; have = (:x,), want = :values)
        native = prepare(jax_stat_edge; have = (:x,), want = :values)
        for x in (Float64[], [0.4])
            got = rk_jax_evaluate(edge, :kernel, (x,)).value
            want = native(x)
            @test isequal(isnan.(got), isnan.(want))
            @test got[.!isnan.(want)] ≈ want[.!isnan.(want)]
        end
    end
end
