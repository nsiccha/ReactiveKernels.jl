using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# Lengths of padded values in a plate cell (snag jax-tuple-valued-1ead4c4c).
# A ragged row's `size` is its length, as its `length` is; and a tuple-valued
# cell's field without a length has none in the next plate, rather than the
# `None` the cell's lengths hold for it.

jax_padded_columns(columns) = reduce(hcat, columns)
# A block's AR(1) path sized by `size(eps, 1)`, as StanBlocks' `@juliacompat`
# bodies bind a `vector[nt]` argument's size.
@traceable function jax_padded_ar_path(eps, phi)
    nt = size(eps, 1)
    d = Array{typeof(phi * eps[1])}(undef, (nt,))
    d[1] = eps[1]
    for t in 2:nt
        d[t] = phi * d[t - 1] + eps[t]
    end
    d
end
@kernel jax_padded_row_means(a::Float64, xs::Vector{Vector{Float64}}) = begin
    means = plate(xs) do x
        a * sum(x) / size(x, 1) + size(x)[1] - length(x)
    end
    total = sum(means)
    return total
end
# The second field is a shared vector: it has no length in the cell.
@kernel jax_padded_unpadded_field(a::Float64, xs::Vector{Vector{Float64}}, base::Vector{Float64},
                                  W::Matrix{Float64}) = begin
    pairs = plate(xs) do x
        (x, (a + sum(x)) .* base)
    end
    second = plate(pairs) do p
        p[2]
    end
    matrix = jax_padded_columns(second)
    total = sum(matrix .* W)
    return total
end
# The second field is derived from the row, so it has the row's length.
@kernel jax_padded_derived_field(phi::Float64, xs::Vector{Vector{Float64}}, W::Matrix{Float64}) = begin
    pairs = plate(xs) do x
        (x, jax_padded_ar_path(x, phi))
    end
    second = plate(pairs) do p
        p[2]
    end
    matrix = jax_padded_columns(second)
    total = sum(matrix .* W)
    return total
end

_jax_padded_attempt(f) = try
    f()
    nothing
catch e
    e
end

@testset "jax_source: padded lengths in plate cells" begin
    ragged = [[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]]
    equal = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    W = [0.5 -1.0; 2.0 0.25; -0.75 1.5]
    base = [0.5, -1.0, 2.0]
    native_means = prepare(jax_padded_row_means)
    native_unpadded = prepare(jax_padded_unpadded_field)
    native_derived = prepare(jax_padded_derived_field)
    @test native_means(0.7, ragged) ≈ 0.7 * (2.0 + 4.0 + 5.5)
    @test_throws DimensionMismatch native_derived(0.6, ragged, W)

    means = jax_source(jax_padded_row_means; have = (:a, :xs), want = :total)
    unpadded = jax_source(jax_padded_unpadded_field; have = (:a, :xs, :base, :W), want = :total)
    derived = jax_source(jax_padded_derived_field; have = (:phi, :xs, :W), want = :total)
    @test occursin("each must fill the padded width", string(derived))

    if !rk_jax_available()
        @test_skip false
    else
        fd(f, x) = (f(x + 1e-6) - f(x - 1e-6)) / 2e-6
        # A row's mean divides by its length, not the padded width.
        r = rk_jax_evaluate(means, :kernel, (0.7, ragged); grad = true)
        @test r.value ≈ native_means(0.7, ragged) rtol = 1e-12
        @test r.grad ≈ fd(a -> native_means(a, ragged), 0.7) rtol = 1e-7

        # Rows of different lengths, one shared column width.
        for xs in (ragged, equal)
            Wx = [0.5 -1.0 0.25; 2.0 0.25 -0.5; -0.75 1.5 1.0][:, 1:length(xs)]
            r = rk_jax_evaluate(unpadded, :kernel, (0.7, xs, base, Wx); grad = true)
            @test r.value ≈ native_unpadded(0.7, xs, base, Wx) rtol = 1e-12
            @test r.grad ≈ fd(a -> native_unpadded(a, xs, base, Wx), 0.7) rtol = 1e-7
        end

        # Columns derived from rows of one length assemble; rows of different
        # lengths are Julia's DimensionMismatch, and the module raises.
        r = rk_jax_evaluate(derived, :kernel, (0.6, equal, W); grad = true)
        @test r.value ≈ native_derived(0.6, equal, W) rtol = 1e-12
        @test r.grad ≈ fd(phi -> native_derived(phi, equal, W), 0.6) rtol = 1e-7
        # The padded width is the longest row's, 3, and so is the column count.
        W3 = [W [1.0, 0.0, -1.0]]
        err = _jax_padded_attempt(() -> rk_jax_evaluate(derived, :kernel, (0.6, ragged, W3)))
        @test err isa ErrorException
        @test occursin("reduce(hcat, columns): the columns differ in length", sprint(showerror, err))

        # No rows: the later plate's empty test reads the tuple-valued plate's
        # first field, whose leading axis is the plate's (a Python tuple has
        # no shape), and `reduce(hcat, ·)` raises as Julia's does.
        @test_throws MethodError native_unpadded(0.7, Vector{Float64}[], base, zeros(3, 0))
        err = _jax_padded_attempt(() -> rk_jax_evaluate(unpadded, :kernel,
                                                        (0.7, Vector{Float64}[], base, zeros(3, 0))))
        @test err isa ErrorException
        @test occursin("reducing over an empty collection", sprint(showerror, err))
    end
end
