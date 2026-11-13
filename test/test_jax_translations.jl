using ReactiveKernels
using Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))
include(joinpath(@__DIR__, "..", "examples", "jax_translations.jl"))

jax_mapped_plain(x) = JAXTranslationExample.triple(x)
@traceable jax_mapped_recorded(x) = JAXTranslationExample.triple(x)
@kernel jax_mapped_wrappers(x) = begin
    plain = jax_mapped_plain(x)
    recorded = jax_mapped_recorded(x)
    total = sum(plain) + sum(recorded)
    return total
end
@kernel jax_mapped_broadcast(x::Matrix{Float64}) = begin
    y = JAXTranslationExample.triple.(x)
    total = sum(y)
    return total
end
@kernel jax_mapped_ragged(xs::Vector{Vector{Float64}}) = begin
    ys = plate(xs) do x
        sum(JAXTranslationExample.triple(x))
    end
    total = sum(ys)
    return total
end

@testset "jax_source: documented package translations" begin
    ex = JAXTranslationExample
    # An ordinary array call uses the whole-call hook; a broadcast uses the
    # scalar mapping. The example owns both, without a source provider.
    scalar = jax_source(ex.mapped; have = (:x,), want = :total, types = (x = Float64,))
    matrix = jax_source(ex.mapped; have = (:x,), want = :total, types = (x = Matrix{Float64},))
    broadcasted = jax_source(jax_mapped_broadcast; have = (:x,), want = :total)
    @test occursin("def _example_triple(", string(scalar))
    @test occursin("def _example_triple(", string(broadcasted))
    @test !occursin("def _example_triple(", string(matrix))
    product = jax_source(ex.mapped_product; have = (:A, :v), want = :total)
    @test occursin(" @ ", string(product))

    A = [0.0 0.2; -0.3 0.1]
    host = jax_source(ex.exponential; have = (:scale, :A), want = :total, data = (; A))
    device = jax_source(ex.exponential; have = (:scale, :A), want = :total)
    @test occursin("scipy.linalg.expm(", string(host))
    @test occursin("jax.scipy.linalg.expm(", string(device))
    @test !occursin("jax.scipy.linalg.expm(", string(host))
    @test string(host) == string(jax_source(ex.exponential; have = (:scale, :A),
        want = :total, data = (A = zeros(4, 4),)))
    if !rk_jax_available()
        @test_skip false
    else
        for x in (0.7, [0.2, -0.4, 0.1], A, zeros(0, 2), ones(3, 4))
            for kernel in (ex.mapped, jax_mapped_wrappers)
                src = jax_source(kernel; have = (:x,), want = :total, types = (x = typeof(x),))
                native = prepare(kernel; have = (:x,), want = :total)
                r = rk_jax_evaluate(src, :kernel, (x,); grad = true)
                @test r.value ≈ native(x) atol = 1e-12
                factor = kernel === ex.mapped ? 3.0 : 6.0
                # The Python driver returns a matrix as a vector of rows.
                expected = x isa Number ? factor : x isa AbstractMatrix ?
                    [fill(factor, size(x, 2)) for _ in axes(x, 1)] : fill(factor, size(x))
                @test r.grad == expected
            end
            prep = jax_source(ex.prepared; have = (:scale, :x), want = :total, data = (; x))
            r = rk_jax_evaluate(prep, :kernel, (0.7, x); grad = true, prepare = :arguments)
            @test r.value ≈ 0.7 * sum(ex.triple(x)) atol = 1e-12
            @test r.grad ≈ sum(ex.triple(x)) atol = 1e-12
        end
        @test rk_jax_evaluate(broadcasted, :kernel, (A,)).value ≈ sum(ex.triple.(A)) atol = 1e-12
        rect = [0.2 -0.4; 0.1 0.7; -0.5 0.3]
        v = [0.8, -0.2]
        @test rk_jax_evaluate(product, :kernel, (rect, v)).value ≈ sum(ex.triple(rect) * v)
        rows = [[0.2, -0.1, 0.8], Float64[], [0.4]]
        ragged = jax_source(jax_mapped_ragged; have = (:xs,), want = :total)
        @test rk_jax_evaluate(ragged, :kernel, (rows,)).value ≈ sum(sum(ex.triple(x)) for x in rows)
        native = prepare(ex.exponential; have = (:scale, :A), want = :total)
        matrix_gradient = jax_source(ex.exponential; have = (:A, :scale), want = :total)
        mktempdir() do dir
            path = joinpath(dir, "matrix_model.py")
            write(path, host)
            consumer = joinpath(@__DIR__, "..", "examples", "jax_translations.py")
            @test success(`$RK_JAX_PYTHON $consumer $path`)
        end
        for M in (A, [0.2;;], [0.1 0.2 0.0; 0.0 -0.1 0.3; -0.2 0.0 0.1])
            for (src, prep) in ((device, false), (host, :arguments))
                r = rk_jax_evaluate(src, :kernel, (0.7, M); grad = true, prepare = prep)
                @test r.value ≈ native(0.7, M) rtol = 1e-12
                @test r.grad ≈ sum(exp(M)) rtol = 1e-12
            end
            r = rk_jax_evaluate(matrix_gradient, :kernel, (M, 0.7); grad = true)
            g = reshape(rk_fd_gradient(u -> native(0.7, reshape(u, size(M))), vec(M)), size(M))
            @test all(isapprox(a, b; rtol = 1e-6) for (a, b) in zip(r.grad, eachrow(g)))
        end
    end
end


# A plate over a matrix's slices: `eachrow(M)` maps M's rows and `eachcol(M)`
# its transpose's, so each cell reads one slice as natively.
@kernel jax_slice_plates(M::Matrix{Float64}, w::Vector{Float64}) = begin
    rows = plate(eachrow(M), w) do row, weight
        weight * sum(row .* row)
    end
    cols = plate(eachcol(M)) do col
        sum(col) * 2.0
    end
    total = sum(rows) + sum(cols)
    return total
end

@testset "jax_source: plates over eachrow and eachcol" begin
    src = jax_source(jax_slice_plates; have = (:M, :w), want = :total)
    @test occursin("jnp.transpose(", string(src))
    native = prepare(jax_slice_plates; have = (:M, :w), want = :total)
    if !rk_jax_available()
        @test_skip false
    else
        for M in ([0.2 -0.4 0.1; 0.3 0.5 -0.2; 1.0 0.0 0.7], zeros(0, 3))
            w = collect(1.0:size(M, 1))
            @test rk_jax_evaluate(src, :kernel, (M, w)).value ≈ native(M, w) atol = 1e-12
        end
    end
end
