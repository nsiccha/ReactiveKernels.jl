using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# `repeat` in Julia's axis order: `repeat(A, counts...)` tiles (`outer`),
# `inner` repeats each entry, and counts beyond `A`'s rank add trailing axes.
# Static counts run in the functions.
@kernel jax_repeat_static(theta::Vector{Float64}, m::Matrix{Float64}) = begin
    entries = vcat(vec(repeat(theta, 2)), vec(repeat(theta, 1, 3)), vec(repeat(m, 2, 3)),
                   vec(repeat(theta; inner = 2)), vec(repeat(m; inner = (1, 2), outer = (2, 1, 2))),
                   vec(repeat(theta; inner = (2, 1), outer = (1, 2))), vec(repeat(m, 1, 1, 2)),
                   vec(repeat(m; outer = (2,))))
    return entries
end
# Julia refuses an `inner` list shorter than `A`'s rank (and then an `outer`
# one too), and a negative count: the module raises as Julia throws
# (`Base._RepeatInnerOuter.check`).
@kernel jax_repeat_short_inner(theta::Vector{Float64}, m::Matrix{Float64}) = begin
    total = sum(theta) * sum(repeat(m; inner = (2,)))
    return total
end
# One prior scale per term, repeated once per category: the category count is
# data, so `prepare_data` computes the scales and one module serves every
# count (snag emit-an-independ-8a0b322b, a categorical model's per-term
# coefficient priors).
@kernel jax_repeat_scales(theta::Vector{Float64}, a::Float64, b::Float64, k::Int) = begin
    scales = repeat([a, b], 1, k)
    z = reshape(theta, 2, :) ./ scales
    total = sum(-0.5 .* z .^ 2 .- log.(scales))
    return total
end
@kernel jax_repeat_outer(theta::Vector{Float64}, s::Vector{Float64}, k::Int) = begin
    scales = repeat(s; outer = (1, k))
    total = sum(-0.5 .* (reshape(theta, length(s), :) ./ scales) .^ 2)
    return total
end

_jax_repeat_host(src) = split(split(string(src), "def prepare_data")[2], "\ndef kernel")[1]

@testset "jax_source: repeat" begin
    src = jax_source(jax_repeat_static; have = (:theta, :m), want = :entries)
    @test occursin("jnp.tile(", src.text) && occursin("jnp.repeat(", src.text)
    # Julia pads a count list with trailing axes, not NumPy's leading ones.
    @test occursin("[..., None]", src.text)

    have = (:theta, :a, :b, :k)
    scales = jax_source(jax_repeat_scales; have, want = :total, data = (; a = 1.2, b = 0.7, k = 2))
    host = _jax_repeat_host(scales)
    @test occursin("np.tile(", host)
    @test !occursin("Computed by the functions", host)
    # One module text across category counts: the count is data.
    @test jax_source(jax_repeat_scales; have, want = :total,
                     data = (; a = 1.2, b = 0.7, k = 5)).text == scales.text
    outer = jax_source(jax_repeat_outer; have = (:theta, :s, :k), want = :total,
                       data = (; s = [1.0, 2.0, 3.0], k = 2))
    @test occursin("np.tile(", _jax_repeat_host(outer))

    # Without data the count is a traced argument, and JAX shapes are static:
    # the emitter names the recipe instead of a trace-time JAX error.
    err = try
        jax_source(jax_repeat_scales; have, want = :total)
        nothing
    catch e
        e
    end
    @test err isa JAXEmissionError
    @test occursin("`repeat` with counts that are not static", err.message)

    short = jax_source(jax_repeat_short_inner; have = (:theta, :m), want = :total)

    if !rk_jax_available()
        @test_skip "RK_JAX_PYTHON not set"
    else
        theta, m = [0.5, -1.0, 2.0], [1.0 2.0; 3.0 4.0; 5.0 6.0]
        native = prepare(jax_repeat_static; have = (:theta, :m), want = :entries)
        @test rk_jax_evaluate(src, "kernel", (theta, m)).value == native(theta, m)

        for k in (1, 2, 5)
            a, b = 1.2, 0.7
            src_k = jax_source(jax_repeat_scales; have, want = :total, data = (; a, b, k))
            @test src_k.text == scales.text
            t = collect(range(-1.0, 1.5; length = 2k))
            f = prepare(jax_repeat_scales; have, want = :total)
            r = rk_jax_evaluate(src_k, "kernel", (t, a, b, k); grad = true, prepare = true)
            @test r.value ≈ f(t, a, b, k) rtol = 1e-12
            @test r.grad ≈ rk_fd_gradient(u -> f(u, a, b, k), t) rtol = 1e-6
        end
        # Where Julia throws, the module raises with Julia's message.
        @test_throws "number of inner repetitions (1) cannot be less than number of dimensions" prepare(
            jax_repeat_short_inner; have = (:theta, :m), want = :total)(theta, m)
        @test_throws "number of inner repetitions (1) cannot be less than number of dimensions" rk_jax_evaluate(
            short, "kernel", (theta, m))
        f = prepare(jax_repeat_scales; have, want = :total)
        @test_throws "no outer repetition count may be negative" f(Float64[], 1.2, 0.7, -1)
        bad = jax_source(jax_repeat_scales; have, want = :total, data = (; a = 1.2, b = 0.7, k = -1))
        @test bad.text == scales.text
        @test_throws "no outer repetition count may be negative" rk_jax_evaluate(
            bad, "kernel", (Float64[], 1.2, 0.7, -1); prepare = true)

        s, k = [1.0, 2.0, 3.0], 2
        t = collect(range(-1.0, 1.0; length = 6))
        g = prepare(jax_repeat_outer; have = (:theta, :s, :k), want = :total)
        @test rk_jax_evaluate(outer, "kernel", (t, s, k); prepare = true).value ≈ g(t, s, k) rtol = 1e-12
    end
end
