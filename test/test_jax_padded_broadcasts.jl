using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# Broadcasts of two padded (ragged) rows of one plate cell (snag
# jax-source-sum-o-d4f6aa97). Each row has its own lengths, so the JAX
# translation saw two different extents and recorded no padded axes for a
# matrix built from both: `sum(cell; dims = 1)` raised. Julia's broadcast
# gives the result the rows' common length, or the longer row's when the
# other has length one.

# A Gaussian mixture over a measured feature row with per-entry standard
# errors: a scalar then a ragged row of features (`x`) and of standard errors
# (`s`), compared with every component's centre `m` and scale `k`. The
# matrices are features × components, the features being the rows' own.
@kernel jax_two_row_mixture(m::Vector{Float64}, X::Vector{Vector{Float64}},
                            S::Vector{Vector{Float64}}, x1::Vector{Float64},
                            k::Vector{Float64}, w::Vector{Float64}) = begin
    lp = plate(X, S, x1, Ref(m), Ref(k), Ref(w)) do xr, sr, x0, ms, ks, ws
        x = vcat(x0, xr)
        s = vcat(0.3, sr)
        sd = sqrt.(ks' .^ 2 .+ s .^ 2)
        cell = -0.5 .* ((x .- ms') ./ sd) .^ 2 .- log.(sd)
        score = vec(sum(cell; dims = 1)) .+ ws
        peak = maximum(score)
        rows = vec(maximum(cell; dims = 2))
        flat = vec(cell)
        peak + log(sum(exp.(score .- peak))) + sum(rows) + sum(flat .* flat[end]) +
            sum(cell) + sum(x .* s) + sum(x + s)
    end
    total = sum(lp)
    return total
end
# Rows broadcast where one has length one: Julia repeats its entry.
@kernel jax_two_row_singleton(a::Float64, A::Vector{Vector{Float64}},
                              B::Vector{Vector{Float64}}) = begin
    lp = plate(A, B) do ar, br
        p = (a .* ar) .* br
        sum(p) + sum(abs2, p .- br) + length(p)
    end
    total = sum(lp)
    return total
end
# Two rows padding different axes (`x .* s'`) would need an extent per axis.
@kernel jax_two_row_outer(a::Float64, X::Vector{Vector{Float64}},
                          S::Vector{Vector{Float64}}) = begin
    lp = plate(X, S) do xr, sr
        t = (a .* xr) .* sr'
        sum(sum(t; dims = 1))
    end
    total = sum(lp)
    return total
end

@testset "jax_source: broadcasts of rows padded by different lengths" begin
    have = (:m, :X, :S, :x1, :k, :w)
    src = jax_source(jax_two_row_mixture; have, want = :total)
    text = string(src)
    @test occursin("def _rk_broadcast_length(", text)
    # No runtime check: one would be a host callback in every evaluation.
    @test !occursin("_rk_check(", text)
    singleton = jax_source(jax_two_row_singleton; have = (:a, :A, :B), want = :total)
    # Operands that pad different axes keep the refusal: their result's
    # padded axes are not known, and reading its padding is a wrong value.
    @test_throws ReactiveKernels.JAXEmissionError jax_source(jax_two_row_outer;
        have = (:a, :X, :S), want = :total)
    m = [0.3, -0.8, 1.05]
    k = [1.0, 0.5, 1.3]
    w = log.([0.2, 0.3, 0.5])
    if !rk_jax_available()
        @test_skip false
    else
        native = prepare(jax_two_row_mixture; have, want = :total)
        # Rows of different lengths, one empty row, and one whole width.
        X = [[0.5, 1.0], Float64[], [0.1, 0.2]]
        S = [[0.2, 0.1], Float64[], [0.4, 0.3]]
        x1 = [0.1, -0.3, 0.5]
        r = rk_jax_evaluate(src, "kernel", (m, X, S, x1, k, w); grad = true)
        @test r.value ≈ native(m, X, S, x1, k, w) rtol = 1e-12
        @test r.grad ≈ rk_fd_gradient(u -> native(u, X, S, x1, k, w), m) rtol = 1e-6 atol = 1e-9
        # A row of length one broadcast against a longer one, at the padded
        # width of one and inside a wider padded row.
        native = prepare(jax_two_row_singleton; have = (:a, :A, :B), want = :total)
        for (A, B) in (([[2.0], [1.5]], [[1.0, 2.0, 3.0], [0.5, 0.25]]),
                       ([[2.0], [1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0], [0.5, 0.25, 0.75]]),
                       ([[2.0], [1.5]], [Float64[], [0.5, 0.25]]))
            r = rk_jax_evaluate(singleton, "kernel", (0.7, A, B); grad = true)
            @test r.value ≈ native(0.7, A, B) rtol = 1e-12
            @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), A, B), [0.7])) rtol = 1e-6
        end
    end
end
