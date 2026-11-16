using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# A loop to a ragged row's length, `for t in 1:length(r)` in a plate cell,
# takes lazy steps to the padded width: a step past the row's end runs no
# body (CONTRIBUTING.md, "Lazy branches") and leaves every outer local as
# Julia's loop leaves it. An array of the row's length then repeats its last
# valid entry in its padding, as every padded value does.

# A write at a fixed position (the reported case: native 21, before 35).
@kernel jax_padded_loop_fixed(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        s = begin
            y = zeros(2)
            for t in 1:length(r)
                y[1] = y[1] + r[t]
            end
            sum(y)
        end
        s
    end
    total = sum(cells)
    return total
end
# A write at the loop index into an array that is not the row's length.
@kernel jax_padded_loop_unpadded(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        s = begin
            y = zeros(3)
            for t in 1:length(r)
                y[t] = r[t]
            end
            sum(y)
        end
        s
    end
    total = sum(cells)
    return total
end
# An array the loop rebinds.
@kernel jax_padded_loop_rebound(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        s = begin
            y = zeros(2)
            for t in 1:length(r)
                y = y .+ r[t]
            end
            sum(y)
        end
        s
    end
    total = sum(cells)
    return total
end
# Writes in a static loop nested in the padded one.
@kernel jax_padded_loop_nested(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        s = begin
            y = zeros(2)
            for t in 1:length(r)
                for j in 1:2
                    y[j] = y[j] + j * r[t]
                end
            end
            sum(y)
        end
        s
    end
    total = sum(cells)
    return total
end
# A tuple carrying an array.
@kernel jax_padded_loop_tuple(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        s = begin
            p = (zeros(2), 0.0)
            for t in 1:length(r)
                p = (p[1] .+ r[t], p[2] + r[t])
            end
            sum(p[1]) + p[2]
        end
        s
    end
    total = sum(cells)
    return total
end
# A write at the loop index into an array of the row's length: its padding
# repeats its last valid entry, so `log` over it is finite and the masked
# reduction's gradient has no NaN.
@kernel jax_padded_loop_rowwise(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        s = begin
            out = zeros(length(r))
            for t in 1:length(r)
                out[t] = exp(r[t])
            end
            sum(log, out)
        end
        s
    end
    total = sum(cells)
    return total
end

# A body that is not finite at a step past the row's end, which Julia never
# runs: `log(0)` and below (todo 19usoc9: eager steps gave `jax.grad` NaN).
@traceable function jax_padded_loop_row_terms(r, a)
    total = zero(a)
    for t in 1:length(r)
        total += a * log(length(r) - t + 1) + r[t]
    end
    return total
end
@kernel jax_padded_loop_past_end(x::Vector{Float64}, R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        jax_padded_loop_row_terms(r, x[1])
    end
    total = sum(cells)
    return total
end

const JAX_PADDED_LOOP_KERNELS = (jax_padded_loop_fixed, jax_padded_loop_unpadded,
                                 jax_padded_loop_rebound, jax_padded_loop_nested,
                                 jax_padded_loop_tuple, jax_padded_loop_rowwise)

@testset "jax_source: loops to a padded row's length keep outer locals past its end" begin
    sources = Dict(k => string(jax_source(k; have = (:R,), want = :total))
                   for k in JAX_PADDED_LOOP_KERNELS)
    # Each step is a lazy branch whose other arm keeps the whole carry; no
    # write or carry is selected element by element past the row's end.
    for k in JAX_PADDED_LOOP_KERNELS
        @test occursin("jax.lax.cond(kept, step_body, lambda c: c, carry)", sources[k])
        @test !occursin("jnp.where(t <= ", sources[k])
    end
    @test occursin(r"y_\d+ = y_\d+\.at\[0\]\.set\(y_\d+\[0\] \+ r\[t - 1\]\)",
                   sources[jax_padded_loop_fixed])
    # An array of the row's length: written at the loop index, then its
    # padding repeats its last valid entry.
    rowwise = sources[jax_padded_loop_rowwise]
    @test occursin(r"out_\d+ = out_\d+\.at\[t - 1\]\.set\(jnp\.exp\(", rowwise)
    @test occursin("jax.lax.dynamic_index_in_dim(", rowwise)
    # An array of another length keeps its entries as they are.
    @test !occursin("dynamic_index_in_dim(", sources[jax_padded_loop_unpadded])

    if !rk_jax_available()
        @test_skip false
    else
        for R in ([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]], [[0.5, 1.5], Float64[], [2.5]])
            lengths = length.(R)
            stops = cumsum(lengths)
            rows(u) = [u[(stop - n + 1):stop] for (n, stop) in zip(lengths, stops)]
            for k in JAX_PADDED_LOOP_KERNELS
                # Julia's `sum(log, out)` raises over an empty row.
                k === jax_padded_loop_rowwise && any(isempty, R) && continue
                f = prepare(k; have = (:R,), want = :total)
                r = rk_jax_evaluate(jax_source(k; have = (:R,), want = :total), "kernel", (R,);
                                    grad = true)
                @test r.value ≈ f(R) rtol = 1e-12
                # The gradient's valid entries, row by row (padding has none).
                g = reduce(vcat, [row[1:n] for (row, n) in zip(r.grad, lengths)])
                @test all(isfinite, reduce(vcat, r.grad))
                @test g ≈ rk_fd_gradient(u -> f(rows(u)), reduce(vcat, R)) rtol = 1e-6 atol = 1e-6
            end
        end
    end
end

@testset "jax_source: a loop to a padded row's length runs no body past the row's end" begin
    src = jax_source(jax_padded_loop_past_end; have = (:x, :R), want = :total)
    @test occursin("jax.lax.cond(kept, step_body, lambda c: c, carry)", string(src))
    if !rk_jax_available()
        @test_skip false
    else
        f = prepare(jax_padded_loop_past_end; have = (:x, :R), want = :total)
        for R in ([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]], [[0.5, 1.5], Float64[], [2.5]])
            x = [0.7]
            r = rk_jax_evaluate(src, "kernel", (x, R); grad = true)
            @test r.value ≈ f(x, R) rtol = 1e-12
            # Eager steps past the end gave NaN here (log of zero and below).
            @test all(isfinite, r.grad)
            @test r.grad ≈ rk_fd_gradient(u -> f(u, R), x) rtol = 1e-6 atol = 1e-6
        end
    end
end
