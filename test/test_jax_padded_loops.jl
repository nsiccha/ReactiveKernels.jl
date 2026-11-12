using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# A loop to a ragged row's length, `for t in 1:length(r)` in a plate cell,
# runs to the padded width; its steps past the row's end must leave every
# outer local as Julia's loop leaves it. A write at the loop index into an
# array of the row's length writes only its padding, which keeps the padding's
# values; any other write, a rebinding and a tuple are kept as they were.

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
# repeats a valid value, so `log` over it is finite and the masked reduction's
# gradient has no NaN.
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

const JAX_PADDED_LOOP_KERNELS = (jax_padded_loop_fixed, jax_padded_loop_unpadded,
                                 jax_padded_loop_rebound, jax_padded_loop_nested,
                                 jax_padded_loop_tuple, jax_padded_loop_rowwise)

@testset "jax_source: loops to a padded row's length keep outer locals past its end" begin
    sources = Dict(k => string(jax_source(k; have = (:R,), want = :total))
                   for k in JAX_PADDED_LOOP_KERNELS)
    # A write that would change an entry past the row's end keeps it there.
    @test occursin(r"y_\d+\.at\[0\]\.set\(jnp\.where\(t <= ", sources[jax_padded_loop_fixed])
    @test occursin(r"\.at\[t - 1\]\.set\(jnp\.where\(t <= ", sources[jax_padded_loop_unpadded])
    # A write at the loop index into the row's padding is unchanged.
    @test occursin(r"out_\d+ = out_\d+\.at\[t - 1\]\.set\(jnp\.exp\(",
                   sources[jax_padded_loop_rowwise])
    @test !occursin("jnp.where(t <= ", sources[jax_padded_loop_rowwise])

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
