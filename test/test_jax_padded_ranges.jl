using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# Ranges to a padded row's length less a constant, and `diff`, in a plate
# cell (snag jax-source-drops-72f21166). `n - c` of a length `n` is bounded by
# the padded width less `c`, so `r[2:n]` and `r[1:(n - 1)]` pad to the same
# width and broadcast as Julia's equal lengths do. Past its stop, a range to a
# data length repeats the stop, as padding repeats a row's last entry: the
# differences' padding is the last valid difference, so `log` of them keeps
# finite gradients.

# The reported spelling: a cell-local length and the two shifted ranges.
@kernel jax_range_lag(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        n = length(r)
        d = r[2:n] .- r[1:(n - 1)]
        sum(d .^ 2)
    end
    total = sum(cells)
    return total
end
# A lag of two.
@kernel jax_range_lag2(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        n = length(r)
        d = r[3:n] .- r[1:(n - 2)]
        sum(exp.(d))
    end
    total = sum(cells)
    return total
end
# An array allocated at `n - 1` and a loop to `n - 1` writing it.
@kernel jax_range_alloc(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        n = length(r)
        s = begin
            d = zeros(n - 1)
            for k in 1:(n - 1)
                d[k] = r[k + 1] - r[k]
            end
            sum(d .^ 2)
        end
        s
    end
    total = sum(cells)
    return total
end
# `log` of the differences: finite only where the padding is a valid difference.
@kernel jax_range_log(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        n = length(r)
        sum(log.(r[2:n] .- r[1:(n - 1)]))
    end
    total = sum(cells)
    return total
end
@kernel jax_diff_row(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        sum(diff(r) .^ 2)
    end
    total = sum(cells)
    return total
end
@kernel jax_diff_log(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        sum(log.(diff(r)))
    end
    total = sum(cells)
    return total
end
@kernel jax_diff_matrix(M::Matrix{Float64}) = begin
    total = sum(diff(M; dims = 2) .^ 2) + sum(diff(M; dims = 1)) + sum(diff(vec(M)))
    return total
end

# Kernels, whether Julia accepts an empty row (`zeros(-1)` does not), and
# whether the cell's function is non-finite at a zero difference (`log`).
const JAX_PADDED_RANGE_KERNELS = ((jax_range_lag, true, false), (jax_range_lag2, true, false),
                                  (jax_range_alloc, false, false), (jax_range_log, true, true),
                                  (jax_diff_row, true, false), (jax_diff_log, true, true))

@testset "jax_source: shifted ranges and diff of a padded row" begin
    lag = string(jax_source(jax_range_lag; have = (:R,), want = :total))
    # Both ranges pad to the padded width less one (before, `1:(n - 1)` ran to
    # the padded width), and past their stops repeat it.
    @test occursin(r"jnp\.minimum\(jnp\.arange\(2, \w+\.shape\[0\] \+ 1\), \w+\)", lag)
    @test occursin(r"jnp\.minimum\(jnp\.arange\(1, max\(\w+\.shape\[0\] - 1, 0\) \+ 1\), \w+\)", lag)
    @test occursin("jnp.diff(", string(jax_source(jax_diff_row; have = (:R,), want = :total)))

    if !rk_jax_available()
        @test_skip false
    else
        # Strictly increasing rows of lengths 0 to 4: the empty and one-entry
        # rows have no differences, the longest fills the padded width.
        for R in ([[0.5, 1.0, 3.0], [2.0], [1.0, 2.0, 4.0, 7.0], [0.3, 0.9]],
                  [[0.5, 1.0, 3.0], Float64[], [1.0, 2.5]],
                  [[0.5, 1.0, 3.0], [1.0, 2.0, 4.0, 7.0], [0.3, 0.9]])
            lengths = length.(R)
            stops = cumsum(lengths)
            rows(u) = [u[(stop - n + 1):stop] for (n, stop) in zip(lengths, stops)]
            for (k, empty_ok, nonfinite_at_zero) in JAX_PADDED_RANGE_KERNELS
                any(isempty, R) && !empty_ok && continue
                f = prepare(k; have = (:R,), want = :total)
                r = rk_jax_evaluate(jax_source(k; have = (:R,), want = :total), "kernel", (R,);
                                    grad = true)
                @test r.value ≈ f(R) rtol = 1e-12
                # A row with no difference has no valid entry its padding could
                # repeat: `log` of that padding gives NaN gradients through the
                # masked sum (todo 0g8sr75, the empty-row padding convention).
                if nonfinite_at_zero && any(n -> n <= 1, lengths)
                    @test_broken all(isfinite, reduce(vcat, r.grad))
                    continue
                end
                # Padding takes no gradient, and none is NaN.
                @test all(isfinite, reduce(vcat, r.grad))
                g = reduce(vcat, [row[1:n] for (row, n) in zip(r.grad, lengths)])
                @test g ≈ rk_fd_gradient(u -> f(rows(u)), reduce(vcat, R)) rtol = 1e-6 atol = 1e-6
            end
        end
        M = [0.1 0.5 0.2; 0.4 -0.3 0.8]
        f = prepare(jax_diff_matrix; have = (:M,), want = :total)
        r = rk_jax_evaluate(jax_source(jax_diff_matrix; have = (:M,), want = :total), "kernel", (M,);
                            grad = true)
        @test r.value ≈ f(M) rtol = 1e-12
        # The gradient comes back as a list of the matrix's rows.
        @test vec(permutedims(reduce(hcat, r.grad))) ≈
              rk_fd_gradient(u -> f(reshape(u, size(M))), vec(M)) rtol = 1e-6 atol = 1e-6
    end
end
