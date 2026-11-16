using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# Sizes of plate cells that hold no padding (snag jax-source-size-612e274c).
# A plate whose cells return arrays of one fixed shape (an earlier plate's
# matrix per entry) fills every stored entry, so a later plate that maps it
# reads each cell as the dense array it is: `size` of a cell is static, also
# along a dimension a helper receives as an argument, as StanBlocks'
# `@juliacompat` bodies check a typed `matrix[3, 3]` argument. A genuinely
# padded row checked that way has its length along the padded axis.

# A dimension check as StanBlocks emits it: `i` is the helper's argument,
# static while JAX traces, not a literal at the `size` call.
@traceable function jax_full_cells_dim(x, i, expected)
    actual = size(x, i)
    actual == expected || throw(DimensionMismatch("dimension mismatch"))
    nothing
end
@traceable function jax_full_cells_row(A, E, k, dt)
    jax_full_cells_dim(A, 1, 3)
    jax_full_cells_dim(A, 2, 3)
    jax_full_cells_dim(E, 1, 3)
    jax_full_cells_dim(E, 2, 3)
    s = A[1, 1] * E[1, 1] + A[2, 2] * E[2, 2] + A[3, 3] * E[3, 3] + A[2, 1] * E[3, 2]
    (s * k, dt + size(E, 1))
end
# A plain helper between the plate and the traceable one.
function jax_full_cells_plain(A, E, k, dt)
    r = jax_full_cells_row(A, E, k, dt)
    (r[1], r[2])::NTuple{2,Float64}
end
# Matrix cells made in one nested plate, read in a later one.
@kernel jax_full_cells_direct(a::Float64, B::Matrix{Float64}, ts::Vector{Vector{Float64}}) = begin
    totals = plate(ts) do steps
        A = a .* B
        mats = plate(steps) do dt
            A .* dt
        end
        vals = plate(steps, mats) do dt, E
            r = jax_full_cells_row(A, E, a, dt)
            r[1] + r[2]
        end
        sum(vals)
    end
    total = sum(totals)
    return total
end
# The same, read in a plate inside a plate over parameters, through a plain
# helper: the matrix cells cross the middle plate as a shared argument.
@kernel jax_full_cells_shared(a::Float64, ks::Vector{Float64}, B::Matrix{Float64},
                              ts::Vector{Vector{Float64}}) = begin
    totals = plate(ts) do steps
        A = a .* B
        mats = plate(steps) do dt
            A .* dt
        end
        sites = plate(ks) do k
            vals = plate(steps, mats) do dt, E
                r = jax_full_cells_plain(A, E, k, dt)
                r[1] + r[2]
            end
            sum(vals)
        end
        sum(sites)
    end
    total = sum(totals)
    return total
end
# A padded row checked along a helper-argument dimension: its length.
@traceable jax_full_cells_mean(x, n) = (jax_full_cells_dim(x, 1, n); sum(x) / size(x, 1))
@kernel jax_full_cells_rows(a::Float64, xs::Vector{Vector{Float64}}) = begin
    means = plate(xs) do x
        a * jax_full_cells_mean(x, length(x))
    end
    total = sum(means)
    return total
end
# Each row checked against three entries: rows of other lengths throw.
@kernel jax_full_cells_three(a::Float64, xs::Vector{Vector{Float64}}) = begin
    means = plate(xs) do x
        a * jax_full_cells_mean(x, 3)
    end
    total = sum(means)
    return total
end

_jax_full_cells_attempt(f) = try
    f()
    nothing
catch e
    e
end

@testset "jax_source: sizes of cells without padding" begin
    B = [-1.0 0.0 0.0; 1.0 -0.5 0.0; 0.0 0.5 -0.25]
    ts = [[0.5, 1.0, 2.0], [0.25], [1.5, 3.0]]
    ks = [0.3, 0.9]
    ragged = [[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]]
    equal = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    native_direct = prepare(jax_full_cells_direct)
    native_shared = prepare(jax_full_cells_shared)
    native_rows = prepare(jax_full_cells_rows)
    native_three = prepare(jax_full_cells_three)
    @test native_rows(0.7, ragged) ≈ 0.7 * (2.0 + 4.0 + 5.5)
    @test_throws DimensionMismatch native_three(0.7, ragged)

    # Before, each cell of `mats` carried lengths (its stored size), and the
    # check of `E` along `i` raised "`size` of a padded value along a computed
    # dimension".
    direct = jax_source(jax_full_cells_direct; have = (:a, :B, :ts), want = :total)
    shared = jax_source(jax_full_cells_shared; have = (:a, :ks, :B, :ts), want = :total)
    rows = jax_source(jax_full_cells_rows; have = (:a, :xs), want = :total)
    three = jax_source(jax_full_cells_three; have = (:a, :xs), want = :total)
    # The matrix cell reaches the helpers without lengths.
    for src in (direct, shared)
        @test occursin(r"def jax_full_cells_row\(A[^)]*\bE\b", string(src))
        @test !occursin("E_lengths", string(src))
    end

    if !rk_jax_available()
        @test_skip false
    else
        fd(f, x) = (f(x + 1e-6) - f(x - 1e-6)) / 2e-6
        r = rk_jax_evaluate(direct, :kernel, (0.7, B, ts); grad = true)
        @test r.value ≈ native_direct(0.7, B, ts) rtol = 1e-12
        @test r.grad ≈ fd(a -> native_direct(a, B, ts), 0.7) rtol = 1e-7
        r = rk_jax_evaluate(shared, :kernel, (0.7, ks, B, ts); grad = true)
        @test r.value ≈ native_shared(0.7, ks, B, ts) rtol = 1e-12
        @test r.grad ≈ fd(a -> native_shared(a, ks, B, ts), 0.7) rtol = 1e-7

        # A padded row's size along the helper's dimension is its length.
        r = rk_jax_evaluate(rows, :kernel, (0.7, ragged); grad = true)
        @test r.value ≈ native_rows(0.7, ragged) rtol = 1e-12
        @test r.grad ≈ fd(a -> native_rows(a, ragged), 0.7) rtol = 1e-7
        # The check runs: rows of three pass, a shorter row raises as Julia throws.
        r = rk_jax_evaluate(three, :kernel, (0.7, equal))
        @test r.value ≈ native_three(0.7, equal) rtol = 1e-12
        err = _jax_full_cells_attempt(() -> rk_jax_evaluate(three, :kernel, (0.7, ragged)))
        @test err isa ErrorException
        @test occursin("dimension mismatch", sprint(showerror, err))
    end
end
