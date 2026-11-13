using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# Logical indexing (snag jax-source-prepa-defc21e0): a `Bool` array index
# selects the entries where it is true. `Bool <: Integer`, and the module used
# to shift a mask as a one-based position vector (`x[mask - 1]`), reading the
# first or the last entry for every element.

# Host (`prepare_data`): rows chosen by a broadcast comparison, then a column.
_jax_mask_rows(table) = table[:, 2] .!= 1
_jax_mask_column(table) = table[_jax_mask_rows(table), 4]
@kernel jax_mask_rows(theta::Vector{Float64}, table::Matrix{Float64}) = begin
    amount = _jax_mask_column(table)
    total = sum(theta[1] .* log.(amount))
    return total
end
# A `Vector{Bool}` input through a plain helper.
_jax_mask_observed(values, present) = values[present]
@kernel jax_mask_observed(theta::Vector{Float64}, values::Vector{Float64},
                          present::Vector{Bool}) = begin
    observed = _jax_mask_observed(values, present)
    total = sum(abs2, observed .- theta[1])
    return total
end
# Whole rows beside the mask; a same-shape matrix mask, in column-major order;
# a list built by `push!`, masked.
@kernel jax_mask_shapes(theta::Vector{Float64}, table::Matrix{Float64}, values::Vector{Float64},
                        present::Vector{Bool}) = begin
    rows = table[present, :]
    big = table[table .> 0.5]
    listed = begin
        out = Float64[]
        for v in values
            push!(out, 2v)
        end
        out[present]
    end
    written = begin
        w = zeros(length(values))
        w[present] = values[present] .* 3.0
        w
    end
    total = theta[1] * sum(rows) + theta[2] * big[1] + theta[1]^2 * big[2] +
        theta[2]^2 * listed[end] + length(big) +
        theta[1] * (written[1] + 2 * written[2] + 3 * written[3])
    return total
end

# Device: masks traced while JAX runs, gathered at padded positions.
@kernel jax_mask_device(x::Vector{Float64}, present::Vector{Bool}) = begin
    total = sum(log.(x[present])) + sum(abs2, x[x .> 1.5]) + length(x[present])
    return total
end
# A write through a mask: an all-false mask writes nothing.
@kernel jax_mask_write(x::Vector{Float64}, present::Vector{Bool}) = begin
    total = begin
        y = zeros(length(x))
        y[present] = x[present] .^ 2
        sum(y .* (1:length(x)))
    end
    return total
end
@kernel jax_mask_device_matrix(A::Matrix{Float64}) = begin
    total = sum(abs2, A[A .> 0.0]) + A[A .> 0.0][1]
    return total
end
# A ragged plate: each cell's row masked by its own (padded) mask row.
@kernel jax_mask_plate(theta::Vector{Float64}, ys::Vector{Vector{Float64}},
                       ms::Vector{Vector{Bool}}) = begin
    cells = plate(ys, ms) do y, m
        sum(abs2, y[m] .- theta[1])
    end
    total = sum(cells)
    return total
end
# Linear indexing of a matrix by a colon, a written range and an integer keeps
# its translation beside the masks.
@kernel jax_mask_linear(A::Matrix{Float64}, k::Int) = begin
    total = sum(abs2, A[:]) + sum(A[2:end]) + A[k] * A[end]
    return total
end
@kernel jax_mask_beside(x::Matrix{Float64}, present::Vector{Bool}) = begin
    total = sum(x[present, :])
    return total
end
@kernel jax_mask_scalar(x::Vector{Float64}, b::Bool) = begin
    total = x[b]
    return total
end

@testset "jax_source: Bool masks select entries" begin
    table = [0.0 1.0 0.0 5.0; 1.0 0.0 0.0 100.0; 2.0 1.0 0.0 7.0;
             3.0 2.0 0.0 200.0; 4.0 1.0 0.0 9.0]
    values, present = [1.0, 2.0, 3.0], [true, false, true]
    host = (
        (jax_mask_rows, (:theta, :table), (; table), ([0.5], table)),
        (jax_mask_observed, (:theta, :values, :present), (; values, present),
         ([0.5], values, present)),
        (jax_mask_shapes, (:theta, :table, :values, :present),
         (; table = table[1:3, :], values, present), ([0.5, 0.25], table[1:3, :], values, present)))
    # Julia's column-major order (row-major would give 5.0 second).
    @test table[1:3, :][table[1:3, :] .> 0.5][2] == 2.0
    for (k, have, data, args) in host
        src = string(jax_source(k; have, want = :total, data))
        # The mask is a NumPy boolean mask, never shifted as positions.
        @test occursin("dtype=bool)", src) || occursin("strict=True", src)
        @test !occursin(r"(present|t_\d+) - 1\b", src)
    end
    # One module text whatever the data's size: lengths and counts stay
    # runtime values, and no body is repeated per row.
    @test string(jax_source(jax_mask_rows; have = (:theta, :table), want = :total,
                            data = (; table))) ==
          string(jax_source(jax_mask_rows; have = (:theta, :table), want = :total,
                            data = (; table = vcat(table, table, table))))
    device = jax_source(jax_mask_device; have = (:x, :present), want = :total)
    @test occursin("jnp.nonzero(", string(device))
    @test_throws "beside another kept dimension" jax_source(jax_mask_beside;
        have = (:x, :present), want = :total)
    @test_throws "invalid index" jax_source(jax_mask_scalar; have = (:x, :b), want = :total)
    if !rk_jax_available()
        @test_skip false
    else
        for (k, have, data, args) in host
            f = prepare(k; have, want = :total)
            src = jax_source(k; have, want = :total, data)
            r = rk_jax_evaluate(src, "kernel", args; prepare = true, grad = true)
            @test r.value ≈ f(args...) rtol = 1e-12
            @test r.grad ≈ rk_fd_gradient(u -> f(u, args[2:end]...), args[1]) rtol = 1e-6 atol = 1e-6
        end
        # A mask of another length fails on the host, as Julia's BoundsError.
        src = jax_source(jax_mask_observed; have = (:theta, :values, :present), want = :total,
                         data = (; values, present))
        @test_throws ErrorException rk_jax_evaluate(src, "kernel", ([0.5], values, [true, false]);
                                                    prepare = true)
        f = prepare(jax_mask_device; have = (:x, :present), want = :total)
        x = [0.5, 2.0, 3.0, 1.0]
        write = jax_source(jax_mask_write; have = (:x, :present), want = :total)
        g = prepare(jax_mask_write; have = (:x, :present), want = :total)
        for m in ([true, false, true, true], [false, true, false, false], falses(4), trues(4))
            m = collect(m)
            for (h, s) in ((f, device), (g, write))
                r = rk_jax_evaluate(s, "kernel", (x, m); grad = true)
                @test r.value ≈ h(x, m) rtol = 1e-12
                @test r.grad ≈ rk_fd_gradient(u -> h(u, m), x) rtol = 1e-6 atol = 1e-6
            end
        end
        @test rk_jax_evaluate(device, "kernel", (Float64[], Bool[])).value == f(Float64[], Bool[])
        f = prepare(jax_mask_device_matrix; have = (:A,), want = :total)
        # Column-major order: 4.0 is the first entry selected (row-major: 2.0).
        A = [-1.0 2.0 3.0; 4.0 5.0 -6.0]
        src = jax_source(jax_mask_device_matrix; have = (:A,), want = :total)
        r = rk_jax_evaluate(src, "kernel", (A,); grad = true)
        @test r.value ≈ f(A) rtol = 1e-12
        @test vec(reduce(hcat, r.grad)') ≈ vec(rk_fd_gradient(u -> f(reshape(u, 2, 3)), vec(A))) rtol = 1e-6
        f = prepare(jax_mask_linear; have = (:A, :k), want = :total)
        src = jax_source(jax_mask_linear; have = (:A, :k), want = :total)
        r = rk_jax_evaluate(src, "kernel", (A, 4); grad = true)
        @test r.value ≈ f(A, 4) rtol = 1e-12
        @test vec(reduce(hcat, r.grad)') ≈ vec(rk_fd_gradient(u -> f(reshape(u, 2, 3), 4), vec(A))) rtol = 1e-6
        f = prepare(jax_mask_plate; have = (:theta, :ys, :ms), want = :total)
        ys = [[1.0, 2.0, 3.0], [4.0], [5.0, 6.0], Float64[]]
        ms = [[true, false, true], [false], [true, true], Bool[]]
        src = jax_source(jax_mask_plate; have = (:theta, :ys, :ms), want = :total)
        r = rk_jax_evaluate(src, "kernel", ([0.5], ys, ms); grad = true)
        @test r.value ≈ f([0.5], ys, ms) rtol = 1e-12
        @test r.grad ≈ rk_fd_gradient(u -> f(u, ys, ms), [0.5]) rtol = 1e-6 atol = 1e-6
    end
end
