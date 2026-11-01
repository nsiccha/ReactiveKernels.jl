module GatherColumnShapeTests
using ReactiveKernels, DifferentiationInterface, Enzyme, LinearAlgebra, Test

# Repeated gathers keep their own column indices, but all integer columns
# have the same row axis. Include the packed-view/reshape/product chain so
# a refused gather cannot hide whole computations further upstream.
@kernel gathered_columns(u::AbstractVector{Float64}, n::Int,
                         scale::AbstractVector{Float64}, L::AbstractMatrix{Float64},
                         ids::AbstractVector{Int}, x::AbstractVector{Float64},
                         y::AbstractVector{Float64}, key::AbstractVector{Int}) = begin
    stop = 4n + 1
    flat = view(u, 2:stop)
    Z = reshape(flat, n, 4)
    B = Z * transpose(scale .* L)
    mu = B[ids, 1] .* x .+ B[ids, 2] .* x.^2 .+
         B[ids, 3] .* sin.(x) .+ B[ids, 4] .* cos.(x)
    dens = plate(mu, y) do m, yi
        -(m - yi)^2 / 2
    end
    return sum(dens)
end

data(n) = (; n, scale = [0.7, 1.1, 0.9, 0.5],
    L = [1.0 0.0 0.0 0.0; 0.2 1.0 0.0 0.0; -0.1 0.3 1.0 0.0; 0.4 0.0 -0.2 1.0],
    ids = [2, 1, 2, n, 1, n], x = [-0.3, 0.2, 0.8, -0.6, 0.5, 0.4],
    y = [0.1, -0.2, 0.3, 0.2, 0.7, -0.1], key = [1, 1, 1, 2, 2, 2])
function reference(u, d, rows)
    Z = reshape(view(u, 2:(4d.n + 1)), d.n, 4)
    F = d.scale .* d.L
    B = Z * transpose(F)
    X = hcat(d.x, d.x.^2, sin.(d.x), cos.(d.x))
    residual = [dot(B[d.ids[i], :], X[i, :]) - d.y[i] for i in rows]
    grad = zero(u)
    for (j, i) in enumerate(rows), k in 1:4
        grad[1 + d.ids[i] + d.n*(k - 1)] -= residual[j] * dot(F[:, k], X[i, :])
    end
    -sum(abs2, residual)/2, grad
end
astsize(x::Expr) = 1 + sum(astsize, x.args; init = 0)
astsize(x) = 0

@kernel gathered_slices(Q::Array{Float64,3}, ids::Vector{Int}, other::Vector{Int},
                        y::Vector{Float64}) = begin
    B = exp.(Q)
    mu = vec(sum(B[ids, 1, :] .+ B[other, 2, :] .+ B[ids, :, 1]; dims = 2))
    dens = plate(mu, y) do m, yi
        -(m - yi)^2 / 2
    end
    return sum(dens)
end

@testset "different literal columns of one gathered source remain local" begin
    have = (:u, :n, :scale, :L, :ids, :x, :y, :key)
    spec = plate_cell(gathered_columns, :dens; index = :i, locality = :error)
    unbound = prepare(spec; have = (have..., :i), want = :dens_cell)
    allocations, sizes = Int[], Int[]
    for n in (4, 400)
        d = data(n)
        u = sin.(1:(4n + 2)) ./ 3
        original = deepcopy((u, d))
        cell = prepare(spec; have = (have..., :i), want = :dens_cell, bound = d)
        part = prepare(plate_partition(gathered_columns, :dens; by = :key,
            locality = :error); have = (have..., :partition),
            want = :partition_total, bound = d)
        for k in (cell, part)
            report = partition_locality(k)
            @test all(e.status !== :whole for e in report)
            for name in (:flat, :Z, :B, :mu)
                @test only(e.status for e in report if e.value === name) === :rows
            end
        end
        for v in (u, -u), i in (1, 4)
            expected, grad = reference(v, d, [i])
            @test cell(v, i) ≈ expected
            @test unbound(v, values(d)..., i) ≈ expected
            ad = prepare_ad(cell, AutoEnzyme(; mode = Enzyme.Reverse), v, i; active = :u)
            value, gradient = ad_value_and_gradient(ad, v, i)
            @test value ≈ expected
            @test gradient ≈ grad
        end
        for label in (1, 2)
            rows = findall(==(label), d.key)
            expected, grad = reference(u, d, rows)
            @test part(u, label) ≈ expected
            ad = prepare_ad(part, AutoEnzyme(; mode = Enzyme.Reverse), u, label; active = :u)
            value, gradient = ad_value_and_gradient(ad, u, label)
            @test value ≈ expected
            @test gradient ≈ grad
        end
        @test partition_footprint(cell, u, 1).u == [1 + d.ids[1] + n*j for j in 0:3]
        cell(u, 1)
        push!(allocations, @allocated cell(u, 1))
        push!(sizes, astsize(code_expr(cell)))
        @test (u, d) == original
    end
    @test allocations[1] == allocations[2]
    @test sizes[1] == sizes[2]
    @info "gathered-column cell growth" groups=(4, 400) allocations sizes
end

@testset "each gathered slice contributes its surviving axes" begin
    k = prepare(plate_cell(gathered_slices, :dens; index = :i, locality = :error);
        have = (:Q, :ids, :other, :y, :i), want = :dens_cell)
    Q = reshape(sin.(1:36) ./ 4, 4, 3, 3)
    ids, other, y = [3, 1, 3, 2], [1, 2, 3, 1], [0.2, -0.1, 0.4, 0.7]
    for i in (1, 2)
        g = ids[i]
        B = exp.(Q)
        residual = sum(B[g, 1, :] .+ B[other[i], 2, :] .+ B[g, :, 1]) - y[i]
        expected = -residual^2 / 2
        @test k(Q, ids, other, y, i) ≈ expected
        grad = zero(Q)
        for j in 1:3
            grad[g, 1, j] -= residual * B[g, 1, j]
            grad[other[i], 2, j] -= residual * B[other[i], 2, j]
            grad[g, j, 1] -= residual * B[g, j, 1]
        end
        ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), Q, ids, other, y, i; active = :Q)
        value, gradient = ad_value_and_gradient(ad, Q, ids, other, y, i)
        @test value ≈ expected
        @test gradient ≈ grad
    end
    @test_throws BoundsError k(Q, [5, 1, 3, 2], other, y, 1)
    @test_throws BoundsError k(Q[:, 1:1, :], ids, other, y, 1)
    @test_throws DimensionMismatch k(Q[:, 1:2, :], ids, other, y, 1)
end
end
