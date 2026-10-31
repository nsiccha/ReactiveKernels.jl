module ProductCellShapeTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# A vector-valued product's row axis is known even when its right factor
# is an expression. The shared coefficients stay whole; the basis and
# predictor run only at the requested rows.
@kernel inline_product(X::AbstractMatrix{Float64}, w::AbstractVector{Float64},
                       z::AbstractVector{Float64}, y::AbstractVector{Float64},
                       key::AbstractVector{Int}, scale::Float64) = begin
    basis = exp.(scale .* X)
    weights = plate(w) do wi
        2 * wi
    end
    predictor = basis * (weights .* z)
    mu = predictor .+ 0.3
    dens = plate(mu, y) do m, yi
        -(m - yi)^2 / 2
    end
    total = sum(dens)
    return total
end

@kernel chained_product(X::AbstractMatrix{Float64}, W::AbstractMatrix{Float64},
                        z::AbstractVector{Float64}, y::AbstractVector{Float64},
                        key::AbstractVector{Int}) = begin
    predictor = X * W * z
    dens = plate(predictor, y) do m, yi
        -(m - yi)^2 / 2
    end
    total = sum(dens)
    return total
end

data(n) = (; X = [sin(i + j) / 3 for i in 1:n, j in 1:3],
    w = [0.2, 0.5, 0.7], y = cos.(1:n), key = [1, 1, fill(2, n - 2)...])
astsize(x::Expr) = 1 + sum(astsize, x.args; init = 0)
astsize(x) = 0

@testset "vector product shapes in cells and partitions" begin
    have = (:X, :w, :z, :y, :key, :scale)
    cell = prepare(plate_cell(inline_product, :dens; index = :i, locality = :error);
        have = (have..., :i), want = :dens_cell)
    spec = plate_partition(inline_product, :dens; by = :key)
    unbound = prepare(spec; have = (have..., :partition), want = :partition_total)
    allocations, sizes = Int[], Int[]
    z = [0.1, -0.2, 0.4]
    for n in (6, 6000)
        d = data(n)
        k = prepare(spec; have = (have..., :partition), want = :partition_total, bound = d)
        report = partition_locality(k)
        @test all(e.status !== :whole for e in report)
        for name in (:basis, :predictor, :mu)
            @test only(e.status for e in report if e.value === name) === :rows
        end
        for scale in (0.4, -0.7), v in (z, -z)
            B = exp.(scale .* d.X)
            residual = B * (2 .* d.w .* v) .+ 0.3 .- d.y
            expected = -sum(abs2, residual[1:2]) / 2
            @test k(v, scale, 1) ≈ expected
            @test unbound(d.X, d.w, v, d.y, d.key, scale, 1) ≈ expected
            @test sum(cell(d.X, d.w, v, d.y, d.key, scale, i) for i in 1:2) ≈ expected
        end
        k(z, 0.4, 1)
        push!(allocations, @allocated k(z, 0.4, 1))
        push!(sizes, astsize(code_expr(k)))
        # Ordinary native Reverse, with an independent analytic gradient.
        ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), z, 0.4, 1; active = :z)
        value, gradient = ad_value_and_gradient(ad, z, 0.4, 1)
        B = exp.(0.4 .* d.X[1:2, :])
        residual = B * (2 .* d.w .* z) .+ 0.3 .- d.y[1:2]
        @test value ≈ -sum(abs2, residual) / 2
        @test gradient ≈ -(B' * residual) .* (2 .* d.w)
    end
    @test allocations[1] == allocations[2]
    @test sizes[1] == sizes[2]
    # Differentiate through the shared weight plate as well as the product.
    d = data(6)
    ad = prepare_ad(unbound, AutoEnzyme(; mode = Enzyme.Reverse),
        d.X, d.w, z, d.y, d.key, 0.4, 1; active = :w)
    value, gradient = ad_value_and_gradient(ad, d.X, d.w, z, d.y, d.key, 0.4, 1)
    B = exp.(0.4 .* d.X[1:2, :])
    residual = B * (2 .* d.w .* z) .+ 0.3 .- d.y[1:2]
    @test value ≈ -sum(abs2, residual) / 2
    @test gradient ≈ -(B' * residual) .* (2 .* z)
    # Julia's dimension checks still apply to the whole product.
    @test_throws DimensionMismatch unbound(d.X, d.w, [0.1, 0.2], d.y, d.key, 0.4, 1)

    # The same proof applies to a multi-factor product, with no single
    # right-operand port and no need to evaluate it just to obtain axes.
    W = [1.0 0.2; -0.3 0.8; 0.7 -0.1]
    c = [0.2, -0.4]
    chain = prepare(plate_partition(chained_product, :dens; by = :key);
        have = (:X, :W, :z, :y, :key, :partition), want = :partition_total)
    expected = -sum(abs2, (d.X * W * c - d.y)[1:2]) / 2
    @test chain(d.X, W, c, d.y, d.key, 1) ≈ expected
    @test only(e.status for e in partition_locality(chain) if e.value === :predictor) === :rows
end
end
