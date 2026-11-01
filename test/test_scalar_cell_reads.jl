module ScalarCellReadTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel anchored(X::AbstractMatrix{Float64}, beta::AbstractVector{Float64},
                 y::AbstractVector{Float64}, key::AbstractVector{Int}) = begin
    basis = exp.(0.2 .* X)
    a = basis * beta
    weight = exp.(a[1])
    shift = a[2] .- a[1]
    mu = weight .+ shift .+ a
    dens = plate(mu, y) do m, yi
        -(m - yi)^2 / 2
    end
    total = sum(dens)
    return total
end

# The shared predictor has its own row axis. Its literal rows must never be
# replaced by the likelihood's selected rows, or broadcast against that axis.
@kernel separate(X::AbstractMatrix{Float64}, beta::AbstractVector{Float64},
                 eta::AbstractVector{Float64}, y::AbstractVector{Float64},
                 key::AbstractVector{Int}) = begin
    basis = exp.(0.2 .* X)
    shared = basis * beta
    weight = exp(shared[1])
    shift = shared[2] .- shared[1]
    mu = weight .+ shift .+ exp.(eta)
    dens = plate(mu, y) do m, yi
        -(m - yi)^2 / 2
    end
    total = sum(dens)
    return total
end

astsize(x::Expr) = 1 + sum(astsize, x.args; init = 0)
astsize(x) = 0

@kernel indexed(x::AbstractVector{Float64}, scale::Float64,
                key::AbstractVector{Int}) = begin
    a = scale .* x
    anchor = getindex(a, 3)
    dens = plate(x) do xi
        anchor + xi
    end
    total = sum(dens)
    return total
end

@testset "scalar index spelling and bounds" begin
    k = prepare(plate_partition(indexed, :dens; by = :key, locality = :error);
        have = (:x, :scale, :key, :partition), want = :partition_total)
    @test k([1.0, 2.0, 3.0], 0.5, [1, 2, 3], 2) == 3.5
    @test partition_footprint(k, [1.0, 2.0, 3.0], 0.5, [1, 2, 3], 2).x == [2, 3]
    @test_throws BoundsError k([1.0, 2.0], 0.5, [1, 2], 2)
end

@testset "scalar indexed demand through a predictor" begin
    beta = [0.2, -0.3]
    allocations, gradient_allocations, structures = Int[], Int[], Int[]
    for spec in (anchored, separate)
        empty!(allocations); empty!(gradient_allocations); empty!(structures)
        have = spec === anchored ? (:X, :beta, :y, :key, :partition) :
                                   (:X, :beta, :eta, :y, :key, :partition)
        partition = plate_partition(spec, :dens; by = :key, locality = :error)
        unbound = prepare(partition; have, want = :partition_total)
        for n in (8, 8000)
            m = spec === anchored ? n : n - 1
            X = [sin(i + j) / 3 for i in 1:m, j in 1:2]
            y, eta = cos.(1:n), sin.(1:n) ./ 10
            key = [1, 1, 2, 2, fill(3, n - 4)...]
            data = spec === anchored ? (; X, y, key) : (; X, eta, y, key)
            kernel = prepare(partition; have, want = :partition_total, bound = data)
            B = exp.(0.2 .* X)
            a = B * beta
            local_mean = spec === anchored ? a : exp.(eta)
            mu = exp(a[1]) .+ (a[2] - a[1]) .+ local_mean
            for label in (1, 2, 3)
                rows = findall(==(label), key)
                @test kernel(beta, label) ≈ -sum(abs2, mu[rows] - y[rows]) / 2
            end
            args = spec === anchored ? (X, beta, y, key, 2) : (X, beta, eta, y, key, 2)
            @test unbound(args...) ≈ kernel(beta, 2)
            footprint = partition_footprint(unbound, args...)
            rows = spec === anchored ? [1, 2, 3, 4] : [1, 2]
            @test footprint.X == [rows; m .+ rows]
            @test footprint.beta === :whole
            @test footprint.y == [3, 4]
            spec === separate && @test footprint.eta == [3, 4]
            report = partition_locality(kernel)
            @test all(e.status !== :whole for e in report)
            @test all(e.status in (:scalar, :rows) for e in report
                      if e.value in (:weight, :shift))
            @test all(e.status === :rows for e in report if e.value in (:a, :shared))
            kernel(beta, 2)
            push!(allocations, @allocated kernel(beta, 2))
            push!(structures, astsize(code_expr(kernel)))
            ad = prepare_ad(kernel, AutoEnzyme(; mode = Enzyme.Reverse), beta, 2; active = :beta)
            value, gradient = ad_value_and_gradient(ad, beta, 2)
            sensitivity = exp(a[1]) .* B[1, :] .+ B[2, :] .- B[1, :]
            expected = zeros(2)
            for row in (3, 4)
                derivative = spec === anchored ? sensitivity + B[row, :] : sensitivity
                expected .-= (mu[row] - y[row]) .* derivative
            end
            @test value ≈ kernel(beta, 2)
            @test gradient ≈ expected
            ad_value_and_gradient!(ad, gradient, beta, 2)
            push!(gradient_allocations, @allocated ad_value_and_gradient!(ad, gradient, beta, 2))
        end
        @test allocations[1] == allocations[2]
        @test gradient_allocations[2] <= gradient_allocations[1] + 1024
        @test structures[1] == structures[2]
    end
end
end
