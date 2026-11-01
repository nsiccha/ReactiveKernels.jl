module PartitionPrefixTypesTests
using ReactiveKernels, Test
const RK = ReactiveKernels

prefix_weights(theta) = view(exp.(theta), :)
@kernel _prefix_typed_rows(Z::Matrix{Float64}, theta::Vector{Float64}, y::Vector{Float64}) = begin
    weights = prefix_weights(theta)
    scaled = Z .* transpose(weights)
    mu = vec(sum(scaled; dims=2))
    density = plate(mu, y) do m, obs
        -(obs - m)^2 / 2
    end
end

@testset "shared cut preserves inferred row operand types" begin
    Z = reshape(collect(range(-0.3, 0.4; length=12)), 4, 3)
    theta = [0.1, -0.2, 0.3]
    y = [0.3, -0.1, 0.2, 0.4]
    spec = plate_cell(_prefix_typed_rows, :density; index=:i, locality=:error)
    original = prepare(spec; have=(:Z, :theta, :y, :i), want=:density_cell,
        bound=(; y))
    parts = RK._partition_shared_parts(original.plan, (:Z => collect(1:4),);
        varying=(:i,), preserve_rows=true)
    @test parts !== nothing
    prefix = prepare(parts.prefix)
    boundary = prefix(theta)
    @test boundary isa SubArray
    residual = prepare(parts.residual)
    @test !any(e -> e.status === :whole, partition_locality(residual))
    for shift in (0.0, 0.2), i in (1, 4, 2)
        point = Z .+ shift
        @test residual(point, theta, i, boundary) ≈ original(point, theta, i)
        @test residual(point, theta, i, copy(boundary)) ≈ original(point, theta, i)
        changed = theta .- shift
        @test residual(point, changed, i, prefix(changed)) ≈ original(point, changed, i)
    end
end
end
