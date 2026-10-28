module ScanTerminalMinimaTests
using ReactiveKernels, Test

include("type_stability_audit.jl")
using .TypeStabilityAudit: findings, rk_or_generated

allocated(f::F, args::Vararg{Any,N}) where {F,N} = @allocated f(args...)

@kernel terminal_scan_minima(xs::Vector{Float64}, seed::Float64) = begin
    trajectory::Vector{Float64} = scan(xs; init = seed,
                                         include_init = true) do carry, x
        next = carry + x
        (next, next)
    end
    relative = @.(100 * (trajectory - trajectory[1]) / trajectory[1])
    low::Float64 = minimum(trajectory)
    relative_low::Float64 = minimum(relative)
    return low, relative_low
end

@kernel terminal_scan_minima_with_trajectory(xs::Vector{Float64}, seed::Float64) = begin
    trajectory::Vector{Float64} = scan(xs; init = seed,
                                         include_init = true) do carry, x
        next = carry + x
        (next, next)
    end
    relative = @.(100 * (trajectory - trajectory[1]) / trajectory[1])
    low::Float64 = minimum(trajectory)
    relative_low::Float64 = minimum(relative)
    return trajectory, low, relative_low
end

@kernel terminal_scan_minima_with_other_reader(xs::Vector{Float64}, seed::Float64) = begin
    trajectory::Vector{Float64} = scan(xs; init = seed,
                                         include_init = true) do carry, x
        next = carry + x
        (next, next)
    end
    low::Float64 = minimum(trajectory)
    total::Float64 = sum(trajectory)
    return low, total
end

@testset "init-including scan terminal minimum" begin
    fused = prepare(terminal_scan_minima)
    materialized = prepare(terminal_scan_minima_with_trajectory)
    expr = string(code_expr(fused))
    @test occursin("scan_minimum", expr)
    @test !occursin("_scan_terminal_samples", expr)
    @test occursin("scan_preallocated", string(code_expr(materialized)))
    @test occursin("scan_preallocated", string(code_expr(
        prepare(terminal_scan_minima_with_other_reader))))
    for (xs, seed) in ((Float64[], 2.0), ([1.0], 2.0),
                       ([1.0, -3.0, 4.0], 2.0),
                       (fill(-0.0, 5), -0.0),
                       ([NaN, 1.0], 2.0),
                       ([1.0, 2.0], NaN),
                       ([ones(1024); NaN], 2.0),
                       ([Inf, -Inf], 2.0),
                       ([1.0, -2.0], 0.0),
                       (collect(range(-0.1, 0.1; length = 129)), 2.0))
        trajectory, expected_low, expected_relative_low = materialized(xs, seed)
        got_low, got_relative_low = fused(xs, seed)
        @test isequal(got_low, expected_low)
        @test isequal(got_relative_low, expected_relative_low)
        @test isequal(trajectory, [seed; accumulate(+, xs; init = seed)])
    end
    # The same scalar residual is used by position batching, including its
    # borrowed-output mode. Neither input nor a returned value is mutated.
    batch = vectorize(terminal_scan_minima; batched = :seed, reuse = true)
    xs = [0.2, -0.5, 0.3]
    seeds = [1.0, 2.0, 4.0]
    saved = copy(xs), copy(seeds)
    got = batch(xs, seeds)
    expected = [fused(xs, seed) for seed in seeds]
    @test all(i -> isequal(got[1][i], expected[i][1]) &&
                   isequal(got[2][i], expected[i][2]), eachindex(seeds))
    @test (xs, seeds) == saved
    native = ReactiveKernels._lower_with_ops(fused.plan)
    tensorized = ReactiveKernels._lower_with_ops(fused.plan; tensorized=true)
    @test native[2] == tensorized[2]
    @test native[3] == tensorized[3]
    @test compile(tensorized[1])(tensorized[2], xs, 2.0) == fused(xs, 2.0)
    for n in (0, 17, 1024)
        xs = fill(-0.001, n)
        fused(xs, 2.0)
        # The specialized call has no trajectory or proxy allocation, at any n.
        @test allocated(fused, xs, 2.0) == 0
    end
end

@kernel typed_minima(xs, seed::Float64) = begin
    trajectory::AbstractVector{Float32} = scan(xs; init=seed, include_init=true) do carry, x
        next = carry + x
        (next, next)
    end
    relative::Vector{Float16} = 100 .* (trajectory .- trajectory[1]) ./ trajectory[1]
    low::Float64 = minimum(trajectory)
    relative_low::Float64 = minimum(relative)
    return low, relative_low
end

@kernel composed_minima(values::Vector{Float64}, initial::Float64) = begin
    (a, b) = terminal_scan_minima(values, initial)
    (c, d) = terminal_scan_minima(values, -initial)
    return a, b, c, d
end

# A concrete inferred output type is required for streaming. This step must
# keep the materialized first-sample typing path.
opaque(x) = Base.inferencebarrier(x)
@kernel opaque_minimum(xs::Vector{Float64}, seed::Float64) = begin
    trajectory::Vector{Float64} = scan(xs; init=seed, include_init=true) do carry, x
        next = opaque(carry + x)
        (next, next)
    end
    low = minimum(trajectory)
    return low
end

@kernel integer_minimum(xs::Vector{Float64}, seed::Float64) = begin
    trajectory::Vector{Float64} = scan(xs; init=seed, include_init=true) do carry, x
        next = carry + x
        (next, next)
    end
    low::Int = minimum(trajectory)
    return low
end

@testset "terminal minimum conversions, composition and fallback" begin
    fused = prepare(typed_minima)
    materialized = prepare(typed_minima; want=(:trajectory, :low, :relative_low))
    for xs in (Float64[], [1.23456789e-5, -0.123456789, 0.5],
               reshape([0.1, -0.5, 0.3, -0.2], 2, 2),
               view([0.1, -0.5, 0.3], 1:2))
        expected = materialized(xs, 1.00000001)
        @test isequal(fused(xs, 1.00000001), expected[2:3])
        @test eltype(expected[1]) === Float32
    end
    xs = [0.2, -0.7, 0.1]
    original = copy(xs)
    scalar = prepare(terminal_scan_minima)
    composed = prepare(composed_minima)
    @test occursin("scan_minimum", string(code_expr(composed)))
    @test @inferred(composed(xs, 2.0)) == (scalar(xs, 2.0)..., scalar(xs, -2.0)...)
    @test isempty(findings(composed, (Vector{Float64}, Float64);
                           recurse=rk_or_generated(@__MODULE__)))
    bound = prepare(composed_minima; bound=(values=xs,))
    @test bound(2.0) == composed(xs, 2.0)
    @test xs == original
    expr = string(code_expr(scalar))
    scalar(ones(1024), 2.0)
    @test string(code_expr(scalar)) == expr
    # WANTing a transformed vector, or adding a whole-array reader, retains it.
    with_relative = prepare(terminal_scan_minima; want=(:relative, :low, :relative_low))
    @test !occursin("scan_minimum", string(code_expr(with_relative)))
    @test with_relative(xs, 2.0)[2:3] == scalar(xs, 2.0)
    opaque_fused = prepare(opaque_minimum)
    opaque_reference = prepare(opaque_minimum; want=(:trajectory, :low))
    for data in (Float64[], xs)
        @test opaque_fused(data, 2.0) == opaque_reference(data, 2.0)[2]
    end
    integer = prepare(integer_minimum)
    @test integer([-0.5], 1.5) === 1
    @test_throws InexactError integer(Float64[], 1.5)

    # Borrowed batch scratch remains valid across lengths and independent copies.
    batch = vectorize(composed_minima; batched=:initial, reuse=true)
    batch_copy = copy(batch)
    for data in (xs, Float64[], ones(17)), seeds in ([1.0, 2.0], [3.0])
        expected = [composed(data, seed) for seed in seeds]
        for f in (batch, batch_copy)
            got = f(data, seeds)
            @test all(j -> all(i -> isequal(got[j][i], expected[i][j]),
                              eachindex(seeds)), 1:4)
        end
    end
end

end # module
