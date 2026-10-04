using Enzyme, LinearAlgebra, Test, Pkg
include("prototype.jl")
using .NativeBDFPrototype

decay(t, y, rate) = [-rate * y[1]]
scalar_loss(u, t0, ts, rate) = sum(interval_bdf(
    decay, u, t0, ts, 2e-6, 3e-6, 4096, rate))

@testset "public decay / all differentiable call coordinates" begin
    u, ts = [1.0], [0.4, 1.6]
    du, dts = zeros(1), zeros(2)
    got = interval_bdf(decay, u, 0.0, ts, 2e-6, 3e-6, 4096, 0.7)
    expected = exp.(-0.7 .* ts)
    @test size(got) == (2, 1)
    @test got[:, 1] ≈ expected rtol = 4e-5
    result = Enzyme.autodiff(Enzyme.Reverse, scalar_loss, Enzyme.Active,
        Enzyme.Duplicated(u, du), Enzyme.Active(0.0),
        Enzyme.Duplicated(ts, dts), Enzyme.Active(0.7))
    @test du[1] ≈ sum(expected) rtol = 4e-5
    @test result[1][2] ≈ 0.7 * sum(expected) rtol = 4e-5
    @test dts ≈ -0.7 .* expected rtol = 4e-5
    @test result[1][4] ≈ -sum(ts .* expected) rtol = 4e-5
    @test u == [1.0] && ts == [0.4, 1.6]
    println("DECAY_COORDINATES ", result, " du=", du, " dts=", dts)
end

# A public stiff linear system, independent of any private application.
# Matrix exponential is an oracle here, never the execution implementation.
linear_rhs(t, y, A, scale, metadata) = scale[1] .* (A * y) .* metadata[1]
function linear_loss(u, A, scale)
    out = interval_bdf(linear_rhs, u, 0.0, [0.03, 0.12],
        1e-9, 1e-10, 10000, A, scale, ([1],))
    sum(out)
end
linear_oracle(u, A, scale) = sum(sum(exp(t * scale[1] * A) * u) for t in (0.03, 0.12))

@testset "matrix and vector arguments / stiff values and Reverse" begin
    u, A, scale = [1.0, 0.4], [-80.0 0.3; 1.1 -0.7], [0.8]
    snapshots = (copy(u), copy(A), copy(scale))
    du, dA, ds = zeros(2), zeros(2, 2), zeros(1)
    @test linear_loss(u, A, scale) ≈ linear_oracle(u, A, scale) rtol = 2e-7
    Enzyme.autodiff(Enzyme.Reverse, linear_loss, Enzyme.Active,
        Enzyme.Duplicated(u, du), Enzyme.Duplicated(A, dA),
        Enzyme.Duplicated(scale, ds))
    expected_du = sum(transpose(exp(t * scale[1] * A)) * ones(2) for t in (0.03, 0.12))
    @test du ≈ expected_du rtol = 2e-6
    h = 1e-5
    for i in eachindex(A)
        plus, minus = copy(A), copy(A)
        plus[i] += h
        minus[i] -= h
        expected = (linear_oracle(u, plus, scale) - linear_oracle(u, minus, scale)) / (2h)
        @test dA[i] ≈ expected rtol = 5e-4 atol = 2e-7
    end
    expected_ds = (linear_oracle(u, A, scale .+ h) - linear_oracle(u, A, scale .- h)) / (2h)
    @test ds[1] ≈ expected_ds rtol = 2e-6
    @test (u, A, scale) == snapshots
    println("STIFF_COORDINATES du=", du, " dA=", dA, " ds=", ds)
end

@testset "output interval budgets and invalid input" begin
    ts = collect(0.001:0.001:0.01)
    counts = step_counts(decay, [1.0], 0.0, ts, 1e-6, 1e-8, 0.7)
    limit = maximum(counts)
    @test sum(counts) > limit
    @test size(interval_bdf(decay, [1.0], 0.0, ts, 1e-6, 1e-8, limit, 0.7)) == (10, 1)
    @test_throws ErrorException interval_bdf(decay, [1.0], 0.0, [0.4], 1e-6, 1e-8, 1, 0.7)
    @test_throws DomainError interval_bdf(decay, [1.0], 0.0, [0.4, 0.3], 1e-6, 1e-8, 4096, 0.7)
    @test_throws DomainError interval_bdf(decay, [1.0], 0.0, [0.4], 0.0, 1e-8, 4096, 0.7)
    println("INTERVAL_STEP_COUNTS ", counts, " per_output_limit=", limit)
end

println("INTERVAL_ACCEPTANCE_PASS")
