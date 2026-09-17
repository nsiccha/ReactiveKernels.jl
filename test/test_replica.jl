using ReactiveKernels
using DifferentiationInterface
using Enzyme
using LinearAlgebra

@kernel replica_transition(
        q::Vector{Float64}, p0::Vector{Float64}, u::Float64,
        stepsize::Float64) = begin
    proposal::Vector{Float64} = q .+ stepsize .* p0
    kinetic::Float64 = 0.5 * dot(p0, p0)
    accept::Bool = log(u) < -0.05 * kinetic
    q_next::Vector{Float64} = accept ? proposal : q
    return (q_next, accept, kinetic)
end

@kernel replica_normal_logpdf(
        x::Vector{Float64}, location::Float64, logscale::Float64) = begin
    standardized = (x .- location) ./ exp(logscale)
    total::Float64 = -0.5 * sum(abs2, standardized) - length(x) * logscale -
                     0.5 * length(x) * log(2π)
    return total
end

@testset "replicated scalar AD" begin
    prepared = prepare_ad(
        replica_normal_logpdf, AutoEnzyme(mode = Enzyme.Reverse),
        collect(0.2:0.1:0.7), 0.3, log(1.2); active = :x, want = :total)
    replicated = replica(prepared; batched = :x)
    positions = reshape(collect(0.0:0.05:1.15), 6, 4)
    values, gradients = replicated(positions, 0.3, log(1.2))

    @test inputs(replicated) == inputs(prepared.kernel)
    @test code_expr(replicated) == code_expr(prepared.kernel)
    @test size(values) == (4,)
    @test size(gradients) == (6, 4)
    for replica_index in axes(positions, 2)
        scalar_value, scalar_gradient = ad_value_and_gradient(
            prepared, positions[:, replica_index], 0.3, log(1.2))
        @test values[replica_index] ≈ scalar_value
        @test gradients[:, replica_index] ≈ scalar_gradient
        @test scalar_gradient ≈
              -(positions[:, replica_index] .- 0.3) ./ 1.2^2
    end

    @test_throws DimensionMismatch replicated(
        positions[:, 1], 0.3, log(1.2))
end

@kernel replica_defaulted(x::Float64, offset::Float64 = 1.0) = begin
    y::Float64 = x + offset
end

@testset "whole-kernel replica transform" begin
    kernel = replica(replica_transition; batched = (:q, :p0, :u))
    q = reshape(collect(1.0:12.0), 3, 4)
    p0 = [0.1 0.5 0.2 0.8;
          0.2 0.4 0.3 0.7;
          0.3 0.3 0.4 0.6]
    u = [0.1, 0.99, 0.2, 0.95]
    stepsize = 0.25

    q_next, accept, kinetic = kernel(q, p0, u, stepsize)
    reference_kinetic = [0.5 * dot(p0[:, i], p0[:, i]) for i in axes(p0, 2)]
    reference_accept = log.(u) .< -0.05 .* reference_kinetic
    reference_q = ifelse.(reshape(reference_accept, 1, :),
                          q .+ stepsize .* p0, q)

    @test q_next == reference_q
    @test accept == reference_accept
    @test kinetic ≈ reference_kinetic
    @test Tuple(value.name for value in inputs(kernel)) ==
          (:q, :p0, :u, :stepsize)
    @test Tuple(value.name for value in outputs(kernel)) ==
          (:q_next, :accept, :kinetic)
    @test code_expr(kernel) == code_expr(prepare(replica_transition))

    @test_throws DimensionMismatch kernel(q[:, 1:3], p0, u, stepsize)
    @test_throws DimensionMismatch kernel(q, p0[:, 1:3], u, stepsize)
    @test_throws ArgumentError replica(replica_transition; batched = ())
    @test_throws ArgumentError replica(replica_transition; batched = (:missing,))

    # Keep the authored named/default signature outside the replica wrapper so
    # the scalar API remains intact as well as its mathematics.
    defaulted = replica(replica_defaulted; batched = :x)
    @test defaulted([2.0, 3.0]) == [3.0, 4.0]
    @test defaulted([2.0, 3.0], 0.5) == [2.5, 3.5]
end
