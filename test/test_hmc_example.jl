using Test

include(joinpath(@__DIR__, "..", "examples", "hmc.jl"))

# The scalar single-chain HMC transition is the one mathematical source of truth RK batches
# (via `replica`, ext/Reactant) into the multi-chain form. This test pins that the scalar
# source is authored in the current `@kernel` syntax, plans/prepares, and samples correctly.
@testset "Scalar HMC transition example" begin
    source = only(HMCExample.all_sources())

    @testset "authored as one scalar @kernel, no data-dependent control flow" begin
        # A single method-free `@kernel` source of truth.
        @test occursin(r"@kernel hmc_transition\(", source)
        # Metropolis accept is a SELECT (ternary), not a branch — the property that makes the
        # scalar source auto-batchable over chains.
        @test occursin("accept ? qL : q", source)
        # Momentum + MH uniform are injected HAVE ports (deterministic transition, no RNG in
        # the kernel) — the backend-neutral seam `replica` and Reactant rely on.
        @test occursin("p0::Vector{Float64}", source)
        @test occursin(r"u::Float64", source)
        # grad_U / pot are ordinary scalar closures (Vector -> Vector / Vector -> scalar),
        # never a matrix batch adapter — `replica` maps them per chain slice.
        @test occursin("grad_U::Function", source)
        @test occursin("pot::Function", source)
        # RK derives the multi-chain form from the SAME scalar source via `replica` — no rewrite.
        @test occursin("replica(hmc_transition; batched = (:q, :p0, :u))", source)
    end

    @testset "prepares and recovers a correlated-Gaussian target single-chain" begin
        artifact = only(map(HMCExample.evaluate_source, HMCExample.all_sources()))
        @test artifact.name === :hmc_transition
        @test size(artifact.samples, 1) == artifact.D
        # The single-chain sampler recovers the target covariance within a loose tolerance.
        @test artifact.cov_relerror < 0.1
        @test artifact.mean_abserror < 0.3
        # A compact straight-line kernel (leapfrog captured as one recipe + the accept select).
        @test artifact.n_recipes ≥ 6
    end

    @testset "replica batches the SAME scalar source into the multi-chain sampler" begin
        artifact = only(map(HMCExample.evaluate_source, HMCExample.all_sources()))
        # One native replica step equals applying the scalar kernel independently per chain.
        @test artifact.replica_matches_scalar
        # Batched output is [params × chains × draws].
        @test size(artifact.batched_samples, 1) == artifact.D
        @test size(artifact.batched_samples, 2) == artifact.n_chains
        @test artifact.n_chains > 1
        # The multi-chain sampler recovers the same target as the single-chain source.
        @test artifact.batched_cov_relerror < 0.1
    end
end
