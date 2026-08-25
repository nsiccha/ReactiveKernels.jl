include(joinpath(@__DIR__, "..", "examples", "preexisting_reactiveobjects.jl"))
include(joinpath(@__DIR__, "..", "examples", "preexisting_reactivehmc.jl"))

using .ReactiveObjectsExamples
using .ReactiveHMCExamples
using LinearAlgebra

@testset "preexisting ReactiveObjects.jl examples" begin
    chain = chain_example()
    @test chain.initial == 7.0
    @test chain.updated == 21.0
    @test chain.recipe_count == 2
    @test chain.allocations == 0

    diamond = diamond_example()
    @test diamond.initial == 5.0
    @test diamond.b_only == 11.0
    @test diamond.updated == 23.0
    @test diamond.calls == (b = 1, c = 1, d = 1)
    @test diamond.allocations == 0

    shared = shared_example()
    @test shared.b == 12.0
    @test shared.c == 18.0
    @test shared.a_calls == 1
    @test shared.allocations == 0
end

@testset "preexisting ReactiveHMC.jl examples" begin
    euclidean = euclidean_examples()
    @test euclidean.gaussian_calls ==
        (pot = 1, grad = 2, metric = 0, metric_grad = 0)
    @test euclidean.relativistic_calls ==
        (pot = 1, grad = 4, metric = 0, metric_grad = 0)
    @test all(isfinite, euclidean.euclidean_phasepoint.pos)
    @test all(isfinite, euclidean.euclidean_phasepoint.mom)
    @test isfinite(euclidean.euclidean_phasepoint.ham)
    @test isfinite(euclidean.relativistic_euclidean_phasepoint.ham)

    pos = [0.25, -0.5]
    mom = [0.4, 0.1]
    geometry_euclidean = @inferred euclidean.gaussian.geometry(pos)
    @test @inferred(euclidean.gaussian.dham_dpos(geometry_euclidean, mom)) == pos
    @test @inferred(euclidean.gaussian.dham_dmom(geometry_euclidean, mom)) == mom

    riemannian = riemannian_examples()
    phasepoint = riemannian.riemannian_phasepoint
    metric_diag = 1 .+ abs2.(pos)
    expected_dmom = mom ./ metric_diag
    expected_dpos = pos .+ pos ./ metric_diag .- pos .* expected_dmom .^ 2
    expected_ham = 0.5 * sum(abs2, pos) +
        0.5 * (sum(log, metric_diag) + dot(mom, expected_dmom))

    @test phasepoint.dham_dmom ≈ expected_dmom
    @test phasepoint.dham_dpos ≈ expected_dpos
    @test phasepoint.ham ≈ expected_ham
    @test riemannian.generalized_leapfrog_calls ==
        (pot = 0, grad = 0, metric = 0, metric_grad = 5)
    @test riemannian.implicit_midpoint_calls ==
        (pot = 0, grad = 0, metric = 0, metric_grad = 6)
    @test riemannian.relativistic_calls ==
        (pot = 0, grad = 0, metric = 0, metric_grad = 1)

    geometry = @inferred riemannian.gaussian.geometry(pos)
    @test @inferred(riemannian.gaussian.dham_dmom(geometry, mom)) ≈ expected_dmom
    @test @inferred(riemannian.gaussian.dham_dpos(geometry, mom)) ≈ expected_dpos
    @test @inferred(riemannian.gaussian.hamiltonian(geometry, mom)) ≈ expected_ham

    softabs = softabs_examples()
    @test softabs.generalized_multistep_calls ==
        (pot = 0, grad = 0, metric = 0, metric_grad = 9)
    @test softabs.relativistic_calls ==
        (pot = 0, grad = 0, metric = 0, metric_grad = 1)
    @test isapprox(
        softabs.riemannian_softabs_phasepoint.ham, phasepoint.ham; rtol = 1e-12,
    )
    @test isapprox(
        softabs.relativistic_riemannian_softabs_phasepoint.ham,
        riemannian.relativistic_riemannian_phasepoint.ham;
        rtol = 1e-12,
    )

    softabs_geometry = @inferred softabs.gaussian.geometry(pos)
    @test all(isfinite, @inferred(
        softabs.gaussian.dham_dpos(softabs_geometry, mom),
    ))
    @test all(isfinite, @inferred(
        softabs.gaussian.dham_dmom(softabs_geometry, mom),
    ))
    @test isfinite(@inferred(
        softabs.gaussian.hamiltonian(softabs_geometry, mom),
    ))
    @test all(isfinite, softabs.generalized_multistep.pos)
    @test all(isfinite, softabs.generalized_multistep.mom)
    @test isfinite(softabs.generalized_multistep.ham)
end
