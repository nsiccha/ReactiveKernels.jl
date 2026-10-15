using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: icar_logdensity
import DifferentiationInterface as DI
import Enzyme

# Independent oracle: -φ'Qφ/2 with Q the graph Laplacian.
function icar_laplacian_reference(phi, node1, node2)
    Q = zeros(length(phi), length(phi))
    for (i, j) in zip(node1, node2)
        Q[i, i] += 1
        Q[j, j] += 1
        Q[i, j] -= 1
        Q[j, i] -= 1
    end
    return -0.5 * phi' * Q * phi
end

@testset "ICAR pairwise-difference law" begin
    law = prepare(icar_logdensity)
    phi = [0.3, -0.1, 0.7, 0.2, -0.4, 1.1]
    node1 = [1, 1, 2, 3, 4, 2]
    node2 = [2, 3, 3, 4, 5, 4]
    @test law(phi, node1, node2) ≈ icar_laplacian_reference(phi, node1, node2) rtol = 1e-14
    # Improper: a constant shift of the field leaves the density unchanged.
    @test law(phi .+ 2.5, node1, node2) ≈ law(phi, node1, node2) rtol = 1e-12
    # Node 6 lies on no edge, so it adds no term; no edges, no density.
    @test law(phi, node1, node2) == law([phi[1:5]; -9.0], node1, node2)
    @test law(phi, Int[], Int[]) == 0.0
    @test_throws DimensionMismatch law(phi, [1, 2], [2])

    # One retained plate over the edges.
    @test count(e -> e.kind === :plate, recipe_inventory(icar_logdensity)) == 1

    # Native Enzyme Reverse: ∂/∂φ_k = -Σ_{edges ∋ k} (φ_k - φ_other).
    backend = DI.AutoEnzyme(; mode = Enzyme.Reverse)
    op = prepare_ad(law, backend, phi, node1, node2; active = :phi)
    gradient = ad_gradient(op, phi, node1, node2)
    expected = zeros(length(phi))
    for (i, j) in zip(node1, node2)
        expected[i] -= phi[i] - phi[j]
        expected[j] += phi[i] - phi[j]
    end
    @test gradient ≈ expected rtol = 1e-12 atol = 1e-14
end
