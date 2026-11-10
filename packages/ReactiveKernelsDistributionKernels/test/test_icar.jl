using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: icar_logdensity,
    icar_rows_logdensity, icar_cols_logdensity
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

@testset "ICAR fields: the rows or columns of one matrix over a shared graph" begin
    rows, cols = prepare(icar_rows_logdensity), prepare(icar_cols_logdensity)
    node1 = [1, 1, 2, 3, 4, 2]
    node2 = [2, 3, 3, 4, 5, 4]
    P = [0.3 -0.1 0.7 0.2 -0.4 1.1;
         0.5 0.2 -0.3 0.9 0.0 -0.6;
         -0.2 0.4 0.1 -0.8 0.6 0.3]
    expected = sum(icar_laplacian_reference(P[k, :], node1, node2) for k in 1:3)
    # The sum of the independent fields' densities, each `icar_logdensity`.
    @test rows(P, node1, node2) ≈ expected rtol = 1e-14
    @test cols(permutedims(P), node1, node2) ≈ expected rtol = 1e-14
    @test rows(P, node1, node2) ≈
        sum(icar_logdensity(P[k, :], node1, node2) for k in 1:3) rtol = 1e-14
    # Improper per field: shifting one field by a constant leaves it unchanged.
    shifted = copy(P)
    shifted[2, :] .+= 3.0
    @test rows(shifted, node1, node2) ≈ rows(P, node1, node2) rtol = 1e-12
    # No fields or no edges: zero; edge lists of different lengths refused.
    @test rows(zeros(0, 6), node1, node2) == 0.0
    @test cols(zeros(6, 0), node1, node2) == 0.0
    @test rows(P, Int[], Int[]) == 0.0
    @test_throws DimensionMismatch rows(P, [1, 2], [2])
    @test_throws DimensionMismatch cols(permutedims(P), [1, 2], [2])

    # One retained edge plate with one field plate inside, whatever K.
    for law in (icar_rows_logdensity, icar_cols_logdensity)
        @test count(e -> e.kind === :plate, recipe_inventory(law)) == 2
    end

    # Native Enzyme Reverse: field k's gradient is its own field's.
    backend = DI.AutoEnzyme(; mode = Enzyme.Reverse)
    expected_gradient = zeros(size(P))
    for k in 1:3, (i, j) in zip(node1, node2)
        expected_gradient[k, i] -= P[k, i] - P[k, j]
        expected_gradient[k, j] += P[k, i] - P[k, j]
    end
    op = prepare_ad(rows, backend, P, node1, node2; active = :phi)
    @test ad_gradient(op, P, node1, node2) ≈ expected_gradient rtol = 1e-12 atol = 1e-14
    Pt = permutedims(P)
    op = prepare_ad(cols, backend, Pt, node1, node2; active = :phi)
    @test ad_gradient(op, Pt, node1, node2) ≈ permutedims(expected_gradient) rtol = 1e-12 atol = 1e-14
end
