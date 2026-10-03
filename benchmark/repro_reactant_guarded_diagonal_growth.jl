# Backend-only boundary: default reverse preserves the guarded vector prior
# numerically, but expands its retained loop when the bound dimension is small.
using Reactant, Enzyme, Test
Reactant.set_default_backend("cpu")

function diagonal_sum(eta, diagonal, K)
    total = zero(eta)
    Reactant.@trace track_numbers=false for i in 2:K
        total += (K - i + 2 * eta - 2) *
            log(Reactant.@allowscalar diagonal[i])
    end
    return total
end

function diagonal_loss(u)
    eta = sum(view(u, 1:1))
    K = length(u) - 1
    diagonal = u[2:(K + 1)]
    total = zero(eta)
    Reactant.@trace if isfinite(eta) && eta > 0
        total = diagonal_sum(eta, diagonal, K)
    end
    return total
end

diagonal_gradient(u) = only(Enzyme.gradient(Enzyme.Reverse,
    Enzyme.Const(diagonal_loss), u))

function operation_inventory(hlo)
    counts = Dict{String,Int}()
    for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", repr(hlo))
        counts[m.match] = get(counts, m.match, 0) + 1
    end
    return counts
end

function retained_work_inventory(ops)
    simplified = ("stablehlo.constant", "stablehlo.reshape", "stablehlo.add",
        "stablehlo.multiply", "stablehlo.subtract", "stablehlo.negate")
    return Dict(name => count for (name, count) in ops if name ∉ simplified)
end

@testset "Reactant guarded diagonal default-reverse growth boundary" begin
    inventories = []
    for K in (2, 4, 8, 16)
        u = [1.5; fill(0.8, K)]
        ru = Reactant.to_rarray(u)
        loss = Reactant.@compile diagonal_loss(ru)
        gradient = Reactant.@compile diagonal_gradient(ru)
        @test Float64(loss(ru)) ≈ diagonal_loss(u) rtol = 1e-12
        # Independent derivatives are an oracle only, never an AD adapter.
        expected = [2 * sum(log, u[3:end]); 0.0;
            [(K - i + 2u[1] - 2) / u[i + 1] for i in 2:K]]
        @test Array(gradient(ru)) ≈ expected rtol = 1e-12
        invalid = Reactant.to_rarray([-1.0; fill(-0.2, K)])
        @test Float64(loss(invalid)) == 0.0
        @test Array(gradient(invalid)) == zeros(K + 1)
        ops = (
            operation_inventory(Reactant.@code_hlo optimize=true diagonal_loss(ru)),
            operation_inventory(Reactant.@code_hlo optimize=true diagonal_gradient(ru)))
        println("guarded diagonal inventory, K=", K, ": ", ops)
        push!(inventories, ops)
    end
    @test inventories[3] == inventories[4]
    @test all(ops -> get(ops, "stablehlo.while", 0) > 0, inventories[3])
    # Small-shape scalar identities may change raw counts; retained control
    # flow, nonlinear work and indexing must agree at every dimension.
    retained = [map(retained_work_inventory, pair) for pair in inventories]
    @test_broken allequal(retained)
end
