# Backend-only boundary: default optimized reverse compiles and keeps lazy
# arithmetic, but a small lane count expands the batch's control-flow regions.
using Reactant, Enzyme, Test
Reactant.set_default_backend("cpu")

function lazy_cell(x0)
    x = Reactant.@allowscalar x0[]
    y = zero(x)
    Reactant.@trace if x > 0
        y = log(x)
    end
    y
end
lazy_batch_loss(v) = sum(only(Reactant.Ops.batch(lazy_cell, [v], Int64[length(v)])))
lazy_batch_gradient(v) = only(Enzyme.gradient(Enzyme.Reverse, lazy_batch_loss, v))

function operation_inventory(hlo)
    counts = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor)\.\w+", hlo)
        counts[m.match] = get(counts, m.match, 0) + 1
    end
    counts
end

@testset "Reactant lazy batch default-reverse growth boundary" begin
    inventories = Dict{String,Int}[]
    for values in ([2.0, -1.0], [2.0, -1.0, 0.5, -3.0, 1.5])
        rv = Reactant.to_rarray(values)
        loss = Reactant.@compile lazy_batch_loss(rv)
        gradient = Reactant.@compile lazy_batch_gradient(rv)
        @test Float64(loss(rv)) ≈ sum(x > 0 ? log(x) : 0.0 for x in values)
        @test Array(gradient(rv)) ≈ [x > 0 ? inv(x) : 0.0 for x in values]
        hlo = repr(Reactant.@code_hlo optimize=true lazy_batch_gradient(rv))
        inventory = operation_inventory(hlo)
        println("lazy batch reverse inventory, lanes=", length(values), ": ", inventory)
        push!(inventories, inventory)
    end
    @test get(first(inventories), "stablehlo.while", 0) == 0
    @test get(last(inventories), "stablehlo.while", 0) > 0
    @test_broken first(inventories) == last(inventories)
end
