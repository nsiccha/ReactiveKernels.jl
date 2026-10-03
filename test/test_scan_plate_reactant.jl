isdefined(@__MODULE__, :ScanPlateADTests) || include("test_scan_plate_ad.jl")
module ScanPlateReactantTests
using ReactiveKernels, Reactant, DifferentiationInterface, Enzyme, Test
using ..ScanPlateADTests: subject_scans

function operations(hlo)
    counts = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith)\.\w+", hlo)
        counts[m.match] = get(counts, m.match, 0)+1
    end
    counts
end

# Exercise the same array-valued cell boundary used by PK: the scan history
# leaves the plate before its outer consumer materializes and reduces it.
path_total(paths) = sum(sum, paths)
path_total(paths::ReactiveKernels._TensorizedPlateBatch) =
    sum(ReactiveKernels._tensorized_plate_materialize(paths))
@kernel collected_subject_scans(q::Vector{Float64}, X::Matrix{Float64}) = begin
    paths = plate(eachcol(X), Ref(q)) do xs, p
        seed = (value=p[1],)
        history = scan(xs, Ref(p); init=seed) do carry, x, params
            next = carry.value + params[2]*x
            ((value=next,), next)
        end
        history
    end
    total = path_total(paths)
    return total
end

@testset "nested subject scans retain compiled structure" begin
    inventories = Dict{String,Int}[]
    q = [0.3, 0.7]
    rq = Reactant.to_rarray(q)
    for (n, G) in ((3, 2), (11, 5))
        X = reshape(sin.(1:n*G), n, G)
        k = prepare(subject_scans; bound=(; X))
        hlo = repr(Reactant.@code_hlo optimize=false k(rq))
        @test count("stablehlo.while", hlo) == 1
        push!(inventories, operations(hlo))
        compiled = Reactant.@compile k(rq)
        @test Float64(compiled(rq)) ≈ k(q)
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), q; active=:q)
        cad = compile_ad_value_and_gradient(ad, rq)
        value, gradient = cad(rq)
        @test Float64(value) ≈ k(q)
        @test Array(gradient) ≈ ad_gradient(ad, q)
    end
    @test first(inventories) == last(inventories)
end

@testset "array-valued subject scans retain compiled structure" begin
    inventories = Dict{String,Int}[]
    q = [0.3, 0.7]
    rq = Reactant.to_rarray(q)
    for (n, G) in ((3, 2), (11, 5))
        X = reshape(sin.(1:n*G), n, G)
        weight = sum((n-i+1)*X[i,s] for s in 1:G for i in 1:n)
        expected = n*G*q[1] + weight*q[2]
        k = prepare(collected_subject_scans; bound=(; X))
        hlo = repr(Reactant.@code_hlo optimize=false k(rq))
        @test count("stablehlo.while", hlo) == 1
        push!(inventories, operations(hlo))
        compiled = Reactant.@compile k(rq)
        @test Float64(compiled(rq)) ≈ expected
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), q; active=:q)
        cad = compile_ad_value_and_gradient(ad, rq)
        value, gradient = cad(rq)
        @test Float64(value) ≈ expected
        @test Array(gradient) ≈ [n*G, weight]
    end
    @test first(inventories) == last(inventories)
end
end
