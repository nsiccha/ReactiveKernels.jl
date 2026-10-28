# Generic fixed-matrix and heterogeneous plate-result acceptance.
module CompoundPlateReactantTests
using ReactiveKernels, Reactant, StaticArrays, Test
using DifferentiationInterface, Enzyme

@kernel matrix_seed_chain(q::Vector{Float64}, xs::Vector{Float64},
        tags::Vector{Int64}) = begin
    scale = q[1]
    values = plate(xs, tags, Ref(scale)) do x, tag, s
        A = SMatrix{2,2}(x*s, s, 1.0, x)
        seed = (state=A, tag=tag, positive=x > 0)
        square = sum(seed.state * seed.state)
        # These selections choose between already-computed valid scalars.
        signed = ifelse(seed.positive, square, -square)
        value = signed + ifelse(seed.tag == 18014398509481985, 0.0, 10.0)
        value
    end
    total = sum(values)
    return total
end

@kernel matrix_values(xs::Vector{Float64}, scale::Float64) = begin
    values = plate(xs, scale) do x, s
        A = SMatrix{2,2}(x*s, s, 1.0, x)
        A
    end
    total = sum(values)
    return values, total
end

function operations(hlo)
    counts = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor)\.\w+", hlo)
        counts[m.match] = get(counts, m.match, 0) + 1
    end
    counts
end

@testset "fixed matrix and typed seed plate batching" begin
    inventories = Dict{String,Int}[]
    reverse_inventories = Dict{String,Int}[]
    k = prepare(matrix_seed_chain)
    q = [0.4]
    for xs in ([-1.0, 1.0], [-2.0, -1.0, 0.3, 1.0, 2.0])
        n = length(xs)
        # These distinct integers become equal if a compound result is packed
        # into Float64 storage. The Bool must also keep its dispatch/type.
        tags = Int64[18014398509481985 - iseven(i) for i in 1:n]
        rq, rx, rt = Reactant.to_rarray.((q, xs, tags))
        expected = sum(sign(x) * (x^2*(q[1]^2+1) + x*(q[1]+1)^2 + 2q[1]) +
            (iseven(i) ? 10.0 : 0.0) for (i,x) in enumerate(xs))
        derivative = sum(sign(x) * (2x^2*q[1] + 2x*(q[1]+1) + 2) for x in xs)
        @test k(q, xs, tags) ≈ expected
        compiled = Reactant.@compile k(rq, rx, rt)
        @test Float64(compiled(rq, rx, rt)) ≈ expected
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), q, xs, tags; active=:q)
        @test ad_gradient(ad, q, xs, tags) ≈ [derivative]
        cad = compile_ad_value_and_gradient(ad, rq, rx, rt)
        value, gradient = cad(rq, rx, rt)
        @test Float64(value) ≈ expected
        @test Array(gradient) ≈ [derivative]
        push!(inventories, operations(repr(Reactant.@code_hlo optimize=false k(rq, rx, rt))))
        reverse_hlo = repr(Reactant.@code_hlo optimize=true ad_value_and_gradient(ad, rq, rx, rt))
        push!(reverse_inventories, operations(reverse_hlo))

        matrices = prepare(matrix_values)
        scale = Reactant.to_rarray(q[1]; track_numbers=true)
        cmatrices = Reactant.@compile matrices(rx, scale)
        pointwise, total = cmatrices(rx, scale)
        reference, reference_total = matrices(xs, q[1])
        @test Array(pointwise) ≈ stack(reference; dims=1)
        @test Float64.(total) ≈ reference_total
    end
    @test first(inventories) == last(inventories)
    @test first(reverse_inventories) == last(reverse_inventories)
    println("typed matrix plate primal inventory: ", first(inventories))
    println("typed matrix plate reverse inventory: ", first(reverse_inventories))

    matrices = prepare(matrix_values)
    empty_xs = Reactant.to_rarray(Float64[])
    scale = Reactant.to_rarray(q[1]; track_numbers=true)
    compiled = Reactant.@compile matrices(empty_xs, scale)
    pointwise, total = compiled(empty_xs, scale)
    @test size(pointwise) == (0, 2, 2)
    @test total isa SMatrix{2,2}
    @test Float64.(total) ≈ last(matrices(Float64[], q[1]))
end

end
