module PlateConsumerLayoutReactantTests
using ReactiveKernels, Reactant, DifferentiationInterface, Enzyme, Test
using StaticArrays: SMatrix
Reactant.set_default_backend("cpu")

# An ordinary consumer must see the same logical lanes as native Julia. These
# helpers deliberately have no method for RK's internal tensorized marker.
pack_lanes(lanes) = vec(stack(lanes))
stack_first(lanes) = stack(lanes; dims=1)
stack_second(lanes) = stack(lanes; dims=2)

@kernel column_pack(q, X) = begin
    lanes = plate(eachcol(X), Ref(q)) do xs, p
        p[1] .* xs
    end
    packed = pack_lanes(lanes)
    first_axis = stack_first(lanes)
    return packed, first_axis, lanes
end

@kernel column_loss(q, X, weights) = begin
    lanes = plate(eachcol(X), Ref(q)) do xs, p
        p[1] .* xs
    end
    packed = pack_lanes(lanes)
    total = sum(packed .* weights)
    return total
end

@kernel matrix_pack(q, X) = begin
    lanes = plate(eachcol(X), Ref(q)) do xs, p
        reshape(p[1] .* xs, 2, 2)
    end
    packed = pack_lanes(lanes)
    middle_axis = stack_second(lanes)
    return packed, middle_axis, lanes
end

@kernel column_sum(q, X) = begin
    lanes = plate(eachcol(X), Ref(q)) do xs, p
        p[1] .* xs
    end
    combined = sum(lanes)
    total = sum(combined)
    return combined, total
end

@kernel fixed_matrix_pack(q, X) = begin
    lanes = plate(eachcol(X), Ref(q)) do xs, p
        SMatrix{2,2}(p[1]*xs[1], p[1]*xs[2], p[1]*xs[3], p[1]*xs[4])
    end
    packed = pack_lanes(lanes)
    combined = sum(lanes)
    return packed, combined, lanes
end

function inventory(hlo)
    counts = Dict{String,Int}()
    for match in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme)\.\w+", repr(hlo))
        counts[match.match] = get(counts, match.match, 0) + 1
    end
    counts
end

@testset "array-valued plate consumers preserve lane layout" begin
    primal_inventories = Dict{String,Int}[]
    reverse_inventories = Dict{String,Int}[]
    q = [0.7]
    rq = Reactant.to_rarray(q)
    for count in (1, 3, 9)
        X = reshape(cos.(1:4count), 4, count)
        weights = sin.(1:4count)
        original_X = copy(X)
        original_weights = copy(weights)
        columns = prepare(column_pack; bound=(; X))
        matrices = prepare(matrix_pack; bound=(; X))
        fixed_matrices = prepare(fixed_matrix_pack; bound=(; X))
        combined = prepare(column_sum; bound=(; X))
        loss = prepare(column_loss; bound=(; X, weights))
        ad = prepare_ad(loss, AutoEnzyme(; mode=Enzyme.Reverse), q; active=:q)
        compiled_columns = Reactant.@compile columns(rq)
        compiled_matrices = Reactant.@compile matrices(rq)
        compiled_fixed = Reactant.@compile fixed_matrices(rq)
        compiled_sum = Reactant.@compile combined(rq)
        compiled_reverse = compile_ad_value_and_gradient(ad, rq)
        reverse_function = let kernel = loss
            input -> only(Enzyme.gradient(Enzyme.Reverse, kernel, input))
        end
        push!(primal_inventories, inventory(Reactant.@code_hlo loss(rq)))
        push!(reverse_inventories, inventory(Reactant.@code_hlo reverse_function(rq)))
        for scale in (0.7, -0.2)
            input = [scale]
            rinput = Reactant.to_rarray(input)
            expected = scale .* vec(X)
            native_packed, native_first, native_lanes = columns(input)
            packed, first_axis, lanes = compiled_columns(rinput)
            @test native_packed ≈ expected
            @test Array(packed) ≈ expected
            @test Array(first_axis) ≈ native_first
            @test Array(lanes) ≈ stack(native_lanes; dims=1)
            native_matrix_packed, native_middle, native_matrices = matrices(input)
            matrix_packed, middle_axis, matrix_lanes = compiled_matrices(rinput)
            @test native_matrix_packed ≈ expected
            @test Array(matrix_packed) ≈ expected
            @test Array(middle_axis) ≈ native_middle
            @test Array(matrix_lanes) ≈ stack(native_matrices; dims=1)
            _, native_fixed_sum, native_fixed_lanes = fixed_matrices(input)
            fixed_packed, fixed_sum, fixed_lanes = compiled_fixed(rinput)
            @test Array(fixed_packed) ≈ expected
            @test Array(fixed_sum) ≈ native_fixed_sum
            @test Array(fixed_lanes) ≈ stack(native_fixed_lanes; dims=1)
            reduced, total = compiled_sum(rinput)
            @test Array(reduced) ≈ vec(sum(scale .* X; dims=2))
            @test Float64(total) ≈ sum(expected)
            value, gradient = compiled_reverse(rinput)
            @test Float64(value) ≈ sum(expected .* weights)
            @test ad_gradient(ad, input) ≈ [sum(vec(X) .* weights)]
            @test Array(gradient) ≈ [sum(vec(X) .* weights)]
            @test input == [scale]
        end
        @test X == original_X
        @test weights == original_weights
    end
    # The default optimizer drops the lanes-to-vector reshape at one lane.
    # Admit exactly that shape simplification; every arithmetic and control
    # operation must still have the same count, with no replicated lane body.
    @test get(primal_inventories[1], "stablehlo.reshape", 0) == 0
    @test get(primal_inventories[2], "stablehlo.reshape", 0) == 1
    @test merge(primal_inventories[1], Dict("stablehlo.reshape" => 1)) ==
        primal_inventories[2]
    @test primal_inventories[2] == primal_inventories[3]
    @test all(==(first(reverse_inventories)), reverse_inventories)
end
end
