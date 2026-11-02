module CellRaggedIndexTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel direct(z::Vector{Float64}, ids::Vector{Int}, coords::Vector{Vector{Int}},
               group_of::Vector{Int}, subject::Vector{Int}) = begin
    groups = plate(ids) do k
        sum(abs2, z[coords[k]])
    end
    density = plate(group_of) do g
        groups[g]
    end
end

@kernel computed(z::Vector{Float64}, ids::Vector{Int}, coords::Vector{Vector{Int}},
                 group_of::Vector{Int}, subject::Vector{Int}) = begin
    field = 2 .* z
    groups = plate(ids) do k
        sum(abs2, field[coords[k]])
    end
    density = plate(group_of) do g
        groups[g]
    end
end

@kernel logical_masks(z::Vector{Float64}, masks::AbstractVector{<:AbstractVector{Bool}},
                      selected::Vector{Int}) = begin
    field = 2 .* z
    groups = plate(masks) do mask
        sum(field[mask])
    end
    density = plate(selected) do i
        groups[i]
    end
end

@testset "logical masks are not proved as ragged position vectors" begin
    z = [0.1, 0.2, 0.3, 0.4]
    selected = [2, 1]
    for vector in (Vector{Bool}, BitVector)
        masks = [vector([true, false, true, false]), vector([false, true, false, true])]
        bound = (; masks, selected)
        full = prepare(logical_masks; have=(:z, :masks, :selected), want=:density, bound)
        @test full(z) ≈ [sum(2 .* z[masks[i]]) for i in selected]
        @test_throws PartitionLocalityError prepare(
            plate_cell(logical_masks, :density; index=:i, locality=:error);
            have=(:z, :masks, :selected, :i), want=:density_cell, bound)
    end
end

const data = (ids=[4, 2, 1, 3, 2], coords=[[4, 2, 4], Int[], [6], [3, 1]],
              group_of=[3, 1, 5, 2, 4, 3], subject=[1, 2, 1, 2, 1, 2])
const have = (:z, keys(data)...)
positions(rows) = reduce(vcat, (data.coords[data.ids[data.group_of[i]]] for i in rows);
                         init=Int[])
astsize(x::Expr) = 1 + sum(astsize, x.args; init=0)
astsize(x) = 1

shares(z, residual, lo) = lo .+ residual .* (exp.(z) ./ sum(exp.(z)))
@kernel shares_rows(z::Vector{Float64}, ids::Vector{Int}, coords::Vector{Vector{Int}},
                    residuals::Vector{Float64}, lo::Vector{Float64},
                    entries::Vector{Vector{Int}}, group_of::Vector{Int}, pos::Vector{Int}) = begin
    groups = plate(ids) do k
        shares(z[coords[k]], residuals[k], lo[entries[k]])
    end
    density = plate(group_of, pos) do g, j
        groups[g][j]^2
    end
end

@testset "bound array-valued group helper retains strict cell locality" begin
    d = (ids=[2, 1], coords=[[4, 2, 4], [3, 1]], residuals=[1.5, 2.0],
         lo=[0.1, 0.2, 0.3, 0.4, 0.5], entries=[[1, 2, 3], [4, 5]],
         group_of=[2, 1, 2, 1, 2], pos=[3, 1, 1, 2, 2])
    k = prepare(plate_cell(shares_rows, :density; index=:i, locality=:error);
                have=(:z, keys(d)..., :i), want=:density_cell, bound=d)
    z = [0.2, -0.1, 0.3, 0.5]
    ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), z, 1; active=:z)
    for i in eachindex(d.group_of)
        group = d.ids[d.group_of[i]]
        selected = d.coords[group]
        p = exp.(z[selected]); p ./= sum(p)
        j = d.pos[i]
        x = d.lo[d.entries[group][j]] + d.residuals[group]*p[j]
        expected_gradient = zeros(length(z))
        for (l, index) in enumerate(selected)
            expected_gradient[index] += 2x*d.residuals[group]*p[j]*((l == j)-p[l])
        end
        @test k(z, i) ≈ x^2
        @test partition_footprint(k, z, i).z == sort(unique(selected))
        value, gradient = ad_value_and_gradient(ad, z, i)
        @test value ≈ x^2
        @test gradient ≈ expected_gradient
    end
end

@testset "ragged indices in bound and live plate operands" begin
    snapshot = deepcopy(data)
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for (model, scale) in ((direct, 1.0), (computed, 2.0)), bind in (false, true)
        bound = bind ? data : NamedTuple()
        cell = prepare(plate_cell(model, :density; index=:i, locality=:error);
                       have=(have..., :i), want=:density_cell, bound)
        partition = prepare(plate_partition(model, :density; by=:subject,
                            index=:label, locality=:error);
                            have=(have..., :label), want=:partition_total, bound)
        args(z, i) = bind ? (z, i) : (z, values(data)..., i)
        for (kernel, rowsets) in ((cell, [[i] for i in eachindex(data.group_of)]),
                                 (partition, [findall(==(s), data.subject) for s in 1:2]))
            @test all(e.status !== :whole for e in partition_locality(kernel))
            ad = prepare_ad(kernel, backend, args(collect(0.1:0.1:0.6), 1)...; active=:z)
            for z in (collect(0.1:0.1:0.6), collect(-0.3:0.2:0.7))
                before = copy(z)
                for (i, rows) in enumerate(rowsets)
                    selected = positions(rows)
                    expected = sum(abs2, scale .* z[selected])
                    @test kernel(args(z, i)...) ≈ expected
                    @test partition_footprint(kernel, args(z, i)...).z == sort(unique(selected))
                    value, gradient = ad_value_and_gradient(ad, args(z, i)...)
                    @test value ≈ expected
                    @test gradient ≈ [2scale^2*z[j]*count(==(j), selected) for j in eachindex(z)]
                end
                @test z == before
            end
        end
    end
    @test data == snapshot
end

@testset "ragged plate source and work stay bounded by selected rows" begin
    structures = Int[]
    allocations = Int[]
    for n in (4, 400)
        d = (ids=collect(1:n), coords=[[2i, 2i-1, 2i] for i in 1:n],
             group_of=collect(n:-1:1), subject=ones(Int, n))
        k = prepare(plate_cell(computed, :density; index=:i, locality=:error);
                    have=(have..., :i), want=:density_cell, bound=d)
        z = fill(0.5, 2n)
        @test k(z, 2) == 3.0
        @test partition_footprint(k, z, 2).z == [2n-3, 2n-2]
        push!(structures, astsize(code_expr(k)))
        k(z, 2)
        push!(allocations, @allocated k(z, 2))
    end
    @test only(unique(structures)) > 0
    @test only(unique(allocations)) >= 0
end
end
