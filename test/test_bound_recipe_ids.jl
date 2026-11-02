module BoundRecipeIdTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel split_cells(z::Vector{Float64}, d::Vector{Float64}, indices::Vector{Int}) = begin
    mu = exp.(z) .+ log.(d)
    cells = plate(indices) do i
        mu[i]^2
    end
end

@kernel split_groups(z::Vector{Float64}, d::Vector{Float64}, groups::Vector{Vector{Int}}) = begin
    mu = exp.(z) .+ log.(d)
    cells = plate(groups) do rows
        sum(abs2, mu[rows])
    end
end

astsize(x::Expr) = 1 + sum(astsize, x.args; init=0)
astsize(x) = 1
unique_ids(p) = length(unique(r.id for r in p.recipes)) == length(p.recipes)

@testset "bound constants keep distinct ids from residual split recipes" begin
    cell = plate_cell(split_cells, :cells; index=:i, locality=:error)
    have = (:z, :d, :indices, :i)
    p = plan(cell; have, want=:cells_cell)
    cache = PreparationCache()
    structures = Int[]
    for n in (4, 400)
        z = collect(range(-0.3, 0.4; length=n))
        d = collect(range(1.1, 2.0; length=n))
        indices = [n, 1, n ÷ 2, 1]
        snapshot = deepcopy((z, d, indices))
        bound = (; d, indices)
        residual = partial_evaluation(p, (cell[:d], cell[:indices]), (d, indices))
        # Splitting leaves a live negative-id recipe for exp.(z). Previously
        # the first hoisted constant reused its id, corrupting cell-form lookup.
        @test any(r -> r.id < 0 && !isempty(r.inputs), residual.recipes)
        @test unique_ids(residual)
        staged = partial_evaluation(p, (cell[:d],), (d,))
        staged = partial_evaluation(staged, (cell[:indices],), (indices,))
        @test unique_ids(staged)
        kernels = (
            prepare(cell; have, want=:cells_cell, bound),
            prepare!(cache, cell; have, want=:cells_cell, bound),
            prepare(staged),
        )
        push!(structures, astsize(code_expr(first(kernels))))
        for k in kernels
            ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), z, 1; active=:z)
            for point in (z, z .+ 0.15), i in eachindex(indices)
                j = indices[i]
                mu = exp(point[j]) + log(d[j])
                expected_gradient = zeros(n)
                expected_gradient[j] = 2mu * exp(point[j])
                @test k(point, i) ≈ mu^2
                @test partition_footprint(k, point, i).z == [j]
                value, gradient = ad_value_and_gradient(ad, point, i)
                @test value ≈ mu^2
                @test gradient ≈ expected_gradient
            end
            @test_throws BoundsError k(z, 0)
            @test_throws BoundsError k(z, length(indices) + 1)
            @test_throws DimensionMismatch k(vcat(z, 0.0), 1)
        end
        rebound = prepare!(cache, cell; have, want=:cells_cell,
                           bound=(d=2d, indices=reverse(indices)))
        @test rebound(z, 1) ≈ (exp(z[last(indices)]) + log(2d[last(indices)]))^2
        @test kernels[2](z, 1) ≈ (exp(z[lastindex(z)]) + log(d[lastindex(d)]))^2
        @test (z, d, indices) == snapshot
    end
    @test only(unique(structures)) > 0
end

@testset "split bound cell shapes retain ragged and empty selections" begin
    z, d = [0.1, -0.2, 0.3], [1.2, 2.1, 1.7]
    groups = [[3, 1, 3], Int[], [2]]
    cell = plate_cell(split_groups, :cells; index=:i, locality=:error)
    k = prepare(cell; have=(:z, :d, :groups, :i), want=:cells_cell, bound=(; d, groups))
    ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), z, 1; active=:z)
    for (i, rows) in enumerate(groups)
        mu = exp.(z) .+ log.(d)
        @test k(z, i) ≈ sum(abs2, mu[rows])
        @test partition_footprint(k, z, i).z == sort(unique(rows))
        value, gradient = ad_value_and_gradient(ad, z, i)
        @test value ≈ sum(abs2, mu[rows])
        @test gradient ≈ [2mu[j] * exp(z[j]) * count(==(j), rows) for j in eachindex(z)]
    end
    empty_domain = prepare(cell; have=(:z, :d, :groups, :i), want=:cells_cell,
                           bound=(; d, groups=Vector{Int}[]))
    @test_throws BoundsError empty_domain(z, 1)
end
end
