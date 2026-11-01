module CellExemplarTypeTests
using Test, ReactiveKernels, DifferentiationInterface, Enzyme
const RK = ReactiveKernels

function weighted_readout(values, indices, weights)
    total = 0.0
    for j in eachindex(indices, weights)
        total += values[indices[j]] * weights[j]
    end
    total
end

@kernel collect_cells(cells) = if eltype(cells) <: Number
    cells
elseif eltype(cells) <: AbstractVector
    isempty(cells) ? eltype(eltype(cells))[] : reduce(vcat, cells)
else
    error("unsupported cells")
end

@kernel collected(u::AbstractVector{Float64}, groups::Vector{Vector{Int}},
        weights::Vector{Vector{Float64}}, indices::Vector{Int}, y::Vector{Float64}) = begin
    values = exp.(u[1:3])
    readout_cells = plate(eachindex(groups)) do i
        weighted_readout(values, groups[i], weights[i])
    end
    readouts = collect_cells(readout_cells)
    selected_cells = plate(eachindex(indices)) do i
        0.1 + (indices[i] > 0 ? readouts[indices[i]] : 0.0)
    end
    selected = collect_cells(selected_cells)
    densities = plate(selected, y) do mean, yi
        -(mean - yi)^2 / 2
    end
    return sum(densities)
end

nodes(ex) = ex isa Expr ? 1 + sum(nodes, ex.args; init=0) : 1

@testset "exact exemplar types reach selected-cell locality analysis" begin
    costs = Tuple{Int,Int}[]
    for n in (16, 1024)
        data = (; groups=[isodd(i) ? [1, 3] : Int[] for i in 1:n],
            weights=[isodd(i) ? [0.4, 0.6] : Float64[] for i in 1:n],
            indices=[i % 3 == 0 ? 0 : i for i in 1:n], y=fill(0.7, n))
        saved = deepcopy(data)
        u = [0.2, -0.3, 0.4]
        spec = plate_cell(collected, :densities; index=:i, locality=:error)
        cell = prepare(spec; bound=data, exemplars=(u, 1))
        full = prepare(collected; bound=data, exemplars=(u,))
        ad = prepare_ad(cell, AutoEnzyme(; mode=Enzyme.Reverse), u, 1; active=:u)
        for shift in (0.0, 0.13)
            x = u .+ shift
            means = [weighted_readout(exp.(x), data.groups[i], data.weights[i]) for i in 1:n]
            expected = [-(0.1 + (i == 0 ? 0.0 : means[i]) - yi)^2 / 2
                        for (i, yi) in zip(data.indices, data.y)]
            @test full(x) ≈ sum(expected)
            for i in (1, 2, 3, n)
                @test cell(x, i) ≈ expected[i]
                result = ad_value_and_gradient(ad, x, i)
                mean = 0.1 + (data.indices[i] == 0 ? 0.0 : means[data.indices[i]])
                grad = zeros(3)
                if data.indices[i] > 0
                    for (j, w) in zip(data.groups[data.indices[i]], data.weights[data.indices[i]])
                        grad[j] += (data.y[i] - mean) * exp(x[j]) * w
                    end
                end
                @test first(result) ≈ expected[i]
                @test last(result) ≈ grad
            end
        end
        @test_throws BoundsError cell(u, n + 1)
        @test_throws ArgumentError cell(view(u, :), 1)
        invalid = (; data..., indices=[n + 1; data.indices[2:end]])
        invalid_cell = prepare(spec; bound=invalid, exemplars=(u, 1))
        @test_throws BoundsError invalid_cell(u, 1)
        @test data == saved
        cell(u, 1)
        push!(costs, (minimum(@allocated(cell(u, 1)) for _ in 1:3), nodes(code_expr(cell))))
        @info "selected cell with exact exemplar" n costs=costs[end]
    end
    @test costs[1] == costs[2]
end
end
