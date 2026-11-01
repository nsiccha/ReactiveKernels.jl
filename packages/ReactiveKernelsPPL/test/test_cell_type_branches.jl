module CellTypeBranchTests
using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using Distributions: Normal, logpdf

@kernel collect_cells(cells) = if eltype(cells) <: Number
    cells
elseif eltype(cells) <: AbstractVector
    isempty(cells) ? eltype(eltype(cells))[] : reduce(vcat, cells)
else
    error("unsupported cells")
end

const COLLECTED = quote
    beta ~ Normal(0, 1)
    @plate for i in eachindex(x)
        signal_cells[i] = beta * x[i]
    end
    signal = collect_cells(signal_cells)
    location = signal .+ shift
    y .~ Normal.(location, 1.0)
end
const DIRECT = quote
    beta ~ Normal(0, 1)
    @plate for i in eachindex(x)
        y[i] ~ Normal(beta * x[i] + shift[i], 1.0)
    end
end
nodes(ex) = ex isa Expr ? 1 + sum(nodes, ex.args; init = 0) : 1

@testset "scalar collected plates retain strict selected-row locality" begin
    costs = Dict{Symbol,Vector{Tuple{Int,Int,Int}}}()
    for (name, source) in ((:collected, COLLECTED), (:direct, DIRECT))
        costs[name] = Tuple{Int,Int,Int}[]
        for n in (16, 1024)
            data = (; x = collect(range(0.5, 2.; length=n)),
                shift = fill(0.2, n), y = fill(0.7, n))
            saved = deepcopy(data)
            bound = bind_data(lower_rkppl(source, data; conditioned=(:y,),
                mod=@__MODULE__), data)
            built = build_kernel(bound)
            u, gradient = [0.3], zeros(1)
            backend = AutoEnzyme(; mode=Enzyme.Reverse)
            full = prepare_sampler(built, bound, u; backend)
            q = prepare_cell_query(built, bound, :y; locality=:error)
            cell = prepare_cell_sampler(built, bound, :y, u; backend, locality=:error)
            for beta in (0.0, 0.3, -0.4)
                u[1] = beta
                value, _ = sampler_value_and_gradient!(full, gradient, u)
                residuals = data.y .- (beta .* data.x .+ data.shift)
                @test value ≈ logpdf(Normal(), beta) + sum(logpdf.(Normal(), residuals))
                @test gradient[1] ≈ -beta + sum(residuals .* data.x)
                for i in (1, n ÷ 2, n)
                    expected = logpdf(Normal(), residuals[i])
                    @test q(u, i) ≈ expected
                    v, _ = cell_value_and_gradient!(cell, gradient, u, i)
                    @test v ≈ expected
                    @test gradient[1] ≈ residuals[i] * data.x[i]
                end
            end
            @test_throws BoundsError q(u, n + 1)
            q(u, 1)
            cell_value_and_gradient!(cell, gradient, u, 1)
            value_bytes = minimum(@allocated(q(u, 1)) for _ in 1:3)
            gradient_bytes = minimum(@allocated(cell_value_and_gradient!(cell,
                gradient, u, 1)) for _ in 1:3)
            push!(costs[name], (value_bytes, gradient_bytes, nodes(code_expr(q.kernel))))
            @info "type-selected collector" name n value_bytes gradient_bytes nodes=costs[name][end][3]
            @test data == saved
        end
        @test costs[name][1] == costs[name][2]
    end
    @test costs[:collected][1][1] <= costs[:direct][1][1] + 128
end
end
