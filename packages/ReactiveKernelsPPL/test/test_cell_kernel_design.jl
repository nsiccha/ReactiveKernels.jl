module CellKernelDesignTests
using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using Distributions: Normal, logpdf

struct RecordingPoint <: AbstractVector{Float64}
    data::Vector{Float64}
    seen::Vector{Int}
end
Base.size(x::RecordingPoint) = size(x.data)
Base.IndexStyle(::Type{RecordingPoint}) = IndexLinear()
Base.getindex(x::RecordingPoint, i::Int) = (push!(x.seen, i); x.data[i])

@kernel function grouped_column(z::AbstractVector{Float64}, groups, coordinates, entries, owner, slot)
    blocks = plate(groups) do g
        exp.(z[coordinates[g]])
    end
    gathered = plate(entries) do i
        block = blocks[owner[i]]
        value::Float64 = block[slot[i]]
        value
    end
    squared = plate(entries) do i
        gathered[i]^2
    end
    return squared
end

const MODEL = quote
    z[axes(packed, 1)] .~ Normal.(0, 1)
    beta[1:2] .~ Normal.(0, 1)
    values = grouped_column(z, groups, coordinates, entries, owner, slot)
    log_values = log.(values)
    X = hcat(one, log_values)
    mu = X * beta .+ offset
    y .~ Normal.(mu, 1)
end

function fixture(n)
    coordinates = [collect((2g-1):(2g)) for g in 1:n]
    owner = repeat(collect(1:n); inner = 2)
    slot = repeat([2, 1], n)
    (; packed = zeros(2n), groups = collect(1:n), coordinates,
        entries = collect(1:2n), owner, slot, one = ones(2n),
        offset = fill(0.1, 2n), y = fill(0.7, 2n))
end

nodes(ex) = ex isa Expr ? 1 + sum(nodes, ex.args; init = 0) : 1

@testset "a grouped kernel column stays local through a design and offset" begin
    costs = Tuple{Int,Int}[]
    for n in (3, 300)
        data = fixture(n)
        saved = deepcopy(data)
        bound = bind_data(lower_rkppl(MODEL, data; conditioned = (:y,), mod = @__MODULE__), data)
        built = build_kernel(bound)
        q = prepare_cell_query(built, bound, :y; locality = :error)
        names = coordinate_names(built.layout)
        zi = [only(findall(==(Symbol("z.$j")), names)) for j in 1:2n]
        bi = [only(findall(==(Symbol("beta.$j")), names)) for j in 1:2]
        u = [0.1sin(j) for j in eachindex(names)]
        sampler = prepare_cell_sampler(built, bound, :y, u;
            locality = :error, backend = AutoEnzyme(; mode = Enzyme.Reverse))
        for cell in (1, 4, 2n)
            g = data.owner[cell]
            zidx = zi[data.coordinates[g][data.slot[cell]]]
            mu = u[bi[1]] + 2u[bi[2]] * u[zidx] + data.offset[cell]
            expected = logpdf(Normal(mu, 1), data.y[cell])
            point = RecordingPoint(copy(u), Int[])
            @test q(point, cell) ≈ expected
            @test sort!(unique!(point.seen)) == sort([zi[data.coordinates[g]]; bi])
            gradient = zeros(length(u))
            value, _ = cell_value_and_gradient!(sampler, gradient, u, cell)
            reference = zeros(length(u))
            residual = data.y[cell] - mu
            reference[bi[1]] = residual
            reference[bi[2]] = residual * 2u[zidx]
            reference[zidx] = residual * 2u[bi[2]]
            @test value ≈ expected
            @test gradient ≈ reference
        end
        q(u, 1)
        push!(costs, (minimum(@allocated(q(u, 1)) for _ in 1:3), nodes(code_expr(q.kernel))))
        @info "grouped design cell" groups=n allocated_bytes=costs[end][1] generated_nodes=costs[end][2]
        @test data == saved
        @test_throws BoundsError q(u, 2n + 1)
    end
    @test costs[1] == costs[2]
end
end
