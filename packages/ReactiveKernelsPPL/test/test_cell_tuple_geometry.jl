module CellTupleGeometryTests
using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using Distributions: Normal, logpdf

struct RecordingPoint <: AbstractVector{Float64}
    data::Vector{Float64}
    seen::Vector{Int}
end
Base.size(x::RecordingPoint) = size(x.data)
Base.IndexStyle(::Type{RecordingPoint}) = IndexLinear()
Base.getindex(x::RecordingPoint, i::Int) = (push!(x.seen, i); x.data[i])

@kernel function grouped(z, groups, coordinates, entries, owner, slot)
    blocks = plate(groups) do g
        exp.(z[coordinates[g]])
    end
    values = plate(entries) do i
        block = blocks[owner[i]]
        value::Float64 = block[slot[i]]
        value
    end
    return values
end

const geometry_calls = Ref(0)
row_count(observed_rows, missing_rows) =
    (geometry_calls[] += 1; length(observed_rows) + length(missing_rows))

@kernel function geometry(observed, observed_rows, missing_rows)
    n = row_count(observed_rows, missing_rows)
    rows = 1:n
    observed_by_row = Dict(zip(observed_rows, observed))
    missing_by_row = Dict(zip(missing_rows, eachindex(missing_rows)))
    observed_component = plate(rows) do row
        get(observed_by_row, row, 0.0)
    end
    order = plate(rows) do row
        haskey(missing_by_row, row) ? n + missing_by_row[row] : row
    end
    return observed_component, order
end

@kernel function complete(observed, missing, order)
    stacked = vcat(observed, missing)
    completed = stacked[order]
    return completed
end

@rkppl effects(groups) = begin
    tau[1:1] .~ LogNormal.(0, 1)
    z[1:length(groups), 1:1] .~ Normal.(0, 1)
    return z .* tau[1]
end

@rkppl population(X) = begin
    beta[1:2] .~ Normal.(0, 1)
    return X * beta
end

function model(computed)
    setup = computed ? :((observed_component, order) =
        geometry(observed, observed_rows, missing_rows)) : quote
        observed_component = observed_data
        order = order_data
    end
    statements = Meta.isexpr(setup, :block) ? setup.args : [setup]
    return quote
        z[axes(packed, 1)] .~ Normal.(0, 1)
        b ~ effects(groups)
        missing = grouped(z, groups, coordinates, entries, owner, slot)
        $(statements...)
        completed = complete(observed_component, missing, order)
        lx = log.(completed)
        X = hcat(one, lx)
        pop ~ population(X)
        ranef = b[effect_rows, 1]
        m = pop .+ ranef
        y .~ Normal.(m, 1)
    end
end

function fixture(n, ragged)
    sizes = ragged ? repeat([0, 1, 5], n ÷ 3) : fill(2, n)
    ends = cumsum(sizes)
    starts = ends .- sizes .+ 1
    coordinates = [collect(a:b) for (a, b) in zip(starts, ends)]
    owner = reduce(vcat, [fill(g, sizes[g]) for g in 1:n])
    slot = reduce(vcat, [collect(k:-1:1) for k in sizes])
    observed = [1.5, 1.7]
    observed_rows = [1, 2]
    missing_rows = collect(3:2n+2)
    (; packed = zeros(2n), groups = collect(1:n), coordinates,
        entries = collect(1:2n), owner, slot, observed, observed_rows, missing_rows,
        observed_data = [observed; zeros(2n)], order_data = [1, 2, ((2n+3):(4n+2))...],
        effect_rows = [1, 1, owner...], one = ones(2n+2), y = ones(2n+2))
end

nodes(ex) = ex isa Expr ? 1 + sum(nodes, ex.args; init = 0) : 1

@testset "tuple-valued data geometry preserves strict cell locality" begin
    for computed in (false, true), ragged in (false, true)
        costs = Tuple{Int,Int}[]
        for n in (3, 300)
            data = fixture(n, ragged)
            saved = deepcopy(data)
            geometry_calls[] = 0
            plan = lower_rkppl(model(computed), data; conditioned = (:y,), mod = @__MODULE__)
            @test geometry_calls[] == 0
            bound = bind_data(plan, data)
            built = build_kernel(bound)
            q = prepare_cell_query(built, bound, :y; locality = :error)
            names = coordinate_names(built.layout)
            index(name) = only(findall(==(Symbol(name)), names))
            zi = [index("z.$j") for j in 1:2n]
            bi = [index("pop.beta.$j") for j in 1:2]
            tau = index("b.tau.1")
            effects_i = [index("b.z.$j.1") for j in 1:n]
            u = [0.1sin(j) for j in eachindex(names)]
            sampler = prepare_cell_sampler(built, bound, :y, u;
                locality = :error, backend = AutoEnzyme(; mode = Enzyme.Reverse))
            for cell in (1, 3, 6, 2n+2), point in (u, u .+ 0.13)
                group = data.effect_rows[cell]
                zidx = cell <= 2 ? nothing :
                    zi[data.coordinates[group][data.slot[cell-2]]]
                lx = zidx === nothing ? log(data.observed[cell]) : point[zidx]
                effect = exp(point[tau]) * point[effects_i[group]]
                mu = point[bi[1]] + point[bi[2]] * lx + effect
                expected = logpdf(Normal(mu, 1), data.y[cell])
                traced = RecordingPoint(copy(point), Int[])
                @test q(traced, cell) ≈ expected
                read = zidx === nothing ? Int[] : zi[data.coordinates[group]]
                @test sort!(unique!(traced.seen)) == sort([read; tau; effects_i[group]; bi])
                gradient = zeros(length(point))
                value, _ = cell_value_and_gradient!(sampler, gradient, point, cell)
                reference = zeros(length(point))
                residual = data.y[cell] - mu
                reference[bi[1]] = residual
                reference[bi[2]] = residual * lx
                reference[tau] = residual * effect
                reference[effects_i[group]] = residual * exp(point[tau])
                zidx === nothing || (reference[zidx] = residual * point[bi[2]])
                @test value ≈ expected
                @test gradient ≈ reference
            end
            q(u, 3)
            push!(costs, (minimum(@allocated(q(u, 3)) for _ in 1:3), nodes(code_expr(q.kernel))))
            @info "tuple geometry cell" computed ragged groups=n allocated_bytes=costs[end][1] generated_nodes=costs[end][2]
            @test geometry_calls[] == (computed ? 1 : 0)
            @test data == saved
            @test_throws BoundsError q(u, 2n+3)
        end
        @test costs[1] == costs[2]
    end
end
end
