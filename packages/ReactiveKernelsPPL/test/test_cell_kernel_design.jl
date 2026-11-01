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

const layout_calls = Ref(0)
function entry_layout(ends)
    layout_calls[] += 1
    (collect(eachindex(ends)), collect(1:last(ends)))
end
@kernel entry_axis(ends) = entry_layout(ends)[2]
@kernel group_axis(ends) = entry_layout(ends)[1]

function computed_model(kind)
    source = deepcopy(MODEL)
    kind === :supplied && return source
    defs = Expr[:(computed_entries = entry_axis(ends))]
    entry = :computed_entries
    if kind === :alias
        push!(defs, :(entry_alias = identity(computed_entries)))
        entry = :entry_alias
    end
    group = :groups
    if kind === :both
        push!(defs, :(computed_groups = group_axis(ends)))
        group = :computed_groups
    end
    i = findfirst(ex -> ex isa Expr && ex.head === :(=) &&
        ex.args[1] === :values, source.args)
    source.args[i] = :(values = grouped_column(z, $group, coordinates, $entry, owner, slot))
    prepend!(source.args, defs)
    return source
end

function fixture(n, ragged)
    sizes = ragged ? repeat([0, 1, 5], n ÷ 3) : fill(2, n)
    ends = cumsum(sizes)
    starts = ends .- sizes .+ 1
    coordinates = [collect(a:b) for (a, b) in zip(starts, ends)]
    owner = reduce(vcat, [fill(g, sizes[g]) for g in 1:n])
    slot = reduce(vcat, [collect(k:-1:1) for k in sizes])
    (; packed = zeros(2n), groups = collect(1:n), ends, coordinates,
        entries = collect(1:2n), owner, slot, one = ones(2n),
        offset = fill(0.1, 2n), y = fill(0.7, 2n))
end

nodes(ex) = ex isa Expr ? 1 + sum(nodes, ex.args; init = 0) : 1

@testset "a grouped kernel column stays local through a design and offset" begin
    builds = Dict{Tuple{Symbol,Int},Any}()
    for kind in (:supplied, :entries, :both, :alias), ragged in (false, true)
        costs = Tuple{Int,Int}[]
        for n in (3, 300)
            data = fixture(n, ragged)
            saved = deepcopy(data)
            layout_calls[] = 0
            source = computed_model(kind)
            plan = lower_rkppl(source, data; conditioned = (:y,), mod = @__MODULE__)
            @test layout_calls[] == 0
            bound = bind_data(plan, data)
            expected_calls = kind === :supplied ? 0 : kind === :both ? 2 : 1
            @test layout_calls[] == expected_calls
            built = build_kernel(bound)
            q = prepare_cell_query(built, bound, :y; locality = :error)
            reused = prepare_cell_query(get(builds, (kind, n), built), bound, :y; locality = :error)
            builds[(kind, n)] = built
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
                @test reused(u, cell) ≈ expected
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
            @info "grouped design cell" kind ragged groups=n allocated_bytes=costs[end][1] generated_nodes=costs[end][2]
            @test layout_calls[] == expected_calls
            @test data == saved
            @test_throws BoundsError q(u, 2n + 1)
        end
        @test costs[1] == costs[2]
    end
end

const coordinate_calls = Ref(0)
copy_coordinates(x) = (coordinate_calls[] += 1; map(copy, x))

@testset "data-only captures keep preparation staging" begin
    for inline in (false, true)
        data = fixture(3, true)
        source = computed_model(:entries)
        inline || pushfirst!(source.args, :(computed_coordinates = copy_coordinates(coordinates)))
        i = findfirst(ex -> ex isa Expr && ex.head === :(=) &&
            ex.args[1] === :values, source.args)
        coords = inline ? :(copy_coordinates(coordinates)) : :computed_coordinates
        source.args[i] = :(values = grouped_column(z, groups, $coords,
            computed_entries, owner, slot))
        coordinate_calls[] = 0
        bound = bind_data(lower_rkppl(source, data; conditioned = (:y,),
            mod = @__MODULE__), data)
        built = build_kernel(bound)
        @test coordinate_calls[] == 0
        q = prepare_cell_query(built, bound, :y; locality = :error)
        @test coordinate_calls[] == 1
        u = fill(0.1, length(coordinate_names(built.layout)))
        @test isfinite(q(u, 1))
        @test isfinite(q(u, 6))
        @test coordinate_calls[] == 1
    end
end

const bound_calls = Ref(0)
lower_bounds(x) = (bound_calls[] += 1; max.(x, 0.0))
@kernel completed(observed, missing, order) = vcat(observed, missing)[order]

const DATA_METADATA_MODEL = quote
    z[axes(lo, 1)] .~ Normal.(0, 1)
    beta[1:2] .~ Normal.(0, 1)
    free = lower_bounds(lo) .+ exp.(z)
    values = completed(observed, free, order)
    lx = log.(values)
    X = hcat(one, lx)
    mu = X * beta .+ offset
    y .~ Normal.(mu, 1)
end

function data_metadata_fixture(n, empty)
    lo = empty ? Float64[] : [isodd(i) ? -0.2 : 0.3 for i in 1:n]
    observed = fill(2.0, n)
    order = empty ? collect(n:-1:1) : vec(permutedims(hcat(n+1:2n, 1:n)))
    (; lo, observed, order, one = ones(length(order)),
        offset = fill(0.1, length(order)), y = fill(0.7, length(order)))
end

@testset "nested data calls provide metadata through sampled compositions" begin
    for empty in (true, false)
        bound_calls[] = 0
        data = data_metadata_fixture(3, empty)
        plan = lower_rkppl(DATA_METADATA_MODEL, data; conditioned = (:y,), mod = @__MODULE__)
        original = deepcopy([(a.name, a.expr, a.label) for a in plan.assignments])
        @test bound_calls[] == 0
        costs = Tuple{Int,Int}[]
        for (k, n) in enumerate((3, 300))
            data = data_metadata_fixture(n, empty)
            saved = deepcopy(data)
            check = ReactiveKernelsPPL._STRUCTURAL_CACHE_CHECK[]
            bound = try
                ReactiveKernelsPPL._STRUCTURAL_CACHE_CHECK[] = true
                bind_data(plan, data)
            finally
                ReactiveKernelsPPL._STRUCTURAL_CACHE_CHECK[] = check
            end
            @test bound_calls[] == k
            @test [(a.name, a.expr, a.label) for a in plan.assignments] == original
            meta = ReactiveKernelsPPL._value_metadata(bound, :values, Set{Symbol}(); data_axes = true)
            @test ReactiveKernels.OutputSignatures.value_axes(meta) == axes(data.y)
            built = build_kernel(bound)
            q = prepare_cell_query(built, bound, :y; locality = :error)
            names = coordinate_names(built.layout)
            zi = [only(findall(==(Symbol("z.$i")), names)) for i in eachindex(data.lo)]
            bi = [only(findall(==(Symbol("beta.$i")), names)) for i in 1:2]
            u = [0.1sin(i) for i in eachindex(names)]
            sampler = prepare_cell_sampler(built, bound, :y, u;
                locality = :error, backend = AutoEnzyme(; mode = Enzyme.Reverse))
            for cell in unique([1, 2, length(data.y)])
                pos = data.order[cell]
                zidx = pos <= n ? nothing : zi[pos - n]
                value = pos <= n ? data.observed[pos] :
                    max(data.lo[pos - n], 0.0) + exp(u[zidx])
                mu = u[bi[1]] + u[bi[2]] * log(value) + data.offset[cell]
                expected = logpdf(Normal(mu, 1), data.y[cell])
                point = RecordingPoint(copy(u), Int[])
                @test q(point, cell) ≈ expected
                wanted = zidx === nothing ? bi : [bi; zidx]
                @test sort!(unique!(point.seen)) == sort(wanted)
                gradient = zeros(length(u))
                got, _ = cell_value_and_gradient!(sampler, gradient, u, cell)
                reference = zeros(length(u))
                residual = data.y[cell] - mu
                reference[bi] = [residual, residual * log(value)]
                zidx === nothing || (reference[zidx] = residual * u[bi[2]] * exp(u[zidx]) / value)
                @test got ≈ expected
                @test gradient ≈ reference
            end
            q(u, 1)
            push!(costs, (minimum(@allocated(q(u, 1)) for _ in 1:3), nodes(code_expr(q.kernel))))
            @test bound_calls[] == k
            @test data == saved
        end
        @test costs[1] == costs[2]
    end
end
end
