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

function weighted_readout(values, indices, weights)
    total = 0.0
    for j in eachindex(indices, weights)
        total += values[indices[j]] * weights[j]
    end
    total
end

const GUARDED_COLLECTED = quote
    beta[1:3] .~ Normal.(prior_mean, prior_sd)
    values = exp.(beta)
    @plate for i in eachindex(groups)
        readout_cells[i] = weighted_readout(values, groups[i], weights[i])
    end
    readouts = collect_cells(readout_cells)
    @plate for i in eachindex(indices)
        selected_cells[i] = 0.1 + (indices[i] > 0 ? readouts[indices[i]] : 0.0)
    end
    selected = collect_cells(selected_cells)
    y .~ Normal.(selected, 1.0)
end

# Same registered-Enzyme empty-window phi gap as the core exemplar test
# (Enzyme snag ordinary-native-1d9d6afe). Pinned acceptance makes it fatal.
function guarded_static_reverse(f)
    try
        f()
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        get(ENV, "RK_TEST_REQUIRE_STATIC_ENZYME", "0") == "1" && rethrow()
        @test occursin("Mismatched activity", sprint(showerror, err))
        @test_broken false
    end
end

@testset "packed exemplar types cross opaque helpers and guarded collectors" begin
    costs = Tuple{Int,Int,Int}[]
    for n in (16, 1024)
        data = (; prior_mean=zeros(3), prior_sd=ones(3),
            groups=[isodd(i) ? [1, 3] : Int[] for i in 1:n],
            weights=[isodd(i) ? [0.4, 0.6] : Float64[] for i in 1:n],
            indices=[i % 3 == 0 ? 0 : i for i in 1:n], y=fill(0.7, n))
        saved = deepcopy(data)
        bound = bind_data(lower_rkppl(GUARDED_COLLECTED, data;
            conditioned=(:y,), mod=@__MODULE__), data)
        built = build_kernel(bound)
        u, gradient = [0.2, -0.3, 0.4], zeros(3)
        backend = AutoEnzyme(; mode=Enzyme.Reverse)
        q = prepare_cell_query(built, bound, :y; locality=:error)
        cell = prepare_cell_sampler(built, bound, :y, u; backend, locality=:error)
        for shift in (0.0, 0.13)
            x = u .+ shift
            for i in (1, 2, 3, n)
                j = data.indices[i]
                mean = 0.1 + (j == 0 ? 0.0 : weighted_readout(exp.(x), data.groups[j], data.weights[j]))
                expected = logpdf(Normal(mean, 1), data.y[i])
                @test q(x, i) ≈ expected
                @test q(view(x, :), Int32(i)) ≈ expected
                @test q(x, CartesianIndex(i)) ≈ expected
                expected_gradient = zeros(3)
                if j > 0
                    for (k, w) in zip(data.groups[j], data.weights[j])
                        expected_gradient[k] += (data.y[i] - mean) * exp(x[k]) * w
                    end
                end
                guarded_static_reverse() do
                    value, _ = cell_value_and_gradient!(cell, gradient, x, i)
                    @test value ≈ expected
                    @test gradient ≈ expected_gradient
                end
            end
        end
        @test_throws BoundsError q(u, n + 1)
        @test data == saved
        q(u, 1)
        cell_value_and_gradient!(cell, gradient, u, 1)
        push!(costs, (minimum(@allocated(q(u, 1)) for _ in 1:3),
            minimum(@allocated(cell_value_and_gradient!(cell, gradient, u, 1)) for _ in 1:3),
            nodes(code_expr(q.kernel))))
        @info "guarded collector with packed exemplar" n costs=costs[end]
    end
    @test costs[1] == costs[2]
end

# A partition's kernels are lowered from the exemplar and a label, as a cell
# query's are, so they take the collector's scalar arm too: each unit reads
# only its own effects, although the response reads the collected location and
# its scale whole (snag whole-vector-res-64373f39). Without that type evidence
# the location was computed whole, so every unit read every unit's `z`.
const PARTITIONED_COLLECTED = quote
    beta ~ Normal(0, 1)
    z[levels(unit)] .~ Normal.(0, 1)
    s[levels(unit)] .~ Normal.(0, 0.5)
    @plate for i in eachindex(x)
        signal_cells[i] = beta * x[i] + z[unit[i]]
    end
    signal = collect_cells(signal_cells)
    sd = exp.(s[unit])
    y .~ Normal.(signal, sd)
end

@testset "partitions take the collector's typed arm and read their own unit" begin
    unit = [1, 2, 3, 1, 2, 3, 2, 1]
    n = length(unit)
    data = (; unit, name = ["unit $g" for g in unit],
        x = collect(range(-1.0, 1.0; length = n)), y = collect(range(0.3, -0.4; length = n)))
    bound = bind_data(lower_rkppl(PARTITIONED_COLLECTED, data; conditioned=(:y,),
        mod=@__MODULE__), data)
    built = build_kernel(bound)
    names = String.(coordinate_names(built.layout))
    at(prefix) = findall(name -> startswith(name, prefix), names)
    beta, z, s = only(at("beta")), at("z"), at("s")
    u = collect(range(-0.3, 0.4; length = built.layout.total))
    density(u, g) = sum(i -> logpdf(Normal(u[beta] * data.x[i] + u[z[g]], exp(u[s[g]])),
        data.y[i]), findall(==(g), unit))
    function gradient_of(u, g)
        gradient = zeros(length(u))
        for i in findall(==(g), unit)
            sd = exp(u[s[g]])
            residual = data.y[i] - (u[beta] * data.x[i] + u[z[g]])
            gradient[beta] += residual * data.x[i] / sd^2
            gradient[z[g]] += residual / sd^2
            gradient[s[g]] += residual^2 / sd^2 - 1
        end
        gradient
    end
    own(g) = sort([beta, z[g], s[g]])
    full = prepare_query(built, bound, :likelihood)
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for partitioned in ((), (:z, :s))
        q = prepare_partition(built, bound, :y; by=:unit, locality=:error, partitioned)
        @test all(entry -> entry.status !== :whole, partition_locality(q))
        @test sum(g -> q(u, g), partition_labels(q)) ≈ full(u)
        for g in 1:3
            @test q(u, g) ≈ density(u, g)
            @test q(view(u, :), g) ≈ density(u, g)
            @test ReactiveKernels.partition_footprint(q.kernel, u, g, nothing).unconstrained == own(g)
        end
        for compact in (true, false)
            sampler = prepare_partition_sampler(built, bound, :y, u; by=:unit, backend,
                compact, partitioned, locality=:error)
            for g in 1:3
                @test partition_footprint(sampler, g) == own(g)
                gradient = zeros(compact ? 3 : length(u))
                value, _ = partition_value_and_gradient!(sampler, gradient, u, g)
                @test value ≈ density(u, g)
                @test gradient ≈ (compact ? gradient_of(u, g)[own(g)] : gradient_of(u, g))
            end
        end
    end
    # A label of another type lowers the same way.
    labels = ["unit $g" for g in 1:3]
    q = prepare_partition(built, bound, :y; by=:name, locality=:error,
        partitioned=(:z => labels, :s => labels))
    @test ReactiveKernels.partition_footprint(q.kernel, u, "unit 2", nothing).unconstrained == own(2)
    @test q(u, "unit 2") ≈ density(u, 2)
end

@testset "cell queries retain Cartesian positions with exact packed types" begin
    data = (; y=reshape(collect(0.1:0.1:0.6), 2, 3))
    source = quote
        beta ~ Normal(0, 1)
        y .~ Normal.(beta, 1.0)
    end
    bound = bind_data(lower_rkppl(source, data; conditioned=(:y,), mod=@__MODULE__), data)
    q = prepare_cell_query(build_kernel(bound), bound, :y; locality=:error)
    u = [0.2]
    for index in CartesianIndices(data.y)
        expected = logpdf(Normal(u[1], 1), data.y[index])
        @test q(u, index) ≈ expected
        @test q(view(u, :), LinearIndices(data.y)[index]) ≈ expected
        @test q(view(u, :), index) ≈ expected
        @test q(view(u, :), Int32(LinearIndices(data.y)[index])) ≈ expected
    end
    u[1] = 0.4
    @test q(view(u, :), CartesianIndex(2, 3)) ≈ logpdf(Normal(0.4, 1), data.y[2, 3])
    @test_throws BoundsError q(u, CartesianIndex(0, 1))
    @test_throws ArgumentError q.kernel(view(u, :), 1)
end
end
