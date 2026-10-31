module QueryFootprintTests
using ReactiveKernels, ReactiveKernelsPPL, Test

function build(ast, data)
    plan = bind_data(lower_rkppl(ast, data; conditioned=(:y,), mod=@__MODULE__), data)
    built = build_kernel(plan)
    plan, built
end

function allocations(footprint, u, i)
    Base.invokelatest(footprint, u, i)
    minimum(@allocated(Base.invokelatest(footprint, u, i)) for _ in 1:3)
end

@testset "value-only query footprints" begin
    bytes = Int[]
    for groups in (4, 2_000)
        group = repeat(collect(1:groups), 2)
        data = (; group, x=repeat([0.0, 1.0], groups), y=zeros(2groups),
            unit=cld.(group, 2))
        original = deepcopy(data)
        plan, built = build(quote
            a ~ Normal(0, 1)
            unused ~ Normal(0, 1)
            z[levels(group), 1:2] .~ Normal.(0, 1)
            mu = a .+ z[group, 1] .+ x .* z[group, 2]
            y .~ Normal.(mu, 1)
        end, data)
        u = zeros(built.layout.total)
        saved = copy(u)
        names = coordinate_names(built.layout)
        cell = prepare_cell_query(built, plan, :y; locality=:error)
        footprint = partition_footprint_kernel(cell.kernel)
        # One preparation serves every cell. Zero x does not prune a read.
        for i in (1, 2, groups, 2groups)
            read = Base.invokelatest(footprint, u, i).unconstrained
            @test names[read] == [:a, Symbol("z.$(group[i]).1"), Symbol("z.$(group[i]).2")]
            @test read == partition_footprint(cell.kernel, u, i).unconstrained
            @test read == Base.invokelatest(footprint, u .+ 0.7, i).unconstrained
            @test cell(u, i) ≈ -0.5log(2pi)
        end
        push!(bytes, allocations(footprint, u, 1))

        if groups == 4
            partition = prepare_partition(built, plan, :y; by=:unit)
            for label in partition_labels(partition)
                cells = findall(==(label), data.unit)
                reads = [Base.invokelatest(footprint, u, i).unconstrained for i in cells]
                read = partition_footprint(partition.kernel, u, label, nothing).unconstrained
                @test read == sort!(unique!(reduce(vcat, reads)))
                @test read == partition_footprint(partition.kernel, u, label, partition.plans[label]).unconstrained
                @test partition(u, label) ≈ sum(cell(u, i) for i in cells)

                # A unit has two disjoint two-coordinate group blocks.
                # The union of cell cliques preserves eight entries; the
                # clique of the unit union would have sixteen.
                eta = Set(findall(n -> startswith(string(n), "z."), names))
                pairs = Set((a, b) for r in reads for a in intersect(r, eta) for b in intersect(r, eta))
                @test length(pairs) == 8
                @test length(intersect(read, eta))^2 == 16
            end
        end
        @test u == saved
        @test data == original
    end
    # The result and working storage follow the selected cell, not all groups.
    @test bytes[2] <= bytes[1] + 1024
end

@testset "cell footprints include both retained lazy arms" begin
    data = (; group=[1, 2, 1], y=[0.0, 0.2, -0.1])
    plan, built = build(quote
        a ~ Normal(0, 1)
        unused ~ Normal(0, 1)
        z[levels(group), 1:2] .~ Normal.(0, 1)
        @plate for i in eachindex(y)
            mu = a > 0 ? z[group[i], 1] : z[group[i], 2]
            y[i] ~ Normal(mu, 1)
        end
    end, data)
    cell = prepare_cell_query(built, plan, :y; locality=:error)
    names = coordinate_names(built.layout)
    for sign in (-1.0, 1.0), i in eachindex(data.y)
        u = fill(sign, built.layout.total)
        read = partition_footprint(cell.kernel, u, i).unconstrained
        # The capture analysis currently reads a conditional's array captures
        # whole. Require both arms without pinning that conservative excess:
        # a future more precise analysis may return just this cell's row.
        expected = [:a, Symbol("z.$(data.group[i]).1"), Symbol("z.$(data.group[i]).2")]
        @test issubset(expected, names[read])
        @test :unused ∉ names[read]
        @test cell(u, i) ≈ -0.5log(2pi) - 0.5(data.y[i] - sign)^2
    end
end

@testset "a ragged observation cell is one outer iteration" begin
    # Keep the gather in the model. A linear predictor can lower to a design
    # product that structurally reads every coefficient, including zero weights.
    data = (; group=[1, 2, 1], y=[[0.1, 0.2], [-0.3], [0.4, 0.5, 0.6]])
    original = deepcopy(data)
    plan, built = build(quote
        a ~ Normal(0, 1)
        z[levels(group)] .~ Normal.(0, 1)
        mu = a .+ exp.(z[group])
        @plate for i in eachindex(y)
            y[i] .~ Normal.(mu[i], 1)
        end
    end, data)
    cell = prepare_cell_query(built, plan, :y; locality=:error)
    footprint = partition_footprint_kernel(cell.kernel)
    u = zeros(built.layout.total)
    names = coordinate_names(built.layout)
    for i in eachindex(data.y)
        read = Base.invokelatest(footprint, u, i).unconstrained
        @test names[read] == [:a, Symbol("z.$(data.group[i])")]
        @test cell(u, i) ≈ sum(v -> -0.5log(2pi) - 0.5(v - 1)^2, data.y[i])
    end
    @test data == original
end
end
