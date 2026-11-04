module QueryFootprintTests
using ReactiveKernels, ReactiveKernelsPPL, LinearAlgebra, SparseArrays, Test
using DifferentiationInterface: AutoEnzyme
import Enzyme

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

# A group effect with one correlation factor per stratum. Its factors' coordinates
# are read by one stacked edge per pair, inside a broadcast, and that one read must
# not make the cells report every subject's coordinates.
@rkppl _qf_stratified(g, s) = begin
    tau[levels(s), 1:2] .~ restricted.(Normal(0, 1), 0.0, Inf)
    @plate for k = levels(s)
        L[k] ~ LKJCholesky(2, 1.0)
    end
    z[levels(g), 1:2] .~ Normal.(0, 1)
    @plate for i = eachindex(g)
        b[i, 1:2] = (tau[s[i], :] .* L[s[i]]) * z[g[i], :]
    end
    return b
end
@rkppl _qf_population(X, n, loc, scale) = begin
    beta[1:n] .~ Normal.(loc, scale)
    return X * beta
end

@testset "a stratified group effect reads its own subject's coordinates" begin
    subject = repeat(["a", "b", "c", "d", "e", "f"], inner = 3)
    arm = repeat(["r", "r", "p", "q", "q", "p"], inner = 3)
    data = (; subject, arm, x = collect(range(-1, 1; length = 18)), y = 0.3 .* sin.(1:18))
    plan, built = build(quote
        effect ~ _qf_stratified(subject, arm)
        ranef = effect[:, 1] .+ effect[:, 2] .* x
        X = hcat(ones(length(subject)), x)
        pop ~ _qf_population(X, 2, 0.0, 1.0)
        mu = pop .+ ranef
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end, data)
    names = coordinate_names(built.layout)
    u = [0.2 * sin(1.3k) for k in 1:built.layout.total]
    cell = prepare_cell_query(built, plan, :y; locality = :error)
    footprint = partition_footprint_kernel(cell.kernel)
    pointwise = Base.invokelatest(prepare_query(built, plan, :pointwise), u)
    levels_of(v) = sort(unique(v))
    for i in eachindex(subject)
        @test cell(u, i) ≈ pointwise.y[i]
        s, g = findfirst(==(arm[i]), levels_of(arm)), findfirst(==(subject[i]), levels_of(subject))
        read = names[Base.invokelatest(footprint, u, i).unconstrained]
        @test Set(read) == Set([:sigma, Symbol("pop.beta.1"), Symbol("pop.beta.2"),
            Symbol("effect.tau.$s.1"), Symbol("effect.tau.$s.2"),
            Symbol("effect.L.1.1"), Symbol("effect.L.1.2"), Symbol("effect.L.1.3"),
            Symbol("effect.z.$g.1"), Symbol("effect.z.$g.2")])
    end
    # The gradient stays within the footprint, whose coordinates it matches.
    backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    sampler = prepare_cell_sampler(built, plan, :y, u; backend)
    gradient = similar(u)
    cell_value_and_gradient!(sampler, gradient, u, 7)
    read = Base.invokelatest(footprint, u, 7).unconstrained
    @test all(j -> j in read || iszero(gradient[j]), eachindex(u))

    # Its shared factors move to the shared prefix, leaving each unit its own z.
    baseline = prepare_partition_sampler(built, plan, :y, u; by = :subject, backend,
        compact = false)
    split = prepare_partition_sampler(built, plan, :y, u; by = :subject, backend,
        compact = false, partitioned = Symbol("effect.z"))
    for (position, label) in enumerate(levels_of(subject))
        local_read = names[partition_footprint(split, label; shared = false)]
        @test Set(local_read) == Set([Symbol("effect.z.$position.1"), Symbol("effect.z.$position.2")])
        expected, reference = similar(u), similar(u)
        value, _ = partition_value_and_gradient!(baseline, reference, u, label)
        got, _ = partition_value_and_gradient!(split, expected, u, label)
        @test got ≈ value
        @test expected ≈ reference
    end
    @test data.subject == subject
end

# A walk whose innovations are correlated within a unit and independent across.
@rkppl _qf_walk(group, step, decay, L, P, W) = begin
    sigma ~ Exponential(2.0)
    eta[1:P, 1:W] .~ Normal.(0, 1)
    correlated = L * eta
    steps = correlated * decay
    walk = sigma .* steps
    cell = (step .- 1) .* P .+ group
    return walk[cell]
end

@rkppl _qf_walk_definition(group, step, decay, values, P, W) = begin
    sigma ~ Exponential(2.0)
    eta[1:P, 1:W] .~ Normal.(0, 1)
    L = sparse(reshape(values, P, P))
    correlated = L * eta
    steps = correlated * decay
    walk = sigma .* steps
    cell = (step .- 1) .* P .+ group
    return walk[cell]
end

@testset "a bound sparse factor keeps a unit's reads to its own rows" begin
    P, W = 6, 3
    within(c) = [1.0 c; c 1.0]
    C = cat(within(0.5), within(0.3), within(0.2); dims = (1, 2))
    factor = Matrix(cholesky(Symmetric(C)).L)
    group, step = repeat(1:P, inner = W), repeat(1:W, outer = P)
    decay = [1.0 0.5 0.25; 0.0 1.0 0.5; 0.0 0.0 1.0]
    columns(L) = (; group, step, decay, L, unit = cld.(group, 2), y = 0.3 .* sin.(eachindex(group)))
    ast = quote
        w ~ _qf_walk(group, step, decay, L, $P, $W)
        a ~ Normal(0, 1)
        mu = a .+ w
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
    fixtures = map((factor, sparse(factor))) do L
        plan, built = build(ast, columns(L))
        (; plan, built, cell = prepare_cell_query(built, plan, :y; locality = :error),
         partition = prepare_partition(built, plan, :y; by = :unit, locality = :error))
    end
    dense, sparse_case = fixtures
    names = coordinate_names(sparse_case.built.layout)
    @test names == coordinate_names(dense.built.layout)
    u = [0.2 * sin(1.3k) for k in 1:length(names)]
    shared = Set([Symbol("w.sigma"), :a, :sigma])
    own(rows) = Set(Symbol("w.eta.$r.$t") for r in rows for t in 1:W)
    for i in eachindex(group)
        @test sparse_case.cell(u, i) ≈ dense.cell(u, i)
        stored = findall(!iszero, factor[group[i], :])
        @test Set(names[partition_footprint(sparse_case.cell.kernel, u, i).unconstrained]) ==
            union(shared, own(stored))
        # A dense factor declares no structure, so its zeros remove no read.
        @test Set(names[partition_footprint(dense.cell.kernel, u, i).unconstrained]) ==
            union(shared, own(1:P))
    end
    # So does a sparse factor the model computes from data.
    definition_plan, definition_built = build(quote
        w ~ _qf_walk_definition(group, step, decay, values, $P, $W)
        a ~ Normal(0, 1)
        mu = a .+ w
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end, (; group, step, decay, values = vec(factor), y = columns(factor).y))
    @test coordinate_names(definition_built.layout) == names
    definition = prepare_cell_query(definition_built, definition_plan, :y; locality = :error)
    for i in (1, 2, 8)
        @test definition(u, i) ≈ dense.cell(u, i)
        @test Set(names[partition_footprint(definition.kernel, u, i).unconstrained]) ==
            union(shared, own(findall(!iszero, factor[group[i], :])))
    end
    backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    samplers = map(fixtures) do f
        prepare_partition_sampler(f.built, f.plan, :y, u; by = :unit, backend, compact = false)
    end
    compact = prepare_partition_sampler(sparse_case.built, sparse_case.plan, :y, u; by = :unit, backend)
    for label in 1:3
        rows = (2label - 1):(2label)
        expected = union(shared, own(rows))
        read = partition_footprint(sparse_case.partition.kernel, u, label, nothing).unconstrained
        @test Set(names[read]) == expected
        planned = partition_footprint(sparse_case.partition.kernel, u, label,
            sparse_case.partition.plans[label]).unconstrained
        @test planned == read
        @test sparse_case.partition(u, label) ≈ dense.partition(u, label)
        dense_gradient, sparse_gradient = similar(u), similar(u)
        dense_value, _ = partition_value_and_gradient!(samplers[1], dense_gradient, u, label)
        sparse_value, _ = partition_value_and_gradient!(samplers[2], sparse_gradient, u, label)
        @test sparse_value ≈ dense_value
        @test sparse_gradient ≈ dense_gradient
        @test all(j -> j in read || iszero(sparse_gradient[j]), eachindex(u))
        positions = partition_footprint(compact, label)
        compact_gradient = zeros(length(positions))
        compact_value, _ = partition_value_and_gradient!(compact, compact_gradient, u, label)
        @test compact_value ≈ sparse_value
        @test compact_gradient ≈ sparse_gradient[positions]
    end
end
end
