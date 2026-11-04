using ReactiveKernelsPPL, Test

module _ParameterMetadataFixtures
using ReactiveKernelsPPL
const forbid = Ref(false)
check() = forbid[] && error("parameter inspection evaluated caller code")
extent(n) = (check(); n)
struct Block
    n::Int
end
ReactiveKernelsPPL.sampling_logdensity(d::Block, x) = (check(); -sum(abs2, x))
ReactiveKernelsPPL.sampling_geometry(::Type{Block}, args, shape) =
    ParameterGeometry((args[1], 2), args[1]; support = :equal_columns,
        constrain = (u, shape, args...) -> (check(); reshape(vcat(u, u), shape)),
        unconstrain = (x, shape, args...) -> (check(); x[:, 1]),
        logjac = (u, shape, args...) -> (check(); 0.0))
struct Scalar end
ReactiveKernelsPPL.sampling_logdensity(::Scalar, x) = (check(); -abs2(x))
ReactiveKernelsPPL.sampling_geometry(::Type{Scalar}, args, shape) =
    ParameterGeometry(shape, 1; support = :real,
        constrain = (u, shape) -> (check(); only(u)),
        unconstrain = (x, shape) -> (check(); [x]),
        logjac = (u, shape) -> (check(); 0.0))
end

@rkppl _metadata_child() = begin
    a ~ Normal(0, 1)
    b[1:2] .~ Normal.(0, 1)
    return a
end

_metadata_build(ast, data = (;); kwargs...) =
    build_kernel(bind_data(lower_rkppl(ast, data; mod = @__MODULE__, kwargs...), data))

function _metadata_leaves(nt, path = ())
    out = Pair{Tuple,Any}[]
    for (name, value) in pairs(nt)
        p = (path..., name)
        value isa NamedTuple ? append!(out, _metadata_leaves(value, p)) : push!(out, p => value)
    end
    return out
end

function _metadata_check(layout)
    records = parameter_metadata(layout)
    leaves = _metadata_leaves(constrain(layout, zeros(layout.total)))
    @test [r.path for r in records] == first.(leaves)
    @test sum((r.ncoordinates for r in records); init = 0) == layout.total
    for (r, (_, value)) in zip(records, leaves)
        @test (r.rank, r.shape) == (ndims(value), size(value))
        @test r.element_shapes === nothing || r.element_shapes == size.(value)
    end
    return Dict(r.path => r for r in records)
end

@testset "parameter metadata: existing builds, authored names and constrained shapes" begin
    built = _metadata_build(quote
        s ~ Normal(0, 1)
        v[1:4] .~ Normal.(0, 1)
        B[1:2, 1:3] .~ Normal.(0, 1)
        empty[1:0] .~ Normal.(0, 1)
        empty_matrix[1:0, 1:3] .~ Normal.(0, 1)
        p ~ Dirichlet([1.0, 2.0, 3.0])
        one ~ Dirichlet([1.0])
        L ~ LKJCholesky(3, 2.0)
        L1 ~ LKJCholesky(1, 2.0)
        eachrow(P[1:2, 1:3]) .~ Dirichlet([1.0, 2.0, 3.0])
        sub ~ _metadata_child()
        var"sub.a" ~ Normal(0, 1)
        derived = s + 1
        y ~ Normal(s, 1)
    end, (; y = 0.5); conditioned = (:y,))
    @test parameter_metadata(built) == parameter_metadata(built.layout)
    bypath = _metadata_check(built.layout)
    expected = Dict(:s => ((), 1), :v => ((4,), 4), :B => ((2, 3), 6),
        :empty => ((0,), 0), :empty_matrix => ((0, 3), 0), :p => ((3,), 2),
        :one => ((1,), 0), :L => ((3, 3), 3), :L1 => ((1, 1), 0), :P => ((2, 3), 4))
    for (name, (shape, n)) in expected
        @test (bypath[(name,)].shape, bypath[(name,)].ncoordinates) == (shape, n)
        @test bypath[(name,)].element_shapes === nothing
    end
    @test bypath[(:sub, :a)].shape == ()
    @test bypath[(:sub, :b)].shape == (2,)
    @test haskey(bypath, (Symbol("sub.a"),))
    @test !haskey(bypath, (:derived,)) && !haskey(bypath, (:y,))
    @test isempty(parameter_metadata(LayoutTable(LayoutEntry[], 0)))
end

@testset "parameter metadata: ragged values and caller-owned geometry without execution" begin
    ast = quote
        k = _ParameterMetadataFixtures.extent(n)
        v[1:k] .~ Normal.(0, 1)
        block[1:n, 1:2] ~ _ParameterMetadataFixtures.Block(n)
        scalar ~ _ParameterMetadataFixtures.Scalar()
        @plate for i in eachindex(m)
            B[i] ~ _ParameterMetadataFixtures.Block(m[i])
            c[i] ~ _ParameterMetadataFixtures.Scalar()
            z[i] .~ Normal.(mu[i], 1)
        end
    end
    snapshots = []
    for m in ([2, 0, 3], Int[])
        mu = [zeros(k, 2) for k in m]
        built = _metadata_build(ast, (; n = 2, m, mu))
        records = parameter_metadata(built)
        push!(snapshots, (built, records))
        _ParameterMetadataFixtures.forbid[] = true
        try
            @test parameter_metadata(built) == records
            @test parameter_metadata(built.layout) == records
        finally
            _ParameterMetadataFixtures.forbid[] = false
        end
        bypath = _metadata_check(built.layout)
        @test (bypath[(:block,)].shape, bypath[(:block,)].ncoordinates) == ((2, 2), 2)
        @test (bypath[(:scalar,)].shape, bypath[(:scalar,)].ncoordinates) == ((), 1)
        @test (bypath[(:c,)].shape, bypath[(:c,)].ncoordinates) == ((length(m),), length(m))
        @test bypath[(:c,)].element_shapes === nothing
        for name in (:B, :z)
            record = bypath[(name,)]
            @test record.rank == 1 && record.shape == (length(m),)
            @test record.element_shapes == [(k, 2) for k in m]
            @test record.ncoordinates == (name === :B ? 1 : 2) * sum(m)
            # Editing a returned container cannot corrupt the built layout.
            push!(record.element_shapes, (99,))
        end
        empty!(parameter_metadata(built))
        @test parameter_metadata(built) == parameter_metadata(built.layout)
        @test only(r for r in parameter_metadata(built) if r.path == (:B,)).element_shapes == [(k, 2) for k in m]
    end
    # Another binding/build cannot silently change an earlier build's shapes.
    @test all(parameter_metadata(built) == saved for (built, saved) in snapshots)
end

@testset "parameter metadata: legacy layout families, split coefficients and namespace order" begin
    entries = LayoutEntry[
        LayoutEntry(:coefficient, :beta, :beta_first, [:b1], 1, 1, :identity),
        LayoutEntry(:sampled, nothing, :s, [:s], 2, 1, :identity),
        LayoutEntry(:coefficient, :beta, :beta_last, [:b2, :b3], 3, 2, :identity),
        LayoutEntry(:plate, nothing, :plate, [:p1, :p2], 5, 2, :identity),
        LayoutEntry(:glm, nothing, :glm, [:g1], 7, 1, :identity),
        LayoutEntry(:scan, nothing, :scan, [:t1, :t2], 8, 2, :identity),
        LayoutEntry(:vector, nothing, :simplex, [:q1, :q2], 10, 2, :simplex),
        LayoutEntry(:vector, nothing, :ordered, Symbol[], 12, 0, :ordered),
        LayoutEntry(:cholesky_corr, nothing, :L, [:l1], 12, 1, :lkj),
        LayoutEntry(:array, nothing, :stack, [:l2, :l3], 13, 2, :lkj_stack,
            NaN, NaN, [2, 2, 2]),
    ]
    layout = LayoutTable(entries, 14, Dict{Symbol,Tuple{Vararg{Symbol}}}(
        :beta => (:a, :beta), :s => (:b, :s), :plate => (:a, :plate)))
    bypath = _metadata_check(layout)
    @test bypath[(:a, :beta)].shape == (3,)
    @test bypath[(:a, :beta)].ncoordinates == 3
    @test [r.path for r in parameter_metadata(layout)][1:3] == [(:a, :beta), (:a, :plate), (:b, :s)]
    @test bypath[(:stack,)].shape == (2, 2, 2)
    @test bypath[(:stack,)].ncoordinates == 2

    # A cell table can contain scalar, empty and matrix cells together.
    cell = LayoutEntry(:array, nothing, :packed, Symbol[], 1, 5, :identity,
        NaN, NaN, [5])
    cells = LayoutTable([cell], 5,
        Dict{Symbol,Tuple{Vararg{Symbol}}}(:packed => (:cells,)),
        Dict{Symbol,Any}(:shapes => [(), (0,), (2, 2)]), Dict{Symbol,Any}(),
        Dict(:packed => (:cells, :shapes)))
    record = _metadata_check(cells)[(:cells,)]
    @test record.shape == (3,) && record.rank == 1
    @test record.element_shapes == [(), (0,), (2, 2)]
    @test record.ncoordinates == 5

    # Dynamic bounds are irrelevant to metadata; they need not be evaluated
    # or even have a supplied value in a standalone layout.
    e = LayoutEntry(:sampled, nothing, :q, [:q], 1, 1, :interval, :lo, :hi)
    bound_layout = LayoutTable([e], 1, Dict{Symbol,Tuple{Vararg{Symbol}}}(),
        Dict{Symbol,Any}(), Dict{Symbol,Any}(:lo => :(error("called lower bound")),
            :hi => :(error("called upper bound"))))
    @test only(parameter_metadata(bound_layout)) ==
        (; path = (:q,), rank = 0, shape = (), element_shapes = nothing, ncoordinates = 1)
end
