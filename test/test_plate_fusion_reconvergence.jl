using ReactiveKernels, Test
isdefined(@__MODULE__, :AuthoredPlateDiamonds) || include("fixtures/authored_plate_diamonds.jl")

plate_outputs(p) = Set(v.name for r in p.recipes for v in r.outputs)

@testset "Plate fusion after downstream reconvergence" begin
    D = AuthoredPlateDiamonds
    RK = ReactiveKernels
    q = [0.3, -0.1]
    sizes = Int[]
    for depth in (1, 3, 6), swapped in (false, true)
        spec = D.spec(depth, swapped)
        p = plan(spec)
        original = copy(p.recipes)
        lowered = RK._fuse_authored_plate_chains(p)
        @test RK._fuse_authored_plate_chains(lowered) === lowered
        @test count(r -> r.op isa RK._AuthoredPlateOp, lowered.recipes) == 1
        cell = only(r.op.kernel for r in lowered.recipes if r.op isa RK._AuthoredPlateOp)
        # The predecessor DAG is copied once, despite its three branch uses.
        @test length(cell.plan.recipes) == 1 + 3depth
        k = prepare(p)
        @test k.plan === p
        @test p.recipes == original
        push!(sizes, D.nodes(code_expr(k)))
        bound_sizes = Int[]
        for n in (0, 1, 13, 1200)
            x = [0.2cos(i) for i in 1:n]
            expected, _ = D.reference(q, x, depth)
            @test k(q, x) ≈ expected atol = 1e-12
            D.allocated(k, q, x)
            @test D.allocated(k, q, x) == 0
            bound = prepare(spec; bound = (; x))
            @test bound(q) ≈ expected atol = 1e-12
            push!(bound_sizes, D.nodes(code_expr(bound)))
        end
        @test allequal(bound_sizes)
    end
    # Fixed graph-size growth, independent of the 0..1200 runtime domain above.
    @test sizes[2] == sizes[1]
    @test sizes[4] == sizes[3]
    @test sizes[6] == sizes[5]
    @test sizes[5] <= 2 * sizes[3]
end

@testset "Reconvergent plate query and ownership boundaries" begin
    D = AuthoredPlateDiamonds
    RK = ReactiveKernels
    spec = D.spec(1)
    q = [0.3, -0.1]
    for n in (0, 1, 13)
        x = [0.2cos(i) for i in 1:n]
        base = sum(q) .* x
        expected, _ = D.reference(q, x, 1)
        query = prepare(spec; want = (:base, :total))
        values, total = query(q, x)
        @test values == base
        @test total ≈ expected atol = 1e-12
        cut = prepare(spec; have = (:base,), want = :total)
        @test cut(base) ≈ expected atol = 1e-12
        shared = plan(spec; want = (:total, :extra, :opaque))
        @test :base in plate_outputs(RK._fuse_authored_plate_chains(shared))
        @test collect(prepare(shared)(q, x)) ≈ [expected, sum(base), sum(abs2, base)]
        atomic = prepare(D.atomic)
        @test atomic(q, x) ≈ sum(base .+ sin.(base)) + n * sum(base) atol = 1e-12
        @test :base in plate_outputs(RK._fuse_authored_plate_chains(plan(D.atomic)))
        independent = prepare(D.independent)
        @test independent(q, x) ≈ sum(sin, base; init = 0.0) + sum(cos, base; init = 0.0)
        @test :base in plate_outputs(RK._fuse_authored_plate_chains(plan(D.independent)))
    end
    matrix = prepare(D.multidimensional)
    for (m, n) in ((0, 2), (1, 1), (3, 2))
        x, z = reshape(collect(1.0:m), m, 1), reshape(collect(1.0:n), 1, n)
        base = (x .+ z) .* sum(q)
        @test matrix(q, x, z) ≈ sum(base .+ sin.(base))
    end
    @test_throws DimensionMismatch matrix(q, ones(2), ones(3))
    @test_throws ArgumentError matrix(q, 1.0, 2.0)
    checked = prepare(D.checked_diamond)
    @test checked(Float64[]) == 0.0
    @test checked([0.0, 1.0, 2.0]) ≈ log(2) + sin(log(2))
    @test_throws DomainError checked([-1.0])
    expanding = prepare(D.expanding_diamond)
    @test_throws DomainError expanding([-1.0], [0.0])
    @test expanding([1.0], Float64[]) == 0.0
    @test expanding([2.0], [0.1, 0.2]) ≈ 2 * (log(2) + sin(log(2))) + 0.3
    @test_throws DimensionMismatch expanding([1.0, 2.0], [1.0, 2.0, 3.0])
end
