# Phase 3: structural CSE, composition, plan caching.
using ReactiveKernels
using Test

@testset "Phase 3 — structural CSE (opt-in)" begin
    @testset "same cse_key + same inputs -> single computation" begin
        g = Graph()
        x = value!(g, :x, Float64); y = value!(g, :y, Float64)
        a1 = value!(g, :a1, Float64); a2 = value!(g, :a2, Float64)
        s = value!(g, :s, Float64)
        f(x, y) = x + y
        add!(g, (x, y) => a1, f; cse_key = :f)
        add!(g, (x, y) => a2, f; cse_key = :f)   # structurally identical -> aliased
        add!(g, (a1, a2) => s, (a, b) -> a + b)

        # a1 and a2 canonicalize to the same node
        @test canon_id(g, a2.id) == canon_id(g, a1.id)
        p = plan(g; have = (x, y), want = (s,))
        # only ONE `f` application survives + the sum
        @test count(r -> r.cse_key === :f, p.recipes) == 1
        ast = string(code_expr(p))
        @test count(m -> true, collect(eachmatch(r"__ops__\[", ast))) == 2  # f once, sum once
        k = prepare(p)
        @test k(2.0, 3.0) == 10.0        # a1=a2=5, s=10
    end

    @testset "without cse_key -> two independent computations" begin
        g = Graph()
        x = value!(g, :x, Float64); y = value!(g, :y, Float64)
        a1 = value!(g, :a1, Float64); a2 = value!(g, :a2, Float64)
        s = value!(g, :s, Float64)
        f(x, y) = x + y
        add!(g, (x, y) => a1, f)          # no cse_key
        add!(g, (x, y) => a2, f)
        add!(g, (a1, a2) => s, (a, b) -> a + b)
        @test canon_id(g, a2.id) != canon_id(g, a1.id)
        p = plan(g; have = (x, y), want = (s,))
        @test count(r -> r.op === f, p.recipes) == 2
    end
end

@testset "Phase 3 — composition (global value identity)" begin
    # Fragment 1: standardize x -> z.  Fragment 2: z -> out.
    x = value(:x, Float64); z = value(:z, Float64)
    g1 = Graph(); add!(g1, x => z, x -> x / 2)

    out = value(:out, Float64)
    g2 = Graph(); add!(g2, z => out, z -> z + 1.0)

    g = compose(g1, g2)
    k = prepare(g; have = (x,), want = (out,))
    @test k(10.0) == 6.0                  # z=5, out=6
    # z is the same identity across fragments -> single chain
    p = plan(g; have = (x,), want = (out,))
    @test length(p.recipes) == 2
end

@testset "Phase 3 — plan caching" begin
    g = Graph()
    x = value!(g, :x, Float64); a = value!(g, :a, Float64); b = value!(g, :b, Float64)
    add!(g, x => a, x -> x + 1.0)
    add!(g, a => b, a -> 2a)

    cache = PreparationCache()
    k1 = prepare!(cache, g; have = (x,), want = (b,))
    k2 = prepare!(cache, g; have = (x,), want = (b,))
    @test k1 === k2                       # reused
    @test length(cache) == 1
    @test k1(1.0) == 4.0

    # different signature -> new entry
    ka = prepare!(cache, g; have = (x,), want = (a,))
    @test length(cache) == 2
    @test ka(1.0) == 2.0

    # graph mutation bumps version -> old cache entry not reused
    add!(g, x => value!(g, :c, Float64), x -> x)
    k3 = prepare!(cache, g; have = (x,), want = (b,))
    @test k3 !== k1
    @test k3(1.0) == 4.0
end
