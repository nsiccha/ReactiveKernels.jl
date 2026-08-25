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

    @testset "typed aliases are atomic and keep producer alternatives" begin
        snapshot(g) = (
            values = copy(g.values),
            recipes = copy(g.recipes),
            producers = Dict(id => copy(rs) for (id, rs) in g.producers),
            aliases = copy(g.aliases),
            version = g.version,
        )

        for (existing_type, new_type) in ((Int, String), (String, Int))
            g = Graph()
            x = value!(g, :x, existing_type)
            existing = value!(g, :existing, existing_type)
            add!(g, x => existing, identity; cse_key = :typed)
            candidate = value(:candidate, new_type)
            before = snapshot(g)
            error = try
                add!(g, x => candidate, identity; cse_key = :typed)
                nothing
            catch err
                err
            end
            @test error isa ArgumentError
            @test occursin("position 1", sprint(showerror, error))
            @test occursin(string(existing_type), sprint(showerror, error))
            @test occursin(string(new_type), sprint(showerror, error))
            @test snapshot(g) == before
            @test !haskey(g.values, candidate.id)
        end

        g = Graph()
        x = value!(g, :x, Int)
        first = value!(g, :first, Int)
        second = value!(g, :second, Float64)
        add!(g; inputs = (x,), outputs = (first, second),
             op = x -> (x, Float64(x)), cse_key = :pair)
        new_first = value(:new_first, Int)
        new_second = value(:new_second, String)
        before = snapshot(g)
        error = try
            add!(g; inputs = (x,), outputs = (new_first, new_second),
                 op = x -> (x, string(x)), cse_key = :pair)
            nothing
        catch err
            err
        end
        @test error isa ArgumentError
        @test occursin("position 2", sprint(showerror, error))
        @test snapshot(g) == before
        @test !haskey(g.values, new_first.id)
        @test !haskey(g.values, new_second.id)

        # A valid alias may merge a value that already has a cheaper producer.
        # Reindex all recipes under the final canonical id so planning retains
        # both alternatives instead of silently selecting only the CSE target.
        g = Graph()
        x = value!(g, :x, Int)
        cheap = value!(g, :cheap, Int)
        target = value!(g, :target, Int)
        add!(g, x => cheap, x -> x + 1; cost = 0, cse_key = :cheap)
        add!(g, x => target, x -> x + 10; cost = 10, cse_key = :target)
        add!(g, x => cheap, x -> x + 10; cost = 10, cse_key = :target)
        @test ReactiveKernels.producers_of(g, cheap.id) == [1, 2]
        selected = plan(g; have = (x,), want = (cheap,))
        @test only(selected.recipes).id == 1
        @test selected.cost == 0
        @test prepare(selected)(2) == 3

        # Repeating the exact same structural output collapses without writing
        # a self-alias, which would make canon_id recurse forever.
        g = Graph()
        x = value!(g, :x, Int)
        y = value!(g, :y, Int)
        add!(g, x => y, identity; cse_key = :same)
        add!(g, x => y, identity; cse_key = :same)
        @test !haskey(g.aliases, y.id)
        @test canon_id(g, y.id) == y.id
        @test length(g.recipes) == 1
    end

    @testset "multi-output alias conflicts reject without mutation" begin
        snapshot(g) = (
            values = copy(g.values),
            recipes = copy(g.recipes),
            producers = Dict(id => copy(rs) for (id, rs) in g.producers),
            aliases = copy(g.aliases),
            version = g.version,
        )

        g = Graph()
        x = value!(g, :x, Int)
        a = value!(g, :a, Int)
        b = value!(g, :b, Int)
        add!(g; inputs = (x,), outputs = (a, b),
             op = x -> (x, x), cse_key = :crossed)
        before = snapshot(g)
        error = try
            add!(g; inputs = (x,), outputs = (b, a),
                 op = x -> (x, x), cse_key = :crossed)
            nothing
        catch err
            err
        end
        @test error isa ArgumentError
        @test occursin("cyclic", sprint(showerror, error))
        @test snapshot(g) == before

        shared = value(:shared, Int)
        before = snapshot(g)
        error = try
            add!(g; inputs = (x,), outputs = (shared, shared),
                 op = x -> (x, x), cse_key = :crossed)
            nothing
        catch err
            err
        end
        @test error isa ArgumentError
        @test occursin("conflicting", sprint(showerror, error))
        @test snapshot(g) == before
        @test !haskey(g.values, shared.id)
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
