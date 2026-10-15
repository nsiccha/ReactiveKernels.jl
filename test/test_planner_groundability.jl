using ReactiveKernels
using Test
using Random

# Independent exhaustive oracle: execute a subset whenever all of its inputs
# are available, then rank complete subsets by cost and recipe count.
function exhaustive_planner_rank(g, have, want)
    best = (Inf, typemax(Int))
    for mask in 0:(2^length(g.recipes) - 1)
        chosen = [r for r in g.recipes if !iszero(mask & (1 << (r.id - 1)))]
        any(r -> r.effectful, chosen) && continue
        available = Set{Int}(canon_id(g, v.id) for v in have)
        pending = copy(chosen)
        while !isempty(pending)
            pos = findfirst(r -> all(v -> canon_id(g, v.id) in available,
                                    r.inputs), pending)
            pos === nothing && break
            r = popat!(pending, pos)
            union!(available, (canon_id(g, v.id) for v in r.outputs))
        end
        isempty(pending) || continue
        all(v -> canon_id(g, v.id) in available, want) || continue
        rank = (sum(r -> r.cost, chosen; init = 0.0), length(chosen))
        best = min(best, rank)
    end
    best
end

function independent_planner_routes(n; inverse_first = false)
    g = Graph()
    x = value!(g, :x, Float64)
    shared = value!(g, :shared, Float64)
    add!(g, x => shared, identity)
    want = Value[]
    for i in 1:n
        a = value!(g, Symbol(:a, i), Float64)
        b = value!(g, Symbol(:b, i), Float64)
        if inverse_first
            add!(g, b => a, identity)
            add!(g, shared => a, identity)
        else
            add!(g, shared => a, identity)
            add!(g, b => a, identity)
        end
        add!(g, a => b, identity)
        push!(want, a)
    end
    g, x, want
end

@testset "exact planner groundability pruning" begin
    @testset "independent cyclic alternatives stay bounded" begin
        for inverse_first in (false, true)
            g, x, want = independent_planner_routes(18; inverse_first)
            snapshot = copy(g.recipes)
            p = plan(g; have = (x,), want)
            @test (p.cost, length(p.recipes)) == (19.0, 19)
            @test prepare(p)(2.5) == Tuple(fill(2.5, 18))
            # This public fixture previously allocated about 2.7 GB. Measure
            # the warmed planner only; the limit leaves room for platform noise.
            @test @allocated(plan(g; have = (x,), want)) < 20_000_000
            @test g.recipes == snapshot
        end
        g, x, want = independent_planner_routes(96; inverse_first = true)
        p = plan(g; have = (x,), want)
        @test (p.cost, length(p.recipes)) == (97.0, 97)
    end

    @testset "multi-output shared routes survive pruning" begin
        g = Graph()
        x = value!(g, :x, Float64)
        a = value!(g, :a, Float64)
        b = value!(g, :b, Float64)
        add!(g, x => a, identity; cost = 1.0)
        add!(g, x => b, identity; cost = 1.0)
        shared = add!(g, x => (a, b), v -> (v, v); cost = 1.25)
        p = plan(g; have = (x,), want = (a, b))
        @test p.cost == 1.25
        @test only(p.recipes) === shared

        # An inverse route may still be needed for its collateral output.
        g = Graph()
        x = value!(g, :x, Float64)
        a = value!(g, :a, Float64)
        b = value!(g, :b, Float64)
        c = value!(g, :c, Float64)
        add!(g, x => a, identity; cost = 0.25)
        add!(g, a => b, identity; cost = 0.25)
        collateral = add!(g, b => (a, c), v -> (v, v); cost = 0.25)
        p = plan(g; have = (x,), want = (a, c))
        @test (p.cost, length(p.recipes)) == (0.75, 3)
        @test collateral in p.recipes
        @test prepare(p)(2.5) == (2.5, 2.5)
    end

    @testset "exhaustive acyclic hypergraphs retain the exact optimum" begin
        rng = MersenneTwister(20261004)
        for _ in 1:150
            g = Graph()
            vals = [value!(g, Symbol(:v, i), Float64) for i in 1:6]
            for _ in 1:9
                split = rand(rng, 1:5)
                ins = Tuple(vals[randperm(rng, split)[1:rand(rng, 0:min(2, split))]])
                candidates = vals[(split + 1):6]
                outs = Tuple(candidates[randperm(rng, length(candidates))[1:rand(rng, 1:min(2, length(candidates)))]])
                add!(g; inputs = ins, outputs = outs, op = identity,
                     cost = rand(rng, (0.0, 0.25, 1.0, 2.0)),
                     effectful = rand(rng) < 0.05)
            end
            have = (vals[1],)
            want = (vals[5], vals[6])
            expected = exhaustive_planner_rank(g, have, want)
            if isfinite(first(expected))
                p = plan(g; have, want)
                @test (p.cost, length(p.recipes)) == expected
            else
                # No subset has an executable order from HAVE (oracle above).
                @test_throws PlanningError plan(g; have, want)
            end
        end
    end
end

@testset "cyclic multi-output planner completeness" begin
    @testset "self-output needs an independent grounding producer" begin
        for direct in (false, true), reverse_order in (false, true)
            g = Graph()
            x, a, b = [value!(g, name, Float64) for name in (:x, :a, :b)]
            routes = [(x => a, x -> 2x, 1.0),
                      (a => (a, b), a -> (a, a^2), 0.0)]
            reverse_order && reverse!(routes)
            for (edge, op, cost) in routes
                add!(g, edge, op; cost)
            end
            direct && add!(g, x => b, x -> 4x^2; cost = 2.0)
            snapshot = copy(g.recipes)
            p = plan(g; have = (x,), want = (b,))
            @test (p.cost, length(p.recipes)) == (1.0, 2)
            k = prepare(p)
            @test @inferred(k(3.0)) == 36.0
            @test k(-2.0) == 16.0
            @test g.recipes == snapshot
            @test p.producer[a.id] === first(p.recipes)
            # Self-coverage never establishes executable availability.
            cycle = only(r for r in g.recipes if length(r.outputs) == 2)
            @test ReactiveKernels._topo(g, [cycle.id], Set([x.id])) === nothing
        end
    end

    @testset "a cycle may be grounded at either input" begin
        for cheaper_a in (false, true), reverse_wants in (false, true)
            g = Graph()
            x, a, b, c, d = [value!(g, name, Float64) for name in (:x, :a, :b, :c, :d)]
            add!(g, a => (b, c), a -> (a, a^2); cost = 0.0)
            add!(g, b => (a, d), b -> (b, b^3); cost = 0.0)
            ground_a = add!(g, x => a, identity; cost = cheaper_a ? 1.0 : 3.0)
            ground_b = add!(g, x => b, identity; cost = cheaper_a ? 3.0 : 1.0)
            want = reverse_wants ? (d, c) : (c, d)
            p = plan(g; have = (x,), want)
            @test (p.cost, length(p.recipes)) == (1.0, 3)
            @test first(p.recipes) === (cheaper_a ? ground_a : ground_b)
            @test prepare(p)(2.0) == (reverse_wants ? (8.0, 4.0) : (4.0, 8.0))
        end
    end

    @testset "ungrounded selections cannot displace executable plans" begin
        g = Graph()
        x, a, b = [value!(g, name, Float64) for name in (:x, :a, :b)]
        add!(g, a => (a, b), a -> (a, a^2); cost = 0.0)
        # No recipe can provide the first a: executing the cycle is impossible.
        @test_throws PlanningError plan(g; have = (x,), want = (b,))
        fallback = add!(g, x => b, abs2; cost = 2.0)
        p = plan(g; have = (x,), want = (b,))
        @test only(p.recipes) === fallback
        @test prepare(p)(3.0) == 9.0
        # HAVE itself can ground a cycle, and remains the authoritative owner.
        p = plan(g; have = (a,), want = (a, b))
        @test (p.cost, length(p.recipes)) == (0.0, 1)
        @test !haskey(p.producer, a.id)
        @test prepare(p)(3.0) == (3.0, 9.0)
    end

    @testset "grounding producers can have their own prerequisites" begin
        g = Graph()
        x, seed, a, b = [value!(g, name, Float64) for name in (:x, :seed, :a, :b)]
        cycle = add!(g, a => (a, b), a -> (a, a^2); cost = 0.0)
        ground = add!(g, seed => a, identity; cost = 0.5)
        prefix = add!(g, x => seed, x -> 2x; cost = 0.5)
        p = plan(g; have = (x,), want = (b,))
        @test p.recipes == [prefix, ground, cycle]
        @test p.cost == 1.0
        @test prepare(p)(3.0) == 36.0
        # Equal total cost is resolved by recipe count, including zero-cost work.
        direct = add!(g, x => b, x -> 4x^2; cost = 1.0)
        @test only(plan(g; have = (x,), want = (b,)).recipes) === direct
    end

    @testset "independent grounding orders do not multiply the search" begin
        g = Graph()
        x = value!(g, :x, Float64)
        want = Value[]
        for i in 1:18
            a = value!(g, Symbol(:a, i), Float64)
            b = value!(g, Symbol(:b, i), Float64)
            add!(g, a => (a, b), a -> (a, a^2); cost = 0.0)
            add!(g, x => a, identity; cost = 1.0)
            push!(want, b)
        end
        p = plan(g; have = (x,), want)
        @test (p.cost, length(p.recipes)) == (18.0, 36)
        @test prepare(p)(2.0) == Tuple(fill(4.0, 18))
        # There are 18! orders in which to add the same grounding recipes.
        @test @allocated(plan(g; have = (x,), want)) < 50_000_000
    end

    @testset "bounded cyclic hypergraphs match the exhaustive oracle" begin
        rng = MersenneTwister(20261015)
        solved = impossible = 0
        for _ in 1:800
            g = Graph()
            vals = [value!(g, Symbol(:v, i), Float64) for i in 1:6]
            for _ in 1:9
                ins = Tuple(vals[randperm(rng, 6)[1:rand(rng, 0:3)]])
                outs = Tuple(vals[randperm(rng, 6)[1:rand(rng, 1:3)]])
                # This is topology-only coverage; these bodies are never run.
                add!(g; inputs = ins, outputs = outs, op = identity,
                     cost = rand(rng, (0.0, 0.25, 1.0, 2.0)),
                     effectful = rand(rng) < 0.05)
            end
            have = Tuple(vals[randperm(rng, 6)[1:rand(rng, 0:2)]])
            want = Tuple(vals[randperm(rng, 6)[1:rand(rng, 1:3)]])
            expected = exhaustive_planner_rank(g, have, want)
            if isfinite(first(expected))
                solved += 1
                p = plan(g; have, want)
                @test (p.cost, length(p.recipes)) == expected
                available = Set{Int}(canon_id(g, v.id) for v in have)
                for r in p.recipes
                    @test all(v -> canon_id(g, v.id) in available, r.inputs)
                    union!(available, (canon_id(g, v.id) for v in r.outputs))
                end
                @test all(v -> canon_id(g, v.id) in available, want)
            else
                impossible += 1
                # Every subset fails the executable-order oracle, not a ban on cycles.
                @test_throws PlanningError plan(g; have, want)
            end
        end
        @test solved > 100
        @test impossible > 100
    end
end
