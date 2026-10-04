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
        available = Set(canon_id(g, v.id) for v in have)
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
                @test_throws PlanningError plan(g; have, want)
            end
        end
    end
end
