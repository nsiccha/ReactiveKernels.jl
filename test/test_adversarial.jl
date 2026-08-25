using ReactiveKernels
using Test

@testset "adversarial planner and lowering contracts" begin
    @testset "authoritative HAVE survives collateral multi-output" begin
        g = Graph()
        x = value!(g, :x, Float64)
        a = value!(g, :a, Float64)
        b = value!(g, :b, Float64)
        out = value!(g, :out, Float64)
        add!(g, x => (a, b), _ -> (100.0, 2.0))
        add!(g, (a, b) => out, +)

        p = plan(g; have = (x, a), want = (out,))
        k = prepare(p)
        @test inputs(k) == (x, a)
        @test k(1.0, 10.0) == 12.0

        st = ReactiveState(g)
        set!(st, x, 1.0)
        set!(st, a, 10.0)
        @test get!(st, out) == 12.0
    end

    @testset "overlapping multi-output producers admit a valid order" begin
        g = Graph()
        x = value!(g, :x, Float64)
        a = value!(g, :a, Float64)
        b = value!(g, :b, Float64)
        c = value!(g, :c, Float64)
        d = value!(g, :d, Float64)
        first_recipe = add!(g, d => (a, b), d -> (d, d + 1))
        source_recipe = add!(g, x => (a, c), x -> (x, x + 2))
        bridge_recipe = add!(g, a => d, identity)

        p = plan(g; have = (x,), want = (b, c))
        @test [r.id for r in p.recipes] ==
              [source_recipe.id, bridge_recipe.id, first_recipe.id]
        @test prepare(p)(3.0) == (4.0, 5.0)
    end

    @testset "cycle diagnostics identify the cycle" begin
        g = Graph()
        a = value!(g, :a, Float64)
        b = value!(g, :b, Float64)
        add!(g, a => b, identity)
        add!(g, b => a, identity)
        err = try
            plan(g; want = (b,))
            nothing
        catch e
            e
        end
        @test err isa PlanningError
        @test occursin("cycle", err.msg)
        @test occursin("b", err.msg)
    end

    @testset "recipe costs enforce the exact-planner domain" begin
        for bad_cost in (-1.0, Inf, NaN)
            g = Graph()
            x = value!(g, :x, Float64)
            y = value!(g, :y, Float64)
            @test_throws ArgumentError add!(g, x => y, identity; cost = bad_cost)
        end
    end

    @testset "effectful recipes are never structurally CSE'd" begin
        for effectful_first in (true, false)
            g = Graph()
            x = value!(g, :x, Float64)
            pure_out = value!(g, :pure_out, Float64)
            effect_out = value!(g, :effect_out, Float64)
            effects = Ref(0)
            add_pure! = () -> add!(g, x => pure_out, v -> v + 1;
                                   cse_key = :same_structure)
            add_effect! = () -> add!(g, x => effect_out,
                                     v -> (effects[] += 1; v + 10);
                                     cse_key = :same_structure, effectful = true)
            if effectful_first
                add_effect!()
                add_pure!()
            else
                add_pure!()
                add_effect!()
            end

            @test length(g.recipes) == 2
            @test canon_id(g, pure_out.id) != canon_id(g, effect_out.id)
            @test count(r -> r.effectful, g.recipes) == 1
            @test prepare(g; have = (x,), want = (pure_out,))(1.0) == 2.0
            @test_throws PlanningError plan(g; have = (x,), want = (effect_out,))
            @test effects[] == 0
        end
    end

    @testset "generated bindings are globally hygienic" begin
        g = Graph()
        ops_named = value!(g, :__ops__, Float64)
        y = value!(g, :y, Float64)
        add!(g, ops_named => y, x -> x + 1)
        @test prepare(g; have = (ops_named,), want = (y,))(2.0) == 3.0

        g2 = Graph()
        a1 = value!(g2, :a, Float64)
        a2 = value!(g2, :a, Float64)
        collision = value!(g2, Symbol("a_", a1.id), Float64)
        z = value!(g2, :z, Float64)
        add!(g2, (a1, a2, collision) => z, (x, y, w) -> 100x + 10y + w)
        k = prepare(g2; have = (a1, a2, collision), want = (z,))
        @test k(1.0, 2.0, 3.0) == 123.0
    end

    @testset "HAVE collapses repeated canonical identities" begin
        g = Graph()
        x = value!(g, :x, Float64)
        k = prepare(g; have = (x, x), want = (x,))
        @test inputs(k) == (x,)
        @test k(4.0) == 4.0
        @test_throws MethodError k()
        @test_throws MethodError k(4.0, 5.0)

        source = value!(g, :source, Float64)
        x_alias = value!(g, :x_alias, Float64)
        out = value!(g, :out, Float64)
        add!(g, source => x, identity; cse_key = :same_x)
        add!(g, source => x_alias, identity; cse_key = :same_x)
        add!(g, (x, x_alias) => out, +)
        alias_kernel = prepare(g; have = (x, x_alias), want = (out,))
        @test inputs(alias_kernel) == (x,)
        @test alias_kernel(3.0) == 6.0
    end
end

@testset "adversarial reactive contracts" begin
    @testset "shared nested provenance remains valid" begin
        g = Graph()
        x = value!(g, :x, Float64)
        a = value!(g, :a, Float64)
        b = value!(g, :b, Float64)
        c = value!(g, :c, Float64)
        out = value!(g, :out, Float64)
        ca = Ref(0)
        cb = Ref(0)
        cc = Ref(0)
        add!(g, x => a, x -> (ca[] += 1; x + 1))
        add!(g, a => b, a -> (cb[] += 1; 2a))
        add!(g, (a, b) => c, (a, b) -> (cc[] += 1; a + b))
        add!(g, c => out, identity)

        st = ReactiveState(g; materialize = (a, b, c))
        set!(st, x, 1.0)
        @test get!(st, a) == 2.0
        @test get!(st, b) == 4.0
        @test get!(st, c) == 6.0
        @test get!(st, out) == 6.0
        @test get!(st, out) == 6.0
        @test (ca[], cb[], cc[]) == (1, 1, 1)
    end

    @testset "checkpoint refuses stale reactive values" begin
        g = Graph()
        x = value!(g, :x, Float64)
        a = value!(g, :a, Float64)
        add!(g, x => a, x -> x + 1)
        st = ReactiveState(g; materialize = (a,))
        set!(st, x, 1.0)
        @test get!(st, a) == 2.0
        set!(st, x, 10.0)
        @test_throws ErrorException checkpoint(st, (a,))
        @test get!(st, a) == 11.0
        @test checkpoint(st, (a,))[canon_id(g, a.id)] == 11.0
    end
end
