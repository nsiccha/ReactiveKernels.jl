# Phase 1 & 2 correctness + hot-path tests (gist §20 items 1-10, §21).
using ReactiveKernels
using Test

# Helper: does the generated Expr mention a call to op index `k`?
calls_opindex(ast, k) = occursin("__ops__[$k]", string(ast))

@testset "Phase 1 — single-producer DAG" begin
    @testset "1. want already in have -> identity kernel" begin
        g = Graph()
        x = value!(g, :x, Float64)
        k = prepare(g; have=(x,), want=(x,))
        @test k(3.5) == 3.5
        @test isempty(k.plan.recipes)
    end

    @testset "2. linear chain -> only required chain selected" begin
        g = Graph()
        x = value!(g, :x, Float64); a = value!(g, :a, Float64); b = value!(g, :b, Float64)
        add!(g, x => a, x -> x + 1.0)
        add!(g, a => b, a -> 2a)
        k = prepare(g; have=(x,), want=(b,))
        @test k(1.0) == 4.0            # a=2, b=4
    end

    @testset "3. branching -> unrelated branch absent from generated code" begin
        g = Graph()
        x = value!(g, :x, Float64); a = value!(g, :a, Float64)
        b = value!(g, :b, Float64); c = value!(g, :c, Float64)
        add!(g, x => a, x -> x)
        add!(g, a => b, a -> a + 1.0)   # wanted branch
        add!(g, a => c, a -> error("c must not run"))  # unrelated branch
        p = plan(g; have=(x,), want=(b,))
        ast = code_expr(p)
        # only 2 recipes selected (x->a, a->b); the c branch must be absent
        @test length(p.recipes) == 2
        s = string(ast)
        @test !occursin("c", s) || !occursin(":c", s)  # c not assigned
        k = prepare(p)
        @test k(5.0) == 6.0
    end

    @testset "4. multiple wants -> shared ancestor executes once" begin
        g = Graph()
        calls = Ref(0)
        x = value!(g, :x, Float64); a = value!(g, :a, Float64)
        b = value!(g, :b, Float64); c = value!(g, :c, Float64)
        add!(g, x => a, x -> (calls[] += 1; x * 2))
        add!(g, a => b, a -> a + 1.0)
        add!(g, a => c, a -> a + 2.0)
        k = prepare(g; have=(x,), want=(b, c))
        calls[] = 0
        res = k(3.0)                    # a=6 (once), b=7, c=8
        @test res == (7.0, 8.0)
        @test calls[] == 1              # shared `a` computed once
        # generated code assigns `a` exactly once
        @test count(!isnothing, [match(r"\ba =", l) for l in split(string(code_expr(k)), '\n')]) == 1
    end

    @testset "5. multi-output recipe -> tuple destructuring / type stability" begin
        g = Graph()
        u = value!(g, :u, Float64); a = value!(g, :a, Float64)
        b = value!(g, :b, Float64); s = value!(g, :s, Float64)
        add!(g; inputs=u, outputs=(a, b), op = u -> (u + 1.0, u + 2.0))
        add!(g, (a, b) => s, (a, b) -> a + b)

        # single want downstream of a multi-output recipe
        ks = prepare(g; have=(u,), want=(s,))
        @test ks(10.0) == 23.0          # a=11, b=12, s=23

        # direct multi-want off the multi-output recipe
        k = prepare(g; have=(u,), want=(a, b))
        @test k(10.0) === (11.0, 12.0)
        @test @inferred(k(10.0)) === (11.0, 12.0)
    end

    @testset "6. impossible want -> informative error" begin
        g = Graph()
        x = value!(g, :x, Float64); z = value!(g, :z, Float64); q = value!(g, :q, Float64)
        add!(g, q => z, q -> q)         # z needs q, q has no producer
        err = try; plan(g; have=(x,), want=(z,)); catch e; e; end
        @test err isa PlanningError
        @test occursin("Cannot produce", err.msg)
        @test occursin("q", err.msg)
    end

    @testset "7. cycle -> informative error" begin
        g = Graph()
        a = value!(g, :a, Float64); b = value!(g, :b, Float64)
        add!(g, a => b, a -> a)
        add!(g, b => a, b -> b)
        err = try; plan(g; have=(), want=(b,)); catch e; e; end
        @test err isa PlanningError
    end
end

@testset "Phase 2 — alternative producers & cost planning" begin
    # gist §21 concrete acceptance example
    g = Graph()
    u = value!(g, :u, Float64)
    a = value!(g, :a, Float64)
    b = value!(g, :b, Float64)
    c = value!(g, :c, Float64)
    r = value!(g, :r, Float64)

    cheap_a(u)    = u + 0.0
    combined_ab(u) = (u + 1.0, u + 10.0)
    make_b(a)     = a + 100.0
    make_c(a)     = a + 1000.0
    finish(b, c)  = b + c

    add!(g, u => a, cheap_a; cost=1.0)
    add!(g, u => (a, b), combined_ab; cost=1.2)
    add!(g, a => b, make_b; cost=1.0)
    add!(g, a => c, make_c; cost=1.0)
    add!(g, (b, c) => r, finish; cost=1.0)

    @testset "8/Query A. want a only -> cheap_a chosen" begin
        ka = plan(g; have=(u,), want=(a,))
        @test ka.cost == 1.0
        @test length(ka.recipes) == 1
        @test ka.recipes[1].op === cheap_a
        k = prepare(ka)
        @test k(5.0) == 5.0
    end

    @testset "9/Query B. want r -> jointly-cheaper combined_ab considered" begin
        kr = plan(g; have=(u,), want=(r,))
        # Route via cheap_a: cheap_a(1) + make_b(1) + make_c(1) + finish(1) = 4.0
        # Route via combined_ab: combined_ab(1.2) + make_c(1) + finish(1) = 3.2
        @test kr.cost == 3.2
        @test any(rc -> rc.op === combined_ab, kr.recipes)
        @test !any(rc -> rc.op === make_b, kr.recipes)  # b comes from combined_ab
        k = prepare(kr)
        # u=2: combined -> a=3, b=12; make_c -> c=1003; finish -> 1015
        @test k(2.0) == 1015.0
    end

    @testset "10/Query C. have boundary is never recomputed" begin
        kc = plan(g; have=(a, b), want=(r,))
        @test length(kc.recipes) == 2   # make_c, finish only
        s = string(code_expr(kc))
        @test !occursin("cheap_a", s) && !occursin("combined_ab", s) && !occursin("make_b", s)
        k = prepare(kc)
        # a=2, b=3: c=make_c(2)=1002, r=finish(3,1002)=1005
        @test k(2.0, 3.0) == 1005.0
    end
end

@testset "Hot-path: no orchestration allocations (gist §10, §21)" begin
    g = Graph()
    a = value!(g, :a, Float64); b = value!(g, :b, Float64)
    c = value!(g, :c, Float64); r = value!(g, :r, Float64)
    add!(g, a => c, a -> a + 1000.0; cost=1.0)
    add!(g, (b, c) => r, (b, c) -> b + c; cost=1.0)
    k = prepare(g; have=(a, b), want=(r,))

    # Measure inside a function barrier so the kernel is a concretely-typed
    # argument (a non-const local binding would otherwise box and mask the 0).
    function alloc2(k, a, b)
        k(a, b)                          # warmup
        @allocated k(a, b)
    end
    @test alloc2(k, 1.0, 2.0) == 0
    @test k(1.0, 2.0) == 1003.0
end

@testset "explain / inspection" begin
    g = Graph()
    u = value!(g, :u, Float64); a = value!(g, :a, Float64)
    b = value!(g, :b, Float64); c = value!(g, :c, Float64); r = value!(g, :r, Float64)
    add!(g, u => a, identity; cost=1.0)
    add!(g, u => (a, b), u -> (u, u); cost=1.2)
    add!(g, a => c, identity; cost=1.0)
    add!(g, (b, c) => r, +; cost=1.0)
    p = plan(g; have=(u,), want=(r,))
    txt = explain(p)
    @test occursin("Have:", txt)
    @test occursin("Want:", txt)
    @test occursin("Total graph cost:", txt)
    @test inputs(p) == (u,)
    @test outputs(p) == (r,)
end
