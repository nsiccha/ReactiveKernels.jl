# Phase 4: reactive/incremental state layer (gist §20 tests 11-18).
using ReactiveKernels
using Test

@testset "11. valid materialized value becomes have for next request" begin
    g = Graph()
    x = value!(g, :x, Float64); y = value!(g, :y, Float64)
    a = value!(g, :a, Float64); b = value!(g, :b, Float64)
    ca = Ref(0)
    add!(g, x => a, xx -> (ca[] += 1; xx + 1.0))
    add!(g, (a, y) => b, (aa, yy) -> aa * yy)

    st = ReactiveState(g; materialize=(a,))
    set!(st, x, 2.0); set!(st, y, 10.0)
    @test get!(st, b) == 30.0
    @test ca[] == 1
    @test get!(st, b) == 30.0        # `a` reused as HAVE, not recomputed
    @test ca[] == 1
end

@testset "12. version update -> dependent cached value recomputed on demand" begin
    g = Graph()
    x = value!(g, :x, Float64); a = value!(g, :a, Float64); b = value!(g, :b, Float64)
    ca = Ref(0)
    add!(g, x => a, xx -> (ca[] += 1; xx + 1.0))
    add!(g, a => b, aa -> 2aa)
    st = ReactiveState(g; materialize=(a,))
    set!(st, x, 2.0)
    @test get!(st, b) == 6.0         # a=3
    @test ca[] == 1
    set!(st, x, 4.0)                 # bump x version -> a stale
    @test get!(st, b) == 10.0        # a=5 recomputed
    @test ca[] == 2
end

@testset "13. alternative-producer provenance: unused path change does not invalidate" begin
    g = Graph()
    p = value!(g, :p, Float64); q = value!(g, :q, Float64); z = value!(g, :z, Float64)
    ch = Ref(0)
    add!(g, p => z, pp -> (ch[] += 1; pp + 1.0); cost=1.0)   # cheaper, chosen
    add!(g, q => z, qq -> (ch[] += 1; qq + 5.0); cost=2.0)   # unused alternative
    st = ReactiveState(g; materialize=(z,))
    set!(st, p, 10.0); set!(st, q, 100.0)
    @test get!(st, z) == 11.0        # via cheap route from p
    @test ch[] == 1
    set!(st, q, 999.0)               # change input on the UNUSED path
    @test get!(st, z) == 11.0        # still valid (prov only records p)
    @test ch[] == 1                  # not recomputed
    set!(st, p, 20.0)                # change the actual dependency
    @test get!(st, z) == 21.0
    @test ch[] == 2                  # now recomputed
end

@testset "14. materialization boundary: only nominated intermediates persist" begin
    g = Graph()
    x = value!(g, :x, Float64); a = value!(g, :a, Float64)
    b = value!(g, :b, Float64); c = value!(g, :c, Float64)
    add!(g, x => a, xx -> xx + 1.0)
    add!(g, a => b, aa -> aa + 1.0)   # not materialized -> ephemeral
    add!(g, b => c, bb -> bb + 1.0)   # want
    st = ReactiveState(g; materialize=(a,))
    set!(st, x, 0.0)
    @test get!(st, c) == 3.0
    @test haskey(st.values, canon_id(g, a.id))   # materialized `a` persisted
    @test !haskey(st.values, canon_id(g, b.id))  # `b` stayed kernel-local
    @test !haskey(st.values, canon_id(g, c.id))  # want not auto-cached
end

@testset "15. kernel reuse: repeated effective signature reuses prepared kernel" begin
    g = Graph()
    x = value!(g, :x, Float64); a = value!(g, :a, Float64); b = value!(g, :b, Float64)
    add!(g, x => a, xx -> xx + 1.0)
    add!(g, a => b, aa -> 2aa)
    st = ReactiveState(g; materialize=(a,))
    set!(st, x, 1.0)
    get!(st, b)                       # signature 1: have {x}
    n1 = length(st.cache)
    get!(st, b)                       # signature 2: have {a}
    n2 = length(st.cache)
    get!(st, b)                       # same signature as #2 -> reuse
    n3 = length(st.cache)
    @test n3 == n2                    # no new kernel prepared
end

@testset "16. frozen cut point: upstream change does not recompute frozen value" begin
    g = Graph()
    X = value!(g, :X, Float64); loc = value!(g, :loc, Float64); sX = value!(g, :sX, Float64)
    cloc = Ref(0)
    add!(g, X => loc, xx -> (cloc[] += 1; xx / 2))
    add!(g, (X, loc) => sX, (xx, ll) -> xx - ll)
    st = ReactiveState(g)
    set!(st, X, 10.0)
    freeze!(st, loc, 100.0)           # authoritative, detached from X
    @test get!(st, sX) == -90.0       # 10 - 100
    @test cloc[] == 0                 # loc producer never ran
    set!(st, X, 20.0)                 # change upstream source
    @test get!(st, sX) == -80.0       # 20 - frozen 100
    @test cloc[] == 0                 # frozen loc still not recomputed
end

@testset "17. explicit have shadows stale cache / producers" begin
    g = Graph()
    raw = value!(g, :raw, Float64); der = value!(g, :der, Float64); out = value!(g, :out, Float64)
    cder = Ref(0)
    add!(g, raw => der, rr -> (cder[] += 1; rr * 10))
    add!(g, der => out, dd -> dd + 1.0)
    st = ReactiveState(g)
    set!(st, raw, 5.0)
    set!(st, der, 7.0)               # supply `der` explicitly
    @test get!(st, out) == 8.0       # uses explicit der, not producer(raw)
    @test cder[] == 0                # producer upstream never consulted
end

@testset "18. replay/checkpoint: phase-1 frozen values reused with phase-2 inputs" begin
    g = Graph()
    X = value!(g, :X, Float64)
    location = value!(g, :location, Float64)
    scale = value!(g, :scale, Float64)
    standardized = value!(g, :standardized, Float64)
    cloc = Ref(0); csc = Ref(0)
    add!(g, X => location, xx -> (cloc[] += 1; xx))          # "mean"
    add!(g, X => scale, xx -> (csc[] += 1; abs(xx) + 1.0))   # "std"
    add!(g, (X, location, scale) => standardized, (xx, l, s) -> (xx - l) / s)

    # Phase 1: derive stats from training data, then checkpoint them.
    st = ReactiveState(g; materialize=(location, scale))
    set!(st, X, 4.0)
    loc, sc = get!(st, (location, scale))
    @test (cloc[], csc[]) == (1, 1)
    cp = checkpoint(st, (location, scale))

    # Phase 2: new state seeded with frozen phase-1 stats + test data.
    st2 = ReactiveState(g; frozen=cp)
    set!(st2, X, 40.0)
    y = get!(st2, standardized)
    @test y == (40.0 - loc) / sc     # frozen stats, phase-2 X
    @test (cloc[], csc[]) == (1, 1)  # phase-1 producers NOT re-run in phase 2
end

@testset "reactive orchestration allocates, but the kernel itself does not" begin
    g = Graph()
    x = value!(g, :x, Float64); a = value!(g, :a, Float64); b = value!(g, :b, Float64)
    add!(g, x => a, xx -> xx + 1.0)
    add!(g, a => b, aa -> 2aa)
    st = ReactiveState(g)
    set!(st, x, 1.0)
    # the prepared kernel reused inside reactive get! is itself zero-alloc
    kern = prepare!(st.cache, g; have=(x,), want=(b,))
    alloc(k, v) = (k(v); @allocated k(v))
    @test alloc(kern, 1.0) == 0
    @test get!(st, b) == 4.0
end
