using ReactiveKernels, ReactiveKernelsPPL, Test

# `bind_data` computes the analyses a plan's structure decides alone (which
# definitions are module data, which inputs are model-level, array axes,
# the structure checks) once per structure, so rebinding one lowering to
# many designs costs each design's data work (snag rkppl-bind-data-deb7f5e5:
# a 70-statement BRM program bound in about 8.5 ms whatever the data size).
# The cache is internal; these tests pin that a cached binding equals an
# uncached one, that rebinding adds no analyses, and that the cache lives
# only as long as its lowering.

const _SCACHE = ReactiveKernelsPPL

ReactiveKernels.@kernel _scache_positions(values, pool) = begin
    positions = ReactiveKernels.plate(values) do v
        Int(findfirst(isequal(v), pool))
    end
    return positions
end
_scache_center(x) = x .- sum(x) / length(x)

const _SCACHE_MODEL = @rkppl begin
    a ~ Normal(0, 1)
    s ~ Exponential(1)
    z[1:2] .~ Normal.(0, 1)
    gi = _scache_positions(g, [1, 2])
    xc = _scache_center(x)
    X = hcat(ones(length(x)), xc)
    b[axes(X, 2)] .~ Normal.(0, 1)
    mu = a .+ X * b .+ z[gi]
    y .~ Normal.(mu, s)
end

_scache_design(n) = (; x = collect(range(-1.0, 1.0; length = n)),
    g = [isodd(i) ? 1 : 2 for i in 1:n], y = 0.1 .* (1:n) .- 0.3)
_scache_without_y(d) = (; x = d.x, g = d.g)
_scache_lower(d) = lower_rkppl(_SCACHE_MODEL, d; conditioned = (:y,))
# The same binding without the cache: outside `bind_data`'s scope every
# analysis computes afresh.
_scache_uncached(plan, d) = _SCACHE._bind_data_latest(plan, Dict{Symbol,Any}(pairs(d)))
_scache_program(b) = kernel_expr(b, assign_layout(b))
_scache_entries(plan) = _SCACHE._structural_cache(plan).entries
_scache_kernel_entries(plan) =
    count(k -> k isa Tuple && first(k) === :prepared_kernel, keys(_scache_entries(plan)))

@testset "structural cache: a cached binding equals an uncached one" begin
    unbound = _scache_lower(_scache_design(4))
    for n in (3, 8, 50)
        d = _scache_design(n)
        b, r = bind_data(unbound, d), _scache_uncached(unbound, d)
        @test _scache_program(b) == _scache_program(r)
        @test isequal(b.columns, r.columns)
        @test b.n_obs == r.n_obs == n
        @test b.roles == r.roles
        # Rebinding a bound plan reads its structure too.
        @test _scache_program(bind_data(b, _scache_design(n + 1))) ==
            _scache_program(_scache_uncached(r, _scache_design(n + 1)))
        # The data-only kernel call evaluates on this design's data.
        @test b.columns[:gi] == d.g
        @test b.columns[:xc] ≈ d.x .- sum(d.x) / n
    end
    # A design without the response is another set of column names.
    d = _scache_without_y(_scache_design(6))
    b, r = bind_data(unbound, d), _scache_uncached(unbound, d)
    @test _scache_program(b) == _scache_program(r)
    @test !haskey(b.columns, :y) || b.columns[:y] isa _SCACHE._Absent
end

@testset "structural cache: rebinding adds no analyses" begin
    unbound = _scache_lower(_scache_design(4))
    first_bound = bind_data(unbound, _scache_design(5))
    entries = length(_scache_entries(unbound))
    @test entries > 0
    for n in (2, 9, 40)
        b = bind_data(unbound, _scache_design(n))
        # The fields binding rebuilds are the earlier binding's objects.
        @test _SCACHE._structure_fields(b) === _SCACHE._structure_fields(first_bound)
    end
    @test length(_scache_entries(unbound)) == entries
    # The data-only kernel is prepared once for the lowering.
    @test _scache_kernel_entries(unbound) == 1
    # Outside `bind_data` nothing is cached: lowering fills its tables as
    # it goes.
    @test !_SCACHE._structural_enabled()
end

@testset "structural cache: the check recomputes every hit" begin
    unbound = _scache_lower(_scache_design(4))
    _SCACHE._STRUCTURAL_CACHE_CHECK[] = true
    try
        for n in (3, 7, 3)
            @test bind_data(unbound, _scache_design(n)) isa _SCACHE.StructuralPlan
        end
        # Positive control: a stale entry fails the check.
        cache = _scache_entries(unbound)
        stale = [k for k in keys(cache) if k isa Tuple && first(k) isa Tuple &&
            first(first(k)) === :module_data_names]
        @test !isempty(stale)
        foreach(k -> cache[k] = Set([:not_a_definition]), stale)
        @test_throws ErrorException bind_data(unbound, _scache_design(5))
        empty!(cache)
        @test bind_data(unbound, _scache_design(5)) isa _SCACHE.StructuralPlan
    finally
        _SCACHE._STRUCTURAL_CACHE_CHECK[] = false
    end
end

@testset "structural cache: concurrent binds of one lowering" begin
    unbound = _scache_lower(_scache_design(4))
    sizes = 3:12
    tasks = [Threads.@spawn(bind_data(unbound, _scache_design(n))) for n in sizes]
    for (n, t) in zip(sizes, tasks)
        b = fetch(t)
        r = _scache_uncached(unbound, _scache_design(n))
        @test _scache_program(b) == _scache_program(r)
        @test isequal(b.columns, r.columns)
    end
end

@noinline function _scache_dropped_lowering()
    plan = _scache_lower(_scache_design(5))
    bind_data(plan, _scache_design(7))
    return WeakRef(plan.assignments)
end

@testset "structural cache: lives as long as its lowering" begin
    ref = _scache_dropped_lowering()
    GC.gc(true)
    GC.gc(true)
    @test ref.value === nothing
end
