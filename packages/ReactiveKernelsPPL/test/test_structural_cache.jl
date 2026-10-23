using ReactiveKernels, ReactiveKernelsPPL, Test

# `bind_data` computes the analyses a plan's structure decides alone (which
# definitions are module data, which inputs are model-level, array axes,
# the structure checks) once per structure, so rebinding one lowering to
# many designs costs each design's data work (snag rkppl-bind-data-deb7f5e5:
# a 70-statement BRM program bound in about 8.5 ms whatever the data size).
# The cache is internal; these tests pin that a cached binding equals an
# uncached one, that rebinding adds no analyses, that keys are exact, that a
# binding computes a value of a column once, and that the cache lives only
# as long as its lowering.

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

# Validation walks a plan's expressions once per structure and tests the
# bound values at its leaves on every binding (snag rkppl-bind-data-96ae5ee3).
# Each model below binds valid data first, so the walk is cached, then a
# design whose values break one leaf: the warm binding fails exactly as an
# uncached one.
_scache_error(f) = try
    f()
    nothing
catch e
    e isa ContractValidationError || rethrow()
    sprint(showerror, e)
end

const _SCACHE_LEAVES = [
    # A positional gather from an array value: integer positions in range.
    (@rkppl(begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        zeta ~ Dirichlet([1.0, 2.0])
        cum = cumsum(vcat(0.0, zeta))
        m = cum[c]
        y .~ Normal.(a .+ b .* m, 1.0)
    end), (; c = [1, 2, 3, 1], y = [0.1, 0.2, 0.3, 0.4]),
        [(; c = [1, 2, 5, 1], y = [0.1, 0.2, 0.3, 0.4]) => "outside 1:3",
         (; c = [1.0, 2.0, 3.0, 1.0], y = [0.1, 0.2, 0.3, 0.4]) => "does not hold integers"]),
    # A gather on a levels axis: every index value is one of its labels.
    (@rkppl(begin
        s ~ Exponential(1)
        z[levels(h)] .~ Normal.(0, s)
        mu = z[g]
        y .~ Normal.(mu, 1.0)
    end), (; h = ["a", "b"], g = ["a", "b", "a"], y = [0.1, 0.2, 0.3]),
        [(; h = ["a", "b"], g = ["a", "c", "a"], y = [0.1, 0.2, 0.3]) => "not a label"]),
    # Elementwise math over a column: a numeric column.
    (@rkppl(begin
        a ~ Normal(0, 1)
        lx = log.(x .+ 3)
        y .~ Normal.(a .+ lx, 1.0)
    end), (; x = [0.1, 0.2, 0.3], y = [0.1, 0.2, 0.3]),
        [(; x = ["p", "q", "r"], y = [0.1, 0.2, 0.3]) => "not numeric"]),
    # An `ifelse` condition column: Bool.
    (@rkppl(begin
        a ~ Normal(0, 1)
        m = ifelse.(flag, x, 0.0)
        y .~ Normal.(a .+ m, 1.0)
    end), (; flag = [true, false, true], x = [0.1, 0.2, 0.3], y = [0.1, 0.2, 0.3]),
        [(; flag = [1, 0, 1], x = [0.1, 0.2, 0.3], y = [0.1, 0.2, 0.3]) => "must be Bool"]),
    # A literal position of a declared array: inside its data-sized axis.
    (@rkppl(begin
        z[1:K] .~ Normal.(0, 1)
        t = z[3]
        y .~ Normal.(t, 1.0)
    end), (; K = 3, y = [0.1, 0.2]),
        [(; K = 2, y = [0.1, 0.2]) => "out of bounds"]),
]

@testset "structural cache: a rebinding runs every data check" begin
    for (model, good, bads) in _SCACHE_LEAVES
        unbound = lower_rkppl(model, good; conditioned = (:y,))
        @test bind_data(unbound, good) isa _SCACHE.StructuralPlan
        for (bad, fragment) in bads
            warm = _scache_error(() -> bind_data(unbound, bad))
            @test warm !== nothing && occursin(fragment, warm)
            @test warm == _scache_error(() -> _scache_uncached(unbound, bad))
            # The valid design still binds after the failure.
            @test bind_data(unbound, good) isa _SCACHE.StructuralPlan
        end
    end
end

# Keys hold a column table's names (and value types) as an interned id, not
# a sorted tuple built per key (snag bind-data-per-de-d92f13be).
@testset "structural cache: signature ids are exact" begin
    cache = _SCACHE._StructuralCache()
    t1 = Dict{Symbol,Any}(:a => 1.0, :b => [1, 2])
    t2 = Dict{Symbol,Any}(:b => [3, 4, 5], :a => 2.0)
    t3 = Dict{Symbol,Any}(:a => 1, :b => [1, 2])
    @test _SCACHE._types_id(cache, t1) == _SCACHE._types_id(cache, t2)
    @test _SCACHE._types_id(cache, t1) != _SCACHE._types_id(cache, t3)
    @test _SCACHE._types_id(cache, t1) != _SCACHE._types_id(cache, Dict{Symbol,Any}(:a => 1.0))
    r1 = Dict(:a => :data, :b => :response)
    roles(r) = _SCACHE._table_id(cache, r, :roles, identity)
    @test roles(r1) == roles(Dict(:b => :response, :a => :data))
    @test roles(r1) != roles(Dict(:a => :predictor, :b => :response))
    @test roles(r1) != _SCACHE._types_id(cache, Dict{Symbol,Any}(:a => 1.0, :b => [1, 2]))
    @test _SCACHE._names_id(cache, keys(t3)) == _SCACHE._names_id(cache, Set([:b, :a]))
    @test _SCACHE._names_id(cache, Set([:a])) != _SCACHE._names_id(cache, Set([:a, :b]))
    # Signatures that share a hash keep their own ids: the stored signature
    # decides.
    key = (:probe, UInt(7), 1)
    id = _SCACHE._signature_id(==(:x), () -> :x, cache, key)
    @test _SCACHE._signature_id(==(:x), () -> :x, cache, key) == id
    @test _SCACHE._signature_id(==(:y), () -> :y, cache, key) ∉ (id, nothing)
end

@testset "structural cache: a binding computes a column's value once" begin
    calls = Ref(0)
    col = ["b", "a", "b"]
    levels_of(c) = () -> (calls[] += 1; sort!(unique(c)))
    # Outside a binding every call computes.
    @test _SCACHE._bind_memo(levels_of(col), :levels, col) == ["a", "b"]
    @test _SCACHE._bind_memo(levels_of(col), :levels, col) == ["a", "b"]
    @test calls[] == 2
    calls[] = 0
    other = copy(col)
    _SCACHE._with_bind_memo() do
        @test _SCACHE._bind_memo(levels_of(col), :levels, col) == ["a", "b"]
        @test _SCACHE._bind_memo(levels_of(col), :levels, col) == ["a", "b"]
        # Another column, equal or not, has its own value.
        @test _SCACHE._bind_memo(levels_of(other), :levels, other) == ["a", "b"]
    end
    @test calls[] == 2
    # Each binding starts afresh.
    _SCACHE._with_bind_memo(() -> _SCACHE._bind_memo(levels_of(col), :levels, col))
    @test calls[] == 3
end

# An absent response holding one array per index is sized from the arrays
# its cells read; its operands still validate, whether each index's array
# has its cell's axes (read as it is) or broadcasts into them.
@testset "structural cache: an absent response's per-index operands validate" begin
    src = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        @plate for i in eachindex(y)
            y[i] .~ censored.(Normal.(a .+ x[i], s), lo[i], Inf)
        end
    end
    x = [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6]]
    fit = Dict{Symbol,Any}(:x => x, :y => [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
        :lo => [[-1.0, -1.0], [-1.0], fill(-1.0, 3)])
    plan = lower_rkppl(src, fit; conditioned = (:y,), mod = @__MODULE__)
    bind_data(plan, fit)
    for lo in ([[-1.0, -0.5], [-1.0], [-1.0, -2.0, -1.0]], [[-1.0], [-1.0], [-2.0]])
        d = Dict{Symbol,Any}(:x => x, :lo => lo)
        b, r = bind_data(plan, d), _scache_uncached(plan, d)
        @test b.columns[:y] isa _SCACHE._Absent
        @test isequal(b.columns, r.columns)
        @test _scache_program(b) == _scache_program(r)
        bad = [copy(c) for c in lo]
        bad[2][1] = NaN
        @test_throws "finite numerics" bind_data(plan, Dict{Symbol,Any}(:x => x, :lo => bad))
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
