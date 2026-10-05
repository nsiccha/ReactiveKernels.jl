# Graph composition and preparation caching (gist §13, §17).

"""
    compose(graphs...) -> Graph

Merge several graphs into one. Because `Value` identities are process-global,
a value shared between fragments is automatically the same node in the result —
composition preserves stable value identity with no explicit port mapping
(gist §13). Structural-CSE aliases are carried over and duplicate recipes across
fragments coalesce.
"""
function compose(gs::Graph...)
    out = Graph()
    for g in gs
        for value in values(g.values)
            _register!(out, value)
        end
        merge!(out.aliases, g.aliases)
    end
    for g in gs
        for r in g.recipes
            add!(out; inputs = r.inputs, outputs = r.outputs, op = r.op,
                 cost = r.cost, cse_key = r.cse_key, effectful = r.effectful,
                 source = r.source)
        end
    end
    out
end

"""
    PreparationCache()

A cache of prepared kernels keyed by graph identity+version and the ordered,
canonicalized have/want signature plus pass identities (gist §17). Cache lookup
happens only in `prepare!`, never on the hot path; a graph mutation bumps its
version and so cannot return a stale kernel.

A bound preparation (`prepare!(cache, …; bound = …)`) is cached one level
deeper: the planned boundary, the compiled data-only prefix and the compiled
residual body are kept per graph version, have/want/pass signature and bound
PORT set, so every later binding of new values to the same ports pays only
the prefix's execution (the bound-data mathematics) and a fresh constant
table. The entry is keyed on the PORTS, not on the bound values or their
types: a later binding of differently typed data on the same ports (a
different plan struct, an `Int` vector for a `Float64` one) reuses the entry
and its compiled bodies, which are generic over those types and specialize
per concrete type on first use like any other Julia call; a lazy branch on a
hoisted constant stays a per-call branch and follows each binding's value.
Only a residual containing authored plates is templated per bound-value type
and shape signature, because that decides whether its plates specialize. The
cache is caller-owned and grows with the distinct boundaries (and those
shapes) prepared through it; it never observes the bound values afterwards and
retains none of them. Rebinding is not specialized on the graph's kernel
types, so a graph's first rebinding in a process compiles nothing once any
graph has rebound. Lookups and first-time preparations hold the cache's lock,
so one cache may serve concurrent tasks.

Plain `prepare(g; …, bound)` and `prepare(spec; …, bound)` reuse the same
bound entries through a cache the graph itself holds (see
[`prepare`](@ref)); a caller-owned cache gives those entries an explicit
lifetime instead, independent of the graph's.
"""
struct PreparationCache
    kernels::Dict{Any,PreparedKernel}
    bound::Dict{Any,_BoundEntry}
    lock::ReentrantLock
end
PreparationCache() = PreparationCache(Dict{Any,PreparedKernel}(),
                                      Dict{Any,_BoundEntry}(), ReentrantLock())

Base.length(c::PreparationCache) = length(c.kernels) + length(c.bound)

_sig(g::Graph, vs) = Tuple(canon_id(g, v.id) for v in _astuple(vs))
_pass_key(pass) = Base.issingletontype(typeof(pass)) ? typeof(pass) : objectid(pass)

"""
    prepare!(cache, g; have, want, passes=(), bound=()) -> PreparedKernel

Like `prepare`, but reuses a cached kernel when the same graph version and
effective have/want/pass signature has been prepared before.

With a non-empty `bound` (one `Value => data` pair or an iterable of them, as
for `prepare`), the returned kernel is the same residual kernel `prepare(g;
have, want, passes, bound)` builds, over the remaining HAVE ports. The cache
keeps everything that does not depend on the bound values — the plan, the
prepared prefix and the compiled residual — so repeating the call with new
data for the same ports runs only the prefix on that data. A residual whose
inner authored plates specialize on the bound data (bound-only cell recipes,
data-bound branch partitions) is still specialized and compiled per binding;
it reuses the plan and prefix only.
"""
function prepare!(cache::PreparationCache, g::Graph; have = (), want = (),
                  passes = (), bound = (), on_error = nothing)
    bound === () || return _prepare_bound!(cache, g, have, want, passes, bound,
                                           objectid(g); on_error)
    key = (objectid(g), g.version, _sig(g, have), _sig(g, want),
           Tuple(objectid(p) for p in passes), on_error)
    lock(cache.lock) do
        get!(cache.kernels, key) do
            prepare(g; have = have, want = want, passes = passes, on_error = on_error)
        end
    end
end

# The bound preparations plain `prepare(g; bound)` reuses. The graph holds them,
# so they live exactly as long as it does, and a mutation (a new version)
# starts afresh rather than keeping entries no lookup can reach. Their keys
# leave out the graph's identity, which is implicit, and name each (singleton)
# pass by its type: neither `objectid` survives serialization, and a graph
# bound while a package image is produced keeps these entries in the image.
struct _GraphPreparations
    version::Int
    cache::PreparationCache
end

const _GRAPH_PREPARATIONS_LOCK = ReentrantLock()

function _graph_preparations(g::Graph)
    lock(_GRAPH_PREPARATIONS_LOCK) do
        memo = g.preparations
        memo isa _GraphPreparations && memo.version == g.version && return memo.cache
        cache = PreparationCache()
        g.preparations = _GraphPreparations(g.version, cache)
        cache
    end
end

# Whether plain `prepare(g; bound, passes)` goes through the graph's cache: a
# non-empty binding whose passes keep one identity across calls. A pass built
# afresh per call (a closure over request data) would key a new entry on every
# call, so such a preparation is not retained.
function _reuses_bound_preparation(@nospecialize(bound), @nospecialize(passes))
    bound === () && return false
    all(pass -> Base.issingletontype(typeof(pass)), passes) || return false
    !isempty(first(_partial_bound_pairs(bound)))
end

_graph_bound_preparation(g::Graph, @nospecialize(have), @nospecialize(want),
                         @nospecialize(passes), @nospecialize(bound), on_error = nothing) =
    _prepare_bound!(_graph_preparations(g), g, have, want, passes, bound, nothing; on_error)

# `graph_key` names the graph in a caller-owned cache (its `objectid`), and is
# `nothing` in the graph's own cache.
function _prepare_bound!(cache::PreparationCache, g::Graph, @nospecialize(have),
                         @nospecialize(want), @nospecialize(passes), @nospecialize(bound),
                         graph_key; on_error = nothing)
    ports, data = _partial_bound_pairs(bound)
    isempty(ports) && return prepare!(cache, g; have = have, want = want,
                                     passes = passes, on_error = on_error)
    key = (:bound, graph_key, g.version, _sig(g, have), _sig(g, want),
           _sig(g, ports), Tuple(_pass_key(p) for p in passes), on_error)
    entry = lock(cache.lock) do
        get!(cache.bound, key) do
            _bound_entry(_kernel_error_policy(plan(g; have = have, want = want), on_error), ports)
        end
    end
    # The per-binding mathematics runs outside the lock.
    hoisted = _bound_hoisted(entry, data)
    residual, specialized = _bound_residual(entry, hoisted)
    # A residual the inner-plate pass rewrote bakes this binding's values in.
    specialized === residual || return prepare(specialized; passes = passes)
    fresh = nothing
    template = lock(cache.lock) do
        entry.template isa _UnbuiltTemplate || return entry.template
        template, kernel = _bound_build(entry, residual, hoisted, passes)
        entry.template = template
        fresh = kernel
        template
    end
    fresh === nothing ? _bound_rebind(entry, template, residual, hoisted, passes) : fresh
end

# The bound-preparation path is untyped (partial_evaluation.jl), so one
# compilation serves every graph and bound type; these directives put it in
# this package's image. Nothing else guarantees it is compiled before the first
# rebinding in a process: the first binding compiled `_bound_rebind` only
# while the inliner split its call above into static calls, and the
# `Union{Nothing,Plan}` residual argument ended that split (snag
# first-rebinding-64d82bfc: 41 ms and 0.55 MB at the first rebinding of the
# ShinyRK simulation graph in every process).
for graph_key in (Nothing, UInt64)
    precompile(_prepare_bound!, (PreparationCache, Graph, Any, Any, Any, Any, graph_key)) ||
        error("no _prepare_bound! method for a $graph_key graph key")
end
for template in (_BoundTemplate, PreparedKernel, Nothing)
    precompile(_bound_rebind, (_BoundEntry, template, Any, Any, Any)) ||
        error("no _bound_rebind method for a $template template")
end
for residual in (Plan, Nothing)
    precompile(_bound_residual_plan, (_BoundEntry, residual, Any)) ||
        error("no _bound_residual_plan method for a $residual residual")
end
for (f, types) in ((_bound_entry, (Plan, Any)), (_bound_hoisted, (_BoundEntry, Tuple)),
                   (_bound_residual, (_BoundEntry, Any)),
                   (_bound_build, (_BoundEntry, Any, Any, Any)),
                   (_partial_residual, (Plan, _PartialBoundary, Any)))
    precompile(f, types) || error("no $f method for $types")
end
