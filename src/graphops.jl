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
        merge!(out.values, g.values)
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
table. The cache is caller-owned and grows with the distinct boundaries (and,
for a residual containing authored plates, the distinct bound-value shapes)
prepared through it; it never observes the bound values afterwards. Lookups
and first-time preparations hold the cache's lock, so one cache may serve
concurrent tasks.
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
                  passes = (), bound = ())
    bound === () || return _prepare_bound!(cache, g, have, want, passes, bound)
    key = (objectid(g), g.version, _sig(g, have), _sig(g, want),
           Tuple(objectid(p) for p in passes))
    lock(cache.lock) do
        get!(cache.kernels, key) do
            prepare(g; have = have, want = want, passes = passes)
        end
    end
end

function _prepare_bound!(cache::PreparationCache, g::Graph, have, want, passes, bound)
    ports, data = _partial_bound_pairs(bound)
    isempty(ports) && return prepare!(cache, g; have = have, want = want, passes = passes)
    key = (:bound, objectid(g), g.version, _sig(g, have), _sig(g, want),
           _sig(g, ports), Tuple(objectid(p) for p in passes))
    entry = lock(cache.lock) do
        get!(cache.bound, key) do
            _bound_entry(plan(g; have = have, want = want), ports)
        end
    end
    # The per-binding mathematics runs outside the lock.
    hoisted = _bound_hoisted(entry, data)
    shape = _bound_shape_key(entry, data, hoisted)
    fresh = nothing
    template = lock(cache.lock) do
        get!(entry.templates, shape) do
            template, kernel = _bound_build(entry, hoisted, passes)
            fresh = kernel
            template
        end
    end
    fresh === nothing || return fresh
    template === nothing && return _bound_specialize(entry, hoisted, passes)
    _bound_rebind(entry, template, hoisted)
end
