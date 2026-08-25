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
                 cost = r.cost, cse_key = r.cse_key, effectful = r.effectful)
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
"""
struct PreparationCache
    kernels::Dict{Any,PreparedKernel}
end
PreparationCache() = PreparationCache(Dict{Any,PreparedKernel}())

Base.length(c::PreparationCache) = length(c.kernels)

_sig(g::Graph, vs) = Tuple(canon_id(g, v.id) for v in _astuple(vs))

"""
    prepare!(cache, g; have, want, passes=()) -> PreparedKernel

Like `prepare`, but reuses a cached kernel when the same graph version and
effective have/want/pass signature has been prepared before.
"""
function prepare!(cache::PreparationCache, g::Graph; have = (), want = (), passes = ())
    key = (objectid(g), g.version, _sig(g, have), _sig(g, want),
           Tuple(objectid(p) for p in passes))
    get!(cache.kernels, key) do
        prepare(g; have = have, want = want, passes = passes)
    end
end
