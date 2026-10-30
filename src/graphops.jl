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
An authored plate that reads a bound argument lowers for that argument's
class — scalar, vector, rank-N array or tuple — as if its port declared it, so
its residual is kept per class: a bound `1:n` and a bound `Vector` domain share
one, a bound scalar operand has another. Element types and extents never split
it. The cache is caller-owned and grows with the distinct boundaries (and those
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

# A loop, not a generator over the (heterogeneous) value tuple, which
# allocated at every element of a rebinding's three signatures; and over
# `_value_iter`, which compiles once for every selection width. A vector, not a
# tuple: a cache key holding an `NTuple{N,ValueId}` compiled the cache's `get`
# and `setindex!` again for every width (8 s at 655 ports).
Base.@nospecializeinfer function _sig(g::Graph, @nospecialize(vs))
    ids = ValueId[]
    for v in _value_iter(vs)
        push!(ids, canon_id(g, (v::Value).id))
    end
    ids
end
_pass_key(pass) = Base.issingletontype(typeof(pass)) ? typeof(pass) : objectid(pass)

"""
    prepare!(cache, g; have, want, passes=(), bound=(), type_stability=nothing, type_stability_level=nothing) -> PreparedKernel

Like `prepare`, but reuses a cached kernel when the same graph version and
effective have/want/pass signature has been prepared before under the same
`type_stability` and `type_stability_level` (see [`prepare`](@ref)).

With a non-empty `bound` (one `Value => data` pair or an iterable of them, as
for `prepare`), the returned kernel is the same residual kernel `prepare(g;
have, want, passes, bound)` builds, over the remaining HAVE ports. The cache
keeps everything that does not depend on the bound values — the plan, the
prepared prefix and the compiled residual — so repeating the call with new
data for the same ports runs only the prefix on that data. A residual whose
inner authored plates cache bound-only cell values shares its compiled body
too, across bindings whose caching decisions agree; one whose plates the pass
partitions by a data-bound branch, or replaces by a bind-time value, is still
specialized and compiled per binding and reuses the plan and prefix only.
"""
# Set while `partition_plan_kernel(spec; …)` prepares: the lowering emits the
# plan's body instead of the kernel's.
const _PLAN_MODE = :ReactiveKernels_partition_plan_mode
_plan_mode() = get(task_local_storage(), _PLAN_MODE, false)
# The exemplars `prepare` received, one value per HAVE port in order, while it
# lowers: the plan typing evaluates the plan on them (`_partition_plan_type`).
const _PLAN_EXEMPLARS = :ReactiveKernels_partition_plan_exemplars
# The record: the exemplars, and the plan port's position and type once the
# lowering has typed it (`typed`), which replaces the port's exemplar type
# (`Nothing`) in the kernel's exact signature.
struct _PlanExemplars
    exemplars::Tuple
    typed::Base.RefValue{Any}
end
_plan_exemplar_record(::Nothing) = nothing
_plan_exemplar_record(exemplars::Tuple) = _PlanExemplars(exemplars, Ref{Any}(nothing))
_plan_exemplars() = get(task_local_storage(), _PLAN_EXEMPLARS, nothing)
_with_plan_exemplars(f, ::Nothing) = f()
_with_plan_exemplars(f, record::_PlanExemplars) =
    task_local_storage(f, _PLAN_EXEMPLARS, record)
_plan_typed_exemplars(types, ::Nothing) = types
function _plan_typed_exemplars(types, record::_PlanExemplars)
    typed = record.typed[]
    typed === nothing && return types
    position, T = typed
    Tuple(i == position ? T : t for (i, t) in enumerate(types))
end

function prepare!(cache::PreparationCache, g::Graph; have = (), want = (),
                  passes = (), bound = (), on_error = nothing, type_stability = nothing,
                  type_stability_level = nothing)
    # One type for every selection's width before the closures capture them.
    haves, wants, binding = _value_list(have), _value_list(want), _bound_list(bound)
    _with_type_stability(type_stability, type_stability_level) do
        binding === () || return _prepare_bound!(cache, g, haves, wants, passes, binding,
                                                 objectid(g); on_error)
        key = (objectid(g), g.version, _sig(g, haves), _sig(g, wants),
               Tuple(objectid(p) for p in passes), on_error, _type_stability(),
               _type_stability_level(), _plan_mode())
        lock(cache.lock) do
            get!(cache.kernels, key) do
                prepare(g; have = haves, want = wants, passes = passes, on_error = on_error)
            end
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
    # Whether there is any pair, without building the pairs `_prepare_bound!`
    # builds (and checks) again.
    bound isa Pair || !isempty(bound)
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
    # A binding checked under one set of type-stability settings never serves
    # another: its rebindings are not checked again. A partition's plan kernel
    # (`partition_plan_kernel`, lowered in plan mode) is another program of
    # the same graph and binding.
    key = (:bound, graph_key, g.version, _sig(g, have), _sig(g, want),
           _sig(g, ports), Tuple(_pass_key(p) for p in passes), on_error,
           _type_stability(), _type_stability_level(), _plan_mode())
    # Locked blocks, not `lock(...) do` closures: a closure's type holds the
    # types it captures (here the signatures in `key`, and below the hoisted
    # values), so it compiled once per program and width.
    entry = @lock cache.lock begin
        existing = get(cache.bound, key, nothing)
        if existing === nothing
            existing = _bound_entry(
                _kernel_error_policy(plan(g; have = have, want = want), on_error), ports)
            cache.bound[key] = existing
        end
        existing
    end
    # The per-binding mathematics runs outside the lock.
    hoisted = _bound_hoisted(entry, data)
    classes = _partial_plate_classes(entry.boundary, hoisted)
    _, specialized, signature = _bound_residual(entry, cache, hoisted, classes)
    # A residual the inner-plate pass rewrote other than through its shared
    # plate rewrites (a data-bound branch partition, a whole-plate value)
    # bakes this binding's structure in.
    signature === nothing && return prepare(specialized; passes = passes)
    # Bindings with the same plate classes, finite loop schedules and plate
    # rewrites share a compiled residual. Extents, index patterns and cached
    # cell values stay runtime constants.
    template_key = (classes, _bound_index_run_classes(entry.index_run_slots, hoisted),
                    signature)
    fresh = nothing
    template = @lock cache.lock begin
        current = get(entry.templates, template_key, _UnbuiltTemplate())
        if current isa _UnbuiltTemplate
            current, fresh = _bound_build(entry, specialized, hoisted, passes)
            entry.templates[template_key] = current
        end
        current
    end
    fresh === nothing ? _bound_rebind(entry, template, specialized, hoisted, passes) : fresh
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
# A cache hit compares its key with the stored one. Signatures are vectors, so
# one key type serves every port count (here without passes).
for on_error in (Nothing, Symbol)
    S = Vector{ValueId}
    for graph_key in (Nothing, UInt64)
        K = Tuple{Symbol,graph_key,Int,S,S,S,Tuple{},on_error,Symbol,Symbol,Bool}
        precompile(isequal, (K, K)) || error("no isequal method for bound key $K")
    end
    K = Tuple{UInt64,Int,S,S,Tuple{},on_error,Symbol,Symbol,Bool}
    precompile(isequal, (K, K)) || error("no isequal method for cache key $K")
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
                   (_bound_residual, (_BoundEntry, PreparationCache, Any, Tuple)),
                   (_bound_fused_residual, (_BoundEntry, PreparationCache, Tuple)),
                   (_bound_build, (_BoundEntry, Any, Any, Any)),
                   (_partial_residual, (Plan, _PartialBoundary, Any)),
                   (_partial_plate_classes, (_PartialBoundary, Any)),
                   (_partial_narrow_plate_inputs, (Graph, _PartialBoundary, Tuple)),
                   (_partial_inner_plates, (Plan, Dict{ValueId,Any}, _PlateRewrites)),
                   (_plate_exclusions, (_PlateRewrites, Int, Plan)),
                   (_plate_analysis, (_PlateRewrites, _PlateAnalysisKey, Recipe,
                                      _AuthoredPlateOp, Set{Int}, Set{ValueId},
                                      Dict{ValueId,Any})),
                   (hash, (_PlateAnalysisKey, UInt)),
                   (isequal, (_PlateAnalysisKey, _PlateAnalysisKey)),
                   # what comparing two analysis keys' cell types dispatches to
                   (isequal, (Type, Type)),
                   (_bound_constants, (Plan,)),
                   (_rebind_table, (Any, Vector{Any})),
                   (_rebind_table, (Any, Vector{Recipe})),
                   (_rebound_table, (_RebindTable, Vector{Any})),
                   (_rebound_table, (_RebindTable, Vector{Recipe})),
                   (_rebound_table, (Vector{Any}, Vector{Int}, Vector{Any})),
                   (_rebound_table, (Vector{Any}, Vector{Int}, Vector{Recipe})),
                   (_ExecutableTable, (Vector{Any},)),
                   (_executable_entries, (_ExecutableTable, Vector{Any})),
                   (_executable_callables, (_ExecutableTable, Vector{Any})),
                   (_executed_prepared_kernel,
                    (Any, Tuple, Tuple, Tuple, Tuple, Plan, Expr, Tuple, Type, Any)))
    precompile(f, types) || error("no $f method for $types")
end
for value in (Any, Number, Tuple, AbstractArray)
    precompile(_bound_plate_class, (value,)) ||
        error("no _bound_plate_class method for $value")
end
