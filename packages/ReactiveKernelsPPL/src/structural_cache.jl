# ── Structural analyses, computed once per plan structure ────────────
#
# Binding repeats analyses that a plan's structure decides alone: which
# definitions are module data, which inputs are model-level, the axes of
# array-valued definitions, the structure checks. `_structural` computes
# each once per structure, so rebinding a lowered plan to new data costs the
# data's work (snag `rkppl-bind-data-deb7f5e5`).
#
# Plans are immutable by contract: `bind_data` and `_with` return new plans
# and never mutate the containers they share. An analysis is then a function
# of the containers it reads, keyed here by their identity. Binding rebuilds
# some fields on every call (roles, conditioned names, sized responses, level
# maps) and interns them, so equal ones are identical. The column table is
# keyed by its names, or by identity for an analysis that reads its values;
# the observation count only where an analysis reads it.
#
# Every plan derived from one lowering shares its `assignments` table, so
# the cache hangs off that table, weakly: it lives while some plan of that
# lowering does. A key or value never holds that table, which would keep
# its own entry alive.

mutable struct _StructuralCache
    const lock::ReentrantLock
    entries::IdDict{Any,Any}
end
_StructuralCache() = _StructuralCache(ReentrantLock(), IdDict{Any,Any}())

# Entries per lowering. A lowering rebound to designs whose sized fields
# differ (other level counts) adds entries per design; past this many the
# cache starts over, so it never grows without bound.
const _STRUCTURAL_CACHE_LIMIT = 4096

# An identity-keyed weak table: `objectid(table) => [(weak table, cache)]`.
# (`WeakKeyDict` compares keys with `isequal`, which would share one cache
# between equal tables of unrelated lowerings.)
const _STRUCTURAL_CACHES = Dict{UInt,Vector{Tuple{WeakRef,_StructuralCache}}}()
const _STRUCTURAL_CACHES_LOCK = ReentrantLock()
const _STRUCTURAL_CACHES_ADDED = Ref(0)

function _structural_cache(plan::StructuralPlan)
    anchor = plan.assignments
    id = objectid(anchor)
    lock(_STRUCTURAL_CACHES_LOCK) do
        bucket = get!(Vector{Tuple{WeakRef,_StructuralCache}}, _STRUCTURAL_CACHES, id)
        for (ref, cache) in bucket
            ref.value === anchor && return cache
        end
        cache = _StructuralCache()
        push!(bucket, (WeakRef(anchor), cache))
        # Drop the caches of collected lowerings now and then.
        if (_STRUCTURAL_CACHES_ADDED[] += 1) % 256 == 0
            for (k, b) in _STRUCTURAL_CACHES
                filter!(e -> e[1].value !== nothing, b)
                isempty(b) && delete!(_STRUCTURAL_CACHES, k)
            end
        end
        return cache
    end
end

_sorted_names(names) = Tuple(sort!(collect(Symbol, names)))
_sorted_roles(roles) = Tuple(sort!(collect(roles); by = first))

# Every field but the anchor and the per-call facets, by identity. Binding
# interns the fields it rebuilds (`_interned`), so equal ones are identical.
@generated function _structure_fields(plan::StructuralPlan)
    Expr(:tuple, (:(getfield(plan, $(QuoteNode(f)))) for f in fieldnames(StructuralPlan)
        if f ∉ (:assignments, :columns, :n_obs))...)
end

# The structure `plan`'s analyses read. `columns` chooses what of the column
# table: `:none` (an analysis that reads none of it), `:names` (the column
# names, no value) or `:identity` (the table itself).
function _structural_key(plan::StructuralPlan; columns::Symbol = :names,
        n_obs::Bool = false)
    cols = columns === :names ? _sorted_names(keys(plan.columns)) :
        columns === :identity ? plan.columns :
        columns === :none ? nothing : throw(ArgumentError("columns = $columns"))
    return (_structure_fields(plan), cols, n_obs ? plan.n_obs : nothing)
end

# The cache serves `bind_data`, whose input is a finished lowering. Lowering
# itself fills its tables as it goes, so outside binding every analysis
# computes afresh.
_structural_enabled() = get(task_local_storage(), :rkppl_structural_cache, false)::Bool
_with_structural_cache(f) = _structural_enabled() ? f() :
    task_local_storage(f, :rkppl_structural_cache, true)

const _UNCACHED = gensym(:uncached)

# Tests set this to recompute every hit and fail on a stale one: a check of
# the key each analysis declares (`test_structural_cache.jl`).
const _STRUCTURAL_CACHE_CHECK = Ref(false)

"""
    _structural(f, plan, tag; columns = :names, n_obs = false, fresh = true)

`f()`, an analysis of `plan` that `tag` (a tuple naming the analysis and its
other arguments, by value) identifies, computed once per structure while
`bind_data` runs. `f` must read only what `_structural_key` keys: no column
unless `columns = :names` (names) or `:identity` (values; for a plan whose
columns no step mutates), and `n_obs` only when `n_obs = true`. Callers
receive a fresh copy, so they may mutate the result; `fresh = false` shares
the cached value with callers that only read it.
"""
function _structural(f, plan::StructuralPlan, tag::Tuple;
        columns::Symbol = :names, n_obs::Bool = false, fresh::Bool = true)
    _structural_enabled() || return f()
    cache = _structural_cache(plan)
    key = (tag, _structural_key(plan; columns, n_obs))
    hit = lock(() -> get(cache.entries, key, _UNCACHED), cache.lock)
    if hit !== _UNCACHED
        _STRUCTURAL_CACHE_CHECK[] && !isequal(f(), hit) && error(
            "structural cache: $(first(tag)) of this plan differs from the " *
            "value cached for its key")
        return fresh ? _fresh(hit) : hit
    end
    # Compute outside the lock: an analysis may consult the cache itself.
    value = f()
    lock(cache.lock) do
        length(cache.entries) >= _STRUCTURAL_CACHE_LIMIT && empty!(cache.entries)
        cache.entries[key] = value
    end
    return fresh ? _fresh(value) : value
end

"""`value`, or the equal object an earlier binding of this lowering produced
(`id` is `value`'s content): binding rebuilds these fields on every call,
and reusing the earlier object keeps the analyses keyed by it valid across
bindings. The result is a plan field, never mutated."""
function _interned(plan::StructuralPlan, id::Tuple, value)
    _structural_enabled() || return value
    cache = _structural_cache(plan)
    return lock(() -> get!(cache.entries, (:interned, id), value), cache.lock)
end

# A cached value is shared; each caller gets its own mutable containers.
_fresh(x) = deepcopy(x)
_fresh(x::Union{Bool,Nothing,Symbol,Number}) = x
_fresh(x::Union{AbstractSet{Symbol},AbstractVector{Symbol}}) = copy(x)
_fresh(x::Tuple) = map(_fresh, x)

"""The callable a data-only definition's call of `spec` runs: `prepare(spec)`,
prepared once per lowering (`spec(args...)` prepares on every call)."""
function _prepared_kernel(plan::StructuralPlan, spec::ReactiveKernels.KernelSpec)
    _structural_enabled() || return Base.invokelatest(ReactiveKernels.prepare, spec)
    cache = _structural_cache(plan)
    key = (:prepared_kernel, spec)
    hit = lock(() -> get(cache.entries, key, _UNCACHED), cache.lock)
    hit === _UNCACHED || return hit
    prepared = Base.invokelatest(ReactiveKernels.prepare, spec)
    lock(() -> (cache.entries[key] = prepared), cache.lock)
    return prepared
end
