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
    # Name sets and column tables this lowering keyed, each with its id
    # (`_signature_id`): `(kind, hash, length) => [signature => id]`.
    signatures::Dict{Tuple{Symbol,UInt,Int},Vector{Pair{Any,Int}}}
    nsignatures::Int
end
_StructuralCache() = _StructuralCache(ReentrantLock(), IdDict{Any,Any}(),
    Dict{Tuple{Symbol,UInt,Int},Vector{Pair{Any,Int}}}(), 0)

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

# Names in a canonical order, for keys. A Symbol is interned for the life of
# the process, so its identity orders it stably and compares as an integer.
_sorted_names(names) = Tuple(sort!(collect(Symbol, names); by = objectid))
_sorted_roles(roles) = Tuple(sort!(collect(roles); by = first))

# Every field but the anchor and the per-call facets, by identity. Binding
# interns the fields it rebuilds (`_interned`), so equal ones are identical.
@generated function _structure_fields(plan::StructuralPlan)
    Expr(:tuple, (:(getfield(plan, $(QuoteNode(f)))) for f in fieldnames(StructuralPlan)
        if f ∉ (:assignments, :columns, :n_obs))...)
end

# A set of names, or a column table's names with each value's type, as an
# integer: one lowering gives equal signatures one id, so a key holding it
# compares and hashes as an integer. Building a sorted tuple for every key
# cost more than many of the analyses it keyed (snag
# `bind-data-per-de-d92f13be`). One pass hashes the signature; a hit is
# compared with the stored signature, so two signatures never share an id.
# Ids are never reused, so emptying the table (past the entry limit) keeps
# every live key exact.
function _signature_id(same, stored, cache::_StructuralCache, key::Tuple{Symbol,UInt,Int})
    lock(cache.lock) do
        bucket = get!(Vector{Pair{Any,Int}}, cache.signatures, key)
        for (signature, id) in bucket
            same(signature) && return id
        end
        length(cache.signatures) >= _STRUCTURAL_CACHE_LIMIT && empty!(cache.signatures)
        id = cache.nsignatures += 1
        push!(get!(Vector{Pair{Any,Int}}, cache.signatures, key), stored() => id)
        return id
    end
end

# Names of a set (each once).
function _names_id(cache::_StructuralCache, names::AbstractSet{Symbol})
    h = UInt(0)
    for n in names
        h ⊻= hash(n)
    end
    return _signature_id(s -> length(s) == length(names) && all(in(s), names),
        () -> Set{Symbol}(names), cache, (:names, h, length(names)))
end
# Outside binding nothing is cached, and a key is never read.
_names_id(plan::StructuralPlan, names) =
    _structural_enabled() ? _names_id(_structural_cache(plan), names) : nothing

function _types_id(cache::_StructuralCache, columns::AbstractDict{Symbol})
    h = UInt(0)
    for (k, v) in columns
        h ⊻= hash(k, objectid(typeof(v)))
    end
    same(s) = length(s) == length(columns) &&
        all(((k, v),) -> get(s, k, Union{}) === typeof(v), columns)
    return _signature_id(same, () -> Dict{Symbol,DataType}(k => typeof(v) for (k, v) in columns),
        cache, (:types, h, length(columns)))
end

# The structure `plan`'s analyses read. `columns` chooses what of the column
# table: `:none` (an analysis that reads none of it), `:names` (the column
# names, no value), `:types` (the names and each value's type) or
# `:identity` (the table itself).
function _structural_key(cache::_StructuralCache, plan::StructuralPlan;
        columns::Symbol = :names, n_obs::Bool = false)
    cols = columns === :names ? _names_id(cache, keys(plan.columns)) :
        columns === :types ? _types_id(cache, plan.columns) :
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
    _structural(f, plan, tag; columns = :names, n_obs = false, fresh = true,
        same = isequal)

`f()`, an analysis of `plan` that `tag` (a tuple naming the analysis and its
other arguments, by value) identifies, computed once per structure while
`bind_data` runs. `f` must read only what `_structural_key` keys: no column
unless `columns = :names` (names), `:types` (names and value types) or
`:identity` (values; for a plan whose columns no step mutates), and `n_obs`
only when `n_obs = true`. Callers receive a fresh copy, so they may mutate
the result; `fresh = false` shares the cached value with callers that only
read it. `same(recomputed, cached)` is the hit check's comparison.
"""
function _structural(f, plan::StructuralPlan, tag::Tuple;
        columns::Symbol = :names, n_obs::Bool = false, fresh::Bool = true,
        same = isequal)
    _structural_enabled() || return f()
    cache = _structural_cache(plan)
    key = (tag, _structural_key(cache, plan; columns, n_obs))
    hit = lock(() -> get(cache.entries, key, _UNCACHED), cache.lock)
    if hit !== _UNCACHED
        _STRUCTURAL_CACHE_CHECK[] && !same(_analysis(f, columns), hit) && error(
            "structural cache: $(first(tag)) of this plan differs from the " *
            "value cached for its key")
        return fresh ? _fresh(hit) : hit
    end
    # Compute outside the lock: an analysis may consult the cache itself.
    value = _analysis(f, columns)
    lock(cache.lock) do
        length(cache.entries) >= _STRUCTURAL_CACHE_LIMIT && empty!(cache.entries)
        cache.entries[key] = value
    end
    return fresh ? _fresh(value) : value
end

# ── Data checks inside structural walks ──────────────────────────────
#
# Some validations walk a plan's expressions (which names a definition
# reads, whether a column sits where a scalar belongs) and test bound values
# only at a few leaves: a gather index's element type and range, a matrix's
# column count. `_data_checks` walks once per structure and records those
# leaves as checks that every binding runs on its own values, so rebinding
# costs the data checks alone (snag `rkppl-bind-data-96ae5ee3`).

const _DATA_CHECKS = :rkppl_data_checks
const _NO_DATA_CHECKS = gensym(:no_data_checks)

# A cached analysis returns a value: a data check it ran would be skipped on
# every hit, unless the analysis is keyed by the column table itself (which
# no step mutates), so it may run checks only then.
_analysis(f, columns::Symbol) = task_local_storage(f, _DATA_CHECKS,
    columns === :identity ? nothing : _NO_DATA_CHECKS)

"""
    _data_check(check, plan)

`check(plan)`, a test of `plan`'s bound values at a leaf of a structural
walk: run now, or recorded while [`_data_checks`](@ref) walks. `check` reads
columns and `n_obs` from its argument only; naming that argument `plan`
keeps it from capturing the walk's plan. It may capture what the walk
derived from the plan's other fields and column names.
"""
function _data_check(check, plan::StructuralPlan)
    sink = get(task_local_storage(), _DATA_CHECKS, nothing)
    sink === _NO_DATA_CHECKS && error("internal: a data check inside a " *
        "cached structural analysis")
    sink === nothing ? check(plan) : push!(sink, check)
    return nothing
end

"""
    _data_checks(walk, plan, tag)

`walk()`, a structural walk of `plan` that reads column names and value
types but no column value or `n_obs` (its data tests are
[`_data_check`](@ref) leaves), once per structure while `bind_data` runs;
then the recorded checks, against `plan`. `tag` names the walk, as for
[`_structural`](@ref).
"""
function _data_checks(walk, plan::StructuralPlan, tag::Tuple)
    checks = _structural(plan, tag; columns = :types, fresh = false,
            same = (a, b) -> map(typeof, a) == map(typeof, b)) do
        sink = Any[]
        task_local_storage(walk, _DATA_CHECKS, sink)
        sink
    end
    for check in checks
        check(plan)
    end
    return nothing
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

# ── Values of one binding ───────────────────────────────────────────
#
# Checks of one binding read some values of a bound column more than once:
# every gather along a `levels(g)` axis looks its labels up in the levels of
# `g`. `_bind_memo` computes each once per `bind_data` call. No step mutates
# a column while binding, so the column's identity keys its value.

const _BIND_MEMO = :rkppl_bind_memo

_with_bind_memo(f) = task_local_storage(f, _BIND_MEMO, IdDict{Any,Any}())

"""`f()`, a value of the bound column `col` that `tag` names: once per
binding while `bind_data` runs, computed afresh elsewhere. The value is
shared within the binding, so callers only read it."""
function _bind_memo(f, tag::Symbol, col)
    memo = get(task_local_storage(), _BIND_MEMO, nothing)
    memo === nothing && return f()
    return get!(f, memo::IdDict{Any,Any}, (tag, col))
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
