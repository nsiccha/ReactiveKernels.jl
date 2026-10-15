# Reading earlier entries of a recurrence's carried arrays (history reads).
#
# The `@scan` loop syntax of `@kernel` bodies and ReactiveKernelsPPL's
# `@scan` blocks share this file. A step reads a carried array `a` at the loop
# index after writing it, at a backward lag `a[t - k]`, or at any EARLIER
# entry whose index provably lies in `1:(t - 1)`: an integer sum or difference
# of literals, the loop index and generator variables (`for s in lo:hi`).
# `a[s]` and `a[t - s]` under `for s in 1:(t - 1)`, and the slice
# `a[(t - k):(t - 1)]`, all qualify. A generator binds its iteration names in
# its body and filters, as in Julia. Front ends raise their own errors; these
# helpers only answer.

# One generator's iteration clauses `(binder, range)` and filter conditions,
# or `nothing` when a clause is not `binder = range`.
function _history_generator_clauses(gen::Expr)
    clauses = Tuple{Any,Any}[]
    conds = Any[]
    for it in gen.args[2:end]
        iters = if Meta.isexpr(it, :filter) && !isempty(it.args)
            push!(conds, it.args[1])
            it.args[2:end]
        else
            Any[it]
        end
        for i in iters
            (Meta.isexpr(i, :(=)) && length(i.args) == 2) || return nothing
            push!(clauses, (i.args[1], i.args[2]))
        end
    end
    return clauses, conds
end

# An index interval is a pair of affine bounds `α * t + β` in the loop index
# `t`, each stored as `(α, β)`.
_history_iv_add(a, b) = ((a[1][1] + b[1][1], a[1][2] + b[1][2]),
    (a[2][1] + b[2][1], a[2][2] + b[2][2]))
_history_iv_neg(a) = ((-a[2][1], -a[2][2]), (-a[1][1], -a[1][2]))

# The interval of an integer index expression over the loop index and the
# generator binders in `env` (binder => interval, or `nothing` when its range
# proves no bound); `nothing` when the expression is not such a sum.
function _history_interval(ex, loopvar, env)
    ex isa Integer && !(ex isa Bool) && return ((0, Int(ex)), (0, Int(ex)))
    ex === loopvar && return ((1, 0), (1, 0))
    ex isa Symbol && return get(env, ex, nothing)
    (Meta.isexpr(ex, :call) && length(ex.args) >= 2) || return nothing
    f = ex.args[1]
    ivs = Any[_history_interval(x, loopvar, env) for x in ex.args[2:end]]
    any(isnothing, ivs) && return nothing
    f === :+ && return foldl(_history_iv_add, ivs)
    f === :- && length(ivs) == 1 && return _history_iv_neg(only(ivs))
    f === :- && length(ivs) == 2 && return _history_iv_add(ivs[1], _history_iv_neg(ivs[2]))
    return nothing
end

_history_is_slice(idx) = Meta.isexpr(idx, :call) && length(idx.args) == 3 &&
    idx.args[1] === :(:)

# The interval a `lo:hi` range gives its binder or slice, or `nothing`.
function _history_range_interval(rng, loopvar, env)
    _history_is_slice(rng) || return nothing
    a = _history_interval(rng.args[2], loopvar, env)
    b = _history_interval(rng.args[3], loopvar, env)
    (a === nothing || b === nothing) && return nothing
    return (a[1], b[2])
end

# True when the bounds lie in `1:(t - 1)` at every loop index `t >= lo`:
# `lower >= 1` and `upper <= t - 1`, each checked where it is tightest (both
# are affine in `t`).
function _history_proves_earlier(lower, upper, lo::Integer)
    (al, bl), (au, bu) = lower, upper
    (al >= 0 && al * lo + bl >= 1) || return false
    (au <= 1 && (au - 1) * lo + bu + 1 <= 0) || return false
    return true
end

# Is `idx` (a point or a `lo:hi` slice) provably earlier than the loop index?
function _history_is_earlier(idx, loopvar, lo::Integer, env)
    bounds = _history_is_slice(idx) ? _history_range_interval(idx, loopvar, env) :
        _history_interval(idx, loopvar, env)
    return bounds !== nothing && _history_proves_earlier(bounds..., lo)
end

# A backward lag `loopvar - k`: the `Int` depth, a `Symbol` depth that is not a
# generator binder (bound data), or `nothing`.
function _history_lag_form(idx, loopvar, env)
    (Meta.isexpr(idx, :call) && length(idx.args) == 3 && idx.args[1] === :- &&
        idx.args[2] === loopvar) || return nothing
    k = idx.args[3]
    k isa Int && !(k isa Bool) && return k
    k isa Symbol && k !== loopvar && !haskey(env, k) && return k
    return nothing
end

# The environment inside generator `gen`: each iteration name maps to the
# interval its range proves, read in the enclosing environment.
function _history_generator_env(gen::Expr, loopvar, env)
    inner = copy(env)
    parts = _history_generator_clauses(gen)
    parts === nothing && return inner
    for (v, rng) in parts[1]
        v isa Symbol && (inner[v] = _history_range_interval(rng, loopvar, inner))
    end
    return inner
end

# Visit every read `state[idx]` of a carried array in `ex` as
# `f(state, idx, env)`, with `env` the generator binders in scope.
function _history_foreach_read(f, ex, states, loopvar, env = Dict{Symbol,Any}())
    ex isa Expr || return nothing
    ex.head === :quote && return nothing
    if ex.head === :generator
        parts = _history_generator_clauses(ex)
        if parts !== nothing
            inner = copy(env)
            for (v, rng) in parts[1]
                _history_foreach_read(f, rng, states, loopvar, inner)
                v isa Symbol && (inner[v] = _history_range_interval(rng, loopvar, inner))
            end
            for c in (ex.args[1], parts[2]...)
                _history_foreach_read(f, c, states, loopvar, inner)
            end
            return nothing
        end
    end
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] in states
        f(ex.args[1], ex.args[2], env)
        _history_foreach_read(f, ex.args[2], states, loopvar, env)
        return nothing
    end
    for a in ex.args
        _history_foreach_read(f, a, states, loopvar, env)
    end
    return nothing
end

# Locals of `names` that `ex` reads, skipping names a generator rebinds.
function _history_local_reads!(out::Set{Symbol}, ex, names, bound = Set{Symbol}())
    if ex isa Symbol
        (ex in names && !(ex in bound)) && push!(out, ex)
        return out
    end
    ex isa Expr || return out
    ex.head === :quote && return out
    if ex.head === :generator
        parts = _history_generator_clauses(ex)
        if parts !== nothing
            inner = copy(bound)
            for (v, rng) in parts[1]
                _history_local_reads!(out, rng, names, inner)
                v isa Symbol && push!(inner, v)
            end
            for c in (ex.args[1], parts[2]...)
                _history_local_reads!(out, c, names, inner)
            end
            return out
        end
    end
    for a in ex.args
        _history_local_reads!(out, a, names, bound)
    end
    return out
end

# --- Fold planning --------------------------------------------------------
#
# A recurrence with several carried arrays is reconstructed one array per RK
# `scan` ("fold"). The fold for array `a` can read `a`'s own earlier entries
# through the scan's `history` argument, and every array produced by an
# earlier fold from its finished trajectory. Each statement is summarized by
# the carried array it writes (or `nothing`), the locals it defines, whether
# it samples, and the expressions it reads.

# Per-statement reads: the locals each statement reads and the carried arrays
# it reads at the current step, at a lag and at earlier entries.
function _history_statement_reads(states, loopvar, statements)
    locals = Set{Symbol}()
    for st in statements
        union!(locals, st.defines)
    end
    return map(statements) do st
        loc, cur, lag, hist = Set{Symbol}(), Set{Symbol}(), Set{Symbol}(), Set{Symbol}()
        for r in st.reads
            _history_local_reads!(loc, r, locals)
            _history_foreach_read(r, states, loopvar) do a, idx, env
                idx === loopvar ? push!(cur, a) :
                    _history_lag_form(idx, loopvar, env) !== nothing ?
                    push!(lag, a) : push!(hist, a)
            end
        end
        (; locals = loc, current = cur, lagged = lag, history = hist)
    end
end

_history_has_reads(reads) = any(r -> !isempty(r.history), reads)

# The statements a fold executes: those producing `output` (or every sampled
# statement, for a density fold with `output === nothing`) and everything they
# read, except arrays already `completed`. `nothing` when an executed
# statement reads the history of an array that is neither `output` nor
# completed.
function _history_fold_closure(statements, reads, output, completed)
    writer = Dict{Symbol,Int}()
    for (i, st) in enumerate(statements)
        st.writes === nothing || (writer[st.writes] = i)
        for d in st.defines
            writer[d] = i
        end
    end
    work = output === nothing ?
        Int[i for (i, st) in enumerate(statements) if st.sampled] :
        Int[writer[output]]
    exec = Set{Int}()
    while !isempty(work)
        i = pop!(work)
        i in exec && continue
        push!(exec, i)
        r = reads[i]
        for l in r.locals
            haskey(writer, l) && push!(work, writer[l])
        end
        for a in union(r.current, r.lagged)
            a in completed || push!(work, writer[a])
        end
        for a in r.history
            a in completed && continue
            a === output || return nothing
            push!(work, writer[a])
        end
    end
    return exec
end

# The order in which the carried arrays are produced, each with the
# statements its fold executes, or `nothing` when some arrays read one
# another's earlier entries (no fold can produce either first). Among the
# feasible arrays the one with the fewest statements goes first.
function _history_fold_order(states, statements, reads)
    completed = Set{Symbol}()
    order = Tuple{Symbol,Set{Int}}[]
    while length(order) < length(states)
        best = nothing
        for a in states
            a in completed && continue
            exec = _history_fold_closure(statements, reads, a, completed)
            exec === nothing && continue
            (best === nothing || length(exec) < length(best[2])) && (best = (a, exec))
        end
        best === nothing && return nothing
        push!(order, best)
        push!(completed, best[1])
    end
    return order
end

# Does the fold executing `exec` read `a`'s earlier entries?
_history_fold_reads_own(reads, exec, a) = any(i -> a in reads[i].history, exec)

# Rebuild generator `gen`: each range maps through `range(ex, env)` in the
# environment of the clauses before it, and the body and filters through
# `body(ex, env)` inside every binder.
function _history_map_generator(range, body, gen::Expr, loopvar, env)
    inner = copy(env)
    function clause(it)
        (Meta.isexpr(it, :(=)) && length(it.args) == 2) || return it
        v, rng = it.args
        mapped = Expr(:(=), v, range(rng, copy(inner)))
        v isa Symbol && (inner[v] = _history_range_interval(rng, loopvar, inner))
        return mapped
    end
    parts = Any[]
    for it in gen.args[2:end]
        if Meta.isexpr(it, :filter) && !isempty(it.args)
            push!(parts, (it.args[1], Any[clause(i) for i in it.args[2:end]]))
        else
            push!(parts, clause(it))
        end
    end
    args = Any[p isa Tuple ? Expr(:filter, body(p[1], inner), p[2]...) : p for p in parts]
    return Expr(:generator, body(gen.args[1], inner), args...)
end
