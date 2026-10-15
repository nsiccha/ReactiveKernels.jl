# Sequential-recurrence surface: `@scan begin <setup>; for t in lo:hi … end end`.
#
# The RK→BRM thin-layer counterpart of `@plate` (independent cells) for the
# SEQUENTIAL primitive, shapes copied from StanBlocks `@scan` verbatim (§35).
# This file is the surface FRONT END: it parses the annotated `@scan` block
# AST into a `ScanSpec` IR node (`parse_scan_block` is a pure syntactic
# function, unit-tested in isolation). Lowering tiers live with their owners:
# surface wiring in `surface.jl`, the array-sampled layout slice in
# `layout.jl`, the centered density + the non-centered `scan(...)`
# carry-fold reconstruction in `generator.jl`, LP use via `ScanSummandTerm`
# (the SB-`ar` slice). The cross-lane coordination gate this header once
# named is resolved (user chose independent+reconcile, scan-lane decision
# `0bowtxh`); the centered primitive landed on main @ `7fd989c`.
#
# Grammar — one or more carried arrays (a tuple carry), literal or bound-integer backward lags:
#   @scan begin
#       x[1] = 0.0                        # setup: every carried array is seeded at
#       d[1] = 0.0                        #   1..m, by a deterministic `=` or a
#       h[1] ~ Normal(0, 1)               #   sampled `~` fill
#       for t in (m+1):T                  # the recurrence; T literal Int or a data length name
#           h[t] ~ Normal(phi*h[t-1], s)  # centered: sample the carried array at the loop index
#           # or, non-centered:
#           #   eps ~ Normal(0, 1)        # per-step local innovation (fresh bare name)
#           #   d[t] = beta*d[t-1] + s*eps # deterministic carry writes, run in order:
#           #   x[t] = x[t-1] + d[t]      #   `d[t]` reads the value written above
#       end
#   end
# A sampled seed or step takes any scalar prior a declaration takes, including
# `Flat()`, half priors and `truncated` / `restricted` with fixed bounds.
# Statements run in order, as in a Julia loop body: a step reads a carried
# array's backward lag `a[t-k]` (1 ≤ k ≤ m; a literal or bound integer), its current value `a[t]`
# once an earlier step of the same iteration wrote it, any earlier entry
# whose index provably lies in `1:(t-1)` (a history read: `a[s]` or
# `a[t - s]` under a generator `for s in 1:(t - 1)`, or a slice
# `a[(t - k):(t - 1)]`), and locals defined by earlier steps. Each carried
# array is written exactly once per step. The observation lives OUTSIDE the
# block (`y .~ Normal.(h, 1)`), so it is not part of `parse_scan_block`.

# The `ScanSpec` / `ScanStep` / `ScanSetup` IR structs live in `contract.jl`
# (alongside the other IR specs, so `StructuralPlan` can reference them); this
# file owns only the surface PARSER that produces a `ScanSpec`.

_scan_fail(msg) = _sfail("@scan: " * msg)

# `Dist(args…)` → (internal family symbol, positional argument expressions,
# support override). `Flat()`, `HalfNormal(s)` / `HalfCauchy(s)` and the
# `truncated(D, lo, hi)` / `restricted(D, lo, hi)` wrappers read exactly as a
# scalar declaration's prior (`_parse_prior`). Arguments stay value
# expressions, so they may read lags, locals and the loop index; bounds stay
# expressions too, and the layout requires them fixed across steps.
function _scan_parse_dist(call, what)
    p = _parse_prior(:_ppl_scan_draw, call, "@scan: $what", identity,
        _scan_bound, fam -> _scan_fail("$what: unknown distribution " *
            "`$(repr(fam))` (admitted: Normal, Cauchy, Exponential, Gamma, " *
            "LogNormal, Beta, InverseGamma, StudentT, TDist, Laplace, " *
            "Logistic, Uniform, Weibull, HalfNormal, HalfCauchy, Flat, " *
            "truncated, restricted)"))
    return p.family, collect(Any, values(p.args)), p.support_override
end

# A `truncated` / `restricted` bound: an infinity literal is `±Inf`; any other
# bound stays a value expression, which the layout requires to be fixed
# across steps.
_scan_bound(b) = b === :Inf ? Inf : b isa Real ? Float64(b) :
    Meta.isexpr(b, :call, 2) && b.args[2] === :Inf && b.args[1] in (:-, :+) ?
        (b.args[1] === :- ? -Inf : Inf) : b

"""
    parse_scan_block(block; label = :scan) -> ScanSpec

Parse the inner `begin … end` block of a `@scan` annotated loop into a
[`ScanSpec`](@ref). Pure and syntactic — see this file's header for the grammar
and boundaries. Every rejection raises `SurfaceLoweringError`.
"""
function parse_scan_block(block; label::Union{Symbol,Nothing} = nothing)
    (block isa Expr && block.head === :block) ||
        _scan_fail("expected a `begin … end` block, got $(repr(block))")
    stmts = Any[a for a in block.args if !(a isa LineNumberNode)]
    isempty(stmts) &&
        _scan_fail("empty block — need seed fill(s) then a trailing `for`")
    forx = last(stmts)
    (forx isa Expr && forx.head === :for) || _scan_fail(
        "the block must END with the recurrence `for` loop " *
        "(setup fills come first)")
    setups_ast = stmts[1:end-1]
    isempty(setups_ast) && _scan_fail(
        "need at least one seed fill (`state[1] ~ Dist(…)` or " *
        "`state[1] = value`) before the loop")

    states, setup, m = _parse_scan_setup(setups_ast)
    loopvar, lo, hi = _parse_scan_for_head(forx, m)

    body = forx.args[2]
    (body isa Expr && body.head === :block) || _scan_fail("malformed loop body")
    step = ScanStep[]
    for s in body.args
        s isa LineNumberNode && continue
        push!(step, _parse_scan_step(s, states, loopvar, lo))
    end
    isempty(step) && _scan_fail("empty recurrence body")
    any(st -> st.indexed, step) || _scan_fail(
        "the loop never writes a carried array (`$(first(states))[$(loopvar)]`; " *
        "each step is a fresh local); a scan must thread the state")
    _scan_check_order(step, states, loopvar)

    maxlag = _scan_maxlag(step, states, loopvar)
    any(k -> k isa Symbol, Iterators.flatten(values(_scan_lags(step, states, loopvar)))) &&
        (maxlag = m)
    m >= maxlag || _scan_fail(
        "the recurrence reads a lag of $(maxlag) but only $(m) initial " *
        "value(s) are seeded per carried array; add fills up to index $(maxlag)")
    return ScanSpec(states, loopvar, lo, hi, setup, step, maxlag,
        something(label, first(states)))
end

# Seed fills: `a[k] ~ Dist(…)` (sampled) or `a[k] = value` (deterministic).
# Each carried array's fills are contiguous `a[1], a[2], …` in order; the
# arrays may interleave, and every array is seeded to the same depth m (the
# loop starts at m + 1). A deterministic seed reads scalars and earlier seeds
# (`d[1] = x[1]`), never the loop's own values.
function _parse_scan_setup(setups_ast)
    states = Symbol[]
    setup = ScanSetup[]
    next = Dict{Symbol,Int}()
    # Every array a fill seeds is carried, whichever fill comes first: a
    # seed reading `d[1]` above `d`'s own fill reads a value not seeded yet.
    carried = Set{Symbol}()
    for s in setups_ast
        lhs = s isa Expr && s.head in (:call, :(=)) ?
            (s.head === :call ? (length(s.args) == 3 ? s.args[2] : nothing) :
                s.args[1]) : nothing
        (lhs isa Expr && lhs.head === :ref && !isempty(lhs.args) &&
            lhs.args[1] isa Symbol) && push!(carried, lhs.args[1])
    end
    for (k, s) in enumerate(setups_ast)
        sampled = s isa Expr && s.head === :call && length(s.args) == 3 &&
            s.args[1] === :~
        assigned = s isa Expr && s.head === :(=) && length(s.args) == 2
        (sampled || assigned) || _scan_fail(
            "setup statement $(k) must be a seed fill `state[i] ~ Dist(…)` " *
            "or `state[i] = value`, got $(repr(s))")
        lhs = sampled ? s.args[2] : s.args[1]
        (lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2) ||
            _scan_fail("setup LHS must be `state[<integer>]`, got $(repr(lhs))")
        nm, idx = lhs.args[1], lhs.args[2]
        nm isa Symbol || _scan_fail("setup array name must be a Symbol")
        idx isa Int || _scan_fail(
            "setup index must be a literal integer, got $(repr(idx)) " *
            "(a slice `state[1:p]` is not supported yet — write the lines out)")
        want = get(next, nm, 1)
        want == 1 && push!(states, nm)
        idx == want || _scan_fail(
            "setup fills of `$(nm)` must be contiguous `$(nm)[1], $(nm)[2], …`; " *
            "expected index $(want), got $(idx)")
        if sampled
            fam, args, support = _scan_parse_dist(s.args[3], "setup `$(nm)[$(idx)]`")
            for a in (args..., _support_args(support)...)
                _scan_check_seed_reads(a, carried, next, "setup `$(nm)[$(idx)]`")
            end
            push!(setup, ScanSetup(nm, idx, :sample, fam, args, nothing, support))
        else
            _scan_check_seed_reads(s.args[2], carried, next,
                "setup `$(nm)[$(idx)]`")
            push!(setup, ScanSetup(nm, idx, :assign, nothing, nothing,
                s.args[2]))
        end
        next[nm] = want + 1
    end
    depths = Dict(a => next[a] - 1 for a in states)
    m = depths[first(states)]
    for a in states
        depths[a] == m || _scan_fail(
            "every carried array is seeded to the same depth (the loop starts " *
            "one past it): `$(first(states))` has $(m) fill(s), `$(a)` has " *
            "$(depths[a])")
    end
    return states, setup, m
end

# A seed reads scalars (parameters, definitions, literals) and seeds filled
# above it (`a[j]`, j a literal index already seeded); anything else of a
# carried array is not a value yet.
function _scan_check_seed_reads(ex, states, next, what)
    if ex isa Symbol
        ex in states && _scan_fail(
            "$what reads the whole carried array `$(ex)`; read a seeded " *
            "element (`$(ex)[1]`)")
        return nothing
    end
    ex isa Expr || return nothing
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] in states
        j = ex.args[2]
        (j isa Int && 1 <= j < get(next, ex.args[1], 1)) || _scan_fail(
            "$what reads `$(ex)`, which is not seeded above it (a seed reads " *
            "scalars and earlier seeds by literal index)")
        return nothing
    end
    for a in ex.args
        _scan_check_seed_reads(a, states, next, what)
    end
    return nothing
end
function _parse_scan_for_head(forx, m)
    (forx.args[1] isa Expr && forx.args[1].head === :(=)) ||
        _scan_fail("malformed `for` head")
    loopvar = forx.args[1].args[1]
    rng = forx.args[1].args[2]
    loopvar isa Symbol || _scan_fail("loop variable must be a Symbol")
    (rng isa Expr && rng.head === :call && rng.args[1] === :(:) &&
     length(rng.args) == 3) ||
        _scan_fail("loop range must be `lo:hi`, got $(repr(rng))")
    lo, hi = rng.args[2], rng.args[3]
    lo isa Int || _scan_fail("loop start must be a literal integer")
    lo == m + 1 || _scan_fail(
        "loop must start at $(m + 1) (one past the $(m) seed fill(s)), got $(lo)")
    (hi isa Int || hi isa Symbol) || _scan_fail(
        "loop bound must be a literal integer or a data length name, " *
        "got $(repr(hi))")
    return loopvar, lo, hi
end

function _parse_scan_step(s, states, loopvar, lo)
    if s isa Expr && s.head === :call && length(s.args) == 3 && s.args[1] === :~
        lhs = s.args[2]
        target, indexed = _scan_step_lhs(lhs, states, loopvar, "`~`")
        fam, args, support = _scan_parse_dist(s.args[3], "step `$(_scan_lhs_show(lhs))`")
        for a in (args..., _support_args(support)...)
            _scan_check_reads(a, states, loopvar, lo)
        end
        return ScanStep(:sample, target, indexed, fam, args, nothing, support)
    elseif s isa Expr && s.head === :(=) && length(s.args) == 2
        lhs = s.args[1]
        target, indexed = _scan_step_lhs(lhs, states, loopvar, "`=`")
        _scan_check_reads(s.args[2], states, loopvar, lo)
        return ScanStep(:assign, target, indexed, nothing, nothing, s.args[2])
    elseif s isa Expr && s.head in (:for, :while)
        _scan_fail("nested loops in a `@scan` body are not supported yet; " *
                   "write an inner sum as a generator, " *
                   "`sum(a[s] * w[$(loopvar) - s] for s in 1:($(loopvar) - 1))`")
    elseif s isa Expr && s.head in (:if, :elseif)
        _scan_fail("branches in a `@scan` body are not supported")
    else
        _scan_fail("a `@scan` step must be `state[$(loopvar)] ~ Dist(…)`, " *
                   "`state[$(loopvar)] = …`, a fresh `local ~ Dist(…)`, or a " *
                   "local assignment; got $(repr(s))")
    end
end

# Classify a step LHS. Returns (target, indexed). `indexed` means a carried
# array is written at the current loop index `state[loopvar]`.
function _scan_step_lhs(lhs, states, loopvar, what)
    if lhs isa Symbol
        lhs in states && _scan_fail(
            "cannot rebind the whole carried array `$(lhs)` in the loop; " *
            "write `$(lhs)[$(loopvar)]`")
        lhs === loopvar && _scan_fail(
            "cannot rebind the loop variable `$(loopvar)` in the loop")
        return lhs, false                      # a fresh per-step local
    elseif lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2
        nm, idx = lhs.args[1], lhs.args[2]
        nm in states || _scan_fail(
            "$(what) writes `$(nm)[…]`, which no setup fill seeds; seed every " *
            "carried array before the loop (`$(nm)[1] = …` or " *
            "`$(nm)[1] ~ Dist(…)`)")
        idx === loopvar || _scan_fail(
            "the carried write must be at the loop index `$(nm)[$(loopvar)]`, " *
            "got `$(_scan_lhs_show(lhs))` (a scan writes each index once, in order)")
        return nm, true
    else
        _scan_fail("$(what) left-hand side must be `state[$(loopvar)]` or a " *
                   "fresh local name, got $(repr(lhs))")
    end
end

_scan_lhs_show(lhs) = string(lhs)

# --- Reads of carried arrays ---------------------------------------------------
#
# A step reads a carried array `a` at the loop index (`a[t]`, once an earlier
# step wrote it), at a backward lag (`a[t - k]`, k a literal or bound integer
# within the seed depth), or at any EARLIER entry whose index provably lies in
# `1:(t - 1)` (a history read). History indices are integer sums and
# differences of literals, the loop index and generator variables
# (`for s in lo:hi`), so `a[s]` and `a[t - s]` under `for s in 1:(t - 1)`, and
# the slice `a[(t - k):(t - 1)]`, all qualify. A generator or comprehension
# binds its iteration names in its body and filters, as in Julia.

# One generator's iteration clauses `(binder, range)` and filter conditions
# (the shared helpers live in ReactiveKernels' `scan_history.jl`).
function _scan_generator_clauses(gen)
    parts = ReactiveKernels._history_generator_clauses(gen)
    parts === nothing && _scan_fail(
        "a generator in a `@scan` step iterates `name in range`, got $(repr(gen))")
    return parts
end

# The binder environment inside generator `gen`: each iteration name maps to
# the interval its range proves. Names that shadow the loop index, a carried
# array or (checked in `_scan_check_order`) a step local are refused.
function _scan_generator_env(gen, states, loopvar, env)
    clauses, _ = _scan_generator_clauses(gen)
    for (v, _) in clauses
        v isa Symbol || _scan_fail(
            "a generator in a `@scan` step iterates one name, got $(repr(v))")
        v === loopvar && _scan_fail(
            "the generator variable `$(v)` shadows the loop variable")
        v in states && _scan_fail(
            "the generator variable `$(v)` shadows the carried array `$(v)`")
    end
    return _history_generator_env(gen, loopvar, env)
end

# Classify the read `state[idx]`: `:current`, a backward lag (the `Int` or
# data `Symbol` depth), or `:history`. Anything else fails.
function _scan_read_kind(idx, state, loopvar, lo, env)
    idx === loopvar && return :current
    k = _history_lag_form(idx, loopvar, env)
    if k !== nothing
        (k isa Symbol || k >= 1) || _scan_fail(
            "lag depth in `$(state)[$(loopvar)-…]` must be a literal or data " *
            "integer ≥ 1, got $(repr(k))")
        return k
    end
    if Meta.isexpr(idx, :call) && length(idx.args) == 3 && idx.args[1] === :+ &&
            idx.args[2] === loopvar && idx.args[3] isa Int && idx.args[3] >= 1
        _scan_fail("forward reference `$(state)[$(idx)]` — a scan reads only " *
                   "values written before the current step")
    end
    _history_is_earlier(idx, loopvar, lo, env) && return :history
    _scan_fail("`$(state)[$(idx)]` reads the carried array at an index not " *
        "proven to lie before the loop index. A step reads " *
        "`$(state)[$(loopvar)]` after writing it, a backward lag " *
        "`$(state)[$(loopvar) - k]`, or an earlier entry whose index lies in " *
        "`1:($(loopvar) - 1)` for every step: `$(state)[s]` or " *
        "`$(state)[$(loopvar) - s]` under `for s in 1:($(loopvar) - 1)`, or a " *
        "slice `$(state)[($(loopvar) - k):($(loopvar) - 1)]`")
end

# Walk an expression rejecting illegal reads of a carried array: a bare read
# of the whole array, a forward lag, or an index not proven earlier than the
# loop index. A carried read's index may itself read only data, literals, the
# loop index and generator binders.
function _scan_check_reads(ex, states, loopvar, lo, env = Dict{Symbol,Any}())
    if ex isa Symbol
        (ex in states && !haskey(env, ex)) && _scan_fail(
            "bare read of the carried array `$(ex)` inside its own loop; read " *
            "a backward lag `$(ex)[$(loopvar)-1]` or its history " *
            "`$(ex)[1:($(loopvar) - 1)]`")
        return nothing
    end
    ex isa Expr || return nothing
    if ex.head === :generator
        inner = _scan_generator_env(ex, states, loopvar, env)
        clauses, conds = _scan_generator_clauses(ex)
        for (_, rng) in clauses
            _scan_check_reads(rng, states, loopvar, lo, inner)
        end
        for c in (ex.args[1], conds...)
            _scan_check_reads(c, states, loopvar, lo, inner)
        end
        return nothing
    end
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] in states
        _scan_check_reads(ex.args[2], states, loopvar, lo, env)
        _scan_read_kind(ex.args[2], ex.args[1], loopvar, lo, env)   # validates
        return nothing
    end
    for a in ex.args
        _scan_check_reads(a, states, loopvar, lo, env)
    end
    return nothing
end

# Generator binders anywhere in `ex`.
function _scan_binders!(out::Set{Symbol}, ex)
    ex isa Expr || return out
    if ex.head === :generator
        for (v, _) in first(_scan_generator_clauses(ex))
            v isa Symbol && push!(out, v)
        end
    end
    for a in ex.args
        _scan_binders!(out, a)
    end
    return out
end

# Standard-Julia order inside one iteration: a local is read only after the
# step that defines it, a carried array's current value `a[t]` only after the
# step that writes it, each carried array is written exactly once, and a local
# is defined once and never shadows a carried array. A generator variable
# never shadows a step local.
function _scan_check_order(step, states, loopvar)
    locals = Set{Symbol}(st.target for st in step if !st.indexed)
    defined = Set{Symbol}()
    written = Set{Symbol}()
    for st in step
        for r in _scan_reads(st)
            shadow = intersect(_scan_binders!(Set{Symbol}(), r), locals)
            isempty(shadow) || _scan_fail(
                "the generator variable `$(first(shadow))` shadows a step local " *
                "of the same name; rename one of them")
            _scan_check_defined(r, states, loopvar, locals, defined, written)
        end
        if st.indexed
            st.target in written && _scan_fail(
                "`$(st.target)[$(loopvar)]` is written twice in one step; each " *
                "carried array is written exactly once per step")
            push!(written, st.target)
        else
            st.target in defined && _scan_fail(
                "the local `$(st.target)` is defined twice in one step; give " *
                "each step local its own name")
            push!(defined, st.target)
        end
    end
    for a in states
        a in written || _scan_fail(
            "the carried array `$(a)` is seeded but never written in the loop " *
            "(write `$(a)[$(loopvar)] = …` or `$(a)[$(loopvar)] ~ …`)")
    end
    return nothing
end

function _scan_check_defined(ex, states, loopvar, locals, defined, written)
    if ex isa Symbol
        (ex in locals && !(ex in defined)) && _scan_fail(
            "the local `$(ex)` is read before the step that defines it")
        return nothing
    end
    ex isa Expr || return nothing
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] in states
        (ex.args[2] === loopvar && !(ex.args[1] in written)) && _scan_fail(
            "`$(ex.args[1])[$(loopvar)]` reads the value being written this " *
            "step before it is written; read a backward lag " *
            "`$(ex.args[1])[$(loopvar)-1]`, or write `$(ex.args[1])[$(loopvar)]` " *
            "in an earlier step")
        return nothing
    end
    args = ex.head === :call ? ex.args[2:end] : ex.args
    for a in args
        _scan_check_defined(a, states, loopvar, locals, defined, written)
    end
    return nothing
end

# Parse the index of a validated lag read `state[loopvar - k]`; return `k`.
function _scan_lag_of(idx, state, loopvar)
    k = _history_lag_form(idx, loopvar, Dict{Symbol,Any}())
    k === nothing && _scan_fail("`$(state)` index must be the backward lag " *
        "`$(loopvar)-<k>`, got $(repr(idx))")
    return k
end

# Distinct backward lags `a[loopvar - k]` read across a scan's steps, per
# carried array (current-value and history reads are not lags).
function _scan_lags(step, states, loopvar)
    lags = Dict{Symbol,Set{Union{Int,Symbol}}}(a => Set{Union{Int,Symbol}}() for a in states)
    for st in step, r in _scan_reads(st)
        _history_foreach_read(r, states, loopvar) do a, idx, env
            k = _history_lag_form(idx, loopvar, env)
            k === nothing || push!(lags[a], k)
        end
    end
    return lags
end

# The statements of a scan in the shared fold-planning form
# (`ReactiveKernels._history_statement_reads`).
_scan_fold_statements(s::ScanSpec) = [(; writes = st.indexed ? st.target : nothing,
    defines = st.indexed ? Set{Symbol}() : Set{Symbol}((st.target,)),
    sampled = st.kind === :sample, reads = _scan_reads(st)) for st in s.step]

_scan_statement_reads(s::ScanSpec) =
    _history_statement_reads(s.states, s.loopvar, _scan_fold_statements(s))

# Does any step read an earlier entry of a carried array?
_scan_has_history(s::ScanSpec) = _history_has_reads(_scan_statement_reads(s))

_scan_maxlag(step, states, loopvar) =
    maximum((maximum((k for k in l if k isa Int); init = 0) for l in values(
        _scan_lags(step, states, loopvar))); init = 0)

# Scalar continuous distributions keep their declared support and density.
# The layout owns the support transforms; recurrence samples remain ordered.
function _scan_shape_gap(s::ScanSpec)
    for st in (s.setup..., s.step...)
        st.kind === :sample || continue
        haskey(SAMPLED_SUPPORT, st.family) || return "scan $(join(s.states, ", ")): " *
            "sampled family :$(st.family) has no scalar continuous support"
    end
    return nothing
end

# Shared lowering-time check; data reads are ordinary captured values.
function _screen_scan(s::ScanSpec, data)
    gap = _scan_shape_gap(s)
    gap === nothing || _scan_fail(gap)
    return nothing
end

# Bound integer lags choose a field of the authored seed-depth window. The
# window's shape stays fixed; the data never expand a recurrence body.
function _resolve_scan(plan::StructuralPlan, s::ScanSpec)
    # Generator binders are never data lags: `a[t - s]` under `for s in …`
    # is a history read.
    function resolve(ex, bound)
        ex isa Expr || return ex
        if ex.head === :generator
            inner = union(bound, Set{Symbol}(v for (v, _) in
                first(_scan_generator_clauses(ex)) if v isa Symbol))
            return Expr(ex.head, map(a -> resolve(a, inner), ex.args)...)
        end
        if ex.head === :ref && length(ex.args) == 2 && ex.args[1] in s.states
            env = Dict{Symbol,Any}(b => nothing for b in bound)
            k = _history_lag_form(ex.args[2], s.loopvar, env)
            if k isa Symbol
                value = get(plan.columns, k, nothing)
                (value isa Integer && !(value isa Bool) && 1 <= value < s.lo) ||
                    _fail(s.label, "scan lag $k must bind an integer in 1:$(s.lo - 1), got $(repr(value))")
                return Expr(:ref, ex.args[1], Expr(:call, :-, s.loopvar, Int(value)))
            end
        end
        return Expr(ex.head, map(a -> resolve(a, bound), ex.args)...)
    end
    resolve(ex) = resolve(ex, Set{Symbol}())
    step = [ScanStep(st.kind, st.target, st.indexed, st.family,
        st.args === nothing ? nothing : map(resolve, st.args), resolve(st.expr),
        st.support) for st in s.step]
    return ScanSpec(s.states, s.loopvar, s.lo, s.hi, s.setup, step, s.maxlag, s.label)
end
