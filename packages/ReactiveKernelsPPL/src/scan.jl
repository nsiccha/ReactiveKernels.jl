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
# Grammar — one or more carried arrays (a tuple carry), literal backward lags:
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
# Statements run in order, as in a Julia loop body: a step reads a carried
# array's backward lag `a[t-k]` (k ≥ 1 literal, k ≤ m), its current value `a[t]`
# once an earlier step of the same iteration wrote it, and locals defined by
# earlier steps. Each carried array is written exactly once per step. The
# observation lives OUTSIDE the block (`y .~ Normal.(h, 1)`), so it is not
# part of `parse_scan_block`.

# The `ScanSpec` / `ScanStep` / `ScanSetup` IR structs live in `contract.jl`
# (alongside the other IR specs, so `StructuralPlan` can reference them); this
# file owns only the surface PARSER that produces a `ScanSpec`.

_scan_fail(msg) = _sfail("@scan: " * msg)

# `Dist(args…)` → (internal family symbol, positional argument expressions).
# Reuses the surface's Distributions.jl family table and positional-arg peeler.
function _scan_parse_dist(call, what)
    (call isa Expr && call.head === :call && !isempty(call.args)) ||
        _scan_fail("$what needs a distribution call, got $(repr(call))")
    fam = call.args[1]
    fam in (:normal, :cauchy, :exponential, :gamma, :lognormal, :beta,
        :inverse_gamma) && _scan_fail(
        "$what: use Distributions.jl constructors (`Normal`, not `$(fam)`)")
    haskey(_PARAM_FAMILIES, fam) || _scan_fail(
        "$what: unknown distribution `$(repr(fam))` (admitted: Normal, " *
        "Cauchy, Exponential, Gamma, LogNormal, Beta, InverseGamma)")
    return _PARAM_FAMILIES[fam], collect(Any, _plain_args(call, "`$(fam)`"))
end

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
        push!(step, _parse_scan_step(s, states, loopvar))
    end
    isempty(step) && _scan_fail("empty recurrence body")
    any(st -> st.indexed, step) || _scan_fail(
        "the loop never writes a carried array (`$(first(states))[$(loopvar)]`; " *
        "each step is a fresh local); a scan must thread the state")
    _scan_check_order(step, states, loopvar)

    maxlag = _scan_maxlag(step, states, loopvar)
    maxlag >= 1 || _scan_fail(
        "no backward lag read of a carried array in the body — independent " *
        "cells are `@plate`, not `@scan`")
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
            fam, args = _scan_parse_dist(s.args[3], "setup `$(nm)[$(idx)]`")
            for a in args
                _scan_check_seed_reads(a, carried, next, "setup `$(nm)[$(idx)]`")
            end
            push!(setup, ScanSetup(nm, idx, :sample, fam, args, nothing))
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

function _parse_scan_step(s, states, loopvar)
    if s isa Expr && s.head === :call && length(s.args) == 3 && s.args[1] === :~
        lhs = s.args[2]
        target, indexed = _scan_step_lhs(lhs, states, loopvar, "`~`")
        fam, args = _scan_parse_dist(s.args[3], "step `$(_scan_lhs_show(lhs))`")
        for a in args
            _scan_check_reads(a, states, loopvar)
        end
        return ScanStep(:sample, target, indexed, fam, args, nothing)
    elseif s isa Expr && s.head === :(=) && length(s.args) == 2
        lhs = s.args[1]
        target, indexed = _scan_step_lhs(lhs, states, loopvar, "`=`")
        _scan_check_reads(s.args[2], states, loopvar)
        return ScanStep(:assign, target, indexed, nothing, nothing, s.args[2])
    elseif s isa Expr && s.head in (:for, :while)
        _scan_fail("nested loops in a `@scan` body are not supported yet")
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

# Walk an expression rejecting illegal reads of a carried array: a bare read
# of the whole array or a forward lag. The admitted reads are a backward lag
# `state[loopvar - k]` (k ≥ 1 literal) and the current value `state[loopvar]`
# (legal once an earlier step wrote it — `_scan_check_order`).
function _scan_check_reads(ex, states, loopvar)
    if ex isa Symbol
        ex in states && _scan_fail(
            "bare read of the carried array `$(ex)` inside its own loop; read " *
            "a backward lag `$(ex)[$(loopvar)-1]`")
        return nothing
    end
    ex isa Expr || return nothing
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] in states
        ex.args[2] === loopvar && return nothing
        _scan_lag_of(ex.args[2], ex.args[1], loopvar)   # validates
        return nothing
    end
    for a in ex.args
        _scan_check_reads(a, states, loopvar)
    end
    return nothing
end

# Standard-Julia order inside one iteration: a local is read only after the
# step that defines it, a carried array's current value `a[t]` only after the
# step that writes it, each carried array is written exactly once, and a local
# is defined once and never shadows a carried array.
function _scan_check_order(step, states, loopvar)
    locals = Set{Symbol}(st.target for st in step if !st.indexed)
    defined = Set{Symbol}()
    written = Set{Symbol}()
    for st in step
        reads = st.kind === :sample ? st.args : Any[st.expr]
        for r in reads
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

# Parse the index of a `state[idx]` read; return the backward lag k (≥1) or fail.
function _scan_lag_of(idx, state, loopvar)
    idx === loopvar && _scan_fail(
        "`$(state)[$(loopvar)]` reads the value being written this step; read a " *
        "backward lag `$(state)[$(loopvar)-1]`")
    if idx isa Expr && idx.head === :call && length(idx.args) == 3 &&
       idx.args[1] === :- && idx.args[2] === loopvar
        k = idx.args[3]
        k isa Int && k >= 1 && return k
        _scan_fail("lag depth in `$(state)[$(loopvar)-…]` must be a literal " *
                   "integer ≥ 1, got $(repr(k))")
    end
    if idx isa Expr && idx.head === :call && length(idx.args) == 3 &&
       idx.args[1] === :+ && idx.args[2] === loopvar
        _scan_fail("forward reference `$(state)[$(loopvar)+…]` — a scan only " *
                   "reads backward lags")
    end
    _scan_fail("`$(state)` index must be the backward lag `$(loopvar)-<k>`, " *
               "got $(repr(idx))")
end

# Distinct backward lags `a[loopvar - k]` read across a scan's steps, per
# carried array (current-value reads `a[loopvar]` are not lags).
function _scan_lags(step, states, loopvar)
    lags = Dict{Symbol,Set{Int}}(a => Set{Int}() for a in states)
    walk(ex) = begin
        if ex isa Expr
            if ex.head === :ref && length(ex.args) == 2 && ex.args[1] in states
                ex.args[2] === loopvar ||
                    push!(lags[ex.args[1]], _scan_lag_of(ex.args[2],
                        ex.args[1], loopvar))
            else
                for a in ex.args
                    walk(a)
                end
            end
        end
    end
    for st in step
        if st.kind === :sample
            for a in st.args
                walk(a)
            end
        else
            walk(st.expr)
        end
    end
    return lags
end

_scan_maxlag(step, states, loopvar) =
    maximum((isempty(l) ? 0 : maximum(l) for l in values(
        _scan_lags(step, states, loopvar))); init = 0)

# Shapes the parser admits but the emitter does not build yet, as one rule
# shared by lowering (`SurfaceLoweringError`, so a program fails where it is
# written) and the generator (`ContractValidationError`, for hand-built
# plans). Returns the first gap's message, or `nothing`:
# - a centered scan (`state[t] ~ dist`) is one carried array, sampled seeds
#   and exactly one step;
# - a non-centered scan writes every carried array deterministically, has at
#   least one per-step innovation, and its latents (sampled seeds,
#   innovations) have real support;
# - no step reads the loop index directly.
const _SCAN_LATENT_FAMILIES = (:normal, :cauchy, :student_t, :laplace, :logistic)

function _scan_shape_gap(s::ScanSpec)
    who = "scan $(join(s.states, ", "))"
    for st in s.step
        reads = st.kind === :sample ? st.args : Any[st.expr]
        any(r -> _scan_mentions(r, s.loopvar, s.states), reads) &&
            return "$who: a " *
            "step uses the loop index `$(s.loopvar)` directly — not supported " *
            "yet (read carried arrays, locals and scalars)"
    end
    carried = [st for st in s.step if st.indexed]
    if !_is_noncentered_scan(s)
        length(s.states) == 1 || return "$who: a centered scan " *
            "(`state[t] ~ dist`) carries one array; a tuple carry writes each " *
            "array deterministically (`state[t] = …`) from sampled innovations"
        all(f -> f.kind === :sample, s.setup) || return "$who: a centered " *
            "scan samples its seeds (`$(only(s.states))[1] ~ dist`); a " *
            "deterministic seed needs the non-centered form " *
            "(`eps ~ Normal(0, 1)`, `$(only(s.states))[t] = …`)"
        length(s.step) == 1 || return "$who: a centered recurrence is " *
            "exactly one step (`$(only(s.states))[$(s.loopvar)] ~ dist`); " *
            "per-step locals need the non-centered form (innovation samples " *
            "+ deterministic carry writes)"
        return nothing
    end
    for st in carried
        st.kind === :sample && return "$who: `$(st.target)[$(s.loopvar)] ~ …` " *
            "samples a carried array while another carried write is " *
            "deterministic — mixing centered and non-centered carried writes " *
            "in one scan is not supported yet"
    end
    real = "real support (Normal, Cauchy, StudentT, Laplace or Logistic)"
    for f in s.setup
        (f.kind === :sample && !(f.family in _SCAN_LATENT_FAMILIES)) &&
            return "$who: the sampled seed `$(f.target)[$(f.index)]` of a " *
            "non-centered scan must have $real; got :$(f.family)"
    end
    innov = [st for st in s.step if st.kind === :sample && !st.indexed]
    for st in innov
        st.family in _SCAN_LATENT_FAMILIES || return "$who: the innovation " *
            "`$(st.target)` must have $real; got :$(st.family)"
    end
    isempty(innov) && return "$who: a non-centered scan needs a per-step " *
        "innovation (`eps ~ Normal(0, 1)`); a fully deterministic recurrence " *
        "is not supported yet"
    return nothing
end

# Whether `ex` reads the loop index `nm` outside a carried array's index
# (`a[t - 1]`, `a[t]` are carried reads, not loop-index reads).
_scan_mentions(ex, nm::Symbol, states) = ex === nm ||
    (ex isa Expr &&
     !(ex.head === :ref && !isempty(ex.args) && ex.args[1] in states) &&
     any(a -> _scan_mentions(a, nm, states),
        ex.head === :call ? ex.args[2:end] : ex.args))

# Lowering-time screen of a parsed scan: an emitter gap (`_scan_shape_gap`)
# or a data read inside the recurrence fails where the program is written.
# Seeds and steps read scalars (parameters, definitions, literals); a data
# column read per step (`y[t]`) is not supported yet.
function _screen_scan(s::ScanSpec, data)
    exprs = Any[]
    for f in s.setup
        f.kind === :sample ? append!(exprs, f.args) : push!(exprs, f.expr)
    end
    for st in s.step
        st.kind === :sample ? append!(exprs, st.args) : push!(exprs, st.expr)
    end
    for ex in exprs
        d = _scan_data_read(ex, data)
        d === nothing || _scan_fail("scan $(join(s.states, ", ")) reads the " *
            "data column `$(d)` — data-varying seeds and steps are not " *
            "supported yet (a scan reads carried arrays, locals and scalars)")
    end
    gap = _scan_shape_gap(s)
    gap === nothing || _scan_fail(gap)
    return nothing
end

function _scan_data_read(ex, data)
    ex isa Symbol && return ex in data ? ex : nothing
    ex isa Expr || return nothing
    for a in (ex.head === :call ? ex.args[2:end] : ex.args)
        d = _scan_data_read(a, data)
        d === nothing || return d
    end
    return nothing
end
