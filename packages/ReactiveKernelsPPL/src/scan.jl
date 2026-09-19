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
# Grammar (v1 — one carried array, literal backward lags):
#   @scan begin
#       h[1] ~ Normal(0, 1)               # setup: literal-index seed fills 1..m
#       for t in (m+1):T                  # the recurrence; T literal Int or a data length name
#           h[t] ~ Normal(phi*h[t-1], s)  # centered: sample the carried array at the loop index
#           # or, non-centered:
#           #   eps ~ Normal(0, 1)        # per-step local innovation (fresh bare name)
#           #   h[t] = phi*h[t-1] + s*eps # deterministic carry write
#       end
#   end
# Carried-array reads in a step RHS must be a backward lag `h[t-k]` (k≥1 literal);
# the setup depth m must be ≥ the maximum lag. The observation lives OUTSIDE the
# block (`y .~ Normal.(h, 1)`), so it is not part of `parse_scan_block`.

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
        "need at least one seed fill (`state[1] ~ Dist(…)`) before the loop")

    state, setup = _parse_scan_setup(setups_ast)
    m = length(setup)
    loopvar, lo, hi = _parse_scan_for_head(forx, m)

    body = forx.args[2]
    (body isa Expr && body.head === :block) || _scan_fail("malformed loop body")
    step = ScanStep[]
    for s in body.args
        s isa LineNumberNode && continue
        push!(step, _parse_scan_step(s, state, loopvar))
    end
    isempty(step) && _scan_fail("empty recurrence body")
    any(st -> st.indexed, step) || _scan_fail(
        "the loop never writes the carried array `$(state)[$(loopvar)]` " *
        "(each step is a fresh local); a scan must thread the state")

    maxlag = _scan_maxlag(step, state, loopvar)
    maxlag >= 1 || _scan_fail(
        "no backward lag read of `$(state)` in the body — independent cells " *
        "are `@plate`, not `@scan`")
    m >= maxlag || _scan_fail(
        "the recurrence reads a lag of $(maxlag) but only $(m) initial " *
        "value(s) are seeded; add `$(state)[1..$(maxlag)]` fills")
    return ScanSpec(state, loopvar, lo, hi, setup, step, maxlag,
        something(label, state))
end

function _parse_scan_setup(setups_ast)
    state = nothing
    setup = ScanSetup[]
    for (k, s) in enumerate(setups_ast)
        (s isa Expr && s.head === :call && length(s.args) == 3 &&
         s.args[1] === :~) || _scan_fail(
            "setup statement $(k) must be `state[$(k)] ~ Dist(…)`, " *
            "got $(repr(s))")
        lhs = s.args[2]
        (lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2) ||
            _scan_fail("setup LHS must be `state[<integer>]`, got $(repr(lhs))")
        nm, idx = lhs.args[1], lhs.args[2]
        nm isa Symbol || _scan_fail("setup array name must be a Symbol")
        idx isa Int || _scan_fail(
            "setup index must be a literal integer, got $(repr(idx)) " *
            "(a slice `state[1:p]` is not supported yet — write the lines out)")
        if state === nothing
            state = nm
        elseif nm !== state
            _scan_fail("all setup fills must seed the same carried array " *
                       "`$(state)` (got `$(nm)`)")
        end
        idx == k || _scan_fail(
            "setup fills must be contiguous `state[1], state[2], …`; " *
            "expected index $(k), got $(idx)")
        fam, args = _scan_parse_dist(s.args[3], "setup `$(nm)[$(idx)]`")
        push!(setup, ScanSetup(idx, fam, args))
    end
    return state, setup
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

function _parse_scan_step(s, state, loopvar)
    if s isa Expr && s.head === :call && length(s.args) == 3 && s.args[1] === :~
        lhs = s.args[2]
        target, indexed = _scan_step_lhs(lhs, state, loopvar, "`~`")
        fam, args = _scan_parse_dist(s.args[3], "step `$(_scan_lhs_show(lhs))`")
        for a in args
            _scan_check_reads(a, state, loopvar)
        end
        return ScanStep(:sample, target, indexed, fam, args, nothing)
    elseif s isa Expr && s.head === :(=) && length(s.args) == 2
        lhs = s.args[1]
        target, indexed = _scan_step_lhs(lhs, state, loopvar, "`=`")
        _scan_check_reads(s.args[2], state, loopvar)
        return ScanStep(:assign, target, indexed, nothing, nothing, s.args[2])
    elseif s isa Expr && s.head in (:for, :while)
        _scan_fail("nested loops in a `@scan` body are not supported yet")
    elseif s isa Expr && s.head in (:if, :elseif)
        _scan_fail("branches in a `@scan` body are not supported")
    else
        _scan_fail("a `@scan` step must be `state[$(loopvar)] ~ Dist(…)`, a " *
                   "fresh `local ~ Dist(…)`, or an assignment; got $(repr(s))")
    end
end

# Classify a step LHS. Returns (target, indexed). `indexed` means the carried
# array is written at the current loop index `state[loopvar]`.
function _scan_step_lhs(lhs, state, loopvar, what)
    if lhs isa Symbol
        lhs === state && _scan_fail(
            "cannot rebind the whole carried array `$(state)` in the loop; " *
            "write `$(state)[$(loopvar)]`")
        return lhs, false                      # a fresh per-step local
    elseif lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2
        nm, idx = lhs.args[1], lhs.args[2]
        nm === state || _scan_fail(
            "$(what) writes `$(nm)[…]`, but v1 threads exactly one carried " *
            "array (`$(state)`); other indexed writes are not supported yet")
        idx === loopvar || _scan_fail(
            "the carried write must be at the loop index `$(state)[$(loopvar)]`, " *
            "got `$(_scan_lhs_show(lhs))` (a scan writes each index once, in order)")
        return state, true
    else
        _scan_fail("$(what) left-hand side must be `$(state)[$(loopvar)]` or a " *
                   "fresh local name, got $(repr(lhs))")
    end
end

_scan_lhs_show(lhs) = string(lhs)

# Walk an expression rejecting illegal reads of the carried array: a bare read
# of the whole array, a current-index read `state[loopvar]`, or a forward lag.
# The only admitted read is a backward lag `state[loopvar - k]` (k≥1 literal).
function _scan_check_reads(ex, state, loopvar)
    if ex isa Symbol
        ex === state && _scan_fail(
            "bare read of the carried array `$(state)` inside its own loop; " *
            "read a backward lag `$(state)[$(loopvar)-1]`")
        return nothing
    end
    ex isa Expr || return nothing
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] === state
        _scan_lag_of(ex.args[2], state, loopvar)   # validates; result used by _scan_maxlag
        return nothing
    end
    for a in ex.args
        _scan_check_reads(a, state, loopvar)
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

function _scan_maxlag(step, state, loopvar)
    maxlag = 0
    walk(ex) = begin
        if ex isa Expr
            if ex.head === :ref && length(ex.args) == 2 && ex.args[1] === state
                maxlag = max(maxlag, _scan_lag_of(ex.args[2], state, loopvar))
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
    return maxlag
end
