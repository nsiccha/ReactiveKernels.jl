# Observed carried arrays of a `@scan` (rkppl-use §5, §13).
#
# Data classifies statements: a carried array whose name is observed data
# is an observed carried array. Its seed and step `~` statements are
# observations: each one's density, at the observed entry, is the
# conditional density of that entry given the earlier ones, and its reads of
# the array read the data. A carried array that is not data keeps its latent
# trajectory.
#
# Lowering states this as an ordinary program. The scan computes each seed's
# and step's conditional density in a carried array of its own, reading the
# observed values, and one dotted `LogDensity` observation of the response
# reads those densities entry by entry:
#
#     @scan begin                          _rkppl_scan_observed_y = _observed_scan_values(:y, y, T)
#         y[1] ~ D1(…)                     @scan begin
#         for t in 2:T                         _rkppl_scan_logdensity_y[1] = <D1 density at y[1]>
#             y[t] ~ D(…, y[t - 1])   ⟶        for t in 2:T
#         end                                      _rkppl_scan_logdensity_y[t] = <D density at y[t]>
#     end                                      end
#                                          end
#                                          y .~ LogDensity.(_observed_scan_logdensity,
#                                              _rkppl_scan_logdensity_y)
#
# so the response keeps every observation contract (`likelihood`,
# `:pointwise` with one density per entry, absent responses, arrays per
# index in a dotted `@plate` cell). In a `@plate` cell the observed array is
# the loop index's array, `y[g][t]`, read in place. The unbound spelling
# `y[g][t] ~ …` in a cell is a latent trajectory per index, exactly as a
# cell-local carried array returned as `y[g]`.

# An observation's density, computed by the scan that reads the observed
# trajectory: the law is the density it is given.
@kernel _observed_scan_logdensity(value, density) = density

const _OBSERVED_SCAN_LAW = GlobalRef(@__MODULE__, :_observed_scan_logdensity)

# Bind-time check of a top-level observed carried array: a vector of real
# numbers with one entry per scan index (no `len` when the scan's length is
# the array's own).
_observed_scan_values(name::Symbol, value) =
    (_check_observed_scan_entries(name, value); value)
function _observed_scan_values(name::Symbol, value, len)
    _check_observed_scan_entries(name, value)
    # More entries would broadcast against fewer densities. The scan reads
    # past fewer entries and fails there, at evaluation, which is where a
    # one-entry array is checked: an absent array's bind-time stand-in has
    # one entry (`_absent_definition_value`).
    n = length(value)
    n == len || n == 1 || _observed_scan_length_error(name, n, len)
    return value
end

# Cell check of one index's observed array; returns the scan's upper bound.
function _observed_scan_cell_bound(name::Symbol, index, value, hi)
    _check_observed_scan_entries("$name[$index]", value)
    hi isa Integer || throw(ArgumentError(
        "the `@scan` upper bound for `$name[$index]` must be an integer; got $(repr(hi))"))
    length(value) == hi || _observed_scan_length_error("$name[$index]", length(value), hi)
    return hi
end
# An absent observed array (a prediction design) has no entries to check.
_observed_scan_cell_bound(::Symbol, index, ::_Absent, hi) = hi

_observed_scan_length_error(name, n, len) = throw(DimensionMismatch(
    "observed `@scan` array `$name` has $n entries, but the scan writes $len; " *
    "an observed carried array holds one entry per scan index"))

function _check_observed_scan_entries(name, value)
    value isa AbstractVector || throw(ArgumentError(
        "observed `@scan` array `$name` must be a vector, one entry per scan index; " *
        "got $(typeof(value))"))
    any(ismissing, value) && throw(ArgumentError(
        "observed `@scan` array `$name` holds missing entries; each entry is " *
        "conditioned on the earlier ones, so missing entries are not built yet"))
    all(x -> x isa Real, value) || throw(ArgumentError(
        "observed `@scan` array `$name` must hold real numbers; got eltype $(eltype(value))"))
    return nothing
end

# The observed density of a seed or step `~ Dist(args…)` at the observed
# value `x`: the stated density with its support normalizer, `-Inf` outside
# the stated support. A seed is one expression, so the normalizer's
# intermediates are written inline.
function _observed_scan_density(call, x, what, translate)
    family, args, support = _scan_parse_dist(call, what)
    args = Any[translate(a) for a in args]
    support = support isa Tuple ? (support[1], map(translate, support[2:end])...) : support
    pre = Expr[]
    density = _scan_density!(pre, (; family, args, support), identity, x,
        :_rkppl_scan_tail)
    for p in pre
        lhs = p.args[1]
        density = _replace_symbol(density, Meta.isexpr(lhs, :(::)) ? lhs.args[1] : lhs,
            p.args[2])
    end
    valid = _observed_scan_support(family, support, args, x)
    valid === nothing && return density
    return :(if $valid
        $density
    else
        $(-Inf)
    end)
end

_replace_symbol(ex, name::Symbol, value) = ex === name ? value :
    ex isa Expr ? Expr(ex.head, (_replace_symbol(a, name, value) for a in ex.args)...) : ex

# The condition that `x` lies in the stated support: the family's natural
# support intersected with a `truncated` / `restricted` / half override.
function _observed_scan_support(family, support, args, x)
    conds = Any[]
    natural = SAMPLED_SUPPORT[family]
    natural === :positive && push!(conds, :($x >= 0))
    natural === :unit && push!(conds, :(0 <= $x), :($x <= 1))
    natural === :interval && push!(conds, :($(args[1]) <= $x), :($x <= $(args[2])))
    if support === :positive
        push!(conds, :($x >= 0))
    elseif support isa Tuple
        head = support[1]
        lo, hi = head === :upper ? (-Inf, support[2]) :
            head === :lower ? (support[2], Inf) : (support[2], support[3])
        lo isa Real && lo == -Inf || push!(conds, :($lo <= $x))
        hi isa Real && hi == Inf || push!(conds, :($x <= $hi))
    end
    isempty(conds) && return nothing
    return foldl((a, b) -> :($a && $b), conds)
end

# A scan block's seed statements: everything before its loop.
_observed_scan_setups(block) = [s for s in block.args
    if !(s isa LineNumberNode) && !Meta.isexpr(s, :for)]

_observed_scan_lhs(s) = Meta.isexpr(s, :call, 3) && s.args[1] === :~ ? s.args[2] :
    Meta.isexpr(s, :(=), 2) ? s.args[1] : nothing

_observed_scan_sample(s) = Meta.isexpr(s, :call, 3) && s.args[1] === :~

# Every statement writing an observed carried array is a `~` statement: an
# assignment would replace an observed value.
function _check_observed_scan_writes(spec::ScanSpec, observed, shown)
    for st in (spec.setup..., spec.step...)
        target = st isa ScanSetup ? st.target : (st.indexed ? st.target : nothing)
        target in observed && st.kind === :assign && _scan_fail(
            "`$(shown(target))` is observed data; its seeds and steps are `~` " *
            "statements, and an assignment `$(shown(target))[…] = …` would replace " *
            "an observed value")
    end
    return nothing
end

# Rewrite one scan block's statements: a `~` writing an observed array at
# index `idx` becomes the density of that entry in the array's density
# trajectory; every other statement is kept, with `translate` applied.
function _observed_scan_rewrite(block, observed, lps, value, translate, hi)
    rewrite(s) = begin
        lhs = _observed_scan_lhs(s)
        if _observed_scan_sample(s) && Meta.isexpr(lhs, :ref, 2)
            a = _observed_scan_target(lhs)
            if a !== nothing && a in observed
                idx = lhs.args[2]
                x = value(a, idx)
                return Expr(:(=), Expr(:ref, lps[a], idx),
                    _observed_scan_density(s.args[3], x,
                        "observation `$(repr(lhs))`", translate))
            end
        end
        return translate(s)
    end
    out = Any[]
    for s in block.args
        if Meta.isexpr(s, :for)
            head = s.args[1]
            range = head.args[2]
            newrange = hi === nothing ? range :
                Expr(:call, range.args[1], range.args[2], hi)
            body = Any[x isa LineNumberNode ? x : rewrite(x) for x in s.args[2].args]
            push!(out, Expr(:for, Expr(:(=), head.args[1], newrange), Expr(:block, body...)))
        else
            push!(out, s isa LineNumberNode ? s : rewrite(s))
        end
    end
    return Expr(:block, out...)
end

# The observed array a write `a[idx]` targets: its name at top level. A cell
# rewrite has already respelled `y[g][idx]` as the placeholder name.
_observed_scan_target(lhs) = lhs.args[1] isa Symbol ? lhs.args[1] : nothing

# Replace reads `a[idx]` of observed arrays by `source(a)[idx]`.
_observed_scan_reads(ex, sources) = ex isa Expr ?
    (Meta.isexpr(ex, :ref) && length(ex.args) >= 2 && ex.args[1] isa Symbol &&
        haskey(sources, ex.args[1]) ?
        Expr(:ref, sources[ex.args[1]], (_observed_scan_reads(a, sources) for a in ex.args[2:end])...) :
        Expr(ex.head, (_observed_scan_reads(a, sources) for a in ex.args)...)) : ex

_observed_scan_statement(lhs, lp) = Expr(:call, :.~, lhs,
    Expr(:., GlobalRef(@__MODULE__, :LogDensity), Expr(:tuple, _OBSERVED_SCAN_LAW, lp)))

# A top-level `@scan` whose carried arrays include observed data: the checked
# observed values, the scan stating their densities, and one observation per
# observed array. `nothing` when no carried array is data.
function _observed_scan_statements(arg::Expr, data, defined, taken)
    block = arg.args[end]
    spec = parse_scan_block(block)
    observed = [a for a in spec.states if a in data]
    isempty(observed) && return nothing
    _check_observed_scan_writes(spec, observed, identity)
    fresh(base) = _cell_scan_fresh!(taken, base)
    values = Dict(a => fresh(Symbol(:_rkppl_scan_observed_, a)) for a in observed)
    lps = Dict(a => fresh(Symbol(:_rkppl_scan_logdensity_, a)) for a in observed)
    out = Any[]
    # The historical unsupplied length name (`for t in 2:T`) takes the
    # consuming response's rows: here, the observed array's own length.
    hi, len = spec.hi, spec.hi
    if hi isa Symbol && !(hi in data) && !(hi in defined)
        hi = fresh(Symbol(:_rkppl_scan_length_, first(observed)))
        len = nothing
    end
    check = GlobalRef(@__MODULE__, :_observed_scan_values)
    for (k, a) in enumerate(observed)
        bound = k == 1 ? len : hi
        push!(out, Expr(:(=), values[a], Expr(:call, check, QuoteNode(a), a,
            (bound === nothing ? () : (bound,))...)))
        k == 1 && len === nothing &&
            push!(out, Expr(:(=), hi, Expr(:call, GlobalRef(Base, :length), values[a])))
    end
    translate(ex) = _observed_scan_reads(ex, values)
    rewritten = _observed_scan_rewrite(block, Set(observed), lps,
        (a, idx) -> Expr(:ref, values[a], idx), translate, hi === spec.hi ? nothing : hi)
    push!(out, Expr(:macrocall, arg.args[1:end-1]..., rewritten))
    append!(out, (_observed_scan_statement(a, lps[a]) for a in observed))
    return out
end

# A `@scan` block in a `@plate for ivar in …` cell whose carried arrays are
# written as the loop index's array, `y[ivar][k]`. Observed data (`y` bound)
# becomes the cell's observation of `y[ivar]`; otherwise `y[ivar]` is the
# cell's per-index output of a latent trajectory.
function _observed_cell_scan(c, ivar, data, taken)
    block = c.args[end]
    names = Symbol[]
    for s in _observed_scan_setups(block)
        lhs = _observed_scan_lhs(s)
        Meta.isexpr(lhs, :ref, 2) && Meta.isexpr(lhs.args[1], :ref, 2) &&
            lhs.args[1].args[1] isa Symbol && lhs.args[1].args[2] === ivar &&
            !(lhs.args[1].args[1] in names) && push!(names, lhs.args[1].args[1])
    end
    isempty(names) && return nothing
    fresh(base) = _cell_scan_fresh!(taken, base)
    placeholders = Dict(y => fresh(Symbol(:_rkppl_scan_, y)) for y in names)
    respell(ex) = ex isa Expr ?
        (Meta.isexpr(ex, :ref) && length(ex.args) >= 2 && Meta.isexpr(ex.args[1], :ref, 2) &&
            ex.args[1].args[1] in names && ex.args[1].args[2] === ivar ?
            Expr(:ref, placeholders[ex.args[1].args[1]], (respell(a) for a in ex.args[2:end])...) :
            Expr(ex.head, (respell(a) for a in ex.args)...)) : ex
    spelled = respell(block)
    observed = [y for y in names if y in data]
    latent = [y for y in names if !(y in data)]
    # Latent per-index trajectories: cell-local carried arrays returned as
    # `y[ivar]`.
    if isempty(observed)
        return Any[Expr(:macrocall, c.args[1:end-1]..., spelled),
            (Expr(:(=), Expr(:ref, y, ivar), placeholders[y]) for y in latent)...]
    end
    # Validate the scan as written, with the loop index's arrays as carried
    # arrays (the cell bound may be any data expression).
    parsed = deepcopy(spelled)
    forx = last(filter(x -> !(x isa LineNumberNode), parsed.args))
    range = forx.args[1].args[2]
    Meta.isexpr(range, :call, 3) && range.args[1] === :(:) ||
        _scan_fail("a cell scan takes the range lo:hi")
    hi = range.args[3]
    range.args[3] = fresh(:_ppl_cell_scan_bound)
    spec = parse_scan_block(parsed)
    obs = Set(placeholders[y] for y in observed)
    shown = Dict(placeholders[y] => Expr(:ref, y, ivar) for y in names)
    _check_observed_scan_writes(spec, obs, p -> shown[p])
    out = Any[]
    # The scan's bound, checked against each observed array's length.
    bound = hi
    for y in observed
        checked = fresh(Symbol(:_rkppl_scan_bound_, y))
        push!(out, Expr(:(=), checked, Expr(:call,
            GlobalRef(@__MODULE__, :_observed_scan_cell_bound), QuoteNode(y), ivar,
            Expr(:ref, y, ivar), bound)))
        bound = checked
    end
    lps = Dict(placeholders[y] => fresh(Symbol(:_rkppl_scan_logdensity_, y)) for y in observed)
    sources = Dict(placeholders[y] => Expr(:ref, y, ivar) for y in observed)
    translate(ex) = _observed_scan_reads(ex, sources)
    rewritten = _observed_scan_rewrite(spelled, obs, lps,
        (a, idx) -> Expr(:ref, sources[a], idx), translate, bound)
    push!(out, Expr(:macrocall, c.args[1:end-1]..., rewritten))
    for y in latent
        push!(out, Expr(:(=), Expr(:ref, y, ivar), placeholders[y]))
    end
    for y in observed
        lp = lps[placeholders[y]]
        output = fresh(Symbol(:_rkppl_scan_density_, y))
        push!(out, Expr(:(=), Expr(:ref, output, ivar), lp))
        push!(out, _observed_scan_statement(Expr(:ref, y, ivar), Expr(:ref, output, ivar)))
    end
    return out
end

# Observed (and per-index) carried arrays of every `@scan` in the program:
# top-level blocks and blocks in `@plate` cells.
function _expand_observed_scans(args, data::Set{Symbol})
    taken = union(data, _expr_names(Expr(:block, args...)))
    defined = Set{Symbol}(a.args[1] for a in args
        if Meta.isexpr(a, :(=), 2) && a.args[1] isa Symbol)
    out = Any[]
    for arg in args
        if Meta.isexpr(arg, :macrocall) && arg.args[1] === Symbol("@scan") &&
                length(arg.args) >= 3 && arg.args[end] isa Expr
            rewritten = _observed_scan_statements(arg, data, defined, taken)
            rewritten === nothing || (append!(out, rewritten); continue)
        elseif Meta.isexpr(arg, :macrocall) && arg.args[1] === Symbol("@plate") &&
                length(arg.args) == 3 && Meta.isexpr(arg.args[3], :for) &&
                Meta.isexpr(arg.args[3].args[1], :(=), 2) &&
                arg.args[3].args[1].args[1] isa Symbol &&
                Meta.isexpr(arg.args[3].args[2], :block)
            loop = arg.args[3]
            ivar = loop.args[1].args[1]
            cells = Any[]
            changed = false
            for c in loop.args[2].args
                expanded = _is_cell_scan_block(c) && c.args[end] isa Expr ?
                    _observed_cell_scan(c, ivar, data, taken) : nothing
                expanded === nothing ? push!(cells, c) :
                    (append!(cells, expanded); changed = true)
            end
            changed && (push!(out, Expr(:macrocall, arg.args[1:2]...,
                Expr(:for, loop.args[1], Expr(:block, cells...)))); continue)
        end
        push!(out, arg)
    end
    return out
end
