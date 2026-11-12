# Shared factors and terms of a summed plate, taken out of its sum (snag
# `normal-plate-kee-8322185b`).
#
# A summed plate adds one value per cell. When that value is `a + c * t`, with
# `a` and `c` reading only values every cell shares and `t` the part that
# reads the cell, the sum is `n * a + c * sum(t)` over the plate's `n` cells:
# the cells compute only `t`, and the shared work runs once per call. A Normal
# observation plate with one scale is the case that matters most:
#
#     z = (x - location) / scale
#     density = -0.5 * log(2pi) - 0.5 * z^2 - log_scale
#
# is `a = -0.5 * log(2pi) - log_scale`, `c = -0.5 * inv(scale)^2` and
# `t = (x - location)^2`. Each cell then neither divides by the scale nor adds
# the hoisted log scale, and the ordinary reverse no longer accumulates the
# scale's adjoint cell by cell: that serial accumulation is what kept the
# reverse loop scalar, and what derivative arithmetic without `reassoc` cannot
# vectorize at all.
#
# A law that guards a degenerate parameter in every cell (`scale == 0 && x ==
# location ? 1.0 : scale`) reads a shared condition first. Such a condition is
# decided once per call: the factored loop assumes the value under which the
# guard no longer reads the cell, and the plate's own loop runs, unchanged,
# whenever it does not hold.
#
# Admission. The factored total is the same real number as the plate's sum.
# Wherever the plate's value is finite, the factored value and its ordinary
# reverse derivatives agree with it to rounding:
# - a shared factor that reads a value (`-0.5 * inv(scale)^2`) is computed once
#   and used only while it is a normal number, so a scale below about 1e-154
#   or above about 1e154 runs the plate's own loop;
# - such a factor multiplies a part the cell did not compute
#   (`(x - location)^2` where the cell squared `z`), which can overflow where
#   the cell does not (a residual above about 1e154), so the partial sum and
#   then the factored total are kept only when finite, and the plate's own
#   loop runs otherwise. A discarded factored loop is still differentiated,
#   with a zero adjoint: each check comes before anything multiplies the value
#   it checks, so that zero never meets an infinite operand and the shared
#   values' derivatives stay those of the plate's own loop;
# - a literal factor (`-1`, `-0.5`) multiplies the cell's own subexpressions,
#   which overflow exactly where the cell does, and is decided before the
#   loop alone.
# Where the plate's value is not finite, derivatives can differ from the
# plate's own: at a scale so small that a Cauchy cell's `z^2` overflows, the
# density is -Inf in both and the scale's derivative NaN here, -n/scale (of
# the wrong sign) in the plate's own loop.
#
# Shared values are evaluated only where every cell would evaluate them (the
# plate is not empty, and they are not inside a branch a cell decides), so no
# exception moves; a cell's read of its own argument at its index is pure and
# in bounds, and is read once at the start of the cell. The pass runs with the
# stable rewrites (`simplify`, on by default; `simplify = false` keeps the
# literal loop), and not under the opt-in generated plate reverse
# (`plate_reverse.jl`), which differentiates the plate's own loop.
#
# How. This is a pass over the lowered body, before the index runs
# (`index_runs.jl`), which then group the factored loop's cells as they would
# the plate's own. It recognises the `@simd` loop of a total-only plate
# (`_index_run_simd_plate`) whose cell ends in its one running-total update,
# and reads the cell's statements as terms `a + c * t` through `+`, `-`, `*`,
# division by a shared value, integer powers and `abs`; any other call that
# reads the cell is opaque and stays as written. It replaces the loop with
#
#     fast = n > 0 && <shared conditions> && total isa Number
#     fast && (factor = c; fast = <factor is a normal number>)   # c reads a value
#     if fast
#         partial = zero(total)
#         <the same loop, each cell adding t to partial>
#         fast = isfinite(partial)                                # c reads a value
#         factored = n * a + c * partial
#         fast = isfinite(factored)                               # c reads a value
#         fast && (total += factored)
#     end
#     if !fast
#         <the plate's own loop>
#     end
#
# A loop it cannot read, or one with nothing to take out (no shared condition,
# and `a` and `c` read no value), is left as it is. The plate's own loop is
# emitted once more beside the factored one, whatever the data's size.

_spf_mark() = "#factored"

# `a + c * t`: `a` and `c` shared expressions (`nothing` for 0 and 1), `t` the
# per-cell part (`nothing` when the value is shared). `orig` computes the value
# as written, for a context that is not algebra (an opaque call's operand).
struct _SPFTerm
    a::Any
    c::Any
    t::Any
    orig::Any
end
_spf_shared(x) = _SPFTerm(x, nothing, nothing, x)
_spf_cell(x) = _SPFTerm(nothing, nothing, x, x)
_spf_isshared(x::_SPFTerm) = x.t === nothing

struct _SPFAbort <: Exception end

mutable struct _SPFCtx
    index::Symbol
    percell::Set{Symbol}           # names the cell binds
    atoms::Vector{Pair{Any,Bool}}  # shared condition => the value assumed
    reads::Vector{Pair{Any,Symbol}}  # a cell's read call => its local
    prefix::Symbol
    ops::Any                       # the lowering's callables table
end

_spf_ref(name::Symbol) = GlobalRef(Base, name)
_spf_call(f::Symbol, args...) = Expr(:call, _spf_ref(f), args...)

# Shared expressions: `nothing` is 0 for `a`, 1 for `c`.
_spf_neg(x) = _spf_call(:-, x)
_spf_negated(x) = Meta.isexpr(x, :call, 2) && x.args[1] == _spf_ref(:-)
_spf_add(x, y) = x === nothing ? y : y === nothing ? x :
    _spf_negated(y) ? _spf_call(:-, x, y.args[2]) : _spf_call(:+, x, y)
_spf_mul(x, y) = x === nothing ? y : y === nothing ? x : _spf_call(:*, x, y)
# `c * t` inside a cell.
_spf_scale(c, t) = c === nothing ? t : (c == -1 ? _spf_call(:-, t) : _spf_call(:*, c, t))

# Whether a shared expression reads a value (a name), rather than being a
# literal arithmetic of constants such as `-0.5 * log(2pi)`.
_spf_reads_value(x) = x isa Symbol ||
    (x isa Expr && x.head !== :quote && any(a -> _spf_reads_value(a), x.head === :call ? x.args[2:end] : x.args))

# --- terms --------------------------------------------------------------------

function _spf_neg(x::_SPFTerm, orig)
    _spf_isshared(x) && return _spf_shared(orig)
    _SPFTerm(x.a === nothing ? nothing : _spf_neg(x.a),
             x.c === nothing ? -1 : (x.c == -1 ? nothing : _spf_neg(x.c)), x.t, orig)
end
function _spf_add(x::_SPFTerm, y::_SPFTerm, orig)
    _spf_isshared(x) && _spf_isshared(y) && return _spf_shared(orig)
    _spf_isshared(x) && return _SPFTerm(_spf_add(x.a, y.a), y.c, y.t, orig)
    _spf_isshared(y) && return _SPFTerm(_spf_add(x.a, y.a), x.c, x.t, orig)
    x.c == y.c && return _SPFTerm(_spf_add(x.a, y.a), x.c, _spf_call(:+, x.t, y.t), orig)
    x.c === nothing && y.c == -1 &&
        return _SPFTerm(_spf_add(x.a, y.a), nothing, _spf_call(:-, x.t, y.t), orig)
    _SPFTerm(_spf_add(x.a, y.a), nothing,
             _spf_call(:+, _spf_scale(x.c, x.t), _spf_scale(y.c, y.t)), orig)
end
# A shared value `s` times `x`.
_spf_times(s, x::_SPFTerm, orig) = _spf_isshared(x) ? _spf_shared(orig) :
    _SPFTerm(x.a === nothing ? nothing : _spf_mul(s, x.a), _spf_mul(s, x.c), x.t, orig)

# The term of expression `e`, its names read through `env`.
function _spf_term(e, env::Dict{Symbol,_SPFTerm}, ctx::_SPFCtx)
    if e isa Symbol
        haskey(env, e) && return env[e]
        e === ctx.index && return _spf_cell(e)
        e in ctx.percell && throw(_SPFAbort())
        return _spf_shared(e)
    end
    e isa Expr || return _spf_shared(e)        # literals, GlobalRefs, QuoteNodes
    parts = _rw_read_parts(e)
    parts !== nothing && return _spf_cell(_spf_read!(parts[2], env, ctx))
    head = e.head
    if head === :block
        statements = Any[a for a in e.args if !(a isa LineNumberNode)]
        length(statements) == 1 || throw(_SPFAbort())
        return _spf_term(only(statements), env, ctx)
    elseif head === :let
        return _spf_let(e, env, ctx)
    elseif head === :&& || head === :||
        length(e.args) == 2 || throw(_SPFAbort())
        first = _spf_term(e.args[1], env, ctx)
        if _spf_isshared(first)
            # A literal decides the condition as written.
            first.orig isa Bool && return first.orig == (head === :&&) ?
                _spf_term(e.args[2], env, ctx) : _spf_shared(first.orig)
            # Decided once per call: the factored cell assumes the value under
            # which the right side is not read (false for `&&`, true for `||`).
            value = head === :||
            push!(ctx.atoms, first.orig => value)
            return _spf_shared(value)
        end
        second = _spf_term(e.args[2], env, ctx)
        return _spf_cell(Expr(head, first.orig, second.orig))
    elseif head === :if
        length(e.args) == 3 || throw(_SPFAbort())
        condition = _spf_term(e.args[1], env, ctx)
        condition.orig === true && return _spf_term(e.args[2], env, ctx)
        condition.orig === false && return _spf_term(e.args[3], env, ctx)
        yes = _spf_term(e.args[2], env, ctx)
        no = _spf_term(e.args[3], env, ctx)
        orig = Expr(:if, condition.orig, yes.orig, no.orig)
        return _spf_isshared(condition) && _spf_isshared(yes) && _spf_isshared(no) ?
            _spf_shared(orig) : _spf_cell(orig)
    elseif head === :call
        return _spf_callterm(e, env, ctx)
    elseif head in (:ref, :., :tuple)
        terms = Any[a isa QuoteNode ? _spf_shared(a) : _spf_term(a, env, ctx) for a in e.args]
        orig = Expr(head, Any[t.orig for t in terms]...)
        return all(_spf_isshared, terms) ? _spf_shared(orig) : _spf_cell(orig)
    end
    throw(_SPFAbort())
end

# `let x = v, …; body end`: the bindings read in order, then the body.
function _spf_let(e::Expr, env, ctx)
    bindings = e.args[1]
    list = Meta.isexpr(bindings, :block) ? Any[b for b in bindings.args if !(b isa LineNumberNode)] :
        Any[bindings]
    inner = copy(env)
    for b in list
        Meta.isexpr(b, :(=), 2) && b.args[1] isa Symbol || throw(_SPFAbort())
        term = _spf_term(b.args[2], inner, ctx)
        # A name in a context that is not algebra reads its value as written.
        inner[b.args[1]] = term
    end
    _spf_term(e.args[2], inner, ctx)
end

# A plate cell's read of an argument at its index: one local per distinct read,
# computed first in the cell.
function _spf_read!(call::Expr, env, ctx::_SPFCtx)
    _rw_resolve(call.args[1], @__MODULE__) in (Base.Broadcast._broadcast_getindex, _plate_read) ||
        throw(_SPFAbort())
    for a in call.args[2:end]
        a === ctx.index && continue
        a isa Symbol && !(a in ctx.percell) && !haskey(env, a) && continue
        throw(_SPFAbort())
    end
    k = findfirst(r -> first(r) == call, ctx.reads)
    k === nothing || return last(ctx.reads[k])
    name = Symbol(ctx.prefix, "_read_", length(ctx.reads) + 1)
    push!(ctx.reads, call => name)
    name
end

function _spf_callterm(e::Expr, env, ctx::_SPFCtx)
    callee = e.args[1]
    callee isa Symbol && (haskey(env, callee) || callee in ctx.percell) && throw(_SPFAbort())
    if callee isa Expr
        head = _spf_term(callee, env, ctx)
        _spf_isshared(head) || throw(_SPFAbort())
        callee = head.orig
    end
    args = e.args[2:end]
    terms = _SPFTerm[_spf_term(a, env, ctx) for a in args]
    orig = Expr(:call, callee, Any[t.orig for t in terms]...)
    all(_spf_isshared, terms) && return _spf_shared(orig)
    f = _rw_resolve(callee, @__MODULE__)
    if f === nothing
        # A call through the callables table (`__callables__[k]`) of a function.
        slot = _callable_slot(callee)
        slot !== nothing && slot <= length(ctx.ops) && ctx.ops[slot] isa Function &&
            (f = ctx.ops[slot])
    end
    if f === Base.:+
        result = terms[1]
        for k in 2:length(terms)
            result = _spf_add(result, terms[k], k == length(terms) ? orig : nothing)
        end
        return length(terms) == 1 ? _SPFTerm(result.a, result.c, result.t, orig) : result
    elseif f === Base.:- && length(terms) == 1
        return _spf_neg(terms[1], orig)
    elseif f === Base.:- && length(terms) == 2
        return _spf_add(terms[1], _spf_neg(terms[2], _spf_call(:-, terms[2].orig)), orig)
    elseif f === Base.:* || f === _native_mul
        cells = findall(!_spf_isshared, terms)
        length(cells) == 1 || return _spf_cell(orig)
        k = only(cells)
        shared = Any[t.orig for (j, t) in enumerate(terms) if j != k]
        s = foldl(_spf_mul, shared)
        return _spf_times(s, terms[k], orig)
    elseif f === Base.:/ && length(terms) == 2 && _spf_isshared(terms[2]) && !_spf_isshared(terms[1])
        return _spf_times(_spf_call(:inv, terms[2].orig), terms[1], orig)
    elseif f === Base.:^ && length(terms) == 2 && args[2] isa Int && !_spf_isshared(terms[1]) &&
            terms[1].a === nothing
        x = terms[1]
        return _SPFTerm(nothing, x.c === nothing ? nothing : Expr(:call, callee, x.c, args[2]),
                        Expr(:call, callee, x.t, args[2]), orig)
    elseif f === Base.abs && length(terms) == 1 && terms[1].a === nothing
        x = terms[1]
        return _SPFTerm(nothing, x.c === nothing ? nothing : Expr(:call, callee, x.c),
                        Expr(:call, callee, x.t), orig)
    end
    _spf_cell(orig)
end
# `_spf_add` with an `orig` of `nothing` is an intermediate of an n-ary sum.
_spf_add(x::_SPFTerm, y::_SPFTerm, ::Nothing) = _spf_add(x, y, _spf_call(:+, x.orig, y.orig))

# --- the loop -----------------------------------------------------------------

# Names bound anywhere in `ex` by assignment or `local`.
function _spf_bound!(out::Set{Symbol}, ex)
    ex isa Expr || return out
    ex.head in (:quote, :inert) && return out
    if ex.head === :(=) && ex.args[1] isa Symbol
        push!(out, ex.args[1])
    elseif ex.head === :local
        for a in ex.args
            a isa Symbol && push!(out, a)
        end
    end
    foreach(a -> _spf_bound!(out, a), ex.args)
    out
end
# Names `ex` reads.
_spf_names!(out::Set{Symbol}, ex) = ex isa Symbol ? push!(out, ex) :
    ex isa Expr && !(ex.head in (:quote, :inert)) ? (foreach(a -> _spf_names!(out, a), ex.args); out) : out

# The factored cell of a summed plate's cell statements, or `nothing`.
function _spf_factor(cellbody::Expr, index::Symbol, ops)
    statements = Any[s for s in cellbody.args if !(s isa LineNumberNode)]
    isempty(statements) && return nothing
    last_statement = statements[end]
    Meta.isexpr(last_statement, :(=), 2) || return nothing
    accumulator, update = last_statement.args
    accumulator isa Symbol &&
        _index_run_call(update, GlobalRef(@__MODULE__, :_plate_total_add), 2) &&
        update.args[2] === accumulator && update.args[3] isa Symbol || return nothing
    value = update.args[3]
    percell = _spf_bound!(Set{Symbol}(), cellbody)
    push!(percell, index)
    ctx = _SPFCtx(index, percell, Pair{Any,Bool}[], Pair{Any,Symbol}[],
                  Symbol(accumulator, _spf_mark()), ops)
    env = Dict{Symbol,_SPFTerm}()
    kept = Pair{Symbol,Any}[]
    try
        for s in statements[1:end-1]
            Meta.isexpr(s, :(=), 2) && s.args[1] isa Symbol || return nothing
            name = s.args[1]
            name === accumulator && return nothing
            term = _spf_term(s.args[2], env, ctx)
            if _spf_isshared(term)
                env[name] = term
            else
                push!(kept, name => term.orig)
                # Structure worth taking out is read as algebra; otherwise
                # the statement's value is read by its name.
                env[name] = term.a === nothing && term.c === nothing ? _spf_cell(name) :
                    _SPFTerm(term.a, term.c, term.t, name)
            end
        end
    catch err
        err isa _SPFAbort && return nothing
        rethrow()
    end
    haskey(env, value) || return nothing
    result = env[value]
    _spf_isshared(result) && return nothing
    a, c, t = result.a, result.c, result.t
    isempty(ctx.atoms) && !_spf_reads_value(a) && !_spf_reads_value(c) && return nothing
    # The statements the per-cell part reads, in their order.
    needed = _spf_names!(Set{Symbol}(), t)
    keep = Pair{Symbol,Any}[]
    for (name, rhs) in reverse(kept)
        name in needed || continue
        pushfirst!(keep, name => rhs)
        _spf_names!(needed, rhs)
    end
    # Each read as the cell spells it (`_rw_read_parts`), computed first.
    reads = Any[Expr(:(=), name, Expr(:block, Expr(:inbounds, true),
                     Expr(:local, Expr(:(=), Symbol(name, :_value), call)),
                     Expr(:inbounds, :pop), Symbol(name, :_value)))
                for (call, name) in ctx.reads if name in needed]
    (; accumulator, a, c, t, atoms = unique(ctx.atoms), prefix = ctx.prefix,
       statements = Any[reads; Any[Expr(:(=), name, rhs) for (name, rhs) in keep]])
end

# `ex` (a plate's `@simd` loop) with its cell statements after the index
# replaced by `statements`.
function _spf_with_cell(ex::Expr, index::Symbol, statements::Vector{Any})
    done = Ref(false)
    _index_run_replace(ex) do node
        done[] && return node
        node.head === :block || return node
        body = Any[a for a in node.args if !(a isa LineNumberNode)]
        isempty(body) && return node
        first_statement = body[1]
        Meta.isexpr(first_statement, :(=), 2) && first_statement.args[1] === index &&
            _index_run_call(first_statement.args[2], GlobalRef(Base, :CartesianIndex), 1) ||
            return node
        done[] = true
        Expr(:block, first_statement, statements...)
    end
end

# The factored form of one plate loop, or `nothing`.
function _spf_plate(ex::Expr, ops)
    plate = _index_run_simd_plate(ex)
    plate === nothing && return nothing
    axes, index, cellbody = plate
    factored = _spf_factor(cellbody, index, ops)
    factored === nothing && return nothing
    (; accumulator, a, c, t, atoms, prefix) = factored
    fast, partial, count, term, total, factor = (Symbol(prefix, s) for s in
        ("", "_partial", "_count", "_term", "_total", "_factor"))
    loop = _spf_with_cell(ex, index, Any[factored.statements;
        Expr(:(=), term, t);
        Expr(:(=), partial, Expr(:call, GlobalRef(@__MODULE__, :_plate_total_add), partial, term))])
    condition = Expr(:call, _spf_ref(:>), count, 0)
    for (atom, value) in atoms
        condition = Expr(:&&, condition, value ? atom : Expr(:call, _spf_ref(:!), atom))
    end
    condition = Expr(:&&, condition, Expr(:call, GlobalRef(Core, :isa), accumulator, _spf_ref(:Number)))
    # A factor that reads a value is computed once, and the factored loop runs
    # only while it is a normal number: a zero, subnormal or infinite factor
    # would lose what each cell's own division keeps. Such a factor multiplies
    # a part the cell did not compute (`(x - location)^2` for `z^2`), which can
    # overflow where the cell's own value does not, so the factored total is
    # kept only when finite. A literal factor (`-1`, `-0.5`) multiplies the
    # cell's own subexpressions, which overflow where the cell does, and is
    # decided before the loop alone: a factored loop whose result were
    # discarded would still be differentiated, its zero adjoint times an
    # infinite partial poisoning the shared values' derivatives with NaN.
    runtime = _spf_reads_value(c)
    combined = _spf_scale(runtime ? factor : c, partial)
    a === nothing || (combined = _spf_call(:+, _spf_call(:*, count, a), combined))
    usable = runtime ? quote
        if $fast
            $factor = $c
            $fast = $(GlobalRef(@__MODULE__, :_plate_factor_usable))($factor)
        end
    end : nothing
    add = :($accumulator = $(GlobalRef(@__MODULE__, :_plate_total_add))($accumulator, $total))
    # The partial sum is checked before the factor multiplies it, and the
    # total before it is kept: every product on a path whose result is
    # discarded then has finite operands, so its zero adjoint stays zero.
    keep = runtime ? quote
        $fast = $(_spf_ref(:isfinite))($partial)
        if $fast
            $total = $combined
            $fast = $(_spf_ref(:isfinite))($total)
            if $fast
                $add
            end
        end
    end : quote
        $total = $combined
        $add
    end
    factored_loop = quote
        if $fast
            $partial = $(GlobalRef(@__MODULE__, :_plate_run_seed))($accumulator)
            $loop
            $keep
        end
        if !$fast
            $ex
        end
    end
    Expr(:block,
         :($count = $(_spf_ref(:length))($(_spf_ref(:CartesianIndices))($axes))),
         :($fast = $condition),
         (usable === nothing ? () : (usable,))...,
         factored_loop.args...)
end

@inline _plate_factor_usable(c::AbstractFloat) = isfinite(c) && abs(c) >= floatmin(c)
@inline _plate_factor_usable(c) = isfinite(c)

# A block this pass produced: one of its statements binds the count.
_spf_produced(ex::Expr) = ex.head === :block && any(ex.args) do a
    Meta.isexpr(a, :(=), 2) && a.args[1] isa Symbol && endswith(String(a.args[1]), _spf_mark() * "_count")
end

function _lower_summed_plate_factors_native(body::Expr, ops)
    _simplify() && !_plate_reverse_enabled() && !_footprint_recording() || return body
    _index_run_replace(body) do ex
        # A loop this pass already rewrote keeps its own loop in the fallback:
        # a fresh node stops the descent, so the pass is idempotent.
        _spf_produced(ex) && return Expr(ex.head, ex.args...)
        ex.head === :let || return ex
        replacement = _spf_plate(ex, ops)
        replacement === nothing ? ex : replacement
    end
end
