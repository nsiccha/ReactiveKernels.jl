# A reverse generated for summed plate programs (decision
# ReactiveKernels:plate-reverse 0ayxb72: "RK differentiates the fused cell
# graph"). The output adjoint of a sum reaches every cell unchanged, so a
# plate program's gradient is one pass over its cells: each cell runs
# forward, then its own reverse sweep seeded with the total's adjoint;
# adjoints of shared values accumulate across cells; a read of a batched
# operand adds its cotangent into that operand's shadow at the cell; and the
# invariants hoisted above the loop are swept once after it.
#
# The reverse is generated from the program's own lowered source, which is
# the plate's cell graph after every lowering decision (inline sources, lazy
# branches, hoisted invariants): each call's partials come from DiffRules
# (the established table ForwardDiff and ReverseDiff use) or, for RK's own
# primitives and calls whose derivative is better spelled through their own
# result, from the overrides below. No rule attaches to a function this
# repository does not own (CONTRIBUTING.md). The primal is the
# program itself, unchanged; lazy branches stay lazy in the reverse too: each
# arm's reverse runs inside that arm, under the stored condition.
#
# A program the generator cannot differentiate (an unknown call, a cell
# shape it does not recognise) keeps the backend's ordinary reverse, and
# `plate_reverse_report` names the reason. Opt-in: `plate_reverse`
# preference or `with_plate_reverse`.

# =============================================== expressions

struct _PRIneligible <: Exception
    reason::String
end
Base.showerror(io::IO, e::_PRIneligible) = print(io, "plate reverse: ", e.reason)
_pr_fail(reason) = throw(_PRIneligible(reason))

# One flattened statement: `lhs = f(args...)`, `f` resolved to its function
# object (`fref` is the expression the generated code calls it through).
struct _PRCall
    lhs::Symbol
    f::Any
    fref::Any
    args::Vector{Any}
end
# A lazy branch: `lhs = cond ? then : else`, each arm its own statements.
struct _PRBranch
    lhs::Symbol
    cond::Any
    then_stmts::Vector{Any}
    then_result::Any
    else_stmts::Vector{Any}
    else_result::Any
end
# A plain alias `lhs = rhs` (a symbol or literal).
struct _PRAlias
    lhs::Symbol
    rhs::Any
end

mutable struct _PRCtx
    mod::Module
    counter::Int
    values::Dict{Symbol,Any}   # program-local names whose values are known (callables)
end
_PRCtx(mod::Module, counter::Int) = _PRCtx(mod, counter, Dict{Symbol,Any}())
_pr_tmp!(ctx::_PRCtx, base = :t) = (ctx.counter += 1; Symbol("_pr_", base, ctx.counter))

# Resolve a call head to its function object and the reference the emitted
# code calls. A head that does not resolve to a constant global is opaque.
function _pr_resolve(head, mod::Module)
    if head isa Symbol
        isdefined(mod, head) || _pr_fail("unresolved call `$head`")
        return getglobal(mod, head), GlobalRef(mod, head)
    elseif head isa GlobalRef
        isdefined(head.mod, head.name) || _pr_fail("unresolved call `$head`")
        return getglobal(head.mod, head.name), head
    elseif Meta.isexpr(head, :., 2) && head.args[2] isa QuoteNode
        parent, _ = _pr_resolve(head.args[1], mod)
        parent isa Module || _pr_fail("call head `$head` is not a module member")
        name = head.args[2].value
        isdefined(parent, name) || _pr_fail("unresolved call `$head`")
        return getglobal(parent, name), GlobalRef(parent, name)
    elseif head isa Function || head isa Type
        return head, head
    end
    _pr_fail("opaque call head `$head`")
end

# A head that resolves is its function; any other head (a program-local
# callable, `getfield(record, :condition)`) is kept as written and only fails
# if the reverse sweep needs its derivative. Program-local callables resolve
# through `ctx.values` when the caller knows their values.
function _pr_resolve_lenient(head, ctx)
    if head isa Symbol && haskey(ctx.values, head)
        return ctx.values[head], head
    end
    try
        return _pr_resolve(head, ctx.mod)
    catch err
        err isa _PRIneligible || rethrow()
        h = head isa Expr ? _pr_flatten_head(head, ctx) : head
        return nothing, h
    end
end
_pr_flatten_head(head, ctx) = head

_pr_atom(x) = x isa Symbol || x isa Number || x isa QuoteNode || x isa GlobalRef ||
    x isa AbstractString || x isa Irrational

# Flatten `ex` into `stmts`; returns the atom (symbol or literal) holding its value.
function _pr_flatten!(stmts::Vector{Any}, ex, ctx::_PRCtx)
    ex isa LineNumberNode && return nothing
    if ex isa Symbol || ex isa Number || ex isa Irrational
        return ex
    elseif ex isa GlobalRef
        # A global constant (π, a module constant) is data.
        return ex
    elseif ex isa QuoteNode
        return ex
    elseif Meta.isexpr(ex, :block)
        result = nothing
        for a in ex.args
            a isa LineNumberNode && continue
            result = _pr_flatten!(stmts, a, ctx)
        end
        return result
    elseif Meta.isexpr(ex, :(=), 2) && (ex.args[1] isa Symbol ||
            Meta.isexpr(ex.args[1], :(::), 2) && ex.args[1].args[1] isa Symbol)
        # `x = rhs` or a typed local `x::T = rhs` (the lowering's names are
        # unique, so each binds once).
        lhs = ex.args[1] isa Symbol ? ex.args[1] : ex.args[1].args[1]
        rhs = _pr_flatten!(stmts, ex.args[2], ctx)
        push!(stmts, _PRAlias(lhs, rhs))
        return lhs
    elseif Meta.isexpr(ex, :let, 2)
        # `let a = x, b = y; body end`: the bindings are aliases (the inline
        # emitter names every local uniquely from its recipe position).
        bindings = ex.args[1]
        for b in (Meta.isexpr(bindings, :block) ? bindings.args : Any[bindings])
            b isa LineNumberNode && continue
            Meta.isexpr(b, :(=), 2) || _pr_fail("unsupported let binding `$b`")
            _pr_flatten!(stmts, b, ctx)
        end
        return _pr_flatten!(stmts, ex.args[2], ctx)
    elseif Meta.isexpr(ex, :local)
        # `local x = rhs` binds like an assignment; a bare declaration is nothing.
        for a in ex.args
            Meta.isexpr(a, :(=), 2) && _pr_flatten!(stmts, a, ctx)
        end
        return nothing
    elseif Meta.isexpr(ex, :meta) || Meta.isexpr(ex, :inbounds) || Meta.isexpr(ex, :loopinfo)
        return nothing
    elseif Meta.isexpr(ex, :call)
        f, fref = _pr_resolve_lenient(ex.args[1], ctx)
        expanded = _pr_expand_stable_helper(f, ex.args[2:end], ctx)
        expanded === nothing || return _pr_flatten!(stmts, expanded, ctx)
        args = Any[_pr_flatten!(stmts, a, ctx) for a in ex.args[2:end]]
        lhs = _pr_tmp!(ctx)
        push!(stmts, _PRCall(lhs, f, fref, args))
        return lhs
    elseif Meta.isexpr(ex, :if) && length(ex.args) in (2, 3)
        cond = _pr_flatten!(stmts, ex.args[1], ctx)
        then_stmts = Any[]; then_result = _pr_flatten!(then_stmts, ex.args[2], ctx)
        else_stmts = Any[]
        else_result = length(ex.args) == 3 ? _pr_flatten!(else_stmts, ex.args[3], ctx) : nothing
        lhs = _pr_tmp!(ctx, :br)
        push!(stmts, _PRBranch(lhs, cond, then_stmts, then_result, else_stmts, else_result))
        return lhs
    elseif Meta.isexpr(ex, :&&, 2) || Meta.isexpr(ex, :||, 2)
        # Short-circuit booleans are nondifferentiable control values.
        cond = _pr_flatten!(stmts, ex.args[1], ctx)
        rest = Any[]; r = _pr_flatten!(rest, ex.args[2], ctx)
        lhs = _pr_tmp!(ctx, :bool)
        push!(stmts, Meta.isexpr(ex, :&&) ?
            _PRBranch(lhs, cond, rest, r, Any[], false) :
            _PRBranch(lhs, cond, Any[], true, rest, r))
        return lhs
    end
    _pr_fail("unsupported expression `$(sprint(show, ex))`")
end

# A stable-rewrite helper call (`_rk_log_of(_native_exp, u)`, emitted by
# `src/stable_rewrites.jl`) expanded into the body of the method its operands
# select, read from Julia's lowered code of that method: the helper is its own
# single source, and the primitives of its stable form (`float`, `log1pexp`,
# `expm1`, …) take their partials as any other call does. An operand the
# generator cannot type is a cell value, which this generator takes to be a
# real scalar throughout (`Float64` selects the same `Real` method). Returns
# `nothing` for any other call, and for a method body with control flow, which
# then keeps the helper call (and fails only if its derivative is needed).
function _pr_expand_stable_helper(f, argexprs, ctx::_PRCtx)
    f isa Function && any(h -> h === f, _RK_STABLE_HELPERS) || return nothing
    types = Any[_pr_operand_type(a, ctx) for a in argexprs]
    match = Base._which(Tuple{typeof(f),types...})
    cis = Base.code_lowered(f, Tuple{types...})
    length(cis) == 1 || return nothing
    ci = only(cis)
    block = Any[]
    slots = Dict{Int,Any}(1 => f)
    for (i, a) in enumerate(argexprs)
        s = _pr_tmp!(ctx, :rw_arg)
        push!(block, :($s = $a))
        slots[i + 1] = s
    end
    ssa = Dict{Int,Any}()
    function conv(x)
        x isa Core.SSAValue && return ssa[x.id]
        x isa Core.SlotNumber && return slots[x.id]
        x isa Core.Argument && return slots[x.n]
        Meta.isexpr(x, :static_parameter) && return match.sparams[x.args[1]]
        x isa Expr && return Expr(x.head, Any[conv(a) for a in x.args]...)
        x
    end
    for (k, st) in enumerate(ci.code)
        if st isa Core.ReturnNode
            isdefined(st, :val) || return nothing
            push!(block, conv(st.val))
            return Expr(:block, block...)
        elseif st isa Union{Core.GotoNode,Core.GotoIfNot} || Meta.isexpr(st, :enter) ||
               Meta.isexpr(st, :leave)
            return nothing
        elseif st === nothing || st isa Core.NewvarNode || Meta.isexpr(st, :meta) ||
               Meta.isexpr(st, :inbounds) || Meta.isexpr(st, :loopinfo)
            ssa[k] = nothing
        elseif Meta.isexpr(st, :(=), 2) && st.args[1] isa Core.SlotNumber
            s = _pr_tmp!(ctx, :rw)
            push!(block, :($s = $(conv(st.args[2]))))
            slots[st.args[1].id] = s
            ssa[k] = s
        elseif Meta.isexpr(st, :call, 3) && st.args[1] == GlobalRef(Base, :getproperty) &&
               st.args[2] isa GlobalRef && st.args[3] isa QuoteNode &&
               isdefined(st.args[2].mod, st.args[2].name) &&
               getglobal(st.args[2].mod, st.args[2].name) isa Module
            # `M.f`, a module member: the reference itself.
            ssa[k] = GlobalRef(getglobal(st.args[2].mod, st.args[2].name), st.args[3].value)
        elseif Meta.isexpr(st, :call)
            s = _pr_tmp!(ctx, :rw)
            push!(block, :($s = $(conv(st))))
            ssa[k] = s
        else
            ssa[k] = conv(st)
        end
    end
    nothing
end

# The type of a helper operand as the generator can know it: a constant
# (function, literal, `Val(k)`), else a real scalar cell value.
function _pr_operand_type(a, ctx::_PRCtx)
    a isa Number && return typeof(a)
    if a isa GlobalRef
        isdefined(a.mod, a.name) && return typeof(getglobal(a.mod, a.name))
    elseif a isa Symbol && haskey(ctx.values, a)
        return typeof(ctx.values[a])
    elseif Meta.isexpr(a, :call, 2) && a.args[1] == GlobalRef(Base, :Val) && a.args[2] isa Integer
        return Val{a.args[2]}
    end
    Float64
end

# ---------------------------------------------------------------- partials

# Calls whose result carries no derivative: predicates, integer and Boolean
# arithmetic, shape and conversion queries.
const _PR_NONDIFF = Set{Any}(Any[Base.:<, Base.:>, Base.:<=, Base.:>=, Base.:(==), Base.:!=,
    Base.:!, Base.isfinite, Base.isnan, Base.isinf, Base.iszero, Base.isone, Base.signbit,
    Base.length, Base.size, Base.axes, Base.eachindex, Base.firstindex, Base.lastindex,
    Base.eltype, Base.typeof, Base.zero, Base.one, Base.oneunit, Base.:&, Base.:|, Base.xor,
    Base.isodd, Base.iseven, Base.Int, Base.round, Base.floor, Base.ceil, Base.trunc, Base.sign])

# Calls whose partials are better spelled through their own result `y` than by
# DiffRules' recomputation (`exp′(x) = exp(x)` would call `exp` twice).
function _pr_result_partials(f, y, args)
    f === Base.exp && return Any[y]
    f === Base.expm1 && return Any[:($y + 1.0)]
    f === Base.sqrt && return Any[:(0.5 / $y)]
    f === Base.:- && length(args) == 1 && return Any[-1.0]
    f === Base.:+ && return Any[1.0 for _ in args]
    f === Base.:- && length(args) == 2 && return Any[1.0, -1.0]
    f === Base.identity && return Any[1.0]
    f === _native_mul && length(args) == 2 && return Any[args[2], args[1]]
    f === Base.muladd && return Any[args[2], args[1], 1.0]
    # Through the divisor's inverse: one `inv` per cell, hoisted with the
    # divisor when the divisor is shared.
    f === Base.:/ && length(args) == 2 && return Any[:(inv($(args[2]))), :(-$y * inv($(args[2])))]
    f === Base.log && length(args) == 1 && return Any[:(inv($(args[1])))]
    f === Base.abs2 && return Any[:(2 * $(args[1]))]
    if f === Base.:^ && length(args) == 2 && args[2] isa Integer
        p = args[2]
        return Any[p == 2 ? :(2 * $(args[1])) : :($p * $(args[1])^$(p - 1)), nothing]
    end
    f === Base.float && return Any[1.0]
    # `convert(T, x)` to a float type (a stable form's `_rw_float`).
    f === Base.convert && length(args) == 2 && return Any[nothing, 1.0]
    if f === Base.:*
        # n-ary product: ∂/∂a_k = ∏_{j≠k} a_j
        return Any[length(args) == 2 ? args[3-k] :
                   Expr(:call, GlobalRef(Base, :*), args[[j for j in eachindex(args) if j != k]]...)
                   for k in eachindex(args)]
    end
    if f === Base.ifelse
        c = args[1]
        return Any[nothing, :($c ? 1.0 : 0.0), :($c ? 0.0 : 1.0)]
    end
    return nothing
end

const _PR_DIFFRULE_MODULES = (:Base, :SpecialFunctions, :LogExpFunctions, :NaNMath)

# The module symbol and name DiffRules registers `f` under, if any.
function _pr_diffrule_key(f, nargs::Int)
    f isa Function || return nothing
    name = nameof(f)
    for M in _PR_DIFFRULE_MODULES
        mod = M === :Base ? Base : _pr_loaded_module(M)
        mod === nothing && continue
        isdefined(mod, name) && getglobal(mod, name) === f || continue
        DiffRules.hasdiffrule(M, name, nargs) && return (M, name)
    end
    nothing
end
function _pr_loaded_module(M::Symbol)
    for (id, mod) in Base.loaded_modules
        id.name == String(M) && return mod
    end
    nothing
end

# DiffRules' expressions name functions by bare or module-qualified symbols;
# resolve every call head to a GlobalRef so the emitted code needs no imports.
function _pr_resolve_heads(ex)
    if Meta.isexpr(ex, :call)
        head = ex.args[1]
        newhead = if head isa Symbol && isdefined(Base, head)
            GlobalRef(Base, head)
        elseif Meta.isexpr(head, :., 2) && head.args[1] isa Symbol && head.args[2] isa QuoteNode
            M = head.args[1] === :Base ? Base : _pr_loaded_module(head.args[1])
            M === nothing ? head : GlobalRef(M, head.args[2].value)
        else
            head
        end
        return Expr(:call, newhead, map(_pr_resolve_heads, ex.args[2:end])...)
    elseif ex isa Expr
        return Expr(ex.head, map(_pr_resolve_heads, ex.args)...)
    end
    ex
end

# Partial derivative expressions of `y = f(args...)`, one per argument
# (`nothing` for an argument with no derivative), or `nothing` when `f` has no
# known derivative (the plate then falls back).
function _pr_partials(f, y::Symbol, args::Vector{Any})
    p = _pr_result_partials(f, y, args)
    p === nothing || return p
    key = _pr_diffrule_key(f, length(args))
    key === nothing && return nothing
    rules = DiffRules.diffrule(key[1], key[2], args...)
    rules = length(args) == 1 ? (rules,) : rules
    Any[r === :NaN ? _pr_fail("`$(key[2])` has no derivative in argument $k") :
        Expr(:let, Expr(:block), _pr_resolve_heads(r)) for (k, r) in enumerate(rules)]
end

# ---------------------------------------------------------------- reverse

# Which statements' results depend on an active symbol.
function _pr_activity!(active::Set{Symbol}, stmts)
    for s in stmts
        if s isa _PRAlias
            s.rhs isa Symbol && s.rhs in active && push!(active, s.lhs)
        elseif s isa _PRCall
            # `_rw_float` computes a stable form's float type.
            (s.f in _PR_NONDIFF || s.f === _rw_float) && continue
            any(a -> a isa Symbol && a in active, s.args) && push!(active, s.lhs)
        elseif s isa _PRBranch
            _pr_activity!(active, s.then_stmts)
            _pr_activity!(active, s.else_stmts)
            (s.then_result isa Symbol && s.then_result in active ||
             s.else_result isa Symbol && s.else_result in active) && push!(active, s.lhs)
        end
    end
    active
end

# Forward statements as Julia expressions; a branch stores its condition.
function _pr_forward_exprs(stmts)
    out = Any[]
    for s in stmts
        if s isa _PRAlias
            push!(out, :($(s.lhs) = $(s.rhs)))
        elseif s isa _PRCall
            push!(out, :($(s.lhs) = $(Expr(:call, s.fref, s.args...))))
        elseif s isa _PRBranch
            then_block = Expr(:block, _pr_forward_exprs(s.then_stmts)..., :($(s.lhs) = $(s.then_result)))
            # An `if` without an `else` arm has the value `nothing` there.
            else_block = Expr(:block, _pr_forward_exprs(s.else_stmts)..., :($(s.lhs) = $(s.else_result)))
            push!(out, Expr(:if, s.cond, then_block, else_block))
        end
    end
    out
end

_pr_bar(s::Symbol) = Symbol(s, :_bar)

# Accumulate `contribution` into the adjoint of `target`.
function _pr_accumulate!(out, defined::Set{Symbol}, target::Symbol, contribution;
                         touched = nothing)
    touched === nothing || push!(touched, target)
    bar = _pr_bar(target)
    if bar in defined
        push!(out, :($bar = $bar + $contribution))
    else
        push!(out, :($bar = $contribution))
        push!(defined, bar)
    end
end

# Symbols whose adjoints the reverse sweep of `stmts` touches.
function _pr_touched!(found::Set{Symbol}, stmts, active)
    for s in stmts
        if s isa _PRAlias
            s.rhs isa Symbol && s.rhs in active && push!(found, s.rhs)
        elseif s isa _PRCall
            for a in s.args
                a isa Symbol && a in active && push!(found, a)
            end
        elseif s isa _PRBranch
            _pr_touched!(found, s.then_stmts, active); _pr_touched!(found, s.else_stmts, active)
            s.then_result isa Symbol && s.then_result in active && push!(found, s.then_result)
            s.else_result isa Symbol && s.else_result in active && push!(found, s.else_result)
        end
    end
    found
end

# Reverse sweep over `stmts`; `defined` tracks which adjoints already exist.
# `leaf(s)` may claim a call statement whose adjoint leaves the sweep (a read
# of a batched operand): it returns the expression that sends the complete
# adjoint `bar` on, or `nothing` to differentiate the call as usual.
function _pr_reverse_exprs!(out, stmts, active::Set{Symbol}, defined::Set{Symbol};
                            leaf = s -> nothing, touched = nothing)
    for s in Iterators.reverse(stmts)
        if s isa _PRAlias
            s.lhs in active && _pr_bar(s.lhs) in defined || continue
            s.rhs isa Symbol && s.rhs in active || continue
            _pr_accumulate!(out, defined, s.rhs, _pr_bar(s.lhs); touched)
        elseif s isa _PRCall
            s.lhs in active && _pr_bar(s.lhs) in defined || continue
            claimed = leaf(s)
            if claimed !== nothing
                push!(out, claimed)
                continue
            end
            partials = _pr_partials(s.f, s.lhs, s.args)
            partials === nothing && _pr_fail("no derivative for `$(s.fref)`")
            ybar = _pr_bar(s.lhs)
            for (a, p) in zip(s.args, partials)
                a isa Symbol && a in active || continue
                p === nothing && _pr_fail("`$(s.fref)` has no derivative in an active argument")
                _pr_accumulate!(out, defined, a, p == 1.0 ? ybar : :($ybar * $p); touched)
            end
        elseif s isa _PRBranch
            s.lhs in active && _pr_bar(s.lhs) in defined || continue
            # Adjoints first reached inside an arm start at zero before it, so
            # both arms leave them defined.
            in_branch = _pr_touched!(Set{Symbol}(), Any[s], active)
            for t in sort!(collect(in_branch))
                bar = _pr_bar(t)
                bar in defined || (push!(out, :($bar = 0.0)); push!(defined, bar))
            end
            arm(stmts_, result) = begin
                o = Any[]; d = copy(defined)
                if result isa Symbol && result in active
                    _pr_accumulate!(o, d, result, _pr_bar(s.lhs); touched)
                end
                _pr_reverse_exprs!(o, stmts_, active, d; leaf, touched)
                Expr(:block, o...)
            end
            push!(out, Expr(:if, s.cond, arm(s.then_stmts, s.then_result),
                            arm(s.else_stmts, s.else_result)))
        end
    end
    out
end

"""
    _pr_scalar_reverse(ex, inputs, active_inputs, mod) -> (forward, reverse, result, bars)

Flatten the scalar expression `ex` over `inputs` and emit its forward
statements, the reverse sweep seeded with `result_bar` (a symbol the caller
binds), the result atom, and the adjoint symbol of each active input that
receives a contribution (inputs absent from `bars` have a zero adjoint).
"""
function _pr_scalar_reverse(ex, active_inputs::Vector{Symbol}, mod::Module; seed::Symbol)
    ctx = _PRCtx(mod, 0)
    stmts = Any[]
    result = _pr_flatten!(stmts, ex, ctx)
    active = _pr_activity!(Set{Symbol}(active_inputs), stmts)
    forward = _pr_forward_exprs(stmts)
    reverse = Any[]
    defined = Set{Symbol}()
    if result isa Symbol && result in active
        push!(reverse, :($(_pr_bar(result)) = $seed))
        push!(defined, _pr_bar(result))
        _pr_reverse_exprs!(reverse, stmts, active, defined)
    end
    bars = Dict{Symbol,Symbol}(a => _pr_bar(a) for a in active_inputs if _pr_bar(a) in defined)
    (; forward, reverse, result, bars)
end

# =============================================== programs

# Calls whose results carry no derivative at the program level: type, shape
# and schedule queries, and the plate's own scaffolding.
const _PR_PROGRAM_INERT = Set{Any}(Any[Base.typeof, Base.eltype, Base.axes, Base.size,
    Base.length, Base.eachindex, Base.isempty, Base.CartesianIndices, Base.CartesianIndex, Base.:!,
    _promote_op, _plate_result_eltype, _plate_require_axes, _plate_broadcast_shape,
    _plate_simd_cells, _plate_spans_axis, _plate_total_seed,
    Base.zero, Base.one, Base.:<, Base.:(==)])

function _pr_call_function(head, mod::Module, values::Dict{Symbol,Any})
    head isa Symbol && haskey(values, head) && return values[head]
    try
        first(_pr_resolve(head, mod))
    catch err
        err isa _PRIneligible || rethrow()
        nothing
    end
end

# Symbols an expression reads, outside inert calls.
function _pr_reads!(found::Set{Symbol}, ex, mod, values;
                    ignore_bindings = false, cell_reads = nothing)
    if ex isa Symbol
        push!(found, ex)
    elseif ex isa Expr
        cell_reads !== nothing && haskey(cell_reads, ex) && return found
        ex.head in (:quote, :meta, :inbounds, :loopinfo, :line) && return found
        if ignore_bindings && Meta.isexpr(ex, :(=), 2)
            lhs = ex.args[1]
            if lhs isa Symbol || Meta.isexpr(lhs, :(::), 2) && lhs.args[1] isa Symbol
                return _pr_reads!(found, ex.args[2], mod, values; ignore_bindings, cell_reads)
            end
        elseif ignore_bindings && ex.head === :local
            for a in ex.args
                # Declarations do not read a value. Initializers and writes
                # through an indexed destination still do.
                a isa Symbol && continue
                Meta.isexpr(a, :(::), 2) && a.args[1] isa Symbol && continue
                _pr_reads!(found, a, mod, values; ignore_bindings, cell_reads)
            end
            return found
        end
        if ex.head === :call
            f = _pr_call_function(ex.args[1], mod, values)
            f !== nothing && f in _PR_PROGRAM_INERT && return found
            for a in ex.args[2:end]
                _pr_reads!(found, a, mod, values; ignore_bindings, cell_reads)
            end
            ex.args[1] isa Expr && _pr_reads!(found, ex.args[1], mod, values;
                                            ignore_bindings, cell_reads)
            return found
        end
        for a in ex.args
            _pr_reads!(found, a, mod, values; ignore_bindings, cell_reads)
        end
    end
    found
end

# Assignment targets inside `ex` with their right-hand sides.
function _pr_assignments!(out, ex)
    ex isa Expr || return out
    if ex.head === :(=) && length(ex.args) == 2
        lhs = ex.args[1]
        name = lhs isa Symbol ? lhs : Meta.isexpr(lhs, :(::), 2) ? lhs.args[1] : nothing
        name isa Symbol && push!(out, name => ex.args[2])
    end
    ex.head === :quote && return out
    for a in ex.args
        _pr_assignments!(out, a)
    end
    out
end

# The dataflow of `ex` as `target => source` pairs, for activity: its
# assignments, plus every in-place write, whose array depends on what it
# writes (`x[i] = v`, and a `f!(x, …)` call by Julia's naming convention), and
# the parameters of an inlined function call, which alias their arguments both
# ways. The native lowering allocates a pointwise plate's output with
# `_plate_similar_output` and fills it inside such a call
# (`buf[index] = value`), so an assignment-only analysis would take a buffer of
# active values for a constant and drop its gradient (a downstream application’s calibrated hourly
# amplitudes, todo 093ym5a). Over-approximating activity only costs a fallback.
function _pr_dataflow!(out, ex)
    ex isa Expr || return out
    ex.head === :quote && return out
    if ex.head === :(=) && length(ex.args) == 2
        lhs = ex.args[1]
        if lhs isa Symbol || Meta.isexpr(lhs, :(::), 2) && lhs.args[1] isa Symbol
            push!(out, (lhs isa Symbol ? lhs : lhs.args[1]) => ex.args[2])
        elseif Meta.isexpr(lhs, :ref) && lhs.args[1] isa Symbol
            push!(out, lhs.args[1] => ex.args[2])
        end
    elseif ex.head === :call
        f = ex.args[1]
        name = f isa Symbol ? f : f isa GlobalRef ? f.name :
               Meta.isexpr(f, :., 2) && f.args[2] isa QuoteNode ? f.args[2].value : nothing
        if name isa Symbol && endswith(String(name), "!") && length(ex.args) >= 2 &&
                ex.args[2] isa Symbol
            push!(out, ex.args[2] => Expr(:tuple, ex.args[3:end]...))
        end
        if Meta.isexpr(f, :function, 2) || Meta.isexpr(f, :->, 2)
            ps = f.args[1]
            for (p, a) in zip(Meta.isexpr(ps, :tuple) ? ps.args : Any[ps], ex.args[2:end])
                pn = _pr_param_name(p)
                pn isa Symbol && a isa Symbol && pn !== a || continue
                push!(out, pn => a)
                push!(out, a => pn)
            end
        end
    end
    for a in ex.args
        _pr_dataflow!(out, a)
    end
    out
end

# Every symbol that depends on an active parameter (a fixed point over the
# body's dataflow; `let` bindings are assignments too).
function _pr_program_activity(body, active_params, mod, values)
    active = Set{Symbol}(active_params)
    pairs = _pr_dataflow!(Pair{Symbol,Any}[], body)
    changed = true
    while changed
        changed = false
        for (lhs, rhs) in pairs
            lhs in active && continue
            reads = _pr_reads!(Set{Symbol}(), rhs, mod, values)
            if !isdisjoint(reads, active)
                push!(active, lhs); changed = true
            end
        end
    end
    active
end

# --------------------------------------------------------------- loop shape

# The `@simd` expansion of a plate cell loop:
#   let r = RANGE; for i = simd_outer_range(r); let n = simd_inner_length(r, i)
#     if zero(n) < n; let j = zero(n); while j < n
#       local LIN = simd_index(r, i, j); CELL; j += 1; $(loopinfo) end ...
# Returns (range, linear, cell) or nothing.
function _pr_simd_loop(ex)
    Meta.isexpr(ex, :let, 2) || return nothing
    binding = _pr_let_binding(ex.args[1])
    binding === nothing && return nothing
    r, range = binding.args
    found = Ref{Any}(nothing)
    function visit(node)
        node isa Expr || return
        if node.head === :while && length(node.args) == 2
            body = node.args[2]
            stmts = Meta.isexpr(body, :block) ? body.args : Any[body]
            k = findfirst(s -> Meta.isexpr(s, :local) && Meta.isexpr(s.args[1], :(=), 2) &&
                Meta.isexpr(s.args[1].args[2], :call) &&
                _pr_is_ref(s.args[1].args[2].args[1], Base, :simd_index), stmts)
            if k !== nothing
                linear = stmts[k].args[1].args[1]
                cells = [s for s in stmts[k+1:end] if Meta.isexpr(s, :block)]
                length(cells) == 1 && (found[] = (range, linear, only(cells)))
                return
            end
        end
        foreach(visit, node.args)
    end
    visit(ex.args[2])
    found[]
end
# Whether call head `x` names `mod.name`: a GlobalRef, a qualified name, or
# the function object itself (the lowering embeds callees as values).
function _pr_is_ref(x, mod, name)
    x isa GlobalRef && return x.mod === mod && x.name === name
    Meta.isexpr(x, :., 2) && return x.args[2] == QuoteNode(name)
    x isa Symbol && return false
    isdefined(mod, name) && x === getglobal(mod, name)
end

# The one binding of a `let`: `let r = x` (`@simd` on Julia 1.10–1.12), or the
# same binding inside a block (1.13's hygienic `@simd`, `let; r = x; end`).
function _pr_let_binding(binding)
    if Meta.isexpr(binding, :block)
        bindings = Any[b for b in binding.args if !(b isa LineNumberNode)]
        length(bindings) == 1 || return nothing
        binding = only(bindings)
    end
    Meta.isexpr(binding, :(=), 2) ? binding : nothing
end

# A loop `begin <simd let> nothing end` wraps the expansion in a block.
function _pr_cell_loop(ex)
    Meta.isexpr(ex, :block) || return _pr_simd_loop(ex)
    lets = [a for a in ex.args if Meta.isexpr(a, :let)]
    rest = [a for a in ex.args if !(a isa LineNumberNode) && !Meta.isexpr(a, :let) && a !== nothing]
    length(lets) == 1 && isempty(rest) || return nothing
    _pr_simd_loop(only(lets))
end

# Whether `ex` holds a loop. The generated reverse differentiates only the
# cell loops it recognizes (`_pr_cell_loop`): any other loop a program keeps
# runs forward only, so one that reads an active value would contribute
# nothing to the gradient.
_pr_has_loop(ex) = ex isa Expr && ex.head !== :quote &&
    (ex.head === :for || ex.head === :while || any(_pr_has_loop, ex.args))

# ----------------------------------------------------------- the transform


mutable struct _PRProgramState
    mod::Module
    values::Dict{Symbol,Any}          # program-local callables
    active::Set{Symbol}
    prepared::Dict{Symbol,Symbol}     # prepared operand => the parameter it reads
    shadow::Dict{Symbol,Symbol}       # batched parameter => its shadow argument
    totals::Set{Symbol}               # running totals
    output::Symbol                    # the program's result
    touched::Set{Symbol}              # outer values whose adjoints received a contribution
    counter::Int
    outer::Set{Symbol}                # active values bound outside every cell loop
end

# The prepared operands of batched parameters:
#   a = Base.broadcastable(param); p = Base.Broadcast.preprocess(nothing, a)
function _pr_prepared_operands(body, params = Symbol[])
    pairs = _pr_assignments!(Pair{Symbol,Any}[], body)
    broadcastables = Dict{Symbol,Symbol}()
    for (lhs, rhs) in pairs
        if Meta.isexpr(rhs, :call, 2) && _pr_is_ref(rhs.args[1], Base, :broadcastable) &&
                rhs.args[2] isa Symbol
            broadcastables[lhs] = rhs.args[2]
        end
    end
    prepared = Dict{Symbol,Symbol}()
    for (lhs, rhs) in pairs
        if Meta.isexpr(rhs, :call, 3) && _pr_is_ref(rhs.args[1], Base.Broadcast, :preprocess) &&
                rhs.args[3] isa Symbol
            a = rhs.args[3]
            if haskey(broadcastables, a)
                prepared[lhs] = broadcastables[a]
            elseif a in params
                # The body already made the operand broadcastable and passes it in.
                prepared[lhs] = a
            end
        end
    end
    prepared, broadcastables
end

# Native direct reads retain the parameter's cell provenance. Approve each
# call, rather than its containing assignment or operand name: another read
# of that operand outside the cell, or at a different index, still needs an
# ordinary derivative. The shadow write below uses the cell's linear index.
function _pr_direct_operands!(prepared, body, params, active_params, broadcastables, mod, values)
    operands = Dict{Symbol,Symbol}(p => p for p in params)
    merge!(operands, broadcastables)
    reads = IdDict{Expr,Nothing}()
    function visit(node)
        node isa Expr && node.head !== :quote || return
        loop = _pr_cell_loop(node)
        if loop === nothing
            foreach(visit, node.args)
            return
        end
        _, linear, cell = loop
        statements = Any[s for s in cell.args if !(s isa LineNumberNode)]
        isempty(statements) && return
        binding = first(statements)
        Meta.isexpr(binding, :(=), 2) && binding.args[1] isa Symbol || return
        index, rhs = binding.args
        Meta.isexpr(rhs, :call, 2) && rhs.args[2] === linear &&
            _pr_call_function(rhs.args[1], mod, values) === Base.CartesianIndex || return
        rebound = _program_bound!(Set{Symbol}(), Expr(:block, statements[2:end]...))
        (index in rebound || linear in rebound) && return
        function read(ex)
            ex isa Expr && ex.head !== :quote || return
            if Meta.isexpr(ex, :call, 3) &&
                    _pr_call_function(ex.args[1], mod, values) === _plate_read &&
                    ex.args[2] isa Symbol && haskey(operands, ex.args[2]) &&
                    ex.args[3] in (index, linear)
                raw = ex.args[2]
                prepared[raw] = operands[raw]
                reads[ex] = nothing
            end
            foreach(read, ex.args)
        end
        foreach(read, statements[2:end])
    end
    visit(body)
    function check(node)
        node isa Expr && node.head !== :quote || return
        if Meta.isexpr(node, :call, 3) &&
                _pr_call_function(node.args[1], mod, values) === _plate_read &&
                get(operands, node.args[2], nothing) in active_params &&
                !haskey(reads, node)
            _pr_fail("a direct plate read does not use its cell index")
        end
        foreach(check, node.args)
    end
    check(body)
    reads
end

_pr_is_read(s, st) = s isa _PRCall &&
    s.f in (Base.Broadcast._broadcast_getindex, _plate_read) &&
    s.args[1] isa Symbol && haskey(st.prepared, s.args[1])

# Reads of active batched operands are active.
function _pr_mark_reads!(active, stmts, st)
    for s in stmts
        if _pr_is_read(s, st) && st.prepared[s.args[1]] in st.active
            push!(active, s.lhs)
        elseif s isa _PRBranch
            _pr_mark_reads!(active, s.then_stmts, st); _pr_mark_reads!(active, s.else_stmts, st)
        end
    end
    active
end

# One fused loop per assignment of the cell's shared conditions: the cell
# forward, then its reverse seeded with the total's adjoint; adjoints of outer
# values accumulate; a read of a batched operand adds its adjoint into that
# operand's shadow at the cell. A branch whose condition reads only shared
# values (the support guard `scale == 0 && x == location`) takes the same arm
# at every cell, so the condition is decided once above the loop and each
# loop keeps only its taken arms; unconditional cell statements that read
# only invariant values then run once above their loop.
# The value an atom has outside the cell: an outer symbol itself, or the
# outer symbol a cell alias chain reaches; `nothing` for a cell value.
function _pr_outer_atom(x, aliases::Dict{Symbol,Any}, cellvars::Set{Symbol})
    while x isa Symbol && haskey(aliases, x)
        x = aliases[x]
    end
    x isa Symbol ? (x in cellvars ? nothing : x) : x
end

# Unconditional statements whose inputs are all invariant (outer values,
# literals or earlier invariant statements): their values are the same at
# every cell. Reads of batched operands vary by cell.
function _pr_invariant_statements(stmts, st, cellvars::Set{Symbol})
    invariant = Set{Symbol}()
    isinv(a) = !(a isa Symbol) || !(a in cellvars) || a in invariant
    for s in stmts
        if s isa _PRAlias
            isinv(s.rhs) && push!(invariant, s.lhs)
        elseif s isa _PRCall
            _pr_is_read(s, st) && continue
            s.f === nothing && !(s.fref isa Symbol) && continue
            all(isinv, s.args) && (s.fref isa Symbol ? isinv(s.fref) : true) && push!(invariant, s.lhs)
        end
    end
    invariant
end

# Rewrite each cell-varying `lhs = x / d` with an invariant divisor `d` as
# `lhs = x * lhs_inverse` after `lhs_inverse = inv(d)`, and add the inverse to
# `invariant`, so it is hoisted above the loop with the divisor.
function _pr_hoist_inverse_divisors(stmts, invariant::Set{Symbol}, cellvars::Set{Symbol})
    out = Any[]
    for s in stmts
        if s isa _PRCall && s.f === Base.:/ && length(s.args) == 2 && !(s.lhs in invariant)
            d = s.args[2]
            if !(d isa Symbol) || d in invariant || !(d in cellvars)
                inverse = Symbol(s.lhs, :_inverse)
                push!(out, _PRCall(inverse, Base.inv, GlobalRef(Base, :inv), Any[d]))
                push!(out, _PRCall(s.lhs, Base.:*, GlobalRef(Base, :*), Any[s.args[1], inverse]))
                push!(invariant, inverse)
                continue
            end
        end
        push!(out, s)
    end
    out
end

# Specialise `stmts` with each known condition symbol fixed to a Bool: a branch
# whose condition is known keeps only its taken arm.
function _pr_fold(stmts, known::Dict{Symbol,Bool})
    out = Any[]
    for s in stmts
        if s isa _PRAlias && s.rhs isa Symbol && haskey(known, s.rhs)
            known[s.lhs] = known[s.rhs]
            push!(out, _PRAlias(s.lhs, known[s.rhs]))
        elseif s isa _PRAlias && s.rhs isa Bool
            known[s.lhs] = s.rhs
            push!(out, s)
        elseif s isa _PRBranch && s.cond isa Symbol && haskey(known, s.cond)
            taken = known[s.cond]
            arm = _pr_fold(taken ? s.then_stmts : s.else_stmts, known)
            append!(out, arm)
            res = taken ? s.then_result : s.else_result
            push!(out, _PRAlias(s.lhs, res))
            res isa Bool && (known[s.lhs] = res)
            res isa Symbol && haskey(known, res) && (known[s.lhs] = known[res])
        elseif s isa _PRBranch
            push!(out, _PRBranch(s.lhs, s.cond, _pr_fold(s.then_stmts, copy(known)), s.then_result,
                                 _pr_fold(s.else_stmts, copy(known)), s.else_result))
        else
            push!(out, s)
        end
    end
    out
end

const _PR_MAX_UNSWITCH = 2

# ------------------------------------------------- factored cell reverse
#
# The reverse of a cell multiplies every cell's adjoints by the same invariant
# factors: the total's adjoint (the seed), literal partials (`-1`, `0.5 · 2`)
# and partials that read only shared values (`1/σ`, `sd`). The sweep keeps each
# cell adjoint as a sum of terms `num · ∏ inv · var`, with `inv` invariant
# factors and `var` the one cell-varying factor (`nothing` for 1), so literal
# chains fold at generation time (`-1 · 0.5 · 2z` is `-z`). A contribution to a
# shared value's adjoint adds only `var` into a partial sum per (value,
# coefficient), which is scaled by `num · ∏ inv` once after the loop; a
# contribution of 1 at the cell's top level is the cell count times the
# coefficient. A batched operand's shadow takes `coefficient · var` per cell,
# the coefficient computed once above the loop. Reassociating the sums changes
# adjoints only at rounding level; the primal is the program's own.
#
# Coefficients are evaluated above the loop. A compound invariant partial
# (`inv(σ)`) is a coefficient only at the cell's top level, which runs at every
# cell as the hoisted invariants do; inside a branch arm it stays a cell
# factor, so an arm's partials are never evaluated where the arm is not taken.

# One term of a cell adjoint: `num · ∏ inv · var`.
struct _PRTerm
    num::Float64
    inv::Vector{Any}
    var::Any                     # a symbol, or `nothing` for 1
end

mutable struct _PRFactored
    st::_PRProgramState
    cellvars::Set{Symbol}
    invariant::Set{Symbol}       # cell statements hoisted above the loop
    shared::Set{Symbol}          # values whose adjoints accumulate across cells
    active::Set{Symbol}
    linear::Symbol
    ncells::Symbol
    pre::Vector{Any}             # above the loop: coefficients and partial sums
    post::Vector{Any}            # after it: each partial sum into its adjoint
    coefs::Dict{Any,Any}         # coefficient key => atom
    sums::Dict{Any,Symbol}       # (value, coefficient key) => partial sum
    counts::Dict{Any,Any}        # (value, coefficient key) => (value, term, unit contributions per cell)
end

_pr_fresh!(fc::_PRFactored, base) = (fc.st.counter += 1; Symbol("_pr_", base, fc.st.counter))

function _pr_is_invariant(fc::_PRFactored, x, toplevel::Bool)
    x isa Symbol && return !(x in fc.cellvars) || x in fc.invariant
    x isa Expr || return true
    toplevel || return false
    found = Set{Symbol}()
    function visit(node)
        if node isa Symbol
            push!(found, node)
        elseif node isa Expr && node.head !== :quote
            # A call head names a function, not a value.
            foreach(visit, node.head === :call ? node.args[2:end] : node.args)
        end
    end
    visit(x)
    all(s -> _pr_is_invariant(fc, s, true), found)
end

_pr_is_mul(h) = h === :* || h == GlobalRef(Base, :*)
_pr_is_neg(ex) = Meta.isexpr(ex, :call, 2) && (ex.args[1] === :- || ex.args[1] == GlobalRef(Base, :-))

# A partial as `num · ∏ inv · ∏ vars`.
function _pr_split_partial(fc::_PRFactored, p, toplevel::Bool)
    num = Ref(1.0); inv = Any[]; vars = Any[]
    function split!(x)
        if x isa Real && !(x isa Bool)
            num[] *= Float64(x)
        elseif Meta.isexpr(x, :call) && _pr_is_mul(x.args[1])
            foreach(split!, x.args[2:end])
        elseif _pr_is_neg(x)
            num[] = -num[]; split!(x.args[2])
        elseif Meta.isexpr(x, :let, 2) && Meta.isexpr(x.args[1], :block) && isempty(x.args[1].args)
            split!(x.args[2])
        elseif Meta.isexpr(x, :block) && count(a -> !(a isa LineNumberNode), x.args) == 1
            split!(only(a for a in x.args if !(a isa LineNumberNode)))
        elseif _pr_is_invariant(fc, x, toplevel)
            push!(inv, x)
        else
            push!(vars, x)
        end
    end
    split!(p)
    num[], inv, vars
end

_pr_coef_key(num, inv) = (num, sort!(map(string, inv)))

# The coefficient `num · ∏ inv` as an atom, computed once above the loop.
function _pr_coef!(fc::_PRFactored, num::Float64, inv::Vector{Any})
    isempty(inv) && return num
    get!(fc.coefs, _pr_coef_key(num, inv)) do
        num == 1.0 && length(inv) == 1 && inv[1] isa Symbol && return inv[1]
        k = _pr_fresh!(fc, :coef)
        prod = length(inv) == 1 ? inv[1] : Expr(:call, GlobalRef(Base, :*), inv...)
        push!(fc.pre, :($k = $(num == 1.0 ? prod : num == -1.0 ? :(-$prod) : :($num * $prod))))
        k
    end
end

_pr_scale_atom(c, v) = v === nothing ? c : c == 1.0 ? v : c == -1.0 ? :(-$v) : :($c * $v)

# `x` as an atom: `x` itself, or a cell temporary bound to it.
function _pr_bind!(out, fc::_PRFactored, x)
    (x isa Symbol || x isa Number) && return x
    t = _pr_fresh!(fc, :f)
    push!(out, :($t = $x))
    t
end

# The value of a sum of terms, bound in the cell.
function _pr_materialize!(out, fc::_PRFactored, terms::Vector{_PRTerm})
    pieces = Any[_pr_scale_atom(_pr_coef!(fc, t.num, t.inv), t.var) for t in terms]
    all(p -> p isa Number, pieces) && return sum(Float64, pieces; init = 0.0)
    _pr_bind!(out, fc, length(pieces) == 1 ? only(pieces) : Expr(:call, GlobalRef(Base, :+), pieces...))
end

# Terms sharing a coefficient merge into one term; more than two coefficients
# collapse into one bound value.
function _pr_normalize!(out, fc::_PRFactored, terms::Vector{_PRTerm})
    length(terms) <= 1 && return terms
    groups = Dict{Any,Vector{_PRTerm}}()
    order = Any[]
    for t in terms
        k = sort!(map(string, t.inv))
        haskey(groups, k) || push!(order, k)
        push!(get!(groups, k, _PRTerm[]), t)
    end
    if length(order) > 2
        v = _pr_materialize!(out, fc, terms)
        return [v isa Number ? _PRTerm(v, Any[], nothing) : _PRTerm(1.0, Any[], v)]
    end
    merged = _PRTerm[]
    for k in order
        g = groups[k]
        if length(g) == 1
            push!(merged, only(g))
        else
            v = _pr_materialize!(out, fc, [_PRTerm(t.num, Any[], t.var) for t in g])
            push!(merged, v isa Number ? _PRTerm(v, first(g).inv, nothing) :
                          _PRTerm(1.0, first(g).inv, v))
        end
    end
    merged
end

# Send `terms` to the adjoint of `a`: a shared value's partial sums, a cell
# value's adjoint bound across a branch (`mutable`), or a cell value's terms.
function _pr_contribute!(out, fc::_PRFactored, adj, mutable, a::Symbol, terms, toplevel::Bool)
    if a in fc.shared
        push!(fc.st.touched, a)
        for t in terms
            key = (a, _pr_coef_key(t.num, t.inv))
            if t.var === nothing && toplevel
                _, _, n = get(fc.counts, key, (a, t, 0))
                fc.counts[key] = (a, t, n + 1)
                continue
            end
            s = get!(fc.sums, key) do
                s = _pr_fresh!(fc, :sum)
                push!(fc.pre, :($s = 0.0))
                push!(fc.post, :($(_pr_bar(a)) = $(_pr_bar(a)) + $(GlobalRef(@__MODULE__, :_pr_scaled))(
                    $(_pr_coef!(fc, t.num, t.inv)), $s)))
                s
            end
            push!(out, :($s = $s + $(t.var === nothing ? 1.0 : t.var)))
        end
    elseif haskey(mutable, a)
        m = mutable[a]
        push!(out, :($m = $m + $(_pr_materialize!(out, fc, terms))))
    else
        append!(get!(adj, a, _PRTerm[]), terms)
    end
end

function _pr_adjoint_terms!(out, fc::_PRFactored, adj, mutable, s::Symbol)
    haskey(mutable, s) && return [_PRTerm(1.0, Any[], mutable[s])]
    haskey(adj, s) || return nothing
    adj[s] = _pr_normalize!(out, fc, adj[s])
end

# Symbols bound inside `stmts`, at any depth.
function _pr_bound!(found::Set{Symbol}, stmts)
    for s in stmts
        s isa Union{_PRAlias,_PRCall} && push!(found, s.lhs)
        if s isa _PRBranch
            push!(found, s.lhs); _pr_bound!(found, s.then_stmts); _pr_bound!(found, s.else_stmts)
        end
    end
    found
end

# The factored reverse sweep of `stmts`.
function _pr_factored_sweep!(out, fc::_PRFactored, stmts, adj, mutable, toplevel::Bool)
    st = fc.st; active = fc.active
    for s in Iterators.reverse(stmts)
        if s isa _PRAlias
            s.rhs isa Symbol && s.rhs in active || continue
            terms = _pr_adjoint_terms!(out, fc, adj, mutable, s.lhs)
            terms === nothing && continue
            _pr_contribute!(out, fc, adj, mutable, s.rhs, terms, toplevel)
        elseif s isa _PRCall
            s.lhs in active || continue
            terms = _pr_adjoint_terms!(out, fc, adj, mutable, s.lhs)
            terms === nothing && continue
            if _pr_is_read(s, st)
                v = _pr_materialize!(out, fc, terms)
                push!(out, :($(GlobalRef(@__MODULE__, :_pr_shadow_add!))(
                    $(st.shadow[st.prepared[s.args[1]]]), $(fc.linear), $v)))
                continue
            end
            partials = _pr_partials(s.f, s.lhs, s.args)
            partials === nothing && _pr_fail("no derivative for `$(s.fref)`")
            for (a, p) in zip(s.args, partials)
                a isa Symbol && a in active || continue
                p === nothing && _pr_fail("`$(s.fref)` has no derivative in an active argument")
                num, inv, vars = _pr_split_partial(fc, p, toplevel)
                pv = isempty(vars) ? nothing : _pr_bind!(out, fc,
                    length(vars) == 1 ? only(vars) : Expr(:call, GlobalRef(Base, :*), vars...))
                new = _PRTerm[_PRTerm(t.num * num, vcat(t.inv, inv),
                    t.var === nothing ? pv : pv === nothing ? t.var : _pr_bind!(out, fc, :($(t.var) * $pv)))
                    for t in terms]
                _pr_contribute!(out, fc, adj, mutable, a, new, toplevel)
            end
        elseif s isa _PRBranch
            s.lhs in active || continue
            terms = _pr_adjoint_terms!(out, fc, adj, mutable, s.lhs)
            terms === nothing && continue
            # Cell values bound outside the branch that an arm contributes to
            # get one bound adjoint, which both arms and later statements add to.
            inside = _pr_bound!(Set{Symbol}(), Any[s])
            for t in sort!(collect(_pr_touched!(Set{Symbol}(), Any[s], active)))
                (t in inside || t in fc.shared || haskey(mutable, t)) && continue
                m = _pr_fresh!(fc, :adj)
                prior = _pr_adjoint_terms!(out, fc, adj, mutable, t)
                push!(out, :($m = $(prior === nothing ? 0.0 : _pr_materialize!(out, fc, prior))))
                mutable[t] = m
                delete!(adj, t)
            end
            arm(stmts_, result) = begin
                o = Any[]
                a2 = Dict{Symbol,Vector{_PRTerm}}(k => copy(v) for (k, v) in adj)
                result isa Symbol && result in active &&
                    _pr_contribute!(o, fc, a2, mutable, result, terms, false)
                _pr_factored_sweep!(o, fc, stmts_, a2, mutable, false)
                Expr(:block, o...)
            end
            push!(out, Expr(:if, s.cond, arm(s.then_stmts, s.then_result),
                            arm(s.else_stmts, s.else_result)))
        end
    end
    out
end

# A value that every cell contributes the same invariant to (`log σ` takes `-1`
# from each cell) takes `n · cells · coefficient` once, after the loop.
function _pr_flush_counts!(fc::_PRFactored)
    for (key, (a, t, n)) in sort!(collect(fc.counts); by = string ∘ first)
        push!(fc.post, :($(_pr_bar(a)) = $(_pr_bar(a)) + $(GlobalRef(@__MODULE__, :_pr_scaled))(
            $(_pr_coef!(fc, t.num, t.inv)), $(n == 1 ? fc.ncells : :($n * $(fc.ncells))))))
    end
end

# A partial sum scaled by its coefficient; a sum that received nothing adds
# nothing, even when the coefficient is not finite (an arm no cell took).
@inline _pr_scaled(k, acc) = iszero(acc) ? zero(k * acc) : k * acc

# ------------------------------------------------------- direct operand reads
#
# A cell reads its operands through broadcast extrusion
# (`_broadcast_getindex(preprocess(nothing, x), CartesianIndex(i))`), which
# selects, per access and dimension, the cell's index or the operand's single
# element. When every operand the cell loop reads spans the plate in every
# dimension, which the plate's shape check then makes the plate's own axes, the
# fused loop reads `x[i]` instead: a check above the loop picks that copy of
# the loop or the extruded one, so the reverse pass carries no per-read select.
@inline _pr_spans(A::Base.Broadcast.Extruded, cells) = ndims(A.x) == ndims(cells) && all(A.keeps)
@inline _pr_spans(::Union{Number,Base.RefValue}, cells) = true
@inline _pr_spans(_, cells) = false
@inline _pr_spans_all(cells, operands...) = all(A -> _pr_spans(A, cells), operands)
@inline _pr_direct_read(A::Base.Broadcast.Extruded, i) = @inbounds A.x[i]
@inline _pr_direct_read(x::Number, i) = x
@inline _pr_direct_read(r::Base.RefValue, i) = r[]

# The prepared operands the statements read, and the statements with each read
# of them direct.
function _pr_direct_reads(stmts, st::_PRProgramState, linear::Symbol, found::Vector{Symbol} = Symbol[])
    out = Any[]
    for s in stmts
        if _pr_is_read(s, st)
            s.args[1] in found || push!(found, s.args[1])
            push!(out, _PRCall(s.lhs, _pr_direct_read, GlobalRef(@__MODULE__, :_pr_direct_read),
                               Any[s.args[1], linear]))
        elseif s isa _PRBranch
            push!(out, _PRBranch(s.lhs, s.cond, _pr_direct_reads(s.then_stmts, st, linear, found)[1],
                                 s.then_result, _pr_direct_reads(s.else_stmts, st, linear, found)[1],
                                 s.else_result))
        else
            push!(out, s)
        end
    end
    out, found
end

function _pr_fused_loop(range, linear, cell::Expr, st::_PRProgramState)
    body = Any[a for a in cell.args if !(a isa LineNumberNode)]
    isempty(body) && _pr_fail("empty cell")
    last_stmt = body[end]
    Meta.isexpr(last_stmt, :(=), 2) && last_stmt.args[1] in st.totals ||
        _pr_fail("cell loop does not end by accumulating the total")
    total = last_stmt.args[1]
    acc = last_stmt.args[2]
    Meta.isexpr(acc, :call, 3) && _pr_is_ref(acc.args[1], @__MODULE__, :_plate_total_add) &&
        acc.args[2] === total || _pr_fail("unsupported total accumulation `$acc`")
    any(s -> !isdisjoint(st.totals, _pr_reads!(Set{Symbol}(), s, st.mod, st.values)),
        body[1:end-1]) && _pr_fail("a cell reads a running total")
    stmts = Any[]
    ctx = _PRCtx(st.mod, st.counter, st.values)
    for s in body[1:end-1]
        _pr_flatten!(stmts, s, ctx)
    end
    density = acc.args[3]
    result = density isa Symbol ? density : _pr_flatten!(stmts, density, ctx)
    st.counter = ctx.counter
    # Symbols the cell binds (at any depth), and its top-level aliases.
    cellvars = _pr_bound!(Set{Symbol}(), stmts)
    # The loop index varies by cell, and so does everything computed from it.
    push!(cellvars, linear)
    aliases = Dict{Symbol,Any}(s.lhs => s.rhs for s in stmts if s isa _PRAlias)
    invariant = _pr_invariant_statements(stmts, st, cellvars)
    # Shared conditions: invariant branch conditions, deduplicated by their
    # defining call over outer atoms (`#scale#op29 == 0` and `#scale#op31 == 0`
    # both read `sigma == 0`).
    defs = Dict{Symbol,_PRCall}(s.lhs => s for s in stmts if s isa _PRCall)
    conds = Dict{Any,Vector{Symbol}}()
    function collect_conds(ss)
        for s in ss
            s isa _PRBranch || continue
            c = s.cond
            if c isa Symbol && c in invariant && haskey(defs, c)
                d = defs[c]
                key = (d.fref, Tuple(_pr_outer_atom(a, aliases, cellvars) for a in d.args))
                any(isnothing, key[2]) || push!(get!(conds, key, Symbol[]), c)
            end
            collect_conds(s.then_stmts); collect_conds(s.else_stmts)
        end
    end
    collect_conds(stmts)
    keys_ = collect(keys(conds))
    length(keys_) > _PR_MAX_UNSWITCH && (keys_ = Any[])
    cond_syms = [gensym(:shared_condition) for _ in keys_]
    pre = Any[:($(cs) = $(Expr(:call, k[1], k[2]...))::Bool) for (cs, k) in zip(cond_syms, keys_)]
    function variant(assignment)
        known = Dict{Symbol,Bool}()
        for (k, v) in zip(keys_, assignment), c in conds[k]
            known[c] = v
        end
        # Fixing a condition fixes its statement too.
        folded = _pr_fold(Any[s isa _PRCall && haskey(known, s.lhs) ? _PRAlias(s.lhs, known[s.lhs]) : s
                              for s in stmts], known)
        inv2 = _pr_invariant_statements(folded, st, cellvars)
        # A cell value divided by an invariant one multiplies by the divisor's
        # inverse, computed once above the loop. This recomputed forward only
        # feeds partials; the primal value is the program's own.
        folded = _pr_hoist_inverse_divisors(folded, inv2, cellvars)
        hoisted = Any[s for s in folded if (s isa _PRAlias || s isa _PRCall) && s.lhs in inv2]
        inloop = Any[s for s in folded if !((s isa _PRAlias || s isa _PRCall) && s.lhs in inv2)]
        active = copy(st.active)
        for _ in 1:2
            _pr_mark_reads!(active, folded, st)
            _pr_activity!(active, folded)
        end
        # The cell's forward: hoisted invariants above the loop, the rest in it.
        forward_in = _pr_forward_exprs(inloop)
        forward_pre = _pr_forward_exprs(hoisted)
        reverse = Any[]
        after = Any[]
        # A hoisted invariant is a shared value of the loop: its adjoint
        # accumulates across cells from its in-loop uses, and its own reverse
        # runs once, after the loop.
        hoisted_active = Symbol[s.lhs for s in hoisted if s.lhs in active]
        init = Any[:($(_pr_bar(h)) = 0.0) for h in hoisted_active]
        defined = Set{Symbol}(_pr_bar(s) for s in st.outer if !(s in st.totals))
        union!(defined, (_pr_bar(h) for h in hoisted_active))
        st.counter += 1
        cells = Symbol(:_pr_cells, st.counter)
        ncells = Symbol(:_pr_ncells, st.counter)
        coefficients = Any[]
        flush = Any[]
        if result isa Symbol && result in active
            shared = Set{Symbol}(s for s in st.outer if !(s in st.totals))
            union!(shared, hoisted_active)
            fc = _PRFactored(st, cellvars, inv2, shared, active, linear, ncells,
                             Any[], Any[], Dict{Any,Any}(), Dict{Any,Symbol}(), Dict{Any,Any}())
            # The cell density's adjoint is the total's (the seed), the same
            # at every cell.
            adj = Dict{Symbol,Vector{_PRTerm}}()
            mutable = Dict{Symbol,Symbol}()
            _pr_contribute!(reverse, fc, adj, mutable, result,
                            [_PRTerm(1.0, Any[_pr_bar(total)], nothing)], true)
            _pr_factored_sweep!(reverse, fc, inloop, adj, mutable, true)
            _pr_flush_counts!(fc)
            coefficients, flush = fc.pre, fc.post
            _pr_reverse_exprs!(after, hoisted, active, copy(defined); touched = st.touched)
        end
        direct_in, read_operands = _pr_direct_reads(inloop, st, linear)
        loop = isempty(read_operands) ?
            :(@inbounds for $linear in $cells
                  $(forward_in...)
                  $(reverse...)
              end) :
            :(if $(GlobalRef(@__MODULE__, :_pr_spans_all))($cells, $(read_operands...))
                  @inbounds for $linear in $cells
                      $(_pr_forward_exprs(direct_in)...)
                      $(reverse...)
                  end
              else
                  @inbounds for $linear in $cells
                      $(forward_in...)
                      $(reverse...)
                  end
              end)
        quote
            $(forward_pre...)
            $(init...)
            $cells = $range
            $ncells = Float64(length($cells))
            $(coefficients...)
            $loop
            $(flush...)
            $(after...)
        end
    end
    isempty(keys_) && return variant(())
    # One loop per assignment of the shared conditions.
    function nest(i, assignment)
        i > length(keys_) && return variant(assignment)
        Expr(:if, cond_syms[i], nest(i + 1, (assignment..., true)), nest(i + 1, (assignment..., false)))
    end
    Expr(:block, pre..., nest(1, ()))
end

@inline _pr_shadow_add!(::Nothing, i, v) = nothing
@inline function _pr_shadow_add!(shadow::AbstractArray, i, v)
    @inbounds shadow[i] += v
    nothing
end

# A straight-line program statement `lhs = rhs` outside cell loops: kept as
# written in the forward, swept in reverse at the end of its block.
struct _PRStatement
    lhs::Symbol
    rhs::Any
end

# Transform one block: its forward statements (cell loops fused, branch arms
# transformed in place), then the reverse of its straight-line active
# statements whose adjoints received a contribution.
function _pr_program_block(stmts, st::_PRProgramState)
    fwd = Any[]
    record = _PRStatement[]
    for s in stmts
        s isa LineNumberNode && continue
        loop = _pr_cell_loop(s)
        if loop !== nothing
            push!(fwd, _pr_fused_loop(loop..., st))
        elseif Meta.isexpr(s, :if)
            arms = Any[s.args[1]]
            for arm in s.args[2:end]
                push!(arms, Expr(:block, _pr_program_block(
                    Meta.isexpr(arm, :block) ? arm.args : Any[arm], st)...))
            end
            push!(fwd, Expr(:if, arms...))
        elseif Meta.isexpr(s, :block)
            append!(fwd, _pr_program_block(s.args, st))
        else
            _pr_has_loop(s) && !isdisjoint(_pr_reads!(Set{Symbol}(), s, st.mod, st.values), st.active) &&
                _pr_fail("a loop over active values that is not a summed cell loop")
            push!(fwd, s)
            for (lhs, rhs) in _pr_assignments!(Pair{Symbol,Any}[], s)
                lhs in st.active || continue
                lhs in st.totals && continue
                if lhs === st.output
                    # `output = _plate_total_value(total)`: the total's adjoint
                    # is the seed, set before the body.
                    Meta.isexpr(rhs, :call, 2) && rhs.args[2] in st.totals ||
                        _pr_fail("unsupported program result `$rhs`")
                    continue
                end
                push!(record, _PRStatement(lhs, rhs))
            end
        end
    end
    rev = Any[]
    for s in Iterators.reverse(record)
        s.lhs in st.touched || continue
        ctx = _PRCtx(st.mod, st.counter, st.values)
        sub = Any[]
        r = _pr_flatten!(sub, s.rhs, ctx)
        st.counter = ctx.counter
        active = _pr_activity!(copy(st.active), sub)
        defined = Set{Symbol}(_pr_bar(a) for a in st.outer if !(a in st.totals))
        if r isa Symbol && r in active && r !== s.lhs
            append!(rev, _pr_forward_exprs(sub))
            push!(rev, :($(_pr_bar(r)) = $(_pr_bar(s.lhs))))
            push!(defined, _pr_bar(r))
            _pr_reverse_exprs!(rev, sub, active, defined; touched = st.touched)
        end
    end
    vcat(fwd, rev)
end

"""
    _pr_program_reverse(ast, active_params, mod, values) -> (expr, scalar, batched)

The reverse of plate program `ast` (a `function (params...) body; result end`
expression). `active_params` are the parameters that carry derivatives;
`values` maps program-local callable parameters to their values. Returns the
reverse function's expression `(params..., ȳ, shadows...) -> cotangents of the
scalar active parameters`, the scalar active parameters in order, and the
batched (per-cell) active parameters, whose shadows follow ȳ in order.
"""
function _pr_program_reverse(ast::Expr, active_params, mod::Module, values::Dict{Symbol,Any})
    Meta.isexpr(ast, :function, 2) && Meta.isexpr(ast.args[1], :tuple) ||
        _pr_fail("not a plate program")
    params = Symbol[p for p in ast.args[1].args]
    body = ast.args[2]
    stmts = Any[a for a in body.args if !(a isa LineNumberNode)]
    output = stmts[end]
    output isa Symbol || _pr_fail("a plate program with several results")
    stmts = stmts[1:end-1]
    # Running totals: seeded (`_plate_total_seed`) or accumulated
    # (`_plate_total_add`), including a total the body seeds and passes in.
    totals = Set{Symbol}()
    assignments = _pr_assignments!(Pair{Symbol,Any}[], body)
    for (lhs, rhs) in assignments
        Meta.isexpr(rhs, :call) && (_pr_is_ref(rhs.args[1], @__MODULE__, :_plate_total_add) ||
            _pr_is_ref(rhs.args[1], @__MODULE__, :_plate_total_seed)) && push!(totals, lhs)
    end
    isempty(totals) && _pr_fail("no plate total")
    terminal_totals = Set{Symbol}(t for t in totals if t === output)
    for (lhs, rhs) in assignments
        lhs === output && Meta.isexpr(rhs, :call, 2) &&
            _pr_is_ref(rhs.args[1], @__MODULE__, :_plate_total_value) &&
            rhs.args[2] in totals && push!(terminal_totals, rhs.args[2])
    end
    isempty(terminal_totals) && _pr_fail("the program's result is not its plate total")
    # A factored reduction also has a partial total whose adjoint is scaled
    # by later arithmetic. Seeding that partial with the output's seed would
    # be wrong. Distinct terminal totals in mutually exclusive branches may
    # each use that seed; counting totals alone would reject those programs.
    issubset(totals, terminal_totals) ||
        _pr_fail("an intermediate plate total needs a propagated adjoint")
    prepared, broadcastables = _pr_prepared_operands(body, params)
    cell_reads = _pr_direct_operands!(prepared, body, params, active_params,
                                    broadcastables, mod, values)
    active = _pr_program_activity(body, active_params, mod, values)
    batched = Symbol[p for p in params if p in active_params && p in Base.values(prepared)]
    scalar = Symbol[p for p in params if p in active_params && !(p in batched)]
    # A marker returns an operand alias, not detached metadata. Permit it
    # only when its result is read exclusively by derivative-inert queries.
    # Keep the call and its checks in the program; a numerical read, another
    # alias or a write through the marker retains the conservative fallback.
    value_reads = _pr_reads!(Set{Symbol}(), body, mod, values; ignore_bindings = true)
    inert_markers = Set(lhs for (lhs, rhs) in assignments if
        Meta.isexpr(rhs, :call) &&
        _pr_is_ref(rhs.args[1], @__MODULE__, :_authored_plate_marker) &&
        !(lhs in value_reads))
    for p in batched
        # A batched parameter reaches the cells only through its prepared
        # operand (aliases such as the plate's axis marker are inert).
        for (lhs, rhs) in assignments
            rhs === p && continue
            p in _pr_reads!(Set{Symbol}(), rhs, mod, values; cell_reads) || continue
            haskey(broadcastables, lhs) && continue
            # `preprocess(nothing, p)`: the body passed p in already broadcastable.
            get(prepared, lhs, nothing) === p && continue
            lhs in inert_markers && Meta.isexpr(rhs, :call) &&
                _pr_is_ref(rhs.args[1], @__MODULE__, :_authored_plate_marker) && continue
            _pr_fail("batched operand `$p` is also read directly by `$lhs`")
        end
    end
    seed = gensym(:ybar)
    shadow = Dict{Symbol,Symbol}(p => Symbol(p, :_shadow) for p in batched)
    # Values bound inside a cell are per cell: their adjoints restart in every
    # cell. Only values bound outside every cell loop accumulate across cells.
    cell_locals = Set{Symbol}()
    function collect_cells(node)
        node isa Expr || return
        loop = _pr_cell_loop(node)
        if loop !== nothing
            for (lhs, _) in _pr_assignments!(Pair{Symbol,Any}[], loop[3])
                push!(cell_locals, lhs)
            end
            return
        end
        foreach(collect_cells, node.args)
    end
    collect_cells(body)
    outer = Set{Symbol}(a for a in active if !(a in cell_locals))
    st = _PRProgramState(mod, values, active, prepared, shadow, totals, output,
                         Set{Symbol}(), 0, outer)
    new_body = _pr_program_block(stmts, st)
    # Every outer adjoint starts at zero; every total's adjoint is the seed.
    init = Any[:($(_pr_bar(a)) = 0.0) for a in sort!(collect(outer)) if !(a in totals)]
    seeds = Any[:($(_pr_bar(t)) = $seed) for t in sort!(collect(totals))]
    checks = Any[:($(GlobalRef(@__MODULE__, :_pr_check_shadow))($(shadow[p]), $p)) for p in batched]
    fexpr = Expr(:function,
        Expr(:tuple, params..., seed, (shadow[p] for p in batched)...),
        Expr(:block, checks..., init..., seeds..., new_body...,
             Expr(:tuple, (_pr_bar(p) for p in scalar)...)))
    fexpr, scalar, batched
end

_pr_check_shadow(::Nothing, x) = nothing
# A tuple operand's cotangent accumulates in a vector the adapter provides.
function _pr_check_shadow(shadow::AbstractVector, x::Tuple)
    length(shadow) == length(x) || throw(ArgumentError(
        "plate reverse: a tuple operand of length $(length(x)) has a shadow of length $(length(shadow))"))
    nothing
end
function _pr_check_shadow(shadow::AbstractArray, x::AbstractArray)
    axes(shadow) == axes(x) && ndims(x) == 1 || throw(ArgumentError(
        "plate reverse: a batched operand's shadow has axes $(axes(shadow)), its value $(axes(x))"))
    nothing
end

# =============================================== rules

"""
    _PlateSumRule{S,B}

An RK-owned callable for one summed plate program: calling it runs the
program (`program`, the plate's own compiled loop, so values are unchanged).
`reverse` is generated from the program's source: `(args..., ȳ, shadows...)`
re-runs the cells, adds each batched operand's cotangent into its shadow and
returns the cotangents of the scalar operands at positions `S`; `B` are the
positions of the batched operands. The Enzyme extension stages it.
"""
struct _PlateSumRule{S,B,P,R} <: Function
    program::P
    reverse::R
end
# One method per operand count, each operand's type a parameter of its own:
# Julia does not otherwise specialize on a `Function` operand it only passes on
# (the program's callables), and the call would dispatch dynamically
# (`_PlateCall` has the same methods).
for count in 0:_PLATE_CALL_MAX_ARGUMENTS
    local arguments = Any[Symbol("argument", index) for index in 1:count]
    local types = Any[Symbol("Argument", index) for index in 1:count]
    local typed = Any[:($a::$T) for (a, T) in zip(arguments, types)]
    @eval @inline (rule::_PlateSumRule)($(typed...)) where {$(types...)} =
        getfield(rule, :program)($(arguments...))
end
_pr_scalar_positions(::_PlateSumRule{S}) where {S} = S
_pr_batched_positions(::_PlateSumRule{S,B}) where {S,B} = B

# Whether the AD path replaces eligible plate programs with generated rules:
# opt-in while the reverse is validated across the corpus. The `plate_reverse`
# preference (`"off"`, the default, or `"on"`) sets it for a project;
# `with_plate_reverse` sets it for the preparations one call makes.
const _PLATE_REVERSE_SCOPE = :ReactiveKernels_plate_reverse
function _plate_reverse_enabled()
    scoped = get(task_local_storage(), _PLATE_REVERSE_SCOPE, nothing)
    scoped === nothing || return scoped::Bool
    value = Preferences.load_preference(@__MODULE__, "plate_reverse", "off")
    value in ("off", "on") || throw(ArgumentError(
        "the ReactiveKernels preference `plate_reverse` must be \"off\" or \"on\"; got $(repr(value))"))
    value == "on"
end

"""
    with_plate_reverse(f, enabled::Bool = true)

Run `f()` with the generated summed-plate reverse `enabled` (or disabled) for
every AD preparation `f` makes, whatever the `plate_reverse` preference says.
A prepared gradient differentiates each eligible summed plate program with a
reverse generated from its own code; every other plate is differentiated by
the backend as before. This also applies to non-allocating kernels after
their cache activity and ownership rewrite. [`plate_reverse_report`](@ref)
reports static eligibility; backend operand annotations can still require
the ordinary reverse at execution.

Include kernel preparation in this scope as well as AD preparation. Native
optimizations chosen while the rule is disabled can produce forms outside
the generated reverse's current coverage; enabling it only for AD can then
retain the backend's ordinary reverse for those regions.
"""
with_plate_reverse(f, enabled::Bool = true) =
    task_local_storage(f, _PLATE_REVERSE_SCOPE, enabled)

# The parameter names of a kernel body `function (__constants__, __callables__, ports...)`.
_pr_param_name(p) = p isa Symbol ? p : Meta.isexpr(p, :(::)) ? p.args[1] : nothing

# Symbols of `body` that depend on `active` (outer dataflow, `_pr_dataflow!`).
function _pr_body_activity(body, active::Set{Symbol})
    pairs = _pr_dataflow!(Pair{Symbol,Any}[], body)
    changed = true
    while changed
        changed = false
        for (lhs, rhs) in pairs
            lhs in active && continue
            if rhs isa Expr && rhs.head === :call && first(rhs.args) isa _PlateProgram
                # a program's result depends on its active arguments
                any(a -> a isa Symbol && a in active, rhs.args[2:end]) || continue
            else
                isdisjoint(_pr_reads!(Set{Symbol}(), rhs, @__MODULE__, Dict{Symbol,Any}()), active) &&
                    continue
            end
            push!(active, lhs); changed = true
        end
    end
    active
end

# The value of a program argument that reads a table: `__callables__[k]`.
function _pr_table_value(arg, kernel)
    Meta.isexpr(arg, :ref, 2) && arg.args[2] isa Integer || return nothing
    arg.args[1] === _CALLABLES_ARG && return (true, kernel.callables[arg.args[2]])
    arg.args[1] === _CONSTANTS_ARG && return (true, kernel.constants[arg.args[2]])
    nothing
end

# A generated rule for the program call `call` (`program(args...)`), or the
# reason it falls back.
# The expression a binding's value comes from: the last expression of a `let`
# or a block, else the expression itself.
function _pr_result_expr(ex)
    while ex isa Expr && ex.head in (:let, :block)
        body = ex.head === :let ? ex.args[2] : ex
        statements = Meta.isexpr(body, :block) ? body.args : Any[body]
        i = findlast(s -> !(s isa LineNumberNode), statements)
        i === nothing && return nothing
        ex = statements[i]
    end
    ex
end

# Whether `ex` can be a lazy cell array (`_CellArray`): a `_cell_array_broadcast`
# or `_cell_array_hcat` result, an alias of a name in `lazy`, or a declaration
# that keeps it (`_cell_array_as`).
_pr_lazy_value(ex::Symbol, lazy) = ex in lazy
function _pr_lazy_value(ex::Expr, lazy)
    ex.head === :call || return false
    f = _cell_array_function(first(ex.args))
    (f === _cell_array_broadcast || f === _cell_array_hcat) && return true
    f === _cell_array_as && length(ex.args) == 3 && return _pr_lazy_value(ex.args[3], lazy)
    f === identity && length(ex.args) == 2 && return _pr_lazy_value(ex.args[2], lazy)
    false
end
_pr_lazy_value(ex, lazy) = false

# The names `body` binds to a value that can be a lazy cell array. Its entries
# are computed on read, and its shadow is the same lazy structure over its
# operands' shadows, which has no entries to add a cell's cotangent into.
function _pr_lazy_cell_arrays(body)
    lazy = Set{Symbol}()
    bindings = _pr_assignments!(Pair{Symbol,Any}[], body)
    changed = true
    while changed
        changed = false
        for (lhs, rhs) in bindings
            lhs in lazy && continue
            _pr_lazy_value(_pr_result_expr(rhs), lazy) || continue
            push!(lazy, lhs)
            changed = true
        end
    end
    lazy
end

function _pr_rule_for(call::Expr, outer_active::Set{Symbol}, kernel,
                      lazy::Set{Symbol} = Set{Symbol}())
    program = first(call.args)::_PlateProgram
    params = Symbol[p for p in program.ast.args[1].args]
    args = call.args[2:end]
    values = Dict{Symbol,Any}()
    active_params = Symbol[]
    for (p, a) in zip(params, args)
        tv = _pr_table_value(a, kernel)
        if tv !== nothing
            values[p] = tv[2]
        elseif a isa Symbol && a in outer_active
            push!(active_params, p)
        end
    end
    isempty(active_params) && return nothing, "no active operand"
    try
        rexpr, scalar, batched = _pr_program_reverse(program.ast, active_params, @__MODULE__, values)
        # A batched operand's cotangent is added into its shadow cell by cell,
        # which a lazy cell array's shadow cannot take: the program keeps the
        # backend's reverse (snag generated-plate-56eaf6e3).
        for p in batched
            a = args[findfirst(==(p), params)]
            a isa Symbol && a in lazy && _pr_fail(
                "batched operand `$a` is a lazy cell array, whose shadow the generated reverse does not write")
        end
        S = Tuple(findfirst(==(p), params) for p in scalar)
        B = Tuple(findfirst(==(p), params) for p in batched)
        # Called through `_PlateCall`'s typed methods: a runtime-generated
        # function's own call method takes its operands as one tuple and does
        # not specialize on the program's callables, so the call would
        # dispatch dynamically and box every operand.
        primal = _PlateCall(compile(program.ast))
        rev = _PlateCall(compile(rexpr))
        return _PlateSumRule{S,B,typeof(primal),typeof(rev)}(primal, rev), rexpr
    catch err
        err isa _PRIneligible || rethrow()
        return nothing, err.reason
    end
end

# The AD body with each eligible plate program a generated rule and every
# other program inline, as `_inline_plate_programs` inlines them, and the
# report of each program: `(result, "rule" or the reason it falls back)`.
function _pr_inline_plate_programs(ast::Expr, kernel, active_ports)
    outer_active = _pr_body_activity(ast.args[2], Set{Symbol}(active_ports))
    lazy = _pr_lazy_cell_arrays(ast.args[2])
    report = Any[]
    function replace(node)
        if node isa Expr && node.head === :(=) && node.args[2] isa Expr &&
                node.args[2].head === :call && first(node.args[2].args) isa _PlateProgram
            rule, info = _pr_rule_for(node.args[2], outer_active, kernel, lazy)
            push!(report, (node.args[1], rule === nothing ? info : "rule"))
            rule === nothing && return node
            return Expr(:(=), node.args[1], Expr(:call, rule, node.args[2].args[2:end]...))
        elseif node isa Expr && node.head !== :quote
            return Expr(node.head, Any[replace(a) for a in node.args]...)
        end
        node
    end
    new = replace(ast)
    # The remaining programs inline, as before.
    _inline_plate_programs(new), report
end

_pr_active_ports(kernel::PreparedKernel, I) = begin
    ports = Symbol[_pr_param_name(p) for p in kernel.ast.args[1].args[3:end]]
    I isa Integer ? Symbol[ports[I]] : Symbol[ports[i] for i in I]
end

_pr_active_ports(kernel::NonAllocatingKernel, I) = begin
    ports = Symbol[_pr_param_name(p) for p in kernel.ast.args[1].args[5:end]]
    I isa Integer ? Symbol[ports[I]] : Symbol[ports[i] for i in I]
end

# The AD kernel (`_ad_inline_plate_programs`) for active selector `I`: with the
# generated plate reverse enabled, each eligible plate program a rule.
Base.@nospecializeinfer function _ad_inline_plate_programs(
        @nospecialize(kernel::PreparedKernel), I)
    _plate_reverse_enabled() && _has_plate_programs(kernel.ast) ||
        return _ad_inline_plate_programs(kernel)
    ast, _ = _pr_inline_plate_programs(kernel.ast, kernel, _pr_active_ports(kernel, I))
    native = compile(ast)
    f = kernel.f isa _ArrayFunction ? _ArrayFunction(native) : native
    _executed_prepared_kernel(f, kernel.constants, kernel.callables, kernel.inputs,
        kernel.outputs, kernel.plan, ast, kernel.lowered_recipes,
        _exact_signature(kernel), kernel.rebinding)
end

"""
    plate_reverse_report(kernel, active = 1) -> Vector{Pair{Symbol,String}}

For a gradient of `kernel` with HAVE port `active` (its position, or a tuple of
positions) active: each plate program's result name and either `"rule"`, when
the source is eligible for generated reverse, or its fallback reason.
Accepts a `PreparedKernel` or `NonAllocatingKernel`. This is a static report,
independent of whether the preference is enabled; cache rewrites and backend
operand annotations can affect the rule used by a prepared gradient.
Single-plate programs are listed even when ordinary native execution inlines
them.
"""
function plate_reverse_report(kernel::Union{PreparedKernel,NonAllocatingKernel}, active = 1)
    _has_plate_programs(kernel.ast) || return Pair{Symbol,String}[]
    _, report = _pr_inline_plate_programs(kernel.ast, kernel, _pr_active_ports(kernel, active))
    Pair{Symbol,String}[Symbol(first(r)) => last(r) for r in report]
end
