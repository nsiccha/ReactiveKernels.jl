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
# repository does not own (docs/src/constraints.md). The primal is the
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
            s.f in _PR_NONDIFF && continue
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
    Base.length, Base.isempty, Base.CartesianIndices, Base.CartesianIndex, Base.:!,
    _promote_op, _plate_result_eltype, _plate_require_axes, _plate_broadcast_shape,
    _plate_simd_cells, _plate_total_seed, Base.zero, Base.one, Base.:<, Base.:(==)])

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
function _pr_reads!(found::Set{Symbol}, ex, mod, values)
    if ex isa Symbol
        push!(found, ex)
    elseif ex isa Expr
        ex.head in (:quote, :meta, :inbounds, :loopinfo, :line) && return found
        if ex.head === :call
            f = _pr_call_function(ex.args[1], mod, values)
            f !== nothing && f in _PR_PROGRAM_INERT && return found
            for a in ex.args[2:end]
                _pr_reads!(found, a, mod, values)
            end
            ex.args[1] isa Expr && _pr_reads!(found, ex.args[1], mod, values)
            return found
        end
        for a in ex.args
            _pr_reads!(found, a, mod, values)
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

# Every symbol that depends on an active parameter (a fixed point over the
# body's assignments; `let` bindings are assignments too).
function _pr_program_activity(body, active_params, mod, values)
    active = Set{Symbol}(active_params)
    pairs = _pr_assignments!(Pair{Symbol,Any}[], body)
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
    binding = ex.args[1]
    Meta.isexpr(binding, :(=), 2) || return nothing
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

# A loop `begin <simd let> nothing end` wraps the expansion in a block.
function _pr_cell_loop(ex)
    Meta.isexpr(ex, :block) || return _pr_simd_loop(ex)
    lets = [a for a in ex.args if Meta.isexpr(a, :let)]
    rest = [a for a in ex.args if !(a isa LineNumberNode) && !Meta.isexpr(a, :let) && a !== nothing]
    length(lets) == 1 && isempty(rest) || return nothing
    _pr_simd_loop(only(lets))
end

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

_pr_is_read(s, st) = s isa _PRCall && s.f === Base.Broadcast._broadcast_getindex &&
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
    cellvars = Set{Symbol}()
    function bound!(ss)
        for s in ss
            s isa Union{_PRAlias,_PRCall} && push!(cellvars, s.lhs)
            if s isa _PRBranch
                push!(cellvars, s.lhs); bound!(s.then_stmts); bound!(s.else_stmts)
            end
        end
    end
    bound!(stmts)
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
        hoisted = Any[s for s in folded if (s isa _PRAlias || s isa _PRCall) && s.lhs in inv2]
        inloop = Any[s for s in folded if !((s isa _PRAlias || s isa _PRCall) && s.lhs in inv2)]
        active = copy(st.active)
        for _ in 1:2
            _pr_mark_reads!(active, folded, st)
            _pr_activity!(active, folded)
        end
        # The cell's forward: hoisted invariants above the loop, the rest in it;
        # the reverse sweep covers every statement (an invariant's adjoint is
        # per cell, through its in-loop uses).
        forward_in = _pr_forward_exprs(inloop)
        forward_pre = _pr_forward_exprs(hoisted)
        reverse = Any[]
        defined = Set{Symbol}(_pr_bar(s) for s in st.outer if !(s in st.totals))
        if result isa Symbol && result in active
            rbar = _pr_bar(result)
            push!(reverse, :($rbar = $(_pr_bar(total))))
            push!(defined, rbar)
            leaf = s -> _pr_is_read(s, st) && s.lhs in active ?
                :($(GlobalRef(@__MODULE__, :_pr_shadow_add!))(
                    $(st.shadow[st.prepared[s.args[1]]]), $linear, $(_pr_bar(s.lhs)))) : nothing
            _pr_reverse_exprs!(reverse, folded, active, defined; leaf, touched = st.touched)
        end
        quote
            $(forward_pre...)
            @inbounds for $linear in $range
                $(forward_in...)
                $(reverse...)
            end
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
    for (lhs, rhs) in _pr_assignments!(Pair{Symbol,Any}[], body)
        Meta.isexpr(rhs, :call) && (_pr_is_ref(rhs.args[1], @__MODULE__, :_plate_total_add) ||
            _pr_is_ref(rhs.args[1], @__MODULE__, :_plate_total_seed)) && push!(totals, lhs)
    end
    isempty(totals) && _pr_fail("no plate total")
    output in totals || any(((lhs, rhs),) -> lhs === output && Meta.isexpr(rhs, :call, 2) &&
        _pr_is_ref(rhs.args[1], @__MODULE__, :_plate_total_value) && rhs.args[2] in totals,
        _pr_assignments!(Pair{Symbol,Any}[], body)) || _pr_fail("the program's result is not its plate total")
    prepared, broadcastables = _pr_prepared_operands(body, params)
    active = _pr_program_activity(body, active_params, mod, values)
    batched = Symbol[p for p in params if p in active_params && p in Base.values(prepared)]
    scalar = Symbol[p for p in params if p in active_params && !(p in batched)]
    for p in batched
        # A batched parameter reaches the cells only through its prepared
        # operand (aliases such as the plate's axis marker are inert).
        for (lhs, rhs) in _pr_assignments!(Pair{Symbol,Any}[], body)
            rhs === p && continue
            p in _pr_reads!(Set{Symbol}(), rhs, mod, values) || continue
            haskey(broadcastables, lhs) && continue
            # `preprocess(nothing, p)`: the body passed p in already broadcastable.
            get(prepared, lhs, nothing) === p && continue
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
the backend as before. [`plate_reverse_report`](@ref) names which plates took
the generated reverse and why the others did not.
"""
with_plate_reverse(f, enabled::Bool = true) =
    task_local_storage(f, _PLATE_REVERSE_SCOPE, enabled)

# The parameter names of a kernel body `function (__constants__, __callables__, ports...)`.
_pr_param_name(p) = p isa Symbol ? p : Meta.isexpr(p, :(::)) ? p.args[1] : nothing

# Symbols of `body` that depend on `active` (outer dataflow over assignments).
function _pr_body_activity(body, active::Set{Symbol})
    pairs = _pr_assignments!(Pair{Symbol,Any}[], body)
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
function _pr_rule_for(call::Expr, outer_active::Set{Symbol}, kernel)
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
    report = Any[]
    function replace(node)
        if node isa Expr && node.head === :(=) && node.args[2] isa Expr &&
                node.args[2].head === :call && first(node.args[2].args) isa _PlateProgram
            rule, info = _pr_rule_for(node.args[2], outer_active, kernel)
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
    plate_reverse_report(kernel::PreparedKernel, active = 1) -> Vector{Pair{Symbol,String}}

For a gradient of `kernel` with HAVE port `active` (its position, or a tuple of
positions) active: each plate program's result name and either `"rule"`, when
its gradient is the generated summed-plate reverse, or the reason the
backend's own reverse differentiates it instead. Plates the lowering keeps
inline (a body with one plate) are not programs and are not listed.
"""
function plate_reverse_report(kernel::PreparedKernel, active = 1)
    _has_plate_programs(kernel.ast) || return Pair{Symbol,String}[]
    _, report = _pr_inline_plate_programs(kernel.ast, kernel, _pr_active_ports(kernel, active))
    Pair{Symbol,String}[Symbol(first(r)) => last(r) for r in report]
end
