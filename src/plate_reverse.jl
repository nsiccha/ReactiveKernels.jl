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
# branches, hoisted invariants): owned scalar rules provide joint cuts of
# their value/partial graph. Other calls' partials come from DiffRules
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
    prefix::Symbol
end
_PRCtx(mod::Module, counter::Int, values::Dict{Symbol,Any}) = _PRCtx(mod, counter, values, :_pr_)
_PRCtx(mod::Module, counter::Int) = _PRCtx(mod, counter, Dict{Symbol,Any}())
_pr_tmp!(ctx::_PRCtx, base = :t) = (ctx.counter += 1; Symbol(ctx.prefix, base, ctx.counter))

# Resolve a call head to its function object and the reference the emitted
# code calls. A head that does not resolve to a constant global is opaque.
function _pr_resolve(head, mod::Module)
    if head isa Symbol
        isdefined(mod, head) && isconst(mod, head) || _pr_fail("unresolved call `$head`")
        return getglobal(mod, head), GlobalRef(mod, head)
    elseif head isa GlobalRef
        isdefined(head.mod, head.name) && isconst(head.mod, head.name) ||
            _pr_fail("unresolved call `$head`")
        return getglobal(head.mod, head.name), head
    elseif Meta.isexpr(head, :., 2) && head.args[2] isa QuoteNode
        parent, _ = _pr_resolve(head.args[1], mod)
        parent isa Module || _pr_fail("call head `$head` is not a module member")
        name = head.args[2].value
        isdefined(parent, name) && isconst(parent, name) || _pr_fail("unresolved call `$head`")
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
    value = _pr_known_value(head, ctx)
    value === nothing || return something(value), head
    try
        return _pr_resolve(head, ctx.mod)
    catch err
        err isa _PRIneligible || rethrow()
        h = head isa Expr ? _pr_flatten_head(head, ctx) : head
        return nothing, h
    end
end
_pr_flatten_head(head, ctx) = head

# Follow only immutable, literal field selections. The actual emitted call
# still reads its receiver at runtime, including when a new binding is used.
function _pr_known_value(ex, ctx)
    ex isa QuoteNode && return Some(ex.value)
    ex isa Symbol && haskey(ctx.values, ex) && return Some(ctx.values[ex])
    ex isa GlobalRef && isdefined(ex.mod, ex.name) && isconst(ex.mod, ex.name) &&
        return Some(getglobal(ex.mod, ex.name))
    if Meta.isexpr(ex, :call, 3) && first(_pr_resolve_lenient(ex.args[1], ctx)) === Core.getfield
        object = _pr_known_value(ex.args[2], ctx)
        return _pr_known_field(object, ex.args[3])
    elseif Meta.isexpr(ex, :ref, 2) && ex.args[2] isa Int
        table = _pr_known_value(ex.args[1], ctx)
        table !== nothing && something(table) isa Tuple &&
            return _pr_known_field(table, ex.args[2])
    end
    nothing
end

function _pr_known_field(object, field)
    object === nothing && return nothing
    value = something(object)
    (field isa Int || field isa QuoteNode && field.value isa Union{Symbol,Int}) || return nothing
    field = field isa QuoteNode ? field.value : field
    if !ismutabletype(typeof(value)) && isdefined(value, field)
        return Some(getfield(value, field))
    end
    nothing
end

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
        # A substituted cell read can copy the entire @inbounds block,
        # including its local name, into multiple operands. Each occurrence
        # returns its own call result. Keep the local assignment for any
        # later explicit use, but feed the fresh call value to this consumer.
        read = _rw_read_parts(ex)
        if read !== nothing
            name, call = read
            _pr_flatten!(stmts, Expr(:(=), name, call), ctx)
            return (last(stmts)::_PRAlias).rhs
        end
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
        rhs isa Symbol && haskey(ctx.values, rhs) && (ctx.values[lhs] = ctx.values[rhs])
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
    elseif Meta.isexpr(ex, :ref, 2) && ex.args[2] isa Int
        table = _pr_known_value(ex.args[1], ctx)
        table !== nothing && something(table) isa Tuple ||
            _pr_fail("unsupported indexed read `$ex`")
        return _pr_flatten!(stmts, Expr(:call, GlobalRef(Core, :getfield), ex.args...), ctx)
    elseif Meta.isexpr(ex, :call)
        f, fref = _pr_resolve_lenient(ex.args[1], ctx)
        # A cell value is a `Float64` here, for which a vector softplus call
        # is the owned rule with its authored partial.
        if f === _vector_rk_log1pexp || f === _vector_log1pexp
            f, fref = _vector_softplus, GlobalRef(@__MODULE__, :_vector_softplus)
        end
        expanded = _pr_expand_stable_helper(f, ex.args[2:end], ctx)
        expanded === nothing || return _pr_flatten!(stmts, expanded, ctx)
        args = Any[_pr_flatten!(stmts, a, ctx) for a in ex.args[2:end]]
        lhs = _pr_tmp!(ctx)
        push!(stmts, _PRCall(lhs, f, fref, args))
        # An immutable closure can carry an owned rule. Resolve its field for
        # call identity only; retain the runtime getfield and its aliases in
        # the emitted program. Mutable records may change after preparation.
        if f === Core.getfield && length(args) == 2
            value = _pr_known_field(_pr_known_value(args[1], ctx), args[2])
            value === nothing || (ctx.values[lhs] = something(value))
        end
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
    (f === Base.exp || f === _native_scalar_exp) && return Any[y]
    f === Base.expm1 && return Any[:($y + 1.0)]
    f === Base.sqrt && return Any[:(0.5 / $y)]
    f === Base.:- && length(args) == 1 && return Any[-1.0]
    f === Base.:+ && return Any[1.0 for _ in args]
    f === Base.:- && length(args) == 2 && return Any[1.0, -1.0]
    f === Base.identity && return Any[1.0]
    f === _native_mul && length(args) == 2 && return Any[args[2], args[1]]
    f === Base.muladd && return Any[args[2], args[1], 1.0]
    f === Base.:/ && length(args) == 2 && return Any[:(1 / $(args[2])), :(-$y / $(args[2]))]
    f === Base.log && length(args) == 1 && return Any[:(inv($(args[1])))]
    f === Base.abs2 && return Any[:(2 * $(args[1]))]
    if f === Base.:^ && length(args) == 2 && args[2] isa Integer
        p = args[2]
        return Any[p == 2 ? :(2 * $(args[1])) : :($p * $(args[1])^$(p - 1)), nothing]
    end
    f === Base.float && return Any[1.0]
    # A Boolean predicate is inactive. Float conversions also arise from a
    # stable form's `_rw_float` and carry the scalar partial unchanged.
    if f === Base.convert && length(args) == 2
        (args[1] === Bool || args[1] == GlobalRef(Core, :Bool)) && return Any[nothing, 0.0]
        return Any[nothing, 1.0]
    end
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

# An argument that selects a value and carries no derivative of its own: the
# condition of `ifelse`. A predicate over active values (`logit > 0`) is such a
# condition, so its dependence on them is no reason to refuse the cell.
_pr_inert_argument(f, k::Int) = f === Base.ifelse && k == 1

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

# Owned scalar rules already provide one graph for their value and partials.
# Record a joint cut only when the reverse sweep reaches the call; an active
# value used solely by a predicate must still execute just its primal. The
# forward sweep later emits this cut at the original (possibly lazy) call site.
function _pr_call_partials(s::_PRCall, active, cuts; cellvars = nothing)
    if !(s.f isa ScalarDerivativeRule)
        p = _pr_partials(s.f, s.lhs, s.args)
        p === nothing || return p
    end
    flags = Tuple(a isa Symbol && a in active for a in s.args)
    call = if s.f isa ScalarDerivativeRule
        Expr(:call, GlobalRef(@__MODULE__, :_derivative_cut_call),
             s.fref, QuoteNode(Val(_activity_mask(flags, 1))), s.args...)
    else
        _pr_source_cut(s.f, s.fref, s.args, flags)
    end
    call === nothing && return nothing
    cut = gensym(:scalar_rule_cut)
    cuts[s.lhs] = (cut, call)
    # A tuple holding cell partials varies with the cell too. Otherwise the
    # factored sweep would treat its fields as shared coefficients.
    cellvars === nothing || push!(cellvars, cut)
    index = 1
    Any[flag ? Expr(:call, GlobalRef(Core, :getfield), cut, index += 1) : nothing
        for flag in flags]
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
function _pr_forward_exprs(stmts, cuts = Dict{Symbol,Tuple{Symbol,Expr}}())
    out = Any[]
    for s in stmts
        if s isa _PRAlias
            push!(out, :($(s.lhs) = $(s.rhs)))
        elseif s isa _PRCall
            if haskey(cuts, s.lhs)
                cut, call = cuts[s.lhs]
                push!(out, :($cut = $call),
                      :($(s.lhs) = $(GlobalRef(Core, :getfield))($cut, 1)))
            else
                push!(out, :($(s.lhs) = $(Expr(:call, s.fref, s.args...))))
            end
        elseif s isa _PRBranch
            then_block = Expr(:block, _pr_forward_exprs(s.then_stmts, cuts)..., :($(s.lhs) = $(s.then_result)))
            # An `if` without an `else` arm has the value `nothing` there.
            else_block = Expr(:block, _pr_forward_exprs(s.else_stmts, cuts)..., :($(s.lhs) = $(s.else_result)))
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

# Carry the binary exponent separately when neither an unscaled product nor
# a quotient coefficient fits. frexp leaves signed zeros and nonfinite
# operands unchanged. Keep this exceptional path out of the SIMD loop body.
const _PRBinaryScalar = Union{Base.IEEEFloat,Base.BitInteger,Bool}
function _pr_product_parts(factors, ::Type{T}) where {T}
    mantissa, power = one(T), 0
    for factor in factors
        m, e = frexp(T(factor))
        mantissa, shift = frexp(mantissa * m)
        power += e + shift
    end
    mantissa, power
end
@noinline function _pr_wide_quotient(factors::Tuple{Vararg{_PRBinaryScalar}}, d::_PRBinaryScalar)
    T = float(promote_type(typeof(d), map(typeof, factors)...))
    v, p = _pr_product_parts(factors, T)
    m, e = frexp(T(d))
    ldexp(v / m, p - e)
end
_pr_wide_quotient(factors, d) = prod(Base.front(factors); init = 1) * (last(factors) / d)

# The same quotients in plain arithmetic, for operands whose products stay in
# the normal range. A product that neither overflows nor loses bits to
# underflow is exact to rounding, and scaling by powers of two changes no
# rounding, so these return the bits of the separate-exponent forms (a
# subnormal quotient is rounded once here and twice there). A zero operand
# gives the same signed zero. Anything else takes the wide form: a product
# that overflowed, one that underflowed from nonzero operands, a value that is
# not finite, or a scalar type other than `Float64`.
@inline _pr_in_range(t::Float64) = floatmin(Float64) <= abs(t) <= floatmax(Float64)
@inline _pr_plain_product(acc::Float64, ok::Bool, ::Tuple{}) = (acc, ok)
@inline function _pr_plain_product(acc::Float64, ok::Bool, factors::Tuple)
    f = Float64(first(factors))
    t = acc * f
    _pr_plain_product(t, ok & (_pr_in_range(t) | (iszero(t) & (iszero(acc) | iszero(f)))),
                      Base.tail(factors))
end
@inline _pr_plain_scalars(::Tuple{}) = true
@inline _pr_plain_scalars(xs::Tuple) =
    first(xs) isa Union{Float64,Base.BitInteger,Bool} && _pr_plain_scalars(Base.tail(xs))
@inline function _pr_quotient(factors::Tuple, d)
    _pr_plain_scalars(factors) && d isa Union{Float64,Base.BitInteger,Bool} ||
        return _pr_wide_quotient(factors, d)
    t, ok = _pr_plain_product(Float64(first(factors)), true, Base.tail(factors))
    ok ? t / Float64(d) : _pr_wide_quotient(factors, d)
end

# Several incoming cotangent terms can cancel. Add their normalized
# products before contracting the quotient partial, even when an individual
# term exceeds the scalar type's exponent range.
@inline function _pr_add_parts(a, b)
    v, p = a
    w, e = b
    iszero(v) && return b
    iszero(w) && return a
    common = max(p, e)
    mantissa, shift = frexp(ldexp(v, p - common) + ldexp(w, e - common))
    mantissa, common + shift
end
@inline _pr_parts_value((v, p)) = iszero(p) ? v : ldexp(v, p)

@noinline function _pr_accumulate_quotient(acc, terms::Tuple{Vararg{Tuple{Vararg{_PRBinaryScalar}}}},
                                           n::_PRBinaryScalar, d::_PRBinaryScalar)
    T = float(promote_type(typeof(n), typeof(d), map(t -> Base.promote_typeof(t...), terms)...))
    value = (zero(T), 0)
    for factors in terms
        value = _pr_add_parts(value, _pr_product_parts(factors, T))
    end
    m, e = frexp(T(d))
    v, p = frexp(T(n))
    partial, shift = frexp(value[1] * v / m)
    _pr_add_parts(acc, (partial, value[2] + p - e + shift))
end
function _pr_accumulate_quotient(acc, terms, n, d)
    (_pr_parts_value(acc) + sum(prod, terms) * (n / d), 0)
end
# The sum of the terms' products, each in plain arithmetic, and whether every
# product and the sum stayed exact to rounding.
@inline _pr_plain_terms(sum::Float64, ok::Bool, ::Tuple{}) = (sum, ok)
@inline function _pr_plain_terms(sum::Float64, ok::Bool, terms::Tuple)
    factors = first(terms)
    _pr_plain_scalars(factors) || return (sum, false)
    t, fine = _pr_plain_product(Float64(first(factors)), true, Base.tail(factors))
    s = sum + t
    _pr_plain_terms(s, ok & fine & isfinite(s), Base.tail(terms))
end
# One row's quotient partial `sum(products) * n / d`, or `nothing` outside
# plain range.
@inline function _pr_plain_partial(terms::Tuple, n, d)
    n isa Union{Float64,Base.BitInteger,Bool} && d isa Union{Float64,Base.BitInteger,Bool} ||
        return nothing
    s, ok = _pr_plain_terms(0.0, true, terms)
    ok || return nothing
    nf = Float64(n)
    t = s * nf
    _pr_in_range(t) | (iszero(t) & (iszero(s) | iszero(nf))) || return nothing
    t / Float64(d)
end
@inline function _pr_sum_quotient(terms, n, d)
    plain = _pr_plain_partial(terms, n, d)
    plain === nothing ? _pr_parts_value(_pr_accumulate_quotient((0.0, 0), terms, n, d)) : plain
end
# A shared cotangent's running sum keeps a zero exponent while every row's
# partial and the sum are plain; the first row outside that range, or a sum
# that already carries an exponent, continues in the separate-exponent form.
@inline function _pr_add_quotient(acc::Tuple{Float64,Int}, terms, n, d)
    if iszero(last(acc))
        plain = _pr_plain_partial(terms, n, d)
        if plain !== nothing && isfinite(plain)
            total = first(acc) + plain
            isfinite(total) && return (total, 0)
        end
    end
    _pr_accumulate_quotient(acc, terms, n, d)
end
@inline _pr_add_quotient(acc, terms, n, d) = _pr_accumulate_quotient(acc, terms, n, d)

function _pr_partial_product(cotangent, partial, quotient)
    if quotient
        _, n, d = partial.args
        n === 1 && return :($cotangent / $d)
        return :($(GlobalRef(@__MODULE__, :_pr_quotient))(($cotangent, $n), $d))
    end
    partial == 1.0 ? cotangent : :($cotangent * $partial)
end

# Reverse sweep over `stmts`; `defined` tracks which adjoints already exist.
# `leaf(s)` may claim a call statement whose adjoint leaves the sweep (a read
# of a batched operand): it returns the expression that sends the complete
# adjoint `bar` on, or `nothing` to differentiate the call as usual.
function _pr_reverse_exprs!(out, stmts, active::Set{Symbol}, defined::Set{Symbol};
                            leaf = s -> nothing, touched = nothing,
                            cuts = Dict{Symbol,Tuple{Symbol,Expr}}())
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
            partials = _pr_call_partials(s, active, cuts)
            partials === nothing && _pr_fail("no derivative for `$(s.fref)`")
            ybar = _pr_bar(s.lhs)
            for (k, (a, p)) in enumerate(zip(s.args, partials))
                a isa Symbol && a in active || continue
                p === nothing && _pr_inert_argument(s.f, k) && continue
                p === nothing && _pr_fail("`$(s.fref)` has no derivative in an active argument")
                _pr_accumulate!(out, defined, a, _pr_partial_product(ybar, p, s.f === Base.:/ && length(s.args) == 2); touched)
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
                _pr_reverse_exprs!(o, stmts_, active, d; leaf, touched, cuts)
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
function _pr_scalar_reverse(ex, active_inputs::Vector{Symbol}, mod::Module; seed::Symbol,
                            values = Dict{Symbol,Any}(), prefix = :_pr_)
    ctx = _PRCtx(mod, 0, values, prefix)
    stmts = Any[]
    result = _pr_flatten!(stmts, ex, ctx)
    active = _pr_activity!(Set{Symbol}(active_inputs), stmts)
    cuts = Dict{Symbol,Tuple{Symbol,Expr}}()
    reverse = Any[]
    defined = Set{Symbol}()
    if result isa Symbol && result in active
        push!(reverse, :($(_pr_bar(result)) = $seed))
        push!(defined, _pr_bar(result))
        _pr_reverse_exprs!(reverse, stmts, active, defined; cuts)
    end
    forward = _pr_forward_exprs(stmts, cuts)
    bars = Dict{Symbol,Symbol}(a => _pr_bar(a) for a in active_inputs if _pr_bar(a) in defined)
    (; forward, reverse, result, bars)
end

# Source records and generated leaf arms already carry their executable
# expression. Differentiate that expression once inside a scalar call, with
# its own argument names and scope. Keep tables and captures as arguments:
# two records with the same code may hold different bound data.
_pr_source_cut(@nospecialize(f), ref, args, flags) = nothing

include("plate_reverse_sources.jl")

_pr_field(ref, field) = Expr(:call, GlobalRef(Core, :getfield), ref, QuoteNode(field))
Base.@nospecializeinfer _pr_source_cut(@nospecialize(f::_KernelSourceOp), ref, args, flags) =
    _pr_source_cut(f.f, _pr_field(ref, :f), args, flags)
Base.@nospecializeinfer _pr_source_cut(@nospecialize(f::Union{_KernelBranch,_KernelSplit}), ref, args, flags) =
    _pr_source_cut(f.call, _pr_field(ref, :call), args, flags)

Base.@nospecializeinfer function _pr_source_cut(@nospecialize(f::_KernelSource), ref, args, flags)
    f.native === nothing || return nothing
    program = _native_generated_function(f.companion)
    _pr_program_cut(program, args, flags, ((f.captures, _pr_field(ref, :captures)),))
end

Base.@nospecializeinfer _pr_source_cut(@nospecialize(f::RuntimeGeneratedFunctions.RuntimeGeneratedFunction), ref, args, flags) =
    _pr_program_cut(f, args, flags, ())

Base.@nospecializeinfer function _pr_source_cut(@nospecialize(f::Union{_KernelBranchCall,_KernelSplitApply}), ref, args, flags)
    record = gensym(:record)
    params = [gensym(:arg) for _ in args]
    body = if f isa _KernelBranchCall
        _native_plate_branch_expr(f, record, params)
    else
        plan = typeof(f).parameters[1]
        _split_apply_body(plan, _kernel_split_spec(f.split), _pr_field(record, :split),
                          i -> params[i], identity)
    end
    ex = Expr(:->, Expr(:tuple, record, params...), body)
    _pr_expression_cut(ex, @__MODULE__, args, flags, ((f, ref),))
end

Base.@nospecializeinfer function _pr_source_cut(@nospecialize(child::_KernelPreparedChild), ref, args, flags)
    k = child.kernel
    kernel_ref = _pr_field(ref, :kernel)
    tables = ((k.constants, _pr_field(kernel_ref, :constants)),
              (k.callables, _pr_field(kernel_ref, :callables)))
    program = k.f isa _ArrayFunction ? k.f.native : k.f
    _pr_program_cut(_native_generated_function(program), args, flags, tables)
end

# Only a final return is a value expression. An early return keeps the
# original ordinary reverse (the scalar sweep has no nonlocal control flow).
function _pr_tail_value(ex)
    Meta.isexpr(ex, :return, 1) && return only(ex.args)
    if Meta.isexpr(ex, :block)
        xs = Any[x for x in ex.args if !(x isa LineNumberNode)]
        isempty(xs) || (xs[end] = _pr_tail_value(xs[end]))
        return Expr(:block, xs...)
    end
    ex
end

# Each branch exports its value, and each local in the scalar sweep has one
# primal and adjoint. Keep let bindings distinct, preserve conversions, and
# require a definition on the path where the local is read.
function _pr_source_body(ex, bound, types, mod; binding = false)
    ex isa Symbol && return get(() -> _pr_fail("unbound scalar source local `$ex`"), bound, ex)
    ex isa Expr || return ex
    ex.head in (:quote, :inert, :meta, :inbounds, :loopinfo) && return ex
    if Meta.isexpr(ex, :(=), 2)
        lhs, rhs = ex.args
        value = _pr_source_body(rhs, bound, types, mod)
        name = _pr_param_name(lhs)
        name isa Symbol || _pr_fail("mutation in scalar source")
        if binding
            delete!(types, name)
        elseif haskey(bound, name)
            _pr_fail("reassignment to `$name` in scalar source")
        end
        Meta.isexpr(lhs, :(::), 2) && (types[name] = lhs.args[2])
        if haskey(types, name)
            T, _ = _pr_resolve(types[name], mod)
            T in (Bool, Float16, Float32, Float64) || _pr_fail("scalar source converts to `$T`")
            value = Expr(:call, GlobalRef(Base, :convert), GlobalRef(Core, nameof(T)), value)
        end
        target = bound[name] = gensym(:source_value)
        return Expr(:(=), target, value)
    elseif Meta.isexpr(ex, :let, 2)
        locals, declared = copy(bound), copy(types)
        header, body = ex.args
        bindings = Meta.isexpr(header, :block) ? header.args : (header,)
        values = Any[_pr_source_body(b, locals, declared, mod; binding = true) for b in bindings]
        return Expr(:let, Expr(:block, values...), _pr_source_body(body, locals, declared, mod))
    elseif ex.head in (:if, :&&, :||)
        cond = _pr_source_body(ex.args[1], bound, types, mod)
        return Expr(ex.head, cond,
            (_pr_source_body(a, copy(bound), copy(types), mod) for a in ex.args[2:end])...)
    elseif ex.head === :local
        values = Any[]
        for a in ex.args
            if Meta.isexpr(a, :(=), 2)
                push!(values, _pr_source_body(a, bound, types, mod))
            else
                name = _pr_param_name(a)
                name isa Symbol || _pr_fail("unsupported declaration in scalar source: `$a`")
                haskey(bound, name) && _pr_fail("redeclaration of scalar source local `$name`")
                Meta.isexpr(a, :(::), 2) && (types[name] = a.args[2])
            end
        end
        return Expr(:block, values...)
    end
    Expr(ex.head, Any[_pr_source_body(a, bound, types, mod) for a in ex.args]...)
end

Base.@nospecializeinfer function _pr_program_cut(@nospecialize(program), args, flags, @nospecialize(known))
    program isa RuntimeGeneratedFunctions.RuntimeGeneratedFunction || return nothing
    ex = _program_expression(program)
    mod = parentmodule(typeof(program).parameters[3])
    _pr_expression_cut(ex, mod, args, flags, known)
end

Base.@nospecializeinfer function _pr_expression_cut(ex, mod, args, flags, @nospecialize(known))
    ex isa Expr && ex.head in (:->, :function) && Meta.isexpr(ex.args[1], :tuple) || return nothing
    # Use Julia's lexical bindings: a let can shadow an argument, and a
    # free name belongs to the program's defining module.
    ex = macroexpand(mod, Expr(Symbol("hygienic-scope"), ex, mod))
    params = ex.args[1].args
    length(params) == length(known) + length(args) || return nothing
    all(p -> _pr_param_name(p) isa Symbol, params) || return nothing
    names = Symbol[_pr_param_name(p) for p in params]
    body = _pr_tail_value(ex.args[2])
    _inline_contains_return(body) && return nothing
    body = _pr_source_body(body, Dict(n => n for n in names), Dict{Symbol,Any}(), mod)
    values = Dict{Symbol,Any}(n => v for (n,(v,_)) in zip(names, known))
    inputs = names[length(known)+1:end][collect(flags)]
    seed = gensym(:seed)
    sweep = _pr_scalar_reverse(body, inputs, mod; seed, values, prefix = gensym(:source))
    result = Expr(:tuple, sweep.result, (get(sweep.bars, n, 0.0) for n in inputs)...)
    code = Expr(:block, :($seed = 1.0), sweep.forward..., sweep.reverse..., result)
    actuals = Any[(ref for (_, ref) in known)..., args...]
    bindings = map(params, names, actuals) do p, name, value
        rhs = p isa Symbol ? value : Expr(:(::), value, p.args[2])
        Expr(:(=), name, rhs)
    end
    Expr(:let, Expr(:block, bindings...), code)
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
        if f isa _PlateProgram
            # A pointwise plate can return nothing and fill an operand. Use
            # the program's actual bindings to trace writes into that array.
            _pr_dataflow!(out, _inline_plate_program(ex, Set{Symbol}()))
        end
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

struct _PRAdditiveGroup
    prefix::Vector{Any}
    finish::Vector{Any}
    index::Symbol
    locals::Vector{Symbol}
end

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
    reads::Dict{Symbol,Int}           # batched parameter => its reads in the program
    arm_initializers::Dict{Symbol,Vector{Pair{Symbol,Any}}}
    joint_cell_cut::Bool
    shared_cell_work::Bool
    groups::IdDict{Expr,_PRAdditiveGroup}
    group_shadows::IdDict{Expr,Expr}
    group_scatters::IdDict{Expr,Expr}
    bound::Vector{Symbol}             # parameters that read the constants table
    sited::Bool                       # the reverse reads its call's site record
end

# How many reads of each batched parameter one call of the program can make,
# through any of its prepared operands: read sites in sequence add up (another
# cell loop, another alias in a cell), the arms of a branch take the larger
# (the native direct and extruded copies of a cell loop). Under one pass a read
# may add unscaled partials into the shadow and scale it once after its loop
# (`_pr_read_reverse!`); that is only right when no other read adds into the
# same shadow.
function _pr_read_counts(body, prepared, mod, values)
    function count(ex)
        counts = Dict{Symbol,Int}()
        ex isa Expr && ex.head !== :quote || return counts
        if ex.head in (:if, :elseif) && length(ex.args) >= 2
            mergewith!(+, counts, count(ex.args[1]))
            arms = Dict{Symbol,Int}()
            for arm in ex.args[2:end]
                mergewith!(max, arms, count(arm))
            end
            return mergewith!(+, counts, arms)
        end
        if Meta.isexpr(ex, :call) && length(ex.args) >= 2 && ex.args[2] isa Symbol &&
                haskey(prepared, ex.args[2])
            f = _pr_call_function(ex.args[1], mod, values)
            if f === Base.Broadcast._broadcast_getindex || f === _plate_read
                p = prepared[ex.args[2]]
                counts[p] = get(counts, p, 0) + 1
            end
        end
        for a in ex.args
            mergewith!(+, counts, count(a))
        end
        counts
    end
    count(body)
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
function _pr_direct_operands!(prepared, body, params, active_params, broadcastables, mod, values, groups)
    operands = Dict{Symbol,Symbol}(p => p for p in params)
    merge!(operands, broadcastables)
    reads = IdDict{Expr,Nothing}()
    function read(ex, indices)
        ex isa Expr && !(ex.head in (:quote, :for, :while)) || return
        if Meta.isexpr(ex, :call, 3) && ex.args[2] isa Symbol && ex.args[3] in indices
            operand = ex.args[2]
            f = _pr_call_function(ex.args[1], mod, values)
            if f === _plate_read && haskey(operands, operand)
                prepared[operand] = operands[operand]
                reads[ex] = nothing
            elseif f === Base.Broadcast._broadcast_getindex && haskey(prepared, operand)
                reads[ex] = nothing
            end
        end
        foreach(arg -> read(arg, indices), ex.args)
    end
    function visit(node)
        node isa Expr && node.head !== :quote || return
        group = get(groups, node, nothing)
        if group !== nothing && !(group.index in group.locals)
            # Each outer read and its scatter use this group's coordinate.
            # Reads in nested cell loops must establish their own coordinate.
            for stmt in group.prefix
                _pr_has_loop(stmt) || read(stmt, (group.index,))
            end
        end
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
        foreach(stmt -> read(stmt, (index, linear)), statements[2:end])
    end
    visit(body)
    function check(node)
        node isa Expr && node.head !== :quote || return
        if Meta.isexpr(node, :call, 3) && !haskey(reads, node)
            f = _pr_call_function(node.args[1], mod, values)
            if f === _plate_read && get(operands, node.args[2], nothing) in active_params
                _pr_fail("a direct plate read does not use its cell index")
            elseif f === Base.Broadcast._broadcast_getindex &&
                    get(prepared, node.args[2], nothing) in active_params
                _pr_fail("a prepared plate read does not use its cell index")
            end
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
    readof::Dict{Symbol,Symbol}  # a gathered read's value (or an alias of it) => the read
    deferred::Dict{Symbol,Vector{Any}}  # read => (shared value, num, inv, other factor) products
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

# A bounded quotient can leave its common coefficient outside the cell
# reduction. Otherwise contract all the factors with the original divisor
# in the cell, so an intermediate product need not be representable.
function _pr_quotient_products(terms)
    map(terms) do term
        factors = Any[term.num, term.inv...]
        term.var === nothing || push!(factors, term.var)
        Expr(:tuple, factors...)
    end
end
function _pr_factored_quotient!(out, fc, terms, partial, toplevel, q)
    _, num, d = partial.args
    if length(terms) != 1
        products = _pr_quotient_products(terms)
        value = :($(GlobalRef(@__MODULE__, :_pr_sum_quotient))(($(products...),), $num, $d))
        return _PRTerm(1.0, Any[], _pr_bind!(out, fc, value))
    end
    t = only(terms)
    negative = _pr_is_neg(num)
    n = negative ? num.args[2] : num
    shared = Any[t.num, t.inv...]
    factors = copy(shared)
    v = t.var
    v === nothing || push!(factors, v)
    n === 1 || push!(factors, n)
    value = :($(GlobalRef(@__MODULE__, :_pr_quotient))(($(factors...),), $d))
    if toplevel && length(shared) <= 3 && all(x -> x isa Union{Symbol,Number}, shared) &&
            _pr_is_invariant(fc, d, true)
        # A factor that is itself a bounded quotient (`z = x / σ`, computed
        # as `x * r`) enters the cell's product as its numerator, and its
        # reciprocal joins the coefficient: the sweep then sums and scatters
        # `x` as a hand-written loop sums the unscaled residual, and `z` is
        # not formed for the reverse. The range check still charges `z`
        # squared, which covers the reciprocals the coefficient now carries
        # (`_pr_targets_fit`).
        recips = Symbol[]
        numerator(x) = if x isa Symbol && haskey(q.fast, x)
            numer, divisor = q.fast[x]
            push!(recips, q.inverses[divisor])
            numer
        else
            x
        end
        fv = v === nothing ? nothing : numerator(v)
        fn = n === 1 ? 1 : numerator(n)
        scale = get!(fc.coefs, (:quotient, Tuple(shared), d, Tuple(recips))) do
            k = _pr_fresh!(fc, :ratio)
            ratio = :(Base.prod(($(shared...),)) / $d)
            for r in recips
                ratio = :($ratio * $r)
            end
            push!(fc.pre, :($k = $(q.normal) ? $ratio : 1.0))
            k
        end
        fast = fv === nothing ? fn : fn === 1 ? fv : :($fv * $fn)
        v === nothing || push!(q.targets, v)
        n === 1 || push!(q.targets, n)
        append!(q.shared, shared)
        push!(q.divisors, d)
        varying = _pr_bind!(out, fc, Expr(:if, q.normal, fast, value))
        return _PRTerm(negative ? -1.0 : 1.0, Any[scale], varying)
    end
    _PRTerm(negative ? -1.0 : 1.0, Any[], _pr_bind!(out, fc, value))
end

function _pr_quotient_contribution!(out, fc, adj, mutable, a, terms, p, toplevel, q)
    if !(a in fc.shared)
        scaled = _pr_factored_quotient!(out, fc, terms, p, toplevel, q)
        _pr_contribute!(out, fc, adj, mutable, a, [scaled], toplevel)
        return
    end
    normal = Any[]
    scaled = _pr_factored_quotient!(normal, fc, terms, p, toplevel, q)
    # A shared cotangent can be finite after cancellation even
    # when individual rows exceed the destination's exponent range.
    state = get!(q.sums, a) do
        temp = _pr_fresh!(fc, :quotient_sum)
        push!(fc.pre, :($temp = (0.0, 0)))
        push!(fc.post, :(if !iszero(first($temp))
            $(_pr_bar(a)) += $(GlobalRef(@__MODULE__, :_pr_parts_value))($temp)
        end))
        temp
    end
    push!(fc.st.touched, a)
    _, n, d = p.args
    products = _pr_quotient_products(terms)
    wide = :($state = $(GlobalRef(@__MODULE__, :_pr_add_quotient))(
        $state, ($(products...),), $n, $d))
    if isempty(scaled.inv)
        push!(out, wide)
    else
        _pr_contribute!(normal, fc, adj, mutable, a, [scaled], toplevel)
        push!(out, Expr(:if, q.normal, Expr(:block, normal...), wide))
    end
end

# The factored reverse sweep of `stmts`.
function _pr_factored_sweep!(out, fc::_PRFactored, stmts, adj, mutable, toplevel::Bool, cuts, q)
    st = fc.st; active = fc.active
    for s in Iterators.reverse(stmts)
        if s isa _PRAlias
            s.rhs isa Symbol && s.rhs in active || continue
            terms = _pr_adjoint_terms!(out, fc, adj, mutable, s.lhs)
            terms === nothing && continue
            _pr_contribute!(out, fc, adj, mutable, s.rhs, terms, toplevel)
        elseif s isa _PRCall
            s.lhs in active || continue
            terms = if s.f === Base.:/ && length(s.args) == 2 && !haskey(mutable, s.lhs)
                get(adj, s.lhs, nothing)
            else
                _pr_adjoint_terms!(out, fc, adj, mutable, s.lhs)
            end
            if _pr_is_read(s, st)
                terms === nothing && !haskey(fc.deferred, s.lhs) && continue
                _pr_read_reverse!(out, fc, adj, mutable, s, terms, toplevel)
                continue
            end
            terms === nothing && continue
            partials = _pr_call_partials(s, active, cuts; cellvars = fc.cellvars)
            partials === nothing && _pr_fail("no derivative for `$(s.fref)`")
            # Ordinary AD already has the joint rule for an owned scalar.
            # A source cut, or an exponential reused by its slope, can save
            # work across the two sweeps even with a full-size row buffer.
            st.shared_cell_work |= s.f === exp || s.f === expm1 ||
                (haskey(cuts, s.lhs) && !(s.f isa ScalarDerivativeRule))
            for (k, (a, p)) in enumerate(zip(s.args, partials))
                a isa Symbol && a in active || continue
                p === nothing && _pr_inert_argument(s.f, k) && continue
                p === nothing && _pr_fail("`$(s.fref)` has no derivative in an active argument")
                if s.f === Base.:/ && length(s.args) == 2
                    _pr_quotient_contribution!(out, fc, adj, mutable, a, terms, p, toplevel, q)
                    continue
                end
                # An arm of `ifelse` takes the cotangent where the condition
                # selects it and zero elsewhere: a select, not a product with 1
                # or 0, so the two arms' contributions fold (`c ? -d : d` for
                # `ifelse(c, -x, x)`), and a cell cotangent that is not finite
                # stays out of the arm not taken. A coefficient that is not
                # finite (the seed) still multiplies both.
                if s.f === Base.ifelse && k in (2, 3) && !_pr_is_invariant(fc, s.args[1], toplevel)
                    c = s.args[1]
                    pick(v) = k == 2 ? :($(GlobalRef(Base, :ifelse))($c, $v, 0.0)) :
                                       :($(GlobalRef(Base, :ifelse))($c, 0.0, $v))
                    new = _PRTerm[_PRTerm(t.num, t.inv,
                        _pr_bind!(out, fc, pick(t.var === nothing ? 1.0 : t.var))) for t in terms]
                    _pr_contribute!(out, fc, adj, mutable, a, new, toplevel)
                    continue
                end
                num, inv, vars = _pr_split_partial(fc, p, toplevel)
                # A shared value's partial that is a gathered read: its sum over
                # the cells can run per group, through the read's own scatter
                # (see the read below), so it waits for the read.
                if toplevel && a in fc.shared && length(vars) == 1 && vars[1] isa Symbol &&
                        haskey(fc.readof, vars[1]) &&
                        all(t -> t.var === nothing || t.var isa Symbol, terms)
                    r = fc.readof[vars[1]]
                    for t in terms
                        push!(get!(fc.deferred, r, Any[]),
                              (a, t.num * num, vcat(t.inv, inv), t.var, vars[1]))
                    end
                    continue
                end
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
                _pr_factored_sweep!(o, fc, stmts_, a2, mutable, false, cuts, q)
                Expr(:block, o...)
            end
            push!(out, Expr(:if, s.cond, arm(s.then_stmts, s.then_result),
                            arm(s.else_stmts, s.else_result)))
        end
    end
    out
end

# The reads of batched operands that the cell reads once, at its top level, and
# every alias of their values: `alias => read`. The program must read the
# operand nowhere else either: another loop, or another prepared alias of the
# same parameter, adds into the same shadow.
function _pr_single_reads(stmts, st::_PRProgramState, active)
    counts = Dict{Symbol,Int}()
    function count!(ss)
        for s in ss
            _pr_is_read(s, st) && (counts[s.args[1]] = get(counts, s.args[1], 0) + 1)
            s isa _PRBranch && (count!(s.then_stmts); count!(s.else_stmts))
        end
    end
    count!(stmts)
    readof = Dict{Symbol,Symbol}()
    for s in stmts
        if _pr_is_read(s, st) && counts[s.args[1]] == 1 && st.prepared[s.args[1]] in active &&
                haskey(st.shadow, st.prepared[s.args[1]]) &&
                get(st.reads, st.prepared[s.args[1]], 0) == 1
            readof[s.lhs] = s.lhs
        elseif s isa _PRAlias && s.rhs isa Symbol && haskey(readof, s.rhs)
            readof[s.lhs] = readof[s.rhs]
        end
    end
    readof
end

# The reverse of a read of a batched operand: its cotangent added into the
# operand's shadow, cell by cell. Under one pass (`_PR_WANT_VALUE`, where the
# shadow is the plate's own zeroed buffer, `view(buffer, g)` for a gathered
# operand) a read whose cotangent is one term `k · v` adds `v` alone and scales
# the buffer by `k` once, after the loop; a shared value's partial `c · u · read`
# with `u == v` (`sd` in `sd * z[g]`) is then `c · Σⱼ zⱼ · bufferⱼ`, one product
# per group instead of one per cell.
function _pr_read_reverse!(out, fc::_PRFactored, adj, mutable, s, terms, toplevel)
    st = fc.st
    shadow = st.shadow[st.prepared[s.args[1]]]
    prepared = s.args[1]
    deferred = toplevel ? pop!(fc.deferred, s.lhs, Any[]) : Any[]
    terms = terms === nothing ? _PRTerm[] : terms
    factorable = toplevel && haskey(fc.readof, s.lhs) && length(terms) <= 1 &&
        !(isempty(terms) && isempty(deferred))
    u = nothing
    if factorable
        u = isempty(terms) ? deferred[1][4] : only(terms).var
        factorable = all(d -> d[4] == u, deferred)
    end
    # The per-cell form: the deferred partials as products, the read's cotangent scaled.
    plain = Any[]
    for (a, num, inv, u, value) in deferred
        pv = u === nothing ? value : _pr_bind!(plain, fc, :($u * $value))
        _pr_contribute!(plain, fc, adj, mutable, a, [_PRTerm(num, inv, pv)], toplevel)
    end
    if !isempty(terms)
        v = _pr_materialize!(plain, fc, terms)
        cot = :($(GlobalRef(@__MODULE__, :_pr_read_cotangent))($shadow, $(s.lhs), $v))
        push!(plain, :($(GlobalRef(@__MODULE__, :_pr_shadow_add!))($shadow, $(fc.linear), $cot)))
    end
    if !factorable
        append!(out, plain)
        return
    end
    one_pass = :($_PR_WANT_VALUE isa Val{true} && !($shadow isa Nothing))
    push!(out, Expr(:if, one_pass,
        :($(GlobalRef(@__MODULE__, :_pr_shadow_add!))($shadow, $(fc.linear), $(u === nothing ? 1.0 : u))),
        Expr(:block, plain...)))
    post = Any[]
    for (a, num, inv, _, _) in deferred
        push!(fc.st.touched, a)
        push!(post, :($(_pr_bar(a)) = $(_pr_bar(a)) + $(GlobalRef(@__MODULE__, :_pr_scaled))(
            $(_pr_coef!(fc, num, inv)),
            $(GlobalRef(@__MODULE__, :_pr_gather_dot))($prepared, $shadow))))
    end
    k = isempty(terms) ? 0.0 : _pr_coef!(fc, only(terms).num, only(terms).inv)
    push!(post, :($(GlobalRef(@__MODULE__, :_pr_scale_buffer!))($shadow, $k)))
    push!(fc.post, Expr(:if, one_pass, Expr(:block, post...)))
end

# Σ over the operand's storage of value × buffer: for a gathered operand
# (`view(z, g)`, prepared as broadcast extrusion) its parent's length, not the
# cells'. The buffer is a view of the same index vector (the Enzyme extension
# builds it from the operand's own indices).
@inline _pr_gather_dot(A::Base.Broadcast.Extruded, buffer) = _pr_gather_dot(A.x, buffer)
function _pr_gather_dot(x::SubArray, buffer::SubArray)
    p = _pr_gather_root(x); q = _pr_gather_root(buffer)
    acc = zero(eltype(p)) * zero(eltype(q))
    @inbounds for j in eachindex(p, q)
        acc += p[j] * q[j]
    end
    acc
end
# A buffer that is its own storage (an `Array` the caller owns) is indexed as
# the operand is: the cell's index, or the single entry of a broadcast operand.
function _pr_gather_dot(x::AbstractArray, buffer::Array)
    acc = zero(eltype(x)) * zero(eltype(buffer))
    @inbounds for j in eachindex(x, buffer)
        acc += x[j] * buffer[j]
    end
    acc
end
_pr_gather_dot(x, buffer) = throw(ArgumentError(
    "plate reverse: a one-pass product through a read of $(typeof(x)) with a buffer $(typeof(buffer))"))
# The plate's own buffer scaled in place, once (`_pr_scaled` per entry, so an
# entry no cell reached stays zero under a coefficient that is not finite): a
# gathered buffer through its parent, each entry once however often the index
# vector repeats it.
_pr_scale_buffer!(::Nothing, k) = nothing
_pr_scale_buffer!(buffer::SubArray, k) = _pr_scale_owned!(_pr_gather_root(buffer), k)
_pr_scale_buffer!(buffer::Array, k) = _pr_scale_owned!(buffer, k)
function _pr_scale_owned!(q, k)
    @inbounds for j in eachindex(q)
        q[j] = _pr_scaled(k, q[j])
    end
    nothing
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
@inline _pr_direct_read(A::Base.Broadcast.Extruded, i) = Base.Broadcast._broadcast_getindex(A, i)
@inline _pr_direct_read(A::Base.Broadcast.Extruded, i::Int) = @inbounds A.x[i]
@inline function _pr_direct_read(A::Base.Broadcast.Extruded, i::CartesianIndex{N}) where {N}
    length(A.keeps) == N || return Base.Broadcast._broadcast_getindex(A, i)
    @inbounds A.x[i]
end
@inline _pr_direct_read(x::Number, i) = x
@inline _pr_direct_read(r::Base.RefValue, i) = r[]

# The prepared operands the statements read, and the statements with each read
# of them direct.
function _pr_direct_reads(stmts, st::_PRProgramState, found::Vector{Symbol} = Symbol[])
    out = Any[]
    for s in stmts
        if _pr_is_read(s, st)
            s.args[1] in found || push!(found, s.args[1])
            push!(out, _PRCall(s.lhs, _pr_direct_read, GlobalRef(@__MODULE__, :_pr_direct_read),
                               s.args))
        elseif s isa _PRBranch
            push!(out, _PRBranch(s.lhs, s.cond, _pr_direct_reads(s.then_stmts, st, found)[1],
                                 s.then_result, _pr_direct_reads(s.else_stmts, st, found)[1],
                                 s.else_result))
        else
            push!(out, s)
        end
    end
    out, found
end

# The sweep's layout parameter also controls the value reduction. Grouped
# tapes factor it, logical tapes accumulate in cell order, and the two-pass
# schedule compiles the reduction away.
const _PR_WANT_VALUE = :_pr_want_value
# The site record of the rule call a reverse runs for (`_PlateRuleSite`, or
# `nothing`): the last parameter of a reverse that keeps a bound operand's
# measured exponents there (`_pr_cached_exponents`), and of no other.
const _PR_SITE = :_pr_site

# The cell's value `result` as affine terms `(num, inv, var)` over the cell's
# top-level statements: `num · ∏ inv · var`, `inv` invariant atoms and `var` a
# cell-varying symbol or `nothing`. A statement that is not a sum, difference,
# negation or product by invariants is a varying term of its own, so the worst
# case is the density itself.
function _pr_value_terms(stmts, invariant::Set{Symbol}, cellvars::Set{Symbol}, result)
    defs = Dict{Symbol,Any}(s.lhs => s for s in stmts if s isa Union{_PRAlias,_PRCall})
    isinv(x) = (x isa Real && !(x isa Bool)) ||
               (x isa Symbol && (!(x in cellvars) || x in invariant))
    terms = Tuple{Float64,Vector{Any},Any}[]
    function walk(x, num::Float64, inv::Vector{Any})
        if x isa Real && !(x isa Bool)
            push!(terms, (num * Float64(x), inv, nothing))
        elseif isinv(x)
            push!(terms, (num, vcat(inv, Any[x]), nothing))
        elseif x isa Symbol && haskey(defs, x)
            d = defs[x]
            args = d isa _PRCall ? d.args : Any[]
            simple = all(a -> a isa Symbol || (a isa Real && !(a isa Bool)), args)
            if d isa _PRAlias
                d.rhs isa Symbol || (d.rhs isa Real && !(d.rhs isa Bool)) ?
                    walk(d.rhs, num, inv) : push!(terms, (num, inv, x))
            elseif simple && d.f === Base.:+
                foreach(a -> walk(a, num, inv), args)
            elseif simple && d.f === Base.:- && length(args) == 2
                walk(args[1], num, inv); walk(args[2], -num, inv)
            elseif simple && d.f === Base.:- && length(args) == 1
                walk(args[1], -num, inv)
            elseif simple && (d.f === Base.:* || d.f === _native_mul) && length(args) == 2 &&
                    count(a -> !isinv(a), args) == 1
                k = findfirst(a -> !isinv(a), args)
                other = args[3 - k]
                other isa Real ? walk(args[k], num * Float64(other), inv) :
                                 walk(args[k], num, vcat(inv, Any[other]))
            else
                push!(terms, (num, inv, x))
            end
        else
            push!(terms, (num, inv, x))
        end
    end
    walk(result, 1.0, Any[])
    terms
end

# A canonical form of a cell value, for telling whether two sums add the same
# term: aliases resolved, a product as an unordered pair, a square as a
# product. `bound` holds the bindings of the block being read (`t = a * b`),
# and a value that the bounded and the exceptional sweep compute differently
# (`if normal; a; else; b; end`) is read as the bounded sweep's. A quotient by
# an invariant divisor (`fast`, the bounded sweep's `x * r`) is its
# numerator's form over that divisor, `(:/, form, divisors)`, with the
# divisors of a product collected outside it. Anything else compares by name,
# and an expression that is not a product equals nothing.
function _pr_summand_key(x, defs::Dict{Symbol,Any}, bound::Dict{Symbol,Any}, normal = nothing,
                         fast = nothing)
    key(y) = _pr_summand_key(y, defs, bound, normal, fast)
    x isa Real && !(x isa Bool) && return Float64(x)
    if x isa Symbol
        haskey(bound, x) && return key(bound[x])
        d = get(defs, x, nothing)
        d isa _PRAlias && return key(d.rhs)
        if d isa _PRCall && length(d.args) == 2
            (d.f === Base.:* || d.f === _native_mul) &&
                return _pr_product_key(key(d.args[1]), key(d.args[2]))
            if d.f === Base.:^ && d.args[2] === 2
                k = key(d.args[1])
                return _pr_product_key(k, k)
            end
            if d.f === Base.:/ && fast !== nothing && haskey(fast, x)
                numer, divisor = fast[x]
                base, divisors = _pr_key_parts(key(numer))
                return (:/, base, (divisors..., divisor))
            end
        end
        return x
    end
    normal !== nothing && Meta.isexpr(x, :if, 3) && x.args[1] === normal && return key(x.args[2])
    if Meta.isexpr(x, :call, 3)
        f = x.args[1]
        f isa GlobalRef && isdefined(f.mod, f.name) && (f = getglobal(f.mod, f.name))
        (f === :* || f === Base.:* || f === _native_mul) &&
            return _pr_product_key(key(x.args[2]), key(x.args[3]))
    end
    gensym(:unshared)
end
function _pr_product_key(a, b)
    (ba, da), (bb, db) = _pr_key_parts(a), _pr_key_parts(b)
    base = repr(ba) <= repr(bb) ? (:*, ba, bb) : (:*, bb, ba)
    isempty(da) && isempty(db) && return base
    (:/, base, Tuple(sort!(Any[da..., db...]; by = repr)))
end
_pr_key_parts(k) = k isa Tuple && length(k) == 3 && k[1] === :/ ? (k[2], k[3]) : (k, ())

# The partial sums that the bounded arm of a quotient contribution adds once
# per cell (`if normal; t = …; S = S + t; else … end`, `_pr_quotient_contribution!`),
# by the canonical form of their summand. `sums` names the sweep's partial
# sums. A sum the cell adds to more than once is no single term's sum.
function _pr_normal_sums(stmts, reverse, normal, sums, fast = nothing)
    defs = Dict{Symbol,Any}(s.lhs => s for s in stmts if s isa Union{_PRAlias,_PRCall})
    updates = Dict{Symbol,Int}()
    function count_updates(node)
        node isa Expr || return
        if Meta.isexpr(node, :(=), 2) && node.args[1] isa Symbol && node.args[1] in sums
            updates[node.args[1]] = get(updates, node.args[1], 0) + 1
        end
        foreach(count_updates, node.args)
    end
    foreach(count_updates, reverse)
    found = Dict{Any,Symbol}()
    for node in reverse
        Meta.isexpr(node, :if, 3) && node.args[1] === normal &&
            Meta.isexpr(node.args[2], :block) || continue
        bound = Dict{Symbol,Any}()
        for stmt in node.args[2].args
            Meta.isexpr(stmt, :(=), 2) && stmt.args[1] isa Symbol || continue
            lhs, rhs = stmt.args
            if lhs in sums
                get(updates, lhs, 0) == 1 && Meta.isexpr(rhs, :call, 3) && rhs.args[1] === :+ &&
                    rhs.args[2] === lhs || continue
                get!(found, _pr_summand_key(rhs.args[3], defs, bound, normal, fast), lhs)
            else
                bound[lhs] = rhs
            end
        end
    end
    defs, found
end

# The statements that add the cell values into `total`: above the loop, in it
# and after it, each only when the sweep wants the value. Where the bounded
# sweep (`normal`) already keeps a partial sum of the same term for a shared
# adjoint, such as the squared residual of a Normal cell for its scale, the
# value reads that sum and adds its own only in the exceptional sweep. A value
# term that is that sum's summand over invariant divisors (the squared
# standardized residual, where the sweep sums the squared residual) reads the
# sum times their reciprocals (`quotients`, the sweep's `q`).
function _pr_value_accumulation(stmts, invariant, cellvars, result, total, add, ncells,
                                st::_PRProgramState; reverse = Any[], normal = nothing,
                                sums = Set{Symbol}(), quotients = nothing)
    want = :($_PR_WANT_VALUE isa Val{true})
    fast = quotients === nothing ? nothing : quotients.fast
    defs, shared = normal === nothing ? (Dict{Symbol,Any}(), Dict{Any,Symbol}()) :
        _pr_normal_sums(stmts, reverse, normal, sums, fast)
    function twin_of(var)
        isempty(shared) && return nothing
        k = _pr_summand_key(var, defs, Dict{Symbol,Any}(), nothing, fast)
        haskey(shared, k) && return shared[k]
        base, divisors = _pr_key_parts(k)
        (isempty(divisors) || !haskey(shared, base)) && return nothing
        Expr(:call, GlobalRef(Base, :*), shared[base], (quotients.inverses[d] for d in divisors)...)
    end
    coef(num, inv) = isempty(inv) ? num :
        num == 1.0 ? Expr(:call, GlobalRef(Base, :*), inv...) :
        Expr(:call, GlobalRef(Base, :*), num, inv...)
    pre = Any[]; parts = Any[]; constant = Any[]
    inloop = Any[:($_PR_WANT_VALUE isa Val{:logical} && ($total = $add($total, $result)))]
    for (num, inv, var) in _pr_value_terms(stmts, invariant, cellvars, result)
        if var === nothing
            push!(constant, coef(num, inv))
        else
            st.counter += 1
            s = Symbol(:_pr_value, st.counter)
            push!(pre, :($s = 0.0))
            update = :($want && ($s = $s + $var))
            twin = twin_of(var)
            if twin === nothing
                push!(inloop, update)
                push!(parts, :($(coef(num, inv)) * $s))
            else
                push!(inloop, Expr(:if, normal, Expr(:block), update))
                push!(parts, :($(coef(num, inv)) * ($normal ? $twin : $s)))
            end
        end
    end
    isempty(constant) || push!(parts, :($ncells * $(Expr(:call, GlobalRef(Base, :+), 0.0, constant...))))
    value = Expr(:call, GlobalRef(Base, :+), 0.0, parts...)
    post = Any[:(if $want && $ncells > 0
                     $total = $add($total, $value)
                 end)]
    pre, inloop, post
end

# A grouped tape has one writable Float64 view through the same index vector
# as its operand. A contiguous portion of that map gives both parents a
# unit-stride loop. Only built-in numeric storage participates, so every read
# and write addressed by the loop can be checked for overlap.
const _PRSpanParent = Union{Vector{Float64},SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}
const _PRSpanGather = SubArray{Float64,1,<:_PRSpanParent,Tuple{Vector{Int}}}
# A gather through one index vector, of plain storage or of another gather:
# `_pr_plain_gather` follows the composition down to its storage.
const _PRPlainGather = SubArray{Float64,1,<:AbstractVector{Float64},Tuple{Vector{Int}}}
# DenseArray is extensible, and a StridedVector can have a user-defined
# parent or index range whose getindex reads other storage.
const _PRPlainIndex = Union{Int,UnitRange{Int},Base.Slice{Base.OneTo{Int}},Base.OneTo{Int}}
const _PRPlainVector{T} = Union{Vector{T},SubArray{T,1,<:Array{T},<:Tuple{Vararg{_PRPlainIndex}}}}
_pr_contiguous_vector(x) = x isa _PRPlainVector{<:Union{Float32,Float64}} && stride(x, 1) == 1
@inline _pr_clear_storage(a, ::Tuple{}) = true
@inline function _pr_clear_storage(a, others::Tuple)
    b = first(others)
    (b === nothing || !_contiguous_storage_overlaps(a, b)) && _pr_clear_storage(a, Base.tail(others))
end
@inline _pr_disjoint_shadows(values, ::Tuple{}) = true
@inline function _pr_disjoint_shadows(values, shadows::Tuple)
    shadow, rest = first(shadows), Base.tail(shadows)
    (shadow === nothing || (_pr_clear_storage(shadow, values) && _pr_clear_storage(shadow, rest))) &&
        _pr_disjoint_shadows(values, rest)
end
_pr_span_possible(cells, value, shadow, operands) = false
@inline _pr_span_operands(::Tuple{}, cells, value, storage) = true
@inline function _pr_span_operands(operands::Tuple, cells, value, storage)
    x = first(operands)
    (x === value || (_pr_contiguous_vector(x) &&
        axes(x) == (cells,) && !_contiguous_storage_overlaps(storage, x))) &&
        _pr_span_operands(Base.tail(operands), cells, value, storage)
end
_pr_separate_gather(cells, value, shadow, operands) = false
@inline function _pr_separate_gather(cells::Base.OneTo{Int}, value::_PRSpanGather,
                                   shadow::_PRSpanGather, operands::Tuple)
    idx = only(parentindices(value))
    storage = parent(shadow)
    length(value) == length(cells) && only(parentindices(shadow)) === idx &&
        axes(storage) == axes(parent(value)) &&
        !_contiguous_storage_overlaps(storage, parent(value)) && !_contiguous_storage_overlaps(storage, idx) &&
        _pr_span_operands(operands, cells, value, storage)
end
@inline function _pr_span_possible(cells::Base.OneTo{Int}, value::_PRSpanGather,
                                   shadow::_PRSpanGather, operands::Tuple)
    idx = only(parentindices(value))
    length(cells) >= 8 && length(value) == length(cells) && @inbounds(idx[2] == idx[1] + 1) &&
        _pr_separate_gather(cells, value, shadow, operands)
end

# The candidate span ends at the bound of either the plate or the parent.
# Check every index before using it. The first failed span leaves the
# remaining cells to their original loop, including their scatter order.
@inline function _pr_span_end(indices, lo, lastcell, start, lastparent)
    1 <= start <= lastparent || return lo - 1
    count = min(lastcell - lo, lastparent - start)
    count >= 7 && @inbounds(indices[lo + 1]) == start + 1 || return lo - 1
    offset = start - lo
    hi = lo + count
    different = 0
    @inbounds @simd for i in lo+2:hi
        different |= xor(indices[i], i + offset)
    end
    iszero(different) ? hi : lo - 1
end

function _pr_span_loop(linear, cells, value, shadow, operands, cellbody, prepared, original)
    lo, hi, start, offset = (Symbol(cells, s) for s in (:_lo, :_hi, :_start, :_offset))
    address = :($linear + $offset)
    inner = _index_run_replace(cellbody) do node
        # The direct-operand check proved each active read uses its cell's
        # index. Inactive operands may also be read at other indices.
        if Meta.isexpr(node, :call, 3) && _pr_is_ref(node.args[1], @__MODULE__, :_plate_read) &&
                get(prepared, node.args[2], nothing) === value
            return :(parent($value)[$address])
        elseif Meta.isexpr(node, :call, 4) && _pr_is_ref(node.args[1], @__MODULE__, :_pr_shadow_add!) &&
                node.args[2] === shadow && node.args[3] === linear
            return :(parent($shadow)[$address] += $(node.args[4]))
        end
        node
    end
    quote
        $hi = 0
        if $_PR_WANT_VALUE isa Val{true} &&
                $(GlobalRef(@__MODULE__, :_pr_span_possible))($cells, $value, $shadow, ($(operands...),))
            $start = @inbounds only(parentindices($value))[1]
            $hi = $(GlobalRef(@__MODULE__, :_pr_span_end))(only(parentindices($value)),
                    1, last($cells), $start, lastindex(parent($value)))
        end
        if $hi > 0
            $lo = 1
            while true
                $start = @inbounds only(parentindices($value))[$lo]
                $offset = $start - $lo
                @inbounds @simd for $linear in $lo:$hi
                    $inner
                end
                $lo = $hi + 1
                $lo > last($cells) && break
                $start = @inbounds only(parentindices($value))[$lo]
                $hi = $(GlobalRef(@__MODULE__, :_pr_span_end))(only(parentindices($value)),
                        $lo, last($cells), $start, lastindex(parent($value)))
                $hi < $lo && break
            end
            @inbounds for $linear in $lo:last($cells)
                $cellbody
            end
        else
            $original
        end
    end
end

# A reordered sweep may read arrays only through the checked plate operands.
# Follow the flattened cell: aliases and primitive scalar arithmetic preserve
# that property. An ordinary index, opaque call or a nested cut could read
# storage aliased by a shadow. Keep its ordered loop. Check external scalar
# operands at the call as well, since a user number can overload arithmetic.
_pr_vector_scalar(x) = x isa Union{Base.IEEEFloat,Base.BitInteger,Bool}
function _pr_vectorized_cell_scalars(stmts, st, linear)
    local_values = Set{Symbol}([linear])
    external = Symbol[]
    for s in stmts
        if s isa _PRAlias
            args = (s.rhs,)
        elseif s isa _PRCall && _pr_is_read(s, st)
            args = (s.args[2],)
        elseif s isa _PRCall && any(f -> f === s.f,
                (+, -, *, /, ^, _native_mul, Base.CartesianIndex, abs, abs2, sin, cos, exp, log))
            args = s.args
        else
            return nothing
        end
        for a in args
            if a isa Symbol
                a in local_values || a in external || push!(external, a)
            elseif !_pr_vector_scalar(a)
                return nothing
            end
        end
        push!(local_values, s.lhs)
    end
    external
end

# A native direct-read loop already uses SIMD for the value. Its two-pass
# reverse can reorder scalar reductions when its writes are independent by
# cell. Check actual storage: singleton broadcasts, indirect views, aliased
# operands and overlapping shadows retain the original sweep order.
@inline function _pr_dense_reverse_cells(cells, operands::Tuple, shadows::Tuple)
    dense(x) = _pr_contiguous_vector(x) && axes(x, 1) == cells
    cells isa Union{Base.OneTo{Int},UnitRange{Int}} && all(dense, operands) &&
        all(x -> x === nothing || dense(x), shadows) &&
        _pr_disjoint_shadows(operands, shadows)
end

# Certify the range of floating-point operands using only plain, disjoint
# storage. What is measured is the range of binary exponents of the nonzero
# values a sweep can read: `(low, high)`, biased exponent fields. A subnormal
# value gives `low == 0` and a value that is not finite `high == 0x7ff`; both
# fail every certificate. `low > high` means no nonzero value was seen.
# Gathered values are measured by scanning their storage once.
const _PRExponents = Tuple{UInt64,UInt64}
const _PR_NO_EXPONENTS = (UInt64(0x7ff), UInt64(0))
const _PR_BAD_EXPONENTS = (UInt64(0), UInt64(0x7ff))
@inline _pr_merge_exponents(a::_PRExponents, b::_PRExponents) = (min(a[1], b[1]), max(a[2], b[2]))
@inline _pr_low_exponent(x::Float64) = (bits = reinterpret(UInt64, abs(x));
    ifelse(iszero(bits), UInt64(0x7ff), bits >> 52))
@inline _pr_high_exponent(x::Float64) = reinterpret(UInt64, abs(x)) >> 52
@inline _pr_exponents(x::Float64) = (_pr_low_exponent(x), _pr_high_exponent(x))
@inline _pr_exponents(x::Union{Base.BitInteger,Bool}) = _pr_exponents(Float64(x))
_pr_exponents(x) = _PR_BAD_EXPONENTS
@inline _pr_bound_storage(x) = x
@inline _pr_bound_storage(x::Base.Broadcast.Extruded) = x.x
# A gather is measured by its storage: every entry a cell can read is one of
# them. A partition's composed gather reads the same storage.
@inline _pr_exponents(x::_PRPlainGather) = _pr_exponents(parent(x))
function _pr_exponents(x::_PRPlainVector{Float64})
    low, high = _PR_NO_EXPONENTS
    @inbounds @simd for i in eachindex(x)
        low = min(low, _pr_low_exponent(x[i]))
        high = max(high, _pr_high_exponent(x[i]))
    end
    (low, high)
end
@inline _pr_all_exponents(range::_PRExponents, ::Tuple{}) = range
@inline _pr_all_exponents(range::_PRExponents, xs::Tuple) =
    _pr_all_exponents(_pr_merge_exponents(range, _pr_exponents(first(xs))), Base.tail(xs))
@inline _pr_scalar_exponents(xs::Tuple) = _pr_all_exponents(_PR_NO_EXPONENTS, xs)
# Whether the sweep's shared-reciprocal arithmetic is exact to rounding for
# nonzero leaves with exponents in `range`. Every nonzero leaf then has a
# magnitude in [2^-down, 2^up]. A target with `(n, d, s)` multiplies `n` leaves,
# divides by `d` and carries `s` bits of slack (`_pr_quotient_bounds!`), so its
# magnitude lies within `n * up + d * down + s` bits above one and
# `n * down + d * up + s` bits below. The sweep squares a target, scales it by
# up to three shared factors over a divisor and sums it over at most an Int's
# worth of cells (63 bits). Both totals stay within 903 of the 1022 bits a
# `Float64` has on either side of one.
@inline _pr_targets_fit(up::Int, down::Int, ::Tuple{}) =
    3up + down + 63 <= 903 && 3down + up <= 903
@inline function _pr_targets_fit(up::Int, down::Int, targets::Tuple)
    n, d, s = first(targets)
    2 * (n * up + d * down + s) + 3up + down + 63 <= 903 &&
        2 * (n * down + d * up + s) + 3down + up <= 903 &&
        _pr_targets_fit(up, down, Base.tail(targets))
end
@inline function _pr_within_range(range::_PRExponents, targets::Tuple)
    low, high = range
    (low > 0) & (high < 0x7ff) || return false
    up = max(0, Int(high) - 1022)
    down = max(0, 1023 - Int(low))
    _pr_targets_fit(up, down, targets)
end
# Several gathered operands, or a partition's composed gather. Each shadow
# is one of the adapter's buffers or a caller's shadow, gathered over storage
# of its own; the sweep reads operands and adds into those shadows only. With
# plain storage on both sides, and no shadow sharing memory with anything the
# sweep reads (an operand's entries or an index vector), no write of the loop
# can change a value its bounds were certified for.
_pr_plain_gather(x) = false
@inline _pr_plain_gather(x::_PRSpanParent) = true
@inline _pr_plain_gather(x::_PRPlainGather) = _pr_plain_gather(parent(x))
@inline _pr_clear_of(storage, x::_PRSpanParent) = !_contiguous_storage_overlaps(storage, x)
@inline _pr_clear_of(storage, x::_PRPlainGather) =
    !_contiguous_storage_overlaps(storage, only(parentindices(x))) && _pr_clear_of(storage, parent(x))
@inline _pr_clear_of(storage, x) = _pr_contiguous_vector(x) && !_contiguous_storage_overlaps(storage, x)
@inline _pr_cell_storage(x::_PRPlainGather, cells) = length(x) == length(cells) && _pr_plain_gather(x)
@inline _pr_cell_storage(x, cells) = _pr_contiguous_vector(x) && axes(x) == (cells,)
# Unrolled over the operand tuples: a closure over mixed operand types would
# box its captures on every call.
@inline _pr_cell_storages(::Tuple{}, cells) = true
@inline _pr_cell_storages(operands::Tuple, cells) =
    _pr_cell_storage(first(operands), cells) && _pr_cell_storages(Base.tail(operands), cells)
@inline _pr_all_clear_of(storage, ::Tuple{}) = true
@inline _pr_all_clear_of(storage, operands::Tuple) =
    _pr_clear_of(storage, first(operands)) && _pr_all_clear_of(storage, Base.tail(operands))
@inline _pr_shadow_clear(shadow::Nothing, operands::Tuple) = true
@inline _pr_shadow_clear(shadow, operands::Tuple) =
    _pr_plain_gather(shadow) && _pr_all_clear_of(_pr_gather_root(shadow), operands)
@inline _pr_shadows_clear(::Tuple{}, operands::Tuple) = true
@inline _pr_shadows_clear(shadows::Tuple, operands::Tuple) =
    _pr_shadow_clear(first(shadows), operands) && _pr_shadows_clear(Base.tail(shadows), operands)
@inline _pr_separate_storage(cells, operands::Tuple, shadows::Tuple) =
    cells isa Base.OneTo{Int} && _pr_cell_storages(operands, cells) &&
        _pr_shadows_clear(shadows, operands)
_pr_bounded_gather(cells, shadow, operands, ::Tuple{}) = false
@inline function _pr_bounded_gather(cells, shadow, operands, xs::Tuple)
    _pr_separate_gather(cells, first(xs), shadow, operands) ||
        _pr_bounded_gather(cells, shadow, operands, Base.tail(xs))
end
# Whether the operands a sweep reads and the shadows it writes are plain,
# disjoint storage, so that no write of the loop changes a measured value.
@inline function _pr_plain_storage(cells, operands::Tuple, shadows::Tuple)
    _pr_dense_reverse_cells(cells, operands, shadows) ||
        (length(shadows) == 1 && _pr_bounded_gather(cells, only(shadows), operands, operands)) ||
        _pr_separate_storage(cells, operands, shadows)
end
# The exponents of everything a sweep reads, beside those already in `range`.
@inline function _pr_storage_exponents(range::_PRExponents, cells, reads::Tuple, shadows::Tuple)
    operands = map(_pr_bound_storage, reads)
    _pr_plain_storage(cells, operands, shadows) || return _PR_BAD_EXPONENTS
    _pr_all_exponents(range, operands)
end

# The same with a site record. `ordinals` has one entry per read: zero for an
# operand measured at this call, else the position of its parameter among the
# reverse's reads of the constants table, whose exponents the site keeps.
@inline function _pr_storage_exponents(range::_PRExponents, cells, reads::Tuple, shadows::Tuple,
                                       site, ordinals::Tuple)
    operands = map(_pr_bound_storage, reads)
    _pr_plain_storage(cells, operands, shadows) || return _PR_BAD_EXPONENTS
    _pr_site_exponents(range, operands, site, ordinals)
end
@inline _pr_site_exponents(range::_PRExponents, ::Tuple{}, site, ::Tuple{}) = range
@inline _pr_site_exponents(range::_PRExponents, xs::Tuple, site, ordinals::Tuple) =
    _pr_site_exponents(
        _pr_merge_exponents(range, _pr_operand_exponents(site, first(ordinals), first(xs))),
        Base.tail(xs), site, Base.tail(ordinals))
@inline _pr_operand_exponents(site, k::Int, x) =
    k == 0 ? _pr_exponents(x) : _pr_cached_exponents(site, k, x)
# Without a tape slot there is nowhere to keep them: measured at each call.
@inline _pr_cached_exponents(site, k::Int, x) = _pr_exponents(x)

# For every target of the sweep's shared-reciprocal arithmetic, how its
# magnitude follows from the leaves': `(n, d, s)`, the leaves it multiplies,
# the leaves it divides by, and bits of slack. A leaf (an outer value or a
# read) is one factor. A literal has its own magnitude as slack. A product
# adds both parts and one bit, a quotient exchanges the divisor's parts, and a
# sum can cancel to one unit in the last place of its smaller operand (53
# bits). `_pr_within_range` charges these against the exponents measured at
# each call. Returns `nothing` when a target is not such arithmetic, or when
# even leaves within 40 bits of one would not fit.
function _pr_quotient_bounds!(q, stmts, st, external, invariant)
    defs = Dict{Symbol,Any}()
    for s in stmts
        s isa Union{_PRCall,_PRAlias} || continue
        (haskey(defs, s.lhs) || s.lhs in external) && return nothing
        # Only inspect storage ahead of calls whose target is known to be pure.
        # A local callable is checked against the value used during preparation.
        if s isa _PRCall && s.fref isa Expr
            return nothing
        end
        defs[s.lhs] = s
    end
    unproved = (0, 0, 1_000_000)
    bounds = Dict{Symbol,NTuple{3,Int}}()
    function bound(x)
        if !(x isa Symbol)
            x isa Union{Float64,Base.BitInteger,Bool} || return unproved
            value = float(x)
            iszero(value) && return (0, 0, 0)
            isfinite(value) && !issubnormal(value) || return unproved
            return (0, 0, abs(exponent(value)) + 1)
        end
        if x in external || x in invariant
            push!(q.shared, x)
            return (1, 0, 0)
        end
        get!(bounds, x) do
            s = get(defs, x, nothing)
            s isa _PRAlias && return bound(s.rhs)
            s isa _PRCall || return unproved
            _pr_is_read(s, st) && return (1, 0, 0)
            if s.f in (+, -) && length(s.args) in (1, 2)
                parts = map(bound, s.args)
                return (maximum(p -> p[1], parts), maximum(p -> p[2], parts),
                        maximum(p -> p[3], parts) + 53)
            end
            if length(s.args) == 2 && s.f in (*, _native_mul)
                a, b = bound(s.args[1]), bound(s.args[2])
                return (a[1] + b[1], a[2] + b[2], a[3] + b[3] + 1)
            end
            if length(s.args) == 2 && s.f === (/) && s.args[2] in q.divisors
                a, b = bound(s.args[1]), bound(s.args[2])
                return (a[1] + b[2], a[2] + b[1], a[3] + b[3] + 1)
            end
            unproved
        end
    end
    targets = unique!(NTuple{3,Int}[bound(x) for x in q.targets])
    _pr_targets_fit(40, 40, Tuple(targets)) ? targets : nothing
end

# Range checks can share the gather loop's reads on long consecutive runs.
# Only a sole zeroed shadow, reached exactly once in the program, can be
# restored for a retry. Plain floating storage and cell-index-only reads
# make the tentative arithmetic free of observable effects.
@inline function _pr_check_in_gather(cells, value, shadow, operands)
    value isa _PRSpanGather && length(parent(value)) >= 128 &&
        _pr_span_possible(cells, value, shadow, operands) &&
        all(x -> eltype(x) === Float64, operands)
end

function _pr_checked_reads(stmts, st, linear, written)
    indices = Set([linear])
    reads = Set{Symbol}()
    constant_reads = Symbol[]
    for s in stmts
        if s isa _PRAlias
            s.rhs in indices && push!(indices, s.lhs)
        elseif _pr_is_read(s, st)
            s.args[2] in indices || return nothing
            operand = s.args[1]
            if st.prepared[operand] === written
            elseif st.prepared[operand] in st.bound
                operand in constant_reads || push!(constant_reads, operand)
            else
                push!(reads, s.lhs)
            end
        elseif s isa _PRCall && s.f === Base.CartesianIndex && s.args == [linear]
            push!(indices, s.lhs)
        elseif !(s isa _PRCall && (s.f in (+, -, *, /, _native_mul, abs, abs2) ||
                (s.f === (^) && length(s.args) == 2 && s.args[2] isa Int && s.args[2] >= 0)))
            return nothing
        end
    end
    reads, constant_reads
end

# The checked sweep measures the exponents of its own reads, beside those of
# `range` (the scalars and the gather's storage, measured above the loop), and
# repeats itself in the separate-exponent form if they do not fit.
function _pr_checked_loop(ordinary, exceptional, visible, reads, check, shadow,
                          coefficients, value_pre, normal, range, targets)
    low, high = gensym(:low_exponent), gensym(:high_exponent)
    inspected = _index_run_replace(ordinary) do node
        if Meta.isexpr(node, :(=), 2) && node.args[1] in reads
            lhs = node.args[1]
            return Expr(:block, node,
                :($low = min($low, $(GlobalRef(@__MODULE__, :_pr_low_exponent))($lhs))),
                :($high = max($high, $(GlobalRef(@__MODULE__, :_pr_high_exponent))($lhs))))
        end
        node
    end
    quote
        if $check
            ($low, $high) = $range
            $(_pr_quotient_loop(inspected, union(visible, [low, high])))
            if !$(GlobalRef(@__MODULE__, :_pr_within_range))(($low, $high), $targets)
                fill!(parent($shadow), 0.0)
                $normal = false
                $(coefficients...)
                $(value_pre...)
                $exceptional
            end
        else
            $(_pr_quotient_loop(ordinary, visible))
        end
    end
end

# Compile the bounded and exceptional loops independently. Placing the cold
# exponent arithmetic alongside the dense loop can prevent its vectorization.
function _pr_quotient_loop(body, visible)
    inputs = _kernel_free_ports(body, visible)
    writes = _pr_assignments!(Pair{Symbol,Any}[], body)
    carried = sort!(collect(intersect(Set(first.(writes)), visible)))
    for name in carried
        name in inputs || push!(inputs, name)
    end
    length(inputs) <= _PLATE_CALL_MAX_ARGUMENTS || return body
    program = _PlateProgram(Expr(:function, Expr(:tuple, inputs...),
                                 Expr(:block, body, Expr(:tuple, carried...))))
    call = Expr(:call, program, inputs...)
    isempty(carried) ? call : Expr(:(=), Expr(:tuple, carried...), call)
end

function _pr_fused_loop(range, linear, cell::Expr, st::_PRProgramState, indexed::Bool)
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
        hoisted = Any[s for s in folded if (s isa _PRAlias || s isa _PRCall) && s.lhs in inv2]
        inloop = Any[s for s in folded if !((s isa _PRAlias || s isa _PRCall) && s.lhs in inv2)]
        active = copy(st.active)
        for _ in 1:2
            _pr_mark_reads!(active, folded, st)
            _pr_activity!(active, folded)
        end
        cuts = Dict{Symbol,Tuple{Symbol,Expr}}()
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
        q = (normal = gensym(:bounded_quotients), targets = Any[], shared = Any[], divisors = Any[],
             sums = Dict{Symbol,Symbol}(), fast = Dict{Symbol,Tuple{Any,Any}}(),
             inverses = Dict{Any,Symbol}())
        # In the bounded sweep a cell quotient by an invariant divisor is its
        # numerator times the divisor's reciprocal, computed once above the loop.
        # `q.fast` maps the quotient, and every alias of it, to its numerator and
        # divisor, so the reverse can move that reciprocal into the coefficients
        # it scales its sums by after the loop (`_pr_factored_quotient!`).
        fast_divisions = Dict{Symbol,Expr}()
        for s in inloop
            if s isa _PRAlias
                s.rhs isa Symbol && haskey(q.fast, s.rhs) && (q.fast[s.lhs] = q.fast[s.rhs])
                continue
            end
            s isa _PRCall && s.f === Base.:/ && length(s.args) == 2 || continue
            x, d = s.args
            d isa Symbol && d in cellvars && !(d in inv2) && continue
            push!(q.targets, x)
            push!(q.divisors, d)
            r = get!(q.inverses, d) do
                r = gensym(:shared_inverse)
                push!(coefficients, :($r = $(q.normal) ? inv($d) : 0.0))
                r
            end
            fast_divisions[s.lhs] = :($x * $r)
            q.fast[s.lhs] = (x, d)
        end
        flush = Any[]
        partial_sums = Set{Symbol}()
        if result isa Symbol && result in active
            shared = Set{Symbol}(s for s in st.outer if !(s in st.totals))
            union!(shared, hoisted_active)
            fc = _PRFactored(st, cellvars, inv2, shared, active, linear, ncells,
                             coefficients, Any[], Dict{Any,Any}(), Dict{Any,Symbol}(), Dict{Any,Any}(),
                             _pr_single_reads(inloop, st, active), Dict{Symbol,Vector{Any}}())
            # The cell density's adjoint is the total's (the seed), the same
            # at every cell.
            adj = Dict{Symbol,Vector{_PRTerm}}()
            mutable = Dict{Symbol,Symbol}()
            _pr_contribute!(reverse, fc, adj, mutable, result,
                            [_PRTerm(1.0, Any[_pr_bar(total)], nothing)], true)
            _pr_factored_sweep!(reverse, fc, inloop, adj, mutable, true, cuts, q)
            st.joint_cell_cut |= !isempty(cuts)
            _pr_flush_counts!(fc)
            coefficients, flush = fc.pre, fc.post
            union!(partial_sums, Base.values(fc.sums))
            _pr_reverse_exprs!(after, hoisted, active, copy(defined); touched = st.touched, cuts)
        end
        # Select joint cuts from the reverse's actual uses before emitting the
        # forward. Hoisted calls stay above the loop; branch-local calls stay lazy.
        forward_pre = _pr_forward_exprs(hoisted, cuts)
        forward_in = _pr_forward_exprs(inloop, cuts)
        # The cell's value, for a sweep that also returns the plate's total
        # (`_PR_WANT_VALUE`): its invariant terms once, `ncells` times, after
        # the loop, and one partial sum per cell-varying term in it.
        value_pre, value_in, value_post = _pr_value_accumulation(
            inloop, inv2, cellvars, result, total, acc.args[1], ncells, st;
            reverse, normal = isempty(q.divisors) ? nothing : q.normal, sums = partial_sums,
            quotients = q)
        direct_in, read_operands = _pr_direct_reads(inloop, st)
        cellbody = Expr(:block, forward_in..., value_in..., reverse...)
        plain_loop = :(@inbounds for $linear in $cells
                           $cellbody
                       end)
        # The native spans-axis arm reads raw operands. Preserve its SIMD
        # reduction when the reverse writes independently by cell. Run-scheduled
        # loops keep their form for the index-run transformation below.
        raw = any(s -> _pr_is_read(s, st) && s.f === _plate_read, inloop)
        scalar_inputs = !indexed && raw && isempty(cuts) ?
            _pr_vectorized_cell_scalars(inloop, st, linear) : nothing
        if scalar_inputs !== nothing
            written = unique(st.prepared[r] for r in read_operands if haskey(st.shadow, st.prepared[r]))
            shadows = map(p -> st.shadow[p], written)
            dense = Expr(:for, :($linear = $cells), cellbody)
            # Only the range-checked quotient variant needs the stronger
            # independence annotation to keep the dense reduction vectorized.
            simd = isempty(q.divisors) ? :(@simd $dense) : :(@simd ivdep $dense)
            loop = :(if $_PR_WANT_VALUE isa Val{false} &&
                            $(GlobalRef(@__MODULE__, :_pr_dense_reverse_cells))(
                                $cells, ($(read_operands...),), ($(shadows...),))
                         @inbounds $simd
                     else
                         $plain_loop
                     end)
            if length(written) == 1
                loop = _pr_span_loop(linear, cells, only(written), only(shadows), read_operands,
                                     cellbody, st.prepared, loop)
            end
            scalars = Expr(:tuple, _pr_bar(total), scalar_inputs...)
            loop = :(if Base.all($(GlobalRef(@__MODULE__, :_pr_vector_scalar)), $scalars)
                         $loop
                     else
                         $plain_loop
                     end)
        elseif isempty(read_operands)
            loop = plain_loop
        else
            loop = :(if $(GlobalRef(@__MODULE__, :_pr_spans_all))($cells, $(read_operands...))
                         @inbounds for $linear in $cells
                             $(_pr_forward_exprs(direct_in, cuts)...)
                             $(value_in...)
                             $(reverse...)
                         end
                     else
                         $plain_loop
                     end)
        end
        certificate = false
        measure = Any[]
        if !isempty(q.divisors)
            function specialize(body, arm)
                _index_run_replace(body) do node
                    if Meta.isexpr(node, :if, 3) && node.args[1] === q.normal
                        return specialize(node.args[arm], arm)
                    elseif arm == 2 && Meta.isexpr(node, :(=), 2) && haskey(fast_divisions, node.args[1])
                        return Expr(:(=), node.args[1], fast_divisions[node.args[1]])
                    end
                    node
                end
            end
            scalars = isempty(cuts) ? _pr_vectorized_cell_scalars(inloop, st, linear) : nothing
            proved = scalars === nothing ? nothing : _pr_quotient_bounds!(q, inloop, st, scalars, inv2)
            if proved !== nothing
                written = unique(st.prepared[r] for r in read_operands if haskey(st.shadow, st.prepared[r]))
                shadows = map(p -> st.shadow[p], written)
                checked = unique(vcat(q.shared, q.divisors))
                targets = Expr(:tuple, (Expr(:tuple, t...) for t in proved)...)
                measured = gensym(:exponents)
                measure = Any[:($measured = $(GlobalRef(@__MODULE__, :_pr_scalar_exponents))(($(checked...),)))]
                structure = :(Base.all($(GlobalRef(@__MODULE__, :_pr_vector_scalar)), ($(scalars...),)) &&
                              !Base.any(iszero, ($(unique(q.divisors)...),)))
                for s in inloop
                    if s isa _PRCall && s.fref isa Symbol
                        structure = :($(s.fref) === $(QuoteNode(s.f)) && $structure)
                    end
                end
                # A read of the constants table has the same entries at
                # every call: its exponents are measured once for the site.
                ordinal(r) = something(findfirst(==(st.prepared[r]), st.bound), 0)
                ordinals = map(ordinal, read_operands)
                cached_exponents = (k, operand) -> Expr(:call,
                    GlobalRef(@__MODULE__, :_pr_cached_exponents), _PR_SITE, k,
                    Expr(:call, GlobalRef(@__MODULE__, :_pr_bound_storage), operand))
                storage = if all(iszero, ordinals)
                    :($(GlobalRef(@__MODULE__, :_pr_within_range))(
                        $(GlobalRef(@__MODULE__, :_pr_storage_exponents))(
                            $measured, $cells, ($(read_operands...),), ($(shadows...),)), $targets))
                else
                    st.sited = true
                    :($(GlobalRef(@__MODULE__, :_pr_within_range))(
                        $(GlobalRef(@__MODULE__, :_pr_storage_exponents))(
                            $measured, $cells, ($(read_operands...),), ($(shadows...),),
                            $_PR_SITE, ($(ordinals...),)), $targets))
                end
                certificate = :($structure && $storage)
                ordinary = specialize(loop, 2)
                exceptional = specialize(plain_loop, 3)
                if !indexed
                    visible = union(st.outer, keys(st.prepared), Base.values(st.prepared),
                                    Base.values(st.shadow), keys(st.values), scalars,
                                    (cells, ncells, _PR_WANT_VALUE, total, _pr_bar(total)))
                    for init_block in (forward_pre, coefficients, value_pre, init)
                        writes = _pr_assignments!(Pair{Symbol,Any}[], Expr(:block, init_block...))
                        union!(visible, first.(writes))
                    end
                    exceptional = _pr_quotient_loop(exceptional, visible)
                    checked_reads = if raw && length(st.shadow) == length(written) == 1 &&
                            get(st.reads, only(written), 0) == 1
                        _pr_checked_reads(inloop, st, linear, only(written))
                    end
                    if checked_reads === nothing || all(isempty, checked_reads)
                        ordinary = _pr_quotient_loop(ordinary, visible)
                    else
                        # The checked sweep measures its dense reads itself.
                        # Above the loop only the gather's storage is scanned,
                        # and a read of the constants table has its exponents
                        # from the site.
                        reads, constant_reads = checked_reads
                        check = gensym(:check_in_gather)
                        gathered, owned = only(written), only(shadows)
                        merged = Expr(:call, GlobalRef(@__MODULE__, :_pr_merge_exponents), measured,
                            Expr(:call, GlobalRef(@__MODULE__, :_pr_exponents), :(parent($gathered))))
                        for operand in constant_reads
                            st.sited = true
                            merged = Expr(:call, GlobalRef(@__MODULE__, :_pr_merge_exponents), merged,
                                          cached_exponents(ordinal(operand), operand))
                        end
                        push!(measure,
                            :($check = $_PR_WANT_VALUE isa Val{true} &&
                                $(GlobalRef(@__MODULE__, :_pr_check_in_gather))(
                                    $cells, $gathered, $owned, ($(read_operands...),))),
                            :($check && ($measured = $merged)))
                        certificate = :($structure && ($check ?
                            $(GlobalRef(@__MODULE__, :_pr_within_range))($measured, $targets) : $storage))
                        # With every dense read measured above the loop the
                        # sweep inspects nothing.
                        ordinary = isempty(reads) ? _pr_quotient_loop(ordinary, visible) :
                            _pr_checked_loop(ordinary, exceptional, visible, reads,
                                check, only(shadows), coefficients, value_pre, q.normal, measured, targets)
                    end
                end
                loop = Expr(:if, q.normal, ordinary, exceptional)
            else
                loop = specialize(plain_loop, 3)
            end
        end
        isempty(q.divisors) || pushfirst!(coefficients, measure..., :($(q.normal) = $certificate))
        quote
            $(forward_pre...)
            $(init...)
            $cells = $range
            $ncells = Float64(length($cells))
            $(coefficients...)
            $(value_pre...)
            $loop
            $(value_post...)
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

# A lazy exp.(x) over one dense vector has exactly one writable input.
# The cell already read the primal y, so its result-based partial composes
# with the cell cotangent without rereading x or materializing exp.(x).
const _PRExpCells = _CellBroadcast{T,1,B} where
    {T<:Union{Float32,Float64},B<:Base.Broadcast.Broadcasted{<:Any,<:Any,typeof(exp),Tuple{Vector{T}}}}
struct _PRMappedShadow{F,V}
    f::F
    storage::V
end
@inline _pr_read_cotangent(shadow, y, v) = v
@generated function _pr_read_cotangent(::_PRMappedShadow{F}, y, v) where {F}
    slope = only(_pr_result_partials(F.instance, :y, (:x,)))
    :(v * $slope)
end
@inline _pr_shadow_add!(s::_PRMappedShadow, i, v) = _pr_shadow_add!(s.storage, i, v)
@inline _pr_shadow_add!(::Nothing, i, v) = nothing
@inline function _pr_shadow_add!(shadow::AbstractArray, i, v)
    # Batched operands are one-dimensional, but may broadcast a single entry
    # over the plate. Its cotangent sums at that entry, just as its primal
    # reads do; the cell index would write beyond a singleton shadow.
    index = length(shadow) == 1 ? firstindex(shadow) : i
    @inbounds shadow[index] += v
    nothing
end

# A straight-line program statement `lhs = rhs` outside cell loops: kept as
# written in the forward, swept in reverse at the end of its block.
struct _PRStatement
    lhs::Symbol
    rhs::Any
end

function _pr_sets_arm_ready(node, flag)
    node isa Expr && node.head !== :quote || return false
    Meta.isexpr(node, :(=), 2) && node.args[1] === flag && node.args[2] === true && return true
    any(arg -> _pr_sets_arm_ready(arg, flag), node.args)
end

# Transform one block: its forward statements (cell loops fused, branch arms
# transformed in place), then the reverse of its straight-line active
# statements whose adjoints received a contribution.
function _pr_program_block(stmts, st::_PRProgramState, indexed_cells; group_index = nothing)
    fwd = Any[]
    record = _PRStatement[]
    arm_sweeps = Dict{Int,Vector{_PRStatement}}()
    for (i, s) in pairs(stmts)
        s isa LineNumberNode && continue
        loop = _pr_cell_loop(s)
        if haskey(st.groups, s)
            group = st.groups[s]
            inner = _pr_program_block(group.prefix, st, indexed_cells; group_index = group.index)
            # Group-local scalar adjoints sum over this group's cells only.
            reset = Any[:($(_pr_bar(a)) = 0.0) for a in group.locals
                        if a in st.outer && a in st.touched && !(a in st.totals)]
            emit = Expr(:if, :(!($_PR_WANT_VALUE isa Val{false})), Expr(:block, group.finish...))
            push!(fwd, Expr(:for, s.args[1], Expr(:block, reset..., inner..., emit)))
        elseif loop !== nothing
            fused = _pr_fused_loop(loop..., st, loop[2] in indexed_cells)
            # A group may have no cells. Even a newly hoisted scalar partial
            # must only run when the original cell has an iteration.
            push!(fwd, group_index === nothing ? fused :
                :(if !isempty($(loop[1])); $fused; end))
        elseif Meta.isexpr(s, :if)
            arms = Any[s.args[1]]
            for arm in s.args[2:end]
                push!(arms, Expr(:block, _pr_program_block(
                    Meta.isexpr(arm, :block) ? arm.args : Any[arm], st, indexed_cells; group_index)...))
            end
            push!(fwd, Expr(:if, arms...))
        elseif Meta.isexpr(s, :block)
            append!(fwd, _pr_program_block(s.args, st, indexed_cells; group_index))
        else
            _pr_has_loop(s) && !isdisjoint(_pr_reads!(Set{Symbol}(), s, st.mod, st.values), st.active) &&
                _pr_fail("a loop over active values that is not a summed cell loop")
            push!(fwd, s)
            haskey(st.group_shadows, s) && push!(fwd, st.group_shadows[s])
            for (lhs, rhs) in _pr_assignments!(Pair{Symbol,Any}[], s)
                if rhs === false && haskey(st.arm_initializers, lhs)
                    # The arm's recipes run inside the loop. Their shared
                    # operands can be defined after this flag's declaration.
                    # Propagate the arm cotangents before those operands.
                    site = findlast(stmt -> _pr_sets_arm_ready(stmt, lhs), stmts)
                    site === nothing && _pr_fail("missing shared arm initialization")
                    push!(get!(Vector{_PRStatement}, arm_sweeps, site), _PRStatement(lhs, rhs))
                    continue
                end
                lhs in st.active || continue
                if lhs in st.totals
                    group_index !== nothing && Meta.isexpr(rhs, :call) &&
                        _pr_is_ref(rhs.args[1], @__MODULE__, :_plate_total_add) &&
                        _pr_fail("nested total addition outside a supported cell loop")
                    continue
                end
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
        haskey(arm_sweeps, i) && append!(record, arm_sweeps[i])
    end
    append!(fwd, _pr_outer_reverse(record, st, group_index))
end

function _pr_outer_reverse(record, st, group_index = nothing)
    rev = Any[]
    for s in Iterators.reverse(record)
        if s.rhs === false && haskey(st.arm_initializers, s.lhs)
            arm = [_PRStatement(lhs, rhs) for (lhs, rhs) in st.arm_initializers[s.lhs]]
            sweep = _pr_outer_reverse(arm, st, group_index)
            isempty(sweep) || push!(rev, Expr(:if, s.lhs, Expr(:block, sweep...)))
            continue
        end
        s.lhs in st.touched || continue
        if s.rhs isa Expr && haskey(st.group_scatters, s.rhs)
            push!(rev, st.group_scatters[s.rhs])
            continue
        end
        ctx = _PRCtx(st.mod, st.counter, st.values)
        sub = Any[]
        r = _pr_flatten!(sub, s.rhs, ctx)
        st.counter = ctx.counter
        active = _pr_activity!(copy(st.active), sub)
        defined = Set{Symbol}(_pr_bar(a) for a in st.outer if !(a in st.totals))
        if r isa Symbol && r in active && r !== s.lhs
            cuts = Dict{Symbol,Tuple{Symbol,Expr}}()
            subrev = Any[]
            _pr_accumulate!(subrev, defined, r, _pr_bar(s.lhs); touched = st.touched)
            leaf = call -> begin
                group_index === nothing && return nothing
                _pr_is_read(call, st) || return nothing
                prepared, index = call.args
                index === group_index || _pr_fail("group operand read at another index")
                shadow = st.shadow[st.prepared[prepared]]
                cot = :($(GlobalRef(@__MODULE__, :_pr_read_cotangent))($shadow, $(call.lhs), $(_pr_bar(call.lhs))))
                :($(GlobalRef(@__MODULE__, :_pr_shadow_add!))($shadow, $index, $cot))
            end
            _pr_reverse_exprs!(subrev, sub, active, defined; leaf, touched = st.touched, cuts)
            append!(rev, _pr_forward_exprs(sub, cuts), subrev)
        end
    end
    rev
end

include("plate_reverse_nested.jl")

# _native_plate_inline_arm declares its shared values outside the cells and
# initializes them on the first visit to their arm. All visits contribute to
# the same cotangents. Sweep those recipes once, only if that arm was taken.
function _pr_arm_initializers!(initializers, node)
    node isa Expr && node.head !== :quote || return
    if Meta.isexpr(node, :if, 2)
        cond, init = node.args
        if _index_run_call(cond, GlobalRef(Base, :!), 1) && Meta.isexpr(init, :block)
            ready = cond.args[2]
            if ready isa Symbol && _gensym_base(ready) == "arm_ready" && !isempty(init.args) &&
                    last(init.args) == Expr(:(=), ready, true)
                entries = init.args[1:end-1]
                all(e -> Meta.isexpr(e, :(=), 2) && e.args[1] isa Symbol, entries) ||
                    _pr_fail("unsupported shared arm initializer")
                body = Pair{Symbol,Any}[e.args[1] => e.args[2] for e in entries]
                get!(initializers, ready, body) == body ||
                    _pr_fail("different initializers for a shared arm")
                return
            end
        end
    end
    foreach(a -> _pr_arm_initializers!(initializers, a), node.args)
end

"""
    _pr_program_reverse(ast, active_params, mod, values) -> (expr, scalar, batched, joint, projections)

The reverse of plate program `ast` (a `function (params...) body; result end`
expression). `active_params` are the parameters that carry derivatives;
`values` maps program-local callable parameters to their values. Returns the
reverse function's expression `(params..., ȳ, want, shadows...) -> (value,
cotangents of the scalar active parameters...)`, the scalar active parameters
in order, the batched (per-cell) active parameters, and whether a cell uses a
joint value/partial cut, and the argument positions for group gather dispatch.
Shadows follow `want` in batched parameter order.
One pass over the cells computes the cotangents for seed ȳ, adding each
batched operand's cotangent into its shadow.

Under `Val(true)` each shadow must be a zeroed buffer the caller owns: an
`Array`, or a gathered operand's buffer, a view of owned storage through the
operand's own index vector. The sweep may add unscaled partials into it and
scale it in place after the loop, and the caller adds it into the operand's
real shadow. Under `Val(:logical)` the sweep accumulates the cell values in
their loop and writes cotangents to zeroed buffers indexed like the operands.
Under `Val(false)` it only adds cotangents into the supplied shadows.
"""
# The summed-plate factoring pass (summed_plate_factors.jl) keeps the plate's
# own loop in the fallback arm of the block it emits. That loop supplies the
# derivative graph, while the primal keeps the factored sum. A factored sum
# that reads frozen least-squares statistics (bound_statistics.jl) leaves no
# cell work for a generated reverse, so the backend differentiates it.
_pr_reads_statistics(x) = x isa Expr && x.head !== :quote &&
    (_index_run_call(x, GlobalRef(@__MODULE__, :_ls_rss)) || any(_pr_reads_statistics, x.args))
# A block the factoring pass produced, also after an embedded plan renamed its
# locals: the count's name then carries the renaming suffix, which
# `_spf_produced` does not accept, and the outer lowering factors the kept loop
# again inside the renamed block's fallback arm.
_pr_factored_block(x) = x isa Expr && x.head === :block && any(x.args) do a
    Meta.isexpr(a, :(=), 2) && a.args[1] isa Symbol &&
        occursin(_spf_mark() * "_count", String(a.args[1]))
end
function _pr_factored_source(ast)
    _index_run_replace(ast) do node
        _pr_factored_block(node) || return node
        _pr_reads_statistics(node) && _pr_fail("the plate is summed from bound statistics")
        fallback = node.args[end]
        Meta.isexpr(fallback, :if, 2) || _pr_fail("unrecognized factored plate")
        own = Any[a for a in (Meta.isexpr(fallback.args[2], :block) ? fallback.args[2].args : Any[fallback.args[2]])
                  if !(a isa LineNumberNode)]
        length(own) == 1 || _pr_fail("unrecognized factored plate")
        _pr_factored_source(only(own))
    end
end

function _pr_program_reverse(ast::Expr, active_params, mod::Module, values::Dict{Symbol,Any},
                             bound::Vector{Symbol} = Symbol[])
    Meta.isexpr(ast, :function, 2) && Meta.isexpr(ast.args[1], :tuple) ||
        _pr_fail("not a plate program")
    params = Symbol[p for p in ast.args[1].args]
    source, index_runs = _pr_index_run_source(_pr_factored_source(ast))
    body = source.args[2]
    initializers = Dict{Symbol,Vector{Pair{Symbol,Any}}}()
    _pr_arm_initializers!(initializers, body)
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
    groups, declared = _pr_additive_groups(body, totals, terminal_totals, assignments)
    prepared, broadcastables = _pr_prepared_operands(body, params)
    active = _pr_program_activity(body, active_params, mod, values)
    # A group's gathered array can itself be active. Its direct reads must
    # pass the same coordinate check as reads of a program argument.
    cell_reads = _pr_direct_operands!(prepared, body, params, active,
                                     broadcastables, mod, values, groups)
    sources, group_shadows, scatters, projections = _pr_group_shadows!(
        cell_reads, groups, params, active, prepared, mod, values, declared)
    batched = Symbol[p for p in params if p in active_params &&
                     (p in Base.values(prepared) || p in sources)]
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
    for assignment in keys(group_shadows)
        p = assignment.args[1]
        shadow[p] = Symbol(p, :_shadow)
    end
    if !isempty(groups)
        for p in Base.values(prepared)
            p in active && !haskey(shadow, p) && _pr_fail("no shadow for nested operand `$p`")
        end
    end
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
    for entries in Base.values(initializers), (lhs, _) in entries
        delete!(cell_locals, lhs)
    end
    outer = Set{Symbol}(a for a in active if !(a in cell_locals))
    counts = _pr_read_counts(body, prepared, mod, values)
    # Several groups may contribute to the same entry. Scaling a whole shadow
    # after each group's loop would also rescale earlier groups' contributions.
    isempty(groups) || empty!(counts)
    # The site record is one more argument of the reverse's call.
    length(params) + length(batched) + 3 <= _PLATE_CALL_MAX_ARGUMENTS || (bound = Symbol[])
    st = _PRProgramState(mod, values, active, prepared, shadow, totals, output,
                         Set{Symbol}(), 0, outer, counts, initializers, false, false,
                         groups, group_shadows, scatters, bound, false)
    new_body = _pr_program_block(stmts, st, keys(index_runs))
    # Every outer adjoint starts at zero; every total's adjoint is the seed.
    init = Any[:($(_pr_bar(a)) = 0.0) for a in sort!(collect(outer)) if !(a in totals)]
    seeds = Any[:($(_pr_bar(t)) = $seed) for t in sort!(collect(totals))]
    checks = Any[:($(GlobalRef(@__MODULE__, :_pr_check_shadow))($(shadow[p]), $p)) for p in batched]
    fexpr = Expr(:function,
        Expr(:tuple, params..., seed, _PR_WANT_VALUE, (shadow[p] for p in batched)...,
             (st.sited ? (_PR_SITE,) : ())...),
        Expr(:block, checks..., init..., seeds..., new_body...,
             Expr(:tuple, output, (_pr_bar(p) for p in scalar)...)))
    # What a cell shares between its value and its partials, for the schedule
    # choice (`_pr_target_expr`): `true` for work the backend's two sweeps
    # would repeat, `:owned` for the joint cut of an owned scalar rule, which
    # the backend shares itself in a flat loop. In a nested total the backend
    # tapes that rule's partials per group, so one flat owned tape pays there
    # too.
    reuse = st.shared_cell_work || (st.joint_cell_cut && !isempty(groups)) ? true :
            st.joint_cell_cut ? :owned : false
    _pr_index_run_schedule(fexpr, index_runs, prepared, shadow, params, batched), scalar, batched, reuse, projections
end

include("plate_reverse_runs.jl")

_pr_check_shadow(::Nothing, x) = nothing
_pr_check_shadow(s::_PRMappedShadow, x::_PRExpCells) =
    _pr_check_shadow(s.storage, only(x.broadcast.args))
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
`reverse` is generated from the program's source: `(args..., ȳ, want,
shadows...)` runs the cells once, adds each batched operand's cotangent into
its shadow and returns the value (when `want` is `Val(true)`) with the scalar
operands' cotangents (positions `S`); `B`
are the positions of the batched operands. The Enzyme extension runs it either
in the augmented primal with a unit seed (scaling afterwards) or in the reverse
pass with the seed; see `_pr_one_pass` there.
"""
struct _PlateSumRule{S,B,P,R,Joint,Projections,CheckedSources} <: Function
    program::P
    reverse::R
end
_PlateSumRule{S,B,P,R,J,G}(p::P, r::R) where {S,B,P,R,J,G} = _PlateSumRule{S,B,P,R,J,G,false}(p, r)
_PlateSumRule{S,B,P,R,J}(p::P, r::R) where {S,B,P,R,J} = _PlateSumRule{S,B,P,R,J,()}(p, r)
_PlateSumRule{S,B,P,R}(p::P, r::R) where {S,B,P,R} = _PlateSumRule{S,B,P,R,false}(p, r)

# Pass the dense leaf itself across the AD rule boundary. Enzyme can then
# tape it when the caller overwrites it after the primal. The by-value
# broadcast descriptor alone does not expose that write to its overwrite flag.
# Keep the descriptor's style and axes so the original program sees the same
# lazy array, including when its single input broadcasts over a larger axis.
struct _PRExpProgram{Inputs,Shadows,F,A}
    f::F
    shapes::A
end
_pr_exp_program(::Val{I}, ::Val{J}, f::F, shapes::A) where {I,J,F,A} =
    _PRExpProgram{I,J,F,A}(f, shapes)
@inline function _pr_exp_operand(x::Vector{T}, shape) where {T}
    bc = Base.Broadcast.Broadcasted(first(shape), exp, (x,), last(shape))
    _CellBroadcast{T,1,typeof(bc)}(bc)
end
@inline _pr_exp_shadow(::Nothing) = nothing
@inline _pr_exp_shadow(storage) = _PRMappedShadow(exp, storage)
@generated function (p::_PRExpProgram{I,J})(args::Vararg{Any,N}) where {I,J,N}
    operands = Any[:(getfield(args, $k)) for k in 1:N]
    for (j, i) in enumerate(I)
        operands[i] = :(_pr_exp_operand($(operands[i]), getfield(p.shapes, $j)))
    end
    for j in J
        operands[j] = :(_pr_exp_shadow($(operands[j])))
    end
    Expr(:call, :(p.f), operands...)
end
@inline function _pr_exp_rule(rule::_PlateSumRule{S,B}, inputs, shadows, shapes) where {S,B}
    p = _pr_exp_program(inputs, Val(()), rule.program, shapes)
    r = _pr_exp_program(inputs, shadows, rule.reverse, shapes)
    _PlateSumRule{S,B,typeof(p),typeof(r),true}(p, r)
end

# The augmented primal sweeps once. A grouped tape holds partials per parent
# entry of an indirect view. A logical tape holds partials per operand entry,
# including each entry of an active tuple. The caller selects the layout
# before Enzyme differentiates the call.
struct _PlateSumRuleOnePass{S,B,R,Layout} <: Function
    rule::R
end
_pr_tape_layout(::_PlateSumRuleOnePass{S,B,R,L}) where {S,B,R,L} = Val(L)
# For a gather that would allocate its tape, below this many cells the
# allocation outweighs the pass it saves
# (eight schools, 8 cells: 0.070 µs in two passes, 0.102–0.118 µs in one).
const _PR_ONE_PASS_MIN_CELLS = 128
# Whether an operand type is a gather one pass can factor through: a `Float64`
# view with an index vector. A unit range (`view(u, 3:10)`, as RKPPL holds a
# parameter block of its flat vector) repeats no entry, so it has no groups to
# factor and its parent is at least as long as the view. Defined before the
# generators that call it, which run in the world of their own definition.
_pr_gather_value(::Type) = false
_pr_gather_value(::Type{<:SubArray{Float64,1,<:AbstractVector,<:Tuple{AbstractVector{<:Integer}}}}) = true
_pr_gather_value(::Type{<:SubArray{Float64,1,<:AbstractVector,<:Tuple{AbstractUnitRange{<:Integer}}}}) = false
# The storage a gather's entries live in. A lane partition of a gathered
# operand (`_lane_view`) is a view of that view: its indices compose, and both
# address the entries of the innermost gather's parent. One-pass buffers,
# their scaling and their products with the operand are per entry of that
# storage, so a partition's lanes share the groups of the whole plate.
@inline _pr_gather_root(x) = x
@inline _pr_gather_root(x::SubArray) =
    _pr_gather_value(typeof(x)) ? _pr_gather_root(parent(x)) : x
# The same composition of index vectors over another root (a tape slice).
@inline _pr_regather(root, x::SubArray) = Base.SubArray(
    _pr_gather_value(typeof(parent(x))) ? _pr_regather(root, parent(x)) : root, parentindices(x))

# Owned storage for the one-pass tape of one rule call site. A preparation that
# owns AD caches (`_ad_na_program`) keeps one slot per generated rule there and
# passes it as the call's last operand; a call without one allocates its tape.
# The tape is leased by removal: the augmented primal takes the vector out and
# leaves the empty one, and the reverse pass puts back the tape it used. No
# flag has to be restored when an exception separates the two passes: the slot
# is then empty, the next call allocates and its reverse pass refills the slot.
# A second call at the site before the first one's reverse pass (the call in a
# loop) finds the slot empty and allocates as well; a changed length does too.
mutable struct _PlateTapeSlot
    tape::Vector{Float64}
    empty::Vector{Float64}
    # A backend adapter that could not take the one pass at this site says so
    # here. The schedule is chosen on operand types and sizes before the
    # backend sees the call; its own annotations can still refuse it.
    declined::Bool
    # The measured exponents (low, high) of each operand the call reads from
    # its program's constants table, by ordinal (`_pr_cached_exponents`).
    exponents::Vector{UInt64}
end
function _PlateTapeSlot()
    empty = Float64[]
    _PlateTapeSlot(empty, empty, false, UInt64[])
end
# A constants-table entry is bound data or a value derived from data alone. A
# slot belongs to one preparation, whose table does not change, so each such
# operand is scanned at the slot's first call and never again. A mark is a
# literal of a program that bindings may share, and keeps nothing.
const _PR_UNMEASURED = typemax(UInt64)
@inline function _pr_cached_exponents(slot::_PlateTapeSlot, k::Int, x)
    kept = slot.exponents
    if 2k <= length(kept)
        low = @inbounds kept[2k - 1]
        low == _PR_UNMEASURED || return (low, @inbounds kept[2k])
    end
    _pr_measure_exponents!(slot, k, x)
end
@noinline function _pr_measure_exponents!(slot::_PlateTapeSlot, k::Int, x)
    low, high = _pr_exponents(x)
    kept = slot.exponents
    while length(kept) < 2k
        push!(kept, _PR_UNMEASURED)
    end
    kept[2k - 1] = low
    kept[2k] = high
    (low, high)
end
@inline _pr_mark_declined!(::Nothing) = nothing

# The mark of one rule call site in a program that owns no AD caches (an
# allocating kernel's). The program passes it as the call's last operand,
# where a non-allocating program passes its tape slot: it says that a
# preparation checks the call, and it leases no tape, so each call allocates
# its own. It holds no state. A program is compiled once for its content and
# every preparation of it runs that code, so nothing the program holds as a
# literal can belong to one preparation; a refused call is recorded for the
# preparation that is evaluating instead (`_pr_checking_sites`).
struct _PlateRuleMark
    ordinal::Int
end
_PlateRuleMark() = _PlateRuleMark(1)
# The trailing operand of a rule call that is no operand of its program.
const _PlateRuleSite = Union{_PlateTapeSlot,_PlateRuleMark}

# A refusal ends the trial derivative before it enters the general reverse.
# In particular, registered Enzyme on Julia 1.13 can lose the roots of a
# constant gathered operand when that reverse reads it from the tape.
struct _PRDeclined <: Exception
    site::_PlateRuleSite
end
const _PR_REFUSAL_SCOPE = :ReactiveKernels_plate_reverse_refusal
@noinline function _pr_mark_declined!(site::_PlateRuleSite)
    site isa _PlateTapeSlot && (site.declined = true)
    get(task_local_storage(), _PR_REFUSAL_SCOPE, false) && throw(_PRDeclined(site))
    nothing
end

# Site numbers follow the structural plate walk, including candidates omitted
# on a retry. The clean program carries the ordinal and allowed signature
# types. The mutable records belong to each preparation.
struct _PRSites
    marks::Vector{_PlateRuleMark}
    allowed::Union{Nothing,Dict{Int,Type}}
    selected::Union{Nothing,Vector{Type}}
    slots::Vector{Pair{Int,_PlateTapeSlot}}
end
_PRSites(marks, allowed, selected) =
    _PRSites(marks, allowed, selected, Pair{Int,_PlateTapeSlot}[])

const _PR_MARKS_SCOPE = :ReactiveKernels_plate_reverse_marks
_pr_marks() = get(task_local_storage(), _PR_MARKS_SCOPE, nothing)::Union{Nothing,_PRSites}
_pr_collecting_marks(f, sites::_PRSites) = task_local_storage(f, _PR_MARKS_SCOPE, sites)
_pr_collecting_marks(f, marks::Vector{_PlateRuleMark}) =
    _pr_collecting_marks(f, _PRSites(marks, nothing, nothing))

# A change of size can switch a reached site between the grouped and logical
# tapes. Their backend activity requirements differ. Record the concrete call
# signature on the selected arm of the native probe. On the same arm in the
# clean program both types are already known, so membership folds to a Bool.
struct _PRObserveSchedule
    selected::Vector{Type}
    ordinal::Int
end
@inline _pr_schedule_matches(::Type{Allowed}, ::T, ::A) where {Allowed,T,A} =
    Tuple{T,A} <: Allowed
function _pr_schedule_matches(probe::_PRObserveSchedule, ::T, ::A) where {T,A}
    old = probe.selected[probe.ordinal]
    probe.selected[probe.ordinal] = Union{old,Tuple{T,A}}
    true
end

function _pr_prepared_call(build, allowed, check_visits)
    _plate_reverse_enabled() || return build(), nothing
    if allowed === nothing && check_visits
        selected = Type[]
        probe = _PRSites(_PlateRuleMark[], nothing, selected)
        candidate = _pr_collecting_marks(build, probe)
        isempty(probe.marks) && return candidate, nothing
        call, point, contexts = candidate
        fill!(selected, Union{})
        try
            Base.invokelatest(call, point, map(DifferentiationInterface.unwrap, contexts)...)
        catch err
            err isa InterruptException && rethrow()
            @warn "The native plate-rule check threw at the exemplar. Preparing the plates inline." exception = err
            fill!(selected, Union{})
        end
        allowed = Dict{Int,Type}(i => types for (i, types) in pairs(selected) if types !== Union{})
        isempty(allowed) && return with_plate_reverse(build, false), nothing
    end
    sites = _PRSites(_PlateRuleMark[], allowed, nothing)
    _pr_collecting_marks(build, sites), sites
end

# The recorder crosses the native call as an ordinary operand. Keeping it out
# of program literals lets simultaneous preparations reuse the compiled probe.
function _pr_probe_callables(callables, sites)
    sites.selected === nothing || isempty(sites.marks) ? callables :
        (callables..., sites.selected)
end

# A successful trial returns the prepared operator. A refused trial returns
# the sites to rebuild, then the new derivative is checked from the start.
# Only a refusal reads the verifier's task-local flag.
function _pr_checking_sites(f, sites)
    (sites === nothing || isempty(sites.marks)) && return f()
    try
        task_local_storage(f, _PR_REFUSAL_SCOPE, true)
    catch err
        err isa _PRDeclined || rethrow()
        ordinal = if err.site isa _PlateRuleMark
            err.site.ordinal
        else
            i = findfirst(pair -> last(pair) === err.site, sites.slots)
            i === nothing ? nothing : first(sites.slots[i])
        end
        any(m -> m.ordinal === ordinal, sites.marks) || return Dict{Int,Type}()
        Dict{Int,Type}(m.ordinal => (sites.allowed === nothing ? Any : sites.allowed[m.ordinal])
                       for m in sites.marks if m.ordinal != ordinal)
    end
end

function _pr_verified_preparation(op, prepared, point, seeds, contexts, evaluated, sites)
    (sites === nothing || isempty(sites.marks)) && return prepared
    if evaluated === nothing
        f, args = _ad_operator_evaluation(op, prepared, point, seeds, contexts)
        Base.code_typed(f, Tuple{map(Core.Typeof, args)...})
        evaluated = try
            Base.invokelatest(f, args...)
            true
        catch err
            err isa Union{InterruptException,_PRDeclined} && rethrow()
            @warn "The derivative threw while verifying plate rules at the exemplar. Preparing the plates inline." exception = err
            false
        end
    end
    evaluated ? prepared : Dict{Int,Type}()
end
@inline function _pr_take_tape!(slot::_PlateTapeSlot, n::Int)
    tape = slot.tape
    length(tape) == n || return zeros(Float64, n)
    slot.tape = slot.empty
    fill!(tape, 0.0)
end
@inline _pr_take_tape!(::Union{Nothing,_PlateRuleMark}, n::Int) = zeros(Float64, n)
@inline _pr_return_tape!(slot::_PlateTapeSlot, tape::Vector{Float64}) = (slot.tape = tape; nothing)
@inline _pr_return_tape!(::Union{Nothing,_PlateRuleMark}, tape::Vector{Float64}) = nothing

# The operand classes a one-pass tape holds a buffer for, on types alone:
# `Float64` vectors, views of them, tuples of `Float64`, and the rows of a
# ragged group (whose projections `_pr_group_projections_cover` checks).
_pr_buffer_operand(::Type) = false
_pr_buffer_operand(::Type{Vector{Float64}}) = true
_pr_buffer_operand(::Type{<:SubArray{Float64,1,<:AbstractVector}}) = true
_pr_buffer_operand(::Type{<:Tuple{Vararg{Float64}}}) = true
_pr_buffer_operand(::Type{Vector{Vector{Float64}}}) = true
# A gathered operand's buffer has one entry per entry of the storage its
# indices address. A plate that reads a few entries of a long vector would
# clear and scale all of it on every call: such a call stays inline.
const _PR_MAX_ROOT_PER_CELL = 8

# The target of a call with operands of types `types`, as an expression in
# `rule` and `args` that `make` turns into the result (the target itself, or
# its call): the one-pass call where a schedule applies, else the program,
# which the caller runs inline. The choice is made here, on operand types and
# sizes: a call that takes the rule has only operands its adapter covers.
function _pr_target_expr(S, B, types, joint, make, one_pass = make)
    inline = make(:(getfield(rule, :program)))
    any(b -> types[b] <: _CellArray, B) && return inline
    all(s -> types[s] <: Union{Float64,Integer}, S) &&
        all(b -> _pr_buffer_operand(types[b]), B) || return inline
    slotted = !isempty(types) && types[end] === _PlateTapeSlot
    tuples = any(b -> types[b] <: Tuple, B)
    gathered = [b for b in B if _pr_gather_value(types[b])]
    rows = any(b -> !(b in gathered) && !(types[b] <: Tuple), B)
    # Cell work the backend's two sweeps would repeat pays for a tape in owned
    # storage. A buffer per row needs work the backend's own scalar rules do
    # not already share (`joint === true`). Buffers per group entry and plain
    # scalar cotangents cost little, so any shared work pays for them.
    reuse = slotted && (joint !== false || tuples)
    logical = slotted && (joint === true || tuples)
    by_entry = one_pass(:(_PlateSumRuleOnePass{$S,$B,typeof(rule),:logical}(rule)))
    isempty(gathered) && return (rows ? logical : reuse) ? by_entry : inline
    buffers = Expr(:call, :+, (:(length(_pr_gather_root(getfield(args, $b)))) for b in gathered)...)
    cells = :(length(getfield(args, $(first(gathered)))))
    if rows
        logical || return inline
        return :($buffers <= _PR_MAX_ROOT_PER_CELL * $cells ? $by_entry : $inline)
    end
    quote
        cells = $cells
        buffers = $buffers
        if ($reuse || cells >= _PR_ONE_PASS_MIN_CELLS) && buffers < cells
            $(one_pass(:(_PlateSumRuleOnePass{$S,$B,typeof(rule),true}(rule))))
        elseif $logical && buffers <= _PR_MAX_ROOT_PER_CELL * cells
            $by_entry
        else
            $inline
        end
    end
end
# Whether a call's last operand is its site record (a tape slot or a mark),
# which is no operand of the program.
_pr_sited(types) = !isempty(types) && types[end] <: _PlateRuleSite
# Whether a reverse program takes its call's site record as a last argument
# (`_PR_SITE`), from the parameter names its compiled type carries.
_pr_sited_reverse(::Type) = false
_pr_sited_reverse(::Type{<:_PlateCall{F}}) where {F} = _pr_sited_reverse(F)
_pr_sited_reverse(::Type{<:RuntimeGeneratedFunctions.RuntimeGeneratedFunction{names}}) where {names} =
    names isa Tuple && !isempty(names) && last(names) === _PR_SITE
_pr_sited_reverse(::Type{<:_PrecompileWarmFunction{F}}) where {F} = _pr_sited_reverse(F)
_pr_sited_reverse(::Type{<:_PRExpProgram{I,J,F}}) where {I,J,F} = _pr_sited_reverse(F)
# The call itself, made in each branch: a target chosen by size and then called
# would be a union of the two wrappers, boxed in the primal. A trailing site
# record is not an operand of the program: only the one-pass call takes it on.
function _pr_rule_expr(S, B, types, joint, dispatch, projections)
    sited = _pr_sited(types)
    count = length(types) - sited
    operands = Any[:(getfield(args, $i)) for i in 1:count]
    if !_pr_group_projections_cover(projections, types)
        dispatch === :_pr_rule_applies && return false
        primal = :(getfield(rule, :program))
        return dispatch === :_pr_rule_target ? primal : Expr(:call, primal, operands...)
    end
    inputs = Tuple(b for b in B if types[b] <: _PRExpCells)
    if !isempty(inputs)
        shadows = Tuple(count + 2 + findfirst(==(i), B) for i in inputs)
        shapes = Expr(:tuple, (:((args[$i].broadcast.style, args[$i].broadcast.axes)) for i in inputs)...)
        for i in inputs
            operands[i] = :(only(getfield(args, $i).broadcast.args))
        end
        sited && push!(operands, :(getfield(args, $(count + 1))))
        extra = dispatch === :_pr_rule_applies ? (:check,) : ()
        return :($dispatch(_pr_exp_rule(rule, Val($inputs), Val($shadows), $shapes),
                           ($(operands...),), $(extra...)))
    end
    dispatch === :_pr_rule_target && return _pr_target_expr(S, B, types, joint, identity)
    dispatch === :_pr_rule_applies &&
        return _pr_target_expr(S, B, types, joint, _ -> false,
                               target -> :(_pr_schedule_matches(check, $target, args)))
    call = target -> Expr(:call, target, operands...)
    one_pass = sited ?
        (target -> Expr(:call, target, operands..., :(getfield(args, $(count + 1))))) : call
    _pr_target_expr(S, B, types, joint, call, one_pass)
end
function _pr_source_checked_expr(S, B, types, joint, dispatch, projections, checked,
                                 sited_reverse::Bool = false)
    body = _pr_rule_expr(S, B, types, joint, dispatch, projections)
    checked || return body
    count = length(types) - _pr_sited(types)
    operands = Any[:(getfield(args, $i)) for i in 1:count]
    primal = :(getfield(rule, :program))
    fallback = dispatch === :_pr_rule_target ? primal :
               dispatch === :_pr_rule_applies ? false : Expr(:call, primal, operands...)
    # Source guards read only forward values. Inference over the reverse
    # must exclude the missing-source return in both value schedules. This
    # is a type query; no cells, shadows or numeric inputs are evaluated.
    shadows = fill(Nothing, length(B))
    inputs = Any[types[i] for i in 1:count]
    site = sited_reverse ? (Nothing,) : ()
    proof(want) = :(_promote_op(getfield(rule, :reverse), $(inputs...), Float64,
                                $(Val{want}), $(shadows...), $(site...)))
    quote
        if _pr_proved_tuple($(proof(true))) && _pr_proved_tuple($(proof(false)))
            $body
        else
            $fallback
        end
    end
end
@inline _pr_proved_tuple(T) = T !== Union{} && T <: Tuple
# The schedule choice and the call it makes are inlined into the rule's
# callable, as the call was before the choice grew: the backend then sees the
# one-pass call (or the program) at the rule call site itself, with the
# activity of the caller's own operands.
_pr_inlined(body) = Expr(:block, Expr(:meta, :inline), body)
@generated _pr_rule_target(rule::_PlateSumRule{S,B,P,R,J,G,C}, args::Tuple) where {S,B,P,R,J,G,C} =
    _pr_source_checked_expr(S, B, args.parameters, J, :_pr_rule_target, G, C, _pr_sited_reverse(R))
@generated _pr_rule_call(rule::_PlateSumRule{S,B,P,R,J,G,C}, args::Tuple) where {S,B,P,R,J,G,C} =
    _pr_inlined(_pr_source_checked_expr(S, B, args.parameters, J, :_pr_rule_call, G, C,
                                        _pr_sited_reverse(R)))
# Whether this call takes a one-pass schedule, from its operand types and
# sizes alone. A call that does not runs the plate inline, where the backend
# differentiates the optimized program exactly as it does without a rule.
@inline _pr_rule_applies(rule, args::Tuple) = _pr_rule_applies(rule, args, Any)
@generated _pr_rule_applies(rule::_PlateSumRule{S,B,P,R,J,G,C}, args::Tuple, check::K) where {S,B,P,R,J,G,C,K} =
    _pr_inlined(_pr_source_checked_expr(S, B, args.parameters, J, :_pr_rule_applies, G, C,
                                        _pr_sited_reverse(R)))
_pr_projections(::Type{<:_PlateSumRule{S,B,P,R,J,G}}) where {S,B,P,R,J,G} = G
@generated _pr_rule_target(rule::_PRGroupBroadcastRule{I,F}, args::Tuple) where {I,F} =
    _pr_group_broadcast_expr(I, _pr_projections(F), args.parameters, :_pr_rule_target)
@generated _pr_rule_call(rule::_PRGroupBroadcastRule{I,F}, args::Tuple) where {I,F} =
    _pr_inlined(_pr_group_broadcast_expr(I, _pr_projections(F), args.parameters, :_pr_rule_call))
@generated _pr_rule_applies(rule::_PRGroupBroadcastRule{I,F}, args::Tuple, check::K) where {I,F,K} =
    _pr_inlined(_pr_group_broadcast_expr(I, _pr_projections(F), args.parameters, :_pr_rule_applies))
# One method per operand count, each operand's type a parameter of its own:
# Julia does not otherwise specialize on a `Function` operand it only passes on
# (the program's callables), and the call would dispatch dynamically
# (`_PlateCall` has the same methods).
for count in 0:_PLATE_CALL_MAX_ARGUMENTS
    local arguments = Any[Symbol("argument", index) for index in 1:count]
    local types = Any[Symbol("Argument", index) for index in 1:count]
    local typed = Any[:($a::$T) for (a, T) in zip(arguments, types)]
    @eval @inline (rule::_PlateSumRule)($(typed...)) where {$(types...)} =
        _pr_rule_call(rule, ($(arguments...),))
    @eval @inline (rule::_PRGroupBroadcastRule)($(typed...)) where {$(types...)} =
        _pr_rule_call(rule, ($(arguments...),))
    @eval @inline (call::_PlateSumRuleOnePass)($(typed...)) where {$(types...)} =
        getfield(getfield(call, :rule), :program)($(arguments...))
    # With its site record, which the program does not take.
    count < _PLATE_CALL_MAX_ARGUMENTS &&
        @eval @inline (call::_PlateSumRuleOnePass)(
                $(typed...), site::_PlateRuleSite) where {$(types...)} =
            getfield(getfield(call, :rule), :program)($(arguments...))
end
_pr_scalar_positions(::_PlateSumRule{S}) where {S} = S
_pr_batched_positions(::_PlateSumRule{S,B}) where {S,B} = B
_pr_batched_positions(rule::_PRGroupBroadcastRule) = _pr_batched_positions(rule.expanded)

# Whether the AD path replaces eligible plate programs with generated rules:
# opt-in while the reverse is validated across the corpus. The `plate_reverse`
# preference (`"off"`, the default, or `"on"`) sets it for a project;
# `with_plate_reverse` sets it for the preparations one call makes.
const _PLATE_REVERSE_SCOPE = :ReactiveKernels_plate_reverse
function _plate_reverse_enabled()
    scoped = get(task_local_storage(), _PLATE_REVERSE_SCOPE, nothing)
    scoped === nothing || return scoped::Bool
    value = Preferences.load_preference(_PREFERENCE_UUID, "plate_reverse", "off")
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
reports static eligibility.

The generated reverse is installed by prepared operators ([`prepare_ad`](@ref)
and the samplers built on it). For native Enzyme, preparation runs a separate
primal at the exemplar to find the sites that select a generated schedule.
It then checks the derivative. Skipped sites, sites that selected inline
execution and calls the backend declined use ordinary differentiation.
Other sites keep their generated reverse for the checked schedules and operand
types. A later size that selects an unchecked layout uses ordinary AD. An exemplar
at which verification throws leaves the plates inline. The one-shot forms
(`ad_gradient(kernel, backend, ...)`) differentiate the plates inline.

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
    users = Dict{Symbol,Vector{Symbol}}()
    values = Dict{Symbol,Any}()
    for (lhs, rhs) in pairs
        for source in _pr_reads!(Set{Symbol}(), rhs, @__MODULE__, values)
            push!(get!(Vector{Symbol}, users, source), lhs)
        end
    end
    pending = collect(active)
    while !isempty(pending)
        source = pop!(pending)
        haskey(users, source) || continue
        for name in users[source]
            name in active && continue
            push!(active, name)
            push!(pending, name)
        end
    end
    active
end

# The names a body binds once to an entry of its constants table and never
# writes: bound data and values derived from data alone, which are the same
# at every call of one preparation.
function _pr_constant_names(body)
    pairs = _pr_dataflow!(Pair{Symbol,Any}[], body)
    writes = Dict{Symbol,Int}()
    for (lhs, _) in pairs
        writes[lhs] = get(writes, lhs, 0) + 1
    end
    Set{Symbol}(lhs for (lhs, rhs) in pairs if writes[lhs] == 1 &&
        Meta.isexpr(rhs, :ref, 2) && rhs.args[1] === _CONSTANTS_ARG && rhs.args[2] isa Integer)
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

function _pr_compile_sum_rule(ast, active_params, values, bound::Vector{Symbol} = Symbol[])
    rexpr, scalar, batched, joint, projections =
        _pr_program_reverse(ast, active_params, @__MODULE__, values, bound)
    params = ast.args[1].args
    S = Tuple(findfirst(==(p), params) for p in scalar)
    B = Tuple(findfirst(==(p), params) for p in batched)
    primal = _PlateCall(compile(ast))
    rev = _PlateCall(compile(rexpr))
    checked = _pr_contains_source_guard(rexpr)
    rule = _PlateSumRule{S,B,typeof(primal),typeof(rev),joint,projections,checked}(primal, rev)
    rule, rexpr, projections
end

function _pr_rule_for(call::Expr, outer_active::Set{Symbol}, kernel,
                      lazy::Set{Symbol} = Set{Symbol}(),
                      constants::Union{Nothing,Set{Symbol}} = nothing)
    program = first(call.args)::_PlateProgram
    params = Symbol[p for p in program.ast.args[1].args]
    args = call.args[2:end]
    values = Dict{Symbol,Any}()
    active_params = Symbol[]
    # The parameters that read the constants table, when the call's site
    # record can keep what is measured of them (`constants` given).
    bound = Symbol[]
    for (p, a) in zip(params, args)
        tv = _pr_table_value(a, kernel)
        if tv !== nothing
            values[p] = tv[2]
            constants !== nothing && a.args[1] === _CONSTANTS_ARG && push!(bound, p)
        elseif a isa Symbol && a in outer_active
            push!(active_params, p)
        elseif a isa Symbol && constants !== nothing && a in constants
            push!(bound, p)
        end
    end
    isempty(active_params) && return nothing, "no active operand"
    try
        rule, rexpr, projections = _pr_compile_sum_rule(program.ast, active_params, values, bound)
        positions = Tuple(i for i in _pr_batched_positions(rule) if args[i] in lazy)
        isempty(positions) && return rule, rexpr
        for i in positions
            any(p -> p.parent == i && !p.batched, projections) || _pr_fail(
                "batched operand `$(args[i])` is a lazy cell array, whose shadow the generated reverse does not write")
        end
        length(args) + 2length(positions) <= _PLATE_CALL_MAX_ARGUMENTS ||
            _pr_fail("the expanded group broadcast exceeds the plate call's operand limit")
        ast, shared = _pr_expand_group_reads(program.ast, positions)
        expanded, info, _ = _pr_compile_sum_rule(ast, [active_params; shared], values, bound)
        return _PRGroupBroadcastRule{positions,typeof(rule),typeof(expanded)}(rule, expanded), info
    catch err
        err isa _PRIneligible || rethrow()
        return nothing, err.reason
    end
end

# The AD body with each eligible plate program a generated rule and every
# other program inline, as `_inline_plate_programs` inlines them, and the
# report of each program: `(result, "rule" or the reason it falls back)`.
# `tape_slot`, when the caller owns AD caches, returns the operand expression
# of a fresh `_PlateTapeSlot` for one rule call: every rule gets one, because
# whether a call takes one pass depends on operand types and sizes that only
# the call knows, and a plate of shared scalars alone keeps its cotangents
# there too. A caller without AD caches gives each rule call a mark of the
# preparation in progress (`_pr_marks`). Either is the call's site record,
# which its preparation reads after one evaluation. A plate whose call cannot
# carry one (no preparation collects marks, or the call is at the operand
# limit) stays inline.
function _pr_inline_plate_programs(ast::Expr, kernel, active_ports; tape_slot = nothing)
    outer_active = _pr_body_activity(ast.args[2], Set{Symbol}(active_ports))
    lazy = _pr_lazy_cell_arrays(ast.args[2])
    report = Any[]
    sites = _pr_marks()
    ordinal = Ref(0)
    # A tape slot belongs to one preparation and can keep what a rule measures
    # of its constants; a mark cannot (`_pr_cached_exponents`).
    constants = tape_slot === nothing ? nothing : _pr_constant_names(ast.args[2])
    function replace(node)
        if node isa Expr && node.head === :(=) && node.args[2] isa Expr &&
                node.args[2].head === :call && first(node.args[2].args) isa _PlateProgram
            ordinal[] += 1
            if sites !== nothing && sites.selected !== nothing
                push!(sites.selected, Union{})
            end
            rule, info = _pr_rule_for(node.args[2], outer_active, kernel, lazy, constants)
            push!(report, (node.args[1], rule === nothing ? info : "rule"))
            rule === nothing && return node
            operands = node.args[2].args[2:end]
            sites !== nothing &&
                length(operands) < _PLATE_CALL_MAX_ARGUMENTS || return node
            allowed = sites.allowed === nothing ? Any : get(sites.allowed, ordinal[], Union{})
            allowed === Union{} && return node
            mark = _PlateRuleMark(ordinal[])
            push!(sites.marks, mark)
            site = tape_slot !== nothing ? tape_slot(mark) : mark
            operands = Any[operands..., site]
            called = Expr(:(=), node.args[1], Expr(:call, rule, operands...))
            # The else arm keeps the program call, which is inlined below.
            check = sites.selected === nothing ? allowed :
                Expr(:call, GlobalRef(@__MODULE__, :_PRObserveSchedule),
                     Expr(:ref, _CALLABLES_ARG, length(kernel.callables) + 1), mark.ordinal)
            condition = Expr(:call, GlobalRef(@__MODULE__, :_pr_rule_applies), rule,
                             Expr(:tuple, operands...), check)
            return Expr(:if, condition, called, node)
        elseif node isa Expr && node.head !== :quote
            return Expr(node.head, Any[replace(a) for a in node.args]...)
        end
        node
    end
    new = replace(ast)
    # The remaining programs inline, as before. A preparation that owns AD
    # caches (the non-allocating program) inlines every plate without a rule,
    # as it does with the generated reverse disabled (`_ad_na_inline_plate`).
    expand = tape_slot === nothing ? _plate_program_or_inline :
        (node, call, declared) -> _inline_plate_program(call, declared)
    _inline_plate_programs(new; expand), report
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
# generated plate reverse enabled, each eligible plate program a rule. Rules
# are installed only for a preparation that verifies them (`_pr_marks`): a
# one-shot derivative differentiates the plates inline.
Base.@nospecializeinfer function _ad_inline_plate_programs(
        @nospecialize(kernel::PreparedKernel), I)
    _plate_reverse_enabled() && _pr_marks() !== nothing && _has_plate_programs(kernel.ast) ||
        return _ad_inline_plate_programs(kernel)
    ast, _ = _pr_inline_plate_programs(kernel.ast, kernel, _pr_active_ports(kernel, I))
    native = compile(ast)
    f = kernel.f isa _ArrayFunction ? _ArrayFunction(native) : native
    ops = _pr_probe_callables(kernel.callables, _pr_marks())
    _executed_prepared_kernel(f, kernel.constants, ops, kernel.inputs,
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
operand annotations can affect the rule used by a prepared gradient. A call of
an eligible plate takes its rule only where a one-pass schedule applies to
the call's operands, and runs inline otherwise. Native Enzyme preparation
checks schedule selection and backend refusals separately for each site.
Single-plate programs are listed even when ordinary native execution inlines
them.
"""
function plate_reverse_report(kernel::Union{PreparedKernel,NonAllocatingKernel}, active = 1)
    _has_plate_programs(kernel.ast) || return Pair{Symbol,String}[]
    _, report = _pr_inline_plate_programs(kernel.ast, kernel, _pr_active_ports(kernel, active))
    Pair{Symbol,String}[Symbol(first(r)) => last(r) for r in report]
end
