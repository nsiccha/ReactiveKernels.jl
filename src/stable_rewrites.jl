# Stable algebraic rewrites of a lowered program (snag `rk-stable-algebr-e6eb8f42`;
# user decision `ReactiveKernels:snag.rk-stable-algebr-e6eb8f42/decisions/2026-11-08T04-13-58-283-17d8927`,
# option 1: on in every preparation, `simplify = false` opts out).
#
# A composition such as `log(exp(u))` is `u` only while `exp(u)` is
# representable (-745 < u < 709): outside, the value is `-Inf` or `Inf`, and
# the ordinary reverse `(1 / exp(u)) * exp(u)` is `0 * Inf = NaN` even where the
# density downstream stays finite. A positive parameter enters a model as
# `exp(u)` (RKPPL's positive transform, BRM's exp link) and is read back on the
# log scale (a law's `log(scale)`, an author's `-log(rate)`), so this is the
# natural spelling of many models. This pass computes such a composition in its
# stable form before the program is compiled; Enzyme then differentiates the
# stable form with its ordinary reverse mode. No derivative is authored.
#
# Admission rule. A rewrite of a subexpression `E` into `E'` is admitted only if
# - `E'` equals `E` as a real function wherever `E` is defined;
# - wherever `E`'s floating-point value is finite, `E'` agrees with it, and with
#   its ordinary-reverse derivative, to within `E`'s own rounding;
# - `E'` may be finite where `E` overflows, underflows or forms `0 * Inf`;
# - `E'` raises every error `E` raises (`exp(log(x))` is `x` only where `x` is
#   proven non-negative by its producer);
# - no lazy branch is evaluated eagerly and no guard moves: a rewrite reads only
#   values its site already reads or that a dominating statement computed.
#
# How. The pass walks the generated body (`_prepare_plan_body`'s AST, plate
# loops still inline; `_prepare`'s otherwise). It recognises a call by the
# *resolved identity* of its callee, never by a name: a `GlobalRef` or a
# qualified or bare name bound to a constant, a recipe's inlined source (`let`
# over its arguments), a call through the callables table read from its
# recipe's source or raw function (a law's hoisted `log(scale)`), and a call
# of one leaf arm of a cell's lazy branch read from that arm's expression
# (`observed ? log(p) : log1p(-p)`, rewritten arm by arm). An argument that is a name
# reads that name's definition when the name is assigned once and the
# definition dominates the site (the same or an enclosing block, earlier).
# A matched composition becomes a call of one of the `_rk_*_of` helpers below,
# whose methods decide by dispatch: the stable form for `Real` operands, the
# literal composition for every other type (a matrix exponential, a custom
# number), so the pass needs no type information. A statement of a pure inner
# call that nothing reads any more is removed: kept, its reverse could still
# form `0 * Inf`. Dotted calls (`log.(exp.(u))`) are rewritten elementwise.
#
# Scope. This covers the programs `prepare` and `prepare_nonallocating` compile
# (and so `prepare_ad`, the RKPPL queries and samplers), inside plate cells
# included. Not yet covered: the reactive compiler, the batched lowering and
# `jax_source`; opaque native functions are never visible to it.

# --- the setting -------------------------------------------------------------

# `simplify` (`_simplify`, `_with_simplify` and the project preference) is
# defined in `simplify_setting.jl`: the summed-plate pass reads it too, inside
# lowerings that run while this module loads.

# --- the stable forms --------------------------------------------------------
#
# Each helper takes the inner callable as its first argument and computes the
# composition as written unless its operands select a stable method. The
# fallbacks spell exactly the authored composition, so a type without a stable
# method (a matrix, an interval, a custom number) runs the literal program.

const _RwExp = Union{typeof(exp),typeof(_native_exp)}
const _RwNonNegative = Union{typeof(exp),typeof(_native_exp),typeof(abs),typeof(abs2),
                             typeof(LogExpFunctions.logistic)}

# The float type the authored arithmetic over these operands produces.
_rw_float(xs...) = float(promote_type(map(typeof, xs)...))

# The inner callable of a dotted helper call, `_rk_log_of.(exp, x)`. Base
# broadcasts a function as a mutable `RefValue`, which a consumer that is not
# inlined, such as a non-allocating step's `apply!!`, allocates on every call;
# this immutable `Ref` is the same zero-dimensional operand.
struct _RwFn{F} <: Ref{F}
    f::F
end
Base.getindex(x::_RwFn) = x.f

# `log(f(x))`.
@_record_source @inline _rk_log_of(f::F, x) where {F} = log(f(x))
@_record_source @inline _rk_log_of(::_RwExp, x::Real) = float(x)
@_record_source @inline _rk_log_of(::typeof(LogExpFunctions.logistic), x::Real) =
    LogExpFunctions.loglogistic(x)

# `log1p(f(x))`.
@_record_source @inline _rk_log1p_of(f::F, x) where {F} = log1p(f(x))
@_record_source @inline _rk_log1p_of(::_RwExp, x::Real) = LogExpFunctions.log1pexp(float(x))

# `log1p(-f(x))`.
@_record_source @inline _rk_log1p_neg_of(f::F, x) where {F} = log1p(-f(x))
@_record_source @inline _rk_log1p_neg_of(::typeof(LogExpFunctions.logistic), x::Real) =
    LogExpFunctions.log1mlogistic(x)
@_record_source @inline _rk_log1p_neg_of(::_RwExp, x::Real) = LogExpFunctions.log1mexp(float(x))

# `log(c - f(x))` with `c` a literal one.
@_record_source @inline _rk_log_one_minus_of(c, f::F, x) where {F} = log(c - f(x))
@_record_source @inline _rk_log_one_minus_of(c::Real, ::typeof(LogExpFunctions.logistic), x::Real) =
    convert(_rw_float(c, LogExpFunctions.logistic(x)), LogExpFunctions.log1mlogistic(x))
@_record_source @inline _rk_log_one_minus_of(c::Real, ::_RwExp, x::Real) =
    LogExpFunctions.log1mexp(convert(_rw_float(c, x), x))

# `log(c + y)` with `c` a literal one.
@_record_source @inline _rk_log_one_plus(c, y) = log(c + y)
@_record_source @inline _rk_log_one_plus(c::Real, y::Real) = log1p(convert(_rw_float(c, y), y))

# `log(c + f(x))` with `c` a literal one.
@_record_source @inline _rk_log_one_plus_of(c, f::F, x) where {F} = log(c + f(x))
@_record_source @inline _rk_log_one_plus_of(c::Real, ::_RwExp, x::Real) =
    LogExpFunctions.log1pexp(convert(_rw_float(c, x), x))

# `log(f(a) + g(b))`. The stable form is `_tie_exact_logaddexp`, whose
# derivative at a tie is the literal one, (1/2, 1/2).
@_record_source @inline _rk_log_add_of(f::F, a, g::G, b) where {F,G} = log(f(a) + g(b))
@_record_source @inline function _rk_log_add_of(::_RwExp, a::Real, ::_RwExp, b::Real)
    T = _rw_float(a, b)
    _tie_exact_logaddexp(convert(T, a), convert(T, b))
end

# `log(sum(f.(x)))`, the broadcast materialized as the native lowering does.
@_record_source @inline _rk_log_sum_of(f::F, x) where {F} =
    log(sum(_native_broadcast_materialize(_native_broadcasted(f, x))))
@_record_source @inline _rk_log_sum_of(::_RwExp, x::AbstractArray{<:Real}) = _rk_logsumexp(x)

# The log-sum-exp of the entries of `x`, shifted by their maximum. Its value is
# `log(sum(exp.(x)))` to within rounding, `-Inf` for no entries and the
# maximum when that is not finite, as the literal gives. Its derivative is the
# softmax whichever entry `maximum` returns at a tie: the shift's two terms
# cancel exactly.
function _rk_logsumexp(x::AbstractArray{<:Real})
    T = float(eltype(x))
    isempty(x) && return T(-Inf)
    m = convert(T, maximum(x))
    isfinite(m) || return m
    s = zero(T)
    @inbounds for v in x
        s += exp(v - m)
    end
    m + log(s)
end

# `logit(f(x))`.
@_record_source @inline _rk_logit_of(f::F, x) where {F} = LogExpFunctions.logit(f(x))
@_record_source @inline _rk_logit_of(::typeof(LogExpFunctions.logistic), x::Real) = float(x)

# `log(p) - log1p(-p)` with `p = f(x)`.
@_record_source @inline _rk_log_odds_of(f::F, x) where {F} = (p = f(x); log(p) - log1p(-p))
@_record_source @inline _rk_log_odds_of(::typeof(LogExpFunctions.logistic), x::Real) = float(x)

# `exp(log(p))` where `p`'s producer is `f`: `p` itself when `f` gives a
# non-negative value.
@_record_source @inline _rk_exp_log_of(::F, p) where {F} = exp(log(p))
@_record_source @inline _rk_exp_log_of(::_RwNonNegative, p::Real) = float(p)

# `f(x) - c` with `c` a literal one.
@_record_source @inline _rk_minus_one_of(f::F, x, c) where {F} = f(x) - c
@_record_source @inline _rk_minus_one_of(::_RwExp, x::Real, c::Real) = expm1(convert(_rw_float(c, x), x))

# `inv(f(x))`.
@_record_source @inline _rk_inv_of(f::F, x) where {F} = inv(f(x))
@_record_source @inline _rk_inv_of(::_RwExp, x::Real) = exp(-float(x))

# `y / f(x)`.
@_record_source @inline _rk_div_by_of(y, f::F, x) where {F} = y / f(x)
@_record_source @inline _rk_div_by_of(y::Real, ::_RwExp, x::Real) = y * exp(-float(x))

# `f(x)^k`: a literal integer power as Julia lowers it (`literal_pow`), any
# other power as a call of `^`.
@_record_source @inline _rk_pow_of(f::F, x, ::Val{k}) where {F,k} = Base.literal_pow(^, f(x), Val(k))
@_record_source @inline _rk_pow_of(::_RwExp, x::Real, ::Val{k}) where {k} = exp(k * float(x))
@_record_source @inline _rk_pow_of(f::F, x, k) where {F} = f(x)^k
@_record_source @inline _rk_pow_of(::_RwExp, x::Real, k::Real) = exp(k * float(x))

# `sqrt(f(x))`.
@_record_source @inline _rk_sqrt_of(f::F, x) where {F} = sqrt(f(x))
@_record_source @inline _rk_sqrt_of(::_RwExp, x::Real) = exp(float(x) / 2)

# The logistic function spelled out, `c1 / (c2 + f(-u))` with literal ones (an
# RKPPL unit transform's `1 / (1 + exp(-u))`): `log` of it, and `log1p` of its
# negation.
@_record_source @inline _rk_log_sigmoid(c1, c2, f::F, u) where {F} = log(c1 / (c2 + f(-u)))
@_record_source @inline _rk_log_sigmoid(c1::Real, c2::Real, ::_RwExp, u::Real) =
    convert(_rw_float(c1, c2, u), LogExpFunctions.loglogistic(u))
@_record_source @inline _rk_log1m_sigmoid(c1, c2, f::F, u) where {F} = log1p(-(c1 / (c2 + f(-u))))
@_record_source @inline _rk_log1m_sigmoid(c1::Real, c2::Real, ::_RwExp, u::Real) =
    convert(_rw_float(c1, c2, u), LogExpFunctions.log1mlogistic(u))

# Every helper the pass emits, by identity, for code that reads a rewritten
# program: the generated plate reverse expands each call into the method its
# operands select (`_pr_expand_stable_helper`, `src/plate_reverse.jl`).
# LOCKSTEP: each `_rw_emit(…, :_rk_…)` helper name below.
# The helpers record their sources (`@_record_source`), so `jax_source`
# translates the same stable forms native execution runs.
const _RK_STABLE_HELPERS = (_rk_log_of, _rk_log1p_of, _rk_log1p_neg_of, _rk_log_one_minus_of,
    _rk_log_one_plus, _rk_log_one_plus_of, _rk_log_add_of, _rk_log_sum_of, _rk_logit_of,
    _rk_log_odds_of, _rk_exp_log_of, _rk_minus_one_of, _rk_inv_of, _rk_div_by_of, _rk_pow_of,
    _rk_sqrt_of, _rk_log_sigmoid, _rk_log1m_sigmoid)

# --- recognising calls --------------------------------------------------------

# A call as the pass reads it: the callee's function object, its argument
# expressions, and whether it is a dotted call.
struct _RwView
    f::Any
    args::Vector{Any}
    broadcast::Bool
end

struct _RwContext
    recipes::Tuple
    counts::Dict{Symbol,Int}    # assignments of each name in the program
    params::Set{Symbol}         # parameters the body never assigns
    reads::Base.RefValue{Int}   # moved cell reads so far (`_rw_fresh`)
end

# The constant a callee expression names, or `nothing`. A binding that is not
# constant can change between preparation and a call, so it never matches.
_rw_binding(m::Module, name::Symbol) =
    isdefined(m, name) && isconst(m, name) ? getglobal(m, name) : nothing
function _rw_resolve(f, mod::Module)
    f isa GlobalRef && return _rw_binding(f.mod, f.name)
    f isa Symbol && return _rw_binding(mod, f)
    f isa Function && return f
    if f isa Expr && f.head === :. && length(f.args) == 2 && f.args[2] isa QuoteNode
        m = f.args[1] isa Module ? f.args[1] : _rw_resolve(f.args[1], mod)
        name = f.args[2].value
        m isa Module && name isa Symbol && return _rw_binding(m, name)
    end
    nothing
end

# The functions whose identity the rules match, and the references the
# rewritten program calls them by.
function _rw_ref(f)
    f === exp && return GlobalRef(Base, :exp)
    f === _native_exp && return GlobalRef(@__MODULE__, :_native_exp)
    f === LogExpFunctions.logistic && return GlobalRef(LogExpFunctions, :logistic)
    f === abs && return GlobalRef(Base, :abs)
    f === abs2 && return GlobalRef(Base, :abs2)
    nothing
end
_rw_is_exp(f) = f === exp || f === _native_exp
_rw_helper(name::Symbol) = GlobalRef(@__MODULE__, name)

# A literal one, as authored source spells it.
_rw_one(x) = x isa Union{Int,Float64,Float32,Float16} && isone(x)

# A value the site can read: a literal, a parameter the body never assigns, a
# name assigned once by a statement that dominates the site, or a plate cell's
# read of its argument (`_rw_read_parts`) over such names.
_rw_available(x, ctx::_RwContext, env) =
    x isa Number || (x isa Symbol && (haskey(env, x) || x in ctx.params)) ||
    _rw_pure_read(x, ctx, env)

# A plate cell reads its argument in place, as one expression per use:
#
#     begin
#         $(Expr(:inbounds, true))
#         local v = Base.Broadcast._broadcast_getindex(prepared, index)
#         $(Expr(:inbounds, :pop))
#         v
#     end
#
# Its local and its call, or `nothing` for any other expression.
function _rw_read_parts(x)
    x isa Expr && x.head === :block || return nothing
    statements = Any[s for s in x.args if !(s isa LineNumberNode)]
    length(statements) == 4 || return nothing
    statements[1] == Expr(:inbounds, true) && statements[3] == Expr(:inbounds, :pop) ||
        return nothing
    declaration = statements[2]
    declaration isa Expr && declaration.head === :local && length(declaration.args) == 1 ||
        return nothing
    assignment = declaration.args[1]
    assignment isa Expr && assignment.head === :(=) && assignment.args[1] isa Symbol ||
        return nothing
    statements[4] === assignment.args[1] || return nothing
    call = assignment.args[2]
    call isa Expr && call.head === :call && !isempty(call.args) || return nothing
    (assignment.args[1], call)
end
# A pure read: of an array and an index the site can read.
function _rw_pure_read(x, ctx::_RwContext, env)
    parts = _rw_read_parts(x)
    parts === nothing && return false
    call = parts[2]
    _rw_resolve(call.args[1], @__MODULE__) === Base.Broadcast._broadcast_getindex &&
        all(a -> a isa Number || (a isa Symbol && (haskey(env, a) || a in ctx.params)),
            call.args[2:end])
end
# An operand as the rewritten call spells it: a read moved out of its statement
# declares a local of its own, never one its statement still declares. The
# local is numbered in the walk's order, never minted with `gensym`, so one
# plan keeps one program (`inline_sources.jl`, `_inline_local_name`).
function _rw_fresh(ctx::_RwContext, x)
    parts = _rw_read_parts(x)
    parts === nothing && return x
    name = Symbol("#stable_read#", ctx.reads[] += 1)
    Expr(:block, Expr(:inbounds, true), Expr(:local, Expr(:(=), name, parts[2])),
         Expr(:inbounds, :pop), name)
end

# `ex` with the names of `subst` replaced.
function _rw_substitute(ex, subst::Dict{Symbol,Any})
    ex isa Symbol && return get(subst, ex, ex)
    ex isa Expr || return ex
    ex.head in (:quote, :inert, :meta) && return ex
    Expr(ex.head, Any[_rw_substitute(a, subst) for a in ex.args]...)
end

# An inlined source, `let <param> = <arg>, … ; <call> end`, as the call over the
# arguments, when every argument is available here and the body is one call.
function _rw_peel_let(ex::Expr, ctx::_RwContext, env)
    length(ex.args) == 2 || return nothing
    bindings, body = ex.args
    pairs = bindings isa Expr && bindings.head === :block ? bindings.args : Any[bindings]
    subst = Dict{Symbol,Any}()
    for b in pairs
        b isa LineNumberNode && continue
        b isa Expr && b.head === :(=) && length(b.args) == 2 && b.args[1] isa Symbol ||
            return nothing
        _rw_available(b.args[2], ctx, env) || return nothing
        subst[b.args[1]] = b.args[2]
    end
    statements = body isa Expr && body.head === :block ?
        Any[s for s in body.args if !(s isa LineNumberNode)] : Any[body]
    length(statements) == 1 || return nothing
    _rw_substitute(only(statements), subst)
end

# A call through the callables table, `(__callables__[k])(args...)`, read from
# recipe `k`'s source when that source is one call of its parameters in order,
# with no captures (a law's hoisted `log_scale = log(scale)`).
function _rw_table_view(ctx::_RwContext, k::Int, args::Vector{Any})
    1 <= k <= length(ctx.recipes) || return nothing
    r = ctx.recipes[k]
    r isa Recipe || return nothing
    op = r.op
    # An exact bare call keeps the function itself in the table (a law's
    # `log_scale = log(scale)`): the table call is that function's call.
    op isa Function && !(op isa _KernelSource) && length(args) == length(r.inputs) &&
        return _RwView(op, args, false)
    op isa _KernelSourceOp || return nothing
    getfield(op, :form) === :fused || return nothing
    f = getfield(op, :f)
    f isa _KernelSource || return nothing
    isempty(_source_captures(f)) && _source_native(f) === nothing || return nothing
    mod = getfield(op, :mod)
    mod isa Module || return nothing
    source = r.source
    source isa Expr && source.head === :call || return nothing
    length(args) == length(r.inputs) == length(source.args) - 1 || return nothing
    names = _recipe_source_names(r)
    length(names) == length(args) || return nothing
    all(i -> source.args[i+1] === names[i], eachindex(names)) || return nothing
    g = _rw_resolve(source.args[1], mod)
    g === nothing ? nothing : _RwView(g, args, false)
end

# `ex` with each bare callee bound to a constant of `mod` replaced by its
# object, so a source's names read as they resolve where it was authored.
function _rw_qualify(ex, mod::Module)
    ex isa Expr || return ex
    ex.head in (:quote, :inert, :meta) && return ex
    args = Any[_rw_qualify(a, mod) for a in ex.args]
    if ex.head === :call && !isempty(args) && args[1] isa Symbol
        f = _rw_binding(mod, args[1])
        f === nothing || (args[1] = f)
    end
    Expr(ex.head, args...)
end

# One leaf arm of a lazy branch in a plate cell, called through the table as
# `(getfield(__callables__[k], :then_arm))(args...)`: recipe `k`'s
# `_KernelBranch` records the recipe arguments each arm reads (its `TI`/`EI`
# positions), and the arm is the closure of that arm's expression over them.
# The arm's call over the table call's arguments, or `nothing`.
function _rw_arm_view(ctx::_RwContext, callee, args::Vector{Any}, env)
    callee isa Expr && callee.head === :call && length(callee.args) == 3 || return nothing
    _rw_resolve(callee.args[1], @__MODULE__) === getfield || return nothing
    k = _callable_slot(callee.args[2])
    k === nothing && return nothing
    which = callee.args[3]
    which isa QuoteNode && which.value in (:then_arm, :else_arm) || return nothing
    1 <= k <= length(ctx.recipes) || return nothing
    r = ctx.recipes[k]
    r isa Recipe && r.op isa _KernelSourceOp || return nothing
    op = r.op
    branch = getfield(op, :f)
    branch isa _KernelBranch || return nothing
    mod = getfield(op, :mod)
    mod isa Module || return nothing
    source = r.source
    source isa Expr && source.head === :if && length(source.args) == 3 || return nothing
    arm = source.args[which.value === :then_arm ? 2 : 3]
    if arm isa Expr && arm.head === :block
        statements = Any[s for s in arm.args if !(s isa LineNumberNode)]
        length(statements) == 1 || return nothing
        arm = only(statements)
    end
    arm isa Expr && arm.head === :call || return nothing
    positions = (typeof(branch)::DataType).parameters[which.value === :then_arm ? 2 : 3]
    positions isa Tuple && all(p -> p isa Int, positions) || return nothing
    names = _recipe_source_names(r)
    length(names) == length(r.inputs) && length(positions) == length(args) &&
        all(p -> 1 <= p <= length(names), positions) || return nothing
    subst = Dict{Symbol,Any}(names[p] => args[i] for (i, p) in enumerate(positions))
    _rw_view(_rw_substitute(_rw_qualify(arm, mod), subst), ctx, env)
end

# A dotted call as the native lowering and the non-allocating decomposition spell it.
_rw_is_broadcasted(f) = f === _native_broadcasted || f === Base.Broadcast.broadcasted
# `__cache_apply__(__caches__[j], __callables__[j], bc)` whose step `j` only
# materializes `bc` (`_step_call`, `_MaterializeStep`).
function _rw_materialize_step(ex::Expr, ctx::_RwContext)
    ex.head === :call && length(ex.args) == 4 && ex.args[1] === _CACHE_APPLY_ARG || return false
    k = _callable_slot(ex.args[3])
    k !== nothing && 1 <= k <= length(ctx.recipes) && ctx.recipes[k] isa _MaterializeStep
end

# The call `ex` makes, or `nothing`.
function _rw_view(ex, ctx::_RwContext, env)
    ex isa Expr || return nothing
    if ex.head === :let
        peeled = _rw_peel_let(ex, ctx, env)
        return peeled === nothing ? nothing : _rw_view(peeled, ctx, env)
    end
    ex.head === :call && !isempty(ex.args) || return nothing
    callee = ex.args[1]
    args = Any[ex.args[2:end]...]
    slot = _callable_slot(callee)
    slot === nothing || return _rw_table_view(ctx, slot, args)
    # A broadcast a non-allocating step materializes into its buffer.
    _rw_materialize_step(ex, ctx) &&
        return (inner = _rw_view(last(args), ctx, env);
                inner !== nothing && inner.broadcast ? inner : nothing)
    f = _rw_resolve(callee, @__MODULE__)
    f === nothing && return nothing
    if f === _native_broadcast_materialize && length(args) == 1
        inner = _rw_view(args[1], ctx, env)
        return inner !== nothing && inner.broadcast ? inner : nothing
    elseif _rw_is_broadcasted(f) && !isempty(args)
        g = _rw_resolve(args[1], @__MODULE__)
        return g === nothing ? nothing : _RwView(g, Any[args[2:end]...], true)
    end
    _RwView(f, args, false)
end

# The call an argument makes: inline, or through the definition of the name it
# is. An inline call's operands move with it; a definition's operands must be
# readable at the site (`movable`).
function _rw_operand(a, ctx::_RwContext, env)
    # A name bound to another name (an inlined source's `let var"#x#opk" = x`)
    # reads that name's definition.
    hops = 0
    while a isa Symbol
        haskey(env, a) || return (nothing, false)
        value = env[a]
        value isa Symbol || return (_rw_view(value, ctx, env), true)
        a = value
        (hops += 1) > 64 && return (nothing, false)
    end
    (_rw_view(a, ctx, env), false)
end
_rw_movable(x, from_definition::Bool, ctx, env) =
    !from_definition || _rw_available(x, ctx, env)

# A dotted outer call may read a scalar or dotted inner call; a scalar outer
# call reads only a scalar one (`log` of a vector is not elementwise).
_rw_fits(outer::_RwView, inner::_RwView) = outer.broadcast || !inner.broadcast

# The rewritten call, in the outer call's form.
_rw_emit(ctx::_RwContext, outer::_RwView, helper::Symbol, args...) = outer.broadcast ?
    Expr(:call, GlobalRef(@__MODULE__, :_native_broadcasted), _rw_helper(helper),
         map(a -> _rw_dotted_operand(_rw_fresh(ctx, a)), args)...) :
    Expr(:call, _rw_helper(helper), map(a -> _rw_fresh(ctx, a), args)...)
# A function operand (`_rw_ref`) of a dotted helper call as an immutable `_RwFn`.
_rw_dotted_operand(a) = a isa GlobalRef && _rw_resolve(a, @__MODULE__) isa Function ?
    Expr(:call, GlobalRef(@__MODULE__, :_RwFn), a) : a

# The inner call of one operand when it is `f(x)` with `f` among `fs` and `x`
# readable at the site: `(view, x)`, else `nothing`.
function _rw_unary(a, ctx, env, outer, pred)
    inner, def = _rw_operand(a, ctx, env)
    inner === nothing && return nothing
    _rw_fits(outer, inner) && length(inner.args) == 1 && pred(inner.f) || return nothing
    x = only(inner.args)
    _rw_movable(x, def, ctx, env) || return nothing
    (inner, x)
end

# `c1 / (c2 + f(-u))` with literal ones and `f` an exponential, as `(f, c1, c2, u)`.
function _rw_sigmoid(a, ctx, env, outer)
    inner, def = _rw_operand(a, ctx, env)
    inner === nothing && return nothing
    _rw_fits(outer, inner) && inner.f === (/) && length(inner.args) == 2 &&
        _rw_one(inner.args[1]) || return nothing
    den = _rw_view(inner.args[2], ctx, env)
    den !== nothing && den.f === (+) && length(den.args) == 2 && !den.broadcast ||
        return nothing
    c2, e = _rw_one(den.args[1]) ? (den.args[1], den.args[2]) :
            _rw_one(den.args[2]) ? (den.args[2], den.args[1]) : (nothing, nothing)
    c2 === nothing && return nothing
    ev = _rw_view(e, ctx, env)
    ev !== nothing && _rw_is_exp(ev.f) && length(ev.args) == 1 && !ev.broadcast ||
        return nothing
    neg = _rw_view(only(ev.args), ctx, env)
    neg !== nothing && neg.f === (-) && length(neg.args) == 1 && !neg.broadcast ||
        return nothing
    u = only(neg.args)
    _rw_movable(u, def, ctx, env) || return nothing
    (ev.f, inner.args[1], c2, u)
end

# --- the rules -----------------------------------------------------------------

_rw_logistic(f) = f === LogExpFunctions.logistic
_rw_exp_or_logistic(f) = _rw_is_exp(f) || _rw_logistic(f)

# `log(…)`.
function _rw_rule_log(outer::_RwView, ctx, env)
    length(outer.args) == 1 || return nothing
    a = only(outer.args)
    # log(sum(exp.(x)))
    if !outer.broadcast
        inner, def = _rw_operand(a, ctx, env)
        if inner !== nothing && !inner.broadcast && inner.f === sum && length(inner.args) == 1
            hit = _rw_unary(only(inner.args), ctx, env, _RwView(sum, Any[], true), _rw_is_exp)
            if hit !== nothing && hit[1].broadcast && _rw_movable(hit[2], def, ctx, env)
                return _rw_emit(ctx, outer, :_rk_log_sum_of, _rw_ref(hit[1].f), hit[2])
            end
        end
    end
    inner, def = _rw_operand(a, ctx, env)
    inner === nothing && return nothing
    _rw_fits(outer, inner) || return nothing
    if inner.f === (+) && length(inner.args) == 2
        p, q = inner.args
        # log(exp(a) + exp(b))
        hp = _rw_unary(p, ctx, env, outer, _rw_is_exp)
        hq = _rw_unary(q, ctx, env, outer, _rw_is_exp)
        if hp !== nothing && hq !== nothing && _rw_movable(hp[2], def, ctx, env) &&
           _rw_movable(hq[2], def, ctx, env)
            return _rw_emit(ctx, outer, :_rk_log_add_of, _rw_ref(hp[1].f), hp[2],
                            _rw_ref(hq[1].f), hq[2])
        end
        # log(1 + exp(x)), log(1 + y)
        c, y = _rw_one(p) ? (p, q) : _rw_one(q) ? (q, p) : (nothing, nothing)
        c === nothing && return nothing
        hy = _rw_unary(y, ctx, env, outer, _rw_is_exp)
        if hy !== nothing && _rw_movable(hy[2], def, ctx, env)
            return _rw_emit(ctx, outer, :_rk_log_one_plus_of, c, _rw_ref(hy[1].f), hy[2])
        end
        _rw_movable(y, def, ctx, env) || return nothing
        return _rw_emit(ctx, outer, :_rk_log_one_plus, c, y)
    elseif inner.f === (-) && length(inner.args) == 2 && _rw_one(inner.args[1])
        # log(1 - logistic(x)), log(1 - exp(x))
        hy = _rw_unary(inner.args[2], ctx, env, outer, _rw_exp_or_logistic)
        hy !== nothing && _rw_movable(hy[2], def, ctx, env) || return nothing
        return _rw_emit(ctx, outer, :_rk_log_one_minus_of, inner.args[1], _rw_ref(hy[1].f), hy[2])
    elseif inner.f === (/) && length(inner.args) == 2
        # log(1 / (1 + exp(-u)))
        s = _rw_sigmoid(a, ctx, env, outer)
        s === nothing && return nothing
        return _rw_emit(ctx, outer, :_rk_log_sigmoid, s[2], s[3], _rw_ref(s[1]), s[4])
    elseif length(inner.args) == 1 && _rw_exp_or_logistic(inner.f)
        x = only(inner.args)
        _rw_movable(x, def, ctx, env) || return nothing
        return _rw_emit(ctx, outer, :_rk_log_of, _rw_ref(inner.f), x)
    end
    nothing
end

# `log1p(…)`.
function _rw_rule_log1p(outer::_RwView, ctx, env)
    length(outer.args) == 1 || return nothing
    a = only(outer.args)
    hit = _rw_unary(a, ctx, env, outer, _rw_is_exp)
    hit === nothing || return _rw_emit(ctx, outer, :_rk_log1p_of, _rw_ref(hit[1].f), hit[2])
    neg, def = _rw_operand(a, ctx, env)
    neg !== nothing && _rw_fits(outer, neg) && neg.f === (-) && length(neg.args) == 1 ||
        return nothing
    b = only(neg.args)
    # log1p(-(1 / (1 + exp(-u))))
    s = _rw_sigmoid(b, ctx, env, outer)
    if s !== nothing && _rw_movable(s[4], def, ctx, env)
        return _rw_emit(ctx, outer, :_rk_log1m_sigmoid, s[2], s[3], _rw_ref(s[1]), s[4])
    end
    # log1p(-logistic(x)), log1p(-exp(x))
    hit = _rw_unary(b, ctx, env, outer, _rw_exp_or_logistic)
    hit !== nothing && _rw_movable(hit[2], def, ctx, env) || return nothing
    _rw_emit(ctx, outer, :_rk_log1p_neg_of, _rw_ref(hit[1].f), hit[2])
end

# `logit(…)`.
function _rw_rule_logit(outer::_RwView, ctx, env)
    length(outer.args) == 1 || return nothing
    hit = _rw_unary(only(outer.args), ctx, env, outer, _rw_logistic)
    hit === nothing ? nothing : _rw_emit(ctx, outer, :_rk_logit_of, _rw_ref(hit[1].f), hit[2])
end

# `exp(log(p))` with `p` from a non-negative producer.
function _rw_rule_exp(outer::_RwView, ctx, env)
    length(outer.args) == 1 || return nothing
    hit = _rw_unary(only(outer.args), ctx, env, outer, f -> f === log)
    hit === nothing && return nothing
    p = hit[2]
    producer, _ = _rw_operand(p, ctx, env)
    producer !== nothing && _rw_fits(outer, producer) && _rw_ref(producer.f) !== nothing &&
        producer.f isa _RwNonNegative || return nothing
    _rw_emit(ctx, outer, :_rk_exp_log_of, _rw_ref(producer.f), p)
end

# `a - b`: `exp(x) - 1`, and the log-odds `log(p) - log1p(-p)` of `p = logistic(x)`.
function _rw_rule_minus(outer::_RwView, ctx, env)
    length(outer.args) == 2 || return nothing
    a, b = outer.args
    if _rw_one(b)
        hit = _rw_unary(a, ctx, env, outer, _rw_is_exp)
        hit === nothing || return _rw_emit(ctx, outer, :_rk_minus_one_of, _rw_ref(hit[1].f), hit[2], b)
    end
    la = _rw_view(a, ctx, env)
    lb = _rw_view(b, ctx, env)
    la !== nothing && lb !== nothing && la.f === log && lb.f === log1p &&
        length(la.args) == 1 && length(lb.args) == 1 && _rw_fits(outer, la) &&
        _rw_fits(outer, lb) || return nothing
    p = only(la.args)
    neg = _rw_view(only(lb.args), ctx, env)
    neg !== nothing && neg.f === (-) && length(neg.args) == 1 && only(neg.args) == p ||
        return nothing
    hit = _rw_unary(p, ctx, env, outer, _rw_logistic)
    hit === nothing ? nothing : _rw_emit(ctx, outer, :_rk_log_odds_of, _rw_ref(hit[1].f), hit[2])
end

# `inv(exp(x))`, `y / exp(x)`, `exp(x)^k`, `sqrt(exp(x))`.
function _rw_rule_inv(outer::_RwView, ctx, env)
    length(outer.args) == 1 || return nothing
    hit = _rw_unary(only(outer.args), ctx, env, outer, _rw_is_exp)
    hit === nothing ? nothing : _rw_emit(ctx, outer, :_rk_inv_of, _rw_ref(hit[1].f), hit[2])
end
function _rw_rule_div(outer::_RwView, ctx, env)
    length(outer.args) == 2 || return nothing
    hit = _rw_unary(outer.args[2], ctx, env, outer, _rw_is_exp)
    hit === nothing ? nothing :
        _rw_emit(ctx, outer, :_rk_div_by_of, outer.args[1], _rw_ref(hit[1].f), hit[2])
end
function _rw_rule_pow(outer::_RwView, ctx, env)
    length(outer.args) == 2 || return nothing
    k = outer.args[2]
    k isa Union{Int,Float64} || return nothing
    hit = _rw_unary(outer.args[1], ctx, env, outer, _rw_is_exp)
    hit === nothing && return nothing
    power = k isa Int ? Expr(:call, GlobalRef(Base, :Val), k) : k
    _rw_emit(ctx, outer, :_rk_pow_of, _rw_ref(hit[1].f), hit[2], power)
end
function _rw_rule_sqrt(outer::_RwView, ctx, env)
    length(outer.args) == 1 || return nothing
    hit = _rw_unary(only(outer.args), ctx, env, outer, _rw_is_exp)
    hit === nothing ? nothing : _rw_emit(ctx, outer, :_rk_sqrt_of, _rw_ref(hit[1].f), hit[2])
end

# The rewritten call `ex`, or `nothing` when no rule applies. A dotted call is
# matched at its lazy `_native_broadcasted` node, a plain one at its call.
function _rw_rules(ex::Expr, ctx::_RwContext, env)
    callee = ex.args[1]
    _callable_slot(callee) === nothing || return nothing
    f = _rw_resolve(callee, @__MODULE__)
    f === nothing && return nothing
    f === _native_broadcast_materialize && return nothing
    outer = _rw_view(ex, ctx, env)
    outer === nothing && return nothing
    g = outer.f
    g === log && return _rw_rule_log(outer, ctx, env)
    g === log1p && return _rw_rule_log1p(outer, ctx, env)
    g === LogExpFunctions.logit && return _rw_rule_logit(outer, ctx, env)
    g === exp && return _rw_rule_exp(outer, ctx, env)
    g === _native_exp && return _rw_rule_exp(outer, ctx, env)
    g === (-) && return _rw_rule_minus(outer, ctx, env)
    g === inv && return _rw_rule_inv(outer, ctx, env)
    g === (/) && return _rw_rule_div(outer, ctx, env)
    g === (^) && return _rw_rule_pow(outer, ctx, env)
    g === sqrt && return _rw_rule_sqrt(outer, ctx, env)
    nothing
end

# A call through the callables table whose recipe is one source call (a law's
# hoisted `log(scale)`): matched as that call over the table call's arguments.
function _rw_table_rules(ex::Expr, ctx::_RwContext, env)
    slot = _callable_slot(ex.args[1])
    slot === nothing && return nothing
    view = _rw_table_view(ctx, slot, Any[ex.args[2:end]...])
    view === nothing && return nothing
    _rw_resolve_view_rule(view, ctx, env)
end
function _rw_resolve_view_rule(outer::_RwView, ctx, env)
    g = outer.f
    g === log && return _rw_rule_log(outer, ctx, env)
    g === log1p && return _rw_rule_log1p(outer, ctx, env)
    g === LogExpFunctions.logit && return _rw_rule_logit(outer, ctx, env)
    g === (-) && return _rw_rule_minus(outer, ctx, env)
    g === inv && return _rw_rule_inv(outer, ctx, env)
    g === (/) && return _rw_rule_div(outer, ctx, env)
    g === sqrt && return _rw_rule_sqrt(outer, ctx, env)
    nothing
end

# --- the walk -----------------------------------------------------------------

const _RW_OPAQUE = (:quote, :inert, :meta, :inbounds, :boundscheck, :loopinfo,
                    :simdloop, :gc_preserve_begin, :gc_preserve_end, :line)
const _RW_CLOSURES = (:->, :function, :do, :generator, :comprehension,
                      :typed_comprehension)

# The name and value of an assignment statement, or `nothing`.
function _rw_assignment(st)
    st isa Expr || return nothing
    if st.head === :(=) && length(st.args) == 2 && st.args[1] isa Symbol
        return (st.args[1], st.args[2])
    elseif st.head === :local && length(st.args) == 1
        inner = st.args[1]
        inner isa Expr && inner.head === :(=) && length(inner.args) == 2 &&
            inner.args[1] isa Symbol && return (inner.args[1], inner.args[2])
    end
    nothing
end

# Assignments of each name anywhere in `ex`; a loop's iteration variable and
# a name a closure binds count as repeated, so neither is ever a definition.
function _rw_count!(counts::Dict{Symbol,Int}, ex)
    ex isa Expr || return counts
    ex.head in _RW_OPAQUE && return counts
    if ex.head === :for
        spec = ex.args[1]
        for s in (spec isa Expr && spec.head === :block ? spec.args : Any[spec])
            s isa Expr && s.head === :(=) && s.args[1] isa Symbol &&
                (counts[s.args[1]] = get(counts, s.args[1], 0) + 2)
        end
        foreach(a -> _rw_count!(counts, a), ex.args[2:end])
        return counts
    end
    if ex.head in _RW_CLOSURES
        # A closure can assign a name of the enclosing body (a boxed capture):
        # every name assigned inside it is never a definition.
        _rw_closure_assigned!(counts, ex)
        return counts
    end
    if ex.head === :(=) && ex.args[1] isa Symbol
        counts[ex.args[1]] = get(counts, ex.args[1], 0) + 1
    elseif ex.head in (:(+=), :(-=), :(*=), :(/=), :(.=)) && ex.args[1] isa Symbol
        counts[ex.args[1]] = get(counts, ex.args[1], 0) + 2
    end
    # A name whose value is written in place (`x[i] = v`, `x .= v`, a mutating
    # `f!(x, …)` call) no longer holds what its assignment computed: never a
    # definition, never an operand to move.
    _rw_mutated!(counts, ex)
    foreach(a -> _rw_count!(counts, a), ex.args)
    counts
end

function _rw_mutated!(counts::Dict{Symbol,Int}, ex::Expr)
    target = nothing
    if ex.head === :(=) && ex.args[1] isa Expr && ex.args[1].head in (:ref, :., :curly)
        target = ex.args[1].args[1]
    elseif ex.head === :(.=)
        target = ex.args[1] isa Expr && ex.args[1].head === :ref ? ex.args[1].args[1] : ex.args[1]
    elseif ex.head === :call && length(ex.args) >= 2
        callee = ex.args[1]
        name = callee isa GlobalRef ? callee.name : callee isa Symbol ? callee :
               callee isa Expr && callee.head === :. && length(callee.args) == 2 &&
               callee.args[2] isa QuoteNode ? callee.args[2].value : nothing
        name isa Symbol && endswith(String(name), "!") && (target = ex.args[2])
    end
    target isa Symbol && (counts[target] = get(counts, target, 0) + 2)
    counts
end

function _rw_closure_assigned!(counts::Dict{Symbol,Int}, ex)
    ex isa Expr || return counts
    ex.head in (:quote, :inert) && return counts
    if ex.head in (:(=), :(+=), :(-=), :(*=), :(/=)) && ex.args[1] isa Symbol
        counts[ex.args[1]] = get(counts, ex.args[1], 0) + 2
    end
    foreach(a -> _rw_closure_assigned!(counts, a), ex.args)
    counts
end

# `ex` rewritten under the definitions `env` holds; blocks and scoped
# constructs see a copy, so a definition never escapes its block.
function _rw_expr(ex, ctx::_RwContext, env::Dict{Symbol,Any})
    ex isa Expr || return ex
    head = ex.head
    head in _RW_OPAQUE && return ex
    head in _RW_CLOSURES && return ex
    if head === :block
        return _rw_block(ex, ctx, copy(env))
    elseif head === :let
        length(ex.args) == 2 || return ex
        bindings = _rw_expr(ex.args[1], ctx, env)
        inner = copy(env)
        for b in (bindings isa Expr && bindings.head === :block ? bindings.args : Any[bindings])
            a = _rw_assignment(b)
            a === nothing || (inner[a[1]] = a[2])
        end
        return Expr(:let, bindings, _rw_expr(ex.args[2], ctx, inner))
    elseif head in (:if, :elseif, :while)
        args = Any[_rw_expr(ex.args[1], ctx, env)]
        for a in ex.args[2:end]
            push!(args, _rw_expr(a, ctx, copy(env)))
        end
        return Expr(head, args...)
    elseif head === :for
        return Expr(:for, ex.args[1], Any[_rw_expr(a, ctx, copy(env)) for a in ex.args[2:end]]...)
    elseif head === :(=) && length(ex.args) == 2
        return Expr(:(=), ex.args[1], _rw_expr(ex.args[2], ctx, env))
    elseif head === :local
        return Expr(:local, Any[_rw_expr(a, ctx, env) for a in ex.args]...)
    elseif head === :call && !isempty(ex.args)
        arm = _rw_arm_view(ctx, ex.args[1], Any[ex.args[2:end]...], env)
        rewritten = arm !== nothing ? _rw_resolve_view_rule(arm, ctx, env) :
                    _callable_slot(ex.args[1]) === nothing ? _rw_rules(ex, ctx, env) :
                    _rw_table_rules(ex, ctx, env)
        rewritten === nothing || (ex = rewritten)
        return Expr(ex.head, ex.args[1], Any[_rw_expr(a, ctx, env) for a in ex.args[2:end]]...)
    end
    Expr(head, Any[_rw_expr(a, ctx, env) for a in ex.args]...)
end

# A block's statements in order, each after the definitions before it.
function _rw_block(ex::Expr, ctx::_RwContext, env::Dict{Symbol,Any})
    out = Expr(:block)
    for st in ex.args
        new = _rw_expr(st, ctx, env)
        push!(out.args, new)
        a = _rw_assignment(new)
        a === nothing || get(ctx.counts, a[1], 0) == 1 && (env[a[1]] = a[2])
    end
    out
end

# --- removing what nothing reads ---------------------------------------------------

# Reads of each name in `ex`: every occurrence but an assignment's target and a
# declaration.
function _rw_reads!(reads::Dict{Symbol,Int}, ex)
    if ex isa Symbol
        reads[ex] = get(reads, ex, 0) + 1
    elseif ex isa Expr
        ex.head in (:quote, :inert, :meta) && return reads
        if ex.head === :(=) && ex.args[1] isa Symbol
            _rw_reads!(reads, ex.args[2])
        elseif ex.head === :local
            for a in ex.args
                a isa Symbol && continue
                a isa Expr && a.head === :(::) && a.args[1] isa Symbol && continue
                _rw_reads!(reads, a)
            end
        else
            foreach(a -> _rw_reads!(reads, a), ex.args)
        end
    end
    reads
end

# Whether `rhs` is a call of a pure, total inner function of the rules over
# names and literals, whose statement can go once nothing reads its name.
function _rw_pure(rhs, ctx::_RwContext)
    rhs isa Symbol && return true
    rhs isa Expr || return false
    _rw_materialize_step(rhs, ctx) && return _rw_pure(last(rhs.args), ctx)
    if rhs.head === :let
        length(rhs.args) == 2 || return false
        bindings, body = rhs.args
        for b in (bindings isa Expr && bindings.head === :block ? bindings.args : Any[bindings])
            b isa LineNumberNode && continue
            b isa Expr && b.head === :(=) && b.args[1] isa Symbol &&
                (b.args[2] isa Symbol || b.args[2] isa Number ||
                 _rw_read_parts(b.args[2]) !== nothing) || return false
        end
        statements = body isa Expr && body.head === :block ?
            Any[s for s in body.args if !(s isa LineNumberNode)] : Any[body]
        return length(statements) == 1 && _rw_pure(only(statements), ctx)
    end
    rhs.head === :call && !isempty(rhs.args) || return false
    simple(a) = a isa Symbol || a isa Number || _rw_read_parts(a) !== nothing
    slot = _callable_slot(rhs.args[1])
    if slot !== nothing
        # A table call of the raw function (`p = logistic(e)` in a cell).
        1 <= slot <= length(ctx.recipes) && ctx.recipes[slot] isa Recipe || return false
        op = ctx.recipes[slot].op
        return (_rw_is_exp(op) || op === LogExpFunctions.logistic) &&
               all(simple, rhs.args[2:end])
    end
    f = _rw_resolve(rhs.args[1], @__MODULE__)
    if f === _native_broadcast_materialize
        return length(rhs.args) == 2 && _rw_pure(rhs.args[2], ctx)
    elseif _rw_is_broadcasted(f)
        length(rhs.args) >= 2 || return false
        g = _rw_resolve(rhs.args[2], @__MODULE__)
        return (_rw_is_exp(g) || g === LogExpFunctions.logistic) &&
               all(simple, rhs.args[3:end])
    end
    (_rw_is_exp(f) || f === LogExpFunctions.logistic) && all(simple, rhs.args[2:end])
end

# Removes, in the blocks the walk rebuilt, the statements of pure inner calls
# whose names nothing reads. Opaque nodes and closures are the input's own
# objects (`_rw_expr` returns them unchanged), so they are never entered.
function _rw_drop_dead!(ex, ctx::_RwContext, reads::Dict{Symbol,Int})
    ex isa Expr || return false
    (ex.head in _RW_OPAQUE || ex.head in _RW_CLOSURES) && return false
    changed = false
    # An inlined source's binding of a name its body no longer reads (the
    # rewritten call reads the definition's operand instead) goes too, so the
    # name it bound can become unread.
    if ex.head === :let && length(ex.args) == 2
        bindings = ex.args[1]
        unread(b) = b isa Expr && b.head === :(=) && length(b.args) == 2 &&
            b.args[1] isa Symbol && get(reads, b.args[1], 0) == 0 &&
            (b.args[2] isa Symbol || b.args[2] isa Number ||
             _rw_read_parts(b.args[2]) !== nothing)
        if bindings isa Expr && bindings.head === :block
            keep = Any[b for b in bindings.args if !unread(b)]
            length(keep) < length(bindings.args) && (bindings.args = keep; changed = true)
        elseif unread(bindings)
            ex.args[1] = Expr(:block)
            changed = true
        end
    end
    if ex.head === :block
        keep = Any[]
        for st in ex.args
            a = st isa Expr && st.head === :(=) ? _rw_assignment(st) : nothing
            if a !== nothing && get(ctx.counts, a[1], 0) == 1 && get(reads, a[1], 0) == 0 &&
               _rw_pure(a[2], ctx)
                changed = true
                continue
            end
            push!(keep, st)
        end
        changed && (ex.args = keep)
    end
    for a in ex.args
        _rw_drop_dead!(a, ctx, reads) && (changed = true)
    end
    changed
end

# --- the pass -------------------------------------------------------------------

"""
    _stable_rewrites(ast::Expr, recipes::Tuple[, definitions]) -> Expr

The generated function `ast` with its numerically fragile compositions in their
stable forms (the file header). `recipes` holds what each position of the
callables table the body calls through runs: a lowered recipe, or a
non-allocating step. `definitions` binds parameters of `ast` to the
expressions that computed them before it, over other parameters of `ast` (a
source-reading backend's earlier statements, `jax_source`), so a composition
across them is found as one within the body is. Returns `ast` itself when
nothing applies.
"""
function _stable_rewrites(ast::Expr, recipes::Tuple,
                          definitions::Dict{Symbol,Any} = Dict{Symbol,Any}())
    ast.head in (:function, :->) && length(ast.args) == 2 &&
        ast.args[1] isa Expr && ast.args[1].head === :tuple || return ast
    counts = _rw_count!(Dict{Symbol,Int}(), ast.args[2])
    params = Set{Symbol}()
    for arg in ast.args[1].args
        name = arg isa Symbol ? arg :
               arg isa Expr && arg.head === :(::) && arg.args[1] isa Symbol ? arg.args[1] : nothing
        name === nothing || get(counts, name, 0) == 0 && push!(params, name)
    end
    ctx = _RwContext(recipes, counts, params, Ref(0))
    body = _rw_expr(ast.args[2], ctx, copy(definitions))
    body isa Expr && body.head === :block || return ast
    body == ast.args[2] && return ast
    while _rw_drop_dead!(body, ctx, _rw_reads!(Dict{Symbol,Int}(), body))
    end
    Expr(ast.head, ast.args[1], body)
end
