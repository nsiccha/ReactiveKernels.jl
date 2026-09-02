# Destination-passing decomposition for the optional non-allocating
# preparation (`prepare_nonallocating`).
#
# The previous non-allocating rewrite routed EVERY selected recipe through one
# opaque per-recipe cache cell. A recipe operation synthesized from captured
# `@kernel` source (`_KernelSourceOp`, form `:fused`) is an anonymous
# function, so the in-place layer has no mutating counterpart for it and every
# call took the allocating fallback — a fused `W * transpose(X) .+ b` or
# `vcat(zeros(1, n), m)` reallocated its full result on each invocation, and a
# fused lazy wrapper such as `reshape(view(u, r), a, b)` was worse than the
# ordinary prepared kernel because storing it into the declared typed slot
# forced a full `convert` copy per call.
#
# This file decomposes such fused sources into primitive steps at preparation
# time, so the `cache_apply` layer sees operations buffers can actually be
# reused for:
#
# - identity-preserving wrappers (`view`, `reshape`, `transpose`, `eachcol`,
#   ranges, scalar arithmetic, …) and isbits-valued calls are emitted inline —
#   they never owned a buffer worth caching;
# - broadcast materializations (dotted calls, and array `getindex` with
#   range/vector indices via `view`) become `_MaterializeStep` destination
#   steps;
# - `vcat` becomes `_ConcatenateStep` and `zeros`/`ones` become
#   `_FillConstructorStep`;
# - every other resolved call stays a generic per-step cache operation, so
#   registered in-place coverage (e.g. `mul!`-backed `*`) applies per step.
#
# Soundness: the fused operation's callable is a module-anonymous function
# with no captured fields, so its free symbols resolve in
# `parentmodule(op.f)`. Decomposition resolves every free symbol against that
# module, requires the binding to be `const`, and emits `GlobalRef`s to those
# exact bindings — never name-based guesses. Any source shape outside the
# grammar (or any resolution failure) falls back to the previous whole-recipe
# cache step, so decomposition never widens behavior; it only exposes the same
# computation at a granularity the in-place layer can reuse.

# --- destination step operations -------------------------------------------
# Core-owned; CALLING one is the exact allocating semantics, so a hand-written
# or absent in-place layer stays correct. The MutatingFunctions extension adds
# buffer-reusing `apply!!` methods for them.

"Materialize a lazily-built broadcast; in-place layers may reuse a destination."
struct _MaterializeStep end
@inline (::_MaterializeStep)(bc) = Base.Broadcast.materialize(bc)

"Concatenate arrays; in-place layers may copy segments into a destination."
struct _ConcatenateStep{F}
    f::F
end
@inline (op::_ConcatenateStep)(args...) = op.f(args...)

"""
Matrix product with a guarded destination. The in-place layer's raw `*`
coverage follows `mul!`'s pre-sized convention (the cache must already have
the result shape), so a batch-size change between calls would hand `mul!` a
stale destination; this step owns the shape/eltype guard and reseeds instead.
"""
struct _MatMulStep end
@inline (::_MatMulStep)(A, B) = A * B

"Array constructor with a known fill value (`zeros`/`ones`)."
struct _FillConstructorStep{F}
    f::F
end
@inline (op::_FillConstructorStep)(dims::Integer...) = op.f(dims...)
_fill_constructor_value(::_FillConstructorStep{typeof(zeros)}) = 0.0
_fill_constructor_value(::_FillConstructorStep{typeof(ones)}) = 1.0

# --- step program accumulator ----------------------------------------------

mutable struct _StepProgram
    ops::Vector{Any}
    caches::Vector{Any}
end

function _step!(prog::_StepProgram, op, cache)
    push!(prog.ops, op)
    push!(prog.caches, cache)
    length(prog.ops)
end

_step_call(j::Int, args...) = Expr(:call, _CACHE_APPLY_ARG,
    Expr(:ref, _CACHES_ARG, j), Expr(:ref, _OPS_ARG, j), args...)
_plain_call(j::Int, args...) = Expr(:call, Expr(:ref, _OPS_ARG, j), args...)

# --- static typing helpers --------------------------------------------------

# Inference-derived upper bound for a call's result; `Any` when unknown.
function _static_type(f, argtypes...)
    all(t -> t isa Type, argtypes) || return Any
    T = try
        Base.promote_op(f, argtypes...)
    catch
        Any
    end
    T === Union{} ? Any : T
end

_nonalloc_slot(::Type{T}) where {T} = Ref{Union{Nothing,T}}(nothing)

# Identity-preserving wrappers that are cheaper to rebuild inline than to
# cache: caching them buys no buffer reuse and may force conversion copies.
const _NONALLOC_LAZY_CALLEES = (
    view, reshape, transpose, adjoint, vec, eachcol, eachrow, identity,
)
_nonalloc_is_lazy(f) = any(l -> l === f, _NONALLOC_LAZY_CALLEES)

function _nonalloc_resolve_const(mod::Module, s::Symbol)
    isdefined(mod, s) || return nothing
    isconst(mod, s) || return nothing
    Some(getglobal(mod, s))
end

function _nonalloc_undotted(s::Symbol)
    str = String(s)
    length(str) > 1 && startswith(str, '.') || return nothing
    s in (:.., :(...)) && return nothing
    Symbol(str[2:end])
end

# --- fused-source decomposition --------------------------------------------

struct _FusedDecomposition
    mod::Module
    argmap::Dict{Symbol,Any}
    argtypes::Dict{Symbol,Any}
    stmts::Vector{Any}
    ops::Vector{Any}
    caches::Vector{Any}
    offset::Int
end

function _emit_step!(ctx::_FusedDecomposition, op, ::Type{T}, args...) where {T}
    push!(ctx.ops, op)
    push!(ctx.caches, _nonalloc_slot(T))
    j = ctx.offset + length(ctx.ops)
    tmp = gensym(:step)
    push!(ctx.stmts, Expr(:(=), tmp, _step_call(j, args...)))
    tmp
end

# Decompose one source node. Returns `(expr, statictype)` or `nothing` when
# the node is outside the supported grammar (the caller then falls back to the
# whole-recipe cache step). `allow_lazy_broadcast` keeps a dotted child lazy
# inside an enclosing dotted call, preserving Julia's dot fusion exactly.
function _decompose(ctx::_FusedDecomposition, node, allow_lazy_broadcast::Bool)
    if node isa Symbol
        haskey(ctx.argmap, node) &&
            return (ctx.argmap[node], get(ctx.argtypes, node, Any))
        resolved = _nonalloc_resolve_const(ctx.mod, node)
        resolved === nothing && return nothing
        return (GlobalRef(ctx.mod, node), typeof(something(resolved)))
    end
    node isa Expr || return (node, typeof(node))
    if node.head === :call && !isempty(node.args)
        callee = node.args[1]
        callee isa Symbol || return nothing
        base = _nonalloc_undotted(callee)
        base === nothing ||
            return _decompose_broadcast(ctx, base, node.args[2:end],
                                        allow_lazy_broadcast)
        return _decompose_call(ctx, callee, node.args[2:end])
    end
    node.head === :ref && length(node.args) >= 2 &&
        return _decompose_getindex(ctx, node.args[1], node.args[2:end])
    nothing
end

function _decompose_arguments(ctx::_FusedDecomposition, rawargs,
                              allow_lazy_broadcast::Bool)
    args = Any[]
    types = Any[]
    for raw in rawargs
        d = _decompose(ctx, raw, allow_lazy_broadcast)
        d === nothing && return nothing
        push!(args, d[1])
        push!(types, d[2])
    end
    (args, types)
end

function _decompose_call(ctx::_FusedDecomposition, callee::Symbol, rawargs)
    haskey(ctx.argmap, callee) && return nothing        # call through a port
    resolved = _nonalloc_resolve_const(ctx.mod, callee)
    resolved === nothing && return nothing
    f = something(resolved)
    decomposed = _decompose_arguments(ctx, rawargs, false)
    decomposed === nothing && return nothing
    args, types = decomposed
    T = _static_type(f, types...)
    fref = GlobalRef(ctx.mod, callee)
    if (isconcretetype(T) && isbitstype(T)) || _nonalloc_is_lazy(f)
        return (Expr(:call, fref, args...), T)
    end
    T isa Type || return nothing
    f === Base.vcat &&
        return (_emit_step!(ctx, _ConcatenateStep(f), T, args...), T)
    if (f === Base.zeros || f === Base.ones) && !isempty(types) &&
       all(t -> t isa Type && t <: Integer, types)
        return (_emit_step!(ctx, _FillConstructorStep(f), T, args...), T)
    end
    if f === Base.:*
        length(types) == 2 &&
            all(t -> t isa Type && isconcretetype(t) &&
                     t <: AbstractVecOrMat, types) ||
            return nothing
        return (_emit_step!(ctx, _MatMulStep(), T, args...), T)
    end
    # The in-place layer's `\` and `/` coverage shares `mul!`'s pre-sized
    # destination convention; without a guarded step, keep the whole-recipe
    # fallback rather than risking a stale destination on a shape change.
    (f === Base.:\ || f === Base.:/) && return nothing
    (_emit_step!(ctx, f, T, args...), T)
end

function _decompose_broadcast(ctx::_FusedDecomposition, base::Symbol, rawargs,
                              allow_lazy_broadcast::Bool)
    haskey(ctx.argmap, base) && return nothing
    resolved = _nonalloc_resolve_const(ctx.mod, base)
    resolved === nothing && return nothing
    f = something(resolved)
    decomposed = _decompose_arguments(ctx, rawargs, true)
    decomposed === nothing && return nothing
    args, types = decomposed
    bc = Expr(:call, GlobalRef(Base.Broadcast, :broadcasted),
              GlobalRef(ctx.mod, base), args...)
    BT = _static_type(Base.Broadcast.broadcasted, typeof(f), types...)
    allow_lazy_broadcast && return (bc, BT)
    T = _static_type(Base.Broadcast.materialize, BT)
    isconcretetype(T) && isbitstype(T) &&
        return (Expr(:call, GlobalRef(Base.Broadcast, :materialize), bc), T)
    T isa Type || return nothing
    (_emit_step!(ctx, _MaterializeStep(), T, bc), T)
end

function _decompose_getindex(ctx::_FusedDecomposition, xraw, idxraws)
    dx = _decompose(ctx, xraw, false)
    dx === nothing && return nothing
    decomposed = _decompose_arguments(ctx, idxraws, false)
    decomposed === nothing && return nothing
    idxs, itypes = decomposed
    T = _static_type(Base.getindex, dx[2], itypes...)
    isconcretetype(T) && isbitstype(T) &&
        return (Expr(:call, GlobalRef(Base, :getindex), dx[1], idxs...), T)
    dx[2] isa Type && dx[2] <: AbstractArray || return nothing
    all(t -> t isa Type &&
             (t <: AbstractRange || t <: AbstractVector{<:Integer} ||
              t <: Colon), itypes) || return nothing
    view_expr = Expr(:call, GlobalRef(Base, :view), dx[1], idxs...)
    VT = _static_type(Base.view, dx[2], itypes...)
    bc = Expr(:call, GlobalRef(Base.Broadcast, :broadcasted),
              GlobalRef(Base, :identity), view_expr)
    BT = _static_type(Base.Broadcast.broadcasted, typeof(identity), VT)
    MT = _static_type(Base.Broadcast.materialize, BT)
    MT isa Type || return nothing
    (_emit_step!(ctx, _MaterializeStep(), MT, bc), MT)
end

# Try to decompose one fused-source recipe statement. Returns
# `(stmts, result_type)` or `nothing` (fall back to the whole-recipe step).
function _decompose_fused_recipe!(prog::_StepProgram, r::Recipe, callargs,
                                  input_types, lhs)
    op = r.op
    kernel_sourceop_form(op) === :fused || return nothing
    fieldcount(typeof(op.f)) == 0 || return nothing     # capturing closure
    argmap = Dict{Symbol,Any}()
    argtypes = Dict{Symbol,Any}()
    for (v, aexpr, at) in zip(r.inputs, callargs, input_types)
        argmap[v.name] = aexpr
        argtypes[v.name] = at
    end
    ctx = _FusedDecomposition(parentmodule(typeof(op.f)), argmap, argtypes,
                              Any[], Any[], Any[], length(prog.ops))
    d = _decompose(ctx, r.source, false)
    d === nothing && return nothing
    append!(prog.ops, ctx.ops)
    append!(prog.caches, ctx.caches)
    stmts = ctx.stmts
    push!(stmts, Expr(:(=), lhs, d[1]))
    (stmts, d[2])
end

# --- authored-plate cache slots ---------------------------------------------

# An authored plate's borrowed cache is seeded as a concrete empty Array so
# later calls reuse it through `broadcast!`. When the batched argument types
# are statically known, the broadcast rank is too, and a rank-concrete slot
# keeps the whole plate call inference-visible instead of loading an unranked
# `Array{T}` on every invocation.
function _plate_broadcast_rank(::Val{A}, argtypes) where {A}
    all(t -> t isa Type && isconcretetype(t), argtypes) || return nothing
    BT = _static_type(_authored_plate_broadcast, Val{A}, argtypes...)
    BT <: Base.Broadcast.Broadcasted && isconcretetype(BT) || return nothing
    axes_type = BT.parameters[2]
    axes_type <: Tuple || return nothing
    length(axes_type.parameters)
end

function _plate_cache_slot(op::_AuthoredPlateOp{K,A}, argtypes) where {K,A}
    T = valtype(only(outputs(op.kernel)))
    N = _plate_broadcast_rank(Val(A), argtypes)
    N === nothing && return Ref{Array{T}}(Vector{T}())
    Ref{Array{T,N}}(Array{T,N}(undef, ntuple(_ -> 0, N)...))
end

function _plate_result_type(op::_AuthoredPlateOp{K,A}, argtypes) where {K,A}
    T = valtype(only(outputs(op.kernel)))
    N = _plate_broadcast_rank(Val(A), argtypes)
    N === nothing ? Array{T} : Array{T,N}
end

# --- statement-level program builder ----------------------------------------

function _nonalloc_argument_types(types::Dict{Symbol,Any}, callargs)
    Any[a isa Symbol ? get(types, a, Any) :
        a isa Expr ? Any : typeof(a) for a in callargs]
end

function _nonalloc_rewrite_recipe!(newbody, prog::_StepProgram, r::Recipe,
                                   types::Dict{Symbol,Any}, lhs, callargs)
    argtypes = _nonalloc_argument_types(types, callargs)
    record!(T) = lhs isa Symbol && (types[lhs] = T)
    op = r.op
    if !r.effectful
        if op isa _AuthoredPlateOp
            j = _step!(prog, op, _plate_cache_slot(op, argtypes))
            push!(newbody.args,
                  Expr(:(=), lhs, _step_call(j, callargs...)))
            record!(_plate_result_type(op, argtypes))
            return
        end
        if op isa _KernelSourceOp
            if !(r.source isa _NoKernelSource)
                dec = _decompose_fused_recipe!(prog, r, callargs, argtypes, lhs)
                if dec !== nothing
                    append!(newbody.args, dec[1])
                    record!(dec[2])
                    return
                end
            end
        else
            # A bare identity operation: isbits results and identity-preserving
            # wrappers need no cache at all.
            T = _static_type(op, argtypes...)
            if (isconcretetype(T) && isbitstype(T)) || _nonalloc_is_lazy(op)
                j = _step!(prog, op, nothing)
                push!(newbody.args, Expr(:(=), lhs, _plain_call(j, callargs...)))
                record!(T)
                return
            end
        end
    end
    # Whole-recipe cache step (the previous behavior). The slot is typed by the
    # inferred result when concrete, so storing a lazy wrapper result never
    # forces a `convert` copy into the declared port type.
    T = _static_type(op, argtypes...)
    slot_type = T isa Type && isconcretetype(T) ? T : valtype(only(r.outputs))
    j = _step!(prog, op, _nonalloc_slot(slot_type))
    push!(newbody.args, Expr(:(=), lhs, _step_call(j, callargs...)))
    record!(T isa Type && T !== Any ? T : valtype(only(r.outputs)))
    nothing
end

# Safety net for statement shapes the builder does not model: every embedded
# `__ops__[i](...)` call still becomes a whole-recipe cache step.
function _nonalloc_rewrite_nested(prog::_StepProgram, p::Plan, node)
    node isa Expr || return node
    args = map(a -> _nonalloc_rewrite_nested(prog, p, a), node.args)
    rewritten = Expr(node.head, args...)
    if rewritten.head === :call && !isempty(rewritten.args)
        slot = _operation_slot(rewritten.args[1])
        if slot !== nothing && 1 <= slot <= length(p.recipes)
            r = p.recipes[slot]
            j = _step!(prog, r.op, _cache_slot(only(r.outputs)))
            return _step_call(j, rewritten.args[2:end]...)
        end
    end
    rewritten
end

"""
    _nonallocating_program(p::Plan, ast::Expr) -> (ast, ops, caches)

Rewrite the un-embedded lowering of `p` into the non-allocating step program:
per-step operations and typed persistent caches, with fused captured sources
decomposed into destination-passing steps where the grammar allows, and the
previous whole-recipe cache step as the universal fallback.
"""
function _nonallocating_program(p::Plan, ast::Expr)
    ast.head === :function ||
        throw(ArgumentError("non-allocating preparation requires a function Expr"))
    signature = ast.args[1]
    signature isa Expr && signature.head === :tuple &&
        !isempty(signature.args) && first(signature.args) === _OPS_ARG ||
        throw(ArgumentError("non-allocating preparation requires the lowered __ops__ signature"))

    prog = _StepProgram(Any[], Any[])
    types = Dict{Symbol,Any}()
    for (v, argexpr) in zip(p.have, signature.args[2:end])
        name = argexpr isa Expr && argexpr.head === :(::) ?
               argexpr.args[1] : argexpr
        name isa Symbol && (types[name] = valtype(v))
    end

    body = ast.args[2]
    newbody = Expr(:block)
    for stmt in body.args
        handled = false
        if stmt isa Expr && stmt.head === :(=) && stmt.args[2] isa Expr &&
           stmt.args[2].head === :call
            call = stmt.args[2]
            slot = _operation_slot(call.args[1])
            if slot !== nothing && 1 <= slot <= length(p.recipes)
                _nonalloc_rewrite_recipe!(newbody, prog, p.recipes[slot],
                                          types, stmt.args[1],
                                          call.args[2:end])
                handled = true
            end
        end
        handled ||
            push!(newbody.args, _nonalloc_rewrite_nested(prog, p, stmt))
    end
    args = Expr(:tuple, _OPS_ARG, _CACHES_ARG, _CACHE_APPLY_ARG,
                signature.args[2:end]...)
    (Expr(:function, args, newbody), Tuple(prog.ops), Tuple(prog.caches))
end
