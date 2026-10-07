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
#   postfix `'`/`.'`, ranges, scalar arithmetic, …) and isbits-valued calls
#   are emitted inline — they never owned a buffer worth caching;
# - broadcast materializations (dotted calls, and array `getindex` with
#   range/vector indices via `view`) become `_MaterializeStep` destination
#   steps;
# - `vcat` becomes `_ConcatenateStep` and `zeros`/`ones` become
#   `_FillConstructorStep`; `sum`/`prod`/`minimum`/`maximum` with a
#   preparation-constant `dims` keyword become `_RowReduceStep`;
# - every other resolved call stays a generic per-step cache operation, so
#   registered in-place coverage (e.g. `mul!`-backed `*`) applies per step.
#
# Soundness: the fused operation's callable is a module-anonymous function
# with no captured fields, so its free symbols resolve in
# the native source closure's defining module. Decomposition resolves every
# free symbol against that module, requires the binding to be `const`, and emits
# `GlobalRef`s to those exact bindings — never name-based guesses. Module-qualified callees
# (`Pkg.f`) resolve through the same rule, requiring every path segment to be
# a `const` module binding. Any source shape outside the grammar (keyword
# calls other than constant-`dims` reductions, a non-`const` global, a call
# through a port — including `dims` read from a port) falls back to the
# previous whole-recipe cache step, so decomposition never widens behavior;
# it only exposes the same computation at a granularity the in-place layer
# can reuse.

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

"""
Row/column reduction with a preparation-constant `dims` keyword
(`sum`/`prod`/`minimum`/`maximum`). The callable is the exact allocating
semantics (`f(A; dims = ...)`); the MutatingFunctions extension reuses the
destination through Base's `reducedim!` family, which runs the same
per-lane `mapreducedim` pass as the allocating twin.
"""
struct _RowReduceStep{F,D}
    f::F
    dims::D
end
@inline (op::_RowReduceStep)(A) = op.f(A; dims = op.dims)

"One-dimensional gather (`x[idx]`); in-place layers may copy into a destination."
struct _GatherStep end
@inline (::_GatherStep)(x, idx...) = x[idx...]
@inline (op::_FillConstructorStep)(dims::Integer...) = op.f(dims...)
_fill_constructor_value(::_FillConstructorStep{typeof(zeros)}) = 0.0
_fill_constructor_value(::_FillConstructorStep{typeof(ones)}) = 1.0

# --- step program accumulator ----------------------------------------------

mutable struct _StepProgram
    ops::Vector{Any}
    caches::Vector{Any}
    # The extension's `cache_apply` callable; `_nonalloc_destination`
    # dispatches on it to ask which steps fill an owned destination.
    cache_apply::Any
end

# --- cache ownership --------------------------------------------------------
# A cache slot holds its step's first result, and every later call offers that
# value to `apply!!` as the destination of the new result. Only storage the
# kernel owns may be offered: `apply!!`'s generic fallback copies the new
# result INTO the cached value, so a first result that aliases caller data (a
# field read such as `sched.xs`, `eachrow` slices of a caller matrix) would
# receive the next call's input — the previous call's caller-owned arrays
# silently overwritten — and an immutable first result (a range) would throw.
# A step therefore keeps a slot only when its result is fresh storage that a
# destination-passing method fills. Every other step is a plain call: its
# result is returned as a borrowed value that may alias an input, and is never
# written into. The generic fallback allocates the result before copying it,
# so the plain call does no more work.
#
# The decomposition's own step operations produce fresh storage by
# construction and carry destination-passing methods in the extension.
_nonalloc_fresh_step(op) = false
_nonalloc_fresh_step(::Union{_MaterializeStep,_GatherStep,_MatMulStep,
                             _ConcatenateStep,_FillConstructorStep,
                             _RowReduceStep,_LaneGather}) = true

# Whether a step with operation `op`, inferred result type `T` and argument
# types `argtypes` keeps a cache slot. The extension adds the method for its
# `cache_apply` callable that also recognizes registered `apply!!` methods.
_nonalloc_destination(cache_apply, op, T, argtypes) = _nonalloc_fresh_step(op)

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
_nonalloc_is_lazy(::_BoundConstant) = true

function _nonalloc_resolve_const(mod::Module, s::Symbol)
    isdefined(mod, s) || return nothing
    isconst(mod, s) || return nothing
    Some(getglobal(mod, s))
end

# Resolve a callee that is either a bare name or a module-qualified path
# (`LogExpFunctions.logistic`, `A.B.f`). Every step must name a `const`
# binding — the same rule bare names follow — and the owner of each path
# segment must itself be a `const` module binding, so the emitted `GlobalRef`
# always names the exact function the fused closure would call. Returns
# `(function value, GlobalRef)` or `nothing`.
function _nonalloc_resolve_function(mod::Module, callee)
    if callee isa Symbol
        r = _nonalloc_resolve_const(mod, callee)
        r === nothing && return nothing
        return (something(r), GlobalRef(mod, callee))
    end
    callee isa Expr && callee.head === :. && length(callee.args) == 2 &&
        callee.args[2] isa QuoteNode || return nothing
    owner = if callee.args[1] isa Symbol
        r = _nonalloc_resolve_const(mod, callee.args[1])
        if r === nothing
            return nothing
        end
        something(r)
    else
        r = _nonalloc_resolve_function(mod, callee.args[1])
        if r === nothing
            return nothing
        end
        first(r)
    end
    owner isa Module || return nothing
    name = callee.args[2].value
    name isa Symbol && isdefined(owner, name) && isconst(owner, name) ||
        return nothing
    (getglobal(owner, name), GlobalRef(owner, name))
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
    cache_apply::Any
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
        callee isa Symbol || callee isa Expr || return nothing
        callee isa Expr &&
            _nonalloc_resolve_function(ctx.mod, callee) === nothing &&
            return nothing
        base = callee isa Symbol ? _nonalloc_undotted(callee) : nothing
        base === nothing ||
            return _decompose_broadcast(ctx, base, node.args[2:end],
                                        allow_lazy_broadcast)
        return _decompose_call(ctx, callee, node.args[2:end])
    end
    # Postfix `'` parses as `Expr(:')` and `.'` as `Expr(:.')` — unary
    # operator-expression heads, not `:call`s with an operator callee. Route
    # them through the ordinary call path under their exact Base names.
    (node.head === Symbol("'") || node.head === Symbol(".'")) &&
        length(node.args) == 1 &&
        return _decompose_call(ctx,
            node.head === Symbol("'") ? :adjoint : :transpose,
            Any[node.args[1]])
    node.head === :ref && length(node.args) >= 2 &&
        return _decompose_getindex(ctx, node.args[1], node.args[2:end])
    node.head === :. && length(node.args) == 2 &&
        return _decompose_dotcall(ctx, node.args[1], node.args[2],
                                  allow_lazy_broadcast)
    nothing
end

# `f.(args)` parses as `Expr(:., f, Expr(:tuple, ...))` — unlike operator-dot
# `a .+ b`, which is a `:call` with a dotted callee. Same broadcast
# semantics, same treatment; `f` may be a bare name or a module-qualified
# path. Anything else in dot position (field access into a non-module value,
# a non-`const` binding) falls back safely.
function _decompose_dotcall(ctx::_FusedDecomposition, func, tup,
                             allow_lazy_broadcast::Bool)
    tup isa Expr && tup.head === :tuple || return nothing
    if func isa Symbol
        return _decompose_broadcast(ctx, func, tup.args, allow_lazy_broadcast)
    end
    rf = _nonalloc_resolve_function(ctx.mod, func)
    rf === nothing && return nothing
    _decompose_broadcast(ctx, rf[1], rf[2], tup.args, allow_lazy_broadcast)
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
    rf = _nonalloc_resolve_function(ctx.mod, callee)
    rf === nothing && return nothing
    _decompose_call_resolved(ctx, callee, rf[1], rf[2], rawargs)
end

function _decompose_call(ctx::_FusedDecomposition, callee, rawargs)
    rf = _nonalloc_resolve_function(ctx.mod, callee)
    rf === nothing && return nothing
    _decompose_call_resolved(ctx, callee, rf[1], rf[2], rawargs)
end

function _decompose_call_resolved(ctx::_FusedDecomposition, callee, f, fref,
                                  rawargs)
    if callee isa Symbol
        haskey(ctx.argmap, callee) && return nothing
    end
    # Keyword calls: only the reduction family with a preparation-constant
    # `dims` keyword leaves the whole-recipe fallback, via `_RowReduceStep`.
    # Every other keyword shape (including port-valued `dims`) is outside the
    # step grammar and keeps the fused closure, which applies the keywords
    # with its original semantics.
    kwargs = nothing
    positional = rawargs
    if !isempty(rawargs) && rawargs[1] isa Expr &&
       rawargs[1].head === :parameters
        kwargs = rawargs[1].args
        positional = rawargs[2:end]
        for kw in kwargs
            kw isa Expr && kw.head === :kw && length(kw.args) == 2 &&
                kw.args[1] === :dims || return nothing
        end
    end
    if kwargs !== nothing
        (f === Base.sum || f === Base.prod ||
         f === Base.minimum || f === Base.maximum) &&
            length(positional) == 1 || return nothing
        dimsval = _nonalloc_dims_const(only(kwargs))
        dimsval === nothing && return nothing
        reduced = _decompose_singleton_reduction(ctx, positional[1])
        reduced !== nothing && return reduced
        decomposed = _decompose_arguments(ctx, positional, false)
        decomposed === nothing && return nothing
        args, types = decomposed
        length(types) == 1 && types[1] isa Type || return nothing
        D = something(dimsval)
        T = _static_type(A -> f(A; dims = D), only(types))
        T isa DataType && isconcretetype(T) && T <: AbstractArray ||
            return nothing
        return (_emit_step!(ctx, _RowReduceStep(f, D), T, args...), T)
    end
    rawargs = positional
    if (f === Base.sum || f === Base.prod ||
        f === Base.minimum || f === Base.maximum) && length(rawargs) == 1
        reduced = _decompose_singleton_reduction(ctx, rawargs[1])
        reduced !== nothing && return reduced
    end
    decomposed = _decompose_arguments(ctx, rawargs, false)
    decomposed === nothing && return nothing
    args, types = decomposed
    T = _static_type(f, types...)
    if (isconcretetype(T) && isbitstype(T)) || _nonalloc_is_lazy(f)
        return (Expr(:call, fref, args...), T)
    end
    if T isa DataType && isconcretetype(T) && !ismutabletype(T)
        # An immutable value owns no reusable buffer: a cache step would only
        # store it back through the passthrough fallback, so emit the
        # construction inline exactly as the fused closure would run it. This
        # covers wrapper structs, tuples, and other immutable constructors;
        # mutable results (arrays, refs, dicts) keep their cache steps below.
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
    # A call without a destination-passing method returns its own result,
    # which may alias an argument: run it inline, never as a cache step.
    _nonalloc_destination(ctx.cache_apply, f, T, types) ||
        return (Expr(:call, fref, args...), T)
    (_emit_step!(ctx, f, T, args...), T)
end

# A preparation-constant reduction axis: an `Int`, `Colon`, or a tuple of
# `Int`s written as a literal in the source. Anything computed (a port read,
# a call) would freeze a preparation-time value into the step table and is
# therefore outside the grammar.
function _nonalloc_dims_const(kw)
    kw isa Expr && kw.head === :kw && length(kw.args) == 2 || return nothing
    v = kw.args[2]
    v isa Int && return Some(v)
    v === Colon() && return Some(Colon())
    v isa Expr && v.head === :tuple && all(a -> a isa Integer, v.args) &&
        return Some(Tuple(v.args))
    nothing
end

# A scalar reduction (`sum`/`prod`/`minimum`/`maximum`) over a provably
# single-element view is exactly that element — no summation algorithm runs, so
# there is no rounding to preserve. Match `view(x, a:a, ...)` with every index
# a syntactic `lo:lo` range over side-effect-free bounds (a Symbol or Number
# used twice evaluates identically, and evaluates once in the rewritten form),
# and only when the view call itself statically resolves to an array (so the
# fused closure's `view` would have succeeded, with identical bounds behavior:
# `a:a` in-bounds ⟺ `a` in-bounds). Anything else returns `nothing` and the
# reduction decomposes normally. This is the packed-read spelling
# `sum(view(unconstrained, i:i))` the query layer generates per scalar layout
# entry, which otherwise boxes one view object per entry per call.
function _decompose_singleton_reduction(ctx::_FusedDecomposition, raw)
    raw isa Expr && raw.head === :call && length(raw.args) >= 3 || return nothing
    vcallee = raw.args[1]
    vcallee isa Symbol || return nothing
    haskey(ctx.argmap, vcallee) && return nothing
    vresolved = _nonalloc_resolve_const(ctx.mod, vcallee)
    vresolved === nothing || something(vresolved) !== Base.view && return nothing
    xraw = raw.args[2]
    vidx = raw.args[3:end]
    # Every index is a syntactic `lo:lo` range over provably idempotent bounds
    # (a name or literal evaluates identically every time, and is evaluated
    # once in the rewritten form).
    for idx in vidx
        idx isa Expr && idx.head === :call && length(idx.args) == 3 &&
            idx.args[1] === Symbol(":") || return nothing
        lo, hi = idx.args[2], idx.args[3]
        lo == hi && lo isa Union{Symbol,Number} || return nothing
    end
    colonresolved = _nonalloc_resolve_const(ctx.mod, Symbol(":"))
    colonresolved === nothing && return nothing
    colonfn = something(colonresolved)
    dx = _decompose(ctx, xraw, false)
    dx === nothing && return nothing
    dx[2] isa Type || return nothing
    # The twin's range construction and `view` call must statically resolve to
    # concrete range and array types: then the twin's `view` provably succeeds
    # (with identical bounds behavior: `a:a` in-bounds ⟺ `a` in-bounds), the
    # view holds exactly one element, and scalar `getindex` is that element.
    trngs = Any[]
    loexprs = Any[]
    lotypes = Any[]
    for idx in vidx
        lo = idx.args[2]
        dlo = _decompose(ctx, lo, false)
        dlo === nothing && return nothing
        push!(loexprs, dlo[1])
        push!(lotypes, dlo[2])
        trng = _static_type(colonfn, dlo[2], dlo[2])
        trng isa DataType && trng <: AbstractRange || return nothing
        push!(trngs, trng)
    end
    tview = _static_type(Base.view, dx[2], trngs...)
    tview isa DataType && tview <: AbstractArray || return nothing
    getexpr = Expr(:call, GlobalRef(Base, :getindex), dx[1], loexprs...)
    T = _static_type(Base.getindex, dx[2], lotypes...)
    (getexpr, T)
end

function _decompose_broadcast(ctx::_FusedDecomposition, base::Symbol, rawargs,
                              allow_lazy_broadcast::Bool)
    haskey(ctx.argmap, base) && return nothing
    rf = _nonalloc_resolve_function(ctx.mod, base)
    rf === nothing && return nothing
    _decompose_broadcast(ctx, rf[1], rf[2], rawargs, allow_lazy_broadcast)
end

function _decompose_broadcast(ctx::_FusedDecomposition, f, fref::GlobalRef,
                              rawargs, allow_lazy_broadcast::Bool)
    decomposed = _decompose_arguments(ctx, rawargs, true)
    decomposed === nothing && return nothing
    args, types = decomposed
    bc = Expr(:call, GlobalRef(Base.Broadcast, :broadcasted),
              fref, args...)
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
    # A dedicated gather step whose arguments are the array and the index
    # values: one-dimensional vector gathers copy without constructing the
    # intermediate view object (which otherwise escapes once per call), while
    # every other shape takes the same view-plus-materialize computation the
    # in-place layer applies below, so behavior never regresses.
    GT = _static_type(Base.getindex, dx[2], itypes...)
    GT isa Type || return nothing
    (_emit_step!(ctx, _GatherStep(), GT, dx[1], idxs...), GT)
end

# Try to decompose one fused-source recipe statement. Returns
# `(stmts, result_type)` or `nothing` (fall back to the whole-recipe step).
function _decompose_fused_recipe!(prog::_StepProgram, r::Recipe, callargs,
                                  input_types, lhs)
    op = r.op
    kernel_sourceop_form(op) === :fused || return nothing
    source_f = _kernel_native_source(op.f)
    fieldcount(typeof(source_f)) == 0 || return nothing # capturing closure
    argmap = Dict{Symbol,Any}()
    argtypes = Dict{Symbol,Any}()
    for (v, aexpr, at) in zip(r.inputs, callargs, input_types)
        argmap[v.name] = aexpr
        argtypes[v.name] = at
    end
    ctx = _FusedDecomposition(parentmodule(typeof(source_f)), argmap, argtypes,
                              Any[], Any[], Any[], length(prog.ops),
                              prog.cache_apply)
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
    if BT <: Base.Broadcast.Broadcasted && isconcretetype(BT)
        axes_type = BT.parameters[2]
        axes_type <: Tuple && return length(axes_type.parameters)
    end
    # Wide tuples of operands can defeat inference of Broadcasted itself.
    # Dense array, tuple and scalar types still prove Julia's result rank.
    ranks = Int[]
    for (i, T) in enumerate(argtypes)
        (i in A || T <: Number) && continue
        if T <: AbstractArray
            push!(ranks, ndims(T))
        elseif T <: Tuple
            push!(ranks, 1)
        else
            return nothing
        end
    end
    maximum(ranks; init=0)
end

# The cache slot and recorded result type use the inferred concrete element
# type (`_authored_plate_result_eltype`) rather than the plan-level `Any` of an
# unannotated body, so an untyped plate materializes a typed buffer here exactly
# as the ordinary native lowering now does — no boxed `Vector{Any}`, and both
# execution paths agree bit-for-bit on the materialized pointwise vector.
_nested_plate_cache(op, argtypes, T, N) = nothing
function _nonalloc_plan_result_types(kernel, argument_types)
    p = kernel.plan
    types = Dict(canon_id(p.graph, v.id) => T for (v, T) in zip(p.have, argument_types))
    for recipe in p.recipes
        inputs = [get(types, canon_id(p.graph, v.id), Any) for v in recipe.inputs]
        T = _nonalloc_recipe_type(recipe.op, inputs)
        for (index, output) in enumerate(recipe.outputs)
            types[canon_id(p.graph, output.id)] = length(recipe.outputs) == 1 ? T :
                T isa DataType && T <: Tuple ? fieldtype(T, index) : Any
        end
    end
    Tuple(get(types, canon_id(p.graph, v.id), Any) for v in p.want)
end
_nonalloc_recipe_type(op, types) = _static_type(op, types...)
function _nonalloc_recipe_type(op::_AuthoredScanOp{K,A,I,H}, types) where {K,A,I,H}
    H && return _static_type(op, types...)
    iterated = [i for i in 2:length(types) if !(i in A)]
    shared = [i for i in 2:length(types) if i in A]
    step_types = Any[types[1], [eltype(types[i]) for i in iterated]..., types[shared]...]
    output = last(_nonalloc_plan_result_types(op.kernel, step_types))
    I && (output = promote_type(types[1], output))
    Vector{output}
end

function _nonalloc_rewrite_recipe!(body, prog::_StepProgram, r::Recipe,
        types::Dict{Symbol,Any}, lhs, args, ::Val{:scan})
    T = _nonalloc_recipe_type(r.op, _nonalloc_argument_types(types, args))
    slot = _step!(prog, identity, _nonalloc_slot(T))
    offset = length(prog.ops)
    for op in r.op.kernel.ops
        _step!(prog, op, nothing)
    end
    cache = Expr(:ref, Expr(:ref, _CACHES_ARG, slot))
    _lower_authored_scan_native!(body, r.op, args, lhs, offset; recycled=cache)
    push!(body.args, Expr(:(=), cache, lhs))
    lhs isa Symbol && (types[lhs] = T)
    nothing
end
# A declared array output (`y::Vector{Float64} = plate(...)`) fixes the
# element type and rank the native kernel's typed local converts the plate's
# result to (`_declare_typed_output!`). Filling a buffer of that element type
# gives the same values without a conversion copy; a cell type that inference
# cannot see through an untyped HAVE port would otherwise leave an `Array{Any}`
# buffer. Returns `(eltype, rank)`, either `nothing` when the declaration
# leaves it open.
function _declared_array_layout(::Type{D}) where {D}
    D <: AbstractArray || return (nothing, nothing)
    E = eltype(D)
    rank = findfirst(n -> D <: AbstractArray{<:Any,n}, 0:16)
    (isconcretetype(E) ? E : nothing, rank === nothing ? nothing : rank - 1)
end

function _nonalloc_plate_eltype(op::_AuthoredPlateOp{K,A}, argtypes,
                                declared::Type = Any) where {K,A}
    T = _authored_plate_result_eltype(op, argtypes)
    E = first(_declared_array_layout(declared))
    E === nothing || return E
    isconcretetype(T) && return T
    elements = [(i in A || argtypes[i] <: Number) ? argtypes[i] : eltype(argtypes[i])
        for i in eachindex(argtypes)]
    only(_nonalloc_plan_result_types(op.kernel, elements))
end
function _nonalloc_plate_rank(op::_AuthoredPlateOp{K,A}, argtypes,
                              declared::Type) where {K,A}
    N = _plate_broadcast_rank(Val(A), argtypes)
    N === nothing ? last(_declared_array_layout(declared)) : N
end
function _plate_cache_slot(op::_AuthoredPlateOp{K,A}, argtypes,
                           declared::Type = Any) where {K,A}
    T = _nonalloc_plate_eltype(op, argtypes, declared)
    N = _nonalloc_plate_rank(op, argtypes, declared)
    nested = _nested_plate_cache(op, argtypes, T, N)
    nested === nothing || return nested
    N === nothing && return Ref{Array{T}}(Vector{T}())
    Ref{Array{T,N}}(Array{T,N}(undef, ntuple(_ -> 0, N)...))
end

function _plate_result_type(op::_AuthoredPlateOp{K,A}, argtypes,
                            declared::Type = Any) where {K,A}
    T = _nonalloc_plate_eltype(op, argtypes, declared)
    N = _nonalloc_plate_rank(op, argtypes, declared)
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
        if op isa _AuthoredScanOp
            _nonalloc_rewrite_recipe!(newbody, prog, r, types, lhs, callargs, Val(:scan))
            return
        end
        if op isa _AuthoredPlateOp
            declared = valtype(only(r.outputs))
            slot = _plate_cache_slot(op, argtypes, declared)
            if slot isa Base.RefValue{<:Array}
                # Lower the cells natively, as ordinary `prepare` does, into the
                # cached buffer (`recycled`), like the scan step. Broadcasting
                # the cell's prepared kernel instead boxed a descriptor holding
                # it on every call (240 B with one `Ref` operand, 352 B with
                # three, Julia 1.10.12): inside this program its `copyto!` is
                # not inlined, and the operation it carries is a constant.
                j = _step!(prog, identity, slot)
                cache = Expr(:ref, Expr(:ref, _CACHES_ARG, j))
                _lower_authored_plate_native!(newbody, prog.ops, Recipe[], op,
                    callargs, r.inputs, lhs, nothing; recycled=cache,
                    element_type=first(_declared_array_layout(declared)))
                # The cell operations the lowering appended keep no cache.
                append!(prog.caches, fill(nothing, length(prog.ops) - length(prog.caches)))
                push!(newbody.args, Expr(:(=), cache, lhs))
                record!(_plate_result_type(op, argtypes, declared))
                return
            end
            j = _step!(prog, op, slot)
            push!(newbody.args,
                  Expr(:(=), lhs, _step_call(j, callargs...)))
            record!(_plate_result_type(op, argtypes, declared))
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
            # A fused source outside the decomposition grammar (control flow,
            # a branch closure) with an isbits result needs no cache either:
            # call it directly, exactly as the bare-operation path does below.
            T = _static_type(op, argtypes...)
            if isconcretetype(T) && isbitstype(T)
                j = _step!(prog, op, nothing)
                push!(newbody.args, Expr(:(=), lhs, _plain_call(j, callargs...)))
                record!(T)
                return
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
    # Whole-recipe step. It keeps a cache slot only when a destination-passing
    # method fills owned storage (see `_nonalloc_destination`); otherwise the
    # operation is called directly and its result is never written into. The
    # slot is typed by the inferred result when concrete, so storing a lazy
    # wrapper result never forces a `convert` copy into the declared port type.
    T = _static_type(op, argtypes...)
    slot_type = T isa Type && isconcretetype(T) ? T : valtype(only(r.outputs))
    if _nonalloc_destination(prog.cache_apply, op, slot_type, argtypes)
        j = _step!(prog, op, _nonalloc_slot(slot_type))
        push!(newbody.args, Expr(:(=), lhs, _step_call(j, callargs...)))
    else
        j = _step!(prog, op, nothing)
        push!(newbody.args, Expr(:(=), lhs, _plain_call(j, callargs...)))
    end
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
            out = only(r.outputs)
            if _nonalloc_destination(prog.cache_apply, r.op, valtype(out), nothing)
                j = _step!(prog, r.op, _cache_slot(out))
                return _step_call(j, rewritten.args[2:end]...)
            end
            j = _step!(prog, r.op, nothing)
            return _plain_call(j, rewritten.args[2:end]...)
        end
    end
    rewritten
end

# --- exemplar-typed preparation ---------------------------------------------
# Cache slots and step selection are fixed at preparation from static types.
# A HAVE port without a concrete declared type leaves everything computed from
# it `Any`-typed, where the native kernel is specialized on the runtime
# argument types instead. Exemplar arguments supply those runtime types: they
# type the program exactly as the arguments of later calls will be typed.
function _nonalloc_exemplar_types(p::Plan, exemplars::Tuple)
    isempty(exemplars) && return nothing
    length(exemplars) == length(p.have) || throw(ArgumentError(
        "prepare_nonallocating received $(length(exemplars)) exemplar " *
        "arguments for $(length(p.have)) HAVE ports " *
        "($(join((v.name for v in p.have), ", "))); pass one value per " *
        "positional HAVE port, in that order"))
    map(Tuple(p.have), exemplars) do port, value
        typeof(value) <: valtype(port) || throw(ArgumentError(
            "the exemplar for HAVE port `$(port.name)` has type " *
            "$(typeof(value)), which is not a $(valtype(port))"))
        typeof(value)
    end
end

# An exemplar-typed program only accepts arguments of the exemplars' types:
# its cache slots are typed for them, so annotate the HAVE signature with them.
function _nonalloc_exact_signature(ast::Expr, have_types::Tuple)
    signature = ast.args[1]
    ports = signature.args[4:end]
    length(ports) == length(have_types) || throw(ArgumentError(
        "non-allocating preparation produced $(length(ports)) HAVE arguments " *
        "for $(length(have_types)) exemplar types"))
    typed = map(ports, have_types) do port, T
        name = port isa Expr && port.head === :(::) ? port.args[1] : port
        Expr(:(::), name, T)
    end
    Expr(:function, Expr(:tuple, signature.args[1:3]..., typed...), ast.args[2])
end

"""
    _nonallocating_program(p::Plan, ast::Expr) -> (ast, ops, caches)

Rewrite the un-embedded lowering of `p` into the non-allocating step program:
per-step operations and typed persistent caches, with fused captured sources
decomposed into destination-passing steps where the grammar allows, and a
whole-recipe step as the universal fallback. Only steps with an owned,
destination-filled result keep a cache (`_nonalloc_destination`).
"""
function _nonallocating_program(p::Plan, ast::Expr; have_types=nothing,
                                cache_apply=nothing)
    ast.head === :function ||
        throw(ArgumentError("non-allocating preparation requires a function Expr"))
    signature = ast.args[1]
    signature isa Expr && signature.head === :tuple &&
        !isempty(signature.args) && first(signature.args) === _OPS_ARG ||
        throw(ArgumentError("non-allocating preparation requires the lowered __ops__ signature"))

    prog = _StepProgram(Any[], Any[], cache_apply)
    types = Dict{Symbol,Any}()
    for (index, (v, argexpr)) in enumerate(zip(p.have, signature.args[2:end]))
        name = argexpr isa Expr && argexpr.head === :(::) ?
               argexpr.args[1] : argexpr
        name isa Symbol && (types[name] = have_types === nothing ? valtype(v) : have_types[index])
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
