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
#   steps; the unary minus of an array operand joins the broadcast lazily;
# - `vcat` becomes `_ConcatenateStep`, with vector literals of numbers
#   (`vcat([a], [b], v)`) concatenated as their elements, and `zeros`/`ones` become
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
# The product the native body computes (`_native_mul`): Base's, or its own loop
# for an operand BLAS cannot take.
@inline (::_MatMulStep)(A, B) = _native_mul(A, B)

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
# The decomposition's own step operations carry destination-passing methods in
# the extension. A partitioned plate's lane gather (`_LaneGather`) is not one:
# it returns a view of its argument, which allocates nothing, so it is a plain
# call, as in the ordinary program. Copying it into an owned vector instead
# cost a gather per call and a scatter in the reverse pass, and left the seeds
# models' non-allocating gradients 3-5 % slower than the ordinary ones.
_nonalloc_fresh_step(op) = false
_nonalloc_fresh_step(::Union{_MaterializeStep,_GatherStep,_MatMulStep,
                             _ConcatenateStep,_FillConstructorStep,
                             _RowReduceStep}) = true

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
    Expr(:ref, _CACHES_ARG, j), Expr(:ref, _CALLABLES_ARG, j), args...)
_plain_call(j::Int, args...) = Expr(:call, Expr(:ref, _CALLABLES_ARG, j), args...)

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

# The type later steps assume for a decomposed recipe's value. The plan's type
# (inference over the whole recipe) and the decomposition's (assembled from its
# parts) both bound the value's type from above, so their intersection does.
# The decomposition's can be the wider one, `Any` for a part inference cannot
# type alone, and it replaced the plan's: a `log.(x)` step over an `x` typed
# `Any` that way kept a `Ref{Any}` slot (`_nonalloc_step_type`). Should the
# two ever disagree, the decomposition is the code that runs: its type.
function _nonalloc_tighter_type(planned, decomposed)
    planned isa Type && decomposed isa Type || return decomposed
    T = typeintersect(planned, decomposed)
    T === Union{} ? decomposed : T
end

# A square dense matrix plus a `UniformScaling` (`K + sigma * I`), as
# LinearAlgebra computes it: a copy of the matrix with `λ` added along its
# diagonal, into a matching cache. Copied element by element, the loop native
# Enzyme reverse differentiates. Another shape, element type or a cache that
# may share memory with the matrix takes Base's own call, and its errors. The
# primal step (the MutatingFunctions extension's `apply!!`) and the
# differentiated one (`_ad_na_arith_loop!`) both fill through it.
@inline function _uniform_add_into!(cache::Matrix{T}, A::Matrix{T},
                                    J::UniformScaling) where {T}
    (Base.promote_op(+, T, typeof(J.λ)) === T && size(A, 1) == size(A, 2) &&
     size(cache) == size(A) && !Base.mightalias(cache, A)) || return A + J
    @inbounds for i in eachindex(A)
        cache[i] = A[i]
    end
    @inbounds for i in 1:size(A, 1)
        cache[i, i] += J.λ
    end
    cache
end
@inline _uniform_add_into!(cache, A, J::UniformScaling) = A + J

_nonalloc_slot(::Type{T}) where {T} = Ref{Union{Nothing,T}}(nothing)
_nonalloc_slot(op, ::Type{T}) where {T} = _nonalloc_slot(T)

# The type of the value a slot of `Ref{S}` holds once seeded: `Any` for the
# `Ref{Any}` of a step whose type preparation could not infer.
_nonalloc_slot_value(::Type{Union{Nothing,T}}) where {T} = T
_nonalloc_slot_value(::Type{T}) where {T} = T

# The type `op(args...)` infers to, folded at compile time (as `_promote_op`):
# the type the native program's plain call of a step has. A cache driver
# reading a slot whose value type is not concrete returns its result as this
# type (`_nonalloc_step_type`), so the slot keeps its buffer without hiding
# the result's type from the rest of the program.
@generated _nonalloc_call_type(op, args::Vararg{Any,N}) where {N} =
    :(Base._return_type(op, $(Tuple{args...})))
@inline _nonalloc_step_type(::Type{S}, op, args...) where {S} =
    isconcretetype(_nonalloc_slot_value(S)) ? nothing : _nonalloc_call_type(op, args...)

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
    # A generated source names its helpers by `GlobalRef` (RKPPL's gathers).
    if callee isa GlobalRef
        r = _nonalloc_resolve_const(callee.mod, callee.name)
        r === nothing && return nothing
        return (something(r), callee)
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
    # A generated source's global by `GlobalRef` (`LinearAlgebra.I`): the
    # constant it names, as a bare name resolves above.
    if node isa GlobalRef
        resolved = _nonalloc_resolve_const(node.mod, node.name)
        resolved === nothing && return nothing
        return (node, typeof(something(resolved)))
    end
    node isa Expr || return (node, typeof(node))
    if node.head === :call && !isempty(node.args)
        callee = node.args[1]
        callee isa Symbol || callee isa Expr || callee isa GlobalRef ||
            return nothing
        callee isa Symbol ||
            _nonalloc_resolve_function(ctx.mod, callee) !== nothing ||
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
    (node.head === :if || node.head === :elseif) && 2 <= length(node.args) <= 3 &&
        return _decompose_branch(ctx, node)
    nothing
end

# `if c; a; else; b; end`, as an authored branch (`_KernelBranch`) spells it:
# the condition, then each arm decomposed into statements of its own, which
# run only when the arm is taken. A missing `else` arm is `nothing`.
function _decompose_branch(ctx::_FusedDecomposition, node::Expr)
    marks = (length(ctx.ops), length(ctx.caches))
    parts = Any[]
    types = Any[]
    for (i, part) in enumerate(node.args)
        d = _decompose_arm(ctx, part)
        if d === nothing
            resize!(ctx.ops, marks[1])
            resize!(ctx.caches, marks[2])
            return nothing
        end
        push!(parts, d[1])
        # An arm that throws (an argument check's `else throw(...)`) returns
        # no value, so the branch has the other arms' type.
        i == 1 || _nonalloc_throws(ctx, part) || push!(types, d[2])
    end
    length(node.args) == 2 && push!(types, Nothing)
    (Expr(node.head, parts...), isempty(types) ? Any : Union{types...})
end

# Whether an arm's value is a call of `throw`, which never returns.
function _nonalloc_throws(ctx::_FusedDecomposition, ex)
    lines = _nonalloc_straight_line(ex)
    lines === nothing && return false
    final = lines[2]
    final isa Expr && final.head === :call && !isempty(final.args) || return false
    callee = final.args[1]
    callee isa Symbol && haskey(ctx.argmap, callee) && return false
    resolved = _nonalloc_resolve_function(ctx.mod, callee)
    resolved !== nothing && first(resolved) === Core.throw
end

# One arm (or condition) of a branch as a block of its own statements and its
# value. Its names stay inside it.
function _decompose_arm(ctx::_FusedDecomposition, ex)
    lines = _nonalloc_straight_line(ex)
    lines === nothing && return nothing
    statements, final = lines
    arm = _FusedDecomposition(ctx.mod, copy(ctx.argmap), copy(ctx.argtypes),
                              Any[], ctx.ops, ctx.caches, ctx.offset,
                              ctx.cache_apply)
    all(st -> _decompose_traced_statement!(arm, st), statements) || return nothing
    d = _decompose(arm, final, false)
    d === nothing && return nothing
    (isempty(arm.stmts) ? d[1] : Expr(:block, arm.stmts..., d[1]), d[2])
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
    f === Base.vcat && any(_nonalloc_vect_operand, rawargs) &&
        return _decompose_vcat_numbers(ctx, rawargs)
    if (f === Base.sum || f === Base.prod ||
        f === Base.minimum || f === Base.maximum) && length(rawargs) == 1
        reduced = _decompose_singleton_reduction(ctx, rawargs[1])
        reduced !== nothing && return reduced
    end
    decomposed = _decompose_arguments(ctx, rawargs, false)
    decomposed === nothing && return nothing
    args, types = decomposed
    T = _static_type(f, types...)
    _nonalloc_is_lazy(f) && return (Expr(:call, fref, args...), T)
    bits = isconcretetype(T) && isbitstype(T)
    if bits || T isa DataType && isconcretetype(T) && !ismutabletype(T)
        # An immutable value owns no reusable buffer: a cache step would only
        # store it back through the passthrough fallback, so emit the
        # construction inline exactly as the fused closure would run it. This
        # covers wrapper structs, tuples, and other immutable constructors;
        # mutable results (arrays, refs, dicts) keep their cache steps below.
        # A registered destination method of an immutable result fills the
        # buffers it holds (RKPPL's simplex and its log-Jacobian), and a
        # `@traceable` helper's body may build arrays of its own (an
        # `MvNormal` density's factor).
        !bits && _nonalloc_destination(ctx.cache_apply, f, T, types) &&
            return (_emit_step!(ctx, f, T, args...), T)
        traced = _decompose_traceable(ctx, f, args, types)
        traced === nothing || return traced
        return (Expr(:call, fref, args...), T)
    end
    T isa Type || return nothing
    f === Base.vcat &&
        return (_emit_step!(ctx, _ConcatenateStep(f), T, args...), T)
    if (f === Base.zeros || f === Base.ones) && !isempty(types) &&
       all(t -> t isa Type && t <: Integer, types)
        return (_emit_step!(ctx, _FillConstructorStep(f), T, args...), T)
    end
    if f === Base.:* || f === _native_mul
        length(types) == 2 &&
            all(t -> t isa Type && isconcretetype(t) &&
                     t <: AbstractVecOrMat, types) ||
            return nothing
        return (_emit_step!(ctx, _MatMulStep(), T, args...), T)
    end
    if f === _native_broadcast_materialize && length(args) == 1 &&
       T isa DataType && isconcretetype(T) && T <: Array
        return (_emit_step!(ctx, _MaterializeStep(), T, args...), T)
    end
    # The in-place layer's `\` and `/` coverage shares `mul!`'s pre-sized
    # destination convention; without a guarded step, keep the whole-recipe
    # fallback rather than risking a stale destination on a shape change.
    (f === Base.:\ || f === Base.:/) && return nothing
    _nonalloc_destination(ctx.cache_apply, f, T, types) &&
        return (_emit_step!(ctx, f, T, args...), T)
    # A `@traceable` helper decomposes as its recorded source over these
    # arguments, as other source-reading compilers translate it.
    traced = _decompose_traceable(ctx, f, args, types)
    traced === nothing || return traced
    # A call without a destination-passing method returns its own result,
    # which may alias an argument: run it inline, never as a cache step.
    (Expr(:call, fref, args...), T)
end

# The authored body of the `@traceable` method that arguments of `types` call
# (`_traceable_source`), decomposed with its formals bound to the decomposed
# arguments and its globals resolved in its own module. Each argument is
# evaluated once, as in the call. `nothing`, with no step added, when the body
# is outside the grammar: the caller then runs the call itself.
function _decompose_traceable(ctx::_FusedDecomposition, f, args, types)
    all(t -> t isa Type, types) || return nothing
    source = _traceable_source(typeof(f), types...)
    source === nothing && return nothing
    names, body, mod, traced = source
    # The source is the `@traceable` method's. A more specific plain method
    # (a `Float64` specialization, say) is the one these arguments call, so
    # the source stands for the call only when its method is the one called.
    called = try
        which(f, Tuple{types...})
    catch
        return nothing
    end
    called.sig == traced || return nothing
    formal_types = _traceable_arguments(source, types, rest -> Tuple{rest...})
    formal_types === nothing && return nothing
    lines = _nonalloc_straight_line(body)
    lines === nothing && return nothing
    statements, body = lines
    marks = (length(ctx.stmts), length(ctx.ops), length(ctx.caches))
    bound = Any[]
    for arg in args
        if arg isa Symbol || arg isa GlobalRef || !(arg isa Expr)
            push!(bound, arg)
        else
            tmp = gensym(:traced_argument)
            push!(ctx.stmts, Expr(:(=), tmp, arg))
            push!(bound, tmp)
        end
    end
    # A Vararg formal is the tuple of the trailing arguments, as in the call.
    bound = _traceable_arguments(source, bound, rest -> begin
        tmp = gensym(:traced_varargs)
        push!(ctx.stmts, Expr(:(=), tmp, Expr(:tuple, rest...)))
        tmp
    end)
    inner = _FusedDecomposition(mod, Dict{Symbol,Any}(zip(names, bound)),
                                Dict{Symbol,Any}(zip(names, formal_types)),
                                ctx.stmts, ctx.ops, ctx.caches, ctx.offset,
                                ctx.cache_apply)
    d = all(st -> _decompose_traced_statement!(inner, st), statements) ?
        _decompose(inner, body, false) : nothing
    d === nothing || return d
    resize!(ctx.stmts, marks[1])
    resize!(ctx.ops, marks[2])
    resize!(ctx.caches, marks[3])
    nothing
end

# A body's statements before its final expression, past line numbers and
# nested blocks: `(statements, final)`, or `nothing` for an empty body. A
# final `return x` is its value `x`.
function _nonalloc_straight_line(ex)
    while ex isa Expr && ex.head === :block
        exprs = filter(a -> !(a isa LineNumberNode), ex.args)
        isempty(exprs) && return nothing
        length(exprs) == 1 ||
            return (exprs[1:end-1], _nonalloc_returned(exprs[end]))
        ex = only(exprs)
    end
    ex isa LineNumberNode ? nothing : (Any[], _nonalloc_returned(ex))
end

_nonalloc_returned(ex) = ex isa Expr && ex.head === :return &&
    length(ex.args) == 1 ? only(ex.args) : ex

# One statement of a traced body, run where the helper's call runs it: an
# assignment `x = value` or `a, b = x, y` binds its names to the decomposed
# values, each evaluated once, and a call (an argument check, say) runs for
# its effect. `false` for anything else, such as a loop or a branch.
function _decompose_traced_statement!(ctx::_FusedDecomposition, st)
    st isa Expr || return false
    if st.head === :(=) && st.args[1] isa Symbol
        return _bind_traced!(ctx, st.args[1], st.args[2])
    elseif st.head === :(=) && st.args[1] isa Expr && st.args[1].head === :tuple &&
           st.args[2] isa Expr && st.args[2].head === :tuple &&
           length(st.args[1].args) == length(st.args[2].args) &&
           all(n -> n isa Symbol, st.args[1].args)
        # Every right-hand value is evaluated before any name is bound.
        values = Any[]
        for value in st.args[2].args
            d = _decompose(ctx, value, false)
            d === nothing && return false
            push!(values, _traced_value!(ctx, d))
        end
        for (name, value) in zip(st.args[1].args, values)
            ctx.argmap[name], ctx.argtypes[name] = value
        end
        return true
    elseif st.head === :call
        d = _decompose(ctx, st, false)
        d === nothing && return false
        d[1] isa Symbol || push!(ctx.stmts, Expr(:(=), gensym(:traced_statement), d[1]))
        return true
    end
    false
end

function _bind_traced!(ctx::_FusedDecomposition, name::Symbol, value)
    d = _decompose(ctx, value, false)
    d === nothing && return false
    ctx.argmap[name], ctx.argtypes[name] = _traced_value!(ctx, d)
    true
end

# A decomposed value as an operand later statements read: a name or a
# constant as it is, any other expression evaluated once, here.
function _traced_value!(ctx::_FusedDecomposition, (expr, T))
    (expr isa Symbol || expr isa GlobalRef || !(expr isa Expr)) && return (expr, T)
    tmp = gensym(:traced_value)
    push!(ctx.stmts, Expr(:(=), tmp, expr))
    (tmp, T)
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
    args = Any[]
    types = Any[]
    for raw in rawargs
        d = _decompose_broadcast_operand(ctx, raw)
        d === nothing && return nothing
        push!(args, d[1])
        push!(types, d[2])
    end
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

# An operand of a dotted call. Base defines the unary minus of an array as
# its elementwise negation (`broadcast_preserving_zero_d(-, A)`), so inside
# the enclosing broadcast an operand `-A`, as in `log1pexp.(-x)`, reads the
# same values as `.-A`: it joins the broadcast lazily instead of materializing
# a negated copy on every call. Every other operand decomposes as before.
function _decompose_broadcast_operand(ctx::_FusedDecomposition, raw)
    if raw isa Expr && raw.head === :call && length(raw.args) == 2 &&
       raw.args[1] === :-
        rf = _nonalloc_resolve_function(ctx.mod, :-)
        if rf !== nothing && rf[1] === Base.:-
            marks = (length(ctx.stmts), length(ctx.ops), length(ctx.caches))
            d = _decompose(ctx, raw.args[2], false)
            d === nothing && return nothing
            A = d[2]
            if A isa Type && isconcretetype(A) && A <: AbstractArray{<:Number}
                BT = _static_type(Base.Broadcast.broadcasted, typeof(Base.:-), A)
                return (Expr(:call, GlobalRef(Base.Broadcast, :broadcasted),
                             rf[2], d[1]), BT)
            end
            # Not an array of numbers: drop the operand's steps and decompose
            # the call as any other.
            resize!(ctx.stmts, marks[1])
            resize!(ctx.ops, marks[2])
            resize!(ctx.caches, marks[3])
        end
    end
    _decompose(ctx, raw, true)
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

# A vector literal `[a, b, ...]` without splats, keywords or rows.
_nonalloc_vect_operand(node) = node isa Expr && node.head === :vect &&
    !any(a -> a isa Expr && a.head in (:..., :parameters, :kw), node.args)

# `vcat([a, b], v)`, as RKPPL writes a coefficient vector of scalar
# parameters (`vcat([alpha], [beta])`) or a scan trajectory after its initial
# values (`vcat([x1, x2], rest)`). Julia builds each literal as a fresh vector
# before concatenating it. When the literal's elements are numbers whose
# promoted type is the element type of the result, those elements are the
# result's rows, each converted once to that type, which is what
# `vcat(a, b, v)` computes. The step concatenates them with the native body's
# `vcat` (`_native_vcat`), which native Enzyme reverse differentiates over
# numbers beside vectors, so neither the literals nor the result are allocated
# on each call. Any other literal keeps the whole-recipe fallback.
function _decompose_vcat_numbers(ctx::_FusedDecomposition, rawargs)
    operands = Any[]       # the step's arguments
    operand_types = Any[]
    authored_types = Any[] # the types `vcat` sees in the source
    literal_eltypes = Any[]
    for raw in rawargs
        if _nonalloc_vect_operand(raw)
            isempty(raw.args) && return nothing
            decomposed = _decompose_arguments(ctx, raw.args, false)
            decomposed === nothing && return nothing
            args, types = decomposed
            all(t -> t isa Type && isconcretetype(t) && t <: Number, types) ||
                return nothing
            P = promote_type(types...)
            push!(literal_eltypes, P)
            push!(authored_types, Vector{P})
            append!(operands, args)
            append!(operand_types, types)
        else
            d = _decompose(ctx, raw, false)
            d === nothing && return nothing
            push!(operands, d[1])
            push!(operand_types, d[2])
            push!(authored_types, d[2])
        end
    end
    T = _static_type(Base.vcat, authored_types...)
    T isa DataType && isconcretetype(T) && T <: Vector && isbitstype(eltype(T)) ||
        return nothing
    all(P -> P === eltype(T), literal_eltypes) || return nothing
    _static_type(_native_vcat, operand_types...) === T || return nothing
    (_emit_step!(ctx, _ConcatenateStep(_native_vcat), T, operands...), T)
end

# Try to decompose one fused-source recipe statement. Returns
# `(stmts, result_type)` or `nothing` (fall back to the whole-recipe step).
function _decompose_fused_recipe!(prog::_StepProgram, r::Recipe, callargs,
                                  input_types, lhs)
    op = r.op
    kernel_sourceop_form(op) === :fused || return nothing
    # The source reads its inputs by its parameter names, not by the input
    # values' names (`_lane_source_params`); a capturing source keeps its step.
    # A lazy branch's source is its `if`, over its all-in-one closure's
    # parameters (`_lane_source_callable`).
    params = _lane_source_params(_lane_source_callable(getfield(op, :f)), r)
    params === nothing && return nothing
    ctx = _FusedDecomposition(_kernel_endpoint_source_module(op),
                              _lane_source_arguments(params, callargs),
                              _lane_source_arguments(params, input_types),
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

# The recorded result type of an authored plate uses the inferred concrete
# element type (`_authored_plate_result_eltype`) rather than the plan-level
# `Any` of an unannotated body, as the ordinary native lowering does.
_nested_plate_cache(op, argtypes, T, N) = nothing
function _nonalloc_plan_result_types(kernel, argument_types)
    p = kernel.plan
    types = _nonalloc_value_types(p, argument_types)
    Tuple(get(types, canon_id(p.graph, v.id), Any) for v in p.want)
end
# Static types of every value of `p`, keyed by canonical value id, from the
# HAVE argument types.
function _nonalloc_value_types(p::Plan, argument_types)
    _plan_value_types(p, argument_types) do recipe, inputs
        _recipe_output_types(recipe, _nonalloc_recipe_type(recipe.op, inputs,
            length(recipe.outputs) == 1 ? valtype(only(recipe.outputs)) : Any))
    end
end
_nonalloc_recipe_type(op, types, declared::Type = Any) = _static_type(op, types...)
function _nonalloc_recipe_type(op::_AuthoredScanOp{K,A,I,H}, types,
                               declared::Type = Any) where {K,A,I,H}
    H && return _static_type(op, types...)
    output = last(_nonalloc_plan_result_types(op.kernel,
                                              _scan_step_argument_types(op, types)))
    _scan_result_type(op, types, output)
end
_nonalloc_recipe_type(op::_AuthoredPlateOp, types, declared::Type = Any) =
    _plate_result_type(op, types, declared)

# A declared array output (`y::Vector{Float64} = plate(...)`) fixes the
# element type and rank the native kernel's typed local converts the plate's
# result to (`_declare_typed_output!`). Returns `(eltype, rank)`, either
# `nothing` when the declaration leaves it open.
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

function _plate_result_type(op::_AuthoredPlateOp{K,A}, argtypes,
                            declared::Type = Any) where {K,A}
    T = _nonalloc_plate_eltype(op, argtypes, declared)
    N = _nonalloc_plate_rank(op, argtypes, declared)
    N === nothing ? Array{T} : Array{T,N}
end

# --- program builder ---------------------------------------------------------

# A plate whose cells return arrays stays one cache step: its slot
# (`_nested_plate_cache`, the MutatingFunctions extension) reuses the cells'
# arrays as well as the outer array, where the native plate lowering would
# allocate every cell's array afresh. Wrapping the plate keeps the lowering
# from inlining it; calling the wrapper is the plate's own call.
struct _NonallocNestedPlate{P,S}
    plate::P
    slot::S
end
(op::_NonallocNestedPlate)(args...) = op.plate(args...)

function _nonalloc_nested_plates(p::Plan, types::AbstractDict)
    g = p.graph
    changed = false
    recipes = map(p.recipes) do r
        r.op isa _AuthoredPlateOp && length(r.outputs) == 1 || return r
        argtypes = Any[get(types, canon_id(g, v.id), Any) for v in r.inputs]
        declared = valtype(only(r.outputs))
        slot = _nested_plate_cache(r.op, argtypes,
            _nonalloc_plate_eltype(r.op, argtypes, declared),
            _nonalloc_plate_rank(r.op, argtypes, declared))
        slot === nothing && return r
        changed = true
        Recipe(r.id, r.inputs, r.outputs, _NonallocNestedPlate(r.op, slot),
               r.cost, r.cse_key, r.effectful, r.source)
    end
    changed || return p
    producer = Dict(canon_id(g, v.id) => r for r in recipes for v in r.outputs)
    Plan(g, p.have, p.want, recipes, producer, p.cost, p.candidates)
end

# A lane slot: the kernel-owned buffer of one materialized authored plate or
# scan output, top-level dotted call or array slice (the `recycle` protocol of
# `_lower_with_ops`, `_lane_reuse`/`_lane_keep!`). Its producer fills it in
# place once it holds a buffer of the right element type and axes, and keeps
# its first allocation otherwise. Its operation-table entry is never called.
struct _NonallocLane end

function _nonalloc_lanes(p::Plan, types::AbstractDict)
    g = p.graph
    have_ids = Set(canon_id(g, v.id) for v in p.have)
    recycle = Pair{ValueId,Symbol}[]
    for r in p.recipes
        # An effect may retain the value it was given; its storage stays fresh.
        length(r.outputs) == 1 && !r.effectful || continue
        cid = canon_id(g, only(r.outputs).id)
        cid in have_ids && continue
        # A scalar slice `x[i]` is a value, not storage.
        T = get(types, cid, Any)
        T isa Type && isconcretetype(T) && isbitstype(T) && continue
        get(p.producer, cid, nothing) === r || continue
        any(pair -> first(pair) == cid, recycle) && continue
        # Fused sources take the step rule instead: their decomposition also
        # reuses the buffers of the calls inside a dotted expression.
        r.op isa Union{_AuthoredPlateOp,_AuthoredScanOp} || continue
        push!(recycle, cid => gensym(Symbol(only(r.outputs).name, :_lane)))
    end
    recycle
end

function _nonalloc_argument_types(types::Dict{Symbol,Any}, callargs)
    Any[a isa Symbol ? get(types, a, Any) :
        a isa Expr ? Any : typeof(a) for a in callargs]
end

# One top-level statement of the native program. An operation call
# `lhs = (__callables__[i])(callargs...)` takes the step rule; everything else
# (a constant's read, the inlined plates, scans and cells, lane-filling
# sources, control flow) is kept.
function _nonalloc_statement!(prog::_StepProgram, recipes::Tuple,
                              types::Dict{Symbol,Any}, stmt)
    stmt isa Expr && stmt.head === :(=) && stmt.args[1] isa Symbol &&
        stmt.args[2] isa Expr && stmt.args[2].head === :call || return Any[stmt]
    call = stmt.args[2]
    call.args[1] == GlobalRef(@__MODULE__, :_cell_array_broadcast) &&
        length(call.args) == 2 &&
        return _nonalloc_cell_broadcast!(prog, types, stmt.args[1], call.args[2])
    i = _callable_slot(call.args[1])
    i !== nothing && 1 <= i <= length(recipes) &&
        length(recipes[i].outputs) == 1 || return Any[stmt]
    _nonalloc_step!(prog, recipes[i], i, types, stmt.args[1],
                    call.args[2:end], stmt)
end

# A broadcast that plate cells only index is spelled `_cell_array_broadcast(bc)`
# (`_lower_cell_arrays_native`): lazy over total arithmetic on dense floating
# storage, otherwise materialized fresh, as `log.(w)` is. When its operand types
# make it materialize, it becomes a materialization step with a buffer of its
# own, which `copyto!`s the same elements.
function _nonalloc_cell_broadcast!(prog::_StepProgram, types::Dict{Symbol,Any},
                                   lhs::Symbol, bc)
    # This path receives native source rather than a recipe call. Decompose
    # its eager operands too: an inline product (and a materialized operand
    # of that product) needs the same destination as a named statement, even
    # when the outer broadcast stays lazy for its elementwise readers.
    ctx = _FusedDecomposition(@__MODULE__, Dict{Symbol,Any}(s => s for s in keys(types)),
                              types, Any[], Any[], Any[], length(prog.ops),
                              prog.cache_apply)
    d = _decompose(ctx, bc, false)
    if d !== nothing
        bc, BT = d
        append!(prog.ops, ctx.ops)
        append!(prog.caches, ctx.caches)
    else
        BT = _nonalloc_expr_type(types, bc)
    end
    T = BT isa Type ? _static_type(_cell_array_broadcast, BT) : Any
    stmts = d === nothing ? Any[] : ctx.stmts
    types[lhs] = T
    if !(T isa DataType && isconcretetype(T) && T <: Array)
        push!(stmts, Expr(:(=), lhs, Expr(:call,
            GlobalRef(@__MODULE__, :_cell_array_broadcast), bc)))
        return stmts
    end
    j = _step!(prog, _MaterializeStep(), _nonalloc_slot(T))
    push!(stmts, Expr(:(=), lhs, _step_call(j,
        Expr(:call, GlobalRef(@__MODULE__, :_cell_array_materialize), bc))))
    stmts
end

# The static type of a program expression over names of known value types:
# `Any` when a name, a callee or a call's result is unknown.
function _nonalloc_expr_type(types::Dict{Symbol,Any}, ex)
    ex isa Symbol && return get(types, ex, Any)
    ex isa GlobalRef && return isdefined(ex.mod, ex.name) && isconst(ex.mod, ex.name) ?
        typeof(getglobal(ex.mod, ex.name)) : Any
    ex isa QuoteNode && return typeof(ex.value)
    ex isa Expr || return typeof(ex)
    ex.head === :call && ex.args[1] isa GlobalRef || return Any
    f = ex.args[1]
    isdefined(f.mod, f.name) && isconst(f.mod, f.name) || return Any
    _static_type(getglobal(f.mod, f.name),
                 (_nonalloc_expr_type(types, a) for a in ex.args[2:end])...)
end

# The step rule: fused captured sources decompose into destination-passing
# steps where the grammar allows; isbits results and identity-preserving
# wrappers are called directly; an operation whose result is fresh storage that
# a destination-passing method fills (`_nonalloc_destination`) keeps a cache
# slot at its own table index; every other operation is called directly.
function _nonalloc_step!(prog::_StepProgram, r::Recipe, i::Int,
                         types::Dict{Symbol,Any}, lhs::Symbol, callargs, stmt)
    op = prog.ops[i]
    if op isa _NonallocNestedPlate
        prog.ops[i] = op.plate
        prog.caches[i] = op.slot
        return Any[Expr(:(=), lhs, _step_call(i, callargs...))]
    end
    argtypes = _nonalloc_argument_types(types, callargs)
    T = _static_type(op, argtypes...)
    if !r.effectful
        if r.op isa _KernelSourceOp && !(r.source isa _NoKernelSource)
            dec = _decompose_fused_recipe!(prog, r, callargs, argtypes, lhs)
            if dec !== nothing
                types[lhs] = _nonalloc_tighter_type(get(types, lhs, Any), dec[2])
                return dec[1]
            end
        end
        if isconcretetype(T) && isbitstype(T)
            # A `@traceable` helper with a numeric result may still build
            # arrays in its body.
            if !(r.op isa _KernelSourceOp)
                dec = _decompose_traceable_recipe!(prog, op, callargs, argtypes, lhs)
                if dec !== nothing
                    types[lhs] = _nonalloc_tighter_type(get(types, lhs, Any), dec[2])
                    return dec[1]
                end
            end
            return Any[stmt]
        end
        !(r.op isa _KernelSourceOp) && _nonalloc_is_lazy(op) && return Any[stmt]
    end
    slot_type = T isa Type && isconcretetype(T) ? T : valtype(only(r.outputs))
    if _nonalloc_destination(prog.cache_apply, op, slot_type, argtypes)
        slot = _nonalloc_slot(op, slot_type)
        slot === nothing && return Any[stmt]
        prog.caches[i] = slot
        return Any[Expr(:(=), lhs, _step_call(i, callargs...))]
    end
    if !r.effectful && !(r.op isa _KernelSourceOp)
        dec = _decompose_traceable_recipe!(prog, op, callargs, argtypes, lhs)
        if dec !== nothing
            types[lhs] = _nonalloc_tighter_type(get(types, lhs, Any), dec[2])
            return dec[1]
        end
    end
    Any[stmt]
end

# A pure recipe whose operation is a `@traceable` helper, such as RKPPL's
# gather of a ranged response operand, decomposes as the helper's source
# (`_decompose_traceable`), so its array results get steps with buffers of
# their own instead of a fresh result on every call.
function _decompose_traceable_recipe!(prog::_StepProgram, op, callargs,
                                      input_types, lhs)
    ctx = _FusedDecomposition(@__MODULE__, Dict{Symbol,Any}(), Dict{Symbol,Any}(),
                              Any[], Any[], Any[], length(prog.ops),
                              prog.cache_apply)
    d = _decompose_traceable(ctx, op, collect(Any, callargs), collect(Any, input_types))
    d === nothing && return nothing
    append!(prog.ops, ctx.ops)
    append!(prog.caches, ctx.caches)
    stmts = ctx.stmts
    push!(stmts, Expr(:(=), lhs, d[1]))
    (stmts, d[2])
end

# The cache slots a program reads.
function _nonalloc_cache_reads!(slots::Set{Int}, node)
    node isa Expr || return slots
    j = _table_slot(node, _CACHES_ARG)
    j === nothing ? foreach(arg -> _nonalloc_cache_reads!(slots, arg), node.args) :
        push!(slots, j)
    slots
end

function _nonalloc_mentions!(names::Set{Symbol}, node)
    if node isa Symbol
        push!(names, node)
    elseif node isa Expr
        foreach(arg -> _nonalloc_mentions!(names, arg), node.args)
    end
    names
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
    ports = signature.args[5:end]
    length(ports) == length(have_types) || throw(ArgumentError(
        "non-allocating preparation produced $(length(ports)) HAVE arguments " *
        "for $(length(have_types)) exemplar types"))
    typed = map(ports, have_types) do port, T
        name = port isa Expr && port.head === :(::) ? port.args[1] : port
        Expr(:(::), name, T)
    end
    Expr(:function, Expr(:tuple, signature.args[1:4]..., typed...), ast.args[2])
end

# The names a program body assigns exactly once, with their right-hand sides:
# the definitions a table root's static type is read through.
function _nonalloc_table_definitions(body)
    definitions = Dict{Symbol,Any}()
    repeated = Set{Symbol}()
    _index_run_visit(body) do ex
        ex.head === :(=) && length(ex.args) == 2 && ex.args[1] isa Symbol || return
        haskey(definitions, ex.args[1]) && push!(repeated, ex.args[1])
        definitions[ex.args[1]] = ex.args[2]
    end
    foreach(name -> delete!(definitions, name), repeated)
    definitions
end

# The static type of `x` in a program body (`_nonallocating_program`): a
# value's inferred type, a constant's own type, or what Julia infers for a call
# of a constant function or operation on arguments of such types. `nothing`
# when no concrete type follows.
function _nonalloc_static_type(x, types, definitions, constants, ops,
                               seen = Set{Symbol}())
    x isa Union{Number,Nothing} && return typeof(x)
    x isa QuoteNode && return typeof(x.value)
    if x isa Symbol
        T = get(types, x, nothing)
        T isa Type && isconcretetype(T) && return T
        (x in seen || !haskey(definitions, x)) && return nothing
        return _nonalloc_static_type(definitions[x], types, definitions, constants, ops,
                                     push!(seen, x))
    end
    slot = _constant_slot(x)
    slot === nothing || return typeof(constants[slot])
    x isa Expr && x.head === :call || return nothing
    head = first(x.args)
    slot = _callable_slot(head)
    f = slot !== nothing ? ops[slot] :
        head isa GlobalRef && isdefined(head.mod, head.name) &&
            isconst(head.mod, head.name) ? getglobal(head.mod, head.name) : nothing
    f === nothing && return nothing
    argtypes = Any[]
    for argument in x.args[2:end]
        T = _nonalloc_static_type(argument, types, definitions, constants, ops, copy(seen))
        T === nothing && return nothing
        push!(argtypes, T)
    end
    T = Base.promote_op(f, argtypes...)
    isconcretetype(T) ? T : nothing
end

"""
    _nonallocating_program(p::Plan; have_types, cache_apply, passes) -> (ast, constants, callables, caches)

The non-allocating step program of `p`. It is the native program ordinary
`prepare` compiles (`_lower_with_ops` over `_lowering_plan(p)`): plate chains
compose, summed plates fold their totals, scans stream into the plates they
feed, and plate cells compute the rows they read. Its remaining allocation
sites are routed to kernel-owned storage:

- each materialized authored plate or scan output, top-level dotted call or
  array slice fills a lane slot that keeps its buffer across calls
  (`_nonalloc_lanes`);
- each other top-level operation call takes the step rule
  (`_nonalloc_step!`);
- a plate whose cells return arrays stays one cache step
  (`_nonalloc_nested_plates`).

The tables are the native lowering's, so every inlined `__constants__[i]` and
`__callables__[i]` reference keeps its meaning; decomposition steps and lane
slots are appended to the callables and caches, which share one index.
"""
function _nonallocating_program(p::Plan; have_types = nothing,
                                cache_apply = nothing, passes = ())
    lowered = _lowering_plan(p)
    argument_types = have_types === nothing ?
        map(valtype, Tuple(lowered.have)) : have_types
    values = _nonalloc_value_types(lowered, argument_types)
    lowered = _nonalloc_nested_plates(lowered, values)
    recycle = _nonalloc_lanes(lowered, values)
    # No typed local declarations: the program is host-only, and declaring
    # `W::Matrix{Float64} = reshape(view(u, r), a, b)` would copy the lazy
    # value into a fresh matrix on every call. Declared plate outputs still
    # fix their buffers' element types. The steps below read the names and
    # value types of `lowered`, so its cells outline nothing (`outline`), and
    # the step program keys its caches on the operation calls
    # (`_nonalloc_statement!`), so this emitter keeps every recipe's call in
    # the table rather than its inlined source. Tables of work reused across
    # an enclosing plate's cells (`_lower_indexed_invariants`) keep lane slots
    # too (`table_lanes`).
    table_lanes = Pair{Symbol,Symbol}[]
    ast, constants, callables, recipes = _lower_with_ops(lowered;
                                        exemplar_types = have_types,
                                        declare = false, recycle,
                                        pushdowns = true, outline = false,
                                        plate_element_types = true,
                                        plate_programs = true,
                                        inline_sources = false,
                                        indexed_lanes = table_lanes)
    isempty(passes) || (ast = transform(ast, passes...))
    names = _varnames(lowered)
    types = Dict{Symbol,Any}(names[cid] => T for (cid, T) in values
                             if haskey(names, cid))
    prog = _StepProgram(Any[callables...], Any[nothing for _ in callables], cache_apply)
    body = Expr(:block)
    for stmt in ast.args[2].args
        append!(body.args, _nonalloc_statement!(prog, recipes, types, stmt))
    end
    # The same stable forms as `prepare` (`stable_rewrites.jl`). After the step
    # rule, whose decomposition spells out the fused sources `prepare` inlines:
    # the pass reads a composition or a branch's arm only from its expression.
    if _simplify()
        body = _stable_rewrites(Expr(:function, ast.args[1], body),
            (recipes..., prog.ops[length(recipes)+1:end]...)).args[2]
        # A step it removed holds no buffer.
        read = _nonalloc_cache_reads!(Set{Int}(), body)
        for j in eachindex(prog.caches)
            j in read || (prog.caches[j] = nothing)
        end
    end
    # A value the native lowering never materializes (a summed plate, a scan
    # streamed into its plate, a row computed in a cell) leaves its lane unused.
    mentioned = _nonalloc_mentions!(Set{Symbol}(), body)
    lanes = Any[]
    for (cid, lane) in recycle
        lane in mentioned || continue
        # Typed by the value's inferred array type, so the program reads it
        # concretely (`_lane_reuse`); an uninferred value keeps a `Ref{Any}`.
        A = get(values, cid, Any)
        slot = A isa Type && A <: Array && isconcretetype(A) ?
            Base.RefValue{Union{Nothing,A}}(nothing) : Ref{Any}(nothing)
        j = _step!(prog, _NonallocLane(), slot)
        push!(lanes, Expr(:(=), lane, Expr(:ref, _CACHES_ARG, j)))
    end
    # A table's slot is typed by the array its allocation returns for the
    # root's static type, as a lane is by its value's type; a root whose type
    # is not inferred keeps a `Ref{Any}`.
    definitions = _nonalloc_table_definitions(body)
    for (lane, root) in table_lanes
        lane in mentioned || continue
        R = _nonalloc_static_type(root, types, definitions, constants, prog.ops)
        A = R === nothing ? Any : Base.promote_op(_indexed_table, R)
        slot = A isa Type && A <: Array && isconcretetype(A) ?
            Base.RefValue{Union{Nothing,A}}(nothing) : Ref{Any}(nothing)
        j = _step!(prog, _NonallocLane(), slot)
        push!(lanes, Expr(:(=), lane, Expr(:ref, _CACHES_ARG, j)))
    end
    prepend!(body.args, lanes)
    ports = ast.args[1].args[3:2 + length(lowered.have)]
    (_extract_plate_programs(Expr(:function,
        Expr(:tuple, _CONSTANTS_ARG, _CALLABLES_ARG, _CACHES_ARG,
             _CACHE_APPLY_ARG, ports...), body)),
     constants, Tuple(prog.ops), Tuple(prog.caches))
end
