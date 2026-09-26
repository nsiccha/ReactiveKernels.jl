# Core graph identities: Value, Recipe, Graph.
#
# These objects are *compile/planning-time metadata only*. None of them are
# consulted inside a prepared kernel (see codegen.jl); the hot path sees only
# ordinary Julia values.

# A process-global counter giving every `Value` a stable identity independent of
# its name or of any particular graph. Identity must not depend solely on the
# name (gist §5), so two values may share a name yet remain distinct.
const _VALUE_COUNTER = Ref(0)
_next_value_id() = (_VALUE_COUNTER[] += 1)

"""
    Value{T}

A stable graph identity for a runtime value of Julia type `T`. `id` is the
identity (globally unique); `name` exists only for diagnostics and for
generated-code readability. Values are immutable and cheap to hash/compare.
"""
struct Value{T}
    id::Int
    name::Symbol
end

Value(name::Symbol, ::Type{T}) where {T} = Value{T}(_next_value_id(), name)

"""
    value(name, T)

Construct a standalone `Value{T}` with a fresh global identity. Use `value!` to
also register it into a graph.
"""
value(name::Symbol, ::Type{T}) where {T} = Value(name, T)

"The declared Julia runtime type of a value."
valtype(::Value{T}) where {T} = T

Base.:(==)(a::Value, b::Value) = a.id == b.id
Base.hash(v::Value, h::UInt) = hash(v.id, hash(:ReactiveKernelsValue, h))
Base.show(io::IO, v::Value{T}) where {T} = print(io, v.name, "::", T)

"""
    Recipe

A pure computation mapping input graph values to one or more output graph
values via `op`. RK does not inspect `op` to prove purity: registering an
ordinary recipe asserts this contract. Set `effectful=true` when the operation
is known not to satisfy it; effectful operations are rejected by the stateless
planner and therefore cannot enter a prepared kernel or plate. `cost` is a
deterministic planning hint (not measured runtime). `cse_key`, when
non-`nothing`, opts the operation into structural CSE (gist §8).
`source` is optional authored-RHS metadata for cold-path readable rendering; it
is kept on the planning recipe rather than the executable operation so it never
enters prepared hot-state tuples.
"""
struct _NoKernelSource end
const _NO_KERNEL_SOURCE = _NoKernelSource()

struct Recipe
    id::Int
    inputs::Tuple{Vararg{Value}}
    outputs::Tuple{Vararg{Value}}
    op::Any
    cost::Float64
    cse_key::Any
    effectful::Bool
    source::Any
end
Recipe(id, inputs, outputs, op, cost, cse_key, effectful) =
    Recipe(id, inputs, outputs, op, cost, cse_key, effectful, _NO_KERNEL_SOURCE)

"""
    _KernelSourceOp{DefToken,Form,F,TF}

An immutable wrapper marking a recipe operation SYNTHESIZED from captured `@kernel` source as
COMPILER-OWNED provenance (RK 07:21). Authoring wraps ONLY the anonymous-closure path of
`_kernel_operation` in this; a bare exact identity (`cholesky`/`+`/…) stays raw and is identity/domain
validated unless it has an explicit compiler-owned tensorized replacement. `DefToken` is a
definition-unique gensym baked in at graph build — NOT a security boundary (an internal
constructor/type parameter cannot prevent deliberate internal misuse); it is trusted only because the
supported authoring path is the ONLY thing that wraps a closure, so an arbitrary public Graph closure is
never auto-wrapped. It makes each fused op a distinct concrete type (survives the prepared ops-tuple,
carries no mutable registry). `Form` (RK 07:24)
distinguishes a `:portcall` — a call THROUGH A PORT, `callable(args…)`, whose first input is the callable
source and the rest are ordered args — from a general `:fused` expression, so a prepared handle can
self-derive the DESTINATION contract (a port-call with one owned buffer + one owned scalar output →
`f(dest, args…)::scalar`) from source SHAPE + typed slot roles, never from a name/Recipe id/inspection.
The call forwards INLINE. A RAW anonymous closure inserted into a Graph carries no wrapper and is
rejected as opaque when captured into a prepared handle.
"""
struct _KernelSourceOp{DefToken,Form,F,TF}
    f::F
    tensor_f::TF
end

_KernelSourceOp(::Val{DefToken}, ::Val{Form}, f::F, tensor_f::TF) where
        {DefToken,Form,F,TF} =
    _KernelSourceOp{DefToken,Form,F,TF}(f, tensor_f)
# Preserve the established internal constructor for compiler fixtures and
# already-authored handles; without an alternate body it uses the same callable
# in both modes.
_KernelSourceOp(token::Val, form::Val, f) = _KernelSourceOp(token, form, f, f)

# Optional tracing extensions classify their scalar/array argument types as
# tensorized.  The tuple fold is ordinary Julia dispatch over argument types,
# so it is resolved while tracing rather than becoming data-dependent control
# flow in the compiled program.
@inline _kernel_source_arg_style(arg) = Val(:native)
# A lazy nested broadcast carries its leaves' style, so marker discovery sees
# through fusion to the traced operands inside.
@inline _kernel_source_arg_style(bc::Base.Broadcast.Broadcasted) =
    _kernel_source_style(bc.args)
@inline _kernel_source_merge(::Val{:tensorized}, style) = Val(:tensorized)
@inline _kernel_source_merge(::Val{:native}, style) = style
@inline _kernel_source_style(::Tuple{}) = Val(:native)
@inline function _kernel_source_style(args::Tuple)
    _kernel_source_merge(
        _kernel_source_arg_style(first(args)),
        _kernel_source_style(Base.tail(args)),
    )
end
# `Vararg{Any,N}` forces specialization on the argument count: Julia's
# default heuristic leaves a Vararg that is merely forwarded unspecialized,
# which materializes the arguments as one boxed tuple. Enzyme then meets that
# tuple as a dynamic `jl_f_tuple` and its runtime tuple rule refuses mixed
# activity (a constant array next to active ones) unless runtime activity is
# switched on. Specialized, the arguments stay individual values.
#
# The forwarding itself is spelled out positionally (no `args...` splat):
# Julia 1.12 inference refuses to unsplat a forwarded tuple of more than 32
# elements into a fixed-arity callee, so a fused closure with 33+ inputs
# (the memo joint's 43-argument log-Jacobian) devolves to dynamic
# `jl_apply_generic` dispatch. The primal still runs, but Enzyme cannot
# differentiate through the dynamic call (snag `joint-decl-memo-9642bb45`).
# A direct N-argument call infers on every version — the same remedy as
# `_prepared_call` (codegen.jl) for the 1.12 RGF splat-allocation cliff.
@inline @generated function _kernel_source_call(::Val{:native},
        op::_KernelSourceOp, args::Vararg{Any,N}) where {N}
    forwarded = [:(getfield(args, $index)) for index in 1:N]
    :(op.f($(forwarded...)))
end
@inline @generated function _kernel_source_call(::Val{:tensorized},
        op::_KernelSourceOp, args::Vararg{Any,N}) where {N}
    forwarded = [:(getfield(args, $index)) for index in 1:N]
    :(op.tensor_f($(forwarded...)))
end
@inline (op::_KernelSourceOp)(args::Vararg{Any,N}) where {N} =
    _kernel_source_call(_kernel_source_style(args), op, args...)
kernel_sourceop_token(::_KernelSourceOp{DefToken}) where {DefToken} = DefToken
kernel_sourceop_form(::_KernelSourceOp{DefToken,Form}) where {DefToken,Form} = Form
# Tensorized fused bodies may mix untraced constant arrays with traced operands.
# Base's generic concatenation and broadcast paths can then allocate host
# containers of traced scalars and copy elementwise — forbidden scalar indexing
# on a traced array — so the tensorized body routes both families through these
# wrappers.  A tracing extension specializes `_tensorized_cat_operand` to
# promote untraced array operands against the discovered traced marker; without
# a traced operand the wrappers reduce to the plain Base operations.
@inline _tensorized_cat_operand(marker, arg) = arg
# A lazy nested broadcast promotes leaf-wise: rebuild it with each leaf routed
# through the operand hook, so host leaves lift against the traced marker
# exactly as if each nest level had materialized on its own.
@inline _tensorized_cat_operand(marker, bc::Base.Broadcast.Broadcasted) =
    Base.Broadcast.broadcasted(bc.f,
        map(arg -> _tensorized_cat_operand(marker, arg), bc.args)...)
@inline _tensorized_getindex(array, indices...) = getindex(array, indices...)
@inline function _tensorized_setindex(array, value, indices...)
    setindex!(array, value, indices...)
    array
end
@inline _tensorized_cat_marker(::Tuple{}) = nothing
@inline _tensorized_cat_marker(args::Tuple) = _tensorized_cat_arg_marker(
    _kernel_source_arg_style(first(args)), first(args), Base.tail(args))
@inline _tensorized_cat_arg_marker(::Val{:tensorized}, arg, rest) = arg
@inline _tensorized_cat_arg_marker(::Val{:native}, arg, rest) =
    _tensorized_cat_marker(rest)
@inline function _tensorized_cat_operands(args::Tuple)
    marker = _tensorized_cat_marker(args)
    marker === nothing ? args :
        map(arg -> _tensorized_cat_operand(marker, arg), args)
end
@inline _tensorized_vcat(args...) = vcat(_tensorized_cat_operands(args)...)
@inline _tensorized_hcat(args...) = hcat(_tensorized_cat_operands(args)...)
@inline _tensorized_cat(args...; dims) =
    cat(_tensorized_cat_operands(args)...; dims = dims)
@inline _tensorized_broadcast(f, args...) =
    _tensorized_materialize(
        Base.broadcasted(f, _tensorized_cat_operands(args)...))

# Nested dotted calls stay lazy so Julia's broadcast fusion survives the
# tensorized lowering: only the OUTERMOST dotted call of a nest materializes
# (via `_tensorized_broadcast` above) — with one exception
# (`_tensorized_lazy_materialize` below): a host-only `Bool` nest.
# Materializing every nest level separately changes WHICH broadcast style
# each level compiles under — a fused dense expression such as
# `tril(X, -1) .+ 0.5 .* Diagonal(diag(X))` splits into an isolated
# `0.5 .* Diagonal(...)` whose structured style trips the `fzeropreserving`
# check on traced numbers (snag `reactant-traced-ff8ff365`) — and allocates
# one temporary per level.  The lazy form promotes exactly like the
# materializing one (same `_tensorized_cat_operands`), so host/traced mixes
# lower identically; the enclosing `broadcasted` nests it exactly as Julia's
# own lowering does.
@inline _tensorized_lazy_broadcast(f, args...) =
    _tensorized_lazy_materialize(
        Base.broadcasted(f, _tensorized_cat_operands(args)...))

# `broadcast(f, ...)` materializes a `Bool`-eltype result into a `BitArray`, and
# a tracing backend's `call_with_reactant` recurses without termination on
# `copyto!(::BitArray, ::Broadcasted)` — Reactant 0.2.284 turns a comparison
# mask such as `Δ .>= 0` into a `StackOverflowError` with no actionable signal
# (it bisects to the wrong op and reads like a broken kernel).  A comparison or
# boolean broadcast over host operands carries no traced value, so its result is
# a compile-time constant: materialize it into a dense `Array{Bool}` instead of a
# `BitArray` — identical values, no `BitArray` copyto!.  Non-`Bool` results and
# any broadcast a backend has already promoted to a traced style keep Base's (and
# Reactant's) own materialize, so this is a container-type normalization only and
# leaves value/shape semantics unchanged.  Per user decision `17bnc6t` this is
# normalized in the `@kernel` lowering rather than in Reactant.
@inline _tensorized_materialize(bc) = Base.materialize(bc)
@inline function _tensorized_materialize(
        bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle{N}}
    ) where {N}
    (N >= 1 && Base.Broadcast.combine_eltypes(bc.f, bc.args) === Bool) ?
        collect(bc) : Base.materialize(bc)
end

# A NESTED host-only `Bool` broadcast (a comparison mask fused inside a
# traced expression, e.g. the PPL varying-dummy `(c .== 2)` inside the LP's
# `.+`/`.*` nest) needs the same `BitArray` normalization as the outermost
# level: a tracing backend standalone-materializes each nested argument of a
# traced broadcast (Reactant's `_copyto!` maps `Base.materialize` over
# `bc.args`), and its `copyto!(::BitArray, ::Broadcasted)` overlay re-enters
# itself without termination (Reactant 0.2.284, the same upstream recursion
# the outermost normalization guards — a bare `StackOverflowError` that
# bisects to the wrong op).  The leaf-wise host promotion does NOT save this
# shape: the promotion marker is the outermost call's first tensorized
# argument, and when every traced operand hides inside a lazy nest the marker
# is the host `Broadcasted` wrapper itself — the backend's promotion hook
# (keyed on a genuine traced marker) never fires, so the mask keeps its host
# `DefaultArrayStyle` and materializes to a `BitArray` (snag
# `dummy-varying-xl-3b05117e`: an outermost `.+` over two lazy nests crashes,
# while the same mask beside a DIRECT traced operand promotes and lowers).
# Dense-materialize the nest HERE (`collect` gives `Array{Bool}`): the
# enclosing levels see an ordinary dense host vector — exactly what the
# outermost normalization already feeds them — so values, shapes, and the
# promotion behavior above are unchanged.  Same predicate as
# `_tensorized_materialize` (`DefaultArrayStyle`, `N >= 1`,
# `combine_eltypes === Bool`); scalar (`N == 0`) nests stay lazy (they
# materialize to a `Bool`, never a `BitArray`), and anything a backend
# already promoted to a traced style keeps its lazy form and fusion.  Per
# user decision `17bnc6t` this normalizes in the `@kernel` lowering, not in
# Reactant.
@inline _tensorized_lazy_materialize(bc) = bc
@inline function _tensorized_lazy_materialize(
        bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle{N}}
    ) where {N}
    (N >= 1 && Base.Broadcast.combine_eltypes(bc.f, bc.args) === Bool) ?
        collect(bc) : bc
end

# `LinearAlgebra.dot(a, b)` with a MIXED host-array × traced operand does not lower
# under a tracing backend: it routes through `conj` on the host vector
# (`MethodError: no method matching conj(::Vector)`).  ONLY that mix is a problem —
# a pure-host dot is ordinary Base, and a pure-traced `dot(q, q)` has the backend's
# own (replica-aware, see `replica`) lowering that existing Reactant kernels rely
# on.  So `_tensorized_dot` DEFAULTS to the native `dot` for every case, and a
# tracing extension specializes ONLY the host-array × traced mix onto
# `_tensorized_normalized_dot` below.  Per user decision `17bnc6t` this normalizes
# in the `@kernel` lowering, not in Reactant.
@inline _tensorized_dot(a, b) = LinearAlgebra.dot(a, b)

# Whether a tensorized-dot operand carries a REAL scalar type.  The default reads
# the element type directly (host arrays/scalars); a tracing backend specializes
# it to see through its traced scalar wrapper — a `TracedRArray{Float64}`'s
# `eltype` is `TracedRNumber{Float64}`, not `Float64`, so a bare `eltype <: Real`
# would misclassify a real traced operand as complex.
@inline _tensorized_real_operand(x) = eltype(x) <: Real

# Normalize the mixed host/traced dot to the value-identical `sum(a .* b)`
# reduction over the promoted broadcast (the friendly form the Reactant benchmark
# authored by hand), so authors can write `dot(data, q)` in the kernel body.
# Value-exact for REAL operands, so it never silently mis-lowers.  COMPLEX
# operands are a LOUD error, never rewritten: `dot` conjugates its FIRST argument,
# so `sum(a .* b)` would silently corrupt a complex-valued result/gradient.
@inline function _tensorized_normalized_dot(a, b)
    (_tensorized_real_operand(a) && _tensorized_real_operand(b)) ||
        throw(ArgumentError(
            "dot(a, b) over complex operands is not lowerable to a tensorized " *
            "reduction: `dot` conjugates its first argument, so `sum(a .* b)` " *
            "would silently corrupt the result. Evaluate this dot on the native " *
            "path, or supply real operands."))
    sum(_tensorized_broadcast(*, a, b))
end

# A factorization call (`cholesky(A)`) in a tensorized body.  A tracing
# backend returns its own factorization type, whose surface can differ from
# `LinearAlgebra.Cholesky` (Reactant's has no `.L`/`.U`).  The tensorized
# companion keeps the authored call unchanged and passes its result through
# this hook; a tracing extension specializes it to wrap its backend type in a
# type the extension owns, so authored code downstream (`C.L`, `C \ b`) never
# needs methods on a foreign type.  Per user decision `17bnc6t` this
# normalizes in the `@kernel` lowering, not in the backend.
@inline _tensorized_factorization(factorization) = factorization

# The sequential-scan primitive `scan(xs..., Ref(shared)...; init) do carry, x…, s… end`
# lowers to this.  `step` is the prepared 2-`want` step kernel
# `(carry, x..., shared...) -> (new_carry, output)`; the scan threads `carry`
# (seeded by `init`) over the `iterated` sequences in lockstep, one element of
# each per step, and collects the per-step outputs.  The default is the ordinary
# native loop — already correct, and the arma11 `errors` recurrence proves the
# native form works.  A tracing backend SPECIALIZES this (on traced `iterated`
# sequences) to emit a `stablehlo.while` carry loop, so the natural sequential
# form lowers under Reactant without unrolling (RK-macro-only per decision
# `17bnc6t`; Reactant untouched).  `iterated` and `shared` are tuples; the common
# case is a one-tuple `iterated`, and `eachindex(iterated...)` validates that
# several sequences share axes (a `DimensionMismatch` otherwise).
@inline function _tensorized_scan(step, init, iterated::Tuple, shared::Tuple)
    marker = _scan_backend_marker(init, iterated, shared)
    _tensorized_scan_lowering(marker, step, init, iterated, shared)
end

# A scan runs on a backend exactly when ANY of its operands is that backend's
# traced value — the carry seed, an iterated sequence (looking through an
# `eachrow`/`eachcol` slices wrapper to its parent), or a shared operand
# (looking through a `Ref`).  Bound host data beside a traced operand is then
# a constant of the traced program, never a reason to fall back to the host
# loop: under the core constraints (`docs/src/constraints.md`) the host loop
# below is the NATIVE lowering, and tracing it would replicate the step body
# once per data element.  A scan whose every operand is host data runs the
# native loop as ordinary host precomputation, emitting no program structure.
@inline _scan_marker_value(x) = x
@inline _scan_marker_value(x::Base.RefValue) = x[]
@inline _scan_marker_value(x::Base.AbstractSlices) = parent(x)
@inline _scan_backend_marker(init, iterated::Tuple, shared::Tuple) =
    _dynamic_tensorized_marker((init, map(_scan_marker_value, iterated)...,
                                map(_scan_marker_value, shared)...))

# The native ordered loop.  `nothing` is the no-backend marker; a backend
# extension specializes `_tensorized_scan_lowering` on its own marker type.
function _tensorized_scan_lowering(::Nothing, step, init, iterated::Tuple,
                                   shared::Tuple)
    idx = eachindex(iterated...)
    isempty(idx) && throw(ArgumentError("scan requires a non-empty sequence"))
    i1 = first(idx)
    carry, out1 = step(init, map(xs -> xs[i1], iterated)..., shared...)
    result = similar(first(iterated), typeof(out1))
    result[i1] = out1
    for i in Iterators.drop(idx, 1)
        carry, out = step(carry, map(xs -> xs[i], iterated)..., shared...)
        result[i] = out
    end
    result
end

# Internal rectangular recurrence boundary. Unlike scan, this returns the final
# carry, which may include fixed-size output buffers. Ragged segments use
# reset/mask/index columns, never dynamic slices or growing containers.
@inline function _rectangular_fold(step, init, columns::Tuple, shared::Tuple, marker)
    isempty(columns) && throw(ArgumentError("a rectangular fold needs columns"))
    n = length(first(columns))
    all(c -> c isa AbstractVector && length(c) == n, columns) ||
        throw(DimensionMismatch("rectangular fold columns must be equal-length vectors"))
    Base.require_one_based_indexing(columns...)
    _rectangular_fold_impl(marker, step, init, columns, shared, n)
end

@inline function _rectangular_fold_impl(marker, step, init, columns, shared, n)
    carry = init
    for i in 1:n
        carry = step(carry, map(c -> c[i], columns), shared...)
    end
    carry
end

# Lazy scalar control: inactive singular/overflowing transitions must not run.
@inline _recurrence_branch(pred, yes, no, args) = pred ? yes(args...) : no(args...)

"""
    _KernelBranch{CI,TI,EI}(call, condition, then_arm, else_arm)

A recipe whose authored right-hand side is a top-level lazy branch
(`c ? a : b`, `if`/`elseif`/`else`, `&&`, `||`) keeps that structure as
metadata beside its ordinary body. Calling it runs `call`, the authored
branch over every recipe argument — exactly the closure an unstructured
recipe would carry — so every lowering that treats the enclosing
`_KernelSourceOp` as opaque is unchanged. The parts are closures over their
OWN free ports, selected from the recipe's ordered arguments by the position
tuples `CI`/`TI`/`EI`; a nested branch arm is itself a `_KernelBranch` over
every argument. Plate partial evaluation reads them: a condition whose ports
are all bound data is evaluated per lane at preparation, and the plate splits
into one plate per taken arm (`_partition_plate_recipe`).
"""
struct _KernelBranch{CI,TI,EI,F,C,T,E}
    call::F
    condition::C
    then_arm::T
    else_arm::E
end
_KernelBranch(::Val{CI}, ::Val{TI}, ::Val{EI}, call::F, condition::C,
              then_arm::T, else_arm::E) where {CI,TI,EI,F,C,T,E} =
    _KernelBranch{CI,TI,EI,F,C,T,E}(call, condition, then_arm, else_arm)
@inline (branch::_KernelBranch)(args...) = branch.call(args...)

# Tensorized authored plates keep slice collections structural instead of
# materializing Base.Slices.  A backend can consume the parent array as one
# batched value, while the generic fallback preserves ordinary eachcol
# broadcast semantics.
struct _TensorizedEachcol{A}
    parent::A
end

struct _TensorizedPlateBatch{A}
    values::A
end

@inline _tensorized_eachcol(parent) = _TensorizedEachcol(parent)
# A backend may represent an in-flight plate value with its own marker type
# (for example per-lane scalars).  It declares that marker here so the recipe
# chain inside one plate body keeps routing through the backend, and it
# specializes the materialize/sum hooks below for that representation.
@inline _tensorized_plate_is_marker(arg) = false
@inline _tensorized_plate_is_marker(
    ::Union{_TensorizedEachcol,_TensorizedPlateBatch}) = true
@inline _tensorized_plate_marker(::Tuple{}) = nothing
@inline function _tensorized_plate_marker(args::Tuple)
    first_arg = first(args)
    _tensorized_plate_is_marker(first_arg) ?
        first_arg : _tensorized_plate_marker(Base.tail(args))
end
@inline _tensorized_plate_fallback_arg(arg) = arg
@inline _tensorized_plate_fallback_arg(arg::_TensorizedEachcol) =
    eachcol(arg.parent)
@inline _tensorized_plate_fallback_arg(arg::_TensorizedPlateBatch) = arg.values
@inline _tensorized_plate_materialize(value) = value
@inline _tensorized_plate_materialize(value::_TensorizedPlateBatch) = value.values
# The authored `sum(pointwise)` consumer of a plate.  Native semantics are
# exactly `sum` over the materialized pointwise vector; a backend that keeps
# the plate as per-lane values may reduce those lanes directly instead of
# first materializing a vector it would immediately reduce.
@inline _tensorized_plate_sum(value) = sum(_tensorized_plate_materialize(value))

# A recipe whose operands carry no plate marker — every operand a host array
# (`bound=` data) or a shared (possibly traced) scalar — broadcasts on the host.
# Route it through `_tensorized_broadcast`, which handles the two ways Base's own
# broadcast breaks a tracing backend here:
#   * an all-host `Bool` recipe (a typed validity local such as the binomial
#     family's `valid::Bool = (observed >= 0) & (observed <= n)`, which becomes
#     its own plate recipe over two bound count vectors) would materialize a
#     `BitArray` — Reactant 0.2.284 recurses without termination on
#     `copyto!(::BitArray, ::Broadcasted)`, a `StackOverflowError` with no
#     actionable signal — so `_tensorized_materialize` collects a dense
#     `Array{Bool}` instead; and
#   * a MIXED host-array/traced-scalar recipe (a scalar parameter with all plate
#     data `bound=`, so only the shared scalar is traced) would infer an abstract
#     `Number` eltype the backend's `similar` cannot allocate — so
#     `_tensorized_cat_operands` promotes the host array operands against the
#     discovered traced marker, giving a concrete traced eltype.
# It reduces to plain `broadcast` for a non-`Bool` all-host recipe, so values and
# shapes are unchanged.
@inline function _tensorized_plate_call(operation, args...)
    marker = _tensorized_plate_marker(args)
    marker === nothing ?
        _tensorized_broadcast(operation, args...) :
        _tensorized_plate_call(marker, operation, args)
end

@inline function _tensorized_plate_call(
        marker::Union{_TensorizedEachcol,_TensorizedPlateBatch},
        operation, args::Tuple)
    unwrapped = map(_tensorized_plate_fallback_arg, args)
    _TensorizedPlateBatch(Base.broadcast(operation, unwrapped...))
end

# Index syntax in a tensorized fused body routes through this hook.  Native
# semantics are exactly Base.getindex; tracing extensions may explicitly
# authorize their backend's scalar gather lowering.
@inline _tensorized_getindex(args...) = getindex(args...)

"""
    Graph()

A mutable builder collecting `Value`s and `Recipe`s plus the producer index the
planner needs. Building executes nothing.
"""
mutable struct Graph
    values::Dict{Int,Value}          # id => Value
    recipes::Vector{Recipe}
    producers::Dict{Int,Vector{Int}} # canonical value id => indices into `recipes`
    aliases::Dict{Int,Int}           # value id => structurally-equal canonical id
    version::Int
end
Graph() = Graph(Dict{Int,Value}(), Recipe[], Dict{Int,Vector{Int}}(),
                Dict{Int,Int}(), 0)

_register!(g::Graph, v::Value) = (g.values[v.id] = v; v)

"""
    canon_id(g, id) -> Int

Resolve a value id to its structural-CSE canonical representative (gist §8).
Absent any structural CSE this is the identity.
"""
canon_id(g::Graph, id::Int) = haskey(g.aliases, id) ? canon_id(g, g.aliases[id]) : id

"""
    value!(g, name, T) -> Value{T}

Create a `Value{T}` and register it into graph `g`.
"""
function value!(g::Graph, name::Symbol, ::Type{T}) where {T}
    v = Value(name, T)
    _register!(g, v)
    g.version += 1
    v
end

_astuple(v::Value) = (v,)
_astuple(t::Tuple) = t
_astuple(v::AbstractVector) = Tuple(v)

function _cse_alias_plan(g::Graph, new_outputs::Tuple, old_outputs::Tuple, cse_key)
    targets = Dict{Int,Int}()
    edges = Pair{Int,Int}[]

    for (position, (new_output, old_output)) in enumerate(zip(new_outputs, old_outputs))
        new_type = valtype(new_output)
        old_type = valtype(old_output)
        new_type === old_type || throw(ArgumentError(
            "structural CSE output type mismatch for key $(repr(cse_key)) at position " *
            "$position: existing output $(old_output.name) has type $old_type, " *
            "new output $(new_output.name) has type $new_type"))

        source = canon_id(g, new_output.id)
        target = canon_id(g, old_output.id)
        source == target && continue
        if haskey(targets, source)
            targets[source] == target || throw(ArgumentError(
                "conflicting structural CSE output mapping for key $(repr(cse_key)) " *
                "at position $position: canonical value $source would map to both " *
                "$(targets[source]) and $target"))
            continue
        end
        targets[source] = target
        push!(edges, source => target)
    end

    # Validate the complete mapping before mutating the graph. In particular,
    # crossed multi-output mappings such as (a, b) => (b, a) must not create a
    # recursive alias chain.
    for start in sort!(collect(keys(targets)))
        seen = Set{Int}()
        current = start
        while haskey(targets, current)
            current in seen && throw(ArgumentError(
                "cyclic structural CSE output mapping for key $(repr(cse_key))"))
            push!(seen, current)
            current = targets[current]
        end
    end
    edges
end

function _reindex_producers!(g::Graph)
    empty!(g.producers)
    for recipe in g.recipes
        indexed = Set{Int}()
        for output in recipe.outputs
            canonical = canon_id(g, output.id)
            canonical in indexed && continue
            push!(get!(g.producers, canonical, Int[]), recipe.id)
            push!(indexed, canonical)
        end
    end
    g
end

"""
    add!(g, inputs => outputs, op; cost=1.0, cse_key=nothing, effectful=false)
    add!(g; inputs, outputs, op, cost=1.0, cse_key=nothing, effectful=false)

Register a recipe `(inputs...) --op--> (outputs...)`. `inputs`/`outputs` may be
a single `Value` or a tuple of `Value`s. Referenced values are auto-registered.
Returns the `Recipe`.
"""
function add!(g::Graph; inputs, outputs, op,
              cost::Real = 1.0, cse_key = nothing, effectful::Bool = false,
              source = _NO_KERNEL_SOURCE)
    ins = _astuple(inputs)
    outs = _astuple(outputs)
    recipe_cost = Float64(cost)
    if !isfinite(recipe_cost) || recipe_cost < 0
        throw(ArgumentError("recipe cost must be finite and non-negative, got $cost"))
    end
    # Opt-in structural CSE (gist §8): if a prior recipe carries the same
    # non-`nothing` cse_key, the same canonical inputs, and the same output
    # arity, it computes the same thing. Alias the new outputs onto the existing
    # producer's outputs instead of adding a duplicate recipe.
    if cse_key !== nothing && !effectful
        canon_ins = Tuple(canon_id(g, v.id) for v in ins)
        for r in g.recipes
            r.effectful && continue
            r.cse_key === nothing && continue
            isequal(r.cse_key, cse_key) || continue
            length(r.outputs) == length(outs) || continue
            Tuple(canon_id(g, v.id) for v in r.inputs) == canon_ins || continue
            alias_plan = _cse_alias_plan(g, outs, r.outputs, cse_key)
            for v in ins; _register!(g, v); end
            for v in outs; _register!(g, v); end
            for (source, target) in alias_plan
                g.aliases[source] = target
            end
            isempty(alias_plan) || _reindex_producers!(g)
            g.version += 1
            return r
        end
    end

    for v in ins; _register!(g, v); end
    for v in outs; _register!(g, v); end
    r = Recipe(length(g.recipes) + 1, ins, outs, op, recipe_cost, cse_key,
               effectful, source)
    push!(g.recipes, r)
    for v in outs
        push!(get!(g.producers, canon_id(g, v.id), Int[]), r.id)
    end
    g.version += 1
    r
end

add!(g::Graph, pair::Pair, op; kwargs...) =
    add!(g; inputs = pair.first, outputs = pair.second, op = op, kwargs...)

"All recipes that can produce value id `vid` (resolved through structural CSE)."
producers_of(g::Graph, vid::Int) = get(g.producers, canon_id(g, vid), Int[])
