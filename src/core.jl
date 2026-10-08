# Core graph identities: Value, Recipe, Graph.
#
# These objects are *compile/planning-time metadata only*. None of them are
# consulted inside a prepared kernel (see codegen.jl); the hot path sees only
# ordinary Julia values.

# A process-global counter giving every `Value` a stable identity independent of
# its name or of any particular graph. Identity must not depend solely on the
# name (gist §5), so two values may share a name yet remain distinct.
const _VALUE_COUNTER = Ref(0)
_next_value_id(floor::Int = 0) =
    (_VALUE_COUNTER[] = max(_VALUE_COUNTER[], floor) + 1)

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

struct _NoKernelSource end
const _NO_KERNEL_SOURCE = _NoKernelSource()

# A source-visible domain check. Backends retain this check at execution
# time; its predicate is nondifferentiable, and the exception is static
# diagnostic data rather than an active mathematical operand.
@inline function _runtime_check(valid, error::Exception)
    valid || throw(error)
    nothing
end

struct _RuntimeCheckCallback{E}
    error::E
end
(check::_RuntimeCheckCallback)(valid) = _runtime_check(valid, check.error)

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
enters prepared hot-state tuples. [`recipe_kind`](@ref) classifies a recipe as an
authored plate, an authored scan or an ordinary recipe.
"""
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
    _TypeOperation{T}

A type used as a recipe's operation, such as the bare constructor call
`y = SVector{2,Float64}(a, b)` in a `@kernel` or `add!(g, x => y, Float64)`.
Calling it calls `T` with the same arguments. Every type has the one Julia
type `DataType` (or `UnionAll`), so a type stored in an operation table leaves
its identity to the runtime value: a call through it, and every result type
derived from it, is uninferred, and the generated body falls back to dynamic
dispatch. Native Enzyme reverse then fails static activity analysis on the
boxed arguments. As a singleton the type travels in the operation's own type.
Recipe registration (`_add_recipe!`) applies it, so every operation table and
type query sees it.
"""
struct _TypeOperation{T} end
@inline (::_TypeOperation{T})(args...) where {T} = T(args...)
_type_operation_type(::_TypeOperation{T}) where {T} = T

"""
    _KernelSourceOp{DefToken,Form,F,TF,IG,CI}

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
`IG` holds the captured source's throw-stripped twin, selected only by an
explicit `on_error = :ignore` preparation; older internal fixtures use `nothing`.
`CI` retains the callable and lowering mode of an exact positional call through
a constant binding, when present. Composition can share these calls without
equating separately authored fused expressions or changing their definition tokens.
The call forwards INLINE. A RAW anonymous closure inserted into a Graph carries no wrapper and is
rejected as opaque when captured into a prepared handle.
"""
struct _KernelSourceOp{DefToken,Form,F,TF,IG,CI}
    f::F
    tensor_f::TF
    ignored_throws::IG
    call_identity::CI
end

# Every authored recipe creates source operations of new concrete types, and
# a constructor call compiles once per concrete type. Graph assembly only stores
# the operation, so build it from its parameters without specializing. The
# prepared kernel still sees the exact concrete operation type.
Base.@nospecializeinfer function _KernelSourceOp(
        @nospecialize(token::Val), @nospecialize(form::Val),
        @nospecialize(f), @nospecialize(tensor_f),
        @nospecialize(ignored_throws = nothing), @nospecialize(call_identity = nothing))
    T = _KernelSourceOp{_val_parameter(token), _val_parameter(form),
                        typeof(f), typeof(tensor_f), typeof(ignored_throws),
                        typeof(call_identity)}
    _kernel_new_instance(T, (f, tensor_f, ignored_throws, call_identity))::_KernelSourceOp
end

Base.@nospecializeinfer _val_parameter(@nospecialize(value::Val)) =
    (typeof(value)::DataType).parameters[1]

# Allocate an instance of a concrete struct type from field values of exactly
# the field types, as its default inner constructor would, without compiling
# that constructor for `T`. `jl_new_structv` is the runtime's own struct
# allocator (also used by the Serialization stdlib); it type-checks each field.
Base.@nospecializeinfer function _kernel_new_instance(@nospecialize(T::DataType),
                                                     @nospecialize(fields::Tuple))
    values = Any[fields...]
    ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, values, length(values))
end

# A module binding can be shadowed by a lexical callable captured in either
# execution body. Its value at construction need not identify later calls,
# especially when Julia retains a reassigned local in a shared Core.Box.
# The runtime callable must also match the binding resolved by lowering,
# which can have replaced the call and removed its lexical capture entirely.
Base.@nospecializeinfer function _KernelSourceOp(
        @nospecialize(token::Val), @nospecialize(form::Val), @nospecialize(f),
        @nospecialize(tensor_f), @nospecialize(ignored_throws), @nospecialize(call_identity),
        @nospecialize(capture_name), @nospecialize(resolved_callable))
    if call_identity !== nothing &&
       (first(call_identity) !== resolved_callable ||
        (capture_name !== nothing &&
         (capture_name in fieldnames(typeof(_kernel_native_source(f))) ||
          capture_name in fieldnames(typeof(_kernel_native_source(tensor_f))))))
        call_identity = nothing
    end
    _KernelSourceOp(token, form, f, tensor_f, ignored_throws, call_identity)
end

# Preserve the established internal constructor for compiler fixtures and
# already-authored handles; without an alternate body it uses the same callable
# in both modes.
_KernelSourceOp(token::Val, form::Val, f) = _KernelSourceOp(token, form, f, f)

# A runtime authoring expression can create a source method newer than its
# caller's world. Keep ordinary, inlineable closure execution whenever that
# method is visible. A generated body carries explicit lexical captures for
# immediate execution of newly composed expressions, including ordinary AD.
# Bodies with Julia's local function/macro/exception scopes keep native lowering.
struct _KernelSourceFunction{F,RF,C} <: Function
    f::F
    runtime_f::RF
    captures::C
end

_kernel_native_source(f) = f
_kernel_native_source(f::_KernelSourceFunction) = f.f

@inline @generated function (f::_KernelSourceFunction)(args::Vararg{Any,N}) where {N}
    forwarded = [:(getfield(args, $index)) for index in 1:N]
    argtypes = Tuple{args...}
    quote
        # The type-only query uses the caller's world without boxing active
        # argument values into a method-lookup array during ordinary AD.
        if hasmethod(f.f, $argtypes)
            Base.@inline f.f($(forwarded...))
        else
            Base.@inline _kernel_runtime_source_call(
                f.runtime_f, f.f, f.captures, $(forwarded...))
        end
    end
end

@inline @generated function _kernel_runtime_source_call(
        runtime_f, f, captures, args::Vararg{Any,N}) where {N}
    forwarded = [:(getfield(args, $index)) for index in 1:N]
    :(Base.@inline RuntimeGeneratedFunctions.generated_callfunc(
        runtime_f, captures, $(forwarded...)))
end

@inline function _kernel_runtime_source_call(::Nothing, f, captures,
                                            args::Vararg{Any,N}) where {N}
    result_type = Core.Compiler.return_type(f, typeof(args))
    Base.invokelatest(f, args...)::(result_type === Union{} ? Any : result_type)
end

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
#
# The native call is CALLSITE-INLINED (`Base.@inline op.f(...)`, spelled out as
# the expression that macro expands to, so the generated body carries no
# macrocall). Julia's inlining heuristic refuses a closure whose body carries a
# loop — a plate cell such as `sum(f(i) for i in slots)` — so without the
# annotation the cell stayed a real function call on every plate coordinate,
# and the loop's invariants (`plan.shifts`, `eachindex`) were recomputed per
# cell. Measured on a 16321-observation, 3-dose superposition plate (Julia
# 1.10.11, x86-64): 70 µs per read as a plain call, 37 µs inlined — the same
# time as the cell body written inline in a hand loop — with allocation and
# values unchanged (snag `generator-plate-e160f4c2`). For a closure the
# heuristic already inlines the annotation changes nothing; the tensorized
# twin traces and needs none. Julia 1.12 inlines this closure on its own.
@inline @generated function _kernel_source_call(::Val{:native},
        op::_KernelSourceOp, args::Vararg{Any,N}) where {N}
    forwarded = [:(getfield(args, $index)) for index in 1:N]
    value = gensym(:value)
    Expr(:block,
         Expr(:inline, true),
         Expr(:local, Expr(:(=), value, :(op.f($(forwarded...))))),
         Expr(:inline, false),
         value)
end
@inline @generated function _kernel_source_call(::Val{:tensorized},
        op::_KernelSourceOp, args::Vararg{Any,N}) where {N}
    forwarded = [:(getfield(args, $index)) for index in 1:N]
    :(op.tensor_f($(forwarded...)))
end
# The entry call forwards positionally too, and folds the style the same way
# `_kernel_source_style` does but spelled out per argument (that function
# recurses through `Base.tail`, itself a splat). Julia's inliner rewrites an
# `args...` splat into a direct call only up to 32 elements
# (`max_tuple_splat`, 1.10 and 1.12 alike); past that the call stays a dynamic
# `Core._apply_iterate`, codegen materializes the arguments in one tuple, and
# Enzyme meets a constant array stored next to active values: a 32-argument
# module call with a bound data vector failed reverse mode with
# `EnzymeRuntimeActivityError` (snag `rkppl-module-cal-79cad594`).
@inline @generated function (op::_KernelSourceOp)(args::Vararg{Any,N}) where {N}
    style = :(Val(:native))
    for index in N:-1:1
        style = :(_kernel_source_merge(
            _kernel_source_arg_style(getfield(args, $index)), $style))
    end
    forwarded = [:(getfield(args, $index)) for index in 1:N]
    :(_kernel_source_call($style, op, $(forwarded...)))
end
kernel_sourceop_token(::_KernelSourceOp{DefToken}) where {DefToken} = DefToken
kernel_sourceop_form(::_KernelSourceOp{DefToken,Form}) where {DefToken,Form} = Form
# The element type of an empty plate is Julia's empty-broadcast type: inferred
# from the authored native cell over the operands' native element types, with
# no cell evaluated. `nothing` for any other callable, or when inference gives
# no concrete type.
_kernel_native_result_type(f, ::Type{<:Tuple}) = nothing
function _kernel_native_result_type(op::_KernelSourceOp, argtypes::Type{<:Tuple})
    T = Core.Compiler.return_type(op.f, argtypes)
    isconcretetype(T) ? T : nothing
end
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
# Typed conversion at the compiler boundary keeps Base.trunc semantics in
# native execution. Tracing extensions can preserve already-integer values.
@inline _tensorized_trunc(::Type{T}, x) where {T<:Integer} = trunc(T, x)
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
@inline function _tensorized_cat_operands(args::Tuple,
        operand = _tensorized_cat_operand)
    marker = _tensorized_cat_marker(args)
    marker === nothing ? args :
        map(arg -> operand(marker, arg), args)
end
# Concatenation treats a scalar as one entry; broadcast must keep it a scalar.
# Give concatenation its own operand hook, with the structural argument rank,
# so a tracing extension can lift scalars and pad missing unit dimensions.
@inline _tensorized_concat_operand(marker, arg, rank) =
    _tensorized_cat_operand(marker, arg)
@inline _tensorized_concat_operands(args::Tuple) =
    _tensorized_cat_operands(args,
        (marker, arg) -> _tensorized_concat_operand(
            marker, arg, Val(maximum(ndims, args))))
@inline _tensorized_vcat(args...) = vcat(_tensorized_concat_operands(args)...)
@inline _tensorized_hcat(args...) = hcat(_tensorized_concat_operands(args)...)
@inline _tensorized_cat(args...; dims) =
    cat(_tensorized_concat_operands(args)...; dims = dims)
# A scalar-vector literal (`[a, b, c]`) in a tensorized body.  The default
# is the plain `Base.vect` construction; a tracing extension builds a real
# traced vector when any element is traced (a host container of traced
# scalars fails downstream — gathering it recurses without termination).
@inline _tensorized_vect(args...) = Base.vect(args...)
# The scalar type a literal element contributes to the promoted element
# type.  The default reads the host type directly; a tracing backend
# specializes it to see through its traced scalar wrapper — a
# `TracedRNumber{Float64}`'s `typeof` is the wrapper, not `Float64`, so a
# bare promotion would abstract the element type.
@inline _tensorized_vect_eltype(x::Number) = typeof(x)
# Build the traced vector from its element type: the host `vect` container
# always builds (wrapper promotion keeps every element assignable), and the
# broadcast conversion lifts it into the traced program — the same
# construction a hand-written `Float64.([a, b, c])` performs, generalized
# over `T`.
@inline _tensorized_vect_construct(::Type{T}, args::Tuple) where {T} =
    T.(Base.vect(args...))
@inline _tensorized_broadcast(f, args...) =
    _tensorized_materialize(
        Base.broadcasted(f, _tensorized_cat_operands(args)...))

@inline _native_broadcast_materialize(value) = Base.materialize(value)
# Base's scalar/zero-dimensional path skips axis instantiation, including
# zero-argument dotted calls for which combine_axes has no method.
@inline _native_broadcast_materialize(
    bc::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}) =
    Base.materialize(bc)
@inline function _native_broadcast_materialize(
        bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle})
    # A deep fused broadcast can exhaust Julia's inlining budget at the
    # unannotated singleton combine_axes method. Passing its mixed active and
    # constant array descriptor across that call defeats native Reverse's
    # static activity analysis. Keep axis discovery at the materialization
    # site; the descriptor stays lazy and Base still owns the element loop.
    # Respect axes supplied by a specialized broadcasted implementation.
    bc.axes === nothing || return Base.materialize(bc)
    ax = Base.@inline Base.Broadcast.combine_axes(bc.args...)
    ready = Base.Broadcast.Broadcasted(bc.style, bc.f, bc.args, ax)
    _native_broadcast_copy(Base.Broadcast.instantiate(ready))
end

@inline function _native_broadcast_copy(bc)
    T = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    isconcretetype(T) || return Base.materialize(bc)
    _native_broadcast_copyto!(similar(bc, T), bc)
end
@inline _native_broadcast_copyto!(dest, bc) = copyto!(dest, bc)
# `similar` has just allocated this dense output, so no input can alias it.
# Extrude with no destination before Base's copy loop: checking aliasing against
# the output would needlessly mix a constant input with a possible fresh copy.
# Other destination types retain their specialized copy/alias protocol.
@inline _native_broadcast_copyto!(dest::Array, bc) =
    copyto!(dest, Base.Broadcast.preprocess(nothing, bc))

# Native concatenation.  Base's methods for dense arrays of one element type,
# `hcat`/`vcat` of `Vector{T}`s and the `typed_hcat`/`typed_vcat`/
# `typed_hvcat` loops behind `hcat`, `vcat` and `hvcat` of `Vector{T}` and
# `Matrix{T}` operands, read each operand from their vararg tuple at a runtime
# index.  When the operands mix constant data with active arrays, native
# Enzyme reverse joins their activities at that load and fails static activity
# analysis with `EnzymeRuntimeActivityError`, although the arguments' activity
# is fixed (`benchmark/repro_enzyme_mixed_activity_concat.jl` reproduces it
# without ReactiveKernels).  Scalar and mixed scalar/array operands, as in
# the literals `[-a 0.0; a -b]` and `[M v; 0.0 1.0]`, reach whichever generic
# method claims them: SparseArrays, which Enzyme loads, extends `hcat`, `vcat`
# and `hvcat` to every list of `Number` and `AbstractVecOrMat{<:Number}`
# operands and sends dense ones through generic `typed_hcat`/`typed_hvcat`
# paths, several times slower than Base's scalar fill.  The native body calls
# these companions instead (`_kernel_native_calls`).  For `Number`, `Vector`
# and `Matrix` operands whose promoted element type is isbits, they return
# Base's result, a freshly allocated `Matrix` or `Vector` of Base's shape,
# element type and values, and write each operand in its own inlined call, so
# every write stays tied to its tuple position.  A `Number` is a 1×1 block,
# and `vcat` of numbers and vectors alone is a vector, as in Base.  Any other
# operand, a non-isbits element type, and every `hcat`/`vcat` layout the
# companions do not lay out call Base itself, which keeps its result and error
# (including `vcat`'s fill of leading numbers across a wider matrix); for these
# operands Base returns a `Matrix` of the promoted element type or throws, so
# that call is asserted to keep the result inferred.  A layout `hvcat`
# rejects (unequal heights or widths, or block-row counts that do not
# describe the operands) throws the companion's own `DimensionMismatch` or
# `ArgumentError`, worded as Base's, and never calls Base: native Enzyme
# reverse on Julia 1.12 cannot compile Base's `hvcat` of mixed scalar and
# array operands even on a path that never runs
# (`benchmark/repro_enzyme_mixed_scalar_hvcat.jl`), and Julia 1.10's `hvcat`
# silently drops operands beyond its block-row counts.
const _NativeCatOperand = Union{Number,Vector,Matrix}
const _NativeColumnOperand = Union{Number,Vector}

@inline _native_cat_width(a::_NativeColumnOperand) = 1
@inline _native_cat_width(a::Matrix) = size(a, 2)

# Sizes over an operand tuple, one inlined call per operand.
@inline _native_cat_width_sum(::Tuple{}) = 0
@inline _native_cat_width_sum(args::Tuple) =
    _native_cat_width(first(args)) + _native_cat_width_sum(Base.tail(args))
@inline _native_cat_height_sum(::Tuple{}) = 0
@inline _native_cat_height_sum(args::Tuple) =
    size(first(args), 1) + _native_cat_height_sum(Base.tail(args))
@inline _native_cat_heights_equal(height, ::Tuple{}) = true
@inline _native_cat_heights_equal(height, args::Tuple) =
    size(first(args), 1) == height &&
    _native_cat_heights_equal(height, Base.tail(args))
@inline _native_cat_widths_equal(width, ::Tuple{}) = true
@inline _native_cat_widths_equal(width, args::Tuple) =
    _native_cat_width(first(args)) == width &&
    _native_cat_widths_equal(width, Base.tail(args))
@inline _native_cat_heights(::Tuple{}) = ()
@inline _native_cat_heights(args::Tuple) =
    (size(first(args), 1), _native_cat_heights(Base.tail(args))...)
@inline _native_cat_widths(::Tuple{}) = ()
@inline _native_cat_widths(args::Tuple) =
    (_native_cat_width(first(args)), _native_cat_widths(Base.tail(args))...)

# Write the operand `a` into `out` (column-major, `stride` rows) with its
# first element at row `row + 1`, column `column + 1`, converting to the
# output's element type as Base's `setindex!` does.  Callers allocate `out`
# from the operands' validated shapes, so every write is in bounds.
@inline function _native_cat_block!(out, stride, row, column, a::Number)
    @inbounds out[column * stride + row + 1] = a
    out
end
@inline function _native_cat_block!(out, stride, row, column, a::Array)
    height = size(a, 1)
    for j in 1:_native_cat_width(a), i in 1:height
        @inbounds out[(column + j - 1) * stride + row + i] =
            a[(j - 1) * height + i]
    end
    out
end
@inline function _native_cat_block!(out::Array{T}, stride, row, column,
                                    a::Array{T}) where {T}
    height = size(a, 1)
    if height == stride
        # Whole columns of `out` (then `row == 0`): one contiguous block.
        Base.unsafe_copyto!(out, column * stride + 1, a, 1, length(a))
    else
        for j in 1:_native_cat_width(a)
            Base.unsafe_copyto!(out, (column + j - 1) * stride + row + 1,
                                a, (j - 1) * height + 1, height)
        end
    end
    out
end

@inline _native_hcat(args...) = hcat(args...)
@inline function _native_hcat(a::_NativeCatOperand, rest::_NativeCatOperand...)
    args = (a, rest...)
    T = Base.promote_eltypeof(args...)
    isbitstype(T) || return hcat(args...)
    height = size(a, 1)
    _native_cat_heights_equal(height, rest) || return hcat(args...)::Matrix{T}
    out = Matrix{T}(undef, height, _native_cat_width_sum(args))
    _native_hcat_copy!(out, height, 0, args)
end
@inline _native_hcat_copy!(out, height, column, ::Tuple{}) = out
@inline function _native_hcat_copy!(out, height, column, args::Tuple)
    a = first(args)
    _native_cat_block!(out, height, 0, column, a)
    _native_hcat_copy!(out, height, column + _native_cat_width(a),
                       Base.tail(args))
end

@inline _native_vcat(args...) = vcat(args...)
@inline function _native_vcat(a::_NativeColumnOperand,
                              rest::_NativeColumnOperand...)
    args = (a, rest...)
    T = Base.promote_eltypeof(args...)
    isbitstype(T) || return vcat(args...)
    out = Vector{T}(undef, _native_cat_height_sum(args))
    _native_vcat_copy!(out, length(out), 0, args)
end
@inline function _native_vcat(a::_NativeCatOperand, rest::_NativeCatOperand...)
    args = (a, rest...)
    T = Base.promote_eltypeof(args...)
    isbitstype(T) || return vcat(args...)
    width = _native_cat_width(a)
    _native_cat_widths_equal(width, rest) || return vcat(args...)::Matrix{T}
    height = _native_cat_height_sum(args)
    out = Matrix{T}(undef, height, width)
    _native_vcat_copy!(out, height, 0, args)
end
@inline _native_vcat_copy!(out, height, row, ::Tuple{}) = out
@inline function _native_vcat_copy!(out, height, row, args::Tuple)
    a = first(args)
    _native_cat_block!(out, height, row, 0, a)
    _native_vcat_copy!(out, height, row + size(a, 1), Base.tail(args))
end

# The errors of layouts `hvcat` rejects, built out of line from sizes only.
@noinline _native_hvcat_count_mismatch(rows, n) = ArgumentError(
    "block-row counts $rows do not describe $n blocks")
@noinline _native_hvcat_height_mismatch(i, expected, got) = DimensionMismatch(
    "mismatched height in block row $i (expected $expected, got $got)")
@noinline _native_hvcat_width_mismatch(i, expected, got) = DimensionMismatch(
    "block row $i has mismatched number of columns (expected $expected, got $got)")

# `hvcat(rows, blocks...)`: block row `i` holds the next `rows[i]` blocks,
# with equal heights within a block row and equal total widths across block
# rows.  `(height, width)` of the result; any other layout throws.
function _native_hvcat_shape(rows::Tuple{Vararg{Int}}, heights::Tuple,
                             widths::Tuple)
    n = length(heights)
    (!isempty(rows) && all(>=(1), rows) && sum(rows) == n) ||
        throw(_native_hvcat_count_mismatch(rows, n))
    k = 0
    height = 0
    width = -1
    for (i, count) in enumerate(rows)
        h = heights[k + 1]
        w = 0
        for _ in 1:count
            k += 1
            heights[k] == h ||
                throw(_native_hvcat_height_mismatch(i, h, heights[k]))
            w += widths[k]
        end
        width < 0 && (width = w)
        w == width || throw(_native_hvcat_width_mismatch(i, width, w))
        height += h
    end
    (height, width)
end

@inline _native_hvcat(rows, args...) = hvcat(rows, args...)
@inline function _native_hvcat(rows::Tuple{Vararg{Int}}, a::_NativeCatOperand,
                               rest::_NativeCatOperand...)
    args = (a, rest...)
    T = Base.promote_eltypeof(args...)
    isbitstype(T) || return hvcat(rows, args...)
    height, width = _native_hvcat_shape(rows, _native_cat_heights(args),
                                        _native_cat_widths(args))
    out = Matrix{T}(undef, height, width)
    _native_hvcat_copy!(out, height, rows, 1, 1, 0, 0, args)
end
@inline _native_hvcat_copy!(out, height, rows, i, j, row, column, ::Tuple{}) =
    out
@inline function _native_hvcat_copy!(out, height, rows, i, j, row, column,
                                     args::Tuple)
    a = first(args)
    _native_cat_block!(out, height, row, column, a)
    # One recursive call per operand: the next block starts a new block row
    # after the last block of row `i`.
    last = j == rows[i]
    _native_hvcat_copy!(out, height, rows,
                        last ? i + 1 : i, last ? 1 : j + 1,
                        last ? row + size(a, 1) : row,
                        last ? 0 : column + _native_cat_width(a),
                        Base.tail(args))
end

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
# several sequences share axes (a `DimensionMismatch` otherwise). Empty
# sequences run no step and yield an empty result. `Val(true)` is the authored
# `include_init = true`: the result is `[init, output…]` in one buffer of
# element type `promote_type(typeof(init), output type)`, exactly the value of
# `vcat([init], outputs)`, and an empty sequence yields `[init]`.
@inline function _tensorized_scan(step, init, iterated::Tuple, shared::Tuple,
                                  include_init::Val = Val(false))
    marker = _scan_backend_marker(init, iterated, shared)
    _tensorized_scan_lowering(marker, step, init, iterated, shared, include_init)
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

# The per-step output type of a scan step applied to these argument types. An
# empty sequence runs no step, so its (empty) result is typed the way Base's
# `accumulate` types one: by inference, `Any` when inference cannot tell.
function _scan_step_output_type(step, argument_types...)
    R = Base.promote_op(step, argument_types...)
    R isa DataType && R <: Tuple && length(R.parameters) == 2 ?
        fieldtype(R, 2) : Any
end

# The indices after a scan's peeled first step, which every native scan loop
# iterates, here and in the generated lowering. For a unit range they are the
# unit range that starts one later; other index collections drop the first.
# Both give the same indices. Native Enzyme reverse can fail on the
# `Iterators.drop` loop of a scan nested in another scan's step, both in one
# body: with an inner sequence that is empty and the same at every outer step,
# it raises `OutOfMemoryError` from the second outer step on. The unit-range
# loop differentiates. `benchmark/repro_enzyme_guarded_inner_loop_cache.jl`
# reproduces both with Enzyme only.
@inline _scan_rest(indices::AbstractUnitRange{<:Integer}) =
    (first(indices) + oneunit(eltype(indices))):last(indices)
@inline _scan_rest(indices) = Iterators.drop(indices, 1)

# The native ordered loop.  `nothing` is the no-backend marker; a backend
# extension specializes `_tensorized_scan_lowering` on its own marker type.
# An empty sequence returns an empty result (or `[init]`) without running the
# step. As in the generated native lowering (`_lower_authored_scan_native!`),
# a concrete inferred output type is the first output's type, so the result is
# allocated once with it before the emptiness test; separate allocations in
# the two arms meet in one value that Enzyme's static activity analysis
# rejects when the empty arm's is never written actively. Otherwise each arm
# allocates, typed by the first output when there is one.
# Keep this ordinary loop visible at its caller. Julia 1.13 / Enzyme's
# readonly analysis otherwise rejects local allocation stores across the call
# boundary when the differentiated closure captures its constant sequence.
@inline function _tensorized_scan_lowering(::Nothing, step, init, iterated::Tuple,
                                   shared::Tuple, ::Val{false} = Val(false))
    idx = eachindex(iterated...)
    T = _scan_step_output_type(
        step, typeof(init), map(eltype, iterated)..., map(typeof, shared)...)
    result = isconcretetype(T) ? similar(first(iterated), T) : nothing
    isempty(idx) && return result === nothing ? similar(first(iterated), T) : result
    i1 = first(idx)
    carry, out1 = step(init, map(xs -> xs[i1], iterated)..., shared...)
    result === nothing && (result = similar(first(iterated), typeof(out1)))
    result[i1] = out1
    for i in _scan_rest(idx)
        carry, out = step(carry, map(xs -> xs[i], iterated)..., shared...)
        result[i] = out
    end
    result
end

@inline function _tensorized_scan_lowering(::Nothing, step, init, iterated::Tuple,
                                   shared::Tuple, ::Val{true})
    idx = eachindex(iterated...)
    allocate(T) = similar(first(iterated), promote_type(typeof(init), T), length(idx) + 1)
    T = _scan_step_output_type(
        step, typeof(init), map(eltype, iterated)..., map(typeof, shared)...)
    result = isconcretetype(T) ? allocate(T) : nothing
    if isempty(idx)
        result === nothing && (result = allocate(T))
        result[1] = init
        return result
    end
    i1 = first(idx)
    carry, out1 = step(init, map(xs -> xs[i1], iterated)..., shared...)
    result === nothing && (result = allocate(typeof(out1)))
    result[1] = init
    result[2] = out1
    position = 2
    for i in _scan_rest(idx)
        carry, out = step(carry, map(xs -> xs[i], iterated)..., shared...)
        position += 1
        result[position] = out
    end
    result
end

# `scan(...; history = h0)`: the step's last argument is a read-only view of
# the result vector being written, so step `j` reads the outputs of steps
# `1:j-1` and `h0` at `j` and after. Every backend hands the step that same
# full-length vector, so a step that reads beyond `j - 1` sees `h0` on each.
# `h0` is a scalar number and fixes the result's element type; each output is
# stored converted to it. The view has no `setindex!`, and a carry holding it
# is refused, because the native view aliases the vector the scan still writes.
struct _ScanHistory{T,A<:AbstractVector{T}} <: AbstractVector{T}
    outputs::A
end
Base.size(h::_ScanHistory) = size(h.outputs)
Base.axes(h::_ScanHistory) = axes(h.outputs)
Base.IndexStyle(::Type{<:_ScanHistory{T,A}}) where {T,A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(h::_ScanHistory, i::Int...) = h.outputs[i...]

function _scan_history_buffer(xs, fill::Number)
    buffer = similar(xs, typeof(fill))
    fill!(buffer, fill)
end

# The native position driver supplies either its first-position scratch or a
# column of the owned output stack. A plate or scan overwrites the matching
# dense buffer in full; `nothing` means allocate. Match element type and axes
# before exposing storage to the generated body.
_replicated_column_type(::Type{Array{T,N}}) where {T,N} =
    SubArray{T,N - 1,Array{T,N},
             Tuple{ntuple(_ -> Base.Slice{Base.OneTo{Int}}, N - 1)...,Int},true}
_replicated_column_admitted(::Type{Array{T,N}}, ::Type{V}) where {T,N,V} =
    N > 1 && _replicated_column_type(Array{T,N}) <: V
@inline _lane_reuse(buffer, ::Type, output_axes) = nothing
@inline @generated function _lane_reuse(buffer, ::Type{T}, output_axes::NTuple{N,Any}) where {T,N}
    # An undeclared WANT's cache is heterogeneous. Narrow it here, where the
    # allocation's element type and rank are known, before entering its loop.
    column = _replicated_column_type(Array{T,N + 1})
    quote
        if buffer isa $column
            # A destination column has the first position's shape. A later
            # shape mismatch cannot be stacked, so reject it before writing.
            # Returning the column unconditionally after that check keeps the
            # hot loop's storage concrete instead of a view/Array union.
            axes(buffer) == output_axes || throw(DimensionMismatch(
                "position outputs must have the same shape at every position"))
            buffer::$column
        else
            buffer isa Array{$T,$N} && axes(buffer) == output_axes ?
                buffer::Array{$T,$N} : nothing
        end
    end
end

# A position intermediate's lane slot (`_lower_replicated_with_ops`): a
# `Ref{Any}` that keeps the dense buffer its producer allocated at an earlier
# position, or an earlier call of a borrowed reader. The slot is reused only
# when element type and axes match; any other value allocates afresh and
# replaces it. Intermediates never leave the call, and every WANT is copied
# into the stacked result before the next position runs, so no live value is
# overwritten (as for the input lanes, `_replicated_lane`).
@inline function _lane_reuse(slot::Base.RefValue{Any}, ::Type{T},
                             output_axes::NTuple{N,Any}) where {T,N}
    buffer = slot[]
    buffer isa Array{T,N} && axes(buffer) == output_axes ? buffer : nothing
end
# Record a freshly allocated buffer in a slot; other recycled values (WANT
# scratch, destination columns, `nothing`) are owned by the position driver.
@inline _lane_keep!(recycled, value) = value
@inline _lane_keep!(slot::Base.RefValue{Any}, value) = (slot[] = value; value)

# Destination forms of a top-level dotted call and an array slice in a
# position residual (`_lane_source_expr`). Each returns exactly the value its
# source would (`_native_broadcast_materialize`, `getindex`): a matching
# recycled buffer is overwritten in full, and otherwise the ordinary result is
# allocated and kept. Every other shape takes the ordinary path.
@inline _lane_broadcast(recycled, value) = _native_broadcast_materialize(value)
@inline _lane_broadcast(recycled,
    bc::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}) =
    _native_broadcast_materialize(bc)
@inline function _lane_broadcast(recycled,
        bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle})
    bc.axes === nothing || return _native_broadcast_materialize(bc)
    ax = Base.@inline Base.Broadcast.combine_axes(bc.args...)
    ready = Base.Broadcast.instantiate(
        Base.Broadcast.Broadcasted(bc.style, bc.f, bc.args, ax))
    T = Base.Broadcast.combine_eltypes(ready.f, ready.args)
    isconcretetype(T) || return Base.materialize(ready)
    buffer = _lane_reuse(recycled, T, ax)
    buffer === nothing && return _lane_keep!(recycled,
        _native_broadcast_copyto!(similar(ready, T), ready))
    copyto!(buffer, ready)
end

const _LaneIndex = Union{Integer,AbstractRange{<:Integer},Colon,AbstractVector{<:Integer}}
@inline _lane_getindex(recycled, A, I...) = A[I...]
# A dense array or a view of one: its non-scalar `getindex` is a fresh
# `Array{T}` of the index shape, which a copy into a matching buffer equals.
@inline function _lane_getindex(recycled,
        A::Union{Array{T},SubArray{T,<:Any,<:Array}}, I::Vararg{_LaneIndex}) where {T}
    all(index -> index isa Integer, I) && return A[I...]
    checkbounds(A, I...)
    J = to_indices(A, I)
    buffer = _lane_reuse(recycled, T, Base.index_shape(J...))
    buffer === nothing && return _lane_keep!(recycled, A[I...])
    copyto!(buffer, view(A, J...))
end

function _scan_history_buffer(recycled, xs, fill::Number)
    buffer = _lane_reuse(recycled, typeof(fill), axes(xs))
    buffer === nothing ?
        _lane_keep!(recycled, _scan_history_buffer(xs, fill)) : fill!(buffer, fill)
end
_scan_history_buffer(recycled, xs, fill) = _scan_history_buffer(xs, fill)
_scan_history_buffer(xs, fill) = throw(ArgumentError(
    "a scan's `history =` value must be a number (it fills the outputs not yet " *
    "written and fixes their element type); got a $(typeof(fill))"))

_scan_holds_history(::Type{<:_ScanHistory}) = true
_scan_holds_history(T::DataType) = (T <: Tuple || T <: NamedTuple) &&
    any(_scan_holds_history, fieldtypes(T))
_scan_holds_history(::Type) = false
@generated function _scan_history_carry(carry)
    _scan_holds_history(carry) ? :(throw(ArgumentError(
        "a `history =` scan step returned its history in the carry; the history " *
        "is read-only and valid only during its step. Carry the values it needs " *
        "instead."))) : :carry
end

@inline function _tensorized_scan_history(step, init, fill, iterated::Tuple,
                                          shared::Tuple)
    marker = _scan_backend_marker(init, iterated, (shared..., fill))
    _tensorized_scan_history_lowering(marker, step, init, fill, iterated, shared)
end

function _tensorized_scan_history_lowering(::Nothing, step, init, fill,
                                           iterated::Tuple, shared::Tuple)
    idx = eachindex(iterated...)
    result = _scan_history_buffer(first(iterated), fill)
    earlier = _ScanHistory(result)
    isempty(idx) && return result
    i1 = first(idx)
    carry, out1 = step(init, map(xs -> xs[i1], iterated)..., shared..., earlier)
    carry = _scan_history_carry(carry)
    result[i1] = out1
    for i in _scan_rest(idx)
        carry, out = step(carry, map(xs -> xs[i], iterated)..., shared..., earlier)
        carry = _scan_history_carry(carry)
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

"""
    _KernelReduction{II,XI,KI,AI}(call, iterator, init, step, index, array,
                                   step_in, step_out)

A recipe whose authored right-hand side is a top-level unfiltered generator
sum with `init` whose term reads one shared vector through Base's total
gather, `sum(… get(A, K, D) … for j in iterator; init = x)`, keeps its parts
as metadata beside its ordinary body. Calling it runs `call`, the authored
expression over every recipe argument, so every lowering that treats the
enclosing `_KernelSourceOp` as opaque is unchanged.

The native plate lowering reads the parts (`_lower_authored_plate_native!`):
when the iterator and `A` are plate invariants it runs the sum dose-outer —
one pass over the cells per element `j`, each cell accumulating
`Base.add_sum(acc, term)` in the authored order — and, when the gather index
advances by exactly one per cell, splits each pass at the window where `K` is
in range, reading `A` there without the bounds test and using `D` outside it.
`iterator`/`init`/`array` are closures over their OWN ports, selected from
the recipe's ordered arguments by the position tuples `II`/`XI`/`AI`;
`index` takes `j` and the ports `KI`; the steps take the accumulator, `j`
(and `step_in` the gathered value) followed by every recipe argument.
"""
struct _KernelReduction{II,XI,KI,AI,F,IT,IN,ST,IX,AR,SI,SO}
    call::F
    iterator::IT
    init::IN
    step::ST
    index::IX
    array::AR
    step_in::SI
    step_out::SO
end
_KernelReduction(::Val{II}, ::Val{XI}, ::Val{KI}, ::Val{AI}, call::F,
                 iterator::IT, init::IN, step::ST, index::IX, array::AR,
                 step_in::SI, step_out::SO) where
        {II,XI,KI,AI,F,IT,IN,ST,IX,AR,SI,SO} =
    _KernelReduction{II,XI,KI,AI,F,IT,IN,ST,IX,AR,SI,SO}(
        call, iterator, init, step, index, array, step_in, step_out)
# The authored body carries a loop, which Julia's inlining heuristic refuses;
# inline it at this call site as `_kernel_source_call` inlines the wrapper.
@inline (reduction::_KernelReduction)(args::Vararg{Any,N}) where {N} =
    @inline reduction.call(args...)

# Inference of nested source operations. A recipe whose source inlines another
# kernel's endpoint (a plate cell `normal(mu, s).logpdf(y)`) embeds that
# kernel's source operations, so one call runs `(op::_KernelSourceOp)(...)`,
# `_kernel_source_call`, the source function and the authored closure, which
# calls the embedded operation through the same methods again. Julia's
# inference treats a caller-to-callee method edge that recurs on its stack as
# possible unbounded recursion: Julia 1.10 compares the inner call with the
# callee's declared `Vararg` signature, which every concrete multi-argument call
# exceeds, and widens it. The inner result, the cell, and a fused plate total
# then infer as `Any`, with a boxed value and an uninlined call per cell, and
# the enclosing kernel's code is cached that way. Only after the inner operation
# was compiled at top level, as a dynamic call from that first kernel does, did
# a later identical kernel infer concretely, so whichever kernel a process
# compiled first stayed slow (snag `first-prepared-s-e848620b`: 5 µs and 87
# allocations against 0.3 µs and 6 per sampler gradient, primal and Enzyme
# alike, on Julia 1.10 and, for other cells, 1.12).
#
# Nesting distinct source operations terminates: an operation embeds only
# operations that already exist, so each level holds a different callable
# type. `recursion_relation` tells inference that such an edge is well founded.
# The same callable again, or one whose type contains its caller's, keeps
# Julia's default limiting.
_source_callable_type(@nospecialize(T)) = false
_source_callable_type(::Type{<:Union{_KernelSourceOp,_KernelSourceFunction,
                                     _KernelBranch,_KernelReduction}}) = true

# The callee a source-call signature runs: its first source-callable parameter
# (the called object, or the operation `_kernel_source_call` forwards).
function _source_call_identity(@nospecialize(sig))
    tuple = Base.unwrap_unionall(sig)
    tuple isa DataType || return nothing
    for parameter in tuple.parameters
        parameter isa DataType && _source_callable_type(parameter) &&
            return parameter
    end
    return nothing
end

function _type_mentions(@nospecialize(T), @nospecialize(target),
                        seen::Base.IdSet{Any} = Base.IdSet{Any}())
    T === target && return true
    T isa DataType || return false
    T in seen && return false
    push!(seen, T)
    for parameter in T.parameters
        _type_mentions(parameter, target, seen) && return true
    end
    return false
end

# Called by inference with the recurring method, the callee method used for
# its limit heuristics, the new call signature, and the signature of the
# earlier frame of the same method; `true` means the recursion is well founded.
function _source_call_recursion_well_founded(
        @nospecialize(method), @nospecialize(callee),
        @nospecialize(sig), @nospecialize(parent_sig))
    inner = _source_call_identity(sig)
    outer = _source_call_identity(parent_sig)
    (inner === nothing || outer === nothing) && return false
    return inner !== outer && !_type_mentions(inner, outer)
end

# Every method a nested source operation re-enters. Run once all of them are
# defined (`ReactiveKernels.jl`, before the precompile workload).
function _mark_source_call_recursion!()
    hasfield(Method, :recursion_relation) || return nothing
    methods_to_mark = Method[
        which(Tuple{_KernelSourceOp,Vararg{Any}}),
        which(Tuple{_KernelSourceFunction,Vararg{Any}}),
        which(Tuple{_KernelBranch,Vararg{Any}}),
        which(Tuple{_KernelReduction,Vararg{Any}}),
        which(Tuple{_IgnoredThrowFunction,Vararg{Any}}),
        methods(_kernel_source_call)...,
        methods(_ignored_throw_call)...,
    ]
    for method in methods_to_mark
        method.recursion_relation = _source_call_recursion_well_founded
    end
    return nothing
end

# Whether a plate's dose-outer lowering of `reduction` keeps the authored
# semantics for these types: a concrete accumulator type `T` that the seed and
# every step return unchanged, an `Int` gather index and a `Vector` gather
# source (or a contiguous column of a dense matrix). `S` is the tuple of the
# recipe's per-cell argument types. Everything
# here is a type computation, folded when the plate body is compiled.
@generated function _plate_reduction_ready(
        reduction::_KernelReduction{II,XI,KI,AI}, ::Type{T},
        ::Type{S}) where {II,XI,KI,AI,T,S<:Tuple}
    types = Any[fieldtype(S, i) for i in 1:fieldcount(S)]
    select(positions) = Any[types[i] for i in positions]
    promote = GlobalRef(Base, :promote_op)
    quote
        isconcretetype($T) || return false
        iterator_type = $promote(reduction.iterator, $(select(II)...))
        element = eltype(iterator_type)
        isconcretetype(element) || return false
        $promote(reduction.init, $(select(XI)...)) === $T || return false
        $promote(reduction.step, $T, element, $(types...)) === $T || return false
        $promote(reduction.index, element, $(select(KI)...)) === Int || return false
        source = $promote(reduction.array, $(select(AI)...))
        _plate_dense_source(source) || return false
        $promote(reduction.step_in, $T, element, eltype(source), $(types...)) === $T ||
            return false
        $promote(reduction.step_out, $T, element, $(types...)) === $T
    end
end
# Whether the coefficient-outer lowering of an `evalpoly(x, c)` cell keeps
# Base's semantics for these types: a concrete cell type `T` that the seed
# `c[end]` already has and every `muladd(x, acc, c[i])` keeps, coefficients in
# an `AbstractVector` (a tuple keeps Base's unrolled method), and a one-axis
# domain. Folded when the plate body is compiled.
@generated function _plate_evalpoly_ready(::Type{T}, ::Type{X}, ::Type{C},
                                          output_axes) where {T,X,C}
    output_axes <: Tuple{Any} && C <: AbstractVector || return false
    quote
        isconcretetype($T) && eltype($C) === $T &&
            $(GlobalRef(Base, :promote_op))($(GlobalRef(Base, :muladd)), $X, $T, $T) === $T
    end
end
# Whether a scan's strip region can run over views of its outer lanes: every
# lane and every other scan sequence is a vector over one common axis.
@inline _plate_strip_ready() = false
@inline _plate_strip_ready(first::AbstractVector, rest...) =
    _plate_strip_same_axis(axes(first, 1), rest...)
@inline _plate_strip_ready(first, rest...) = false
@inline _plate_strip_same_axis(axis) = true
@inline _plate_strip_same_axis(axis, lane::AbstractVector, rest...) =
    axes(lane, 1) == axis && _plate_strip_same_axis(axis, rest...)
@inline _plate_strip_same_axis(axis, lane, rest...) = false
# Steps per strip of a strip-fused scan: the strip buffers of its plates stay
# in the first-level cache between the plate cells and the steps that read them.
const _PLATE_STRIP = 128

# Cells per tile of a coefficient-outer pass: the accumulators and the cell
# values of one tile stay in the first-level cache across the passes.
const _PLATE_FOLD_TILE = 256

_plate_dense_source(::Type) = false
_plate_dense_source(::Type{<:Vector}) = true
_plate_dense_source(::Type{S}) where {T,P<:Matrix,S<:SubArray{T,1,P}} =
    S === _replicated_column_type(P)

# The gather index of one dose-outer pass as a function of the cell: the
# reduction's `index` part at the pass element over the cell's arguments, each
# either read per cell from a plate argument whose axes are the plate's (so
# every cell is in bounds, and a caller's `@inbounds` reaches the read, as in
# Base's broadcast) or shared. A struct, not a closure: a closure in a
# generated kernel body is an opaque closure over `Any` arguments, which
# dispatches dynamically on every call.
struct _PlateCellArgument{A}
    values::A
end
struct _PlateSharedArgument{A}
    value::A
end
Base.@propagate_inbounds _plate_argument(argument::_PlateCellArgument, cell) =
    Base.Broadcast._broadcast_getindex(argument.values, cell)
@inline _plate_argument(argument::_PlateSharedArgument, cell) = argument.value
Base.@propagate_inbounds _plate_arguments(::Tuple{}, cell) = ()
Base.@propagate_inbounds _plate_arguments(arguments::Tuple, cell) =
    (_plate_argument(first(arguments), cell),
     _plate_arguments(Base.tail(arguments), cell)...)
struct _PlateCellIndex{F,J,A<:Tuple}
    index::F
    element::J
    arguments::A
end
Base.@propagate_inbounds (index::_PlateCellIndex)(cell) =
    index.index(index.element, _plate_arguments(index.arguments, cell)...)
# Whether every per-cell argument of a gather index spans the plate's axes.
@inline _plate_spans(output_axes) = true
@inline _plate_spans(output_axes, values, rest...) =
    axes(values) == output_axes && _plate_spans(output_axes, rest...)

# The first gather index of a dose-outer pass and whether the index advances
# by exactly one per cell in `cells` order. A pure index map that passes this
# check reads one contiguous window of the source. `index` reads only cells of
# `cells` (`_PlateCellIndex` over arguments spanning them). Kept out of line: compiled
# alone, a shifted-lattice index (`t - shift`) folds the whole scan away, while
# inlined into the pass nest it ran as a full extra pass.
@noinline function _plate_affine_index(index, cells)
    n = length(cells)
    n == 0 && return (false, 0)
    first_index = (@inbounds index(cells[1]))::Int
    first_index > typemax(Int) - (n - 1) && return (false, first_index)
    for position in 2:n
        (@inbounds index(cells[position])) == first_index + (position - 1) ||
            return (false, first_index)
    end
    (true, first_index)
end

# Every cell of a plate, in coordinate order. A single axis iterates its range
# directly: `CartesianIndices` iteration of one axis is a loop the compiler does
# not vectorize (measured: a third of a dose-outer superposition read went to
# its `__inc` on the seed pass), while several axes keep the Cartesian
# iteration, which avoids converting linear positions per cell.
@inline _plate_cells(cells::CartesianIndices{1}) =
    Base.Generator(CartesianIndex, only(cells.indices))
@inline _plate_cells(cells::CartesianIndices) = cells

# The cell positions whose gather index `first_index + position - 1` lies in
# `source`'s range, as `lo:hi` within `1:n` (empty when none do).
@inline function _plate_gather_window(source::AbstractVector, first_index::Int,
                                      n::Int)
    lo = clamp(widen(firstindex(source)) - first_index + 1, 1, n + 1)
    hi = clamp(widen(lastindex(source)) - first_index + 1, lo - 1, n)
    (Int(lo), Int(hi))
end

# Tensorized authored plates keep slice collections structural instead of
# materializing Base.Slices.  A backend can consume the parent array as one
# batched value, while the generic fallback preserves ordinary eachcol
# broadcast semantics.
struct _TensorizedEachcol{A}
    parent::A
end

struct _TensorizedPlateBatch{A,S}
    values::A
    schema::S
end

_TensorizedPlateBatch(values) = _TensorizedPlateBatch(values, nothing)

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
# Keep array-valued lanes identifiable until their consumer decides how to
# arrange them. A bare lanes-leading tensor is not a collection of arrays:
# `stack` would flatten its scalar entries and silently lose the lane layout.
@inline _tensorized_plate_pointwise(value) = _tensorized_plate_materialize(value)
@inline _tensorized_plate_pointwise(value::_TensorizedPlateBatch{<:AbstractArray}) =
    ndims(value.values) > 1 ? value : value.values
@inline _tensorized_plate_pointwise(
    value::_TensorizedPlateBatch{<:Tuple,<:AbstractArray}) = value

# The marker is owned by RK, so ordinary helpers can use Base.stack without a
# backend-specific helper method. Only the fixed tensor rank determines this
# permutation; no recipe or indexing operation is replicated per lane.
Base.stack(value::_TensorizedPlateBatch; dims = :) =
    _tensorized_plate_stack(_tensorized_plate_materialize(value), dims)
@inline _tensorized_plate_stack(value::AbstractArray, ::Colon) =
    _tensorized_plate_stack(value, ndims(value))
@inline function _tensorized_plate_stack(value::AbstractArray, dim::Integer)
    rank = ndims(value)
    1 <= dim <= rank || throw(ArgumentError("stack dimension must be in 1:$rank"))
    permutation = ntuple(i -> i == dim ? 1 : i < dim ? i + 1 : i, rank)
    permutedims(value, permutation)
end
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
    _tensorized_plate_dispatch(operation, args)
end

@inline function _tensorized_plate_dispatch(operation, args::Tuple)
    _tensorized_plate_default_call(operation, args)
end

@inline function _tensorized_plate_default_call(operation, args::Tuple)
    marker = _tensorized_plate_marker(args)
    marker === nothing ?
        _tensorized_broadcast(operation, args...) :
        _tensorized_plate_call(marker, operation, args)
end

# Branch metadata identifies the exact condition ports. A scalar or atomic
# operand has no lane axis, so a condition reading only those operands is one
# lazy decision around the whole batch. Keep lane-dependent conditions in the
# cell. In particular, a singleton lane array is still a lane operand.
@inline _plate_branch_shared(arg) = false
@inline _plate_branch_shared(arg::Number) = true
@inline _plate_branch_shared(arg::Base.RefValue) = true
@inline _plate_branch_value(arg) = arg
@inline _plate_branch_value(arg::Base.RefValue) = arg[]
@inline _plate_branch_operands(predicate, args::Tuple) = args
@inline _plate_branch_empty(arg) = false
@inline _plate_branch_empty(arg::AbstractArray) = isempty(arg)
@inline _plate_branch_empty(arg::_TensorizedEachcol) = size(arg.parent, 2) == 0
@inline _plate_branch_empty(arg::_TensorizedPlateBatch) =
    _plate_branch_empty(arg.values)
@inline _plate_branch_empty(args::Tuple) = any(_plate_branch_empty, args)

@inline @generated function _plate_branch_arguments(args::Tuple, ::Val{I}) where {I}
    Expr(:tuple, [:(getfield(args, $index)) for index in I]...)
end

# Keep the fixed set of shared operands as direct arguments to the authored
# predicate. A mapped tuple followed by a splat loses its argument types when
# tracing a generated source callable after a bound-data branch split.
@inline @generated function _plate_branch_predicate(
        condition, args::Tuple{Vararg{Any,N}}) where {N}
    forwarded = [:(_plate_branch_value(getfield(args, $index))) for index in 1:N]
    :(condition($(forwarded...)))
end

struct _PlateBranchArm{I,O}
    operation::O
end

@inline @generated function (arm::_PlateBranchArm{I})(args::Vararg{Any,N}) where {I,N}
    operation = :(getfield(arm, :operation))
    inputs = Any[:(getfield(args, $index)) for index in I]
    # An arm may ignore every lane operand (a constant fallback, for example).
    # Preserve the original broadcast domain and marker with ignored anchors;
    # selecting an arm must not shrink its axes or turn it into a host loop.
    for index in 1:N
        index in I && continue
        operation = :(_LaneAnchored($operation))
        pushfirst!(inputs, :(getfield(args, $index)))
    end
    :(_tensorized_plate_call($operation, $(inputs...)))
end

@inline function _tensorized_plate_dispatch(
        operation::_KernelSourceOp{D,F,N,T}, args::Tuple) where
        {D,F,N<:_KernelBranch,T<:_KernelBranch}
    branch = operation.tensor_f
    CI, TI, EI = typeof(branch).parameters[1:3]
    condition_args = _plate_branch_arguments(args, Val(CI))
    # Empty domains run no cells and must not evaluate a new shared condition.
    # Their existing batch lowering also owns shape/type validation.
    if !all(_plate_branch_shared, condition_args) || _plate_branch_empty(args)
        return _tensorized_plate_default_call(operation, args)
    end
    predicate = _plate_branch_predicate(branch.condition, condition_args)
    args = _plate_branch_operands(predicate, args)
    yes = _KernelSourceOp(Val(D), Val(F), operation.f.then_arm,
        branch.then_arm, operation.ignored_throws)
    no = _KernelSourceOp(Val(D), Val(F), operation.f.else_arm,
        branch.else_arm, operation.ignored_throws)
    _recurrence_branch(predicate, _PlateBranchArm{TI,typeof(yes)}(yes),
        _PlateBranchArm{EI,typeof(no)}(no), args)
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
    ReactiveKernels.traced(f, args...)

The call a kernel makes in place of `f(args...)` when a tracing backend
(Reactant) compiles it.  The default is `f(args...)`.  Add a method for your
own function to give it a tracing implementation, while native execution keeps
calling `f` itself:

```julia
# native: a hand-written in-place loop, fast but not traceable
superpose(plan, units, weights) = ...
# under tracing: any traceable implementation, here a prepared plate kernel
ReactiveKernels.traced(::typeof(superpose), plan, units, weights) =
    SUPERPOSE_KERNEL(plan, units, weights)
```

The method may be any Julia the backend can trace: a prepared kernel, a
`derivative_rule`, or backend code in the package's own Reactant extension.
Dispatch selects it by the authored argument types, so annotate the arguments
that choose the implementation and leave traced arguments untyped.
[`@traceable`](@ref) defines such a method from a helper's own body.

A kernel routes a positional call through `traced` only in the code a tracing
backend runs, and only for functions Base, Core and ReactiveKernels do not
own; a call with keyword arguments is not routed.  Define the method before
the `@kernel` that calls `f`: a recipe that is exactly `x = f(ports...)` keeps
`f` itself as its operation unless `f` already has a `traced` method when the
kernel is defined.  The two implementations are separate code: test the traced
one against the native one.
"""
@inline traced(f::F, args::Vararg{Any,N}) where {F,N} = f(args...)

# `get(A, i, default)` in a tensorized fused body routes through this hook: the
# total gather that reads `A[i]` when `i` is a valid index and yields `default`
# otherwise (a causal response that is zero before its dose, a lookup table
# with a fill value).  Native semantics are exactly `Base.get`.  Base decides
# with a lazy branch on the bounds test, so a tracing extension keeps that
# branch lazy on a traced index: the read of `A[i]` stays inside the taken arm
# and an out-of-range index is never read (`docs/src/constraints.md`).
@inline _tensorized_get(args...) = get(args...)

# A value a retained generator loop reads but never carries.
# `ReactantCore.@trace for` hands every variable its body references to the
# tracer as a loop argument, and the tracer rebuilds a struct with traced
# fields — impossible for a host struct whose fields are concretely typed (a
# schedule plan holding a `Vector{Int}`), so a loop reading such a value failed
# with `NoFieldMatchError`.  An untraced capture crosses the loop boundary in
# this wrapper, which a tracing extension passes through unchanged.  A traced
# capture stays a loop argument, entered as a fresh tracer object
# (`_loop_capture_traced`): `@trace` writes each loop result back into every
# traced object the body reads, so the caller's own tracer would be rebound
# and returned as an aliased program output, which XLA export rejects for a
# zero-sized one.  A partly traced tuple or named tuple (a model holding
# traced scalars beside a host matrix) is opened leaf by leaf, so its host
# leaves cross wrapped too: handed bare to the tracer, a host matrix became a
# matrix of traced scalars, one loop argument per element, which a traced
# index cannot gather.  Without a tracing backend the wrapper is opened again
# at every use and nothing else changes.
struct _LoopHostValue{T}
    value::T
end
ReactantCore.is_traced(::_LoopHostValue) = false
ReactantCore.is_traced(::_LoopHostValue, ::Base.IdSet) = false
@inline _loop_capture(x) =
    ReactantCore.is_traced(x) ? _loop_capture_traced(x) : _LoopHostValue(x)
_loop_capture_traced(x) = x
_loop_capture_traced(x::Union{Tuple,NamedTuple}) = map(_loop_capture, x)
@inline _loop_open(x) = x
@inline _loop_open(x::_LoopHostValue) = x.value
@inline _loop_open(x::Union{Tuple,NamedTuple}) = map(_loop_open, x)

# A carried local of a retained loop, re-bound just before the loop.  When
# anything the loop reads is traced (the `witnesses`: its captures and its
# carry), a tracing extension replaces the value by a fresh traced copy
# (`_loop_seed_traced`), because `ReactantCore.@trace` carries a value only by
# updating a traced object that exists before the loop: a host seed such as
# `acc = 0.0` has none, and the loop's result was silently dropped (the loop
# returned `0.0`).  With nothing traced the value is returned unchanged.
@inline _loop_seed(x, witnesses...) =
    _loop_any_traced(witnesses) ? _loop_seed_traced(x) : x
@inline _loop_any_traced(::Tuple{}) = false
@inline _loop_any_traced(witnesses::Tuple) =
    ReactantCore.is_traced(first(witnesses)) || _loop_any_traced(Base.tail(witnesses))
_loop_seed_traced(x) = x

# Whether a retained `for` loop's range is an untraced host range with no
# elements, so the loop is skipped instead of traced (`_kernel_tensorized_loop`).
# A range with a traced bound is never skipped.  The colon form takes the
# bounds `@trace for` reads from `a:b` / `a:s:b` syntax.
@inline _loop_host_empty(range) = !ReactantCore.is_traced(range) && isempty(range)
@inline _loop_host_empty_colon(bounds...) =
    !_loop_any_traced(bounds) && isempty((:)(bounds...))

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
    # The value-independent work of this graph's bound preparations, reused by
    # `prepare(…; bound)` (`_graph_preparations`, graphops.jl); `nothing` until
    # the first one. Not part of the graph's meaning.
    preparations::Any
    # This graph's identity high-water mark travels with a package image.
    # A dependency's process-local counter does not retain allocations made
    # while precompiling a consumer, so it alone cannot extend a loaded graph.
    value_id_floor::Int
end
Graph(values, recipes, producers, aliases, version) =
    Graph(values, recipes, producers, aliases, version, nothing)
Graph(values, recipes, producers, aliases, version, preparations) =
    Graph(values, recipes, producers, aliases, version, preparations,
          maximum(keys(values); init = 0))
Graph() = Graph(Dict{Int,Value}(), Recipe[], Dict{Int,Vector{Int}}(),
                Dict{Int,Int}(), 0)

function _register!(g::Graph, v::Value)
    g.values[v.id] = v
    g.value_id_floor = max(g.value_id_floor, v.id)
    v
end

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
    v = Value{T}(_next_value_id(g.value_id_floor), name)
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
    _add_recipe!(g, inputs, outputs, op, cost, cse_key, effectful, source)
end

# Compare canonical value ids elementwise. `==` on tuples of unknown length is an
# abstract call that any package method on tuples invalidates.
function _canonical_ids_equal(g::Graph, values::Tuple{Vararg{Value}}, ids::Vector{Int})
    length(values) == length(ids) || return false
    for (value, id) in zip(values, ids)
        canon_id(g, value.id) == id || return false
    end
    true
end

# Registration stores callable and provenance values as graph metadata. Its body
# does not need a new inferred executable for every numerical operation type.
Base.@nospecializeinfer function _add_recipe!(g::Graph,
        inputs, outputs, @nospecialize(op),
        cost::Real, @nospecialize(cse_key), effectful::Bool, @nospecialize(source))
    ins = _astuple(inputs)
    outs = _astuple(outputs)
    recipe_cost = Float64(cost)
    if !isfinite(recipe_cost) || recipe_cost < 0
        throw(ArgumentError("recipe cost must be finite and non-negative, got $cost"))
    end
    op isa Type && (op = _TypeOperation{op}())
    # Opt-in structural CSE (gist §8): if a prior recipe carries the same
    # non-`nothing` cse_key, the same canonical inputs, and the same output
    # arity, it computes the same thing. Alias the new outputs onto the existing
    # producer's outputs instead of adding a duplicate recipe.
    if cse_key !== nothing && !effectful
        canon_ins = Int[canon_id(g, v.id) for v in ins]
        for r in g.recipes
            r.effectful && continue
            r.cse_key === nothing && continue
            isequal(r.cse_key, cse_key) || continue
            length(r.outputs) == length(outs) || continue
            _canonical_ids_equal(g, r.inputs, canon_ins) || continue
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
