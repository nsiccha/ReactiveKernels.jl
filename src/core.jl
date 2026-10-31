# Core graph identities: Value, Recipe, Graph.
#
# These objects are *compile/planning-time metadata only*. None of them are
# consulted inside a prepared kernel (see codegen.jl); the hot path sees only
# ordinary Julia values.

# Every `Value` has an identity independent of its name or of any particular
# graph. Identity must not depend solely on the name (gist §5), so two values
# may share a name yet remain distinct.
#
# Identities also cross process boundaries: a package image keeps the values its
# kernels were authored with, and the process that loads it makes values of its
# own. An identity is therefore the namespace that allocated it plus a number
# within that namespace. Each process allocates in a namespace of its own
# (`__init__`), so values from different package images and from the loading
# process never coincide, and a value one image refers to from another keeps its
# identity: loading preserves which namespace object it belongs to.
"""
    ValueNamespace

An allocator of `Value` identities. Identities from different namespaces are
distinct; a namespace is compared by object identity.
"""
mutable struct ValueNamespace
    # Identities allocated so far. Atomic because graphs may be built from
    # concurrent tasks, all allocating in this process's namespace.
    @atomic count::Int
    # The hash and order of identities. A namespace's address differs in every
    # process that loads it from a package image; this key travels with it.
    key::UInt
end
ValueNamespace(key::UInt) = ValueNamespace(0, key)
# A namespace is an identity, not data: a deep copy of a graph or a table keyed
# by value identities refers to the same values, so it keeps the namespace.
Base.deepcopy_internal(namespace::ValueNamespace, ::IdDict) = namespace

"""
    ValueId

The identity of a `Value`: the namespace that allocated it and its number there.
"""
struct ValueId
    namespace::ValueNamespace
    number::Int
end
Base.:(==)(a::ValueId, b::ValueId) =
    a.namespace === b.namespace && a.number == b.number
Base.hash(id::ValueId, h::UInt) = hash(id.number, hash(id.namespace.key, h))
# Allocation order within a namespace; namespaces in the order of their keys.
Base.isless(a::ValueId, b::ValueId) = a.namespace === b.namespace ?
    isless(a.number, b.number) : isless(a.namespace.key, b.namespace.key)
Base.show(io::IO, id::ValueId) =
    print(io, "ValueId(", id.number, "@", string(id.namespace.key; base = 16), ")")

# An identity as one `Int`, for the stateful kernel factory: its plan keys are
# type parameters, which a `ValueId` (it refers to its mutable namespace) cannot
# be. The number is the same in every process. It is not an identity by
# itself, so a graph whose identities would share one is refused
# (`_check_canon_ints`) rather than having two values merged.
function _canon_int(id::ValueId)
    n = reinterpret(Int, hash(id.number, id.namespace.key))
    n == 0 ? 1 : n
end
function _check_canon_ints(g)
    length(Set(_canon_int(id) for id in keys(g.values))) == length(g.values) ||
        throw(ArgumentError("two values of one graph share a kernel-factory " *
                            "identity number; rebuild the graph"))
    g
end

# The namespace of this package's own image. A process that loads it replaces
# it with one of its own (`__init__`).
const _VALUE_NAMESPACE = Ref(ValueNamespace(rand(RandomDevice(), UInt)))
function _next_value_id()
    namespace = _VALUE_NAMESPACE[]
    ValueId(namespace, @atomic namespace.count += 1)
end

function __init__()
    # A process that loads package images keeps a fixed key, so the order of
    # its own values relative to imported ones is the same in every session; a
    # process that generates an image gives that image a key of its own.
    generating = ccall(:jl_generating_output, Cint, ()) != 0
    _VALUE_NAMESPACE[] = ValueNamespace(generating ? rand(RandomDevice(), UInt) : zero(UInt))
    # Programs that package images loaded from now on compiled (`compile`).
    push!(Base.package_callbacks, _register_image_programs)
    # This process reads its own project's type-stability preferences.
    foreach(preference -> preference[] = nothing, _TYPE_STABILITY_PREFERENCES)
    nothing
end

"""
    Value{T}

A stable graph identity for a runtime value of Julia type `T`. `id` is the
identity (a `ValueId`, unique across processes and package images); `name`
exists only for diagnostics and for generated-code readability. Values are
immutable and cheap to hash/compare.
"""
struct Value{T}
    id::ValueId
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
    _KernelSourceOp{F,IG,CI}

An immutable wrapper marking a recipe operation SYNTHESIZED from captured `@kernel` source as
COMPILER-OWNED provenance (RK 07:21). Authoring wraps ONLY the anonymous-closure path of
`_kernel_operation` in this; a bare exact identity (`cholesky`/`+`/…) stays raw and is identity/domain
validated (`_kernel_source_op_callee` lists the Base callees that stay wrapped). `token` is a
definition-unique gensym baked in at graph build — NOT a security boundary (an internal
constructor cannot prevent deliberate internal misuse); it is trusted only because the
supported authoring path is the ONLY thing that wraps a closure, so an arbitrary public Graph closure is
never auto-wrapped. The token and the form are values, not type parameters (RK 0.1.27): the
operation's type is its source's, so every build of one program shares one executable type
and a prepared kernel holds the plan's operations themselves, with no execution copy.
`form` (RK 07:24)
distinguishes a `:portcall` — a call THROUGH A PORT, `callable(args…)`, whose first input is the callable
source and the rest are ordered args — from a general `:fused` expression, so a prepared handle can
self-derive the DESTINATION contract (a port-call with one owned buffer + one owned scalar output →
`f(dest, args…)::scalar`) from source SHAPE + typed slot roles, never from a name/Recipe id/inspection.
`f` is the source record (`_KernelSource`) or the metadata around one (`_KernelBranch`,
`_KernelSplit`, `_KernelReduction`); `mod` the module the source's names resolve in
(`nothing` for an operation built without one). A prepared kernel's callables table holds
`f`, never this wrapper: the record is executable state, the wrapper is provenance.
`IG` holds the captured source's throw-stripped twin, selected only by an
explicit `on_error = :ignore` preparation; older internal fixtures use `nothing`.
`CI` retains the callable and lowering mode of an exact positional call through
a constant binding, when present. Composition can share these calls without
equating separately authored fused expressions or changing their definition tokens.
The call forwards INLINE. A RAW anonymous closure inserted into a Graph carries no wrapper and is
rejected as opaque when captured into a prepared handle.
"""
struct _KernelSourceOp{F,IG,CI}
    token::Symbol
    form::Symbol
    f::F
    ignored_throws::IG
    call_identity::CI
    mod::Any
end

# Every authored recipe creates source operations of new concrete types, and
# a constructor call compiles once per concrete type. Graph assembly only stores
# the operation, so build it from its parameters without specializing. The
# prepared kernel still sees the exact concrete operation type.
Base.@nospecializeinfer function _KernelSourceOp(
        @nospecialize(token::Val), @nospecialize(form::Val),
        @nospecialize(f),
        @nospecialize(ignored_throws = nothing), @nospecialize(call_identity = nothing),
        @nospecialize(mod = nothing))
    T = _KernelSourceOp{typeof(f), typeof(ignored_throws), typeof(call_identity)}
    _kernel_new_instance(T, (_val_parameter(token)::Symbol, _val_parameter(form)::Symbol,
                             f, ignored_throws, call_identity, mod))::_KernelSourceOp
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

# A module binding can be shadowed by a lexical callable captured in the
# execution body. Its value at construction need not identify later calls,
# especially when Julia retains a reassigned local in a shared Core.Box.
# The runtime callable must also match the binding resolved by lowering,
# which can have replaced the call and removed its lexical capture entirely.
Base.@nospecializeinfer function _KernelSourceOp(
        @nospecialize(token::Val), @nospecialize(form::Val), @nospecialize(f),
        @nospecialize(ignored_throws), @nospecialize(call_identity),
        @nospecialize(capture_name), @nospecialize(resolved_callable), @nospecialize(mod))
    if call_identity !== nothing &&
       (first(call_identity) !== resolved_callable ||
        (capture_name !== nothing &&
         capture_name in _source_capture_names(f)))
        call_identity = nothing
    end
    _KernelSourceOp(token, form, f, ignored_throws, call_identity, mod)
end

# An authored source's executable record (RK 0.1.27, stage 3 of the
# operation-table migration): what runs, as data. `captures` are the values
# the source closed over (a `NamedTuple`, empty when none). `companion` is the
# source compiled as a runtime-generated function over `(captures, params...)`,
# callable from any later world without a method query; its type carries the
# canonical source's id and the authored parameter names, so every build of
# one program shares it and the compiled code behind it, and the names are
# read from the type. `native` is the closure Julia itself lowered, kept only
# for a source Julia must own (a local function, a generator or
# comprehension, an exception scope, a shared `Core.Box` cell): such a record
# has no companion and calls the closure, with a latest-world fallback where
# its method is not yet visible.
#
# The record holds executable state only: no token, parameter-name or module
# value. The token and the module live on the recipe's operation
# (`_KernelSourceOp`), which the plan and `lowered_recipes` keep; the
# parameter names are read from the companion's type (its argument names
# after the captures environment, a gensym among them in `_canonical_locals`'
# spelling, so two builds of one source have one type). The callables table
# of a prepared kernel holds the record, so a capture-free table is free of
# heap references that Enzyme would otherwise have to prove read-only
# (b259aa13f: such fields cost the Imperial COVID model's gradients 446 → 762
# allocations per call and 5 → 25 minutes of compile time).
struct _KernelSource{C,R,N} <: Function
    captures::C
    companion::R
    native::N
end

# One instance for every source type (`@nospecialize`, `getfield`): lowering
# reads these for every recipe of a program.
Base.@nospecializeinfer _source_record(@nospecialize(f)) = nothing
Base.@nospecializeinfer _source_record(@nospecialize(f::_KernelSource)) = f
Base.@nospecializeinfer _source_captures(@nospecialize(f::_KernelSource)) = getfield(f, :captures)
# The native closure of a source Julia owns, else `nothing`.
Base.@nospecializeinfer _source_native(@nospecialize(f::_KernelSource)) = getfield(f, :native)
# The source's parameter names when it takes `n`, else `nothing`: the
# companion's argument names after its captures environment (its type; a
# gensym parameter in `_canonical_locals`' spelling, `##name#k`), or the
# native closure's method. `_spelled_parameter_name` maps a name to the
# spelling of the recipe's source.
Base.@nospecializeinfer function _source_params(@nospecialize(f::_KernelSource), n)
    native = getfield(f, :native)
    native === nothing || return _closure_parameter_names(native, n)
    companion = _native_generated_function(getfield(f, :companion))
    names = (typeof(companion)::DataType).parameters[1]
    names isa Tuple && length(names) == n + 1 || return nothing
    Symbol[names[2:end]...]
end
# The same, as `source` spells them.
Base.@nospecializeinfer function _source_params(@nospecialize(f::_KernelSource), n,
                                                @nospecialize(source))
    params = _source_params(f, n)
    params === nothing && return nothing
    spelled = _source_symbols!(Set{Symbol}(), source)
    Symbol[_spelled_parameter_name(name, spelled) for name in params]
end
# The captured names of the source behind `f` (a record, or metadata whose
# call is one); none for any other callable.
Base.@nospecializeinfer function _source_capture_names(@nospecialize(f))
    record = _source_record(f)
    record === nothing ? () : keys(getfield(record, :captures))
end

# Defined before the generated call below: a generator runs in the world of
# its definition and sees no later method.
# The companion's body over the captures and the arguments, as
# `RuntimeGeneratedFunctions.generated_callfunc` splices it, so the call is
# this record method and not that generated function's: a source embedding
# other sources (a plate cell calling an endpoint) nests calls of one method
# over distinct record types, which `_mark_source_call_recursion!` tells
# inference is well founded, where nested `generated_callfunc` frames over
# distinct generated-function types hit Julia's recursion limit and infer
# `Any` (the atlas model's branch cells and a cell query's gathered cell lost
# their concrete types). `nothing` when the body is not in the cache.
function _companion_call_body(@nospecialize(R), forwarded)
    R isa DataType && R <: _PrecompileWarmFunction && (R = R.parameters[1])
    R isa DataType && R <: RuntimeGeneratedFunctions.RuntimeGeneratedFunction || return nothing
    argnames, cache_tag, _, id = R.parameters[1:4]
    body = RuntimeGeneratedFunctions._lookup_body(cache_tag, id)
    body === nothing && return nothing
    argnames isa Tuple && length(argnames) == length(forwarded) + 1 || return nothing
    # Every argument is read before any parameter is bound: a parameter may
    # be named like this method's own `f` or `args`.
    temporaries = [gensym(:argument) for _ in forwarded]
    environment = gensym(:captures)
    setup = Any[Expr(:(=), environment, :(getfield(f, :captures)))]
    append!(setup, (Expr(:(=), temporary, arg) for (temporary, arg) in zip(temporaries, forwarded)))
    push!(setup, Expr(:(=), argnames[1], environment))
    append!(setup, (Expr(:(=), name, temporary)
                    for (name, temporary) in zip(argnames[2:end], temporaries)))
    Expr(:block, Expr(:meta, :inline), setup..., body)
end

# The call is a forwarding shim and is never a call boundary: both generated
# bodies declare themselves inline. `@inline` on a `@generated` definition does
# not reach the generated code, and the call-site `@inline` of
# `_kernel_source_call` did not inline the shim once the source body was large
# (a scan step's log-sum-exp): such a step became a call per scan step, and
# native Enzyme reverse through it cost 2.5–3× the same kernel with the step
# inlined (HMM forward pass, snag `rkppl-hmm-forwar-89257681`).
@inline @generated function (f::_KernelSource{C,R,N})(args::Vararg{Any,K}) where {C,R,N,K}
    forwarded = [:(getfield(args, $index)) for index in 1:K]
    if N === Nothing
        spliced = _companion_call_body(R, forwarded)
        spliced === nothing || return spliced
        return Expr(:block, Expr(:meta, :inline),
            :(Base.@inline RuntimeGeneratedFunctions.generated_callfunc(
                getfield(f, :companion), getfield(f, :captures), $(forwarded...))))
    end
    # A native closure's method may be newer than the caller's world (a
    # kernel authored at run time). The query is `Core._hasmethod` on the
    # complete signature, which inference folds to a constant in the caller's
    # world on every supported Julia; `hasmethod` validates its tuple type in
    # a loop that stayed in every call from Julia 1.12 on, so a plate cell
    # walked its argument types once per cell and did not vectorize.
    signature = Expr(:curly, GlobalRef(Core, :Tuple),
                     :($(GlobalRef(Core, :Typeof))(getfield(f, :native))), args...)
    quote
        $(Expr(:meta, :inline))
        if Core._hasmethod($signature)
            Base.@inline getfield(f, :native)($(forwarded...))
        else
            Base.@inline _kernel_latest_native_call(getfield(f, :native), $(forwarded...))
        end
    end
end

@inline function _kernel_latest_native_call(f, args::Vararg{Any,N}) where {N}
    result_type = Core.Compiler.return_type(f, typeof(args))
    Base.invokelatest(f, args...)::(result_type === Union{} ? Any : result_type)
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
# heuristic already inlines the annotation changes nothing. Julia 1.12 inlines
# this closure on its own.
@inline @generated function _kernel_source_call(
        op::_KernelSourceOp, args::Vararg{Any,N}) where {N}
    forwarded = [:(getfield(args, $index)) for index in 1:N]
    value = gensym(:value)
    Expr(:block,
         Expr(:inline, true),
         Expr(:local, Expr(:(=), value, :(op.f($(forwarded...))))),
         Expr(:inline, false),
         value)
end
# The entry call forwards positionally too. Julia's inliner rewrites an
# `args...` splat into a direct call only up to 32 elements
# (`max_tuple_splat`, 1.10 and 1.12 alike); past that the call stays a dynamic
# `Core._apply_iterate`, codegen materializes the arguments in one tuple, and
# Enzyme meets a constant array stored next to active values: a 32-argument
# module call with a bound data vector failed reverse mode with
# `EnzymeRuntimeActivityError` (snag `rkppl-module-cal-79cad594`).
@inline @generated function (op::_KernelSourceOp)(args::Vararg{Any,N}) where {N}
    forwarded = [:(getfield(args, $index)) for index in 1:N]
    :(_kernel_source_call(op, $(forwarded...)))
end
# A deep copy of a plan copies its operations; the authoring module is a
# global object the copy shares (`deepcopy` refuses a `Module`).
function Base.deepcopy_internal(op::_KernelSourceOp, stackdict::IdDict)
    haskey(stackdict, op) && return stackdict[op]
    copy = _KernelSourceOp(Val(getfield(op, :token)), Val(getfield(op, :form)),
                           Base.deepcopy_internal(getfield(op, :f), stackdict),
                           Base.deepcopy_internal(getfield(op, :ignored_throws), stackdict),
                           Base.deepcopy_internal(getfield(op, :call_identity), stackdict),
                           getfield(op, :mod))
    stackdict[op] = copy
end

# One instance for every operation type: the readers below run for every
# recipe of a program.
Base.@nospecializeinfer kernel_sourceop_token(@nospecialize(op::_KernelSourceOp)) =
    getfield(op, :token)::Symbol
Base.@nospecializeinfer kernel_sourceop_form(@nospecialize(op::_KernelSourceOp)) =
    getfield(op, :form)::Symbol
# The element type of an empty plate is Julia's empty-broadcast type: inferred
# from the authored native cell over the operands' native element types, with
# no cell evaluated. `nothing` for any other callable, or when inference gives
# no concrete type.
_kernel_native_result_type(f, ::Type{<:Tuple}) = nothing
Base.@nospecializeinfer function _kernel_native_result_type(
        @nospecialize(op::_KernelSourceOp), @nospecialize(argtypes::Type{<:Tuple}))
    T = Core.Compiler.return_type(getfield(op, :f), argtypes)
    isconcretetype(T) ? T : nothing
end
# Read-only graph gathers may share their parent's storage, including in queries.
# Keep a packed vector's contiguous slice as the parent: Base.view would eagerly
# materialize range[indices] to flatten this pair of views. As with other borrowed
# graph values, neither the parent nor the indices are writable cache storage.
@inline _readonly_gather(array::AbstractVector, indices::AbstractVector{<:Integer}) =
    view(array, indices)
@inline function _readonly_gather(
        array::SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true},
        indices::Vector{Int})
    checkbounds(array, indices)
    return Base.SubArray(array, (indices,))
end

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
# and `Matrix` operands, and contiguous range views of a `Vector` with their
# matrix reshapes, whose promoted element type is isbits, they return
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
# A contiguous range view of a `Vector` (such as a parameter block read in
# place from a packed vector) stores its elements as the vector does, and so
# does that view reshaped to a matrix: both are dense column-major operands.
# Base concatenates them beside arrays through a loop over a `Union` of
# operand types, which native Enzyme reverse rejects
# (`IllegalTypeAnalysisException` in `_typed_vcat`).
const _NativeVectorView = Base.FastContiguousSubArray{<:Any,1,<:Vector}
const _NativeVector = Union{Vector,_NativeVectorView}
const _NativeMatrix = Union{Matrix,Base.ReshapedArray{<:Any,2,<:_NativeVectorView}}
const _NativeCatOperand = Union{Number,_NativeVector,_NativeMatrix}
const _NativeColumnOperand = Union{Number,_NativeVector}

@inline _native_cat_width(a::_NativeColumnOperand) = 1
@inline _native_cat_width(a::_NativeMatrix) = size(a, 2)

# A layout `hcat`/`vcat` does not lay out calls Base for its error (or `vcat`'s
# fill of leading numbers).  Native Enzyme compiles that call even where it
# never runs, so a view operand reaches it as a copy: Base's values and errors,
# over the operand types it already differentiates.
@inline _native_cat_base_operand(a) = a
@inline _native_cat_base_operand(a::_NativeVectorView) = copy(a)
@inline _native_cat_base_operand(
    a::Base.ReshapedArray{<:Any,2,<:_NativeVectorView}) = copy(a)

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
@inline function _native_cat_block!(out, stride, row, column, a::AbstractArray)
    height = size(a, 1)
    for j in 1:_native_cat_width(a), i in 1:height
        @inbounds out[(column + j - 1) * stride + row + i] =
            a[(j - 1) * height + i]
    end
    out
end
# A dense operand of the output's element type is copied with
# `unsafe_copyto!` on Julia 1.10 only.  There, native Enzyme reverse
# differentiates the copy, while the element loop above fails static activity
# analysis on an empty fresh operand (the empty-allocation branch of
# `reactivekernels-use` §7ai; `test_native_concat_ad.jl`'s zero-row cases).
# From Julia 1.11, Enzyme differentiates `unsafe_copyto!` of array memory as
# an untyped `memmove` (its `myunsafe_copyto!` override), and native reverse
# fails with `EnzymeNoTypeError` ("Cannot deduce type of copy") whenever type
# analysis cannot recover the element type elsewhere: measured on Julia 1.12
# and 1.13 for `vcat([a], v)` in a kernel prepared with `bound=` data, the
# trajectory `@rkppl` builds for a scan
# (`test_native_concat_ad.jl`, "bound operands").  The typed element loop
# differentiates there, empty operands included.
@static if VERSION < v"1.11.0-"
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
end

@inline _native_hcat(args...) = hcat(args...)
@inline function _native_hcat(a::_NativeCatOperand, rest::_NativeCatOperand...)
    args = (a, rest...)
    T = Base.promote_eltypeof(args...)
    isbitstype(T) || return hcat(args...)
    height = size(a, 1)
    _native_cat_heights_equal(height, rest) ||
        return hcat(map(_native_cat_base_operand, args)...)::Matrix{T}
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
    _native_cat_widths_equal(width, rest) ||
        return vcat(map(_native_cat_base_operand, args)...)::Matrix{T}
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

# Native matrix products.  Base's `*` of a matrix and a vector or matrix with
# an operand that is not strided, such as a view of a lazily indexed vector,
# runs LinearAlgebra's generic `mul!`.  On Julia 1.10 that method allocates
# two 10 KB tile buffers per call, whatever the operand sizes, and it reaches
# its kernel through a call native Enzyme reverse cannot resolve statically,
# so the derivative runs through `runtime_generic_augfwd` and allocates the
# tiles' shadows besides: one row of a 2-column random-effect block
# (`z[r:r, :] * transpose(tau .* L)`, read from RKPPL's compact-cell overlay)
# cost 88 KB and 48 µs per gradient (snag `compact-cell-gra-ebf0a114`).  The
# native body calls this companion instead (`_kernel_native_calls`).  For two
# operands built only from `Array`s (and `SubArray`, `ReshapedArray`,
# `Adjoint` and `Transpose` wrappers of them, or another type that opts in
# through `_native_mul_operand`), one of them not strided, with one-based axes
# and isbits `Real` element types, it returns Base's result: a fresh `Matrix`
# (or `Vector` for a vector right factor) of Base's element type, summed in the
# generic method's order over the shared dimension (each product added in turn
# to a zero start).  Every other operand list, including two strided operands
# (which Base may send to BLAS), row vectors, structured and static matrices,
# mismatched sizes and more than two factors, calls Base itself.
@inline _native_mul_operand(::Array) = true
@inline _native_mul_operand(
    A::Union{SubArray,Base.ReshapedArray,LinearAlgebra.Adjoint,LinearAlgebra.Transpose}) =
    _native_mul_operand(parent(A))
@inline _native_mul_operand(_) = false

@inline _native_mul_strided(::StridedVecOrMat) = true
@inline _native_mul_strided(::LinearAlgebra.AdjOrTrans{<:Any,<:StridedVecOrMat}) = true
@inline _native_mul_strided(_) = false

# Whether `_native_mul` computes `A * B` with its own loop (the operands
# above), rather than leaving it to Base. The non-allocating program's product
# step (`_MatMulStep`) asks the same question for its cached destination.
@inline _native_mul_loops(A, B) = false
@inline function _native_mul_loops(A::AbstractMatrix{TA},
                                   B::AbstractVecOrMat{TB}) where {TA<:Real,TB<:Real}
    isbitstype(TA) && isbitstype(TB) || return false
    A isa LinearAlgebra.AdjOrTransAbsVec && return false
    B isa LinearAlgebra.AdjOrTransAbsVec && return false
    _native_mul_operand(A) && _native_mul_operand(B) || return false
    _native_mul_strided(A) && _native_mul_strided(B) && return false
    Base.has_offset_axes(A, B) && return false
    size(A, 2) == size(B, 1) || return false
    isbitstype(Base.promote_op(LinearAlgebra.matprod, TA, TB))
end

# Adds `A * B` into `C` in the generic method's order over the shared
# dimension, each product added in turn; `C` holds zeros on entry.
@inline function _native_mul_accumulate!(C, A, B)
    m, n = size(A)
    @inbounds for j in 1:size(B, 2), k in 1:n
        b = B[k, j]
        for i in 1:m
            C[i, j] += A[i, k] * b
        end
    end
    C
end

@inline _native_mul(args...) = *(args...)
@inline function _native_mul(A::AbstractMatrix{TA},
                             B::AbstractVecOrMat{TB}) where {TA<:Real,TB<:Real}
    _native_mul_loops(A, B) || return A * B
    T = Base.promote_op(LinearAlgebra.matprod, TA, TB)
    m = size(A, 1)
    C = B isa AbstractVector ? zeros(T, m) : zeros(T, m, size(B, 2))
    _native_mul_accumulate!(C, A, B)
end

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
# reproduces both with Enzyme only. On Julia 1.12 and later, registered Enzyme
# also fails the unit-range loop there: it sizes the skipped loop's reverse
# cache with a negative count (EnzymeAD/Enzyme.jl#3787,
# `benchmark/repro_enzyme_skipped_inner_loop_count.jl`). That is Enzyme's to
# repair; the indices stay as they are.
@inline _scan_rest(indices::AbstractUnitRange{<:Integer}) =
    (first(indices) + oneunit(eltype(indices))):last(indices)
@inline _scan_rest(indices) = Iterators.drop(indices, 1)

# The sequential-scan primitive `scan(xs...; init) do carry, x… end` (shared values
# captured by the step, or `Ref(shared)` operands)
# lowers to this.  `step` is the prepared 2-`want` step kernel
# `(carry, x..., shared...) -> (new_carry, output)`; the scan threads `carry`
# (seeded by `init`) over the `iterated` sequences in lockstep, one element of
# each per step, and collects the per-step outputs in an ordinary native loop.
# `iterated` and `shared` are tuples; the common
# case is a one-tuple `iterated`, and `eachindex(iterated...)` validates that
# several sequences share axes (a `DimensionMismatch` otherwise). Empty
# sequences run no step and yield an empty result. `Val(true)` is the authored
# `include_init = true`: the result is `[init, output…]` in one buffer of
# element type `promote_type(typeof(init), output type)`, exactly the value of
# `vcat([init], outputs)`, and an empty sequence yields `[init]`.
#
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
@inline function _scan_loop(step, init, iterated::Tuple, shared::Tuple,
                            ::Val{false} = Val(false))
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

@inline function _scan_loop(step, init, iterated::Tuple, shared::Tuple, ::Val{true})
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
#
# `history = leading` (a vector) puts the given entries first: the result is
# `[leading; outputs]`, step `j` writes entry `length(leading) + j` and reads
# the leading entries and earlier outputs below it, and entries not written
# yet read `zero` of the leading entries' element type. A recurrence seeded
# with `a[1:m]` then reads its whole trajectory from one vector.
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
function _scan_history_buffer(xs, leading::AbstractVector)
    T = eltype(leading)
    buffer = similar(xs, T, length(leading) + length(xs))
    fill!(buffer, zero(T))
    copyto!(buffer, firstindex(buffer), leading, firstindex(leading), length(leading))
end
_scan_history_buffer(xs, columns::NamedTuple) =
    map(column -> _scan_history_buffer(xs, column), columns)

# Where step outputs start in a history buffer.
_scan_history_offset(::Number) = 0
_scan_history_offset(leading::AbstractVector) = length(leading)
_scan_history_offset(columns::NamedTuple) = map(_scan_history_offset, columns)

# The element type of a history scan's result from its history value's type.
_scan_history_element(::Type{T}) where {T} = T
_scan_history_element(::Type{V}) where {V<:AbstractVector} = eltype(V)
_scan_history_result_type(::Type{T}) where {T} = Vector{_scan_history_element(T)}
function _scan_history_result_type(::Type{NamedTuple{names,types}}) where {names,types}
    NamedTuple{names,Tuple{map(_scan_history_result_type, fieldtypes(types))...}}
end

_scan_history_views(buffer::AbstractVector) = _ScanHistory(buffer)
_scan_history_views(columns::NamedTuple) = map(_scan_history_views, columns)
@inline function _scan_history_record!(buffer::AbstractVector, offset, index, output)
    buffer[offset + index] = output
    nothing
end
@inline function _scan_history_record!(buffers::NamedTuple{names}, offsets::NamedTuple{names},
                                       index, outputs::NamedTuple{names}) where {names}
    map(buffers, offsets, outputs) do buffer, offset, output
        _scan_history_record!(buffer, offset, index, output)
    end
    nothing
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

# The intermediates of a plate cell in a position residual keep lane slots
# across cells (`_plate_nested_native!`), but the cell's result is stored by
# reference in the plate's output. A result that shares memory with a slot's
# buffer (an opaque function returned the buffer or a view of it, a container
# holds it) takes the buffer with it: the slot is emptied, so a later cell
# allocates afresh instead of overwriting a stored result.
@inline function _lane_release!(result, slots::Vararg{Base.RefValue{Any},N}) where {N}
    isbits(result) && return result
    foreach(slots) do slot
        buffer = slot[]
        buffer === nothing || !_lane_may_reference(result, buffer) || (slot[] = nothing)
    end
    result
end

# The lane slots of an RK-owned callee called from a position residual
# (`_lane_call`): one state per call site, kept in that site's slot. A value
# that may hold one of these buffers releases the whole state, so the next
# call starts a fresh one (`_lane_may_reference`).
struct _LaneCalleeState
    slots::Vector{Base.RefValue{Any}}
end
@inline function _lane_callee_state!(slot::Base.RefValue{Any}, n::Int)
    state = slot[]
    state isa _LaneCalleeState && length(state.slots) == n && return state
    fresh = _LaneCalleeState(Base.RefValue{Any}[Base.RefValue{Any}(nothing) for _ in 1:n])
    slot[] = fresh
    fresh
end
function _lane_state_may_reference(result, state::_LaneCalleeState)
    for slot in state.slots
        held = slot[]
        held === nothing || !_lane_may_reference(result, held) || return true
    end
    false
end

# A plate of the position residual that stores its cells keeps one
# table per lane slot of a value computed at every cell (`_CELL_TABLES`): entry
# `k` holds the buffer cell `k`'s stored result held when the plate last ran,
# or a spare. A plate at the residual's top level runs once per position, and
# its stored results are dead once the next position (or, for a borrowed
# reader, the next call) runs it again: a WANT is stored by value, as numbers
# (`_replicated_output` refuses arrays of arrays). A plate inside another
# plate's cell runs once per outer cell, and its stored results are dead once
# the next outer cell runs it again, except those the outer cell's result
# holds, which leave the table (`_lane_tables_release!`). So cell `k` takes its
# entry into the slot before computing (`_lane_cell_take!`). A result that
# holds the slot's buffer gives
# it to the entry and the slot gets the entry's spare (`_lane_cell_release!`);
# every other cell keeps reusing the slot. Only the slot is ever written, and a
# buffer leaves the table only at its own cell, so no stored result is
# overwritten while it is live. A plate whose cells' element type `T` holds no
# array (numbers, tuples of them) keeps no table: its results never hold a
# buffer, and a table lookup per cell would only cost time. `every` says
# whether the slot's value is computed at every cell, a constant of the
# compiled body (`every_cell` in `_lower_authored_plate_loop_native!`); if
# not, one computed value can serve several cells, so the slot is released
# instead, as `_lane_release!` releases it. The take receives
# `every && _replicated_may_contain_array(T)`, decided before the plate's cells.
@inline function _lane_cell_table(table::Base.RefValue{Any}, cell::Int)
    entries = table[]
    entries isa Vector{Any} || (entries = Any[]; table[] = entries)
    entries = entries::Vector{Any}
    while length(entries) < cell
        push!(entries, nothing)
    end
    entries
end
@inline function _lane_cell_take!(slot::Base.RefValue{Any}, table::Base.RefValue{Any},
                                  cell::Int, takes::Bool)
    takes || return nothing
    entries = _lane_cell_table(table, cell)
    held = @inbounds entries[cell]
    held === nothing && return nothing
    @inbounds entries[cell] = slot[]
    slot[] = held
    nothing
end
@inline function _lane_cell_release!(result, cell::Int,
        slots::Vararg{Tuple{Base.RefValue{Any},Base.RefValue{Any},Bool},N}) where {N}
    isbits(result) && return result
    foreach(slots) do (slot, table, every)
        buffer = slot[]
        (buffer === nothing || !_lane_may_reference(result, buffer)) && return
        every || return (slot[] = nothing)
        entries = _lane_cell_table(table, cell)
        @inbounds slot[] = entries[cell]
        @inbounds entries[cell] = buffer
    end
    result
end

# The outer cell's result drops every entry of the tables of the plates inside
# that cell (`_plate_nested_native!`) it may hold: the next outer cell then
# allocates afresh for that inner cell instead of overwriting a live value.
@inline function _lane_tables_release!(result,
        tables::Vararg{Base.RefValue{Any},N}) where {N}
    isbits(result) && return result
    foreach(tables) do table
        entries = table[]
        entries isa Vector{Any} || return
        for cell in eachindex(entries::Vector{Any})
            held = @inbounds entries[cell]
            held === nothing || !_lane_may_reference(result, held) ||
                (@inbounds entries[cell] = nothing)
        end
    end
    result
end

# Any other value may hold an array.
# The walk runs once per cell. It specializes on the result only: a slot's
# buffer is untyped (`Ref{Any}`), so a method specialized on it would be found
# at run time and box the result, a tuple included. Tuples recurse by type and
# arrays loop, so the walk allocates nothing (`any` with a closure over a
# heterogeneous tuple boxed every item).
_lane_may_reference(result, @nospecialize(buffer)) = true
_lane_may_reference(::Number, @nospecialize(buffer)) = false
function _lane_may_reference(result::AbstractArray, @nospecialize(buffer))
    buffer isa _LaneCalleeState && return _lane_state_may_reference(result, buffer)
    _lane_mightalias(result, buffer) && return true
    _replicated_may_contain_array(eltype(result)) || return false
    for item in result
        _lane_may_reference(item, buffer) && return true
    end
    false
end
_lane_may_reference(result::NamedTuple, @nospecialize(buffer)) =
    _lane_may_reference(values(result), buffer)
_lane_may_reference(::Tuple{}, @nospecialize(buffer)) = false
@inline _lane_may_reference(result::Tuple, @nospecialize(buffer)) =
    _lane_may_reference(first(result), buffer) ||
    _lane_may_reference(Base.tail(result), buffer)
# `Base.mightalias` on the untyped buffer was found at run time on every check
# (about 110 ns, 8 % of a borrowed population read, snag
# lane-cell-take-e-45c66e49). A slot holds a dense `Array` (`_lane_reuse`),
# usually of the result's own type: those checks are compiled in.
@inline _lane_mightalias(result::A, @nospecialize(buffer)) where {A<:AbstractArray} =
    buffer isa A ? Base.mightalias(result, buffer::A) :
    buffer isa Array ? Base.mightalias(result, buffer::Array) :
    buffer isa AbstractArray && Base.mightalias(result, buffer)

# The buffer an authored plate or scan fills (`_lane_allocation`): the matching
# recycled buffer, or `allocate(args...)`, kept when `recycled` is a lane slot.
# - Not inlined, with one fixed-arity method per allocation arity. Inline, the
#   reused and the fresh buffer merge into a phi of derived data pointers in
#   the filling loop, which reverse-mode Enzyme cannot keep rooted for the
#   garbage collector; a non-inlined varargs method boxes its arguments on
#   Julia 1.10; a tuple of them stores an active array, which Enzyme rejects.
# - A lane slot of a differentiated non-allocating program (`_ad_na_program`)
#   is a `Ref{Array{T,N}}` seeded with an owned copy of the exemplar call's
#   buffer (`_ad_na_concretize_cache`), and is only read: one inline load,
#   no store, no merge. A call that may store into the slot made Enzyme cache
#   loads conservatively (about 1 µs per gradient on small models), and a
#   merged buffer lost its derivative in scans. That program keeps the
#   exemplar's shapes and element types; another one fails loudly.
const _LANE_ACQUIRE_ARITIES = 1:4
for arity in _LANE_ACQUIRE_ARITIES
    args = [Symbol(:allocation_argument_, i) for i in 1:arity]
    @eval @noinline function _lane_acquire(recycled, ::Type{T}, output_axes,
                                           allocate::F, $(args...)) where {T,F}
        buffer = _lane_reuse(recycled, T, output_axes)
        buffer === nothing ? _lane_keep!(recycled, allocate($(args...))) : buffer
    end
    @eval @inline function _lane_acquire(slot::Base.RefValue{B}, ::Type{T},
            output_axes::NTuple{N,Any}, allocate::F,
            $(args...)) where {B<:Array,T,N,F}
        buffer = slot[]
        buffer isa Array{T,N} && axes(buffer) == output_axes ||
            _lane_changed(buffer, T, output_axes)
        buffer::Array{T,N}
    end
    # No buffer to offer (the position driver's `nothing`, or a lane the
    # exemplar call never filled): allocate inline, as the ordinary program.
    @eval @inline _lane_acquire(::Nothing, ::Type{T}, output_axes, allocate::F,
                                $(args...)) where {T,F} = allocate($(args...))
end

@noinline _lane_changed(buffer, ::Type{T}, output_axes) where {T} =
    throw(ArgumentError(
        "a non-allocating gradient fills each materialized value into the " *
        "buffer it prepared from the exemplar call, a $(typeof(buffer)) with " *
        "axes $(axes(buffer)); this call's value is an array of $T with axes " *
        "$output_axes. Prepare the gradient again for the new shape"))

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

# `reduce(vcat, A)` (`_lane_source_expr`). For a vector of dense vectors,
# Base's method allocates `similar(A[1], T, n)`, a `Vector{T}` of the promoted
# element type and the total length, and copies the cells in order
# (`Base._typed_vcat!`); a matching recycled buffer is filled the same way.
# Any other argument, no cells (whose Base call throws) and a cell that may
# share memory with the buffer take Base's call.
@inline _lane_reduce_vcat(recycled, A) = reduce(vcat, A)
@inline function _lane_reduce_vcat(recycled,
        A::AbstractVector{<:Union{Array{<:Any,1},SubArray{<:Any,1,<:Array}}})
    isempty(A) && return reduce(vcat, A)
    T = mapreduce(eltype, promote_type, A)
    n = 0
    for cell in A
        n += length(cell)
    end
    buffer = _lane_reuse(recycled, T, (Base.OneTo(n),))
    buffer === nothing && return _lane_keep!(recycled, reduce(vcat, A))
    for cell in A
        Base.mightalias(cell, buffer) && return _lane_keep!(recycled, reduce(vcat, A))
    end
    Base._typed_vcat!(buffer, A)
end

# `zeros`, `ones` and `fill` of integer dimensions in a lazy-branch arm
# (`_lane_source_expr`): Base's fresh array, or a matching recycled buffer
# filled with the same element. Other arguments (dimension tuples, a negative
# dimension's error) take Base's call.
@inline _lane_zeros(recycled, args...) = zeros(args...)
@inline _lane_zeros(recycled, dims::Vararg{Integer}) = _lane_zeros(recycled, Float64, dims...)
@inline _lane_zeros(recycled, ::Type{T}, dims::Vararg{Integer}) where {T} =
    _lane_filled(recycled, zeros, zero(T), T, dims)
@inline _lane_ones(recycled, args...) = ones(args...)
@inline _lane_ones(recycled, dims::Vararg{Integer}) = _lane_ones(recycled, Float64, dims...)
@inline _lane_ones(recycled, ::Type{T}, dims::Vararg{Integer}) where {T} =
    _lane_filled(recycled, ones, one(T), T, dims)
@inline _lane_fill(recycled, args...) = fill(args...)
@inline _lane_fill(recycled, value::Number, dims::Vararg{Integer}) =
    _lane_filled(recycled, fill, value, typeof(value), dims)
@inline function _lane_filled(recycled, make::F, value, ::Type{T}, dims) where {F,T}
    fresh() = make === fill ? fill(value, dims...) : make(T, dims...)
    all(d -> d >= 0, dims) || return fresh()
    buffer = _lane_reuse(recycled, T, map(d -> Base.OneTo(Int(d)), dims))
    buffer === nothing && return _lane_keep!(recycled, fresh())
    fill!(buffer, value)
end

_scan_history_buffer(recycled, xs, fill::Number) = fill!(
    _lane_acquire(recycled, typeof(fill), axes(xs), _scan_history_buffer, xs, fill),
    fill)
function _scan_history_buffer(recycled, xs, leading::AbstractVector)
    T = eltype(leading)
    buffer = _lane_acquire(recycled, T,
        (Base.OneTo(length(leading) + length(xs)),), _scan_history_buffer, xs,
        leading)
    fill!(buffer, zero(T))
    copyto!(buffer, firstindex(buffer), leading, firstindex(leading), length(leading))
end
_scan_history_buffer(recycled, xs, fill) = _scan_history_buffer(xs, fill)
_scan_history_buffer(xs, fill) = throw(ArgumentError(
    "a scan's `history =` value must be a number (it fills the outputs not yet " *
    "written and fixes their element type), a vector of leading entries, " *
    "or a named tuple of these column specifications; " *
    "got a $(typeof(fill))"))

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

function _scan_history_loop(step, init, fill, iterated::Tuple, shared::Tuple)
    idx = eachindex(iterated...)
    result = _scan_history_buffer(first(iterated), fill)
    offset = _scan_history_offset(fill)
    earlier = _scan_history_views(result)
    isempty(idx) && return result
    i1 = first(idx)
    carry, out1 = step(init, map(xs -> xs[i1], iterated)..., shared..., earlier)
    carry = _scan_history_carry(carry)
    _scan_history_record!(result, offset, i1, out1)
    for i in _scan_rest(idx)
        carry, out = step(carry, map(xs -> xs[i], iterated)..., shared..., earlier)
        carry = _scan_history_carry(carry)
        _scan_history_record!(result, offset, i, out)
    end
    result
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

In a plate cell, a leaf arm may also carry its own graph, `then_spec` /
`else_spec`: a `KernelSpec` over the arm's ports (by name) whose one output is
the arm's value, authored as a cell body would be (`nothing` otherwise). The
native plate lowering reads them when the condition reads only shared
operands: the condition is decided once per call and the cells run the taken
arm's graph, whose shared recipes are hoisted above the cell loop like any
other cell invariant (`_plate_shared_branch`).
"""
struct _KernelBranch{CI,TI,EI,F,C,T,E}
    call::F
    condition::C
    then_arm::T
    else_arm::E
    # Cold lowering metadata, untyped: prepared operation tables carry this
    # branch, and a graph in its type would be compiled with every program.
    then_spec::Any
    else_spec::Any
end
_KernelBranch(ci::Val, ti::Val, ei::Val, call, condition, then_arm, else_arm) =
    _KernelBranch(ci, ti, ei, call, condition, then_arm, else_arm, nothing, nothing)
# Built once per authored branch, so assemble it without compiling a
# constructor per closure type (as `_KernelSourceOp` does).
Base.@nospecializeinfer function _KernelBranch(
        @nospecialize(ci::Val), @nospecialize(ti::Val), @nospecialize(ei::Val),
        @nospecialize(call), @nospecialize(condition), @nospecialize(then_arm),
        @nospecialize(else_arm), @nospecialize(then_spec), @nospecialize(else_spec))
    T = _KernelBranch{_val_parameter(ci), _val_parameter(ti), _val_parameter(ei),
                      typeof(call), typeof(condition), typeof(then_arm),
                      typeof(else_arm)}
    _kernel_new_instance(T, (call, condition, then_arm, else_arm,
                             then_spec, else_spec))::_KernelBranch
end
# Forwarded positionally and inlined (`Vararg{Any,N}` specializes on the
# count; a merely forwarded `args...` is left unspecialized and boxes the
# arguments as one tuple): the table holds the branch itself, which the
# generated body calls directly.
@inline @generated function (branch::_KernelBranch)(args::Vararg{Any,N}) where {N}
    forwarded = [:(getfield(args, $index)) for index in 1:N]
    :(Base.@inline getfield(branch, :call)($(forwarded...)))
end

"""
    _KernelSplit{S}(call, wholes, splits, root)

An authored expression (a fused recipe's right-hand side, or one leaf arm of a
`_KernelBranch`) whose strict subexpressions read fewer of its arguments than
their enclosing expression keeps those subexpressions as metadata beside its
ordinary body. Under some `bound=` binding each could read only bound data
while the expression around it does not. Calling it runs `call`, the authored
expression over every argument, so every lowering that treats the enclosing
`_KernelSourceOp` as opaque is unchanged.

`S` is `(ports, root, nodes)`: the expression's argument names in order, the
root's `(direct, kids)` and, for each split node in post-order,
`(own, direct, kids)`. `own` are the argument positions the node's whole
subexpression reads, `direct` those it reads outside its nested split nodes,
and `kids` the indices of those nested nodes. `wholes[i]` evaluates node `i`
from its `own` arguments. `splits[i]` (`nothing` for a node without kids)
evaluates it from its `direct` arguments followed by its kids' values, and
`root` evaluates the whole expression from the root's `direct` arguments
followed by the root kids' values. Plate partial evaluation reads them
(`_partial_split_recipes`): a node whose arguments are all bound data becomes
its own recipe there, so preparation evaluates it once per lane.
"""
struct _KernelSplit{S,F,W,P,R}
    call::F
    wholes::W
    splits::P
    root::R
end
# Built once per authored recipe, of a new concrete type each time: construct
# it without compiling its constructor per type, as `_KernelSourceOp` does.
Base.@nospecializeinfer function _KernelSplit(@nospecialize(spec::Val), @nospecialize(call),
                                              @nospecialize(wholes::Tuple),
                                              @nospecialize(splits::Tuple),
                                              @nospecialize(root))
    T = _KernelSplit{_val_parameter(spec), typeof(call), typeof(wholes),
                     typeof(splits), typeof(root)}
    _kernel_new_instance(T, (call, wholes, splits, root))::_KernelSplit
end
_kernel_split_spec(::_KernelSplit{S}) where {S} = S
# Source readers of the whole expression (destination forms, nonallocating
# decomposition) read the authored closure the split metadata wraps.
Base.@nospecializeinfer _source_record(@nospecialize(split::_KernelSplit)) =
    _source_record(getfield(split, :call))
# Forwarded positionally and inlined, as a source record's call forwards.
@inline @generated function (split::_KernelSplit)(args::Vararg{Any,N}) where {N}
    forwarded = [:(getfield(args, $index)) for index in 1:N]
    :(Base.@inline split.call($(forwarded...)))
end

"""
    _KernelSplitApply{P}(split)

A split expression (`_KernelSplit`) evaluated with some of its nodes supplied
as values: its arguments are the expression's own, followed by one value per
supplied node. `P` holds the root's kids in order, each `(:value, j)` (the
`j`-th supplied value), `(:whole, k)` (node `k` from its own arguments) or
`(:split, k, kids)` (node `k` from its direct arguments and its kids, given
the same way). Everything else is computed exactly where the authored
expression computes it, so a lazy arm holding the expression still runs only
when taken.
"""
struct _KernelSplitApply{P,S}
    split::S
end
_KernelSplitApply(::Val{P}, split::S) where {P,S} = _KernelSplitApply{P,S}(split)
@inline @generated function (apply::_KernelSplitApply{P,S})(args::Vararg{Any,N}) where {P,S,N}
    ports, root, nodes = S.parameters[1]
    count = length(ports)
    arg(i) = :(getfield(args, $i))
    split = :(getfield(apply, :split))
    function node(plan)
        plan[1] === :value && return arg(count + plan[2])
        k = plan[2]
        plan[1] === :whole && return :(Base.@inline getfield(getfield($split, :wholes), $k)(
            $(map(arg, nodes[k][1])...)))
        :(Base.@inline getfield(getfield($split, :splits), $k)(
            $(map(arg, nodes[k][2])...), $(map(node, plan[3])...)))
    end
    :(Base.@inline getfield($split, :root)($(map(arg, root[1])...), $(map(node, P)...)))
end
Base.@nospecializeinfer _source_record(@nospecialize(apply::_KernelSplitApply)) =
    _source_record(getfield(apply, :split))

# One arm of a `_KernelBranchCall`: its operation over the arguments at `I`.
struct _KernelArm{I,F}
    f::F
end
@inline @generated function (arm::_KernelArm{I})(args::Vararg{Any,N}) where {I,N}
    :(Base.@inline getfield(arm, :f)($((:(getfield(args, $i)) for i in I)...)))
end

"""
    _KernelBranchCall{CI,TI,EI,Names}(condition, then_arm, else_arm)

The lazy branch of a `_KernelBranch` whose arms preparation rebuilt: the
condition over the arguments at `CI`, then exactly one arm through
`_recurrence_branch` (the native ternary). `Names` are its arguments' names in the recipe's readable source.
"""
struct _KernelBranchCall{CI,TI,EI,Names,C,T,E}
    condition::C
    then_arm::T
    else_arm::E
end
_KernelBranchCall(::Val{CI}, ::Val{TI}, ::Val{EI}, ::Val{Names}, condition::C,
                  then_arm::T, else_arm::E) where {CI,TI,EI,Names,C,T,E} =
    _KernelBranchCall{CI,TI,EI,Names,C,T,E}(condition, then_arm, else_arm)
# The body of a `_KernelBranchCall` over the arguments at `P`. A native `Bool`
# condition takes one arm with the ternary `_recurrence_branch` would apply,
# and an arm that is itself a rebuilt branch is expanded in place rather than
# entered through this generated method again: Julia 1.10's inference widens a
# recursive call of one method with a different signature, so nested rebuilt
# branches would otherwise make the whole cell infer `Any` (and native Reverse
# then meets a dynamic call). Any other condition keeps `_recurrence_branch`.
function _branch_call_body(b, @nospecialize(B), P, path)
    CI, TI, EI = B.parameters[1], B.parameters[2], B.parameters[3]
    T, E = fieldtype(B, :then_arm), fieldtype(B, :else_arm)
    c = Symbol("##branch_condition#", path)
    read(positions) = Any[:(getfield(args, $(P[i]))) for i in positions]
    then_arm, else_arm = :(getfield($b, :then_arm)), :(getfield($b, :else_arm))
    arms = :(_recurrence_branch($c, _KernelArm{$TI,$T}($then_arm),
        _KernelArm{$EI,$E}($else_arm), ($(read(eachindex(P))...),)))
    quote
        $c = getfield($b, :condition)($(read(CI)...))
        $c isa Bool ? ($c ? $(_branch_arm_body(then_arm, T, Tuple(P[i] for i in TI), path * "t")) :
                            $(_branch_arm_body(else_arm, E, Tuple(P[i] for i in EI), path * "e"))) :
            $arms
    end
end
function _branch_arm_body(arm, @nospecialize(A), P, path)
    if A <: _KernelBranch && fieldtype(A, :call) <: _KernelBranchCall
        return _branch_call_body(:(getfield($arm, :call)), fieldtype(A, :call), P, path)
    end
    :(Base.@inline $arm($((:(getfield(args, $i)) for i in P)...)))
end
@inline @generated function (b::_KernelBranchCall{CI,TI,EI})(args::Vararg{Any,N}) where {CI,TI,EI,N}
    _branch_call_body(:b, b, Tuple(1:N), "")
end

"""
    _KernelReduction{II,XI,KI,AI}(call, iterator, init, step, index, array,
                                   step_in, step_out)

A recipe whose authored right-hand side is a top-level unfiltered generator
sum with `init`, `sum(term for j in iterator; init = x)`, whose term either
reads one shared vector through Base's total gather (`get(A, K, D)`) or is
plain arithmetic (`_kernel_fold_arithmetic`), keeps its parts as metadata
beside its ordinary body. Calling it runs `call`, the authored expression over
every recipe argument, so every lowering that treats the enclosing
`_KernelSourceOp` as opaque is unchanged.

The native plate lowering reads the parts (`_lower_authored_plate_native!`):
when the iterator (and `A`) are plate invariants it runs the sum fold-outer —
one pass over the cells per element `j`, each cell accumulating
`Base.add_sum(acc, term)` in the authored order. For a gathered sum
(dose-outer) whose gather index advances by exactly one per cell, each pass
splits at the window where `K` is in range, reading `A` there without the
bounds test and using `D` outside it.
`iterator`/`init`/`array` are closures over their OWN ports, selected from
the recipe's ordered arguments by the position tuples `II`/`XI`/`AI`;
`index` takes `j` and the ports `KI`; the steps take the accumulator, `j`
(and `step_in` the gathered value) followed by every recipe argument. A
gather-free sum has empty `KI`/`AI` and `nothing` for `index`, `array`,
`step_in` and `step_out`.
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
_source_callable_type(::Type{<:Union{_KernelSourceOp,_KernelSource,
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
        which(Tuple{_KernelSource,Vararg{Any}}),
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

# Whether a plate's fold-outer lowering of `reduction` keeps the authored
# semantics for these types: a concrete accumulator type `T` that the seed and
# every step return unchanged and, for a gathered sum, an `Int` gather index
# and a `Vector` gather source (or a contiguous column of a dense matrix). `S`
# is the tuple of the recipe's per-cell argument types. Everything here is a
# type computation, folded when the plate body is compiled.
@generated function _plate_reduction_ready(
        reduction::_KernelReduction{II,XI,KI,AI,F,IT,IN,ST,IX}, ::Type{T},
        ::Type{S}) where {II,XI,KI,AI,F,IT,IN,ST,IX,T,S<:Tuple}
    types = Any[fieldtype(S, i) for i in 1:fieldcount(S)]
    select(positions) = Any[types[i] for i in positions]
    promote = GlobalRef(Base, :promote_op)
    fold = quote
        isconcretetype($T) || return false
        iterator_type = $promote(reduction.iterator, $(select(II)...))
        element = eltype(iterator_type)
        isconcretetype(element) || return false
        $promote(reduction.init, $(select(XI)...)) === $T || return false
        $promote(reduction.step, $T, element, $(types...)) === $T || return false
    end
    IX === Nothing && return quote
        $fold
        true
    end
    quote
        $fold
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
# an `AbstractVector` (a tuple keeps Base's unrolled method), a one-axis
# domain, and an `x` that is not `Complex` (Base evaluates a complex `x` with
# a different recurrence, `_evalpoly(z::Complex, p)`). Folded when the plate
# body is compiled.
@generated function _plate_evalpoly_ready(::Type{T}, ::Type{X}, ::Type{C},
                                          output_axes) where {T,X,C}
    output_axes <: Tuple{Any} && C <: AbstractVector && !(X <: Complex) ||
        return false
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

# Cells per tile of a fold-outer pass: the accumulators and the cell values of
# one tile stay in the first-level cache across the passes.
const _PLATE_FOLD_TILE = 256
# Whether a plate's output has one axis, which a fold-outer tile walks directly.
@inline _plate_one_axis(::Tuple{Any}) = true
@inline _plate_one_axis(output_axes) = false

# Cells per chunk of a coefficient-outer `evalpoly` cell: independent Horner
# chains whose accumulators and `x` values stay in vector registers across the
# coefficients, enough of them to cover the multiply-add latency.
const _PLATE_HORNER_CHUNK = 32

# `dest[i] = evalpoly(x[i], c)` for every `i` in `cells`, as Base computes it
# for a vector `c`: the seed `c[end]`, then `muladd(x[i], acc, c[j])` for
# `j = length(c)-1:-1:1`. A chunk of `_PLATE_HORNER_CHUNK` consecutive cells
# runs its chains side by side, its accumulators and `x` values in registers
# across the coefficients; the cells after the last full chunk run one at a
# time. The callers guarantee a one-based `c` and `x` and `dest` indexable at
# every cell; an empty `c` raises Base's `BoundsError` at the seed, as Base's
# `p[end]` does. Its own function: inlined into a large kernel body, the
# chunk's consecutive reads lose their adjacency to LLVM and its registers to
# the rest of the body.
@noinline function _plate_evalpoly_chunks!(dest, x, c, cells::AbstractUnitRange{Int})
    seed = c[lastindex(c)]
    degree = length(c) - 1
    cell = first(cells)
    stop = last(cells)
    while cell <= stop - (_PLATE_HORNER_CHUNK - 1)
        _plate_evalpoly_chunk!(dest, x, c, cell, seed, degree, Val(_PLATE_HORNER_CHUNK))
        cell += _PLATE_HORNER_CHUNK
    end
    while cell <= stop
        xi = @inbounds x[cell]
        acc = seed
        for j in degree:-1:1
            acc = muladd(xi, acc, @inbounds c[j])
        end
        @inbounds dest[cell] = acc
        cell += 1
    end
    dest
end
@generated function _plate_evalpoly_chunk!(dest, x, c, start::Int, seed, degree::Int,
                                           ::Val{W}) where {W}
    xs = [Symbol(:x_, k) for k in 1:W]
    accs = [Symbol(:acc_, k) for k in 1:W]
    quote
        $((:($(xs[k]) = @inbounds x[start + $(k - 1)]) for k in 1:W)...)
        $((:($(accs[k]) = seed) for k in 1:W)...)
        for j in degree:-1:1
            cj = @inbounds c[j]
            $((:($(accs[k]) = muladd($(xs[k]), $(accs[k]), cj)) for k in 1:W)...)
        end
        $((:(@inbounds dest[start + $(k - 1)] = $(accs[k])) for k in 1:W)...)
        nothing
    end
end

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
# The `@simd` iteration space of a pure total-only plate (see
# `_lower_authored_plate_cells_native!`): one axis is its own indexable range,
# several are their `CartesianIndices`. The loop body rebuilds the cell index
# with `CartesianIndex`, which accepts either element.
@inline _plate_simd_cells(axes::Tuple{Any}) = only(axes)
@inline _plate_simd_cells(axes::Tuple) = CartesianIndices(axes)

# The cell positions whose gather index `first_index + position - 1` lies in
# `source`'s range, as `lo:hi` within `1:n` (empty when none do).
@inline function _plate_gather_window(source::AbstractVector, first_index::Int,
                                      n::Int)
    lo = clamp(widen(firstindex(source)) - first_index + 1, 1, n + 1)
    hi = clamp(widen(lastindex(source)) - first_index + 1, lo - 1, n)
    (Int(lo), Int(hi))
end

"""
    Graph()

A mutable builder collecting `Value`s and `Recipe`s plus the producer index the
planner needs. Building executes nothing.
"""
mutable struct Graph
    values::Dict{ValueId,Value}          # id => Value
    recipes::Vector{Recipe}
    producers::Dict{ValueId,Vector{Int}} # canonical value id => indices into `recipes`
    aliases::Dict{ValueId,ValueId}       # value id => structurally-equal canonical id
    version::Int
    # The value-independent work of this graph's bound preparations, reused by
    # `prepare(…; bound)` (`_graph_preparations`, graphops.jl); `nothing` until
    # the first one. Not part of the graph's meaning.
    preparations::Any
end
Graph(values, recipes, producers, aliases, version) =
    Graph(values, recipes, producers, aliases, version, nothing)
Graph() = Graph(Dict{ValueId,Value}(), Recipe[], Dict{ValueId,Vector{Int}}(),
                Dict{ValueId,ValueId}(), 0)

_register!(g::Graph, v::Value) = (g.values[v.id] = v; v)

"""
    canon_id(g, id) -> ValueId

Resolve a value id to its structural-CSE canonical representative (gist §8).
Absent any structural CSE this is the identity.
"""
canon_id(g::Graph, id::ValueId) =
    haskey(g.aliases, id) ? canon_id(g, g.aliases[id]) : id

# The canonical identity of `v` in `g` as one `Int` (`_canon_int(::ValueId)`).
_canon_int(g::Graph, v::Value) = _canon_int(canon_id(g, v.id))

"""
    value!(g, name, T) -> Value{T}

Create a `Value{T}` and register it into graph `g`.
"""
function value!(g::Graph, name::Symbol, ::Type{T}) where {T}
    v = Value{T}(_next_value_id(), name)
    _register!(g, v)
    g.version += 1
    v
end

_astuple(v::Value) = (v,)
_astuple(t::Tuple) = t
_astuple(v::AbstractVector) = Tuple(v)

# A HAVE/WANT selection as one `Vector{Value}`, and a non-empty binding as one
# vector of `Value => data` pairs. A tuple of `Value`s or of pairs has a type of
# its own for every selection, so preparation code specialized on it (a keyword
# method's body, a closure that captures it) compiled again for every program,
# in time superlinear in the width: 27 of the 30 s of preparing a kernel with
# 655 bound ports (snag stack-overflow-i-549143aa). Anything else (one value or
# pair, an entry of another kind) is returned as it is, for the callee to take or
# refuse as before.
Base.@nospecializeinfer function _value_list(@nospecialize(vs))
    vs isa Tuple || return vs
    list = Value[]
    for i in 1:nfields(vs)
        v = getfield(vs, i)
        v isa Value || return vs
        push!(list, v)
    end
    list
end

# The values `_astuple(vs)` iterates, and `collect(Value, _astuple(vs))`,
# without compiling for the type of a tuple `vs`.
Base.@nospecializeinfer function _value_iter(@nospecialize(vs))
    list = _value_list(vs)
    list isa AbstractVector ? list : _astuple(list)
end
Base.@nospecializeinfer _value_vector(@nospecialize(vs)) = collect(Value, _value_iter(vs))

Base.@nospecializeinfer function _bound_list(@nospecialize(bound))
    bound isa Tuple && nfields(bound) > 0 || return bound
    list = Pair{Value,Any}[]
    for i in 1:nfields(bound)
        entry = getfield(bound, i)
        entry isa Pair{<:Value} || return bound
        push!(list, entry)
    end
    list
end

function _cse_alias_plan(g::Graph, new_outputs::Tuple, old_outputs::Tuple, cse_key)
    targets = Dict{ValueId,ValueId}()
    edges = Pair{ValueId,ValueId}[]

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
        seen = Set{ValueId}()
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
        indexed = Set{ValueId}()
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
function _canonical_ids_equal(g::Graph, values::Tuple{Vararg{Value}}, ids::Vector{ValueId})
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
        canon_ins = ValueId[canon_id(g, v.id) for v in ins]
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
producers_of(g::Graph, vid::ValueId) = get(g.producers, canon_id(g, vid), Int[])
