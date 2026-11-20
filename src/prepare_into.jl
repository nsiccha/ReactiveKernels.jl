# The ordinary prepared program supplies the vector semantics, including any
# passes the caller already applied. Only its terminal materialization can be
# deferred directly to the destination. The buffered path below reads this
# same prepared body.
struct _IntoBroadcast{B}
    value::B
end

function _defer_final_broadcast(ex)
    ex isa Expr || return nothing
    if Meta.isexpr(ex, :call, 2) &&
       ex.args[1] == GlobalRef(@__MODULE__, :_native_broadcast_materialize)
        return Expr(:call, GlobalRef(@__MODULE__, :_IntoBroadcast), ex.args[2])
    elseif ex.head in (:let, :block) && !isempty(ex.args)
        tail = _defer_final_broadcast(last(ex.args))
        tail === nothing || return Expr(ex.head, ex.args[1:end-1]..., tail)
    end
    nothing
end

function _into_program(k::PreparedKernel)
    original = code_expr(k)
    Meta.isexpr(original, :function, 2) || return k
    signature, body = original.args
    Meta.isexpr(body, :block) && length(body.args) >= 2 || return k
    store, ret = body.args[end-1:end]
    Meta.isexpr(ret, :return, 1) && Meta.isexpr(store, :(=), 2) || return k
    name, rhs = store.args
    name isa Symbol && only(ret.args) === name || return k
    # A previous declaration can convert the RHS on assignment, and a closure
    # may see the binding. In either case keep the ordinary computation intact.
    (_mentions_symbol(signature, name) ||
     any(stmt -> _mentions_symbol(stmt, name), @view(body.args[1:end-2]))) && return k
    deferred = _defer_final_broadcast(rhs)
    deferred === nothing && return k
    ast = Expr(:function, signature, Expr(:block, body.args[1:end-2]..., Expr(:return, deferred)))
    _prepared_kernel(compile(ast), k.constants, k.callables, k.inputs, k.outputs,
                     k.plan, ast, k.lowered_recipes; exact_signature = _exact_signature(k))
end

mutable struct _IntoScratch{S,K}
    kernel::K
    @atomic busy::Bool
end

# An intermediate can be retained across calls only when it cannot escape to
# user code. Read a straight-line numerical region of the *executed* program,
# including its let bindings, then let the usual buffered lowering choose the
# destination steps. This never rebuilds the selected plan or reruns a pass.
const _IntoNumeric = Union{_NativeBroadcastNumber,Array{<:_NativeBroadcastNumber},BitArray}
const _INTO_NUMERIC_CALLS = (+, -, *, /, min, max, abs, abs2, sqrt, exp, log,
    copy, reverse, sort, sum, prod, minimum, maximum, cumsum, cumprod, length,
    size, zeros, ones, fill, vcat, hcat, _NONALLOC_LAZY_CALLEES..., _native_vcat, _native_mul,
    _native_broadcasted, _native_broadcast_materialize)

function _into_operand(ctx, ex, k)
    ex isa Symbol && return haskey(ctx.argmap, ex) ? ex : nothing
    ex isa _NativeBroadcastNumber && return ex
    if ex isa GlobalRef
        r = _nonalloc_resolve_const(ex.mod, ex.name)
        r === nothing && return nothing
        value = something(r)
        (value isa _IntoNumeric || any(f -> f === value, _INTO_NUMERIC_CALLS)) || return nothing
        return ex
    end
    ex isa Expr || return nothing
    j = _constant_slot(ex)
    if j !== nothing && 1 <= j <= length(k.constants)
        value = k.constants[j]
        value isa _IntoNumeric || return nothing
        name = gensym(:bound)
        ctx.argmap[name], ctx.argtypes[name] = ex, typeof(value)
        return name
    end
    Meta.isexpr(ex, :call) && !isempty(ex.args) || return nothing
    callee = first(ex.args)
    j = _callable_slot(callee)
    if j !== nothing && 1 <= j <= length(k.callables)
        f = k.callables[j]
        any(x -> x === f, _INTO_NUMERIC_CALLS) || return nothing
        callee = _const_call_ref(f)
    end
    callee isa GlobalRef || return nothing
    r = _nonalloc_resolve_const(callee.mod, callee.name)
    r !== nothing && any(f -> f === something(r), _INTO_NUMERIC_CALLS) || return nothing
    args = Any[]
    for arg in @view ex.args[2:end]
        operand = _into_operand(ctx, arg, k)
        operand === nothing && return nothing
        if operand isa Expr
            # Complete each argument before computing the next. Decomposition
            # can otherwise hoist a later buffer step ahead of an earlier error.
            d = _decompose(ctx, operand, false)
            d === nothing && return nothing
            name = gensym(:operand)
            ctx.argmap[name], ctx.argtypes[name] = _traced_value!(ctx, d)
            operand = name
        end
        push!(args, operand)
    end
    Expr(:call, callee, args...)
end

function _into_bind!(ctx, stmt, k, locals = nothing)
    Meta.isexpr(stmt, :(=), 2) && stmt.args[1] isa Symbol || return false
    name = stmt.args[1]
    name in (_CONSTANTS_ARG, _CALLABLES_ARG) && return false
    # The native emitter gives each let fresh names. A pass that shadows or
    # changes an outer binding keeps Julia's ordinary lexical evaluation.
    locals !== nothing && haskey(ctx.argmap, name) && return false
    d = _into_expression!(ctx, stmt.args[2], k, locals)
    d === nothing && return false
    ctx.argmap[name], ctx.argtypes[name] = _traced_value!(ctx, d)
    locals === nothing || push!(locals, name)
    true
end

function _into_expression!(ctx, ex, k, locals = nothing)
    if Meta.isexpr(ex, :let, 2)
        bindings, body = ex.args
        names = Symbol[]
        for stmt in (Meta.isexpr(bindings, :block) ? bindings.args : (bindings,))
            _into_bind!(ctx, stmt, k, names) || return nothing
        end
        d = _into_expression!(ctx, body, k, names)
        for name in names
            delete!(ctx.argmap, name)
            delete!(ctx.argtypes, name)
        end
        return d
    elseif Meta.isexpr(ex, :block) && !isempty(ex.args)
        for stmt in @view ex.args[1:end-1]
            stmt isa LineNumberNode || _into_bind!(ctx, stmt, k, locals) || return nothing
        end
        return _into_expression!(ctx, last(ex.args), k, locals)
    end
    operand = _into_operand(ctx, ex, k)
    operand === nothing ? nothing : _decompose(ctx, operand, false)
end

# A single directly writable broadcast needs no workspace or lease.
_into_direct_operand_type(T) = T <: _IntoNumeric || T <: Base.RefValue{<:_NativeBroadcastNumber}
function _into_direct_operand_type(T::Type{<:Base.Broadcast.Broadcasted})
    isconcretetype(T) && T <: Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle} &&
        fieldtype(T, :axes) === Nothing && fieldtype(T, :f) <: _NativeBroadcastArithmetic &&
        let args = fieldtypes(fieldtype(T, :args))
            length(args) <= 32 && all(_into_direct_operand_type, args)
        end
end
_into_direct_result(T) = false
_into_direct_result(::Type{_IntoBroadcast{B}}) where {B} = _into_direct_operand_type(B)

function _into_scratch(k, direct, types, cache_apply)
    all(T -> isconcretetype(T) && T <: _IntoNumeric, types) || return nothing
    ast = code_expr(k)
    Meta.isexpr(ast, :function, 2) || return nothing
    signature, body = ast.args
    Meta.isexpr(signature, :tuple) && length(signature.args) == 2 + length(types) &&
        signature.args[1:2] == [_CONSTANTS_ARG, _CALLABLES_ARG] || return nothing
    Meta.isexpr(body, :block) && !isempty(body.args) &&
        Meta.isexpr(last(body.args), :return, 1) || return nothing
    ports = signature.args[3:end]
    ctx = _FusedDecomposition(@__MODULE__, Dict{Symbol,Any}(), Dict{Symbol,Any}(),
        Any[], Any[], Any[], 0, cache_apply)
    for (port, T) in zip(ports, types)
        name = Meta.isexpr(port, :(::), 2) ? port.args[1] : port
        name isa Symbol || return nothing
        name in (_TABLES_ARG, _CACHES_ARG, _CACHE_APPLY_ARG) && return nothing
        ctx.argmap[name], ctx.argtypes[name] = name, T
    end
    for stmt in @view body.args[1:end-1]
        stmt isa LineNumberNode || _into_bind!(ctx, stmt, k) || return nothing
    end
    d = _into_expression!(ctx, only(last(body.args).args), k)
    d === nothing && return nothing
    result, T = d
    isempty(ctx.caches) && return nothing
    isconcretetype(T) && T <: Array{<:_NativeBroadcastNumber} || return nothing
    if direct !== k && length(ctx.ops) == 1 && only(ctx.ops) isa _MaterializeStep &&
       _into_direct_result(Base.promote_op(direct, types...))
        return nothing
    end
    push!(ctx.stmts, Expr(:return, result))
    buffered = Expr(:function, Expr(:tuple, _CONSTANTS_ARG, _CALLABLES_ARG,
        _CACHES_ARG, _CACHE_APPLY_ARG, ports...), Expr(:block, ctx.stmts...))
    S = Tuple{types...}
    kernel = NonAllocatingKernel{S}(compile(_nonallocating_body(buffered)), k.constants,
        Tuple(ctx.ops), Tuple(ctx.caches), cache_apply, k.inputs, k.outputs, k.plan, buffered)
    _IntoScratch{S,typeof(kernel)}(kernel, false)
end

struct _IntoWriter{K,B,S}
    program::K
    bound::B
    scratch::S
end

"""
    prepare_into(spec, exemplars...; kwargs...)
    prepare_into(kernel::PreparedKernel, exemplars...)

Prepare a Julia writer for one array result. Call `write!(dst, args...)` to
fill and return `dst`. Pass a writable array with the result's axes and storage
disjoint from runtime and bound inputs. Each call chooses its destination.
Values use Julia's ordinary array element conversion when needed.

A terminal native array broadcast can fill the destination directly. With
MutatingFunctions loaded and concrete input types from exemplars or declared
ports, a writer can reuse private buffers in straight-line dense numerical
recipes before copying the result. Other recipes evaluate once and copy.
Array elements are copied shallowly, as with `copyto!`. A call that throws
during a direct broadcast can leave a partially written destination.

For a `KernelSpec`, `Graph` or `Plan`, selection and binding keywords are
those of [`prepare`](@ref). Positional exemplars describe the remaining HAVE
ports in order. A prepared kernel keeps its selected computation and passes.
"""
function prepare_into(source::Union{KernelSpec,Graph,Plan}, exemplars...;
                      bound = source isa KernelSpec ? NamedTuple() : (), kwargs...)
    # Normalize an iterable of pairs once, including a single-use iterator.
    binding = bound isa Union{Pair,NamedTuple,Tuple} ? bound : collect(bound)
    held = binding isa NamedTuple ? values(binding) : last(_partial_bound_pairs(binding))
    k = prepare(source; bound = binding, exemplars = isempty(exemplars) ? nothing : exemplars, kwargs...)
    _prepare_into(k, held, exemplars)
end

function prepare_into(k::Union{PreparedKernel,_KernelSignatureCallable{<:PreparedKernel}}, exemplars...)
    program = k isa PreparedKernel ? k : k.target
    if !isempty(exemplars)
        _nonalloc_exemplar_types(program.plan, exemplars, :prepare_into)
        signature = _exact_signature(program)
        signature === Nothing || exemplars isa signature ||
            _prepared_argument_types_error(program, signature, exemplars)
    end
    _prepare_into(k, (), exemplars)
end

_prepare_into(k::_KernelSignatureCallable, bound, exemplars) =
    _KernelSignatureCallable(_prepare_into(k.target, bound, exemplars), k.signature)
_prepare_into(k::_ShapedKernel, bound, exemplars) =
    throw(ArgumentError("prepare_into requires one array result; select its WANT port"))

function _prepare_into(k::PreparedKernel, bound, exemplars)
    length(k.outputs) == 1 || throw(ArgumentError("prepare_into requires one array result"))
    direct = _into_program(k)
    ext = Base.get_extension(@__MODULE__, :ReactiveKernelsMutatingFunctionsExt)
    types = if !isempty(exemplars)
        map(typeof, exemplars)
    elseif _exact_signature(k) !== Nothing
        fieldtypes(_exact_signature(k))
    else
        map(valtype, k.inputs)
    end
    scratch = ext === nothing ? nothing : _into_scratch(k, direct, types, ext.cache_apply!)
    _IntoWriter(direct, bound, scratch)
end

@inline function (callable::_KernelSignatureCallable{<:_IntoWriter})(
        dst::AbstractArray, args...; kwargs...)
    write = (resolved...) -> callable.target(dst, resolved...)
    _kernel_signature_invoke(_KernelSignatureCallable(write, callable.signature), args, NamedTuple(kwargs))
end

@inline function (w::_IntoWriter)(dst::AbstractArray, args::Vararg{Any,N}) where {N}
    _into_aliases(dst, (args, w.bound, w.program.constants, w.program.callables)) &&
        throw(ArgumentError("destination overlaps an input to the prepared kernel"))
    _into_run!(w.scratch, w.program, dst, args)
end

@inline _into_run!(scratch, ordinary, dst, args) = _write_result!(dst, ordinary(args...))

# Base knows the index geometry of views. For contiguous dense storage, also
# catch distinct array headers around overlapping regions of the same memory.
_into_storage_overlap(dst, src) = Base.mightalias(dst, src)
const _IntoDense = Union{Array,SubArray{<:Any,<:Any,<:Array}}
@inline function _into_storage_overlap(dst::_IntoDense, src::_IntoDense)
    if isbitstype(eltype(dst)) && isbitstype(eltype(src)) &&
       (dst isa Array || Base.iscontiguous(dst)) && (src isa Array || Base.iscontiguous(src))
        return _contiguous_storage_overlaps(dst, src)
    end
    Base.mightalias(dst, src) && return true
    # Noncontiguous views of the same array use Base's index check above.
    # Distinct dense headers may wrap overlapping memory at different offsets.
    a, b = parent(dst), parent(src)
    a !== b && isbitstype(eltype(a)) && isbitstype(eltype(b)) && _contiguous_storage_overlaps(a, b)
end

# Only publish a borrowed buffer through ordinary, eager destination storage.
# A custom copyto! can retain its source. Overlapping invocations (including a
# recursive call) and reverse tapes likewise need independent primal storage.
@inline function _into_run!(scratch::_IntoScratch{S}, ordinary, dst::_IntoDense, args) where {S}
    if !(args isa S) || !(dst isa StridedArray{<:_NativeBroadcastNumber}) ||
       EnzymeCore.within_autodiff() ||
       !(@atomicreplace :acquire scratch.busy false => true).success
        return _write_result!(dst, ordinary(args...))
    end
    try
        _write_result!(dst, scratch.kernel(args...))
    catch
        # A BoundsError can expose an intermediate array to its caller.
        # Subsequent calls must never overwrite storage reachable through it.
        for slot in scratch.kernel.caches
            slot[] = nothing
        end
        rethrow()
    finally
        @atomic :release scratch.busy = false
    end
end

# Inspect only data reachable by the call. Numerical arrays take a single
# storage check. Containers can be recursive or partly initialized.
@inline _into_aliases(dst, value, seen = nothing) = Base.@inline _into_fields_alias(dst, value, seen)
_into_aliases(dst, value::Union{Module,Type,Symbol,String}, seen) = false
_into_aliases(dst, k::PreparedKernel, seen) = _into_aliases(dst, (k.constants, k.callables), seen)

function _into_seen(value, seen)
    seen === nothing && (seen = Base.IdSet{Any}())
    value in seen && return nothing
    push!(seen, value)
end

@inline function _into_aliases(dst, value::AbstractArray, seen)
    _into_storage_overlap(dst, value) && return true
    isbitstype(eltype(value)) && return false
    # A lazy array's getindex can compute or throw, even in an arm the kernel
    # never takes. Inspect its stored operands without evaluating its elements.
    value isa Array || return _into_fields_alias(dst, value, seen)
    seen = _into_seen(value, seen)
    seen === nothing && return false
    any(i -> isassigned(value, i) && _into_aliases(dst, value[i], seen), eachindex(value))
end

@generated function _into_fields_alias(dst, value::T, seen) where {T}
    (isbitstype(T) || !isstructtype(T)) && return false
    checks = [:(isdefined(value, $i) && _into_aliases(dst, getfield(value, $i), seen))
              for i in 1:fieldcount(T)]
    any_field = foldr((a, b) -> :($a || $b), checks; init = false)
    ismutabletype(T) || return any_field
    quote
        seen = _into_seen(value, seen)
        seen !== nothing && $any_field
    end
end

@inline function _into_check_axes(dst, source)
    axes(dst) == axes(source) || throw(DimensionMismatch(
        "destination axes $(axes(dst)) differ from result axes $(axes(source))"))
end

_write_result!(dst, value) = throw(ArgumentError("prepare_into expected an array result, got $(typeof(value))"))
@inline function _write_result!(dst, value::AbstractArray)
    _into_check_axes(dst, value)
    # Validate conversions in temporary storage before publishing any element.
    source = eltype(value) <: eltype(dst) ? value : copyto!(similar(dst, axes(dst)), value)
    copyto!(dst, source)
    dst
end

@inline _write_result!(dst, b::_IntoBroadcast) =
    _write_result!(dst, _native_broadcast_materialize(b.value))
@inline function _write_result!(dst, b::_IntoBroadcast{<:Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{N}}}) where {N}
    bc = b.value
    # Stay within native arithmetic on ordinary numeric storage. A user-defined
    # function or array may observe the destination during element evaluation.
    # The fallback finishes all those reads before the caller's first write.
    if N != 0 && !_native_broadcast_wide(bc) && _native_broadcast_operand(bc)
        ready = Base.Broadcast.instantiate(bc)
        T = Base.Broadcast.combine_eltypes(ready.f, ready.args)
        if isconcretetype(T) && T <: eltype(dst) && !_into_aliases(dst, bc.args)
            _into_check_axes(dst, ready)
            _native_broadcast_copyto!(dst, ready)
            return dst
        end
    end
    _write_result!(dst, _native_broadcast_materialize(bc))
end
