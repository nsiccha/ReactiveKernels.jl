# The ordinary prepared program supplies the vector semantics, including any
# passes the caller already applied. Only its terminal materialization can be
# deferred to the writer. Every other result goes through the same copy path.
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

struct _IntoWriter{K,B}
    program::K
    bound::B
end

"""
    prepare_into(spec, exemplars...; kwargs...)
    prepare_into(kernel::PreparedKernel, exemplars...)

Prepare a Julia writer for one array result. Call `write!(dst, args...)` to
fill and return `dst`. Pass a writable array with the result's axes and storage
disjoint from runtime and bound inputs. Each call chooses its destination.
Values use Julia's ordinary array element conversion when needed.

A terminal native array broadcast can fill the destination directly. Other
recipes evaluate once and copy their result into it. Array elements are
copied shallowly, as with `copyto!`. A call that throws during a direct
broadcast can leave a partially written destination.

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
    _prepare_into(k, held)
end

function prepare_into(k::Union{PreparedKernel,_KernelSignatureCallable{<:PreparedKernel}}, exemplars...)
    program = k isa PreparedKernel ? k : k.target
    if !isempty(exemplars)
        _nonalloc_exemplar_types(program.plan, exemplars, :prepare_into)
        signature = _exact_signature(program)
        signature === Nothing || exemplars isa signature ||
            _prepared_argument_types_error(program, signature, exemplars)
    end
    _prepare_into(k, ())
end

_prepare_into(k::_KernelSignatureCallable, bound) =
    _KernelSignatureCallable(_prepare_into(k.target, bound), k.signature)
_prepare_into(k::_ShapedKernel, bound) =
    throw(ArgumentError("prepare_into requires one array result; select its WANT port"))

function _prepare_into(k::PreparedKernel, bound)
    length(k.outputs) == 1 || throw(ArgumentError("prepare_into requires one array result"))
    _IntoWriter(_into_program(k), bound)
end

@inline function (callable::_KernelSignatureCallable{<:_IntoWriter})(
        dst::AbstractArray, args...; kwargs...)
    write = (resolved...) -> callable.target(dst, resolved...)
    _kernel_signature_invoke(_KernelSignatureCallable(write, callable.signature), args, NamedTuple(kwargs))
end

@inline function (w::_IntoWriter)(dst::AbstractArray, args::Vararg{Any,N}) where {N}
    _into_aliases(dst, (args, w.bound, w.program.constants, w.program.callables)) &&
        throw(ArgumentError("destination overlaps an input to the prepared kernel"))
    _write_result!(dst, w.program(args...))
end

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
