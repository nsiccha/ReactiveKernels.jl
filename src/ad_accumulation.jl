# Additive reverse derivatives of scalar directional contractions. Directions
# act on a subset of HAVE ports; the enclosing reverse remains active over the
# complete requested boundary. In particular, it differentiates the values of
# the other ports even though the inner forward passes hold them fixed.
struct _ADPartialCall{I,F}
    call::F
end

_ad_partial_point(point, ::Val{I}) where {I} =
    I isa Int ? getfield(point, I) : map(i -> point[i], I)

# I is a boundary selector, not a bound array's length or contents.
@generated function _ad_replace_partial(point::P, localpoint, ::Val{I}) where {P<:Tuple,I}
    indices = _ad_selector_indices(I)
    Expr(:tuple, (begin
        k = findfirst(==(j), indices)
        k === nothing ? :(getfield(point, $j)) :
            I isa Int ? :localpoint : :(getfield(localpoint, $k))
    end for j in 1:fieldcount(P))...)
end

@inline (f::_ADPartialCall{I})(localpoint, point, contexts...) where {I} =
    Base.@inline f.call(_ad_replace_partial(point, localpoint, Val(I)), contexts...)

@inline function _ad_directional_value(point, f, direction, backend, contexts...)
    only(DifferentiationInterface.pushforward(f, backend, point, (direction,),
        map(DifferentiationInterface.Constant, contexts)...))
end

struct _ADContraction{O,I,F,B,C}
    call::F
    first_backend::B
    second_backend::C
end
@inline (f::_ADContraction{0})(point, direction, weight, contexts...) =
    weight * (Base.@inline f.call(point, contexts...))
@inline function (f::_ADContraction{1,I})(point, direction, weight, contexts...) where {I}
    weight * _ad_directional_value(_ad_partial_point(point, Val(I)),
        _ADPartialCall{I,typeof(f.call)}(f.call), direction,
        f.first_backend, point, contexts...)
end
@inline function (f::_ADContraction{2,I})(point, direction, weight, contexts...) where {I}
    weight * _ad_directional_value(_ad_partial_point(point, Val(I)),
        _ad_directional_value, direction, f.second_backend,
        _ADPartialCall{I,typeof(f.call)}(f.call), direction,
        f.first_backend, point, contexts...)
end

struct PreparedADAccumulator{P,F,I}
    prepared::P
    call::F
end

_ad_accumulation_supported(backend) = false
_ad_accumulate_backend!(backend, call, destination, point, direction, weight, contexts) =
    throw(ArgumentError("additive AD requires a backend supporting accumulation into supplied shadows"))

function _ad_contraction_backends(backend, order)
    order == 0 && return (backend, nothing, nothing)
    backend isa DifferentiationInterface.SecondOrder || throw(ArgumentError(
        "an order-$order accumulator requires a SecondOrder backend"))
    order == 1 && return (backend.outer, backend.inner, nothing)
    _ad_check_third_backend(backend)
    backend.outer.outer, backend.inner, backend.outer.inner
end

"""
    prepare_ad_accumulator(kernel, backend, args...; active, directional=active, order=0)

Prepare additive derivatives of one scalar WANT. `active` names the HAVE ports
whose derivatives are accumulated; `directional` names a subset of them.
`order = 0`, `1`, or `2` differentiates `f`, `Df[v]`, or `D²f[v,v]`,
respectively. Directions act only on `directional`. The result remains
differentiable with respect to **every** `active` port, including the ports
held fixed by the directional passes. This gives mixed derivatives without
allocating zero directions for large shared operands.

The backend is a reverse backend for order zero, `SecondOrder(reverse,
forward)` for order one, or `SecondOrder(SecondOrder(reverse, forward),
forward)` for order two, as for `prepare_ad_hvp` and `prepare_ad_third`.
Currently accumulation is supported by native reverse `AutoEnzyme` with a
constant function annotation. Load Enzyme before preparation.

Use [`ad_accumulator_storage`](@ref) once to allocate a matching destination,
then [`ad_accumulate!`](@ref) repeatedly. Storage follows `active` as a tuple,
even for one port; scalar ports use `Ref` storage. The operator and its
destinations serve one caller at a time.
"""
function prepare_ad_accumulator(kernel::PreparedKernel, backend, args...;
        active, directional = active, order::Integer = 0, kwargs...)
    order in (0, 1, 2) || throw(ArgumentError("accumulator order must be 0, 1, or 2"))
    reverse, first_backend, second_backend = _ad_contraction_backends(backend, order)
    _ad_accumulation_supported(reverse) || throw(ArgumentError(
        "additive AD requires native reverse AutoEnzyme with a constant function annotation"))
    selectors = active isa Tuple ? active : (active,)
    allindices = _ad_active_selector(kernel, selectors)
    directionindices = _ad_active_selector(kernel, directional)
    positions = map(_ad_selector_indices(directionindices)) do index
        position = findfirst(==(index), allindices)
        position === nothing && throw(ArgumentError("directional ports must be included in active"))
        position
    end
    I = directionindices isa Int ? only(positions) : positions
    prepared = prepare_ad(kernel, reverse, args...; active = selectors, kwargs...)
    call = _ADContraction{Int(order),I,typeof(prepared.call),
        typeof(first_backend),typeof(second_backend)}(
            prepared.call, first_backend, second_backend)
    PreparedADAccumulator{typeof(prepared),typeof(call),I}(prepared, call)
end

"""
    ad_accumulator_storage(prepared, args...)

Allocate zero derivative storage for the active boundary of a
[`PreparedADAccumulator`](@ref) at `args`. Do this once per accumulation pass,
or reuse and zero the storage between passes, not between individual terms.
"""
function ad_accumulator_storage(q::PreparedADAccumulator, args...; kwargs...)
    point, _ = _ad_prepared_arguments(q.prepared, args, NamedTuple(kwargs))
    _ad_zero_cotangent(point)
end

"""
    ad_accumulate!(prepared, destination, direction, weight, args...)

Add `weight * ∇Dⁿf[direction,…]` to `destination`, where `n` is the prepared
order, and return the weighted scalar contraction. For order zero pass
`direction = nothing`. Destination arrays and refs must match the active
point's types and shapes and must not alias primal inputs. No destination is
cleared or copied; a repeated call adds another contribution. A direction
follows the `directional` selector's structure, not the full active boundary.
`weight` and the direction are held fixed in the enclosing reverse pass.
"""
function ad_accumulate!(q::PreparedADAccumulator{P,F,I}, destination, direction,
        weight::Real, args...; kwargs...) where {P,F,I}
    point, contexts = _ad_prepared_arguments(q.prepared, args, NamedTuple(kwargs))
    typeof(destination) === typeof(point) && _ad_tangent_matches(point, destination) ||
        throw(DimensionMismatch("accumulator storage must match the active point's types and shapes"))
    tangent = if q.call isa _ADContraction{0}
        nothing
    else
        localpoint = _ad_partial_point(point, Val(I))
        _ad_check_tangents(localpoint, (direction,))
        _ad_tangent_point(localpoint, direction)
    end
    _ad_accumulate_backend!(q.prepared.backend, q.call, destination, point,
        tangent, weight, contexts)
end
