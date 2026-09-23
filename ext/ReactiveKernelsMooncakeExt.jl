# Mooncake adapter for rules generated from one pure-math graph
# (`src/derivative_rules.jl`). Every `rrule!!`/`frule!!` here is generic over
# the rule: a scalar rule's partials are combined with the covector or the
# directions by the scalar chain rule; a vector rule's authored branches run
# through its forward and reverse cuts, with the pullback holding exactly the
# residuals the reverse cut reads. No derivative mathematics is authored here,
# and nothing attaches to a function ReactiveKernels does not own.
module ReactiveKernelsMooncakeExt

using ReactiveKernels: ScalarDerivativeRule, derivative_cut, DerivativeRule,
    forward_cut, reverse_cut, reverse_residuals, has_forward_branch,
    has_reverse_branch
import Mooncake
using Mooncake: CoDual, Dual, MinimalCtx, NoRData, NoTangent, @is_primitive,
    primal, tangent, zero_fcodual

# A generated rule's fields are operation tables: nothing to differentiate.
Mooncake.tangent_type(::Type{<:ScalarDerivativeRule}) = NoTangent
Mooncake.tangent_type(::Type{<:DerivativeRule}) = NoTangent

@is_primitive MinimalCtx Tuple{ScalarDerivativeRule,Vararg}
@is_primitive MinimalCtx Tuple{DerivativeRule,Vararg}

# Argument handling shared by both rule kinds. A real scalar returns its
# cotangent as rdata; an `Array` of reals accumulates into its fdata (Mooncake
# keeps array tangents in the forward data) and returns `NoRData`; an integer
# is not differentiable. Anything else is refused rather than guessed.
@inline _rdata(x::CoDual{<:Base.IEEEFloat}, cotangent) =
    convert(typeof(primal(x)), cotangent)
@inline function _rdata(x::CoDual{<:Array{<:Base.IEEEFloat}}, cotangent)
    tangent(x) .+= cotangent
    NoRData()
end
@inline _rdata(::CoDual{<:Integer}, cotangent) = NoRData()
_rdata(x::CoDual, cotangent) = throw(ArgumentError(
    "generated Mooncake rule: unsupported argument type $(typeof(primal(x))); " *
    "rule inputs must be real scalars, arrays of reals, or integers"))

@inline _direction(x::Dual) = _direction(primal(x), tangent(x))
@inline _direction(x, ::NoTangent) = zero(x)
@inline _direction(x, dx) = dx

# ----------------------------------------------------------------- scalar ----

function Mooncake.frule!!(f::Dual{<:ScalarDerivativeRule{Name,N}},
        xs::Vararg{Dual,N}) where {Name,N}
    cut = derivative_cut(primal(f), ntuple(_ -> true, Val(N)), map(primal, xs)...)
    partials = Base.tail(cut)
    Dual(first(cut), sum(map((partial, x) -> partial * _direction(x), partials, xs)))
end

function Mooncake.rrule!!(f::CoDual{<:ScalarDerivativeRule{Name,N}},
        xs::Vararg{CoDual,N}) where {Name,N}
    cut = derivative_cut(primal(f), ntuple(_ -> true, Val(N)), map(primal, xs)...)
    partials = Base.tail(cut)
    function scalar_rule_pullback(ȳ)
        (NoRData(), map((x, partial) -> _rdata(x, ȳ * partial), xs, partials)...)
    end
    zero_fcodual(first(cut)), scalar_rule_pullback
end

# ----------------------------------------------------------------- vector ----

function Mooncake.frule!!(f::Dual{<:DerivativeRule{Name,N}},
        xs::Vararg{Dual,N}) where {Name,N}
    rule = primal(f)
    has_forward_branch(rule) || throw(ArgumentError(
        "forward-mode Mooncake through $(rule): the rule has no forward branch " *
        "(author a directions + tangent branch in its graph)"))
    y, ẏ = forward_cut(rule, map(primal, xs)..., map(_direction, xs)...)
    Dual(y, ẏ)
end

# The output covector: an array result's cotangent accumulates into the fdata
# handed out with the primal; a scalar result's arrives as rdata.
@inline _output(y::Array{<:Base.IEEEFloat}) = (dy = zero(y); (CoDual(y, dy), dy))
@inline _output(y) = (zero_fcodual(y), nothing)
@inline _covector(ȳ, ::Nothing) = ȳ
@inline _covector(::NoRData, dy) = dy

# Mooncake carries no activity information, so the pullback runs the
# all-active cut; it retains only the inputs that cut reads.
function Mooncake.rrule!!(f::CoDual{<:DerivativeRule{Name,N}},
        xs::Vararg{CoDual,N}) where {Name,N}
    rule = primal(f)
    has_reverse_branch(rule) || throw(ArgumentError(
        "reverse-mode Mooncake through $(rule): the rule has no reverse branch " *
        "(author a covector + cotangents branch in its graph)"))
    mask = Val(2^N - 1)
    values = map(primal, xs)
    kept = reverse_residuals(rule, mask)
    residuals = ntuple(i -> kept[i] ? values[i] : nothing, Val(N))
    y, dy = _output(rule(values...))
    function vector_rule_pullback(ȳ)
        cotangents = reverse_cut(rule, mask, residuals..., _covector(ȳ, dy))
        (NoRData(), map(_rdata, xs, cotangents)...)
    end
    y, vector_rule_pullback
end

end # module
