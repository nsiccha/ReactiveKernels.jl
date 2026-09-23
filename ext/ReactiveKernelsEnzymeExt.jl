# Enzyme adapter for rules generated from one pure-math graph
# (`src/derivative_rules.jl`). Every method here is generic over the rule: the
# activity pattern of the call selects the graph cut, and the cut's partials
# are combined with Enzyme's directions or covectors by the scalar chain rule.
# No derivative mathematics is authored in this file, and nothing attaches to a
# function ReactiveKernels does not own.
module ReactiveKernelsEnzymeExt

using ReactiveKernels: ScalarDerivativeRule, derivative_cut, DerivativeRule, forward_cut,
    reverse_cut, reverse_residuals, has_forward_branch, has_reverse_branch
using ReactiveKernels: _activity_mask
import Enzyme
using Enzyme: Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated,
    BatchDuplicatedNoNeed
using Enzyme.EnzymeCore: Annotation
import Enzyme.EnzymeRules
using Enzyme.EnzymeRules: FwdConfig, RevConfig, AugmentedReturn, needs_primal,
    needs_shadow, width, overwritten

@inline _values(args::Tuple) = map(arg -> arg.val, args)

# ---------------------------------------------------------------- reverse ----

# Activity patterns are read off the argument TYPES, so they are literal
# tuples at code-generation time and every cut selection folds statically.
@generated _reverse_active(args::Tuple) = Tuple(T <: Active for T in args.parameters)

function EnzymeRules.augmented_primal(
        config::RevConfig, rule::Const{<:ScalarDerivativeRule}, ::Type{RT},
        args::Vararg{Annotation,N}) where {RT,N}
    RT <: Union{Const,Active} || throw(ArgumentError(
        "ScalarDerivativeRule reverse rule: unsupported return activity $(RT); " *
        "a real scalar result is Const or Active"))
    active = _reverse_active(args)
    values = _values(args)
    if any(active) && RT <: Active
        cut = derivative_cut(rule.val, active, values...)
        primal = first(cut)
        tape = Base.tail(cut)
    else
        primal = rule.val(values...)
        tape = nothing
    end
    AugmentedReturn(needs_primal(config) ? primal : nothing, nothing, tape)
end

function EnzymeRules.reverse(
        ::RevConfig, ::Const{<:ScalarDerivativeRule}, dret::Active, tape,
        args::Vararg{Annotation,N}) where {N}
    _reverse_shadows(dret.val, tape, args)
end

function EnzymeRules.reverse(
        ::RevConfig, ::Const{<:ScalarDerivativeRule}, ::Type{<:Const}, tape,
        args::Vararg{Annotation,N}) where {N}
    ntuple(_ -> nothing, Val(N))
end

@inline _reverse_shadows(ȳ, ::Nothing, args::Tuple) = map(_ -> nothing, args)
@inline _reverse_shadows(ȳ, tape::Tuple, ::Tuple{}) = ()
@inline function _reverse_shadows(ȳ, tape::Tuple, args::Tuple)
    arg = first(args)
    if arg isa Active
        (_reverse_shadow(ȳ, first(tape)),
         _reverse_shadows(ȳ, Base.tail(tape), Base.tail(args))...)
    else
        (nothing, _reverse_shadows(ȳ, tape, Base.tail(args))...)
    end
end
@inline _reverse_shadow(ȳ::Number, partial) = ȳ * partial
# Batch width > 1: one covector per lane.
@inline _reverse_shadow(ȳ::Tuple, partial) = map(lane -> lane * partial, ȳ)

# ---------------------------------------------------------------- forward ----

const _Tangent = Union{Duplicated,DuplicatedNoNeed,BatchDuplicated,BatchDuplicatedNoNeed}
@generated _forward_active(args::Tuple) = Tuple(T <: _Tangent for T in args.parameters)

function EnzymeRules.forward(
        config::FwdConfig, rule::Const{<:ScalarDerivativeRule}, ::Type{RT},
        args::Vararg{Annotation,N}) where {RT,N}
    values = _values(args)
    if RT <: Const || !needs_shadow(config)
        primal = rule.val(values...)
        return needs_primal(config) ? primal : nothing
    end
    active = _forward_active(args)
    if any(active)
        cut = derivative_cut(rule.val, active, values...)
        primal = first(cut)
        tangent = _forward_accumulate(
            _zero_tangent(Val(width(config)), primal), Base.tail(cut), args)
    else
        primal = rule.val(values...)
        tangent = _zero_tangent(Val(width(config)), primal)
    end
    needs_primal(config) || return tangent
    width(config) == 1 ? Duplicated(primal, tangent) :
        BatchDuplicated(primal, tangent)
end

@inline _zero_tangent(::Val{1}, primal) = zero(primal)
@inline _zero_tangent(::Val{W}, primal) where {W} = ntuple(_ -> zero(primal), Val(W))

@inline _forward_accumulate(acc, ::Tuple{}, ::Tuple{}) = acc
@inline function _forward_accumulate(acc, partials::Tuple, args::Tuple)
    arg = first(args)
    if arg isa _Tangent
        _forward_accumulate(_forward_add(acc, first(partials), arg.dval),
                            Base.tail(partials), Base.tail(args))
    else
        _forward_accumulate(acc, partials, Base.tail(args))
    end
end
@inline _forward_add(acc::Number, partial, direction::Number) =
    acc + partial * direction
@inline _forward_add(acc::Tuple, partial, directions::Tuple) =
    map((lane, direction) -> lane + partial * direction, acc, directions)

# ================================================================ vector ====
# Rules with authored forward/reverse branches (`DerivativeRule`). Reverse
# mode stages the graph: the augmented primal runs the primal cut, hands
# Enzyme a zero shadow for an array result (Enzyme accumulates the covector
# into it), and retains exactly the inputs the selected reverse cut reads
# (copied when Enzyme says the argument may be overwritten before the reverse
# pass); the reverse pass runs that cut with the covector and accumulates the
# cotangents into the argument shadows. Forward mode runs the forward cut with
# Enzyme's directions (zero for inactive inputs), once per batch lane.

const _ReverseActive = Union{Active,Duplicated,BatchDuplicated}
@generated _reverse_active_vector(args::Tuple) =
    Tuple(T <: _ReverseActive for T in args.parameters)

@inline _zero_shadow(::Val{1}, primal) = _zero_like(primal)
@inline _zero_shadow(::Val{W}, primal) where {W} = ntuple(_ -> _zero_like(primal), Val(W))
@inline _zero_like(x::AbstractArray) = zero(x)
@inline _zero_like(x::Number) = zero(x)

# Retain the residual inputs for the reverse cut: `nothing` for inputs the cut
# never reads, a copy for an array Enzyme may overwrite before the reverse pass.
@inline function _retain_residuals(mask::NTuple{N,Bool}, values::Tuple, overwritten_flags) where {N}
    ntuple(Val(N)) do i
        mask[i] || return nothing
        value = values[i]
        # `overwritten` includes the function object at position 1.
        (value isa AbstractArray && overwritten_flags[i + 1]) ? copy(value) : value
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfig, rule::Const{<:DerivativeRule}, ::Type{RT},
        args::Vararg{Annotation,N}) where {RT,N}
    active = _reverse_active_vector(args)
    values = _values(args)
    primal = rule.val(values...)
    shadow = needs_shadow(config) ? _zero_shadow(Val(width(config)), primal) : nothing
    if !any(active) || RT <: Const
        return AugmentedReturn(needs_primal(config) ? primal : nothing, shadow, nothing)
    end
    has_reverse_branch(rule.val) || throw(ArgumentError(
        "reverse-mode Enzyme through $(rule.val): the rule has no reverse branch " *
        "(author a covector + cotangents branch in its graph)"))
    mask = _activity_mask(active, 1)
    residuals = _retain_residuals(reverse_residuals(rule.val, Val(mask)), values,
                                  overwritten(config))
    AugmentedReturn(needs_primal(config) ? primal : nothing, shadow, (shadow, residuals))
end

function EnzymeRules.reverse(
        config::RevConfig, rule::Const{<:DerivativeRule}, dret, tape,
        args::Vararg{Annotation,N}) where {N}
    tape === nothing && return ntuple(_ -> nothing, Val(N))
    shadow, residuals = tape
    active = _reverse_active_vector(args)
    mask = _activity_mask(active, 1)
    _reverse_lanes(Val(width(config)), rule.val, Val(mask), dret, shadow, residuals, args)
end

# Width 1: one covector, one cotangent per active input.
@inline function _reverse_lanes(::Val{1}, rule, ::Val{M}, dret, shadow, residuals, args::Tuple) where {M}
    ȳ = dret isa Active ? dret.val : shadow
    cotangents = reverse_cut(rule, Val(M), residuals..., ȳ)
    _apply_cotangents(args, cotangents)
end
# Width W: one covector per lane; Active arguments return a tuple of lanes,
# batched arguments accumulate lane by lane.
@inline function _reverse_lanes(::Val{W}, rule, ::Val{M}, dret, shadow, residuals, args::Tuple) where {W,M}
    lanes = ntuple(Val(W)) do w
        ȳ = dret isa Active ? dret.val[w] : shadow[w]
        reverse_cut(rule, Val(M), residuals..., ȳ)
    end
    _apply_cotangent_lanes(args, lanes, Val(W))
end

@inline _apply_cotangents(::Tuple{}, ::Tuple) = ()
@inline function _apply_cotangents(args::Tuple, cotangents::Tuple)
    arg = first(args)
    if arg isa Active
        (first(cotangents), _apply_cotangents(Base.tail(args), Base.tail(cotangents))...)
    elseif arg isa Duplicated
        _accumulate!(arg.dval, first(cotangents))
        (nothing, _apply_cotangents(Base.tail(args), Base.tail(cotangents))...)
    else
        (nothing, _apply_cotangents(Base.tail(args), cotangents)...)
    end
end
@inline _apply_cotangent_lanes(::Tuple{}, lanes, ::Val) = ()
@inline function _apply_cotangent_lanes(args::Tuple, lanes::Tuple, ::Val{W}) where {W}
    arg = first(args)
    if arg isa Active
        value = ntuple(w -> first(lanes[w]), Val(W))
        (value, _apply_cotangent_lanes(Base.tail(args), map(Base.tail, lanes), Val(W))...)
    elseif arg isa BatchDuplicated
        for w in 1:W
            _accumulate!(arg.dval[w], first(lanes[w]))
        end
        (nothing, _apply_cotangent_lanes(Base.tail(args), map(Base.tail, lanes), Val(W))...)
    else
        (nothing, _apply_cotangent_lanes(Base.tail(args), lanes, Val(W))...)
    end
end
@inline function _accumulate!(shadow::AbstractArray, cotangent)
    shadow .+= cotangent
    nothing
end

@inline _direction(arg::Union{Duplicated,DuplicatedNoNeed}) = arg.dval
@inline _direction(arg::Annotation) = _zero_like(arg.val)
@inline _direction(arg::Union{BatchDuplicated,BatchDuplicatedNoNeed}, w::Int) = arg.dval[w]
@inline _direction(arg::Annotation, w::Int) = _zero_like(arg.val)

function EnzymeRules.forward(
        config::FwdConfig, rule::Const{<:DerivativeRule}, ::Type{RT},
        args::Vararg{Annotation,N}) where {RT,N}
    values = _values(args)
    if RT <: Const || !needs_shadow(config)
        primal = rule.val(values...)
        return needs_primal(config) ? primal : nothing
    end
    has_forward_branch(rule.val) || throw(ArgumentError(
        "forward-mode Enzyme through $(rule.val): the rule has no forward branch " *
        "(author a directions + tangent branch in its graph)"))
    if width(config) == 1
        primal, tangent = forward_cut(rule.val, values..., map(_direction, args)...)
        return needs_primal(config) ? Duplicated(primal, tangent) : tangent
    end
    lanes = ntuple(Val(width(config))) do w
        forward_cut(rule.val, values..., map(arg -> _direction(arg, w), args)...)
    end
    tangents = map(last, lanes)
    needs_primal(config) ? BatchDuplicated(first(lanes)[1], tangents) : tangents
end

end # module
