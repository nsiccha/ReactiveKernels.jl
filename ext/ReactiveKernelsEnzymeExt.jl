# Enzyme adapter for rules generated from one pure-math graph
# (`src/derivative_rules.jl`). Every method here is generic over the rule: the
# activity pattern of the call selects the graph cut, and the cut's partials
# are combined with Enzyme's directions or covectors by the scalar chain rule.
# No derivative mathematics is authored in this file, and nothing attaches to a
# function ReactiveKernels does not own.
module ReactiveKernelsEnzymeExt

using ReactiveKernels: ScalarDerivativeRule, derivative_cut
import Enzyme
using Enzyme: Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated,
    BatchDuplicatedNoNeed
using Enzyme.EnzymeCore: Annotation
import Enzyme.EnzymeRules
using Enzyme.EnzymeRules: FwdConfig, RevConfig, AugmentedReturn, needs_primal,
    needs_shadow, width

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

end # module
