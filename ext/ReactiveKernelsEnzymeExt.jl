# Enzyme adapter for rules generated from one pure-math graph
# (`src/derivative_rules.jl`). The rule methods are generic: the call's
# activity pattern selects the graph cut, and its partials combine with
# Enzyme's directions or covectors by the scalar chain rule. The native AD
# preparation below chooses a bounded forward batch over the unchanged RK
# callable. No derivative mathematics is authored here, and nothing attaches
# to a function ReactiveKernels does not own. This extension also declares
# which Enzyme modes retain WANT values during a prepared gradient's primal
# sweep.
module ReactiveKernelsEnzymeExt

using ReactiveKernels: ScalarDerivativeRule, derivative_cut, DerivativeRule, forward_cut,
    has_forward_branch, has_reverse_branch, stage_primal, stage_reverse
using ReactiveKernels: _stage_residual_sources
import ReactiveKernels
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
        config::RevConfig, ::Const{<:ScalarDerivativeRule}, ::Type{<:Const}, tape,
        args::Vararg{Annotation,N}) where {N}
    # A constant result contributes zero even when an input annotation is
    # Active. Enzyme requires a numeric cotangent for that input slot.
    map(arg -> arg isa Active ? _zero_tangent(Val(width(config)), arg.val) : nothing, args)
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
# mode stages the graph: the augmented primal runs the primal stage of the
# selected activity pattern (`stage_primal`: the primal plus the reverse
# cut's covector-independent residuals, computed once), hands Enzyme a zero
# shadow for an array result (Enzyme accumulates the covector into it), and
# retains those residuals; the reverse pass runs `stage_reverse` with the
# covector and accumulates the cotangents into the argument shadows. Forward mode runs the forward cut with
# Enzyme's directions (zero for inactive inputs), once per batch lane.

const _ReverseActive = Union{Active,Duplicated,BatchDuplicated}
# The activity mask as a TYPE (bit `i` set when argument `i` is active), read
# off the annotation types at code generation. Cut selection and the
# no-derivative branch then never depend on constant propagation, which
# inference skips for large rule types: a value-level mask there left
# `augmented_primal` returning a Union of tapes, and Enzyme compiled `reverse`
# against the `nothing` one.
@generated function _reverse_mask(args::Tuple)
    mask = 0
    for (i, T) in enumerate(args.parameters)
        T <: _ReverseActive && (mask |= 1 << (i - 1))
    end
    :(Val{$mask}())
end

@inline _zero_shadow(::Val{1}, primal) = _zero_like(primal)
@inline _zero_shadow(::Val{W}, primal) where {W} = ntuple(_ -> _zero_like(primal), Val(W))
@inline _zero_like(x::AbstractArray) = zero(x)
@inline _zero_like(x::Number) = zero(x)

# Retain the staged residuals across the reverse pass. An input array Enzyme
# says may be overwritten before then is copied, and so is a residual that IS
# the returned primal array (the caller owns it and may overwrite it); fresh
# intermediates of the primal stage are owned here and kept as they are.
@inline function _retain_residuals(sources::NTuple{R,Int}, residuals::NTuple{R,Any},
        primal, overwritten_flags) where {R}
    ntuple(Val(R)) do k
        value = residuals[k]
        value isa AbstractArray || return value
        source = sources[k]
        # `overwritten` includes the function object at position 1.
        source > 0 ? (overwritten_flags[source + 1] ? copy(value) : value) :
            value === primal ? copy(value) : value
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfig, rule::Const{<:DerivativeRule}, ::Type{RT},
        args::Vararg{Annotation,N}) where {RT,N}
    mask = _reverse_mask(args)
    values = _values(args)
    if mask === Val(0) || RT <: Const
        primal = rule.val(values...)
        shadow = needs_shadow(config) ? _zero_shadow(Val(width(config)), primal) : nothing
        return AugmentedReturn(needs_primal(config) ? primal : nothing, shadow, nothing)
    end
    has_reverse_branch(rule.val) || throw(ArgumentError(
        "reverse-mode Enzyme through $(rule.val): the rule has no reverse branch " *
        "(author a covector + cotangents branch in its graph)"))
    primal, staged = stage_primal(rule.val, mask, values...)
    shadow = needs_shadow(config) ? _zero_shadow(Val(width(config)), primal) : nothing
    residuals = _retain_residuals(_stage_residual_sources(rule.val, mask), staged, primal,
                                  overwritten(config))
    AugmentedReturn(needs_primal(config) ? primal : nothing, shadow, (shadow, residuals))
end

function EnzymeRules.reverse(
        config::RevConfig, rule::Const{<:DerivativeRule}, dret, tape,
        args::Vararg{Annotation,N}) where {N}
    # No derivative flowed (constant result or no active input): an Active
    # input still needs a numeric zero cotangent.
    tape === nothing && return map(arg -> arg isa Active ?
        _zero_tangent(Val(width(config)), arg.val) : nothing, args)
    shadow, residuals = tape
    _reverse_lanes(Val(width(config)), rule.val, _reverse_mask(args), dret, shadow,
                   residuals, args)
end

# Width 1: one covector, one cotangent per active input.
@inline function _reverse_lanes(::Val{1}, rule, ::Val{M}, dret, shadow, residuals, args::Tuple) where {M}
    ȳ = dret isa Active ? dret.val : shadow
    cotangents = stage_reverse(rule, Val(M), residuals, ȳ)
    _apply_cotangents(args, cotangents)
end
# Width W: one covector per lane; Active arguments return a tuple of lanes,
# batched arguments accumulate lane by lane.
@inline function _reverse_lanes(::Val{W}, rule, ::Val{M}, dret, shadow, residuals, args::Tuple) where {W,M}
    lanes = ntuple(Val(W)) do w
        ȳ = dret isa Active ? dret.val[w] : shadow[w]
        stage_reverse(rule, Val(M), residuals, ȳ)
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

# ------------------------------------------------- retained WANT outputs ----

# A reverse-mode Enzyme gradient runs the differentiated target once, at the
# requested point, and keeps the target's writes, so WANT values retained in
# that sweep (`prepare_ad(...; retain)`) are its primal values.
ReactiveKernels.ad_retains_primal_sweep(
    ::ReactiveKernels.DifferentiationInterface.AutoEnzyme{
        <:Union{Nothing,Enzyme.ReverseMode}}) = true

# A small active vector can carry one forward tangent lane per coordinate.
# Enzyme differentiates the *same* native RK callable, so every plate cell,
# prior, transform, and bound-data read follows its ordinary primal path. This
# is a scheduling choice, not a derivative rule for a likelihood or operation.
const _FORWARD_BATCH_MODE = Enzyme.set_abi(Enzyme.ForwardWithPrimal, Enzyme.InlineABI)
const _ReverseAutoEnzyme = ReactiveKernels.DifferentiationInterface.AutoEnzyme{
    <:Union{Nothing,Enzyme.ReverseMode}}

struct _NativeForwardBatchAD{K,R,B}
    reverse::R
    basis::B
end

function ReactiveKernels._native_forward_batch_eligible(
        ::_ReverseAutoEnzyme, kernel::ReactiveKernels.PreparedKernel,
        point::Vector{Float64})
    1 <= length(point) <= 4 || return false
    plan = kernel.plan
    recipes = plan.recipes
    all(r -> !r.effectful && !(r.op isa ReactiveKernels._AuthoredScanOp),
        recipes) || return false
    any(recipes) do recipe
        recipe.op isa ReactiveKernels._AuthoredPlateOp || return false
        all(r -> !r.effectful && !(r.op isa Union{
                ReactiveKernels._AuthoredPlateOp,
                ReactiveKernels._AuthoredScanOp}),
            recipe.op.kernel.plan.recipes) || return false
        consumer = ReactiveKernels._authored_plate_sum_recipe(plan, recipe)
        consumer === nothing && return false
        output_id = ReactiveKernels.canon_id(
            plan.graph, only(recipe.outputs).id)
        any(w -> ReactiveKernels.canon_id(plan.graph, w.id) == output_id,
            plan.want) && return false
        !any(candidate -> candidate !== consumer && any(input ->
            ReactiveKernels.canon_id(plan.graph, input.id) == output_id,
            candidate.inputs), recipes)
    end
end

function ReactiveKernels._prepare_native_forward_batch_ad(
        kernel::ReactiveKernels.PreparedKernel, resolver,
        backend::_ReverseAutoEnzyme, args::Tuple, kwargs::NamedTuple, active;
        retain::Tuple = (), warm::Bool = false)
    isempty(retain) || throw(ArgumentError(
        "native forward batching does not retain further WANT values"))
    reverse = ReactiveKernels._prepare_ad(kernel, resolver, backend,
        args, kwargs, active; warm = false)
    point, contexts = ReactiveKernels._ad_prepared_arguments(
        reverse, args, kwargs)
    point isa Vector{Float64} || throw(ArgumentError(
        "native forward batching needs one active Vector{Float64} port"))
    1 <= length(point) <= 4 || throw(ArgumentError(
        "native forward batching supports one to four active coordinates"))
    all(c -> c isa ReactiveKernels.DifferentiationInterface.Constant,
        contexts) || throw(ArgumentError(
        "native forward batching needs read-only Constant contexts"))
    K = length(point)
    basis = ntuple(K) do direction
        [coordinate == direction ? 1.0 : 0.0 for coordinate in 1:K]
    end
    prepared = _NativeForwardBatchAD{K,typeof(reverse),typeof(basis)}(
        reverse, basis)
    warm && Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
        prepared, similar(point), args...; kwargs...)
    prepared
end

function _forward_batch_value_gradient!(prepared::_NativeForwardBatchAD{K},
        gradient::AbstractVector, point::Vector{Float64}, contexts) where {K}
    axes(gradient) == axes(point) || throw(DimensionMismatch(
        "gradient axes $(axes(gradient)) differ from active point axes $(axes(point))"))
    annotations = map(c -> Enzyme.Const(c.data), contexts)
    tangents, value = Enzyme.autodiff(_FORWARD_BATCH_MODE,
        prepared.reverse.call, Enzyme.BatchDuplicated(point, prepared.basis),
        annotations...)
    @inbounds for coordinate in 1:K
        gradient[coordinate] = getfield(tangents, coordinate)
    end
    value, gradient
end

function _forward_batch_value_gradient!(prepared::_NativeForwardBatchAD{1},
        gradient::AbstractVector, point::Vector{Float64}, contexts)
    axes(gradient) == axes(point) || throw(DimensionMismatch(
        "gradient axes $(axes(gradient)) differ from active point axes $(axes(point))"))
    annotations = map(c -> Enzyme.Const(c.data), contexts)
    tangent, value = Enzyme.autodiff(_FORWARD_BATCH_MODE,
        prepared.reverse.call, Enzyme.Duplicated(point, prepared.basis[1]),
        annotations...)
    gradient[1] = tangent
    value, gradient
end

function ReactiveKernels.ad_value_and_gradient!(prepared::_NativeForwardBatchAD{K},
        gradient, args...; kwargs...) where {K}
    point, contexts = ReactiveKernels._ad_prepared_arguments(
        prepared.reverse, args, NamedTuple(kwargs))
    if point isa Vector{Float64} && length(point) == K &&
            all(c -> c isa ReactiveKernels.DifferentiationInterface.Constant,
                contexts)
        return _forward_batch_value_gradient!(prepared, gradient, point, contexts)
    end
    ReactiveKernels.ad_value_and_gradient!(prepared.reverse, gradient,
                                           args...; kwargs...)
end

function ReactiveKernels.ad_value_and_gradient(prepared::_NativeForwardBatchAD,
        args...; kwargs...)
    point, _ = ReactiveKernels._ad_prepared_arguments(
        prepared.reverse, args, NamedTuple(kwargs))
    point isa AbstractVector || return ReactiveKernels.ad_value_and_gradient(
        prepared.reverse, args...; kwargs...)
    ReactiveKernels.ad_value_and_gradient!(prepared, similar(point), args...;
                                            kwargs...)
end

ReactiveKernels.ad_gradient(prepared::_NativeForwardBatchAD, args...; kwargs...) =
    last(ReactiveKernels.ad_value_and_gradient(prepared, args...; kwargs...))

ReactiveKernels.ad_value_gradient_and_retained!(prepared::_NativeForwardBatchAD,
        gradient, args...; kwargs...) =
    ReactiveKernels.ad_value_gradient_and_retained!(prepared.reverse,
        gradient, args...; kwargs...)

ReactiveKernels.ad_value_gradient_and_retained(prepared::_NativeForwardBatchAD,
        args...; kwargs...) =
    ReactiveKernels.ad_value_gradient_and_retained(prepared.reverse,
        args...; kwargs...)

ReactiveKernels.compile_ad_gradient(prepared::_NativeForwardBatchAD,
        args...; kwargs...) =
    ReactiveKernels.compile_ad_gradient(prepared.reverse, args...; kwargs...)

ReactiveKernels.compile_ad_value_and_gradient(prepared::_NativeForwardBatchAD,
        args...; kwargs...) =
    ReactiveKernels.compile_ad_value_and_gradient(prepared.reverse,
                                                  args...; kwargs...)

end # module
