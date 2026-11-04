# ChainRules adapter for rules generated from one pure-math graph
# (`src/derivative_rules.jl`). Every `frule`/`rrule` here is generic over the
# rule: the scalar chain rule combines a scalar rule's partials with the
# tangents or the covector; a vector rule's authored branches run through its
# forward cut and staged reverse cuts, with the pullback holding exactly the
# residuals the reverse stage reads. No derivative mathematics is authored here, and
# nothing attaches to a function ReactiveKernels does not own.
module ReactiveKernelsChainRulesCoreExt

using ReactiveKernels: ScalarDerivativeRule, derivative_cut, DerivativeRule,
    forward_cut, has_forward_branch, has_reverse_branch, stage_primal,
    stage_reverse, _rule_activity, _derivative_inputs, rule_inputs
import ChainRulesCore
using ChainRulesCore: AbstractZero, NoTangent, unthunk

@inline _tangent(x, ẋ::AbstractZero) = zero(x)
@inline _tangent(x, ẋ) = unthunk(ẋ)

# ----------------------------------------------------------------- scalar ----

function ChainRulesCore.frule((_, ẋs...), rule::ScalarDerivativeRule{Name,N},
        xs::Vararg{Any,N}) where {Name,N}
    cut = derivative_cut(rule, ntuple(_ -> true, Val(N)), xs...)
    primal = first(cut)
    partials = Base.tail(cut)
    tangent = sum(map((x, ẋ, partial) -> partial * _tangent(x, ẋ), xs, ẋs, partials))
    primal, tangent
end

struct ScalarRulePullback{P}
    partials::P
end
function (pullback::ScalarRulePullback)(ȳ)
    covector = unthunk(ȳ)
    (NoTangent(), map(partial -> covector * partial, pullback.partials)...)
end

function ChainRulesCore.rrule(rule::ScalarDerivativeRule{Name,N},
        xs::Vararg{Any,N}) where {Name,N}
    cut = derivative_cut(rule, ntuple(_ -> true, Val(N)), xs...)
    first(cut), ScalarRulePullback(Base.tail(cut))
end

# ----------------------------------------------------------------- vector ----

function ChainRulesCore.frule((_, ẋs...), rule::DerivativeRule{Name,N},
        xs::Vararg{Any,N}) where {Name,N}
    has_forward_branch(rule) || throw(ArgumentError(
        "frule through $(rule): the rule has no forward branch"))
    length(ẋs) == N || throw(ArgumentError("frule: expected $(N) input tangents"))
    activity = _rule_activity(rule)
    _check_data_tangents(rule, activity, ẋs)
    forward_cut(rule, xs..., map(_tangent, _derivative_inputs(activity, xs),
                                _derivative_inputs(activity, ẋs))...)
end

@generated function _check_data_tangents(rule, ::Val{D}, tangents::Tuple) where {D}
    checks = [:(unthunk(tangents[$i]) isa AbstractZero || throw(ArgumentError(
        "frule: data-only input :" * string(rule_inputs(rule)[$i]) *
        " requires NoTangent() or ZeroTangent()")))
        for i in 1:length(tangents.parameters) if D & (1 << (i - 1)) == 0]
    Expr(:block, checks..., :(nothing))
end

# ChainRules carries no activity information, so the pullback selects every
# declared differentiable input and restores data positions as NoTangent().
@generated function _input_cotangents(::Val{D}, ::Val{N}, cotangents::Tuple) where {D,N}
    index = 0
    values = map(1:N) do i
        D & (1 << (i - 1)) == 0 ? :(NoTangent()) : :(cotangents[$(index += 1)])
    end
    Expr(:tuple, values...)
end

struct RulePullback{R,M,T}
    rule::R
    residuals::T
end
function (pullback::RulePullback{R,M})(ȳ) where {R,M}
    cotangents = M == 0 ? () :
        stage_reverse(pullback.rule, Val(M), pullback.residuals, unthunk(ȳ))
    (NoTangent(), _input_cotangents(Val(M), Val(length(rule_inputs(pullback.rule))), cotangents)...)
end

function ChainRulesCore.rrule(rule::DerivativeRule{Name,N}, xs::Vararg{Any,N}) where {Name,N}
    has_reverse_branch(rule) || throw(ArgumentError(
        "rrule through $(rule): the rule has no reverse branch"))
    _rrule(rule, _rule_activity(rule), xs...)
end

function _rrule(rule, ::Val{M}, xs...) where {M}
    y, residuals = M == 0 ? (rule(xs...), ()) : stage_primal(rule, Val(M), xs...)
    y, RulePullback{typeof(rule),M,typeof(residuals)}(rule, residuals)
end

end # module
