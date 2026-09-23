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
    stage_reverse
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
    forward_cut(rule, xs..., map(_tangent, xs, ẋs)...)
end

# ChainRules carries no activity information, so the pullback is the
# all-active staged reverse cut; it holds that cut's residuals only.
struct RulePullback{R,M,T}
    rule::R
    residuals::T
end
function (pullback::RulePullback{R,M})(ȳ) where {R,M}
    (NoTangent(), stage_reverse(pullback.rule, Val(M), pullback.residuals, unthunk(ȳ))...)
end

function ChainRulesCore.rrule(rule::DerivativeRule{Name,N}, xs::Vararg{Any,N}) where {Name,N}
    has_reverse_branch(rule) || throw(ArgumentError(
        "rrule through $(rule): the rule has no reverse branch"))
    mask = 2^N - 1
    y, residuals = stage_primal(rule, Val(mask), xs...)
    y, RulePullback{typeof(rule),mask,typeof(residuals)}(rule, residuals)
end

end # module
