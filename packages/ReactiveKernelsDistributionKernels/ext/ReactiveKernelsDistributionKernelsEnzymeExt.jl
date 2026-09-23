# INTERIM reverse-mode Enzyme rules for DistributionKernels' OWN `loggamma` and
# `logbeta` entry points — to be REPLACED by adapters generated from one
# pure-math graph (ReactiveKernels:review todo 2026-09-23T03-00-11-762-1q9sudt;
# policy: docs/src/constraints.md, "Derivative rules come from one mathematical
# graph, never by hand"). The rules attach to functions this package owns,
# never to SpecialFunctions' (that would be type piracy and would change every
# Enzyme user in the session).
#
# Why a rule at all: Enzyme 0.13's reverse mode aborts the process (an LLVM
# assertion in its shadow-allocation caching) while differentiating
# SpecialFunctions' pure-Julia `logabsgamma` port inside lazily evaluated
# branches of non-inlined functions differentiated together, one inside a loop
# — the shape every guarded `logpdf` plus an observation plate produces
# (`benchmark/repro_enzyme_lgamma_branch.jl`, Enzyme + SpecialFunctions only).
# With a rule the owned entry points are primitives for Enzyme, so that body is
# never differentiated; the derivatives are exact (`digamma`). Reverse mode
# only: forward mode differentiates the bodies and never failed.
module ReactiveKernelsDistributionKernelsEnzymeExt

using Enzyme: Const, Active
import Enzyme.EnzymeRules
using Enzyme.EnzymeRules: RevConfig, AugmentedReturn, needs_primal
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma, logbeta
using SpecialFunctions: digamma

const _F64 = Union{Const{Float64},Active{Float64}}

# loggamma(x): d/dx = digamma(x)
function EnzymeRules.augmented_primal(
        config::RevConfig, func::Const{typeof(loggamma)},
        ::Type{<:Union{Const,Active}}, x::_F64)
    AugmentedReturn(needs_primal(config) ? func.val(x.val) : nothing,
                    nothing, nothing)
end
EnzymeRules.reverse(
        ::RevConfig, ::Const{typeof(loggamma)}, dret::Active, tape,
        x::Active{Float64}) = (dret.val * digamma(x.val),)
EnzymeRules.reverse(
        ::RevConfig, ::Const{typeof(loggamma)}, dret::Type{<:Const}, tape,
        x::Active{Float64}) = (nothing,)
EnzymeRules.reverse(
        ::RevConfig, ::Const{typeof(loggamma)}, dret, tape,
        x::Const{Float64}) = (nothing,)

# logbeta(a, b): ∂/∂a = digamma(a) - digamma(a + b),
#                ∂/∂b = digamma(b) - digamma(a + b)
function EnzymeRules.augmented_primal(
        config::RevConfig, func::Const{typeof(logbeta)},
        ::Type{<:Union{Const,Active}}, a::_F64, b::_F64)
    AugmentedReturn(needs_primal(config) ? func.val(a.val, b.val) : nothing,
                    nothing, nothing)
end
function EnzymeRules.reverse(
        ::RevConfig, ::Const{typeof(logbeta)}, dret::Active, tape,
        a::_F64, b::_F64)
    dsum = digamma(a.val + b.val)
    (a isa Active ? dret.val * (digamma(a.val) - dsum) : nothing,
     b isa Active ? dret.val * (digamma(b.val) - dsum) : nothing)
end
EnzymeRules.reverse(
        ::RevConfig, ::Const{typeof(logbeta)}, dret::Type{<:Const}, tape,
        a::_F64, b::_F64) = (nothing, nothing)

end # module
