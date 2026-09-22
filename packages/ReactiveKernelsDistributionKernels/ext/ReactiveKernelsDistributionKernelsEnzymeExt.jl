# Reverse-mode Enzyme rules for the special functions the distribution
# sources call inside lazy support guards.
#
# `loggamma(::Float64)` reaches Enzyme as the C call `lgamma_r`, and Enzyme's
# built-in handling of that call asserts (LLVM `cast<Instruction>` in
# `handleKnownCallDerivatives`) when the call sits inside a lazily evaluated
# branch of a non-inlined function that is differentiated together with a
# second such branch — the shape every `logpdf` with a support guard plus a
# plate over observations produces (`benchmark/repro_enzyme_lgamma_branch.jl`
# is the Enzyme-only reproducer).  Giving `loggamma` a Julia-level derivative
# (`digamma`) makes it a primitive for Enzyme, so the C-level handler is never
# reached.  `logbeta` reaches the same call through `logabsgamma`, so both
# `loggamma` and `logabsgamma` (value and sign) get rules.
module ReactiveKernelsDistributionKernelsEnzymeExt

using Enzyme: Const, Active
import Enzyme.EnzymeRules
using Enzyme.EnzymeRules: RevConfig, AugmentedReturn, needs_primal
using SpecialFunctions: loggamma, logabsgamma, digamma

for F in (typeof(loggamma), typeof(logabsgamma))
    @eval begin
        function EnzymeRules.augmented_primal(
                config::RevConfig, func::Const{$F},
                ::Type{<:Union{Const,Active}},
                x::Union{Const{Float64},Active{Float64}})
            AugmentedReturn(needs_primal(config) ? func.val(x.val) : nothing,
                            nothing, nothing)
        end
        EnzymeRules.reverse(
                ::RevConfig, ::Const{$F}, dret::Type{<:Const}, tape,
                x::Active{Float64}) = (nothing,)
        EnzymeRules.reverse(
                ::RevConfig, ::Const{$F}, dret, tape, x::Const{Float64}) =
            (nothing,)
    end
end

EnzymeRules.reverse(
        ::RevConfig, ::Const{typeof(loggamma)}, dret::Active, tape,
        x::Active{Float64}) = (dret.val * digamma(x.val),)
# `logabsgamma` returns `(value, sign)`; only the value carries a cotangent.
EnzymeRules.reverse(
        ::RevConfig, ::Const{typeof(logabsgamma)}, dret::Active, tape,
        x::Active{Float64}) = (dret.val[1] * digamma(x.val),)

end # module
