# Standalone CPU reproducer: only Enzyme and SpecialFunctions are required.
#
# Enzyme reverse mode aborts the Julia process (an LLVM assertion, not a
# catchable error) when two lazily evaluated branches around `lgamma_r`-backed
# calls (`loggamma`, `logbeta`) sit in non-inlined functions that are
# differentiated together, one of them inside a loop:
#   Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"'
#   cast<llvm::Instruction, llvm::Value> at Enzyme/CallDerivatives.cpp
#   handleKnownCallDerivatives ... recursivelyHandleSubfunction
# Each piece alone, the fully inlined shape, and a branchless `ifelse` select
# all differentiate fine.  This is the shape every guarded `logpdf` plus an
# observation plate produces in ReactiveKernelsDistributionKernels, which
# therefore registers Julia-level reverse rules for `loggamma` and
# `logabsgamma` (its Enzyme extension); with those rules the gradient below
# matches central differences.  Recorded on strato2, Enzyme 0.13.204 /
# SpecialFunctions 2, Julia 1.10.11, 2026-09-23.
using Enzyme, SpecialFunctions

@noinline prior_op(x, a, b) =
    (x > 0) & (x < 1) ? (a - 1) * log(x) + (b - 1) * log1p(-x) - logbeta(a, b) : -Inf
@noinline cell_op(k, n, logp, log1mp) =
    (k >= 0) & (k <= n) ?
        (loggamma(n + 1.0) - loggamma(k + 1.0) - loggamma(n - k + 1.0) +
         k * logp + (n - k) * log1mp) : -Inf
function density(x, ks, ns)
    total = prior_op(x, 2.0, 2.0)
    logp = log(x); log1mp = log1p(-x)
    for i in eachindex(ks)
        total += cell_op(ks[i], ns[i], logp, log1mp)
    end
    total
end

ks = [3, 5, 2]; ns = [10, 10, 10]
h = 1e-6
reference = (density(0.3 + h, ks, ns) - density(0.3 - h, ks, ns)) / 2h
@assert isapprox(reference, 20 / 3; rtol = 1e-6)
println("central difference: ", reference)
println("Enzyme: ", Enzyme.gradient(Reverse, x -> density(x, ks, ns), 0.3))   # aborts here
