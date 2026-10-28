# Standalone CPU reproducer: only Reactant, Enzyme and
# DifferentiationInterface are required (zero ReactiveKernels frames).
#
# A Reactant-compiled `value_and_gradient` where `s = exp(u[2])` feeds
# every lane through BOTH `./ s` and `-log(s)`, while a `u[1]`-dependent
# term sits outside the reduction, silently miscompiles under the default
# pipeline: the second gradient coordinate reads exactly `+(n-1)` high —
# each lane's `-1` adjoint from `-log(s)` is counted ONCE instead of `n`
# times. The primal is correct, and the SAME trace through
# `optimize = :only_enzyme` is value- and gradient-correct, so the trace
# and Enzyme's reverse are right and a default-pipeline optimization pass
# breaks them. `optimize = :no_slice_slice` still miscompiles: this is a
# DIFFERENT pass than the §7j `slice_slice` pattern. Recorded on strato2,
# Reactant 0.2.285 / Enzyme 0.13.204, Julia 1.10.11, 2026-09-23:
#   native reference:  g = [10.526941513649327, 10.606426725904297]
#   default pipeline:  g = [10.526941513649327, 12.606426725904297]  (WRONG)
#   :only_enzyme:      g matches the reference to 1e-15
# Bisected trigger (every other shape compiles correctly): likelihood
# alone (any `mu` construction), likelihood + constant, likelihood +
# `u[2]`-only term, scalar-only, and broadcast-without-`z` all pass; a
# second-slice (`u[1]`) term — even a linear `+ a` — is REQUIRED, as is
# the `./ s` lane path alongside the `-log(s)` path.
using Reactant, Enzyme
import DifferentiationInterface: AutoEnzyme, value_and_gradient

const _REPRO_Y = [1.0, 2.0, 1.5]

function _repro_posterior(u)
    a = sum(view(u, 1:1))
    s = exp(sum(view(u, 2:2)))
    mu = fill(a, 3)
    z = (_REPRO_Y .- mu) ./ s
    return sum(-z .* z .- log(s)) + a
end

const _REPRO_BE = AutoEnzyme(; mode = Enzyme.Reverse)
const _REPRO_U0 = [0.2, -0.1]

ref_val, ref_g = value_and_gradient(_repro_posterior, _REPRO_BE, _REPRO_U0)
fn = (t,) -> value_and_gradient(_repro_posterior, _REPRO_BE, t)
traced = Reactant.to_rarray(_REPRO_U0)

# The trace + Enzyme reverse are correct ...
oe_val, oe_g = Reactant.compile(fn, (traced,); sync = true,
    optimize = :only_enzyme)(Reactant.to_rarray(_REPRO_U0))
@assert Float64(oe_val) ≈ ref_val
@assert Array(oe_g) ≈ ref_g

# ... and the default pipeline miscompiles them (fails here).
got_val, got_g = Reactant.compile(fn, (traced,); sync = true)(
    Reactant.to_rarray(_REPRO_U0))
@assert Float64(got_val) ≈ ref_val   # the primal is exact
@assert Array(got_g) ≈ ref_g         # the gradient is +2.0 high on coord 2
