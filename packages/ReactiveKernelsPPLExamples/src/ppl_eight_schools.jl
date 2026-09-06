module PPLEightSchoolsExample

# EXPERIMENTAL — NOT REVIEWED / NOT APPROVED. This module exists only so the
# posteriordb benchmark 4th side (ReactiveKernels:ppl:benchmark) can `import` a
# stable builder for the `@ppl`-authored eight-schools model instead of copying
# test-local code. Like `@ppl`/`PPLGibbs` it is deliberately NOT exported and is
# NOT part of the consumer API (`reactivekernels-use` does not mention it). Reach
# it only via the fully qualified path
# `ReactiveKernelsPPLExamples.PPLEightSchoolsExample.build_ppl_eight_schools`.
# Do not depend on it until `@ppl` is reviewed and approved.

using ReactiveKernels
using ..ReactiveKernelsPPLExamples.PPLMacro: @ppl
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

# Centered eight-schools authored in the experimental `@ppl` front-end. The data
# ports are named `observations` / `observation_scales` to match the hand-authored
# EightSchoolsExample kernel exactly, and the effects vector uses the LITERAL size
# 8 (no size port), so this KernelSpec is queried with the identical
# have/want/bound surface and the identical q = [μ, log_τ, θ…] packing as
# `build_eight_schools_graph()`:
#
#   μ ~ Normal(0, 5); τ ~ HalfCauchy(0, 5); θⱼ ~ Normal(μ, τ); yⱼ ~ Normal(θⱼ, σⱼ)
#
# `positive(cauchy(…))` supplies the half-Cauchy (+log 2 normalization and
# log/exp transform with Jacobian log_τ). Density-equivalent to the hand-authored
# kernel to machine precision — see test/test_ppl_posteriordb_parity.jl.
@ppl _ppl_eight_schools(observations::Vector{Float64},
                        observation_scales::Vector{Float64}) = begin
    mu ~ normal(0.0, 5.0)
    tau ~ positive(cauchy(0.0, 5.0))
    theta::vector[8] ~ normal(mu, tau)
    observations ~ normal(theta, observation_scales)
end

"""
    build_ppl_eight_schools() -> ReactiveKernels.KernelSpec

EXPERIMENTAL. Build the centered eight-schools model authored with the
experimental `@ppl` front-end, returned as a `ReactiveKernels.KernelSpec` queried
exactly like the hand-authored `EightSchoolsExample.build_eight_schools_graph()`:

```julia
prepare(build_ppl_eight_schools();
        have = (:unconstrained, :observations, :observation_scales),
        want = :posterior)(q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)
```

with `q = [μ, log_τ, θ…]` (10 coordinates). Density-equivalent to the
hand-authored kernel to machine precision (regression: `test_ppl_posteriordb_parity.jl`).

Unexported and not part of the consumer API — for the benchmark 4th side and
parity checks only. A fresh, independent graph is returned on each call.
"""
build_ppl_eight_schools() = compose(_ppl_eight_schools)

end # module PPLEightSchoolsExample
