# Ordinary reverse-mode gradients through the NATIVE solve: plain
# `AutoEnzyme(mode=Enzyme.Reverse)` with no `runtime_activity`, no priming,
# and no hand-written sensitivity rule. Checks against central differences.
#
# Two Enzyme sharp edges shape this file (both probed, not assumed):
# - the losses call the top-level `solve_ode` directly: a closure with
#   kwargs (`solve_p(p; kw...)`) leaves a generic call Enzyme cannot prove
#   readonly, while the identical top-level kwargs call specializes fine;
# - the solver stores fresh state vectors without `copy` (safe by the
#   functional core's construction): `copy` of an active vector either
#   aborts reverse mode or silently zeroes the gradient.
using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme

const NATIVE_REVERSE_BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

@testset "native ordinary reverse gradients" begin
    u0 = (1.0, 2.0)
    tspan = (0.0, 3.0)
    p0 = (0.5, 1.5)
    saveat = (1.0, 2.0)

    loss_endpoint(p) = sum(solve_ode(exponential_decay, [u0...], tspan,
        Tsit5(); p=p, abstol=1e-10, reltol=1e-8).u[end])
    loss_saveat(p) = sum(sum, solve_ode(exponential_decay, [u0...], tspan,
        Tsit5(); p=p, abstol=1e-10, reltol=1e-8, saveat=saveat).u)
    loss_u0(x) = sum(solve_ode(exponential_decay, x, tspan, Tsit5();
        p=[p0...], abstol=1e-10, reltol=1e-8).u[end])

    @test gradient(loss_endpoint, NATIVE_REVERSE_BACKEND, [p0...]) ≈
          central_gradient(loss_endpoint, [p0...]) atol = 1e-6
    @test gradient(loss_saveat, NATIVE_REVERSE_BACKEND, [p0...]) ≈
          central_gradient(loss_saveat, [p0...]) atol = 1e-6
    @test gradient(loss_u0, NATIVE_REVERSE_BACKEND, [u0...]) ≈
          central_gradient(loss_u0, [u0...]) atol = 1e-6
end
