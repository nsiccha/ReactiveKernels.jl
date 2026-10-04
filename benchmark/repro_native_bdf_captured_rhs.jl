# Capability reproducer for the current owned BDF adapter, not an acceptance
# substitution: the final independent derivative assertion currently fails.
# Run in an environment declaring all five packages below.
using ReactiveKernels, OrdinaryDiffEqBDF, SciMLBase, SciMLSensitivity, Enzyme, Test

function captured_rhs_loss(p)
    rhs = (t, u) -> -p[1] .* u
    sum(rk_ode_bdf_tol(rhs, [1.0], 0.0, [0.4, 1.6], 1e-8, 1e-8, 10000))
end

p, dp = [0.7], zeros(1)
times = [0.4, 1.6]
@test captured_rhs_loss(p) ≈ sum(exp.(-p[1] .* times)) rtol=2e-6
Enzyme.autodiff(Enzyme.Reverse, captured_rhs_loss, Enzyme.Active, Enzyme.Duplicated(p, dp))
expected = sum(-times .* exp.(-p[1] .* times))
@test p == [0.7]
println("Captured RHS parameter: observed=", dp[1], " analytic=", expected)
@test dp[1] ≈ expected rtol=3e-6
