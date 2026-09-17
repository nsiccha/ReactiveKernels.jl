# Declarative PPL kernel: Lotka–Volterra adaptive ODE

This is the posteriordb `lotka_volterra` model (posterior
`hudson_lynx_hare-lotka_volterra`) on the full Hudson Bay lynx–and–hare
dataset. It is the first dynamical-system example in the PPL corpus: the two
state trajectories are produced by the model's explicitly configured adaptive
RK45 call, not by a fixed-step rewrite.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/lotka_volterra.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/lotka_volterra.jl).
In Stan semantics, the positive vectors `theta`, `z_init`, and `sigma` enter
through exponentials, and the unconstrained posterior adds their eight log
Jacobians. The ODE is

```math
\begin{aligned}
\dot z_1 &= (\alpha-\beta z_2)z_1, &
\dot z_2 &= (-\gamma+\delta z_1)z_2,
\end{aligned}
```

integrated from time zero with `reltol = 1e-5`, `abstol = 1e-3`, and
`maxiters = 500`. The Julia module uses `DP5()` for this natural foreign-solver
node. DP5 and Stan's Boost dopri5 implementation are different adaptive
integrators even at the same controls; acceptance therefore reports the
per-model BridgeStan discrepancy rather than silently widening a tolerance.

The solver boundary is deliberately authored for ordinary native AD:

1. `ODEProblem` is constructed once outside the differentiated model callable.
2. Active constrained `z_init` and `theta` vectors are supplied to `solve` as
   the `u0` and `p` keyword values.
3. The module never calls `remake` inside AD and never enables Enzyme runtime
   activity or hand-writes sensitivities.

This shape matters because constructing or remaking a problem inside AD stores
active values in the problem object and can force Enzyme runtime-activity
analysis. The natural adaptive call remains a foreign opaque node. Accepted
evidence for this node: native primal/value only. Ordinary-Reverse gradient
is unsupported pending core snag `plain-enzyme-rev-3dc5d563`, and compiled
Reactant primal/gradient is unsupported pending the separate Reactant
survey; either resolution may change this boundary.

The panel below shows three views of the model: **Raw input** (the exact
executed source), a readable **Generated kernel**, and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :LotkaVolterraExample, :LOTKA_VOLTERRA_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_lotka_volterra!,
)
```

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.LotkaVolterraExample.demo()'
```
