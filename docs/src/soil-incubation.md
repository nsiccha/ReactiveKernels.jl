# Declarative PPL kernel: soil incubation two-pool carbon

This is the posteriordb `soil_incubation` model (posterior
`soil_carbon-soil_incubation`) on the full real evolved-CO2 dataset. It tracks
carbon in two pools with feedback and no inputs, integrated with Stan's
unconfigured adaptive RK45 call. The ODE is

```math
\begin{aligned}
\dot C_1 &= -k_1 C_1 + \alpha_{12} k_2 C_2, \\
\dot C_2 &= -k_2 C_2 + \alpha_{21} k_1 C_1,
\end{aligned}
```

with the initial state partitioned by `C(0) = (\gamma C_{\mathrm{tot}},
(1-\gamma) C_{\mathrm{tot}})` and the observed evolved CO2
`eCO_2(t) = C_{\mathrm{tot}} - (C_1(t) + C_2(t))`.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/soil_incubation.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/soil_incubation.jl).
In Stan semantics, the pool rates `k1`, `k2`, `alpha21`, `alpha12`, and `sigma`
enter through exponentials while the partitioning coefficient `gamma` in
`[0, 1]` enters through a logistic transform with its exact log Jacobian. The
data file additionally carries `eCO2sd`, which the Stan data block does not
declare, so it is not a model input.

The unconfigured Stan RK45 defaults map to explicit `DP5()` controls
`reltol = 1e-6`, `abstol = 1e-6`, and `maxiters = 1_000_000`. DP5 and Stan's
Boost dopri5 implementation are different adaptive integrators even at the
same controls; acceptance therefore reports the per-model BridgeStan
discrepancy rather than silently widening a tolerance.

The solver boundary is deliberately authored for ordinary native AD:

1. `ODEProblem` is constructed once outside the differentiated model callable.
2. Both the active initial pool state (the in-graph gamma partition) and the
   transformed theta vector are supplied to `solve` as the `u0` and `p`
   keyword values.
3. The module never calls `remake` inside AD and never enables Enzyme runtime
   activity or hand-writes sensitivities.

This shape matters because constructing or remaking a problem inside AD stores
active values in the problem object and can force Enzyme runtime-activity
analysis. The natural adaptive call remains a foreign opaque node, so loaded
Reactant behavior is a separate acceptance axis rather than a claim made on
this page.

The panel below shows three views of the model: **Raw input** (the exact
executed source), a readable **Generated kernel**, and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :SoilIncubationExample, :SOIL_INCUBATION_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_soil_incubation!,
)
```

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.SoilIncubationExample.demo()'
```
