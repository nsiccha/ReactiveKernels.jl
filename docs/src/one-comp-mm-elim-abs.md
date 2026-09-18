# Declarative PPL kernel: one-compartment Michaelis–Menten

This is the posteriordb `one_comp_mm_elim_abs` model (posterior
`one_comp_mm_elim_abs-one_comp_mm_elim_abs`) on the full real drug
concentration dataset. It is the lane's only stiff example: a one-compartment
pharmacokinetic system with first-order absorption and Michaelis–Menten
elimination, integrated with Stan's unconfigured adaptive BDF call. The ODE is

```math
\begin{aligned}
\dot C &= \mathrm{dose}(t) - \frac{V_m}{V}\frac{C}{K_m + C}, &
\mathrm{dose}(t) &= \begin{cases}
e^{-k_a t} D k_a / V & t > 0, \\
0 & t = 0.
\end{cases}
\end{aligned}
```

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/one_comp_mm_elim_abs.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/one_comp_mm_elim_abs.jl).
In Stan semantics, the positive parameters `k_a`, `K_m`, `V_m`, and `sigma`
enter through exponentials, and the unconstrained posterior adds their four
log Jacobians. The dose and compartment volume arrive as Stan real data
`x_r = {D, V}`; they are named module constants captured outside AD, exactly
like the prebuilt problem itself, because only the initial state and theta can
pass through `solve`'s active `u0`/`p` keywords.

The unconfigured Stan BDF defaults map to explicit `FBDF()` controls
`reltol = 1e-10`, `abstol = 1e-10`, and `maxiters = 100_000_000`. FBDF and
Stan's CVODES BDF implementation are different adaptive integrators even at
the same controls; acceptance therefore reports the per-model BridgeStan
discrepancy rather than silently widening a tolerance.

The solver boundary is deliberately authored for ordinary native AD:

1. `ODEProblem` is constructed once outside the differentiated model callable.
2. The bound raw initial concentration and transformed theta vector are
   supplied to `solve` as the `u0` and `p` keyword values.
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
    @__MODULE__, :OneCompMMElimAbsExample, :ONE_COMP_MM_ELIM_ABS_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_one_comp_mm_elim_abs!,
)
```

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.OneCompMMElimAbsExample.demo()'
```
