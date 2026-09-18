# Declarative PPL kernel: SIR with environmental bacteria

This is the posteriordb `sir` model (posterior `sir-sir`) on the full real
outbreak dataset. It extends the susceptible–infected–removed state with a
bacterial environmental compartment and uses Stan's unconfigured adaptive RK45
call.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/sir.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/sir.jl).
The fixed transformed-data half-saturation constant is `kappa = 10^6`. The ODE
is

```math
\begin{aligned}
\dot y_1 &= -\frac{\beta y_4}{y_4+\kappa}y_1, \\
\dot y_2 &= \frac{\beta y_4}{y_4+\kappa}y_1-\gamma y_2, \\
\dot y_3 &= \gamma y_2, \\
\dot y_4 &= \xi y_2-\delta y_4.
\end{aligned}
```

The four positive parameters use Stan's exponential transforms and Jacobians.
The Poisson observations are consecutive decrements of the susceptible state:
the first is `y0[1] - y[1,1]`, and observation `n > 1` uses
`y[n-1,1] - y[n,1]`. Bacterial measurements have a log-normal likelihood with
standard deviation `0.15`. No decrement is clamped or rewritten.

The natural foreign-solver node maps Stan's BridgeStan 2.9 / Stan 2.39
unconfigured RK45 defaults to explicit Julia `DP5()` controls:

```text
relative tolerance = 1e-6
absolute tolerance = 1e-6
maximum steps      = 1_000_000
```

As with Lotka–Volterra, Julia DP5 and Stan's Boost dopri5 implementation are
different adaptive integrators. Acceptance records the per-model BridgeStan
comparison rather than silently widening a tolerance.

The `ODEProblem` is built once outside the differentiated model callable. The
bound raw initial state and transformed five-value solver parameter vector are
passed to `solve` as `u0` and `p`. The source never calls `remake` inside AD,
enables runtime activity, or hand-authors sensitivities.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :SIRExample, :SIR_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_sir!,
)
```

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.SIRExample.demo()'
```
