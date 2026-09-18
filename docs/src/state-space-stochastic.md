# Declarative PPL kernel: state_space_stochastic (structural DLM)

This example ports the
`state_space_stochastic_level_stochastic_seasonal` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`uk_drivers-state_space_stochastic_level_stochastic_seasonal`) into the
declarative-`@kernel` style — a structural time-series (dynamic linear model) of
UK monthly driver deaths, authored on the FULL real data (`n = 192` months)
loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/state_space_stochastic.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/state_space_stochastic.jl).

A stochastic (random-walk) LEVEL `mu`, a stochastic SEASONAL component whose
12-month contributions sum to ≈ 0, and two regression covariates:

```math
\begin{aligned}
\mu_t &\sim \operatorname{Normal}(\mu_{t-1}, \sigma_2), \quad t = 2..n, \\
\text{seasonal}_t &\sim \operatorname{Normal}\!\Big(-\!\!\sum_{k=t-11}^{t-1} \text{seasonal}_k,\; \sigma_1\Big), \quad t = 12..n, \\
\hat{y} &= \mu + \beta x + \lambda w, \qquad y_i \sim \operatorname{Normal}(\hat{y}_i + \text{seasonal}_i, \sigma_3), \\
\sigma &\sim \operatorname{Student\_t}(4, 0, 1).
\end{aligned}
```

The genuine sequential structure lives in the transition priors — the level's
random walk and the seasonal window sum, evaluated over the free state vectors —
so they are authored as vectorized in-graph reductions (an adjacent-slice plate
for the level, a banded 0/1 matvec for the trailing-window seasonal sum). The
unconstrained vector is `(mu[1..n], seasonal[1..n], beta, lambda, sigma[1..3])`:
the level `mu` is a **bounded vector** with data-directed bounds
`mean(y) ± 3·sd(y)` (per-element logit transform, summed log-Jacobian), and the
three scales are a `positive_ordered[3]` (`σ₁ = exp(z₁)`, `σₖ = σₖ₋₁ + exp(zₖ)`;
log-Jacobian `Σ zₖ`).

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :StateSpaceStochasticExample, :STATE_SPACE_STOCHASTIC_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_state_space_stochastic!,
)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/forecast_batch_gate.jl`
`@compile`s both the primal and the gradient and asserts they match the native
evaluation and the reference `.stan` (via BridgeStan). The data-bounded logit
level transform, the `positive_ordered` cumulative-exp scales, and the
trailing-window seasonal matvec all lower cleanly. Parity is asserted at the
gate's tested reference-valid probe points — six native probes per case, Reactant
axes at the first probe — on the gate's pinned `benchmark/all80-env` toolchain
(BridgeStan 2.9 / Stan 2.39, Reactant, Enzyme and DifferentiationInterface as
resolved there); it is a tested-point result, not a claim over every finite
input.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.StateSpaceStochasticExample.demo()'
```
