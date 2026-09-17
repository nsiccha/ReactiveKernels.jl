# Declarative PPL kernel: losscurve_sislob (insurance loss development)

This example ports the `losscurve_sislob` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`loss_curves-losscurve_sislob`) into the declarative-`@kernel` style — a
hierarchical insurance loss-development ("chain-ladder") curve, authored on the
FULL real data (`n_cohort = 10`, `n_time = 10`, `n_data = 55`) loaded through
PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/losscurve_sislob.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/losscurve_sislob.jl).

Each cohort's ultimate loss ratio `LRᵢ` is drawn hierarchically; the expected
loss at development time `t` is `LRᵢ · premiumᵢ · gf(t)`, where the growth factor
`gf(t) ∈ (0, 1)` is a Weibull or log-logistic CDF-shaped curve selected by the
`growthmodel_id` data flag:

```math
\begin{aligned}
\text{gf}(t) &= \begin{cases}
  1 - \exp\!\big(-(t/\theta)^{\omega}\big) & \text{growthmodel\_id} = 1 \ (\text{Weibull}),\\
  t^{\omega} / (t^{\omega} + \theta^{\omega}) & \text{otherwise (log-logistic)},
\end{cases}\\
\mu\_LR \sim \operatorname{Normal}(0, 0.5),&\quad sd\_LR \sim \operatorname{LogNormal}(0, 0.5),\\
LR_i \sim \operatorname{LogNormal}(\mu\_LR, sd\_LR),&\quad \omega,\theta \sim \operatorname{LogNormal}(0, 0.5),\ loss\_sd \sim \operatorname{LogNormal}(0, 0.7),\\
\text{loss}_d &\sim \operatorname{Normal}\!\big(LR_{c_d}\, premium_{c_d}\, \text{gf}(t_d),\; loss\_sd\cdot premium_{c_d}\big).
\end{aligned}
```

The unconstrained vector is `(log ω, log θ, log LR[1..10], μ_LR, log sd_LR,
log loss_sd)`; every positive scale carries a log/exp support transform whose
summed log-Jacobian is added. The growth factor is a genuine in-graph
data→parameter transformation: the `growthmodel_id` flag stays a **live bound
port** and selects Weibull or log-logistic per cell (`ifelse` evaluates both
arms, so binding the flag *selects* the branch — it does not prune the other's
computation). Both arms are authored in an overflow-safe form — `(t/θ)^ω`
inside the exponential for Weibull, and the algebraically equal
`1/(1+(θ/t)^ω)`-style stable form for log-logistic — so the eagerly evaluated
but unselected arm cannot overflow at extreme `ω`; the committed gate probes
exactly that at `ω = 1000`, `θ = max(t)`. Cohort/time indices are integer-array
gathers.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel**, and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :LosscurveSislobExample, :LOSSCURVE_SISLOB_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_losscurve_sislob!,
)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/forecast_batch_gate.jl`
`@compile`s both the primal and the gradient and asserts they match the native
evaluation and the reference `.stan` (via BridgeStan). The reused `normal` and
`lognormal` endpoints and the integer-array gathers all lower cleanly. Parity is
asserted at the gate's tested reference-valid probe points — six native probes
per case plus the `ω = 1000` inactive-branch stress probe, with the Reactant
axes at the first probe and the stress probe — on the gate's pinned
`benchmark/all80-env` toolchain (BridgeStan 2.9 / Stan 2.39, Reactant, Enzyme
and DifferentiationInterface as resolved there); it is a tested-point result,
not a claim over every finite input.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.LosscurveSislobExample.demo()'
```
