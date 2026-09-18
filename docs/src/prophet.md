# Declarative PPL kernel: prophet (piecewise-trend forecasting, linear trend)

This example ports the `prophet` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`rstan_downloads-prophet`) into the declarative-`@kernel` style — Facebook
Prophet's piecewise-trend forecasting model, authored on the FULL real data
(`T = 1169`, `K = 34`, `S = 25`) loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/prophet.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/prophet.jl).

!!! note "Linear trend only"
    Stan's `prophet` supports a **linear** (`trend_indicator == 0`) or a
    **logistic** (`== 1`) trend. The authoritative `rstan_downloads` data selects
    the LINEAR trend, and this graph implements **only** that mode.
    `build_prophet_graph` validates the trend flag and throws an explicit
    unsupported-mode error for the logistic trend rather than returning a wrong
    or sentinel density. The logistic trend is unimplemented and unverified here
    (a separately-scoped item); its natural `logistic_gamma` recurrence is
    documented as Stan pseudocode in
    `ProphetExample.PROPHET_LOGISTIC_TREND_REFERENCE_SOURCE` and now authors via
    RK's lockstep multi-sequence [`scan`](scan.md).

A piecewise trend (`k` base rate plus per-changepoint adjustments `delta`) with
additive/multiplicative seasonality regressors:

```math
\begin{aligned}
A_{ij} &= \mathbf{1}\{t_i \ge t\_change_j\} \quad (\text{data-only changepoint incidence}), \\
\text{trend} &= (k + A\,\delta)\odot t + \big(m + A\,(-t\_change \odot \delta)\big), \\
\mu &= \text{trend} \odot \big(1 + X(\beta\odot s_m)\big) + X(\beta\odot s_a), \qquad y_i \sim \operatorname{Normal}(\mu_i, \sigma\_obs), \\
k, m &\sim \operatorname{Normal}(0, 5), \quad \delta \sim \operatorname{DoubleExponential}(0, \tau), \\
\sigma\_obs &\sim \operatorname{Normal}(0, 0.5), \quad \beta_j \sim \operatorname{Normal}(0, sigmas[j]).
\end{aligned}
```

The unconstrained vector is `(k, m, delta[1..25], log sigma_obs, beta[1..34])`;
only `sigma_obs` carries an exp support transform (log-Jacobian). The changepoint
incidence matrix `A`, the linear trend, and the additive/multiplicative
seasonality are in-graph data→parameter transformations. The Laplace
(double-exponential) changepoint prior and the reused `normal` endpoint are
imported from `ReactiveKernelsDistributionKernels`.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :ProphetExample, :PROPHET_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_prophet!,
)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/forecast_batch_gate.jl`
`@compile`s both the primal and the gradient and asserts they match the native
evaluation and the reference `.stan` (via BridgeStan). The changepoint-incidence
comparison-mask matvec, the trend/seasonality, and the reused `normal`/`laplace`
endpoints all lower cleanly. Parity is asserted at the gate's tested
reference-valid probe points — six native probes per case, Reactant axes at the
first probe — on the gate's pinned `benchmark/all80-env` toolchain (BridgeStan
2.9 / Stan 2.39, Reactant, Enzyme and DifferentiationInterface as resolved
there); it is a tested-point result, not a claim over every finite input.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.ProphetExample.demo()'
```
