# Declarative PPL kernel: accel_splines (brms penalized splines)

This example ports the `accel_splines` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`mcycle_splines-accel_splines`) into the declarative-`@kernel` style — a brms
2.10.0 penalized-spline regression of the motorcycle-acceleration data, with a
spline for BOTH the mean and the (log-linked) residual scale, authored on the
FULL real data (`N = 133`, 38 knots per spline) loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/accel_splines.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/accel_splines.jl).

Each spline enters as a fixed basis-function design matrix (`Zs_*`) times
standardized coefficients `zs_*` scaled by a half-Student-t `sds_*`; the
linear-effect design matrices `Xs_*` carry flat `bs_*`:

```math
\begin{aligned}
\mu &= \text{Intercept} + X_s\, b_s + Z_{s,1,1}\, (sds_{1,1}\, zs_{1,1}), \\
\sigma &= \exp\!\big(\text{Intercept}_\sigma + X_{s,\sigma}\, b_{s,\sigma} + Z_{s,\sigma,1,1}\, (sds_{\sigma,1,1}\, zs_{\sigma,1,1})\big), \\
Y_i &\sim \operatorname{Normal}(\mu_i, \sigma_i), \\
\text{Intercept} &\sim \operatorname{Student\_t}(3, -13, 36), \quad zs_{\cdot} \sim \operatorname{Normal}(0, 1), \\
sds_{\cdot} &\sim \operatorname{Student\_t}(3, 0, 36)\ \text{truncated to } > 0 \ (\text{the } +\log 2 \text{ half-Student-t normalizer}).
\end{aligned}
```

The unconstrained vector is `(Intercept, bs, zs_1_1[1..38], log sds_1_1,
Intercept_sigma, bs_sigma, zs_sigma_1_1[1..38], log sds_sigma_1_1)`; only the two
spline standard deviations carry an exp support transform (summed log-Jacobian).
The mean and log-scale linear predictors are in-graph data→parameter
transformations (design-matrix products). The `prior_only` flag is Stan's data
switch (`if (!prior_only)`): it is validated and
converted once at the raw-data boundary (`ACCEL_PRIOR_ONLY`) and selects the
authoritative posterior node through `accel_splines_posterior_want` — planning
`:prior_only_posterior` does not compute the likelihood recipe at all, exactly
mirroring Stan's skipped branch. The committed gate probes that mode with the
unconstrained `Intercept_sigma` at `-800` (where the deselected likelihood's
σ = exp(-800) would underflow to 0).

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :AccelSplinesExample, :ACCEL_SPLINES_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_accel_splines!,
)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/forecast_batch_gate.jl`
`@compile`s both the primal and the gradient and asserts they match the native
evaluation and the reference `.stan` (via BridgeStan). The design-matrix
products and reused `normal`/`student_t` endpoints all lower cleanly. Parity is
asserted at the gate's tested reference-valid probe points — six native probes
per case, plus the alternate-flag (`prior_only = 1`) case with its
`Intercept_sigma = -800` stress probe, with the Reactant axes at the first probe
and the stress probe — on the gate's pinned `benchmark/all80-env` toolchain
(BridgeStan 2.9 / Stan 2.39, Reactant, Enzyme and DifferentiationInterface as
resolved there); it is a tested-point result, not a claim over every finite
input.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.AccelSplinesExample.demo()'
```
