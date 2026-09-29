# Native varying-source PK/PD cell

`ReactiveKernelsPPL` composes the Gamma-transit/two-compartment PK response,
the normalized dose/concentration effectiveness GP, and piecewise-exact PBMC
and CSF indirect-response updates. The native emitter keeps one batched call;
subject, dose, basis and grid traversal remain runtime loops. Native Enzyme
reverse uses ordinary Julia math and the existing generated transit rule.

Declare the raw schedule inside an `@rkppl` model:

```julia
vs = varyingsource_pkpd_schedule(
    obs = (:subject, :time, :assay),
    dose = (:dose_subject, :dose_time, :amount, :treatment_key),
    discretization = :discretization_lags,
)
```

Assay codes are `1` for PK, `2` for PBMC and `3` for CSF. Subjects cover
`1:n`. Dose times must be nondecreasing within each subject and amounts must
be positive. Equal-time doses remain distinct. PBMC and CSF observation times
must strictly increase within each subject and assay; PK times may repeat.
The raw discretization column contains nonnegative, nondecreasing cumulative
lags from dosing. Treatment keys combine vessel and diet; binding renumbers
them by first appearance within each subject.

Binding owns both PD boundary/midpoint grids, their shared PK reference grid,
lag indices and observation maps. A dose-free subject measured only at its
initial PD state can have an empty PK reference grid. An absent assay skips
its PD and placebo evaluation. Rebinding rebuilds the grids from the new raw
columns, and `validate_data` rejects altered materialized products. A grid
that cannot advance with the supplied discretization lags is rejected.

Inside the grouped plate, the full cell takes 31 arguments:

```julia
reads = varyingsource_pkpd_read_locs(
    vs, dose_log_rate, dose_log_mode, dose_log_F,
    gp_unit_weights, dose_slope, conc_slope, dose_scale, conc_scale, eff_scale,
    placebo_unit_weights, placebo_length_scale, placebo_sd,
    csf_unit_weights, csf_length_scale, csf_sd, placebo_lo, placebo_hi,
    log_Vc, log_k10, log_k12, log_k21,
    log_baseline_pbmc, log_kout, log_theta1_pbmc, log_theta2_pbmc,
    log_baseline_csf, log_theta1_csf, log_theta2_csf,
    log_absorption_rate, log_absorption_mode,
)
mu = reads[vs.obs_map]
```

The three dose modifiers accept scalars or vectors in original dose-row order.
Dose predictors may contain continuous, factor, offset and monotonic diet
terms. The 13 subject log-parameter predictors use subject rows. A predictor
used on both axes is rejected; split it into separate predictors. As in the
[PK slice](varyingsource-pk.md), treatment-map value `j` selects source column
`j` among the first `maximum(treatment_map)` dose columns, preserving the
original program's selection even when that treatment first occurs later.

The three innovation vectors are separate ports. The effectiveness vector is
square and column-major, with at least `2×2` entries. Each placebo vector has
at least two entries. They can be bound data or concrete `:vector_normal`
`VectorParameter` declarations in a `StructuralPlan`. Length scales and
amplitudes are positive constrained values. `placebo_lo` and `placebo_hi`
define the raw time domain; the placebo basis clamps to this domain before
mapping to `[-1,1]` with padding `L=1.5`.

These transforms do not add priors. The original model's GP length-scale
prior has lower bound `(6/pi)*sqrt(log(100)/(k^2-1))` and upper bound `2`, so
its effectiveness prior requires `k ≥ 4`, although the mathematical transform
admits `k = 2`. The cell does not select a statistical model or its priors.

Centered correlated subject effects use the existing varying-draws surface:

```julia
d ~ varying_draws(subject, [1, 1]; centered=true, eta=2., sd=Exponential(0.6666666666666666))
r_vc ~ varying_slice(d, 1:1)
r_k10 ~ varying_slice(d, 2:2)
```

This samples `b_flat_<group>` directly under a multivariate Gaussian with
Cholesky factor `Diagonal(tau)*L`. The flattening order is group first, margin
second. Julia's `Exponential` argument is a scale: `2/3` corresponds to Stan
rate `1.5`. `constrain` returns the direct `b_<group>` matrix. Centered draws
require plain grouping; multi-membership and stratified grouping are rejected.

For assay-specific additive/proportional observation scales, scalar model
parameters can be gathered by the raw assay column inside the plate:

```julia
s_add = [s_add_pk, s_add_pbmc, s_add_csf][assay]
s_prop = [s_prop_pk, s_prop_pbmc, s_prop_csf][assay]
dv .~ CensoredAddpropnormal.(mu, s_add, s_prop, lloq)
```

The scale is `sqrt.(s_add.^2 .+ (mu .* s_prop).^2)`. Values at or below LLOQ
take the log-CDF arm, and values above LLOQ take the density arm. Raw
observation/LLOQ columns suffice; no censoring flag is required. Gather entries
must be numeric literals or scalar model names, and binding checks integer
assay indices and their range.

This consumer supports native execution. Reactant execution is explicitly
rejected. The emitter binds `series_rtol=1e-15, watson_terms=8`; Watson
truncation error is separate from the adaptive-series tolerance. Standalone
math callers can bind different PK controls with `prepare_varyingsource_pkpd`.
The existing [17-coordinate PK benchmark](https://github.com/nsiccha/ReactiveKernels.jl/tree/main/benchmark/varyingsource_pk) is a
PK slice measurement; it does not measure this full statistical model.
The [full posterior benchmark](https://github.com/nsiccha/ReactiveKernels.jl/tree/main/benchmark/varyingsource_pkpd)
records the complete synthetic model, coordinate map, precision checks and
timings at 232 and 583 coordinates.
