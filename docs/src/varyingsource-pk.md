# Varying-source PK slice

`ReactiveKernelsPPL` supplies the PK component of Bruno's `varyingsource3`
twin at source checkpoint `896137dd`. A Gamma transit input feeds a linear
two-compartment system. Each treatment reuses one unit response; doses then
accumulate in order, with a normalized two-dimensional GP adjusting each dose
from its amount and the concentration produced by earlier doses.

The native cell is `varyingsource_pk_concentration`. Its subject-local data
are a reference-grid size, dose amounts, treatment map, unique lags, dose-major
lag indices, and dosing-time indices. The remaining inputs are per-dose log
rate/mode/F modifiers, an effectiveness surface from
`varyingsource_effectiveness(weights, dose_slope, conc_slope)`, and six log
parameters: central volume, three microconstants, absorption rate, and
absorption mode. `varyingsource_pk_locs` gathers observation indices from this
grid, preserving duplicates and their order.

The GP basis has padding 1.5. Dose uses the log domain 10000–200000 without
clamping; concentration is clamped to 0–2000 before its log1p map. The surface
is normalized at dose 10000 and concentration zero. Coefficients supplied to
this slice are already scaled GP weights, rather than the full model's
standardized innovations and length-scale hyperparameters.

## Grouped emission

Declare raw observation and dose columns with:

```julia
vs = varyingsource_pk_schedule(obs = (:subject, :time),
    dose = (:dose_subject, :dose_time, :dose_amount, :treatment_key))
```

`treatment_key` is the combined treatment/diet identifier (for the twin,
`treatment + 100*diet`). Binding renumbers it by first appearance within each
subject. Observation subjects must cover `1:n`; dose subjects must refer to
them. Dose times must be nondecreasing within a subject, and dose amounts must
be finite and positive. Equal-time doses remain separate. Subject axes may be
interleaved, and observation times may repeat or arrive in any order.

A grouped cell calls:

```julia
@plate pk_locations for s in 1:kernel_nsub_pk_locations
    reads = varyingsource_pk_read_locs(vs, dose_log_rate, dose_log_mode,
        dose_log_F, gp_weights, dose_slope, conc_slope,
        log_Vc, log_k10, log_k12, log_k21, log_absorption_rate, log_absorption_mode)
    mu = reads[vs.obs_map]
    concentration .~ Normal.(mu, sigma)
    mu
end
```

The six log parameters are outer subject predictors or model scalars.
The three dose modifiers may be scalars, subject predictors, or flat vectors
in the original dose-row order. In-cell dotted arithmetic can form these
vectors from bound dose covariates and model coefficients. This does not yet
extract arbitrary BRM dose-axis predictors. GP slopes are scalars or subject
predictors. `gp_weights` is a shared square coefficient vector in Julia column
order: bound data, or a concrete-size `VectorParameter` with family
`:vector_normal` in the typed plan. The latter prior belongs to that plan; it
does not reproduce the twin's GP innovation prior automatically.

Binding builds separate cumulative ends for reference times, doses, lags, and
the dose-major lag-index products. It preserves original dose-row indices and
builds `obs_map` into the flat reference concentrations. Hand-bound products
are verified by exact rebuilding. Emission produces one call to
`varyingsource_pk_read_locs_over_subjects`; subject, treatment, lag, dose, and
reference traversal remain ordinary runtime loops. Dose-free subjects skip
parameter indexing, GP construction, and unit solves and return zero PK.

Treatment map value `j` selects source column `j`, assembled from dose-column
modifiers `j`. This preserves the deployed twin's selection of the first
`maximum(treatment_map)` source columns, even if treatment `j` first appears
on a later dose. Changing that selection would change the modeled twin.

## Numerical and backend limits

The default cell binds `series_rtol=1e-15` and `watson_terms=8`. The series
tolerance and asymptotic Watson truncation error are separate. These controls
do not establish a production accuracy policy. Direct cell users can bind
different controls with `prepare_varyingsource_pk(; series_rtol, watson_terms)`.
The grouped emitter currently uses the default controls. The
[unit-response comparison](https://github.com/nsiccha/ReactiveKernels.jl/tree/main/benchmark/transit_twocmt)
measures reverse gradients against actual Stan BDF; it does not measure this
complete PK slice or the full posterior. The separate
[emitted PK-slice comparison](https://github.com/nsiccha/ReactiveKernels.jl/tree/main/benchmark/varyingsource_pk)
measures the full 17-coordinate slice density and reverse gradient on public
synthetic 3/30-subject workloads, including GP feedback, likelihood, priors,
and the sigma Jacobian. It records 14.5–50× speedups with gradient differences
below production Stan's measured differences on those cases. It does not
establish full-twin or real-fit performance.

Native Enzyme reverse differentiates the ordinary cell and GP arithmetic.
The transit primitive uses its existing generated mathematical reverse rule.
Reactant execution fails explicitly: compiled transit-rule and sequential-dose
control flow are not established for this cell. The slice does not yet include
PD/placebo grids, full centered hierarchical subject effects, GP prior
transforms, or BRM-side grouped extraction. Those remain full-twin work.
