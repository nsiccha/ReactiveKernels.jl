"""
    prepare_filter_smc(filter_target, calibration::SMCCarriedReference;
                       parameters=nothing, parameter_id=nothing, backend,
                       calibration_model_id=filter_target.signature.model_id)

Prepare bounded-window parameter learning over an authored finite-state filter.
`parameters(u)` maps an unconstrained position to the complete constrained
parameter named tuple expected by `filter_target`; use the model's existing
prepared parameter transformation. `parameter_id` identifies that pure,
unchanging mapping and its frozen preprocessing. It must change when they do.
Omit both to use the native mapping from `carry_smc(completed; parameters=...)`.
`backend` prepares ordinary AD of the smooth Gaussian reference only. Load the
SMC and filtering dependencies. No derivative of particle resampling is used.

The same `SMCCarriedReference` can initialize `prepare_smc_window` and this
filter. Its `frame.model_id` must match `calibration_model_id`. By default this
is the filter's `model_id`. When a consumer has separately verified that its
batch and streaming IDs name compatible source/frozen semantics, it can pass
the exact batch ID explicitly. Both identities remain bound into checkpoints;
the ordinary filter's identity is unchanged, including for boundary transfer.
A completed
SMC cloud is supplied through `carry_smc` (Gaussian moment approximation).
The prepared target retains no calibration observations or trajectories.
"""
function prepare_filter_smc end

"""
    tune_filter_smc(target; rng, init, kwargs...)
    tune_filter_smc(state; rng, init, kwargs...)

Tune fixed NUTS proposals on the carried reference with the existing independent
`tune_smc` pilot. Pass its schedule as `SMCOptions(tuning=schedule)` to
`initialize_filter_smc`. The final temperature's kernel is frozen for the
stream: its `moves` sets the proposals per member at each replenishment, and
its NUTS settings drive only explicit `rejuvenate_smc` recovery. That kernel is
never adapted from production members, even at a window boundary.
Replenishment itself proposes population-scaled random-walk moves; see
`initialize_filter_smc`. This tunes a reference-reversible proposal, not a
noisy likelihood.
The state overload tunes its **current window** reference for explicit recovery
with `rejuvenate_smc(state; tuning, moves)`. Supply pilot initial positions and
RNGs independently of production members. A longer reference pilot alone does
not establish exploration of the observed-window posterior.
"""
function tune_filter_smc end

"""
    initialize_filter_smc(target; rng::Xoshiro, members=64, particles=128,
                          window_length=32, ess_fraction=0.5,
                          options=SMCOptions(), history_limit=32,
                          boundary=nothing, max_reflections=16)

Initialize an owned parameter ensemble from its carried Gaussian and a separate
conditional particle filter per member. Each new record reweights members by
their predictive scores. Replenishment is triggered by the effective sample
size over *distinct* parameter positions: the weights of bitwise-identical
members are summed, so resampled copies count once. When it falls below
`ess_fraction*members`, or any member has zero likelihood, resample the current
parameter/conditional-filter pairs by their updated weights and move the
selected members. Their retained window likelihood estimates and conditional
clouds survive selection. Only one fixed-budget replenishment is attempted per
step.

Each move proposes a Gaussian random walk whose covariance is `2.38^2/d` times
the members' weighted covariance before resampling (Roberts, Gelman and Gilks'
scaling), fixed for the whole replenishment. That covariance keeps up to
`max_reflections` leading correlation directions of the members, fitted in their
marginally standardized frame with `WarmupHMC.fit_gaussian_moments`, so
coordinates on very different scales are treated alike. The retained directions
keep their weighted variance, and the remaining directions share the remaining
standardized variance equally (probabilistic PCA). At most the
distinct-position ESS minus 2 directions are kept. With none, or
`max_reflections=0`, the walk is diagonal with the members' weighted marginal
spread. Work and storage are O(members × d × directions), and no d × d matrix is
formed. A diagonal walk leaves the narrow parameter combinations that the data
pin down, such as a hierarchical intercept plus its group effects, and is
rejected there. Its particle-Metropolis acceptance ratio is
the window Gaussian prior ratio times the proposed/current *window* likelihood
estimate ratio. Each proposal rebuilds conditional states from the frozen
boundary. Rejections retain the previous estimate and conditional cloud. The
kernel contributes only `moves`, the proposals per member. A coordinate whose
weighted spread is zero cannot move: a population whose spread has vanished
stays collapsed and fails as described below. Its recovery needs the
independently tuned kernel of `rejuvenate_smc`.

After `window_length` records, replace the parameter posterior by a Gaussian
with the members' weighted mean and their weighted covariance, summarized as
above (up to `max_reflections` leading correlation directions). Then
independently resample the weighted finite states into a pool of `particles`
states as the next boundary, and draw a new ensemble and independent initial
state clouds from these summaries. A diagonal boundary (`max_reflections=0`) treats the
marginal widths of correlated parameters as independent information. Where the
data inform only combinations, such as a hierarchical intercept and its group
effects, the next window then contracts the population again, and every window
compounds it. The calibration you supply is the first window's prior, so carry
it with its correlations too (`carry_smc(fit; parameters, basis=:fit)`); a
diagonal calibration over-contracts the first window. This discards
parameter–state dependence and Gaussianizes the parameter posterior; it is an
explicit assumed-density approximation, **not exact full-history SMC²**. With a
finite ensemble, the spread in directions that the data never inform still
drifts downward from window to window, because each boundary estimates it
again from the members; more members reduce the drift.
Nonpositive/nonfinite marginal scales fail with `SMCDegeneracyError` at
rollover; no scale is invented.
Full spatial coincidence is checked on input and **after** the configured
mutation pass, so resampling from one parent still gets its chance to move.
An error preserves the input state in `error.state` and the rejected summary
in `error.diagnostics`; no observation or RNG draw is committed on failure.
Attempt explicit `rejuvenate_smc` recovery before retrying that observation.
A constant coordinate alone is allowed within a window, until a Gaussian
boundary needs its positive marginal scale. Cost and retained state are bounded
by members, particles, groups and the chosen window length, not elapsed time.

Use `filter_step`/`filter_stream`, `filter_checkpoint` and
`resume_filter(target, checkpoint)`. Records include observations, inputs and
optional active groups; missing observations keep the ordinary filter meaning.
This learning route consumes new data once. For state-only training warm-up,
first use the ordinary filter, then pass its state as `boundary`. Its weighted
finite states are pooled independently of the new parameter draws; its time
and group clocks are preserved, but its training scores are not added again.
The boundary filter must have the same source/configuration identity. The
learning state retains the sampled pool, never that old filter or its history.
Caller RNGs, input states and observations remain unchanged on success/failure.

Supported read-only diagnostics on the returned state are `positions`
(unconstrained static coordinates, one member per column), `parameter_mass`
(normalized weights), `time`, `window`, `window_length`, `records` (the retained
window; use `length(state.records)`), `predictive_logscore`, `log_normalizer`
and `history`. Each retained history entry exposes `time`, `window`,
`window_steps`, `parameter_ess_before`, `parameter_ess_after`,
`distinct_ess_before`, `distinct_ess_after`, `replenished`, `boundary_reset`,
`boundary_directions`, `unique_parents`, `proposals`, `accepted`,
`distinct_members`, `moved_members`, `coordinate_sd`, `kernel`,
`proposal_scale`, `proposal_directions`, `kind`, `likelihood_evaluations`,
`reference_evaluations` and `predictive_logscore`. `kind` is `:assimilate` or
`:rejuvenate`. `parameter_ess_*` counts members, copies included;
`distinct_ess_*` counts distinct parameter positions and triggers
replenishment. `proposal_scale` is the random walk's per-coordinate standard
deviation in an assimilation replenishment (empty otherwise), and
`proposal_directions` the correlation directions it kept.
`boundary_directions` is the number the step's rollover kept (0 without one).
`reference_evaluations` counts
NUTS steps on the Gaussian reference, which only `:rejuvenate` recovery takes.
`moved_members` counts final positions differing from the selected parent (or
recovery input); accepted proposals may leave a position unchanged.
`coordinate_sd` is the weighted marginal spread in unconstrained coordinates.
History is capped by `history_limit`; counting `replenished` there counts only
retained steps. For lifetime counts, accumulate the latest entry after each
step with `history_limit >= 1`. Equal weights after resampling do not demonstrate
mixing; inspect pre-resampling ESS, ancestry, actual movement and spread too.
"""
function initialize_filter_smc end


# The authored filtering extension owns finite-state storage and transitions.
# The SMC extension uses these narrow internal operations without retranscribing
# a model, resampling engine or FilterState's representation.
function _filter_boundary_initialize end
function _filter_step_or_zero end
function _filter_with_rng end
function _filter_boundary end
function _filter_resample end
function _filter_initial_boundary end
