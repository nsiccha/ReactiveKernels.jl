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

Tune fixed NUTS proposals on the carried reference with the existing independent
`tune_smc` pilot. Pass its schedule as `SMCOptions(tuning=schedule)` to
`initialize_filter_smc`. The final temperature's numerical kernel is frozen
for the stream. It is never adapted from production members, even at a window
boundary. This tunes a reference-reversible proposal, not a noisy likelihood.
"""
function tune_filter_smc end

"""
    initialize_filter_smc(target; rng::Xoshiro, members=64, particles=128,
                          window_length=32, ess_fraction=0.5,
                          options=SMCOptions(), history_limit=32,
                          boundary=nothing)

Initialize an owned parameter ensemble from its carried Gaussian and a separate
conditional particle filter per member. Each new record reweights members by
their predictive scores. Below `ess_fraction*members`, draw a fresh ensemble
from the current window prior, replay only the retained window and importance
weight by its particle likelihood estimates, then resample and rejuvenate.
Only one replenishment is attempted per step; diagnostics show remaining ESS.

A rejuvenation proposal is a fixed WarmupHMC NUTS transition reversible with
respect to the current Gaussian prior. Its particle-Metropolis acceptance
ratio is the proposed/current *window* likelihood estimate. Each proposal
rebuilds conditional states from the frozen boundary. Rejections retain the
previous estimate and conditional cloud. Fixed-kernel settings stay unchanged.

After `window_length` records, replace the parameter posterior by its weighted
Gaussian marginal-moment summary and independently resample the weighted finite
states into a pool of `particles` states as the next boundary. Draw a new ensemble and independent initial state
clouds from these summaries. This discards parameter–state dependence and
Gaussianizes the parameter posterior; it is an explicit assumed-density
approximation, **not exact full-history SMC²**. Degenerate marginal variance
fails visibly rather than inventing a scale. Cost and retained state are bounded
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
