"""
    prepare_filter(model::RKPPLModel; model_id, observations=(:y,), inputs=(),
                   parameters=(;), data=(;), drifting=(), groups=(;), frozen=FrozenValues())
    prepare_filter(model::RKPPLModel, bound::StructuralPlan; kwargs...)

Compile finite-lag `@scan` seeds, transitions and observation laws from the
authored model, including expanded submodels and `@kernel` providers with
ordinary finite `scan` operations. Load `Distributions`, `Random` and
`SequentialMonteCarlo` 0.7. `WarmupHMC` is not needed for filtering.
`parameters` are calibrated, constrained values held fixed for the entire
stream. `data` holds fixed model inputs, never an accumulating observation
history. `frozen` substitutes a `FrozenValues` record directly, without retaining
its original training inputs. New observations and covariates arrive as named
tuples to `filter_step`.
The bound-plan overload defaults to `frozen=bound.frozen`, obtains required
fixed inputs from that binding, and retains only those read by the projected
transition. It excludes observation/input histories and batch horizons.
The returned `target.data` contains those selected fixed values; checkpoint
resume compares them independently of `model_id`.
Computed covariates/designs are recomputed from new input ports; only explicitly
frozen definitions use their fitted values. Pass an exported `FrozenValues`
record explicitly when the binding has not yet applied it. Unfrozen fixed
inputs actually read by the transition remain retained. Scoped data names use
the same authored dotted paths as fixed parameters.
The scan's upper bound names the batch horizon; filtering extends the same
recurrence one time slice at a time. Its seed observations are assimilated too.
Models with no carried state use the same path with empty carries. Temporal
`hcat` design columns and their products with fixed coefficient vectors are
projected one row at a time, including constant `ones`/`zeros`/`fill` columns
whose dimension is `length` of a stream port.

`drifting` names sampled carried states that the author interprets as dynamic
parameters. Their authored transitions define the dynamic-parameter model;
there is no implicit parameter jitter or reinterpretation of old carries.
The ordinary cumulative `update_smc` remains the static-parameter replay route.

The initial implementation stores scalar Float64 states with literal bounded
lags, ordinary scalar definitions, normalized Distributions laws, and one level
of fixed-size `@plate` groups. `groups=(;g=1:G)` fixes the plate's group axis;
`filter_step(...;groups=(;g=[2,5]))` advances only selected groups, each on its
own clock. Models mixing global and grouped recurrences require synchronized
slices. All groups share each particle's global parameters.
The authored plate domain must match that fixed axis: `eachindex`/`length`
of a stream port or a fixed bound range is supported; domain mismatches fail.
Arbitrary-history scans, growing groups,
general whole-trajectory transformations are refused. Omitted observation names
or `missing` entries skip conditioning but advance selected groups; omitted
groups do not advance. Missing observed recurrence states are drawn from their
authored law. Observation/input arrays retain the original group positions.
Other whole-vector calls cannot be projected as elementwise work; author pointwise
operations with broadcasts. Opaque scalar user functions must themselves have
bounded work/storage per invocation. Kernel-provider scans currently require
scalar carry/output, invariant seeds/shared arguments and `include_init=false`.
`model_id` must change with captured code, frozen preprocessing or semantics;
captured functions must remain pure and unchanged. `program` on the returned
target is the actual generated host program, including its random draws.
"""
function prepare_filter end

"""
    filter_parameter_names(model; observations=(:y,), inputs=(), data=(;))
    filter_parameters(model, constrained_values; kwargs...)

Identify static declarations outside authored scans, expanding the model's
submodels. Names use authored dotted paths. `filter_parameters` selects and
copies only those values from a flat or nested constrained named tuple/dict.
Historical latent seeds, innovations and carried paths are excluded. Use it
on each calibrated posterior draw before passing an ensemble to the filter;
no retained training trajectories are needed. Inputs and already pinned values
are not fixed parameters to extract. Requires the filtering extension.
"""
function filter_parameter_names end
function filter_parameters end

"""
    initialize_filter(target; rng::Random.Xoshiro, particles=1000,
                      ensemble=nothing, weights=nothing, history_limit=32)

Create an owned cloud before time 1, without advancing the supplied RNG.
An optional fixed `ensemble` is a vector of complete parameter named tuples
with the same keys/types as `target.parameters`. `particles` is the number of
conditional state particles **per ensemble member**.
`weights` gives the ensemble's initial normalized probability masses (uniform
by default). Filtering reweights and resamples this finite approximation; it
does not replenish static parameter diversity. Resampling stays within each
member's conditional state cloud. Inspect `parameter_mass`, ESS,
and surviving lineages for degeneracy. No implicit prior is sampled or scored.
"""
function initialize_filter end

"""
    filter_step(state, observation::NamedTuple; inputs=(;), condition_state_only=false,
                groups=nothing)

Advance one authored time slice and return a new conditioned cloud. The
`predictive_logscore` is the log weighted mean of the new observation likelihood
over the propagated, **pre-assimilation** particles; `log_normalizer` accumulates
these scores. Then SequentialMonteCarlo weights/resamples the whole particle,
including its independently owned finite carries. Each fixed parameter member
retains its own conditional cloud; its mass is updated by its predictive score.
`carries[i].name[end, group]` is a current state value; grouped state names are
qualified as `Symbol("group_binder.name")`. `weights` combine member mass and
equal conditional particle weights after resampling.
Each bounded history entry contains the predictive score, the previous
parameter-mass-weighted mean conditional ESS, parameter ESS, unique immediate
parents and surviving original lineages. These expose state and parameter
degeneracy; they do not guarantee Monte Carlo accuracy.

For parameters already calibrated on a training history, warm up the finite
states on that history with `condition_state_only=true`. This conditions each
member's latent states while preserving its parameter mass and the accumulated
`log_normalizer`; it avoids counting the training likelihood twice. Cold
initialization followed by ordinary steps updates both. Resume from a compatible
joint checkpoint needs no warm-up. The warm-up flag is recorded in diagnostics.

Every engine invocation has exactly two iterations; even scalar engine
diagnostics have bounded length. State and checkpoint retention is independent
of elapsed time at fixed particle count, lag depth, group count and
`history_limit`. Input state and RNG survive any failed step unchanged.
All-zero, NaN and positive-infinite likelihood weights fail visibly.
"""
function filter_step end

"""
    filter_stream(state, records; steps=typemax(Int), stop=state->false)

Process records `(observation=(; y=...), inputs=(; x=...))`. Check `stop` before
requesting each record; it may implement a cooperative interrupt. Return at
the last completed observation. No iterator or stream history is retained.
Use a stateful iterator when resuming consumption of the same live stream.
"""
function filter_stream end

"""
    filter_checkpoint(state)
    resume_filter(target, checkpoint)

Copy a versioned data-only checkpoint: finite carries, fixed parameter ensemble,
weights, ancestry, bounded diagnostics and continuation RNG. Rebuild `target`
from the original authored model and fixed data after restart. Resume checks
the model/port/configuration signature, Julia/package versions and state shape.
No compiled closures or stream observations are serialized. Exact continuation
requires identical captured code and versions. Persist only trusted checkpoints
with Julia Serialization; atomic file replacement remains the caller's job.
"""
function filter_checkpoint end
function resume_filter end
