"""
    SMCReference(location, scale)

A normalized product Normal distribution in the model's **unconstrained**
coordinates. This is an importance reference, not a draw from the model prior.
The bridge scores its actual normalized density and the full model posterior
(including dependent priors and the Jacobian). Both vectors are copied.
"""
struct SMCReference
    location::Vector{Float64}
    scale::Vector{Float64}
    function SMCReference(location::AbstractVector, scale::AbstractVector)
        length(location) == length(scale) || throw(DimensionMismatch("reference location and scale"))
        all(isfinite, location) || throw(ArgumentError("reference location must be finite"))
        all(x -> isfinite(x) && x > 0, scale) || throw(ArgumentError("reference scale must be finite and positive"))
        new(Vector{Float64}(location), Vector{Float64}(scale))
    end
end

"""
    SMCTuningSchedule

Data-only result of `tune_smc`. `temperatures` and `kernels` describe a finite,
precomputed schedule. Each kernel records `stepsize`, `metric` (scales), `moves`
and `max_tree_depth`. Production selects the nearest temperature knot (ties
select the higher knot), using only its temperature and this configuration.
`source` and `target` bind the schedule to the exact bridge; `reference` is the
normalized initial reference, or `nothing` for a posterior update. `pilot`
records the copied initial/final RNG, initial position and bounded pilot work.

Arrays in this record must remain read-only. Initialization, updates and resume
copy and validate the schedule. Checkpoints include it; resuming never reruns
pilot adaptation. A schedule is not a stationary posterior sample.
"""
struct SMCTuningSchedule{S,T,R,K,P}
    source::S
    target::T
    reference::R
    temperatures::Vector{Float64}
    kernels::Vector{K}
    pilot::P
end

"""
    SMCOptions(; ess_fraction=0.8, stepsize=0.25, metric=nothing,
               moves=2, max_tree_depth=6, max_stages=1000,
               min_increment=1e-12, history_limit=128, tuning=nothing)

Adaptive bridge and fixed NUTS configuration. `metric` is a vector of positive
diagonal **scales**, not variances or precisions; `nothing` means ones. The
metric, step size and tree depth stay fixed during mutation, with no adaptation
using the particles being moved. `tuning=tune_smc(...)` instead selects a fixed
kernel from an independent, precomputed, bridge-specific pilot schedule.
Scheduled kernel settings override `stepsize`, `metric`, `moves` and
`max_tree_depth` for production mutation. Each stage resamples, then takes `moves`
invariant NUTS transitions. `ess_fraction` selects the next temperature by
bisection on the weight ESS, before resampling. A bridge that cannot progress
by `min_increment`, or needs more than `max_stages`, throws without advancing
its input state. Only the last `history_limit` stage summaries are retained.
"""
struct SMCOptions
    ess_fraction::Float64
    stepsize::Float64
    metric::Union{Nothing,Vector{Float64}}
    moves::Int
    max_tree_depth::Int
    max_stages::Int
    min_increment::Float64
    history_limit::Int
    tuning::Union{Nothing,SMCTuningSchedule}
    function SMCOptions(; ess_fraction=0.8, stepsize=0.25, metric=nothing,
            moves=2, max_tree_depth=6, max_stages=1000, min_increment=1e-12,
            history_limit=128, tuning=nothing)
        0 < ess_fraction < 1 || throw(ArgumentError("ess_fraction must be between 0 and 1"))
        isfinite(stepsize) && stepsize > 0 || throw(ArgumentError("stepsize must be finite and positive"))
        metric === nothing || (metric isa AbstractVector && all(x -> isfinite(x) && x > 0, metric)) ||
            throw(ArgumentError("metric must be a vector of finite positive scales"))
        moves >= 1 || throw(ArgumentError("moves must be positive"))
        max_tree_depth >= 1 || throw(ArgumentError("max_tree_depth must be positive"))
        max_stages >= 1 || throw(ArgumentError("max_stages must be positive"))
        0 < min_increment < 1 || throw(ArgumentError("min_increment must be between 0 and 1"))
        history_limit >= 1 || throw(ArgumentError("history_limit must be positive"))
        tuning === nothing || tuning isa SMCTuningSchedule ||
            throw(ArgumentError("tuning must be a SMCTuningSchedule or nothing"))
        new(ess_fraction, stepsize, metric === nothing ? nothing : Vector{Float64}(metric),
            moves, max_tree_depth, max_stages, min_increment, history_limit, deepcopy(tuning))
    end
end

"""
    prepare_smc(built, binding, u0; backend, model_id, binding_id, kwargs...)

Prepare a reusable SMC posterior target using `prepare_sampler`. Load
`SequentialMonteCarlo`, `WarmupHMC`, `LogDensityProblems` and `Random` first.
`model_id` identifies the model's source coordinate frame, including frozen
preprocessing and caller-owned geometry; `binding_id` identifies these exact
bound data. Use stable strings, and change them when their meanings change.
Updates additionally check the packed layout; generated programs may change
with the data (for example, a longer scan). Each build must match its own
binding. `model_id` is a caller-supplied identity, not an inferred code hash. Bindings,
captured data and code must remain read-only for a target's lifetime.
`kwargs` are forwarded to `prepare_sampler`.
"""
function prepare_smc end

"""
    tune_smc(target; rng::Xoshiro, init, reference=nothing, previous=nothing,
             temperatures=(0.0, 0.05, 0.2, 0.5, 1.0), pilot_steps=64,
             options=SMCOptions())

Prepare all mutation tuning **before** creating a production cloud. `init` is
an explicit finite unconstrained position and `rng` an explicit independent
pilot RNG; both are copied. Do not supply production-derived positions or RNGs.
The pilot follows the specified fixed temperature grid, adapting through
WarmupHMC's continuous diagonal kernel, then freezes each knot's scale and step
size. Its draws are discarded. It does not adapt from production particles.

For an initial bridge, supply a normalized `SMCReference` (default standard
Normal). The reference remains fixed throughout tuning and production. For a
cumulative-data update pass `previous=old_prepared_target` and no reference.
The returned schedule is valid only for these exact endpoints. Supply it as
`SMCOptions(tuning=schedule)`. The production ESS temperature rule remains
adaptive; this API does not claim exactly unbiased finite-particle evidence.

`temperatures` must strictly increase from 0 to 1; `pilot_steps` is the positive
number of adaptive transitions per knot. `options` supplies the initial metric,
production move count and tree-depth limit; it must not itself contain tuning.
"""
function tune_smc end

"""
    smc_reference(state_or_schedule)

Return an owned, data-only normalized reference for an initial bridge, or
`nothing` for a posterior-to-posterior update. An update uses the previous
prepared posterior as its source and does not need the original reference.
"""
function smc_reference end
smc_reference(schedule::SMCTuningSchedule) = deepcopy(schedule.reference)

"""
    SMCNonfiniteError <: Exception

An invalid SMC endpoint or bridge evaluation. `position` is a copied, exact
unconstrained proposal; `value` is the evaluated log density or log-density
ratio, and `gradient` is a copied gradient when one was evaluated (otherwise
`nothing`). `quantity` is `:density`, `:gradient` or `:logratio`; `endpoint` is
`:source`, `:target` or `:bridge`. `temperature` is the bridge temperature at
evaluation, before advancement for weight/ratio checks.

NaN and +Inf densities and nonfinite gradients at finite densities fail fast.
The existing -Inf outside-support convention is preserved. Initial clouds
are evaluated by their first `smc_step`, not by `initialize_smc`.

Error display is bounded, including display with a backtrace: inspect the
fields for the full numerical diagnostic, and `catch_backtrace()` for the
caller stack. Mutation failures cross the worker-task boundary as this same
exception, without a nested task trace. The input cloud and RNG are preserved.
"""
struct SMCNonfiniteError <: Exception
    position::Vector{Float64}
    value::Float64
    gradient::Union{Nothing,Vector{Float64}}
    quantity::Symbol
    endpoint::Symbol
    temperature::Float64
    function SMCNonfiniteError(position, value, gradient, quantity, endpoint, temperature)
        new(copy(position), value, gradient === nothing ? nothing : copy(gradient),
            quantity, endpoint, temperature)
    end
end

function Base.showerror(io::IO, err::SMCNonfiniteError)
    print(io, "SMCNonfiniteError: invalid ", err.quantity, " at ", err.endpoint,
        " endpoint, temperature=", err.temperature, ", value=", err.value,
        ", dimension=", length(err.position),
        ". Full numerical diagnostic: error.position, error.value, error.gradient.")
end
Base.show(io::IO, err::SMCNonfiniteError) = showerror(io, err)
# Prepared target types can contain the entire model. Their stack frames are
# not a bounded diagnostic; the ordinary catch_backtrace API remains available.
Base.showerror(io::IO, err::SMCNonfiniteError, bt; backtrace=true) = showerror(io, err)

"""
    initialize_smc(target; rng, particles=1000,
        reference=SMCReference(zeros(d), ones(d)), options=SMCOptions())

Draw the initial reference cloud, at temperature zero. Copies `rng` (an
explicit `Random.Xoshiro`) and never advances the caller's RNG. The returned
state owns its particle positions, normalized weights and continuation RNG.
Call `smc_step` for one completed stage, or `smc` to finish the bridge.
With `options.tuning`, an omitted reference uses the schedule's exact reference;
an explicit different reference rejects before drawing any particles.
"""
function initialize_smc end

"""
    smc(target; rng, particles=1000, reference=..., options=SMCOptions(), stages=typemax(Int))
    smc(state; stages=typemax(Int))

Run up to `stages` complete adaptive stages. A completed state's `temperature`
is exactly 1. Positions are columns in the original unconstrained frame;
`weights` are normalized. `log_normalizer` estimates the full target integral
from the normalized reference. It is model evidence only when the authored
density includes the necessary normalizing constants. No input state is mutated.
"""
function smc end

"""
    smc_step(state)

Return the next complete resample/move stage. `history` holds bounded summaries
(temperature, ESS before resampling, log normalizer increment, unique parents,
surviving initial ancestors, moves and divergences). `ancestors` maps the most
recent stage to its input cloud, and `lineage` tracks the initial cloud.
Each summary's `kernel` records the actual scales, step size, moves per particle
and depth limit; `tuning_index` is its schedule knot (zero when untuned).
`evaluations` sums the transition evaluation counts; `mean_squared_movement`
is the mean squared unconstrained displacement from each resampled parent to
its final child, after all moves. These describe mutation work and movement,
not independent evidence that a posterior has been explored adequately.
Failures propagate; the input state and its RNG remain usable.
"""
function smc_step end

"""
    update_smc(previous, target; options=previous.options, stages=typemax(Int))

Bridge a **completed** posterior cloud to a target for caller-supplied
**cumulative** data in the same model and coordinate frame. The bridge density
is `(1-beta)*old_posterior + beta*new_posterior` in log space, so priors and
Jacobians occur once. Every mutation evaluates the required old/new histories
from their beginning. This is full-history updating, not constant-cost streaming
or a latent-state particle filter. No old terminal scan state is reused.
If `previous.options` contains tuning for its old bridge, the default rejects:
supply a freshly prepared update schedule, or explicit untuned `SMCOptions`.
"""
function update_smc end

"""
    smc_checkpoint(state)
    resume_smc(target, checkpoint; previous=nothing)

Copy the completed-stage state into a versioned, data-only checkpoint. Persist
it with Julia `Serialization` in a trusted file. Rebuild the target after a
restart and pass it to `resume_smc`; a data-update checkpoint also needs the
old prepared target as `previous`. IDs, program/layout and endpoint densities
are checked. Queries, compiled code and AD caches are never serialized.
Exact seeded continuation requires the same Julia/package versions, data,
model code and configuration. This API does not write files or promise atomic
disk persistence. The caller owns file replacement and model/data retention.
"""
function smc_checkpoint end
function resume_smc end
