"""
    SMCReference(location, scale)

A normalized Gaussian distribution in the model's **unconstrained**
coordinates. This is an importance reference, not a draw from the model prior.
The bridge scores its actual normalized density and the full model posterior
(including dependent priors and the Jacobian). `scale` is a vector of diagonal
scales or WarmupHMC's plain `scale_data` record (load the SMC dependencies first).
The latter represents a correlated, bounded-reflection Gaussian without a dense
covariance. Fields `location` and `scale` own validated copies; treat them as
read-only. `smc_reference` returns another owned copy. A public two-dimensional
example is `scale=(; diagonal=[sqrt(2.0),sqrt(0.02)], indices=[[1,2]],
reflections=[[cos(pi/8),sin(pi/8)]])`, with covariance
`[1.01 0.99; 0.99 1.01]`. See WarmupHMC's `scale_from_data`/`scale_data`
contract for the operator representation. All sampling, scoring and mutation
use the rebuilt data operator, including its floating-point representation.

Selective carries also produce `(kind=:diagonal_lowrank, diagonal=D,
vectors=Q, values=v)`, representing covariance
`Diagonal(sqrt.(D)) * (I + Q * Diagonal(v) * Q') * Diagonal(sqrt.(D))`.
Here `D` is positive, `Q` has orthonormal columns and `v > -1` elementwise.
This compact covariance is accepted as an importance reference, including for
prepared-density targets; it is not a WarmupHMC mutation metric. Its storage
is linear in dimension at fixed rank, with no dense covariance.
"""
struct SMCReference{S}
    location::Vector{Float64}
    scale::S
    function SMCReference(location::AbstractVector, scale)
        copied = _smc_scale(scale)
        length(location) == _smc_scale_dimension(copied) || throw(DimensionMismatch("reference location and scale"))
        all(isfinite, location) || throw(ArgumentError("reference location must be finite"))
        new{typeof(copied)}(Vector{Float64}(location), copied)
    end
end

function _smc_scale(scale::AbstractVector)
    all(x -> isfinite(x) && x > 0, scale) || throw(ArgumentError("scales must be finite and positive"))
    return Vector{Float64}(scale)
end
_smc_scale(scale) = throw(ArgumentError("scale must be a positive vector or supported WarmupHMC scale_data record; load the SMC dependencies"))
_smc_scale_dimension(scale::AbstractVector) = length(scale)
_smc_scale_dimension(scale::NamedTuple) = length(scale.diagonal)

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
               min_increment=1e-12, history_limit=128, tuning=nothing,
               nthreads=Threads.nthreads())

Adaptive bridge and fixed NUTS configuration. `metric` is a vector of positive
diagonal **scales**, or WarmupHMC's plain `scale_data` record for a correlated
bounded-reflection scale; `nothing` means ones. These are not precisions. The
metric, step size and tree depth stay fixed during mutation, with no adaptation
using the particles being moved. `tuning=tune_smc(...)` instead selects a fixed
kernel from an independent, precomputed, bridge-specific pilot schedule.
Scheduled kernel settings override `stepsize`, `metric`, `moves` and
`max_tree_depth` for production mutation. Each stage resamples, then takes `moves`
invariant NUTS transitions. `ess_fraction` selects the next temperature by
bisection on the weight ESS, before resampling. A bridge that cannot progress
by `min_increment`, or needs more than `max_stages`, throws without advancing
its input state. `nthreads` bounds the number of concurrent particle-mutation
workers (default `Threads.nthreads()`); each owns its sampler and AD scratch.
Weighting, resampling and child-seed draws retain their serial order. Particle
counts need not divide the worker count. A fixed seed and worker count reproduce
the cloud and diagnostics; `nthreads=1` preserves the serial random stream.
Use an AD backend supporting concurrent evaluation (for Enzyme, the dogfood
concurrent-GC repair is required until released). Only the last `history_limit`
stage summaries are retained.
"""
struct SMCOptions
    ess_fraction::Float64
    stepsize::Float64
    metric::Union{Nothing,Vector{Float64},NamedTuple}
    moves::Int
    max_tree_depth::Int
    max_stages::Int
    min_increment::Float64
    history_limit::Int
    tuning::Union{Nothing,SMCTuningSchedule}
    nthreads::Int
    function SMCOptions(; ess_fraction=0.8, stepsize=0.25, metric=nothing,
            moves=2, max_tree_depth=6, max_stages=1000, min_increment=1e-12,
            history_limit=128, tuning=nothing, nthreads=Threads.nthreads())
        0 < ess_fraction < 1 || throw(ArgumentError("ess_fraction must be between 0 and 1"))
        isfinite(stepsize) && stepsize > 0 || throw(ArgumentError("stepsize must be finite and positive"))
        metric = metric === nothing ? nothing : _smc_scale(metric)
        nthreads isa Integer && nthreads >= 1 || throw(ArgumentError("nthreads must be a positive integer"))
        moves >= 1 || throw(ArgumentError("moves must be positive"))
        max_tree_depth >= 1 || throw(ArgumentError("max_tree_depth must be positive"))
        max_stages >= 1 || throw(ArgumentError("max_stages must be positive"))
        0 < min_increment < 1 || throw(ArgumentError("min_increment must be between 0 and 1"))
        history_limit >= 1 || throw(ArgumentError("history_limit must be positive"))
        tuning === nothing || tuning isa SMCTuningSchedule ||
            throw(ArgumentError("tuning must be a SMCTuningSchedule or nothing"))
        new(ess_fraction, stepsize, metric,
            moves, max_tree_depth, max_stages, min_increment, history_limit, deepcopy(tuning), nthreads)
    end
end

"""
    prepare_smc(built, binding, u0; backend, model_id, binding_id, kwargs...)
    prepare_smc(make_problem, u0; backend, model_id, binding_id, frame_id)

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

The two-argument method accepts a zero-argument factory `make_problem()` returning
a prepared `LogDensityProblems` target with `dimension`, `logdensity` and
`logdensity_and_gradient` (capability order at least one). `u0` supplies its
coordinate dimension, not production particles. The supplied density already
includes all priors, coordinate Jacobians and approximation terms: SMC adds none.
Its gradient is used directly; `backend` prepares only the Gaussian reference's
derivatives. A gradient returned in reusable storage is copied before retention.

Each factory call must return the same density in the same coordinate frame,
with independent mutable workspaces. Preparation calls it once; initialization,
tuning, update, resume and additional mutation workers construct owned execution
instances. Factories are invoked before launching mutation workers, never once
per particle. Read-only data may be shared; do not share mutable solver state.
Use `() -> deepcopy(problem)` only when copying is valid for that problem.
Execution instances along one state's continuation may share scratch; do not
concurrently advance or checkpoint the same continuation. Separate initializations
and resumes have independent problem state.

`model_id` covers density code and approximation configuration (including any
inner-solver settings), `binding_id` covers exact data, and `frame_id` covers
ordered coordinate meanings, transforms and preprocessing. These stable strings
are caller assertions, not inferred hashes. Change them when semantics change.
Updates require matching model/frame/dimension; resume and tuning also require
the same binding. Checkpoints contain identities and endpoint scores, never the
factory or its workspaces. Rebuild with the same code, data and configuration
before `resume_smc`. This adapter does not infer named RKPPL parameter geometry.
"""
function prepare_smc end

"""
    SMCCarriedReference

Data-only Gaussian summary of a posterior in a fixed unconstrained coordinate
frame. Construct with [`carry_smc`](@ref). The owned `reference`, `frame` and
`calibration_id` contain no old observations, prepared queries or trajectories.
Treat them as read-only. The frame includes the caller's model/preprocessing
identity and the packed layout; the calibration ID identifies this summary.
For full-frame targets, `fit_diagnostics` is `nothing` for externally supplied
or fixed-basis summaries; `basis=:fit` carries the bounded WarmupHMC fitting
diagnostics as owned data. Selective carries also report joint recompression.
"""
struct SMCCarriedReference{R,F,D}
    reference::R
    frame::F
    calibration_id::String
    fit_diagnostics::D
end
SMCCarriedReference(reference,frame,calibration_id::String) =
    SMCCarriedReference(reference,frame,calibration_id,nothing)

"""
    carry_smc(target, reference::SMCReference; calibration_id)
    carry_smc(selective_target, reference::SMCReference; calibration_id, maxrank=16)
    carry_smc(completed_state; calibration_id, scale=nothing, parameters=nothing,
              basis=nothing, max_reflections=16, maxiter=100, progress=nothing)

Copy a calibrated Gaussian posterior into the target's coordinate frame for
bounded-window updates. The first form accepts an externally calibrated
reference. A tuning schedule's reference is a normalized proposal, not a
posterior moment estimate. The second form uses the completed
cloud's weighted mean and, by default, its marginal standard deviations
(population moments, not a dense covariance). Supply a diagonal scale or a
portable WarmupHMC `scale_data` record to preserve learned correlations.
Zero or nonfinite marginal scales require an explicit valid scale.

`basis=portable_scale` instead fits the completed cloud's weighted marginal
variances after whitening by that scale, then replaces its diagonal magnitudes
while retaining its learned reflection directions. Thus proposal scale inflation
does not become posterior variance. This costs O(particles × dimension × bounded
reflections), with O(dimension) working storage and no dense covariance. It
discards cross-covariances in the supplied basis: exact covariance requires that
basis to diagonalize the posterior covariance. Use an independently learned
orientation, for example `smc_reference(schedule).scale`, in the selected frame.
`basis` and explicit `scale` are mutually exclusive; `scale` is copied as given.

`basis=:fit` learns directions from the completed weighted cloud through
WarmupHMC's public `fit_gaussian_moments`, then carries its mean and portable
scale. `max_reflections`, `maxiter` and `progress` apply to this fit. Use this
route when a supplied pilot basis omits posterior correlations. It fits the
selected coordinates when `parameters` is given, without slicing full-frame
geometry. Work is O(d N (1 + k maxiter)), workspace O(d N + d k + N), and retained
size O(d (1 + k)); no dense covariance or old cloud is retained. These are
posterior summary moments, not production-kernel adaptation. The owned
`fit_diagnostics` contains `reflections`, `iterations`, `relative_residuals`,
`converged` and `effective_sample_size`. Residual convergence does not certify
covariance accuracy; finite reflection/iteration budgets may lose correlations.

This is an assumed-density approximation to the old posterior. A Gaussian
summary is exact only for a Gaussian posterior with matching moments; fitting
it from a finite cloud introduces Monte Carlo error too. The cloud itself is
not retained. This summary is a prior for subsequent windows, not production
kernel adaptation. Mutation tuning still uses independent pilot RNGs/positions.

`parameters=(:name, Symbol("nested.name"), ...)` selects complete authored
static parameter blocks from a completed full-frame fit, excluding historical scan
coordinates. The producer packs their unconstrained marginal and carries only
their portable native geometry. `scale`, if supplied, must be in this selected
frame; slicing a full correlated scale does not produce its marginal scale.
Use [`smc_parameters`](@ref) to restore the selected constrained tuple.
Static transform dependencies must also be selected. Data-only scalar bound
expressions are folded once, so old observations are not retained. Custom
geometry containing callable objects is not represented by this native
projection; use a static-only calibration and an explicit mapping for it.
For selective windows (`coordinates=:auto`), omit `parameters`: carrying
integrates the inactive conditional into the full joint Gaussian.
The external-reference form bounds joint rank by `maxrank`; completed states
use `max_reflections`. Completed selective states return `.fit_diagnostics.marginal`
for the compact fit and `.joint` for recompression; the external form returns
the joint diagnostics directly. Their `rank`, `discarded_rank` and
`relative_loss` describe the retained rank and discarded correction's spectral
norm in the diagonal-whitened frame, not posterior accuracy.
"""
function carry_smc end

"""
    prepare_smc_window(built, binding, u0; previous::SMCCarriedReference,
                       backend, model_id, binding_id, warm=false, coordinates=:all)

Prepare a static-parameter target `log(previous.reference) + loglikelihood`
for **new observations only**. The likelihood and its ordinary AD derivative
come from the model's `:likelihood` query. Original priors and coordinate
Jacobians are not added again: the carried reference is already a density in
unconstrained coordinates. Parameter transformations in the likelihood remain.

With `coordinates=:all`, the model/preprocessing identity and packed layout
must match `previous`.
Keep fitted preprocessing frozen and pass it through the binding. Retained
storage and each evaluation depend on this window, never on elapsed history.
Use the filter for latent trajectories whose transitions are evaluated as
successive observations arrive.

With `coordinates=:auto`, the prepared target uses only the old coordinates
read by this window's likelihood or fresh priors, and all fresh coordinates.
Authored paths and coordinate labels match the previous carry; packed offsets
may change. Rebuild and bind the current program when an array extent changes,
while keeping `model_id` and fitted preprocessing fixed. The target is
`q_old(active) * p(fresh | active) * likelihood(window | active, fresh)`.
Fresh factors include their native transforms and Jacobians once. Dependencies
of fresh priors, including old hyperparameters absent from the likelihood,
belong to the active set. Joint factors and coupled transforms remain intact.
Fresh events are selected by [`factor_coordinates`](@ref) ownership, never by
their read support. A density on a derived expression that owns no coordinates
but reads fresh ones cannot be assigned automatically and is rejected rather
than omitted. Fresh-factor discovery inherits [`density_factors`](@ref)'s
explicit limitations for noncentered scans and stochastic scans inside cells.
The likelihood must satisfy native partition locality. Sparse observations
selecting elements of simplex/ordered slice-transformed arrays currently lack
an observation row lowering and are rejected; their fresh atomic priors are
supported. No full-coordinate fallback is performed.

Inspect [`smc_window_coordinates`](@ref) before supplying a compact initial
point, reference or mutation metric. Particle positions and gradients use that
compact order. [`smc_recover`](@ref) draws the inactive old coordinates lazily
from their retained conditional; no full joint particle cloud is stored.
`carry_smc(completed; max_reflections=16, ...)` integrates that conditional
against the fitted compact Gaussian, flattens the joint covariance and caps
its rank at the requested budget. Its full location vector follows the current
layout, followed by retained coordinates absent from the current program.
Recompression diagnostics report discarded rank and the spectral norm of the
discarded covariance correction in the diagonal-whitened frame; this is not
a posterior accuracy bound.

Use the ordinary `tune_smc`, `initialize_smc`, `smc`, `smc_checkpoint` and
`resume_smc` APIs. The default initial reference is the carried prior for
`:all`, or its active marginal with independent standard-normal fresh
coordinates for `:auto`. This proposal does not replace the fresh priors in
the target. A
checkpoint resumes from the rebuilt window target alone, without old data.
Its log normalizer is the new window's predictive normalizer under the
approximate carried prior, not exact full-history evidence. Supply each new
window once; overlapping windows would count their observations again.
"""
function prepare_smc_window end

"""
    tune_smc(target; rng::Xoshiro, init, reference=nothing, previous=nothing,
             temperatures=(0.0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0),
             pilot_steps=64, pilot=:continuous, max_reflections=16,
             recording_target=128, pilot_evaluations=1000, reference_inflation=2.0,
             options=SMCOptions())

Prepare all mutation tuning **before** creating a production cloud. `init` is
an explicit finite unconstrained position and `rng` an explicit independent
pilot RNG; both are copied. Do not supply production-derived positions or RNGs.
The pilot follows the specified fixed temperature grid, adapting through
WarmupHMC's continuous diagonal kernel (`pilot=:continuous`), then freezes each
knot's scale and step size. `pilot=:adaptive` instead runs ordinary adaptive
warm-up at each knot and exports its active bounded-reflection or diagonal scale
with that scale's own step size. Each run starts from an explicit diagonal
initializer, avoiding a nonportable Pathfinder factor. Its draws are discarded.
It does not adapt from production particles.

For an initial bridge, supply a normalized `SMCReference` (default standard
Normal). With `pilot=:adaptive`, `reference=:pilot` first runs a separate pilot
at the full target and learns a normalized Gaussian from its retained draw mean
and variances in the active scale's reflection basis. Cross-covariances in that
basis are discarded. `reference_inflation` multiplies the fitted standard
deviations (default 2; finite and positive). It applies only to `reference=:pilot`;
1 uses the uninflated fitted moments, not the active mutation scale. The active
mutation scale and its paired step size are exported unchanged by each knot's
pilot. The reference is fixed before the grid pilots and production. Fitting
costs O(N*d*k) for N retained draws, d coordinates and k bounded reflections
(O(N*d) without reflections), with no dense covariance.

Pilot exploration and wider references do not guarantee posterior coverage.
`schedule.pilot.reference_work` records the inflation and native-coordinate
`draw_sd`, alongside pilot work. A proposal reference is not a posterior-moment
summary of the production cloud.
`smc_reference(schedule)` returns the resulting data-only reference. For a
cumulative-data update pass `previous=old_prepared_target` and no reference.
The returned schedule is valid only for these exact endpoints. Supply it as
`SMCOptions(tuning=schedule)`. The production ESS temperature rule remains
adaptive; this API does not claim exactly unbiased finite-particle evidence.

`temperatures` must strictly increase from 0 to 1; `pilot_steps` is the positive
number of continuous adaptive transitions per knot. For `pilot=:adaptive` it is
the retained-draw floor per warm-up, not an exact count; `pilot_evaluations`
sets its first evaluation window, `recording_target` bounds its retained pool,
and `max_reflections` bounds the learned operator. The schedule retains only
scale data, coordinate spread summaries and work records, including each pilot
seed; its evaluation counts exclude upstream initialization and the initial
step-size search.
`options` supplies the initial diagonal metric,
production move count and tree-depth limit; it must not itself contain tuning.
The default grid includes near-zero knots. Supply a model-specific grid when
its tempering regime needs finer resolution; a finite grid is not an automatic
guarantee that the learned kernels explore every intermediate target well.
"""
function tune_smc end

"""
    smc_reference(state_or_schedule)

Return an owned, data-only normalized reference for an initial bridge, or
`nothing` for a posterior-to-posterior update. An update uses the previous
prepared posterior as its source and does not need the original reference.
This returns the initial proposal distribution, not posterior moments from a
completed cloud. Use `carry_smc(completed; basis=:fit, ...)` to fit posterior
moments before carrying a summary into a new window.
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
    SMCDegeneracyError <: Exception

All particle positions are exactly equal in a positive-dimensional SMC cloud.
`state` is the preserved input state (not a copy; treat it as read-only), including
its unchanged continuation RNG and normalizer. `phase` is `:input`, `:stage` or
`:rejuvenation`. `diagnostics` is the rejected stage summary, or `nothing` when
the input itself was collapsed. A rejected stage is not appended to `state.history`.

This detects full spatial collapse, not partial degeneracy or inadequate mixing.
Constant density ratios, a constant coordinate, a single initial ancestor and
zero movement alone do not trigger it. Zero-dimensional targets are exempt.
Use independently prepared tuning with [`rejuvenate_smc`](@ref) to attempt
recovery; a failure must not be treated as a completed posterior.
"""
struct SMCDegeneracyError{S,D} <: Exception
    state::S
    phase::Symbol
    diagnostics::D
end

function Base.showerror(io::IO, err::SMCDegeneracyError)
    print(io, "SMCDegeneracyError: all ", size(err.state.positions, 2),
        " particles coincide (", err.phase, "), input temperature=", err.state.temperature,
        ". Input state preserved in error.state; inspect error.diagnostics and use ",
        "rejuvenate_smc with independently prepared tuning to attempt recovery.")
end
Base.show(io::IO, err::SMCDegeneracyError) = showerror(io, err)
Base.showerror(io::IO, err::SMCDegeneracyError, bt; backtrace=true) = showerror(io, err)

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
An exactly collapsed positive-dimensional cloud throws [`SMCDegeneracyError`](@ref),
including when supplied at temperature 1. Reaching temperature 1 does not otherwise
certify convergence or adequate exploration.
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
`evaluations` sums the transition evaluation counts; `mean_acceptance` averages
the transitions' tree acceptance probabilities (`nothing` for zero dimensions).
`mean_squared_movement`
is the mean squared unconstrained displacement from each resampled parent to
its final child, after all moves. These describe mutation work and movement,
not independent evidence that a posterior has been explored adequately.
`coordinate_sd` gives native unconstrained-coordinate standard deviations of
the post-mutation, equally weighted cloud; `reference_sd` gives the initial
Gaussian reference's marginal standard deviations, or `nothing` for an exact
posterior-to-posterior update. These vectors retain O(d*history_limit) values,
not historical clouds. Compare spreads across stages and independent runs;
neither a large first temperature step nor a high ESS alone diagnoses coverage.
`nthreads` records the number of mutation workers, capped by available default
threads and particles (one for a zero-dimensional target). Exact diagnostic
replay requires the same available thread count as well as the same options.
Failures propagate; the input state and its RNG remain usable.
Full spatial collapse is checked before temperature selection and after mutation,
including the final stage. It throws [`SMCDegeneracyError`](@ref) with the preserved
input and, for a failed mutation, the rejected stage's diagnostics. Summary `kind`
is `:temper` for this operation or `:rejuvenate` for explicit fixed-temperature work.
"""
function smc_step end

"""
    rejuvenate_smc(state; tuning::SMCTuningSchedule, moves::Integer)

Attempt recovery or additional exploration using exactly `moves` fixed-kernel
NUTS transitions per particle at `state.temperature`. `tuning` must be prepared
independently of the production particles, for these exact source/target endpoints
and reference. Supply pilot initial positions and RNGs independently; include
knots near the temperature needing recovery when calling `tune_smc`.

There is no weighting, resampling, temperature advance, normalizer update, or
movement-based stopping rule. Input state and RNG are preserved. The result owns
new positions and RNG, identity `ancestors`, copied weights/lineage, and the newly
validated schedule for subsequent `smc` stages. The requested `moves` overrides
the selected kernel's move count for this call only. A bounded history entry with
`kind=:rejuvenate` records the actual work and zero normalizer increment; `stage`
still counts tempering stages, so recovery does not spend `max_stages`.
Checkpoint/resume retains the new schedule and exact continuation without retuning.

A still-collapsed result throws `SMCDegeneracyError` with the input state and the
attempt's diagnostics. No automatic retry is made. Distinct output particles do
not prove mixing; compare independent fits and posterior diagnostics. Rejuvenation
cannot repair errors in normalizer increments accumulated before this call.
Zero-dimensional targets perform no transitions and consume no RNG.
"""
function rejuvenate_smc end

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
