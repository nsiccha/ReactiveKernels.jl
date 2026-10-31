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
    SMCOptions(; ess_fraction=0.8, stepsize=0.25, metric=nothing,
               moves=2, max_tree_depth=6, max_stages=1000,
               min_increment=1e-12, history_limit=128)

Adaptive bridge and fixed NUTS configuration. `metric` is a vector of positive
diagonal **scales**, not variances or precisions; `nothing` means ones. The
metric, step size and tree depth stay fixed during mutation, with no adaptation
using the particles being moved. Each stage resamples, then takes `moves`
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
    function SMCOptions(; ess_fraction=0.8, stepsize=0.25, metric=nothing,
            moves=2, max_tree_depth=6, max_stages=1000, min_increment=1e-12,
            history_limit=128)
        0 < ess_fraction < 1 || throw(ArgumentError("ess_fraction must be between 0 and 1"))
        isfinite(stepsize) && stepsize > 0 || throw(ArgumentError("stepsize must be finite and positive"))
        metric === nothing || (metric isa AbstractVector && all(x -> isfinite(x) && x > 0, metric)) ||
            throw(ArgumentError("metric must be a vector of finite positive scales"))
        moves >= 1 || throw(ArgumentError("moves must be positive"))
        max_tree_depth >= 1 || throw(ArgumentError("max_tree_depth must be positive"))
        max_stages >= 1 || throw(ArgumentError("max_stages must be positive"))
        0 < min_increment < 1 || throw(ArgumentError("min_increment must be between 0 and 1"))
        history_limit >= 1 || throw(ArgumentError("history_limit must be positive"))
        new(ess_fraction, stepsize, metric === nothing ? nothing : Vector{Float64}(metric),
            moves, max_tree_depth, max_stages, min_increment, history_limit)
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
    initialize_smc(target; rng, particles=1000,
        reference=SMCReference(zeros(d), ones(d)), options=SMCOptions())

Draw the initial reference cloud, at temperature zero. Copies `rng` (an
explicit `Random.Xoshiro`) and never advances the caller's RNG. The returned
state owns its particle positions, normalized weights and continuation RNG.
Call `smc_step` for one completed stage, or `smc` to finish the bridge.
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
