module ReactiveKernelsPPLSMCExt

using ReactiveKernelsPPL
using ReactiveKernels
using Random
using LinearAlgebra: Diagonal, logabsdet
using SequentialMonteCarlo: SMCModel, SMCIO, smc!
using WarmupHMC: FixedKernelNUTSState, fixed_kernel_nuts_step
using WarmupHMC: ContinuousNUTSState, continuous_nuts_step
import WarmupHMC, SequentialMonteCarlo, LogDensityProblems
import ReactiveKernelsPPL as PPL

PPL._smc_scale(scale::NamedTuple) = WarmupHMC.scale_data(WarmupHMC.scale_from_data(scale))
_metric(scale::AbstractVector) = Diagonal(scale)
_metric(scale::NamedTuple) = WarmupHMC.scale_from_data(scale)

struct SMCTarget{Q,B,S}
    query::Q
    backend::B
    signature::S
end

# Geometry metadata is structural; neither one entry per data row nor a dense
# covariance is introduced. The caller's model_id additionally covers external
# geometry/preprocessing semantics that cannot be inferred from coordinate names.
_sampling_id(::Nothing) = nothing
_sampling_id(s) = (; s.rhs, s.broadcast, s.constructor_broadcast)
_entry(e) = (e.kind, e.predictor, e.name, e.labels, e.offset, e.size,
    e.transform, e.lo, e.hi, e.dims, map(_entry, e.scan_blocks), _sampling_id(e.sampling))
_layout(layout) = (layout.total, map(_entry, layout.entries), layout.name_paths)

function PPL.prepare_smc(built, binding::StructuralPlan, u0::AbstractVector;
        backend, model_id::AbstractString, binding_id::AbstractString, kwargs...)
    isempty(model_id) && throw(ArgumentError("model_id must name the model and coordinate frame"))
    isempty(binding_id) && throw(ArgumentError("binding_id must name the exact bound data"))
    hasproperty(built, :program) || throw(ArgumentError("prepare_smc needs a build_kernel result"))
    query = prepare_sampler(built, binding, u0; backend, kwargs...)
    signature = deepcopy((; model_id=String(model_id), binding_id=String(binding_id),
        program=built.program, layout=_layout(query.layout)))
    return SMCTarget(query, backend, signature)
end

@kernel _normal_reference(u::Vector{Float64}, location::Vector{Float64}, scale::Vector{Float64}) = begin
    density = -length(u) * log(2pi) / 2 - sum(log.(scale)) -
        sum(abs2, (u .- location) ./ scale) / 2
end

@kernel _structured_normal_reference(u::Vector{Float64}, location::Vector{Float64},
        scale::AbstractMatrix{Float64}, logdet::Float64) = begin
    z = scale \ (u .- location)
    density = -length(u) * log(2pi) / 2 - logdet - sum(abs2, z) / 2
end

struct ReferenceTarget{R,K,A,C}
    reference::R
    kernel::K
    ad::A
    arguments::C
end

function _reference(target, reference)
    length(reference.location) == target.query.layout.total ||
        throw(DimensionMismatch("reference and target dimensions differ"))
    r = SMCReference(reference.location, reference.scale)
    kernel, arguments = if r.scale isa AbstractVector
        prepare(_normal_reference; want=:density), (r.location, r.scale)
    else
        scale = _metric(r.scale)
        prepare(_structured_normal_reference; want=:density),
            (r.location, scale, first(logabsdet(scale)))
    end
    ad = Base.invokelatest(prepare_ad, kernel, target.backend,
        copy(r.location), arguments...; active=:u)
    return ReferenceTarget(r, kernel, ad, arguments)
end

_density(t::SMCTarget, u) = t.query(u)
_density(t::ReferenceTarget, u) = Base.invokelatest(t.kernel, u, t.arguments...)::Float64
_gradient(t::SMCTarget, g, u) = first(sampler_value_and_gradient!(t.query, g, u))
_gradient(t::ReferenceTarget, g, u) = first(Base.invokelatest(ad_value_and_gradient!,
    t.ad, g, u, t.arguments...))

function _checked(value, u, endpoint, temperature, gradient=nothing)
    (isfinite(value) || value == -Inf) ||
        throw(SMCNonfiniteError(u, value, gradient, :density, endpoint, temperature))
    return value
end

function _checked_gradient(value, g, u, endpoint, temperature)
    _checked(value, u, endpoint, temperature, g)
    isfinite(value) && !all(isfinite, g) &&
        throw(SMCNonfiniteError(u, value, g, :gradient, endpoint, temperature))
    return value
end

struct BridgeTarget{A,B}
    source::A
    target::B
    temperature::Float64
    dimension::Int
end
LogDensityProblems.dimension(t::BridgeTarget) = t.dimension
LogDensityProblems.capabilities(::Type{<:BridgeTarget}) = LogDensityProblems.LogDensityOrder{1}()

# Endpoints branch explicitly: 0 * -Inf must never contaminate the other end.
function LogDensityProblems.logdensity(t::BridgeTarget, u)
    t.temperature == 1 && return _checked(_density(t.target, u), u, :target, t.temperature)
    t.temperature == 0 && return _checked(_density(t.source, u), u, :source, t.temperature)
    a = _checked(_density(t.source, u), u, :source, t.temperature)
    b = _checked(_density(t.target, u), u, :target, t.temperature)
    return _checked((1-t.temperature) * a + t.temperature * b, u, :bridge, t.temperature)
end

function LogDensityProblems.logdensity_and_gradient(t::BridgeTarget, u)
    # Sampler queries own scratch space and are used serially. Each returned
    # gradient is fresh: WarmupHMC may retain it in a trajectory.
    a, b = t.temperature == 1 ? (t.target, nothing) :
        t.temperature == 0 ? (t.source, nothing) : (t.source, t.target)
    g = similar(u)
    endpoint = t.temperature == 1 ? :target : :source
    value = _checked_gradient(_gradient(a, g, u), g, u, endpoint, t.temperature)
    if b !== nothing
        h = similar(u)
        other = _checked_gradient(_gradient(b, h, u), h, u, :target, t.temperature)
        value = (1-t.temperature) * value + t.temperature * other
        g .= (1-t.temperature) .* g .+ t.temperature .* h
        _checked_gradient(value, g, u, :bridge, t.temperature)
    end
    return value, g
end

struct SMCStage
    update::Int
    stage::Int
    temperature::Float64
    ess_before_resampling::Float64
    log_normalizer_increment::Float64
    unique_parents::Int
    surviving_ancestors::Int
    moves::Int
    divergences::Int
    kernel::NamedTuple
    tuning_index::Int
    evaluations::Int
    mean_squared_movement::Float64
    mean_acceptance::Union{Nothing,Float64}
end
Base.:(==)(a::SMCStage, b::SMCStage) =
    all(getfield(a, k) == getfield(b, k) for k in fieldnames(SMCStage))

struct SMCState{A,B}
    source::A
    target::B
    positions::Matrix{Float64}
    weights::Vector{Float64}
    temperature::Float64
    log_normalizer::Float64
    rng::Xoshiro
    options::SMCOptions
    ancestors::Vector{Int}
    lineage::Vector{Int}
    history::Vector{SMCStage}
    update::Int
    stage::Int
end

function _kernel(options, d)
    return (; stepsize=options.stepsize,
        metric=options.metric === nothing ? ones(d) : deepcopy(options.metric),
        moves=options.moves, max_tree_depth=options.max_tree_depth)
end

function _validate_kernel(kernel, d)
    scale = PPL._smc_scale(kernel.metric)
    PPL._smc_scale_dimension(scale) == d || throw(DimensionMismatch("scheduled metric dimension differs"))
    isfinite(kernel.stepsize) && kernel.stepsize > 0 ||
        throw(ArgumentError("scheduled step size must be finite and positive"))
    kernel.moves isa Integer && kernel.moves >= 1 || throw(ArgumentError("invalid scheduled move count"))
    kernel.max_tree_depth isa Integer && kernel.max_tree_depth >= 1 ||
        throw(ArgumentError("invalid scheduled tree depth"))
    return kernel
end

function _temperatures(temperatures)
    grid = Float64[temperatures...]
    length(grid) >= 2 && first(grid) == 0 && last(grid) == 1 &&
        all(isfinite, grid) && all(>(0), diff(grid)) ||
        throw(ArgumentError("tuning temperatures must strictly increase from 0 to 1"))
    return grid
end

function _options(options, d, source, target)
    options.metric === nothing || PPL._smc_scale_dimension(options.metric) == d ||
        throw(DimensionMismatch("metric and target dimensions differ"))
    _validate_kernel(_kernel(options, d), d)
    tuning = options.tuning
    if tuning !== nothing
        _temperatures(tuning.temperatures)
        length(tuning.temperatures) == length(tuning.kernels) ||
            throw(ArgumentError("tuning schedule has a missing kernel"))
        foreach(k -> _validate_kernel(k, d), tuning.kernels)
        _same(tuning.source, _source_id(source)) && _same(tuning.target, target.signature) ||
            throw(ArgumentError("SMC tuning belongs to a different bridge; rebuild it for these exact endpoints"))
        _same(tuning.reference, source isa ReferenceTarget ? source.reference : nothing) ||
            throw(ArgumentError("SMC tuning reference differs from its source"))
    end
    return deepcopy(options)
end

function _stage_kernel(options, d, beta)
    tuning = options.tuning
    tuning === nothing && return _kernel(options, d), 0
    upper = searchsortedfirst(tuning.temperatures, beta)
    i = upper <= 1 ? 1 : upper > length(tuning.temperatures) ? length(tuning.temperatures) :
        beta-tuning.temperatures[upper-1] < tuning.temperatures[upper]-beta ? upper-1 : upper
    return tuning.kernels[i], i
end

function PPL.tune_smc(target::SMCTarget; rng::Xoshiro, init::AbstractVector,
        reference=nothing, previous=nothing,
        temperatures=(0.0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0),
        pilot_steps::Integer=64, pilot::Symbol=:continuous,
        max_reflections::Integer=16, recording_target::Integer=128,
        pilot_evaluations::Integer=1000, options=SMCOptions())
    d = target.query.layout.total
    length(init) == d || throw(DimensionMismatch("pilot initial position and target dimensions differ"))
    d > 0 || throw(ArgumentError("tuning needs at least one unconstrained coordinate"))
    all(isfinite, init) || throw(ArgumentError("pilot initial position must be finite"))
    pilot_steps >= 1 || throw(ArgumentError("pilot_steps must be positive"))
    pilot in (:continuous, :adaptive) || throw(ArgumentError("pilot must be :continuous or :adaptive"))
    max_reflections >= 0 || throw(ArgumentError("max_reflections must be nonnegative"))
    recording_target >= 1 || throw(ArgumentError("recording_target must be positive"))
    pilot_evaluations >= 1 || throw(ArgumentError("pilot_evaluations must be positive"))
    options.tuning === nothing || throw(ArgumentError("pilot options must not already contain tuning"))
    grid = _temperatures(temperatures)
    scale = options.metric === nothing ? ones(d) : options.metric
    scale isa AbstractVector || throw(ArgumentError("pilot initialization requires diagonal scales"))
    _validate_kernel(_kernel(options, d), d)
    pilot_rng = copy(rng)
    reference_work = nothing
    if reference === :pilot
        previous === nothing && pilot === :adaptive || throw(ArgumentError(
            "reference=:pilot requires an initial bridge and pilot=:adaptive"))
        initial_source = _reference(target, SMCReference(zeros(d), ones(d)))
        learned = _adaptive_pilot(BridgeTarget(initial_source, target, 1.0, d),
            pilot_rng, init, scale, pilot_steps, max_reflections, recording_target,
            pilot_evaluations, options.max_tree_depth)
        reference = SMCReference(learned.location, learned.scale)
        reference_work = learned.work
    end
    reference === nothing || reference isa SMCReference ||
        throw(ArgumentError("reference must be SMCReference, :pilot, or nothing"))
    source = if previous === nothing
        _reference(target, reference === nothing ? SMCReference(zeros(d), ones(d)) : reference)
    else
        previous isa SMCTarget || throw(ArgumentError("previous must be a prepared target, not a particle cloud"))
        reference === nothing || throw(ArgumentError("posterior update tuning takes no reference"))
        _compatible(previous.signature, target.signature)
        previous
    end
    options = _options(options, d, source, target)
    if pilot === :adaptive
        kernels, work = NamedTuple[], NamedTuple[]
        position = Vector{Float64}(init)
        for beta in grid
            learned = _adaptive_pilot(BridgeTarget(source,target,beta,d), pilot_rng,
                position, scale, pilot_steps, max_reflections, recording_target,
                pilot_evaluations, options.max_tree_depth)
            push!(kernels, _validate_kernel((; stepsize=learned.stepsize,
                metric=learned.scale, moves=options.moves,
                max_tree_depth=options.max_tree_depth), d))
            push!(work, learned.work)
            position = learned.position
        end
        return SMCTuningSchedule(deepcopy(_source_id(source)), deepcopy(target.signature),
            source isa ReferenceTarget ? deepcopy(source.reference) : nothing,
            grid, kernels, (; initial_rng=copy(rng), final_rng=copy(pilot_rng),
                init=Vector{Float64}(init), steps=Int(pilot_steps), work,
                method=pilot, max_reflections=Int(max_reflections),
                recording_target=Int(recording_target),
                pilot_evaluations=Int(pilot_evaluations), reference_work))
    end
    chain = ContinuousNUTSState(copy(rng), BridgeTarget(source, target, first(grid), d);
        init=(; position=Vector{Float64}(init), squared_scale=scale.^2))
    kernels = typeof(_kernel(options, d))[]
    work = NamedTuple{(:transitions, :evaluations, :divergences),Tuple{Int,Int,Int}}[]
    for beta in grid
        bridge = BridgeTarget(source, target, beta, d)
        before = chain.total_evaluation_counter
        divergences = 0
        for _ in 1:pilot_steps
            transition = continuous_nuts_step(chain, bridge)
            divergences += transition.divergent
        end
        kernel = (; stepsize=Float64(chain.stepsize), metric=copy(chain.scale.diag),
            moves=options.moves, max_tree_depth=options.max_tree_depth)
        push!(kernels, _validate_kernel(kernel, d))
        push!(work, (; transitions=Int(pilot_steps),
            evaluations=Int(chain.total_evaluation_counter-before), divergences))
    end
    return SMCTuningSchedule(deepcopy(_source_id(source)), deepcopy(target.signature),
        source isa ReferenceTarget ? deepcopy(source.reference) : nothing,
        grid, kernels, (; initial_rng=copy(rng), final_rng=copy(chain.rng),
            init=Vector{Float64}(init), steps=Int(pilot_steps), work, method=pilot))
end

function _adaptive_pilot(bridge, rng, init, scale, steps, reflections, recording, evaluations, depth)
    seed = rand(rng, UInt64)
    result = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(seed), bridge;
        init=(; position=Vector{Float64}(init), squared_scale=scale.^2),
        n_draws=steps, n_evaluations=evaluations, recording_target=recording,
        max_reflections=reflections, max_tree_depth=depth,
        nonlinear_adapt=false, monitor_ess=false)
    data = WarmupHMC.scale_data(result.scale_options[result.active_transformation])
    draws = result.posterior_position
    return (; scale=data, stepsize=Float64(result.stepsize),
        location=vec(sum(draws; dims=2))./size(draws,2),
        position=Vector{Float64}(result.position_and_gradient.q),
        work=(; seed, retained_draws=size(draws,2),
            evaluations=Int(result.total_evaluation_counter),
            divergences=Int(result.n_divergent_samples),
            family=result.active_transformation))
end

PPL.smc_reference(state::SMCState) =
    state.source isa ReferenceTarget ? deepcopy(state.source.reference) : nothing

function PPL.initialize_smc(target::SMCTarget; rng::Xoshiro, particles::Integer=1000,
        reference=nothing, options=SMCOptions())
    particles >= 2 || throw(ArgumentError("SMC needs at least two particles"))
    if reference === nothing
        reference = options.tuning === nothing ?
            SMCReference(zeros(target.query.layout.total), ones(target.query.layout.total)) :
            smc_reference(options.tuning)
        reference === nothing && throw(ArgumentError("posterior update tuning cannot initialize a reference cloud"))
    end
    source = _reference(target, reference)
    r = copy(rng)
    d, n = target.query.layout.total, Int(particles)
    options = _options(options, d, source, target)
    positions = _reference_draws(r, source.reference, d, n)
    return SMCState(source, target, positions, fill(1/n, n), 0.0, 0.0,
        r, options, collect(1:n), collect(1:n), SMCStage[], 0, 0)
end

function _reference_draws(rng, reference::SMCReference{<:AbstractVector}, d, n)
    return reference.location .+ reference.scale .* randn(rng, d, n)
end
function _reference_draws(rng, reference::SMCReference{<:NamedTuple}, d, n)
    scale = _metric(reference.scale)
    positions = Matrix{Float64}(undef, d, n)
    for i in 1:n
        positions[:, i] = reference.location .+ scale * randn(rng, d)
    end
    return positions
end

# Stable ESS for the equal-weight input cloud. This calculation chooses a
# temperature; SequentialMonteCarlo performs the actual weighting/resampling.
function _ess(delta, increment)
    shift = maximum(delta)
    shift == -Inf && throw(ArgumentError("all SMC particles have zero target weight"))
    s = 0.0
    s2 = 0.0
    for x in delta
        w = exp(increment * x - increment * shift)
        s += w
        s2 += w*w
    end
    return s*s/s2
end

function _temperature(delta, beta, options)
    remaining = 1-beta
    threshold = options.ess_fraction * length(delta)
    _ess(delta, remaining) >= threshold && return 1.0
    lo, hi = 0.0, remaining
    for _ in 1:60
        mid = (lo+hi)/2
        if _ess(delta, mid) >= threshold
            lo = mid
        else
            hi = mid
        end
    end
    lo >= min(options.min_increment, remaining) && beta+lo > beta ||
        throw(ArgumentError("adaptive SMC bridge cannot advance at the requested ESS; " *
            "increase the cloud/support overlap or deliberately lower ess_fraction"))
    return beta+lo
end

# Particles carry indices only. Library ancestor copies cannot alias parameter
# buffers, and zero-argument constructors do not need a data-sized type parameter.
mutable struct IndexParticle
    index::Int
    logweight::Float64
end
IndexParticle() = IndexParticle(0, 0.0)

function _library_stage(state, beta, delta, seed)
    d, n = size(state.positions)
    positions = similar(state.positions)
    ancestors = Vector{Int}(undef, n)
    starts, finishes, divergences = Ref(0), Ref(0), Ref(0)
    evaluations, movement, acceptance = Ref(0), Ref(0.0), Ref(0.0)
    target = BridgeTarget(state.source, state.target, beta, d)
    options, _ = _stage_kernel(state.options, d, beta)
    metric = _metric(options.metric)
    mutation! = function (out, rng, step, parent, scratch)
        if step == 1
            i = starts[] += 1
            out.index = i
            out.logweight = (beta-state.temperature) * delta[i]
        else
            i = finishes[] += 1
            ancestors[i] = parent.index
            u = copy(state.positions[:, parent.index])
            if d > 0
                chain = FixedKernelNUTSState(Xoshiro(rand(rng, UInt64)), u;
                    metric, stepsize=options.stepsize, max_tree_depth=options.max_tree_depth)
                for _ in 1:options.moves
                    transition = fixed_kernel_nuts_step(chain, target)
                    chain = transition.state
                    divergences[] += transition.divergent
                    evaluations[] += transition.stats.steps
                    acceptance[] += transition.stats.acceptance_rate
                end
                u = chain.position
            end
            movement[] += sum(abs2, u .- view(state.positions, :, parent.index))
            positions[:, i] = u
            out.index = i
            out.logweight = 0.0
        end
        return nothing
    end
    potential = (step, particle, scratch) -> particle.logweight
    model = SMCModel(mutation!, potential, 2, IndexParticle, Nothing)
    output = SMCIO{IndexParticle,Nothing}(n, 2, 1, false)
    # SMC 0.7 has no rng keyword. Seed only this owned child task. On our
    # Julia >=1.10 floor task creation does not advance the parent's RNG.
    task = @task begin
        Random.seed!(seed)
        try
            smc!(model, output)
            nothing
        catch err
            # Transport only our numerical diagnostic as data. Other failures
            # keep their ordinary task exception and original backtrace.
            err isa SMCNonfiniteError || rethrow()
            err
        end
    end
    schedule(task)
    result = fetch(task) # join every worker before returning or throwing
    result isa SMCNonfiniteError && throw(result)
    return positions, ancestors, output.logZhats[end], n*output.esses[1],
        divergences[], evaluations[], movement[]/n,
        d == 0 ? nothing : acceptance[]/(n*options.moves)
end

function PPL.smc_step(state::SMCState)
    state.temperature == 1 && return state
    state.stage < state.options.max_stages || throw(ArgumentError(
        "SMC reached max_stages before temperature 1; the input state is preserved"))
    d, n = size(state.positions)
    delta = Vector{Float64}(undef, n)
    for i in 1:n
        u = state.positions[:, i]
        a = _checked(_density(state.source, u), u, :source, state.temperature)
        b = _checked(_density(state.target, u), u, :target, state.temperature)
        isfinite(a) || throw(ArgumentError("SMC source density is zero at particle $i"))
        delta[i] = b-a
        isfinite(b) && !isfinite(delta[i]) &&
            throw(SMCNonfiniteError(u, delta[i], nothing, :logratio, :bridge, state.temperature))
    end
    beta = _temperature(delta, state.temperature, state.options)
    rng = copy(state.rng)
    positions, ancestors, increment, ess, divergences, evaluations, movement, acceptance =
        _library_stage(state, beta, delta, rand(rng, UInt64))
    isfinite(increment) || throw(DomainError(increment, "non-finite SMC normalizer increment"))
    log_normalizer = state.log_normalizer+increment
    isfinite(log_normalizer) || throw(DomainError(log_normalizer, "SMC log normalizer overflowed"))
    lineage = state.lineage[ancestors]
    kernel, tuning_index = _stage_kernel(state.options, d, beta)
    stage = SMCStage(state.update, state.stage+1, beta, ess, increment,
        length(Set(ancestors)), length(Set(lineage)), d == 0 ? 0 : n*kernel.moves,
        divergences, deepcopy(kernel), tuning_index, evaluations, movement, acceptance)
    history = [state.history; stage]
    if length(history) > state.options.history_limit
        deleteat!(history, 1:length(history)-state.options.history_limit)
    end
    return SMCState(state.source, state.target, positions, fill(1/n, n), beta,
        log_normalizer, rng, state.options, ancestors, lineage,
        history, state.update, state.stage+1)
end

function PPL.smc(state::SMCState; stages::Integer=typemax(Int))
    stages >= 0 || throw(ArgumentError("stages must be nonnegative"))
    for _ in 1:stages
        state.temperature == 1 && break
        state = smc_step(state)
    end
    return state
end
PPL.smc(target::SMCTarget; stages::Integer=typemax(Int), kwargs...) =
    smc(initialize_smc(target; kwargs...); stages)

# Compare nested signature containers without depending on Expr's embedded-array
# identity equality or Dict iteration order.
_same(a, b) = PPL._same_program(a, b)
_same(a::SMCReference, b::SMCReference) = a.location == b.location && a.scale == b.scale
_same(a::Tuple, b::Tuple) = length(a) == length(b) && all(_same(x,y) for (x,y) in zip(a,b))
_same(a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) && _same(values(a), values(b))
_same(a::AbstractVector, b::AbstractVector) = axes(a) == axes(b) && all(_same(x,y) for (x,y) in zip(a,b))
_same(a::AbstractDict, b::AbstractDict) = keys(a) == keys(b) && all(_same(a[k],b[k]) for k in keys(a))

function _compatible(a, b)
    # A new history may generate a different program (e.g. a longer scan).
    # Each target is checked against its own binding by prepare_sampler.
    # model_id identifies the authored model; layout checks the coordinates.
    a.model_id == b.model_id && _same(a.layout, b.layout) ||
        throw(ArgumentError("SMC update changes the model or source coordinate frame; start a new cloud"))
end

function PPL.update_smc(previous::SMCState, target::SMCTarget;
        options=previous.options, stages::Integer=typemax(Int))
    previous.temperature == 1 || throw(ArgumentError("complete the previous bridge before updating data"))
    _compatible(previous.target.signature, target.signature)
    d, n = size(previous.positions)
    state = SMCState(previous.target, target, copy(previous.positions), copy(previous.weights),
        0.0, previous.log_normalizer, copy(previous.rng), _options(options,d,previous.target,target),
        collect(1:n), copy(previous.lineage), copy(previous.history), previous.update+1, 0)
    return smc(state; stages)
end

_versions() = (; julia=VERSION, ppl=Base.pkgversion(PPL), rk=Base.pkgversion(ReactiveKernels),
    smc=Base.pkgversion(SequentialMonteCarlo), nuts=Base.pkgversion(WarmupHMC))
_source_id(t::SMCTarget) = (; kind=:posterior, signature=t.signature)
_source_id(t::ReferenceTarget) = (; kind=:reference, reference=t.reference)
function _scores(t, positions, endpoint, temperature)
    return map(axes(positions, 2)) do i
        u = positions[:, i]
        _checked(_density(t, u), u, endpoint, temperature)
    end
end

function PPL.smc_checkpoint(state::SMCState)
    return deepcopy((; schema_version=1, versions=_versions(),
        source=_source_id(state.source), target=state.target.signature,
        positions=state.positions, weights=state.weights, temperature=state.temperature,
        log_normalizer=state.log_normalizer, rng=state.rng, options=state.options,
        ancestors=state.ancestors, lineage=state.lineage, history=state.history,
        update=state.update, stage=state.stage,
        source_scores=_scores(state.source, state.positions, :source, state.temperature),
        target_scores=_scores(state.target, state.positions, :target, state.temperature)))
end

function PPL.resume_smc(target::SMCTarget, checkpoint; previous=nothing)
    cp = deepcopy(checkpoint)
    cp.schema_version == 1 || throw(ArgumentError("unsupported SMC checkpoint schema"))
    cp.versions == _versions() || throw(ArgumentError("SMC checkpoint Julia/package versions differ"))
    _same(cp.target, target.signature) || throw(ArgumentError("SMC checkpoint target/model/binding differs"))
    d, n = size(cp.positions)
    d == target.query.layout.total && n >= 2 && all(isfinite, cp.positions) ||
        throw(ArgumentError("invalid SMC checkpoint particle positions"))
    source = if cp.source.kind === :reference
        previous === nothing || throw(ArgumentError("a reference checkpoint takes no previous posterior"))
        _reference(target, cp.source.reference)
    elseif cp.source.kind === :posterior
        previous isa SMCTarget || throw(ArgumentError("data-update checkpoint needs the previous prepared target"))
        _same(cp.source.signature, previous.signature) || throw(ArgumentError("SMC checkpoint previous binding differs"))
        previous
    else
        throw(ArgumentError("unknown SMC checkpoint source"))
    end
    _scores(source, cp.positions, :source, cp.temperature) == cp.source_scores &&
        _scores(target, cp.positions, :target, cp.temperature) == cp.target_scores ||
        throw(ArgumentError("SMC checkpoint endpoint densities differ; rebuild with the original code and data"))
    0 <= cp.temperature <= 1 && isfinite(cp.log_normalizer) ||
        throw(ArgumentError("invalid SMC checkpoint bridge state"))
    length(cp.weights) == n && all(==(1/n), cp.weights) ||
        throw(ArgumentError("SMC checkpoint must follow a completed resample/move stage"))
    length(cp.ancestors) == length(cp.lineage) == n &&
        all(i -> 1 <= i <= n, cp.ancestors) && all(i -> 1 <= i <= n, cp.lineage) ||
        throw(ArgumentError("invalid SMC checkpoint ancestry"))
    return SMCState(source, target, cp.positions, cp.weights, cp.temperature,
        cp.log_normalizer, cp.rng, _options(cp.options,d,source,target), cp.ancestors,
        cp.lineage, cp.history, cp.update, cp.stage)
end

end
