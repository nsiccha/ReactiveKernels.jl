# The outer continuation is data only. The active sampler payload remains
# WarmupHMC's public checkpoint representation, never a second sampler state.
const _TUNING_CHECKPOINT_FILE = "tuning_latest.jls"

function _tuning_source(target, previous, reference)
    if previous === nothing
        return _reference(target, reference)
    end
    previous isa AbstractSMCTarget || throw(ArgumentError(
        "previous must be a prepared target, not a particle cloud"))
    _compatible(previous.signature, target.signature)
    return _execution_target(previous)
end

function _write_tuning_checkpoint(directory, checkpoint)
    directory === nothing && return nothing
    mkpath(directory)
    path, io = mktemp(directory)
    try
        serialize(io, checkpoint)
        close(io)
        # rename replaces one directory entry atomically; mv(force=true) can
        # remove the previous checkpoint before installing its replacement.
        Base.Filesystem.rename(path, joinpath(directory, _TUNING_CHECKPOINT_FILE))
    finally
        isopen(io) && close(io)
        isfile(path) && rm(path)
    end
    return nothing
end

function _capture_tuning(cp, source, target; warmup=cp.warmup)
    position = warmup === nothing ? cp.position : warmup.position_and_gradient.q
    positions = [cp.config.init, position]
    if cp.config.grid_start === :independent && cp.phase !== :reference
        for pilot in cp.pilots
            pilot.warmup === nothing && continue
            push!(positions, pilot.warmup.position_and_gradient.q)
        end
    end
    probes = reduce(hcat, positions)
    return deepcopy(merge(cp, (; warmup, probes,
        source_scores=_scores(source, probes, :source, _tuning_temperature(cp)),
        target_scores=_scores(target, probes, :target, _tuning_temperature(cp)))))
end

_tuning_temperature(cp) = cp.phase === :reference || cp.phase === :complete ?
    1.0 : cp.config.temperatures[cp.knot]

function _tuning_event(cp, boundary; window=0, retained_draws=0, evaluations=0)
    return (; phase=cp.phase, knot=cp.phase === :reference ? 0 : cp.knot,
        temperature=_tuning_temperature(cp), boundary,
        completed_knots=length(cp.kernels), window=Int(window),
        retained_draws=Int(retained_draws), evaluations=Int(evaluations))
end

function _record_tuning(cp, event)
    push!(cp.history, event)
    length(cp.history) > cp.config.history_limit && popfirst!(cp.history)
    return cp
end

function _validate_tuning_checkpoint(cp, target, previous, config)
    cp isa NamedTuple && hasproperty(cp, :kind) && cp.kind === :smc_tuning &&
        hasproperty(cp, :schema_version) &&
        cp.schema_version == (config.grid_start === :independent ? 2 : 1) ||
        throw(ArgumentError("unsupported SMC tuning checkpoint"))
    cp.versions == _versions() || throw(ArgumentError(
        "SMC tuning checkpoint Julia/package versions differ"))
    _same(cp.target, target.signature) || throw(ArgumentError(
        "SMC tuning checkpoint target/model/binding/frame differs"))
    _same(cp.config, config) || throw(ArgumentError(
        "SMC tuning checkpoint configuration differs; pass the original tuning keywords"))
    _same(cp.previous, previous === nothing ? nothing : previous.signature) ||
        throw(ArgumentError("SMC tuning checkpoint previous endpoint differs"))
    source = _tuning_source(target, previous, cp.reference)
    valid_phase = if config.grid_start === :independent && cp.phase !== :reference
        _valid_grid_checkpoint(cp, _dimension(target))
    else
        (cp.phase === :reference ? cp.knot == 1 && config.reference === :pilot :
        cp.phase === :grid ? 1 <= cp.knot <= length(config.temperatures) :
        cp.phase === :complete && cp.knot == length(config.temperatures) + 1 &&
            cp.warmup === nothing && cp.seed === nothing) &&
            length(cp.kernels) == length(cp.work) == cp.knot - 1
    end
    valid_phase ||
        throw(ArgumentError("invalid SMC tuning checkpoint phase"))
    foreach(kernel -> _validate_kernel(kernel, _dimension(target)), cp.kernels)
    _scores(source, cp.probes, :source, _tuning_temperature(cp)) == cp.source_scores &&
        _scores(target, cp.probes, :target, _tuning_temperature(cp)) == cp.target_scores ||
        throw(ArgumentError("SMC tuning checkpoint endpoint densities differ; use the original code and data"))
    return source
end

function _tuning_schedule(cp, source)
    c = cp.config
    return SMCTuningSchedule(deepcopy(_source_id(source)), deepcopy(cp.target),
        source isa ReferenceTarget ? deepcopy(source.reference) : nothing,
        copy(c.temperatures), deepcopy(cp.kernels),
        (; initial_rng=copy(cp.initial_rng), final_rng=copy(cp.rng), init=copy(c.init),
            steps=c.pilot_steps, work=deepcopy(cp.work), method=:adaptive,
            max_reflections=c.max_reflections, recording_target=c.recording_target,
            pilot_evaluations=c.pilot_evaluations, reference_work=deepcopy(cp.reference_work),
            grid_start=c.grid_start))
end

function _run_tuning_pilot(bridge, cp, progress, callback)
    c = cp.config
    kwargs = (; init=(; position=copy(cp.position), squared_scale=c.scale.^2),
        n_draws=c.pilot_steps, n_evaluations=c.pilot_evaluations,
        recording_target=c.recording_target, max_reflections=c.max_reflections,
        max_tree_depth=c.max_tree_depth, nonlinear_adapt=false, monitor_ess=false,
        progress, callback)
    cp.warmup === nothing && return WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(cp.seed), bridge; kwargs...)
    # The public resume entry point reads cp_latest.jls. Materialize only that
    # payload in an owned temporary directory, keeping the durable outer/inner
    # checkpoint one atomic record. No target or runtime handle is serialized.
    return mktempdir() do directory
        serialize(joinpath(directory, "cp_latest.jls"), cp.warmup)
        wrapped = function (state, boundary)
            # WarmupHMC checkpoints before calling us. Retain its latest file
            # only; otherwise numbered windows accumulate during a long resume.
            for path in readdir(directory; join=true)
                name = basename(path)
                (name == "cp_init.jls" || occursin(r"^cp_window_[0-9]+\.jls$", name)) && rm(path)
            end
            return callback(state, boundary)
        end
        WarmupHMC.adaptive_warmup_mcmc(Xoshiro(cp.seed), bridge;
            kwargs..., callback=wrapped, checkpoint_dir=directory, resume=true)
    end
end

function _tune_adaptive(target; rng, init, reference, previous,
        reference_inflation, grid, pilot_steps, max_reflections,
        recording_target, pilot_evaluations, options, scale,
        progress, callback, checkpoint_dir, resume, overwrite, grid_start)
    resume isa Bool || resume isa SMCTuningPartial || throw(ArgumentError(
        "resume must be a Bool or SMCTuningPartial"))
    resume === false || !overwrite || throw(ArgumentError("resume and overwrite cannot be combined"))
    overwrite && checkpoint_dir === nothing && throw(ArgumentError("overwrite needs checkpoint_dir"))
    reference === :pilot && previous !== nothing && throw(ArgumentError(
        "reference=:pilot requires an initial bridge"))
    reference === nothing || reference === :pilot || reference isa SMCReference ||
        throw(ArgumentError("reference must be SMCReference, :pilot, or nothing"))
    if previous !== nothing
        previous isa AbstractSMCTarget || throw(ArgumentError(
            "previous must be a prepared target, not a particle cloud"))
        reference === nothing || throw(ArgumentError("posterior update tuning takes no reference"))
    end
    config = deepcopy((; init=Vector{Float64}(init), reference,
        reference_inflation=Float64(reference_inflation), temperatures=grid,
        pilot_steps=Int(pilot_steps), max_reflections=Int(max_reflections),
        recording_target=Int(recording_target), pilot_evaluations=Int(pilot_evaluations),
        scale, moves=options.moves, max_tree_depth=options.max_tree_depth,
        history_limit=options.history_limit, grid_start))
    path = checkpoint_dir === nothing ? nothing :
        joinpath(checkpoint_dir, _TUNING_CHECKPOINT_FILE)
    if resume === true
        path !== nothing && isfile(path) || throw(ArgumentError(
            "resume=true needs checkpoint_dir containing tuning_latest.jls"))
    elseif resume === false && path !== nothing && isfile(path) && !overwrite
        throw(ArgumentError("SMC tuning checkpoint already exists; use resume=true or overwrite=true"))
    end
    cp = if resume === false
        d = _dimension(target)
        initial_reference = previous !== nothing ? nothing : reference === :pilot ?
            SMCReference(zeros(d), ones(d)) : reference === nothing ?
            deepcopy(_default_reference(target)) : deepcopy(reference)
        (; kind=:smc_tuning, schema_version=grid_start === :independent ? 2 : 1, versions=_versions(),
            target=deepcopy(target.signature),
            previous=previous === nothing ? nothing : deepcopy(previous.signature),
            config, initial_rng=copy(rng), rng=copy(rng),
            reference=initial_reference, reference_work=nothing,
            kernels=NamedTuple[], work=NamedTuple[], history=NamedTuple[],
            phase=reference === :pilot ? :reference : :grid, knot=1,
            position=Vector{Float64}(init), seed=nothing, warmup=nothing,
            pilots=NamedTuple[])
    else
        deepcopy(resume isa SMCTuningPartial ? resume.checkpoint : deserialize(path))
    end
    source = resume === false ? _tuning_source(target, previous, cp.reference) :
        _validate_tuning_checkpoint(cp, target, previous, config)
    _options(options, _dimension(target), source, target)
    observing = callback !== nothing || progress !== nothing ||
        checkpoint_dir !== nothing || resume !== false
    return with_progress(progress; description="SMC tuning") do tuning_progress
        while cp.phase !== :complete
            if interrupt_requested(tuning_progress)
                event = _tuning_event(cp, :pending)
                cp = _capture_tuning(_record_tuning(cp, event), source, target)
                _write_tuning_checkpoint(checkpoint_dir, cp)
                update_progress!(tuning_progress; result=:partial)
                return SMCTuningPartial(cp, event)
            end
            if cp.seed === nothing
                if cp.phase === :grid && grid_start === :independent
                    return _tune_independent_grid(cp, source, target, options,
                        tuning_progress, callback, checkpoint_dir, observing)
                end
                cp = merge(cp, (; seed=rand(cp.rng, UInt64)))
            end
            if observing
                cp = _capture_tuning(cp, source, target)
                _write_tuning_checkpoint(checkpoint_dir, cp)
            end
            beta = _tuning_temperature(cp)
            bridge = BridgeTarget(source, target, beta, _dimension(target))
            label = cp.phase === :reference ? "Reference pilot" :
                "Grid pilot $(cp.knot)/$(length(grid)) (temperature=$beta)"
            stopped = Ref(false)
            result = with_progress(tuning_progress; description=label, transient=true) do pilot_progress
                hook = function (state, boundary)
                    payload = WarmupHMC.checkpoint_payload(state)
                    event = _tuning_event(cp, boundary; window=payload.outer_counter,
                        retained_draws=size(payload.posterior_position, 2),
                        evaluations=payload.total_evaluation_counter)
                    cp = _capture_tuning(_record_tuning(cp, event), source, target;
                        warmup=payload)
                    _write_tuning_checkpoint(checkpoint_dir, cp)
                    update_progress!(tuning_progress; phase=label, boundary,
                        window=event.window, evaluations=event.evaluations,
                        completed_knots=event.completed_knots)
                    requested = callback !== nothing && callback(event, boundary) === true
                    stopped[] = requested || interrupt_requested(pilot_progress)
                    return stopped[]
                end
                value = _run_tuning_pilot(bridge, cp, pilot_progress, observing ? hook : nothing)
                stopped[] |= interrupt_requested(pilot_progress)
                return value
            end
            if stopped[] || size(result.posterior_position, 2) < pilot_steps
                # Do not fit a reference, export a kernel, or consume the next
                # seed from an unfinished sampler, including a zero-draw init.
                cp.warmup === nothing && error("adaptive pilot stopped without a checkpoint boundary")
                update_progress!(tuning_progress; result=:partial)
                return SMCTuningPartial(deepcopy(cp), last(cp.history))
            end
            learned = _finish_adaptive_pilot(result, cp.seed;
                reference_inflation=cp.phase === :reference ? reference_inflation : nothing)
            event = _tuning_event(cp, :pilot; retained_draws=learned.work.retained_draws,
                evaluations=learned.work.evaluations)
            if cp.phase === :reference
                cp = merge(cp, (; reference=SMCReference(learned.location, learned.reference_scale),
                    reference_work=learned.work, phase=:grid, seed=nothing, warmup=nothing))
                source = _tuning_source(target, previous, cp.reference)
            else
                push!(cp.kernels, _validate_kernel((; stepsize=learned.stepsize,
                    metric=learned.scale, moves=options.moves,
                    max_tree_depth=options.max_tree_depth), _dimension(target)))
                push!(cp.work, learned.work)
                knot = cp.knot + 1
                cp = merge(cp, (; knot, phase=knot > length(grid) ? :complete : :grid,
                    position=learned.position, seed=nothing, warmup=nothing))
            end
            event = merge(event, (; completed_knots=length(cp.kernels)))
            _record_tuning(cp, event)
            if observing
                cp = _capture_tuning(cp, source, target)
                _write_tuning_checkpoint(checkpoint_dir, cp)
            end
            update_progress!(tuning_progress; completed_knots=length(cp.kernels),
                phase=label, boundary=:pilot)
            requested = callback !== nothing && callback(event, :pilot) === true
            if cp.phase !== :complete && (requested || interrupt_requested(tuning_progress))
                update_progress!(tuning_progress; result=:partial)
                return SMCTuningPartial(deepcopy(cp), event)
            end
        end
        update_progress!(tuning_progress; result=:ready)
        return _tuning_schedule(cp, source)
    end
end
