# Independent grid pilots retain their own numerical continuations. Worker
# slots are execution resources only; they never choose a seed or a start.
function _valid_grid_checkpoint(cp, d)
    pilots = cp.pilots
    if isempty(pilots)
        return cp.phase === :grid && cp.knot == 1 && isempty(cp.kernels) &&
            isempty(cp.work) && cp.seed === nothing && cp.warmup === nothing
    end
    length(pilots) == length(cp.config.temperatures) || return false
    for p in pilots
        p.seed isa UInt64 || return false
        if p.kernel === nothing
            p.work === nothing || return false
        else
            p.work !== nothing && p.warmup === nothing || return false
            _validate_kernel(p.kernel, d)
        end
    end
    pending = findfirst(p -> p.kernel === nothing, pilots)
    return cp.seed === nothing && cp.warmup === nothing &&
        cp.knot == (pending === nothing ? length(pilots) + 1 : pending) &&
        cp.phase === (pending === nothing ? :complete : :grid) &&
        _same(cp.kernels, [p.kernel for p in pilots if p.kernel !== nothing]) &&
        _same(cp.work, [p.work for p in pilots if p.kernel !== nothing])
end

function _grid_tuning_event(cp, i, boundary; kwargs...)
    return _tuning_event(merge(cp, (; phase=:grid, knot=i)), boundary; kwargs...)
end

function _finish_grid_slot(cp, i, learned)
    kernel = _validate_kernel((; stepsize=learned.stepsize, metric=learned.scale,
        moves=cp.config.moves, max_tree_depth=cp.config.max_tree_depth),
        length(cp.config.init))
    cp.pilots[i] = merge(cp.pilots[i], (; warmup=nothing, kernel, work=learned.work))
    pending = findfirst(p -> p.kernel === nothing, cp.pilots)
    return merge(cp, (; kernels=NamedTuple[p.kernel for p in cp.pilots if p.kernel !== nothing],
        work=NamedTuple[p.work for p in cp.pilots if p.kernel !== nothing],
        knot=pending === nothing ? length(cp.pilots) + 1 : pending,
        phase=pending === nothing ? :complete : :grid))
end

function _tune_independent_grid(cp, source, target, options, tuning_progress,
        callback, checkpoint_dir, observing)
    if isempty(cp.pilots)
        cp = merge(cp, (; pilots=NamedTuple[(; seed=rand(cp.rng, UInt64),
            warmup=nothing, kernel=nothing, work=nothing) for _ in cp.config.temperatures]))
    end
    if observing
        cp = _capture_tuning(cp, source, target)
        _write_tuning_checkpoint(checkpoint_dir, cp)
    end
    pending = findall(p -> p.kernel === nothing, cp.pilots)
    workers = min(options.nthreads, Threads.nthreads(), length(pending))
    # Construct factories serially too: the factory need not support concurrent
    # calls, while each resulting evaluator must be exclusively owned.
    bridge = BridgeTarget(source, target, first(cp.config.temperatures), _dimension(target))
    targets = [_worker_target(bridge) for _ in 1:workers]
    jobs = Channel{Int}(length(pending))
    foreach(i -> put!(jobs, i), pending)
    close(jobs)
    guard = ReentrantLock()
    cancelled = Threads.Atomic{Bool}(false)
    stop_event = Ref{Union{Nothing,NamedTuple}}(nothing)
    tasks = Task[]
    for worker in 1:workers
        push!(tasks, Threads.@spawn try
            for i in jobs
                cancelled[] && break
                local_cp = lock(guard) do
                    p = cp.pilots[i]
                    merge(cp, (; knot=i, position=copy(cp.config.init),
                        seed=p.seed, warmup=p.warmup))
                end
                beta = local_cp.config.temperatures[i]
                t = targets[worker]
                bridge_i = BridgeTarget(t.source, t.target, beta, t.dimension)
                label = "Grid pilot $i/$(length(local_cp.config.temperatures)) (temperature=$beta)"
                stopped = Ref(false)
                result = with_progress(tuning_progress; description=label, transient=true) do pilot_progress
                    hook = function (state, boundary)
                        if !observing
                            stopped[] = cancelled[]
                            return stopped[]
                        end
                        # The running sampler will mutate its own arrays again;
                        # the published continuation must own its snapshot.
                        payload = deepcopy(WarmupHMC.checkpoint_payload(state))
                        lock(guard) do
                            cp.pilots[i] = merge(cp.pilots[i], (; warmup=payload))
                            event = _grid_tuning_event(cp, i, boundary;
                                window=payload.outer_counter,
                                retained_draws=size(payload.posterior_position, 2),
                                evaluations=payload.total_evaluation_counter)
                            _record_tuning(cp, event)
                            if checkpoint_dir !== nothing
                                _write_tuning_checkpoint(checkpoint_dir, _capture_tuning(cp, source, target))
                            end
                            update_progress!(tuning_progress; phase=label, boundary,
                                window=event.window, evaluations=event.evaluations,
                                completed_knots=event.completed_knots)
                            requested = callback !== nothing && callback(event, boundary) === true
                            if requested || interrupt_requested(pilot_progress)
                                cancelled[] = true
                                stop_event[] === nothing && (stop_event[] = event)
                            end
                            stopped[] = cancelled[]
                            return stopped[]
                        end
                    end
                    _run_tuning_pilot(bridge_i, local_cp, pilot_progress, hook)
                end
                if stopped[] || size(result.posterior_position, 2) < local_cp.config.pilot_steps
                    cancelled[] = true
                    continue
                end
                learned = _finish_adaptive_pilot(result, local_cp.seed)
                lock(guard) do
                    cp = _finish_grid_slot(cp, i, learned)
                    event = _grid_tuning_event(cp, i, :pilot;
                        retained_draws=learned.work.retained_draws,
                        evaluations=learned.work.evaluations)
                    _record_tuning(cp, event)
                    if checkpoint_dir !== nothing
                        _write_tuning_checkpoint(checkpoint_dir, _capture_tuning(cp, source, target))
                    end
                    update_progress!(tuning_progress; completed_knots=length(cp.kernels),
                        phase=label, boundary=:pilot)
                    requested = callback !== nothing && callback(event, :pilot) === true
                    if requested || interrupt_requested(tuning_progress)
                        cancelled[] = true
                        stop_event[] === nothing && (stop_event[] = event)
                    end
                end
            end
        catch err
            cancelled[] = true
            err isa SMCNonfiniteError && return err
            rethrow()
        end)
    end
    # No task may keep evaluating targets or writing checkpoints after return,
    # including when another task's sampler, callback or checkpoint write fails.
    for task in tasks
        try
            wait(task)
        catch err
            err isa TaskFailedException || rethrow()
        end
    end
    for result in map(fetch, tasks)
        result isa SMCNonfiniteError && throw(result)
    end
    if cp.phase === :complete
        update_progress!(tuning_progress; result=:ready)
        return _tuning_schedule(cp, source)
    end
    event = stop_event[] === nothing ? _tuning_event(cp, :pending) : stop_event[]
    snapshot = _capture_tuning(cp, source, target)
    _write_tuning_checkpoint(checkpoint_dir, snapshot)
    update_progress!(tuning_progress; result=:partial)
    return SMCTuningPartial(snapshot, event)
end
