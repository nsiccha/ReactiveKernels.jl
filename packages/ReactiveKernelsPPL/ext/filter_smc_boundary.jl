# Included by the authored filtering extension. These operations preserve the
# existing FilterState and its validation/transition implementation.
PPL._filter_resample(scores,seed) = _resample(scores,seed)

PPL._filter_initial_boundary(target,::Nothing,rng,n) = nothing
function PPL._filter_initial_boundary(target::FilterTarget,state::FilterState,rng,n)
    _same(target.signature,state.target.signature) ||
        _fail("initial boundary belongs to a different filter")
    parents,_,_ = _resample(log.(state.weights),rand(rng,UInt64))
    selected = parents[rand(rng,1:length(parents),n)]
    return (; carries=deepcopy(state.carries[selected]),time=state.time,
        group_times=deepcopy(state.group_times))
end

function PPL._filter_with_rng(state::FilterState,rng::Xoshiro)
    return FilterState(state.target,deepcopy(state.carries),deepcopy(state.ensemble),
        copy(state.parameter_mass),copy(state.log_parameter_mass),copy(state.weights),
        state.particles,copy(rng),state.time,state.predictive_logscore,state.log_normalizer,
        copy(state.ancestors),copy(state.lineage),copy(state.history),state.history_limit,
        deepcopy(state.group_times))
end

function PPL._filter_boundary(members,masses,rng)
    first_member = first(members)
    all(m -> m.time==first_member.time && _same(m.group_times,first_member.group_times),members) ||
        _fail("conditional state clocks differ at the window boundary")
    weights = [masses[j]*weight for (j,member) in enumerate(members) for weight in member.weights]
    parents,_,_ = _resample(log.(weights),rand(rng,UInt64))
    n = first_member.particles
    # Uniformly subsample the library's whole resampled population. Taking
    # its first N entries would bias an ordered/systematic resampler. A pool
    # of N states keeps each proposed member's boundary work independent of M.
    selected = parents[rand(rng,1:length(parents),n)]
    carries = [deepcopy(members[div(i-1,n)+1].carries[mod(i-1,n)+1]) for i in selected]
    return (; carries,time=first_member.time,
        group_times=deepcopy(first_member.group_times))
end

function PPL._filter_boundary_initialize(target::FilterTarget,boundary;
        rng::Xoshiro,particles::Integer,parameters)
    state = initialize_filter(target;rng,particles,ensemble=[parameters],history_limit=0)
    boundary === nothing && return state
    boundary.time >= 0 || _fail("invalid boundary time")
    !isempty(boundary.carries) || _fail("invalid boundary cloud size")
    for carry in boundary.carries
        keys(carry)==keys(target.prototype) &&
            all(size(a)==size(b) && all(isfinite,a) for (a,b) in zip(carry,target.prototype)) ||
            _fail("invalid boundary carry shape/value")
    end
    keys(boundary.group_times)==keys(target.signature.groups) &&
        all(length(t)==length(g) && all(x -> 0<=x<=boundary.time,t)
            for (t,g) in zip(boundary.group_times,target.signature.groups)) ||
        _fail("invalid boundary group clocks")
    r = copy(rng)
    parents = rand(r,1:length(boundary.carries),particles)
    carries = [deepcopy(boundary.carries[j]) for j in parents]
    return FilterState(target,carries,state.ensemble,state.parameter_mass,
        state.log_parameter_mass,state.weights,state.particles,r,boundary.time,
        0.0,0.0,state.ancestors,state.lineage,state.history,state.history_limit,
        deepcopy(boundary.group_times))
end
