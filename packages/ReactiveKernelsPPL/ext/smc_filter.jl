# Included by the SMC extension. Finite-state transitions and conditional
# resampling remain in the authored filtering extension.
_same(a::SMCCarriedReference,b::SMCCarriedReference) =
    _same(a.reference,b.reference) && _same(a.frame,b.frame) && a.calibration_id==b.calibration_id &&
    _same(a.fit_diagnostics,b.fit_diagnostics)

struct FilterReferenceTarget{Q,B,S,R} <: AbstractSMCTarget
    query::Q
    backend::B
    signature::S
    prior::R
end
_density(t::FilterReferenceTarget,u) = _density(t.prior,u)
_gradient(t::FilterReferenceTarget,g,u) = _gradient(t.prior,g,u)
_default_reference(t::FilterReferenceTarget) = deepcopy(t.prior.reference)

struct SMCFilterTarget{T,C,F,R,S}
    filter::T
    calibration::C
    parameters::F
    proposal::R
    signature::S
end

function _filter_reference(calibration, backend)
    d = length(calibration.reference.location)
    query = (; layout=(; total=d))
    signature = (; model_id=calibration.frame.model_id,layout=calibration.frame.layout,
        kind=:filter_reference, calibration_id=calibration.calibration_id,
        reference=calibration.reference)
    prototype = SMCTarget(query,backend,signature)
    prior = _reference(prototype,calibration.reference)
    return FilterReferenceTarget(query,backend,signature,prior)
end

function PPL.prepare_filter_smc(filter, calibration::SMCCarriedReference;
        parameters=nothing, parameter_id=nothing, backend,
        calibration_model_id::AbstractString=filter.signature.model_id)
    if parameters===nothing
        parameters=PPL._smc_parameter_map(calibration)
        parameter_id===nothing && (parameter_id="carried-native-geometry/v1")
    end
    parameter_id isa AbstractString && !isempty(parameter_id) ||
        throw(ArgumentError("parameter_id must identify the coordinate mapping"))
    calibration_model_id == calibration.frame.model_id ||
        throw(ArgumentError("calibration model/preprocessing identity differs from the expected calibration_model_id"))
    proposal = _filter_reference(calibration,backend)
    mapped = parameters(copy(calibration.reference.location))
    typeof(mapped) == typeof(filter.parameters) ||
        throw(ArgumentError("parameter mapping must return the filter's complete parameter schema"))
    signature = deepcopy((; filter=filter.signature, calibration,
        calibration_model_id=String(calibration_model_id),
        parameter_id=String(parameter_id)))
    return SMCFilterTarget(filter,deepcopy(calibration),parameters,proposal,signature)
end

PPL.tune_filter_smc(target::SMCFilterTarget;kwargs...) = tune_smc(target.proposal;kwargs...)

struct SMCFilterStage
    kind::Symbol
    time::Int
    window::Int
    window_steps::Int
    predictive_logscore::Float64
    parameter_ess_before::Float64
    parameter_ess_after::Float64
    distinct_ess_before::Float64
    distinct_ess_after::Float64
    replenished::Bool
    boundary_reset::Bool
    unique_parents::Int
    proposals::Int
    accepted::Int
    likelihood_evaluations::Int
    reference_evaluations::Int
    distinct_members::Int
    moved_members::Int
    coordinate_sd::Vector{Float64}
    kernel::NamedTuple
    proposal_scale::Vector{Float64}
end

# Anchored moments give exactly zero for a constant coordinate, even when a
# weighted sum of identical large values would round away from that value.
function _filter_moments(positions,mass)
    anchor = positions[:,1]
    offset = zeros(length(anchor))
    for j in axes(positions,2), i in eachindex(anchor)
        offset[i] += mass[j]*(positions[i,j]-anchor[i])
    end
    variance = zeros(length(anchor))
    for j in axes(positions,2), i in eachindex(anchor)
        variance[i] += mass[j]*abs2((positions[i,j]-anchor[i])-offset[i])
    end
    return anchor+offset,sqrt.(variance)
end

# Group bitwise-identical parameter columns (signed zeros merged), summing their
# masses. Hash each column once: O(members*coordinates), never pairwise.
function _filter_groups(positions,mass)
    groups = Dict{Vector{Float64},Float64}()
    for j in axes(positions,2)
        key = map(x -> iszero(x) ? 0.0 : x,positions[:,j])
        groups[key] = get(groups,key,0.0)+mass[j]
    end
    return groups
end
_filter_distinct(positions) = length(_filter_groups(positions,zeros(size(positions,2))))
# Copies of one parameter value count once: member ESS cannot see that
# resampled duplicates with identical conditional scores share their weight.
_filter_distinct_ess(positions,mass) = inv(sum(abs2,values(_filter_groups(positions,mass))))

function _filter_history(state,stage)
    history = [state.history;stage]
    length(history)>state.history_limit && deleteat!(history,1:length(history)-state.history_limit)
    return history
end
Base.:(==)(a::SMCFilterStage,b::SMCFilterStage) =
    all(getfield(a,k)==getfield(b,k) for k in fieldnames(SMCFilterStage))

struct SMCFilterState{T,C,R,M,B}
    target::T
    calibration::C
    proposal::R
    positions::Matrix{Float64}
    members::Vector{M}
    parameter_mass::Vector{Float64}
    log_parameter_mass::Vector{Float64}
    boundary::B
    records::Vector{NamedTuple}
    rng::Xoshiro
    time::Int
    window::Int
    window_length::Int
    ess_fraction::Float64
    kernel::NamedTuple
    history::Vector{SMCFilterStage}
    history_limit::Int
    predictive_logscore::Float64
    log_normalizer::Float64
end

PPL.tune_filter_smc(state::SMCFilterState;kwargs...) = tune_smc(state.proposal;kwargs...)

function Base.showerror(io::IO, err::PPL.SMCDegeneracyError{<:SMCFilterState})
    state = err.state
    print(io, "SMCDegeneracyError: learning-filter degeneracy (", err.phase,
        "), input time=", state.time, ", window=", state.window,
        ", members=", length(state.members))
    diagnostic = err.diagnostics
    if diagnostic !== nothing
        invalid = findall(x -> !(isfinite(x) && x>0),diagnostic.coordinate_sd)
        print(io, ", invalid marginal scales=", length(invalid))
        if diagnostic isa SMCFilterStage
            print(io, ", distinct=", diagnostic.distinct_members,
                ", distinct ESS=", round(diagnostic.distinct_ess_after;sigdigits=3),
                ", moved=", diagnostic.moved_members, ", accepted=", diagnostic.accepted,
                "/", diagnostic.proposals)
        end
    end
    print(io, ". Input state/RNG preserved in error.state; a failed filter_step has not ",
        "consumed its observation. Inspect error.diagnostics and its kernel; use ",
        "tune_filter_smc(error.state; rng, init) with independent pilot inputs and ",
        "rejuvenate_smc(error.state; tuning, moves), then retry the observation. ",
        "Distinct members do not establish mixing.")
end

function _filter_cloud(target,calibration,boundary,rng,n,k)
    d = length(calibration.reference.location)
    positions = _reference_draws(rng,calibration.reference,d,k)
    members = map(axes(positions,2)) do j
        parameters = target.parameters(copy(positions[:,j]))
        PPL._filter_boundary_initialize(target.filter,boundary;
            rng=Xoshiro(rand(rng,UInt64)),particles=n,parameters)
    end
    return positions,members
end

function PPL.initialize_filter_smc(target::SMCFilterTarget; rng::Xoshiro,
        members::Integer=64,particles::Integer=128,window_length::Integer=32,
        ess_fraction=0.5,options=SMCOptions(),history_limit::Integer=32,boundary=nothing)
    members >= 2 || throw(ArgumentError("parameter learning needs at least two members"))
    particles >= 1 || throw(ArgumentError("at least one state particle is required"))
    window_length >= 1 || throw(ArgumentError("window_length must be positive"))
    0 < ess_fraction <= 1 || throw(ArgumentError("ess_fraction must be in (0,1]"))
    history_limit >= 0 || throw(ArgumentError("history_limit must be nonnegative"))
    d = length(target.calibration.reference.location)
    d > 0 || throw(ArgumentError("parameter learning needs an unconstrained coordinate"))
    options = _options(options,d,target.proposal.prior,target.proposal)
    kernel,_ = _stage_kernel(options,d,1.0)
    r = copy(rng)
    pool = PPL._filter_initial_boundary(target.filter,boundary,r,Int(particles))
    positions,clouds = _filter_cloud(target,target.calibration,pool,r,Int(particles),Int(members))
    return SMCFilterState(target,deepcopy(target.calibration),target.proposal,positions,clouds,
        fill(1/members,members),fill(-log(members),members),pool,NamedTuple[],r,
        pool === nothing ? 0 : pool.time,1,Int(window_length),Float64(ess_fraction),deepcopy(kernel),SMCFilterStage[],
        Int(history_limit),0.0,0.0)
end

function _filter_logsum(values)
    all(x -> isfinite(x) || x == -Inf,values) ||
        throw(ArgumentError("invalid parameter likelihood estimate"))
    m = maximum(values)
    m > -Inf || throw(ArgumentError("all parameter members have zero window likelihood"))
    return m+log(sum(x -> exp(x-m),values))
end

function _window_filter(target,boundary,records,u,rng,n)
    state = PPL._filter_boundary_initialize(target.filter,boundary;
        parameters=target.parameters(copy(u)),rng,particles=n)
    evaluations = 0
    for record in records
        evaluations += n
        state = PPL._filter_step_or_zero(state,record.observation;
            inputs=record.inputs,groups=record.groups)
        state === nothing && return nothing,-Inf,evaluations
    end
    return state,state.log_normalizer,evaluations
end

function _filter_rollover(state,rng)
    location,coordinate_sd = _filter_moments(state.positions,state.parameter_mass)
    invalid_coordinates = findall(x -> !(isfinite(x) && x>0),coordinate_sd)
    isempty(invalid_coordinates) && all(isfinite,location) ||
        throw(PPL.SMCDegeneracyError(state,:rollover,
            (;coordinate_sd,invalid_coordinates,kernel=deepcopy(state.kernel))))
    reference = SMCReference(location,coordinate_sd)
    boundary = PPL._filter_boundary(state.members,state.parameter_mass,rng)
    calibration = SMCCarriedReference(reference,deepcopy(state.calibration.frame),
        string("filter-window-",string(state.window+1;base=16,pad=16)))
    proposal = _filter_reference(calibration,state.proposal.backend)
    positions,members = _filter_cloud(state.target,calibration,boundary,rng,
        first(state.members).particles,length(state.members))
    return calibration,proposal,boundary,positions,members
end

# Explicit recovery only: fixed NUTS proposals reversible for the window's
# Gaussian prior, independent of the members' (possibly vanished) spread.
function _filter_move!(positions,members,target,proposal,boundary,records,kernel,rng)
    n,k = first(members).particles,length(members)
    scores = [m.log_normalizer for m in members]
    accepted,moved,evaluations,reference_evaluations = 0,0,0,0
    reference = BridgeTarget(proposal.prior,proposal,1.0,size(positions,1))
    prototype = FixedKernelNUTSState(Xoshiro(0),positions[:,1];
        metric=_metric(kernel.metric),stepsize=kernel.stepsize,max_tree_depth=kernel.max_tree_depth)
    for j in 1:k
        initial = positions[:,j]
        for _ in 1:kernel.moves
            chain = FixedKernelNUTSState(prototype; rng=Xoshiro(rand(rng,UInt64)),position=positions[:,j])
            transition = fixed_kernel_nuts_step(chain,reference)
            reference_evaluations += transition.stats.steps
            u = transition.state.position
            candidate,score,work = _window_filter(target,boundary,records,u,
                Xoshiro(rand(rng,UInt64)),n)
            evaluations += work
            # The proposal is reversible for the window's Gaussian prior.
            # Keep the previous likelihood estimate AND cloud on rejection.
            if log(rand(rng)) < score-scores[j]
                positions[:,j] = u
                members[j] = candidate
                scores[j] = score
                accepted += 1
            end
        end
        moved += positions[:,j] != initial
    end
    return accepted,moved,evaluations,reference_evaluations
end

# Random-walk scale for a d-dimensional Gaussian target (Roberts, Gelman and
# Gilks 1997), applied per coordinate to the population's weighted spread.
_filter_walk_scale(sd) = (2.38/sqrt(length(sd))) .* sd

# Population-scaled particle-Metropolis moves (the SMC² move of Chopin, Jacob
# and Papaspiliopoulos). The scale is fixed for the whole replenishment, so
# each move is invariant for prior × window likelihood estimate.
function _filter_walk!(positions,members,target,proposal,boundary,records,moves,scale,rng)
    n,k,d = first(members).particles,length(members),size(positions,1)
    scores = [m.log_normalizer for m in members]
    priors = [_density(proposal.prior,positions[:,j]) for j in 1:k]
    accepted,moved,evaluations = 0,0,0
    for j in 1:k
        initial = positions[:,j]
        for _ in 1:moves
            u = positions[:,j] .+ scale .* randn(rng,d)
            prior = _density(proposal.prior,u)
            candidate,score,work = _window_filter(target,boundary,records,u,
                Xoshiro(rand(rng,UInt64)),n)
            evaluations += work
            # Symmetric proposal: the prior ratio times the window likelihood
            # estimate ratio. Keep the previous estimate AND cloud on rejection.
            if log(rand(rng)) < (prior-priors[j])+(score-scores[j])
                positions[:,j] = u
                members[j] = candidate
                scores[j] = score
                priors[j] = prior
                accepted += 1
            end
        end
        moved += positions[:,j] != initial
    end
    return accepted,moved,evaluations
end

function _filter_replenish(positions,members,logmass,target,proposal,boundary,records,kernel,rng)
    # Scale from the weighted population before resampling discards its spread.
    _,spread = _filter_moments(positions,exp.(logmass))
    scale = _filter_walk_scale(spread)
    parents,_,_ = PPL._filter_resample(logmass,rand(rng,UInt64))
    positions = positions[:,parents]
    members = [PPL._filter_with_rng(members[j],Xoshiro(rand(rng,UInt64))) for j in parents]
    accepted,moved,evaluations =
        _filter_walk!(positions,members,target,proposal,boundary,records,kernel.moves,scale,rng)
    return positions,members,length(unique(parents)),accepted,moved,evaluations,scale
end

function PPL.filter_step(state::SMCFilterState,observation::NamedTuple;
        inputs::NamedTuple=(;),groups=nothing)
    _check_cloud(state,state.positions,:input)
    rng = copy(state.rng)
    reset = length(state.records) == state.window_length
    calibration,proposal,boundary,positions,members = reset ? _filter_rollover(state,rng) :
        (state.calibration,state.proposal,state.boundary,state.positions,state.members)
    k,n = length(members),first(members).particles
    logmass = reset ? fill(-log(k),k) : copy(state.log_parameter_mass)
    records = reset ? NamedTuple[] : copy(state.records)
    push!(records,deepcopy((;observation,inputs,groups)))
    nextmembers = similar(members)
    for j in 1:k
        next = PPL._filter_step_or_zero(members[j],observation;inputs,groups)
        if next === nothing
            # A zero-weight member has no conditional posterior. It cannot be
            # selected for a PMMH move; preserve only its bounded old state
            # until the replenishment below replaces the ensemble.
            nextmembers[j] = members[j]
            logmass[j] = -Inf
        else
            logmass[j] += next.predictive_logscore
            nextmembers[j] = next
        end
    end
    score = _filter_logsum(logmass)
    logmass .-= score
    mass = exp.(logmass)
    ess = inv(sum(abs2,mass))
    distinct_ess = _filter_distinct_ess(positions,mass)
    replenish = distinct_ess < state.ess_fraction*k || any(==(-Inf),logmass)
    unique_parents,accepted,moved = k,0,0
    proposal_scale = Float64[]
    evaluations = k*n
    if replenish
        positions,nextmembers,unique_parents,accepted,moved,work,proposal_scale =
            _filter_replenish(positions,nextmembers,logmass,state.target,proposal,boundary,records,
                state.kernel,rng)
        evaluations += work
        mass,logmass = fill(1/k,k),fill(-log(k),k)
    end
    total = state.log_normalizer+score
    isfinite(total) || throw(ArgumentError("streaming filter normalizer overflow"))
    window = state.window+Int(reset)
    _,coordinate_sd = _filter_moments(positions,mass)
    stage = SMCFilterStage(:assimilate,state.time+1,window,length(records),score,ess,
        inv(sum(abs2,mass)),distinct_ess,_filter_distinct_ess(positions,mass),replenish,reset,
        unique_parents,replenish ? k*state.kernel.moves : 0,accepted,evaluations,0,
        _filter_distinct(positions),moved,coordinate_sd,deepcopy(state.kernel),proposal_scale)
    _check_cloud(state,positions,:filter_step,stage)
    return SMCFilterState(state.target,calibration,proposal,positions,nextmembers,
        mass,logmass,boundary,records,rng,state.time+1,window,state.window_length,
        state.ess_fraction,state.kernel,_filter_history(state,stage),state.history_limit,score,total)
end

function PPL.rejuvenate_smc(state::SMCFilterState; tuning::SMCTuningSchedule,moves::Integer)
    moves>=1 || throw(ArgumentError("rejuvenation moves must be positive"))
    options = _options(SMCOptions(;tuning),size(state.positions,1),state.proposal.prior,state.proposal)
    nextkernel,_ = _stage_kernel(options,size(state.positions,1),1.0)
    kernel = merge(nextkernel,(;moves=Int(moves)))
    rng = copy(state.rng)
    positions = copy(state.positions)
    members = [PPL._filter_with_rng(m,copy(m.rng)) for m in state.members]
    accepted,moved,evaluations,reference_evaluations = _filter_move!(positions,members,
        state.target,state.proposal,state.boundary,state.records,kernel,rng)
    _,coordinate_sd = _filter_moments(positions,state.parameter_mass)
    k = length(members)
    ess = inv(sum(abs2,state.parameter_mass))
    stage = SMCFilterStage(:rejuvenate,state.time,state.window,length(state.records),0.0,
        ess,ess,_filter_distinct_ess(state.positions,state.parameter_mass),
        _filter_distinct_ess(positions,state.parameter_mass),false,false,k,k*kernel.moves,
        accepted,evaluations,reference_evaluations,_filter_distinct(positions),moved,
        coordinate_sd,deepcopy(kernel),Float64[])
    _check_cloud(state,positions,:rejuvenation,stage)
    # At a full boundary, a recovery that still cannot supply a Gaussian
    # marginal must fail here, before a subsequent rollover consumes data.
    if length(state.records)==state.window_length && !all(x -> isfinite(x) && x>0,coordinate_sd)
        throw(PPL.SMCDegeneracyError(state,:rejuvenation,stage))
    end
    return SMCFilterState(state.target,state.calibration,state.proposal,positions,members,
        copy(state.parameter_mass),copy(state.log_parameter_mass),deepcopy(state.boundary),
        deepcopy(state.records),rng,state.time,state.window,state.window_length,state.ess_fraction,
        deepcopy(nextkernel),_filter_history(state,stage),state.history_limit,
        state.predictive_logscore,state.log_normalizer)
end

function PPL.filter_checkpoint(state::SMCFilterState)
    return deepcopy((; schema_version=2,kind=:parameter_learning,versions=_versions(),
        signature=state.target.signature,calibration=state.calibration,
        positions=state.positions,members=map(filter_checkpoint,state.members),
        parameter_mass=state.parameter_mass,log_parameter_mass=state.log_parameter_mass,
        boundary=state.boundary,records=state.records,rng=state.rng,time=state.time,
        window=state.window,window_length=state.window_length,
        ess_fraction=state.ess_fraction,kernel=state.kernel,history=state.history,
        history_limit=state.history_limit,predictive_logscore=state.predictive_logscore,
        log_normalizer=state.log_normalizer))
end

function PPL.resume_filter(target::SMCFilterTarget,checkpoint)
    c = deepcopy(checkpoint)
    c.schema_version==2 && c.kind===:parameter_learning && c.versions==_versions() ||
        throw(ArgumentError("parameter filter checkpoint schema/versions differ"))
    _same(c.signature,target.signature) ||
        throw(ArgumentError("parameter filter checkpoint model, mapping or frame differs"))
    _compatible(c.calibration.frame,target.calibration.frame)
    d,k = size(c.positions)
    d == length(c.calibration.reference.location) && k >= 2 && all(isfinite,c.positions) ||
        throw(ArgumentError("invalid parameter checkpoint positions"))
    length(c.members)==length(c.parameter_mass)==length(c.log_parameter_mass)==k ||
        throw(ArgumentError("invalid parameter checkpoint ensemble size"))
    all(x -> isfinite(x) && x >= 0,c.parameter_mass) &&
        isapprox(sum(c.parameter_mass),1.0;atol=1e-12) &&
        isapprox(_filter_logsum(c.log_parameter_mass),0.0;atol=1e-12) &&
        all(isapprox.(exp.(c.log_parameter_mass),c.parameter_mass;atol=1e-15)) ||
        throw(ArgumentError("invalid parameter checkpoint masses"))
    c.window_length>=1 && 0<=length(c.records)<=c.window_length && c.window>=1 && c.time>=0 ||
        throw(ArgumentError("invalid parameter checkpoint window"))
    c.history_limit>=0 && length(c.history)<=c.history_limit &&
        0<c.ess_fraction<=1 && isfinite(c.log_normalizer) && isfinite(c.predictive_logscore) &&
        c.rng isa Xoshiro || throw(ArgumentError("invalid parameter checkpoint diagnostics/RNG"))
    _validate_kernel(c.kernel,d)
    proposal = _filter_reference(c.calibration,target.proposal.backend)
    members = map(c.members) do member
        resume_filter(target.filter,member)
    end
    all(m -> m.time==c.time && length(m.ensemble)==1 && m.particles==first(members).particles,members) ||
        throw(ArgumentError("invalid conditional checkpoint clocks or sizes"))
    for j in 1:k
        _same(target.parameters(copy(c.positions[:,j])),only(members[j].ensemble)) ||
            throw(ArgumentError("checkpoint parameter mapping differs at a retained position"))
    end
    boundary_time = c.boundary === nothing ? 0 : c.boundary.time
    boundary_time+length(c.records)==c.time ||
        throw(ArgumentError("checkpoint boundary and window clocks differ"))
    # Validate the pooled state distribution through its owning extension.
    PPL._filter_boundary_initialize(target.filter,c.boundary;
        parameters=only(first(members).ensemble),rng=Xoshiro(0),particles=1)
    return SMCFilterState(target,c.calibration,proposal,c.positions,members,
        c.parameter_mass,c.log_parameter_mass,c.boundary,c.records,c.rng,c.time,
        c.window,c.window_length,c.ess_fraction,c.kernel,c.history,c.history_limit,
        c.predictive_logscore,c.log_normalizer)
end

function PPL.filter_stream(state::SMCFilterState,records;steps::Integer=typemax(Int),stop=state->false)
    steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    iterator_state = nothing
    first_record = true
    for _ in 1:steps
        stop(state) && break
        item = first_record ? iterate(records) : iterate(records,iterator_state)
        item === nothing && break
        record,iterator_state = item
        first_record = false
        state = filter_step(state,record.observation;inputs=get(record,:inputs,(;)),
            groups=get(record,:groups,nothing))
    end
    return state
end
