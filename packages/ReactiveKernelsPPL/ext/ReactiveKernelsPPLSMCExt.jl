module ReactiveKernelsPPLSMCExt

using ReactiveKernelsPPL
using ReactiveKernels
using Random
using LinearAlgebra: Diagonal
using SequentialMonteCarlo: SMCModel, SMCIO, smc!
using WarmupHMC: FixedKernelNUTSState, fixed_kernel_nuts_step
import WarmupHMC, SequentialMonteCarlo, LogDensityProblems
import ReactiveKernelsPPL as PPL

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

struct ReferenceTarget{K,A}
    reference::SMCReference
    kernel::K
    ad::A
end

function _reference(target, reference)
    length(reference.location) == target.query.layout.total ||
        throw(DimensionMismatch("reference and target dimensions differ"))
    r = SMCReference(reference.location, reference.scale)
    kernel = prepare(_normal_reference; want=:density)
    ad = Base.invokelatest(prepare_ad, kernel, target.backend,
        copy(r.location), r.location, r.scale; active=:u)
    return ReferenceTarget(r, kernel, ad)
end

_density(t::SMCTarget, u) = t.query(u)
_density(t::ReferenceTarget, u) = Base.invokelatest(t.kernel, u,
    t.reference.location, t.reference.scale)::Float64
_gradient(t::SMCTarget, g, u) = first(sampler_value_and_gradient!(t.query, g, u))
_gradient(t::ReferenceTarget, g, u) = first(Base.invokelatest(ad_value_and_gradient!,
    t.ad, g, u, t.reference.location, t.reference.scale))

function _checked(value)
    (isfinite(value) || value == -Inf) || throw(DomainError(value,
        "SMC target returned NaN or +Inf; use -Inf only for points outside support"))
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
    t.temperature == 1 && return _checked(_density(t.target, u))
    t.temperature == 0 && return _checked(_density(t.source, u))
    a, b = _checked(_density(t.source, u)), _checked(_density(t.target, u))
    return _checked((1-t.temperature) * a + t.temperature * b)
end

function LogDensityProblems.logdensity_and_gradient(t::BridgeTarget, u)
    # Sampler queries own scratch space and are used serially. Each returned
    # gradient is fresh: WarmupHMC may retain it in a trajectory.
    a, b = t.temperature == 1 ? (t.target, nothing) :
        t.temperature == 0 ? (t.source, nothing) : (t.source, t.target)
    g = similar(u)
    value = _checked(_gradient(a, g, u))
    if b !== nothing
        h = similar(u)
        other = _checked(_gradient(b, h, u))
        value = _checked((1-t.temperature) * value + t.temperature * other)
        g .= (1-t.temperature) .* g .+ t.temperature .* h
    end
    isfinite(value) && !all(isfinite, g) && throw(DomainError(g,
        "SMC target returned a non-finite gradient at a finite density"))
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
end

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

function _options(options, d)
    options.metric === nothing || length(options.metric) == d ||
        throw(DimensionMismatch("metric and target dimensions differ"))
    return deepcopy(options)
end

function PPL.initialize_smc(target::SMCTarget; rng::Xoshiro, particles::Integer=1000,
        reference=SMCReference(zeros(target.query.layout.total), ones(target.query.layout.total)),
        options=SMCOptions())
    particles >= 2 || throw(ArgumentError("SMC needs at least two particles"))
    source = _reference(target, reference)
    r = copy(rng)
    d, n = target.query.layout.total, Int(particles)
    options = _options(options, d)
    positions = reference.location .+ reference.scale .* randn(r, d, n)
    return SMCState(source, target, positions, fill(1/n, n), 0.0, 0.0,
        r, options, collect(1:n), collect(1:n), SMCStage[], 0, 0)
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
    target = BridgeTarget(state.source, state.target, beta, d)
    options = state.options
    metric = Diagonal(options.metric === nothing ? ones(d) : options.metric)
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
                end
                u = chain.position
            end
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
        smc!(model, output)
    end
    schedule(task)
    fetch(task) # propagate callback/library failures; no detached work
    return positions, ancestors, output.logZhats[end], n*output.esses[1], divergences[]
end

function PPL.smc_step(state::SMCState)
    state.temperature == 1 && return state
    state.stage < state.options.max_stages || throw(ArgumentError(
        "SMC reached max_stages before temperature 1; the input state is preserved"))
    d, n = size(state.positions)
    delta = Vector{Float64}(undef, n)
    for i in 1:n
        u = state.positions[:, i]
        a, b = _checked(_density(state.source, u)), _checked(_density(state.target, u))
        isfinite(a) || throw(ArgumentError("SMC source density is zero at particle $i"))
        delta[i] = _checked(b-a)
        isfinite(b) && !isfinite(delta[i]) && throw(DomainError(delta[i],
            "SMC log density ratio overflowed"))
    end
    beta = _temperature(delta, state.temperature, state.options)
    rng = copy(state.rng)
    positions, ancestors, increment, ess, divergences =
        _library_stage(state, beta, delta, rand(rng, UInt64))
    isfinite(increment) || throw(DomainError(increment, "non-finite SMC normalizer increment"))
    log_normalizer = state.log_normalizer+increment
    isfinite(log_normalizer) || throw(DomainError(log_normalizer, "SMC log normalizer overflowed"))
    lineage = state.lineage[ancestors]
    stage = SMCStage(state.update, state.stage+1, beta, ess, increment,
        length(Set(ancestors)), length(Set(lineage)), d == 0 ? 0 : n*state.options.moves, divergences)
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
        0.0, previous.log_normalizer, copy(previous.rng), _options(options,d),
        collect(1:n), copy(previous.lineage), copy(previous.history), previous.update+1, 0)
    return smc(state; stages)
end

_versions() = (; julia=VERSION, ppl=Base.pkgversion(PPL), rk=Base.pkgversion(ReactiveKernels),
    smc=Base.pkgversion(SequentialMonteCarlo), nuts=Base.pkgversion(WarmupHMC))
_source_id(t::SMCTarget) = (; kind=:posterior, signature=t.signature)
_source_id(t::ReferenceTarget) = (; kind=:reference, reference=t.reference)
_scores(t, positions) = [_checked(_density(t, positions[:, i])) for i in axes(positions,2)]

function PPL.smc_checkpoint(state::SMCState)
    return deepcopy((; schema_version=1, versions=_versions(),
        source=_source_id(state.source), target=state.target.signature,
        positions=state.positions, weights=state.weights, temperature=state.temperature,
        log_normalizer=state.log_normalizer, rng=state.rng, options=state.options,
        ancestors=state.ancestors, lineage=state.lineage, history=state.history,
        update=state.update, stage=state.stage,
        source_scores=_scores(state.source, state.positions),
        target_scores=_scores(state.target, state.positions)))
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
    _scores(source, cp.positions) == cp.source_scores &&
        _scores(target, cp.positions) == cp.target_scores ||
        throw(ArgumentError("SMC checkpoint endpoint densities differ; rebuild with the original code and data"))
    0 <= cp.temperature <= 1 && isfinite(cp.log_normalizer) ||
        throw(ArgumentError("invalid SMC checkpoint bridge state"))
    length(cp.weights) == n && all(==(1/n), cp.weights) ||
        throw(ArgumentError("SMC checkpoint must follow a completed resample/move stage"))
    length(cp.ancestors) == length(cp.lineage) == n &&
        all(i -> 1 <= i <= n, cp.ancestors) && all(i -> 1 <= i <= n, cp.lineage) ||
        throw(ArgumentError("invalid SMC checkpoint ancestry"))
    return SMCState(source, target, cp.positions, cp.weights, cp.temperature,
        cp.log_normalizer, cp.rng, _options(cp.options,d), cp.ancestors,
        cp.lineage, cp.history, cp.update, cp.stage)
end

end
