module MutationProfileBGenericNUTSSupport

using ReactiveKernels
using ReactiveKernelsNUTSExamples
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "..", "benchmark",
                 "nuts_kernel_authoring_fixture_b.jl"))

const RK = ReactiveKernels
const NEX = ReactiveKernelsNUTSExamples
const F = NUTSBMutationAuthoringFixture

potential(position) = sum(abs2, position) / 2
# The factory uses the destination form; stateless derived-field repairs use
# the functional form of the same model-gradient authority.
gradient(position) = (potential(position), copy(position))
gradient(destination, position) = begin
    destination .= position
    potential(position)
end

struct EndpointAuthority <: Function end
struct StatisticsAuthority <: Function end
(::EndpointAuthority)(args...; kwargs...) = error("functional only")
(::StatisticsAuthority)(args...; kwargs...) = error("functional only")

struct LeapfrogLowering{T}
    transition::T
end
function (lowering::LeapfrogLowering)(effect, point)
    (arguments=(lowering.transition(point),), result=nothing,
     effect_state=effect)
end

function statistics_lowering(effect, state)
    n_steps = state.n_steps + one(state.n_steps)
    unit = one(state.dham)
    rate = (unit - unit / n_steps) * state.acceptance_rate +
        (unit / n_steps) *
        ifelse(state.dham >= zero(state.dham), unit, exp(state.dham))
    updated = merge(state, (; n_steps, acceptance_rate=rate))
    (arguments=(updated,), result=nothing,
     effect_state=(; n_steps, acceptance_rate=rate))
end

_unmeasured(f, label) = f()

function _generic_program(max_depth=2; measure=_unmeasured,
                          stepsize=0.1, replay=nothing)
    endpoint = measure("endpoint") do
        RK.compile_state_transition(
        F.euclidean_phasepoint, RK.partial(F.leapfrog!; stepsize),
        (potential, gradient, Diagonal([1.0]), [1.0], [0.25]))
    end
    point = RK.initial_transition_state(endpoint)
    structured = RK.structured_state_port(endpoint)
    step_source = EndpointAuthority()
    step_port = RK.effect_lowering_port(
        step_source, Tuple{typeof(point)}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=RK.total_functional_lowering(
            LeapfrogLowering(endpoint)))
    stats_source = StatisticsAuthority()
    stats_port = RK.effect_lowering_port(
        stats_source, Tuple{RK.StatefulStateValue}, Nothing;
        written_arguments=(1,),
        initial_effect_state=(n_steps=0, acceptance_rate=0.0),
        functional_lowering=RK.total_functional_lowering(
            statistics_lowering))

    static_values = RK._sm_finite_static_values(structured)
    proposal_contract = RK._sm_finite_structural_contract(
        F.fillf(deepcopy, point, max_depth + 2); static_values)
    tree_contract = RK._sm_finite_structural_contract(
        F.fillf(F.tree, point, max_depth + 1))
    bindings = RK.stateful_compiler_bindings(
        init=structured, fwd=structured, bwd=structured,
        proposals=proposal_contract, trees=tree_contract,
        step_f=step_port, stats_f=stats_port)
    kernel = measure("source_preparation") do
        RK.compile_stateful(
        F.nuts_state, bindings, point;
        step_f=step_source, max_depth,
        min_dham=-1000.0, stats_f=stats_source)
    end
    state = kernel(point; step_f=step_source, max_depth,
                   min_dham=-1000.0, stats_f=stats_source)
    snapshot = RK.stateful_snapshot(state)

    if isnothing(replay)
        tokens = Tuple(isodd(index) ? :uniform : :exponential
                       for index in 1:64)
        replay = RK.OrderedRNGReplay(
            reshape([0.0], 1, 1), fill(false, 32), fill(1.0, 32),
            tokens)
    end
    bounds = measure("control_bounds") do
        RK.stateful_control_bounds(
        kernel, Val(:step!), state;
        argument_types=Tuple{typeof(replay)})
    end
    transition = measure("functional_lowering") do
        RK.functionalize_stateful(kernel, Val(:step!), bounds)
    end
    (; endpoint, point, structured, kernel, snapshot, replay, bounds,
       transition, max_depth, stepsize)
end

mutable struct SourceReplayRNG <: AbstractRNG
    uniforms::Vector{Bool}
    exponentials::Vector{Float64}
    events::Vector{Symbol}
    normal_index::Int
    uniform_index::Int
    exponential_index::Int
    event_index::Int
end

function _source_event!(rng::SourceReplayRNG, expected::Symbol)
    rng.event_index <= length(rng.events) || error(
        "source RNG exhausted its event tape")
    observed = rng.events[rng.event_index]
    observed === expected || error(
        "source RNG expected $expected, observed $observed")
    rng.event_index += 1
end

function Random.rand(rng::SourceReplayRNG, ::Type{Bool})
    _source_event!(rng, :uniform)
    rng.uniform_index <= length(rng.uniforms) || error(
        "source RNG exhausted its Bool tape")
    value = rng.uniforms[rng.uniform_index]
    rng.uniform_index += 1
    value
end

function Random.randexp(rng::SourceReplayRNG)
    _source_event!(rng, :exponential)
    rng.exponential_index <= length(rng.exponentials) || error(
        "source RNG exhausted its exponential tape")
    value = rng.exponentials[rng.exponential_index]
    rng.exponential_index += 1
    value
end

function _source_values(pf)
    values = Dict{Int,Any}()
    metric = Diagonal([1.0])
    for slot in RK.kernel_plan_slots(RK.kernel_prepared_plan(pf))
        name = String(slot.path[1])
        values[slot.canon] = name == "pot_f" ? potential :
            name == "grad_f" ? gradient :
            name == "metric" ? metric :
            name == "chol_metric" ? cholesky(metric) :
            startswith(name, "##node") ? 0.0 :
            name == "pos" ? [1.0] :
            name == "mom" ? [0.25] :
            name in ("dpot_dpos", "dham_dpos", "dkin_dmom", "dham_dmom") ?
                [0.0] : 0.0
    end
    values
end

function _source_setup(max_depth; stepsize=0.1)
    pf = RK._prepare_factory(
        F.euclidean_phasepoint, RK.kernel_registration(F.leapfrog!))
    frame = NEX._construct_nuts_frame(
        pf, _source_values(pf), max_depth;
        step_f=RK.partial(F.leapfrog!; stepsize),
        stats_f=F.nuts_stats!, min_dham=-1000.0)
    RK.compile_prepared_initialization(
        pf, typeof(frame.init), typeof(frame.shared))(
            frame.init, frame.shared, RK.kernel_prepared_handles(pf))
    NEX._seed_nuts_children!(frame)
    compiled = NEX.compile_nuts(
        pf, F.nuts_state, F.refresh_momentum!!, F.nuts!!, frame)
    (; pf, frame, compiled)
end

function _source_program(max_depth, replay; stepsize=0.1)
    source = _source_setup(max_depth; stepsize)
    event_names = Dict(RK._sm_ordered_rng_event_code(name) => name
                       for name in (:normal, :uniform, :exponential))
    rng = SourceReplayRNG(
        copy(replay.uniforms), copy(replay.exponentials),
        [event_names[token] for token in replay.event_tokens],
        1, 1, 1, 1)
    source.compiled.step!(source.frame, source.compiled.scratch, rng)
    (; source..., rng)
end

# Record conditional source RNG events instead of guessing an alternating
# tape. At larger depths the recursive proposal decisions consume additional
# exponentials, and short-circuit acceptance can omit them altogether.
mutable struct SourceRecordingRNG <: AbstractRNG
    uniforms::Vector{Bool}
    exponentials::Vector{Float64}
    events::Vector{Symbol}
end
function Random.rand(rng::SourceRecordingRNG, ::Type{Bool})
    push!(rng.events, :uniform)
    push!(rng.uniforms, false)
    false
end
function Random.randexp(rng::SourceRecordingRNG)
    push!(rng.events, :exponential)
    push!(rng.exponentials, 1.0)
    1.0
end

function recorded_source(max_depth; stepsize=0.1, transitions=2)
    transitions >= 1 || throw(ArgumentError("transitions must be positive"))
    source = _source_setup(max_depth; stepsize)
    rng = SourceRecordingRNG(Bool[], Float64[], Symbol[])
    receipts = Any[]
    for _ in 1:transitions
        source.compiled.step!(source.frame, source.compiled.scratch, rng)
        push!(receipts, (
            dynamic=_source_logical_dynamic(source),
            cursors=(1, length(rng.uniforms) + 1,
                     length(rng.exponentials) + 1, length(rng.events) + 1)))
    end
    replay = RK.OrderedRNGReplay(
        zeros(1, 1), rng.uniforms, rng.exponentials, rng.events)
    (; source, replay, receipts, max_depth, stepsize)
end

function _endpoint_matrix(values, field)
    reduce(hcat, (copy(getproperty(value, field)) for value in values))
end

function _generic_dynamic(state)
    points = (state.init, state.fwd, state.bwd, state.proposals...)
    trees = state.trees
    (D=length(state.init.pos), NPP=length(points), NT=length(trees),
     pp_pos=_endpoint_matrix(points, :pos),
     pp_mom=_endpoint_matrix(points, :mom),
     pp_dpot=_endpoint_matrix(points, :dpot_dpos),
     pp_dkin=_endpoint_matrix(points, :dkin_dmom),
     pp_pot=[point.pot for point in points],
     pp_kin=[point.kin for point in points],
     pp_ham=[point.ham for point in points],
     lw=_endpoint_matrix(trees, :log_weight),
     tb_mom=_endpoint_matrix((tree.bwd for tree in trees), :mom),
     tb_dh=_endpoint_matrix((tree.bwd for tree in trees), :dham_dmom),
     tbf_mom=_endpoint_matrix((tree.bwd_fwd for tree in trees), :mom),
     tbf_dh=_endpoint_matrix((tree.bwd_fwd for tree in trees), :dham_dmom),
     sm_b=_endpoint_matrix((tree.summed_mom for tree in trees), :bwd),
     sm_f=_endpoint_matrix((tree.summed_mom for tree in trees), :fwd),
     gofwd=state.gofwd, may_sample=state.may_sample,
     may_continue=state.may_continue, diverged=state.diverged,
     n_steps=state.n_steps, reached_depth=state.reached_depth,
     acceptance_rate=state.acceptance_rate, dham=state.dham)
end

function _source_currentness(source)
    plan = RK.kernel_prepared_plan(source.pf)
    bwd = source.frame.bwd
    (; mom=NEX._canon_current(
           bwd, NEX.kernel_plan_named_slot_val(plan, Val(:mom))),
       dkin_dmom=NEX._canon_current(
           bwd, NEX.kernel_plan_named_slot_val(plan, Val(:dkin_dmom))),
       dham_dmom=NEX._canon_current(
           bwd, NEX.kernel_plan_named_slot_val(plan, Val(:dham_dmom))))
end

function _source_logical_dynamic(source)
    # `_nuts_frame_to_tensors` is a private physical-cache census.  A source
    # read of a dirty derived endpoint field first executes its prepared ensure.
    # Materialize those reads on a topology-preserving copy so the semantic
    # receipt compares the generic compiler's public normalized state with the
    # source program's logical values without mutating the continuing oracle.
    frame = source.frame
    # Copy only the owned endpoint carriers.  The frame also holds shared
    # callable authorities (including Modules), which are intentionally not
    # deepcopy-able and need not be copied for a read-only logical projection.
    points = map(deepcopy,
        (frame.init, frame.fwd, frame.bwd, frame.proposals...))
    handles = RK.kernel_prepared_handles(source.pf)
    fields = sort!(collect(NEX.derived_fields(source.pf)))
    ensures = Tuple(NEX.compile_prepared_ensure(
        source.pf, typeof(frame.init), typeof(frame.shared), field)
        for field in fields)
    for point in points, ensure in ensures
        ensure(point, frame.shared, handles)
    end
    raw = NEX._nuts_frame_to_tensors(frame)
    source_matrix(slot) = reduce(hcat,
        (copy(NEX._nuts_ppfield(point, slot)) for point in points))
    merge(raw, (;
        pp_pos=source_matrix(4), pp_mom=source_matrix(5),
        pp_dpot=source_matrix(8), pp_dkin=source_matrix(10),
        pp_pot=[NEX._nuts_ppfield(point, 7) for point in points],
        pp_kin=[NEX._nuts_ppfield(point, 11) for point in points],
        pp_ham=[NEX._nuts_ppfield(point, 12) for point in points]))
end

_rng_receipt(rng) = (;
    normal_index=rng.normal_index,
    uniform_index=rng.uniform_index,
    exponential_index=rng.exponential_index,
    event_index=rng.event_index,
    events=copy(rng.events))

function build_case(; measure=_unmeasured, max_depth=2,
                    stepsize=0.1, replay=nothing)
    prepared = _generic_program(max_depth; measure, stepsize, replay)
    result = measure("native_first") do
        prepared.transition(prepared.snapshot, prepared.replay)
    end
    generic = (; prepared..., result)
    source = measure("source_oracle") do
        _source_program(generic.max_depth, generic.replay; stepsize)
    end
    source_raw_dynamic = NEX._nuts_frame_to_tensors(source.frame)
    source_currentness = _source_currentness(source)
    source_dynamic = _source_logical_dynamic(source)
    source_first_rng = _rng_receipt(source.rng)
    generic_dynamic = _generic_dynamic(generic.result.state)

    second_result = generic.transition(
        generic.result.state, generic.result.arguments[1])
    source.compiled.step!(source.frame, source.compiled.scratch, source.rng)
    second = (;
        result=second_result,
        generic_dynamic=_generic_dynamic(second_result.state),
        source_raw_dynamic=NEX._nuts_frame_to_tensors(source.frame),
        source_currentness=_source_currentness(source),
        source_dynamic=_source_logical_dynamic(source),
        source_rng=_rng_receipt(source.rng))

    (; generic..., source, source_first_rng, source_raw_dynamic,
       source_currentness, source_dynamic, generic_dynamic, second)
end

end
