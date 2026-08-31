module WalnutsCompilerSupport

using LinearAlgebra
using ReactiveKernels

include(joinpath(@__DIR__, "..", "..", "benchmark",
                 "walnuts_kernel_authoring_fixture.jl"))

const RK = ReactiveKernels
const WFX = WalnutsKernelAuthoringFixture

const ORACLE_CASES = (
    (name="base_grid_accept", stiffness=1.0, theta=1.0, rho=0.3,
     macro_time=0.1, max_step_halvings=4, min_micro_steps=1,
     max_error=1.0, accepted=true, micro_steps=1,
     candidate_theta=1.0249999999999999,
     candidate_rho=0.19875000000000001,
     candidate_joint=-0.54506328124999992,
     base_accept=0.99993672075221618),
    (name="dyadic_reverse_accept", stiffness=10.0, theta=1.0, rho=0.3,
     macro_time=0.5, max_step_halvings=4, min_micro_steps=1,
     max_error=0.5, accepted=true, micro_steps=10,
     candidate_theta=0.075936222076415927,
     candidate_rho=-3.1054732918739321,
     candidate_joint=-4.8508137323873521,
     base_accept=0.045331641611676264),
    (name="reverse_grid_reject", stiffness=0.1, theta=-2.0, rho=-3.0,
     macro_time=3.0, max_step_halvings=4, min_micro_steps=1,
     max_error=1.0, accepted=false, micro_steps=4,
     candidate_theta=-9.1381250000000005,
     candidate_rho=-1.223390625,
     candidate_joint=-4.9236087364501966,
     base_accept=0.33200259295192019),
    (name="all_grids_reject", stiffness=10.0, theta=1.0, rho=0.3,
     macro_time=0.5, max_step_halvings=4, min_micro_steps=1,
     max_error=1.0e-6, accepted=false, micro_steps=15,
     candidate_theta=0.08240002202929364,
     candidate_rho=-3.1504141277854654,
     candidate_joint=-4.9965034064272675,
     base_accept=0.045331641611676264),
)

potential(stiffness) = position ->
    oftype(first(position), 0.5) * stiffness * sum(abs2, position)

gradient(stiffness) = (destination, position) -> begin
    value = oftype(first(position), 0.5) * stiffness * sum(abs2, position)
    destination .= stiffness .* position
    value
end

struct EndpointEffectAuthority <: Function end
struct StatisticsEffectAuthority <: Function end

(::EndpointEffectAuthority)(args...; kwargs...) = throw(ArgumentError(
    "endpoint effect authority is functional-only"))
(::StatisticsEffectAuthority)(args...; kwargs...) = throw(ArgumentError(
    "statistics effect authority is functional-only"))

struct GaussianLeapfrogLowering{T}
    stiffness::T
end

function (lowering::GaussianLeapfrogLowering)(effect, point; stepsize)
    half = oftype(stepsize, 0.5)
    first_mom = point.mom .- half .* stepsize .* point.dham_dpos
    pos = point.pos .+ stepsize .* first_mom
    dpot_dpos = lowering.stiffness .* pos
    mom = first_mom .- half .* stepsize .* dpot_dpos
    dkin_dmom = copy(mom)
    pot = oftype(first(pos), 0.5) * lowering.stiffness * sum(abs2, pos)
    kin = oftype(first(mom), 0.5) * sum(abs2, mom)
    updated = merge(point, (;
        pos, mom, pot, dpot_dpos, dkin_dmom, kin, ham=pot + kin,
        dham_dpos=dpot_dpos, dham_dmom=dkin_dmom))
    (arguments=(updated,), result=nothing, effect_state=effect)
end

function statistics_lowering(effect, state)
    n_steps = effect.n_steps + one(effect.n_steps)
    unit = one(state.dham)
    rate = (unit - unit / n_steps) * effect.acceptance_rate +
        (unit / n_steps) *
        ifelse(state.dham >= zero(state.dham), unit, exp(state.dham))
    (arguments=(state,), result=nothing,
     effect_state=(; n_steps, acceptance_rate=rate))
end

function endpoint(stiffness, theta, rho)
    pot_f = potential(stiffness)
    grad_f = gradient(stiffness)
    spec = WFX.euclidean_phasepoint
    transition = RK.compile_state_transition(
        spec, RK.partial(WFX.leapfrog!; stepsize=0.1),
        (pot_f, grad_f, Diagonal([1.0]), [theta], [rho]))
    transition, RK.initial_transition_state(transition)
end

function _build_case_setup(case; max_depth=1, min_dham=-1000.0,
                           directions=fill(false, max_depth),
                           exponentials=fill(1.0, max(2^max_depth, 1)))
    endpoint_transition, point = endpoint(
        case.stiffness, case.theta, case.rho)
    structured = RK.structured_state_port(endpoint_transition)
    step_source = EndpointEffectAuthority()
    step_port = RK.effect_lowering_port(
        step_source, Tuple{typeof(point)}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=RK.total_functional_lowering(
            GaussianLeapfrogLowering(case.stiffness)))
    stats_source = StatisticsEffectAuthority()
    stats_port = RK.effect_lowering_port(
        stats_source, Tuple{RK.StatefulStateValue}, Nothing;
        written_arguments=(),
        initial_effect_state=(n_steps=0, acceptance_rate=zero(case.theta)),
        functional_lowering=RK.total_functional_lowering(
            statistics_lowering))
    bindings = RK.stateful_compiler_bindings(
        init=structured,
        fwd=structured,
        bwd=structured,
        candidate=structured,
        reverse_candidate=structured,
        step_f=step_port,
        stats_f=stats_port,
    )
    kernel = RK.compile_stateful(
        WFX.walnuts_state, bindings, point;
        step_f=step_source, macro_time=case.macro_time, max_depth,
        max_step_halvings=case.max_step_halvings,
        min_micro_steps=case.min_micro_steps, max_error=case.max_error,
        min_dham, stats_f=stats_source)
    state = kernel(
        point; step_f=step_source, macro_time=case.macro_time, max_depth,
        max_step_halvings=case.max_step_halvings,
        min_micro_steps=case.min_micro_steps, max_error=case.max_error,
        min_dham, stats_f=stats_source)
    snapshot = RK.stateful_snapshot(state)
    static_values = RK._sm_finite_static_values(structured)
    proposal_contract = RK._sm_finite_structural_contract(
        snapshot.proposals; static_values)
    proposal_raw = RK._sm_finite_structural_pack(
        proposal_contract, snapshot.proposals)
    tree_contract = RK._sm_finite_structural_contract(snapshot.trees)
    tree_raw = RK._sm_finite_structural_pack(
        tree_contract, snapshot.trees)
    (; endpoint_transition, structured, kernel, snapshot,
       step_source, step_port, stats_source, stats_port,
       directions, exponentials, proposal_contract, proposal_raw,
       tree_contract, tree_raw)
end

function _build_case_transition(setup; max_iterations=1_000_000)
    RK.functionalize_stateful(
        setup.kernel, Val(:step!); max_iterations,
        argument_types=Tuple{
            typeof(setup.directions),typeof(setup.exponentials)})
end

function build_case(case; max_depth=1, min_dham=-1000.0,
                    directions=fill(false, max_depth),
                    exponentials=fill(1.0, max(2^max_depth, 1)),
                    max_iterations=1_000_000)
    setup = _build_case_setup(
        case; max_depth, min_dham, directions, exponentials)
    transition = _build_case_transition(setup; max_iterations)
    merge(setup, (; transition))
end

function result_values(result)
    state = result.state
    stats = result.effects.stats_f
    (
        candidate_theta=only(state.candidate.pos),
        candidate_rho=only(state.candidate.mom),
        candidate_joint=-state.candidate.ham,
        accepted=state.macro_accepted[1],
        macro_count=state.macro_count,
        total_micro_steps=state.total_micro_steps,
        attempted_micro_steps=state.attempted_micro_steps[1],
        forward_attempts=state.forward_attempts[1],
        reverse_checks=state.reverse_checks[1],
        acceptance_rate=stats.acceptance_rate,
        reached_depth=state.reached_depth,
        n_steps=stats.n_steps,
        diverged=state.diverged,
        replay_overflow=state.replay_overflow,
        control_overflow=result.control_overflow,
        direction_calls=state.direction_index,
        exponential_calls=state.exponential_index,
    )
end

end # module WalnutsCompilerSupport
