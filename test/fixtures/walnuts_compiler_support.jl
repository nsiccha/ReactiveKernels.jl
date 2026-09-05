module WalnutsCompilerSupport

using LinearAlgebra
using ReactiveKernels

include(joinpath(@__DIR__, "..", "..", "benchmark",
                 "walnuts_kernel_authoring_fixture_b.jl"))

const RK = ReactiveKernels
const WFX = WALNUTSBMutationAuthoringFixture
const NFX = WFX.NUTSBMutationAuthoringFixture

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

struct GaussianGradient{T} <: Function
    stiffness::T
end
gradient(stiffness) = GaussianGradient(stiffness)
function (gradient::GaussianGradient)(position)
    value = oftype(first(position), 0.5) * gradient.stiffness * sum(abs2, position)
    (value, gradient.stiffness .* position)
end
function (gradient::GaussianGradient)(destination, position)
    value = oftype(first(position), 0.5) * gradient.stiffness * sum(abs2, position)
    destination .= gradient.stiffness .* position
    value
end

struct EndpointEffectAuthority <: Function end
struct StatisticsEffectAuthority <: Function end

(::EndpointEffectAuthority)(args...; kwargs...) = throw(ArgumentError(
    "endpoint effect authority is functional-only"))
(::StatisticsEffectAuthority)(args...; kwargs...) = throw(ArgumentError(
    "statistics effect authority is functional-only"))

struct EndpointLowering{T}
    transition::T
end

function (lowering::EndpointLowering)(effect, point; stepsize)
    (arguments=(lowering.transition(point, (; stepsize)),),
     result=nothing, effect_state=effect)
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

function endpoint(stiffness, theta, rho)
    pot_f = potential(stiffness)
    grad_f = gradient(stiffness)
    spec = NFX.euclidean_phasepoint
    transition = RK.compile_state_transition(
        spec, WFX.leapfrog!,
        (pot_f, grad_f, Diagonal([1.0]), [theta], [rho]);
        runtime_controls=(stepsize=0.1,))
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
            EndpointLowering(endpoint_transition)))
    stats_source = StatisticsEffectAuthority()
    stats_port = RK.effect_lowering_port(
        stats_source, Tuple{RK.StatefulStateValue}, Nothing;
        written_arguments=(1,),
        initial_effect_state=(n_steps=0, acceptance_rate=zero(case.theta)),
        functional_lowering=RK.total_functional_lowering(
            statistics_lowering))
    static_values = RK._sm_finite_static_values(structured)
    proposal_contract = RK._sm_finite_structural_contract(
        WFX.fillf(deepcopy, point, max_depth + 2); static_values)
    tree_contract = RK._sm_finite_structural_contract(
        WFX.fillf(WFX.tree, point, max_depth + 1))
    bindings = RK.stateful_compiler_bindings(
        init=structured,
        fwd=structured,
        bwd=structured,
        candidate=structured,
        reverse_candidate=structured,
        proposals=proposal_contract,
        trees=tree_contract,
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
    proposal_raw = RK._sm_finite_structural_pack(
        proposal_contract, snapshot.proposals)
    tree_raw = RK._sm_finite_structural_pack(
        tree_contract, snapshot.trees)
    (; endpoint_transition, structured, kernel, snapshot,
       step_source, step_port, stats_source, stats_port,
       directions, exponentials, proposal_contract, proposal_raw,
       tree_contract, tree_raw)
end

function _build_case_transition(setup; max_iterations=nothing)
    state = setup.snapshot
    # The finest dyadic grid bounds integrate!'s runtime trip count. The
    # same limit covers the coarser reverse grids and the retry loop.
    iterations = isnothing(max_iterations) ? max(
        state.max_depth, state.max_step_halvings,
        foldl((steps, _) -> Base.Checked.checked_mul(steps, 2),
            1:(state.max_step_halvings - 1); init=state.min_micro_steps)) :
        max_iterations
    bounds = RK.stateful_control_bounds(
        setup.kernel, Val(:step!), state;
        recursion_bound=:max_depth, max_iterations=iterations,
        argument_types=Tuple{
            typeof(setup.directions),typeof(setup.exponentials)})
    RK.functionalize_stateful(
        setup.kernel, Val(:step!), bounds)
end

function build_case(case; max_depth=1, min_dham=-1000.0,
                    directions=fill(false, max_depth),
                    exponentials=fill(1.0, max(2^max_depth, 1)),
                    max_iterations=nothing)
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
