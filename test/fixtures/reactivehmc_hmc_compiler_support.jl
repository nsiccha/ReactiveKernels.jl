module ReactiveHMCHMCCompilerSupport

using ReactiveKernels
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "examples",
                 "preexisting_reactivehmc.jl"))
include(joinpath(@__DIR__, "..", "..", "benchmark",
                 "reactivehmc_integrator_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "..", "benchmark",
                 "reactivehmc_hmc_kernel_fixture.jl"))

const RK = ReactiveKernels

potential(position) = sum(abs2, position) / 2
gradient(position) = (potential(position), position)
step_marker(point) = nothing
stats_marker(state) = nothing

step_lowering(transition) = (effect, point) -> (
    arguments=(transition(point),),
    result=nothing,
    effect_state=effect,
)

function stats_lowering(effect, state)
    next = effect.count + one(effect.count)
    errors = ntuple(length(effect.errors)) do index
        ifelse(next == index, state.dham, effect.errors[index])
    end
    (
        arguments=(state,),
        result=nothing,
        effect_state=(errors=errors, count=next),
    )
end

function build_case(case)
    stepsize = case["stepsize"]
    n_steps = case["n_steps"]
    min_dham = case["min_dham"]
    normal = case["normal_draw"]
    exponentials = case["exponential_draws"]

    bundle = ReactiveHMCExamples.euclidean_phasepoint_kernels(
        Val(:gaussian), potential, gradient, Diagonal(ones(2)),
        case["initial_position"], zeros(2))
    endpoint = compile_state_transition(
        bundle.spec,
        partial(ReactiveHMCIntegratorFixture.generalized_leapfrog!;
                stepsize, n_fi_steps=1),
        values(bundle.sources),
    )
    point = initial_transition_state(endpoint)
    structured = RK.structured_state_port(endpoint)
    step_port = RK.effect_callable_port(
        step_marker, Tuple{typeof(point)}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=step_lowering(endpoint),
    )

    stats_port = RK.effect_callable_port(
        stats_marker, Tuple{RK.StatefulStateValue}, Nothing;
        written_arguments=(),
        initial_effect_state=(
            errors=ntuple(_ -> 0.0, n_steps), count=0),
        functional_lowering=stats_lowering,
    )
    bindings = RK.stateful_compiler_bindings(
        init=structured,
        fwd=structured,
        step_f=step_port,
        stats_f=stats_port,
    )
    kernel = RK.compile_stateful(
        ReactiveHMCHMCFixture.hmc_state, bindings, point;
        n_steps, min_dham, step_f=step_marker, stats_f=stats_marker)
    state = kernel(point; n_steps, min_dham,
                   step_f=step_marker, stats_f=stats_marker)
    snapshot = RK.stateful_snapshot(state)
    replay = RK.OrderedRNGReplay(
        reshape(normal, :, 1), Bool[false],
        isempty(exponentials) ? [0.0] : exponentials)
    transition = RK.functionalize_stateful(
        kernel, Val(:step!);
        max_iterations=max(n_steps, 1),
        argument_types=Tuple{typeof(replay)})
    (; bundle, endpoint, structured, kernel, transition, snapshot, replay)
end

function result_values(result)
    count = Int(result.effects.stats_f.count)
    (
        init_pos=result.state.init.pos,
        init_mom=result.state.init.mom,
        init_ham=result.state.init.ham,
        fwd_pos=result.state.fwd.pos,
        fwd_mom=result.state.fwd.mom,
        fwd_ham=result.state.fwd.ham,
        dham=result.state.dham,
        diverged=result.state.diverged,
        energy_errors=result.effects.stats_f.errors[1:count],
        normal_calls=result.arguments[1].normal_index - 1,
        exponential_calls=result.arguments[1].exponential_index - 1,
        rng_overflow=result.arguments[1].overflow,
        control_overflow=result.control_overflow,
    )
end

end # module ReactiveHMCHMCCompilerSupport
