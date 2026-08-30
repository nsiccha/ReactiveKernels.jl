module ReactiveHMCHMCCompilerSupport

using ReactiveKernels
using ReactiveKernelsCompatibilityExamples: ReactiveHMCExamples
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "benchmark",
                 "reactivehmc_integrator_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "..", "benchmark",
                 "reactivehmc_hmc_kernel_fixture.jl"))

const RK = ReactiveKernels

potential(position) = sum(abs2, position) / 2
gradient(position) = (potential(position), position)

struct EndpointEffectAuthority <: Function end

(::EndpointEffectAuthority)(point) = throw(ArgumentError(
    "endpoint effect authority is functional-only"))

mutable struct StatisticsRecorder{T} <: Function
    errors::Vector{T}
    count::Int
end

function (stats::StatisticsRecorder)(state)
    stats.count += 1
    stats.errors[stats.count] = state.dham
    nothing
end

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

function build_case(case; potential_f=potential, gradient_f=gradient,
                    numeric_type=Float64)
    T = numeric_type
    stepsize = T(case["stepsize"])
    n_steps = case["n_steps"]
    min_dham = T(case["min_dham"])
    normal = T.(case["normal_draw"])
    exponentials = Float64.(case["exponential_draws"])

    bundle = ReactiveHMCExamples.euclidean_phasepoint_kernels(
        Val(:gaussian), potential_f, gradient_f, Diagonal(ones(T, 2)),
        T.(case["initial_position"]), zeros(T, 2))
    endpoint = compile_state_transition(
        bundle.spec,
        partial(ReactiveHMCIntegratorFixture.generalized_leapfrog!;
                stepsize, n_fi_steps=1),
        values(bundle.sources),
    )
    point = initial_transition_state(endpoint)
    structured = RK.structured_state_port(endpoint)
    step_source = EndpointEffectAuthority()
    step_port = RK.effect_lowering_port(
        step_source, Tuple{typeof(point)}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=RK.total_functional_lowering(
            step_lowering(endpoint)),
    )

    diagnostic_type = typeof(point.ham)
    stats_source = StatisticsRecorder(zeros(diagnostic_type, n_steps), 0)
    stats_port = RK.effect_callable_port(
        stats_source, Tuple{RK.StatefulStateValue}, Nothing;
        written_arguments=(),
        initial_effect_state=(
            errors=ntuple(_ -> zero(diagnostic_type), n_steps), count=0),
        functional_lowering=RK.total_functional_lowering(stats_lowering),
    )
    bindings = RK.stateful_compiler_bindings(
        init=structured,
        fwd=structured,
        step_f=step_port,
        stats_f=stats_port,
    )
    kernel = RK.compile_stateful(
        ReactiveHMCHMCFixture.hmc_state, bindings, point;
        n_steps, min_dham, step_f=step_source, stats_f=stats_source)
    state = kernel(point; n_steps, min_dham,
                   step_f=step_source, stats_f=stats_source)
    snapshot = RK.stateful_snapshot(state)
    replay = RK.OrderedRNGReplay(
        reshape(normal, :, 1), Bool[false],
        isempty(exponentials) ? [0.0] : exponentials,
        Tuple(Symbol(event) for event in case["rng_events"]))
    transition = RK.functionalize_stateful(
        kernel, Val(:step!);
        max_iterations=max(n_steps, 1),
        argument_types=Tuple{typeof(replay)})
    (; bundle, endpoint, structured, kernel, state, transition, snapshot, replay,
       step_source, stats_source, step_port, stats_port)
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
        rng_event_calls=result.arguments[1].event_index - 1,
        rng_overflow=result.arguments[1].overflow,
        control_overflow=result.control_overflow,
    )
end

end # module ReactiveHMCHMCCompilerSupport
