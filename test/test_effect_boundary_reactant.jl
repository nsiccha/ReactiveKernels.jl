using ReactiveKernels
using Reactant
using LinearAlgebra
using Test
import Reactant: @compile

include(joinpath(@__DIR__, "fixtures", "stateful_functional_contracts.jl"))
const _EFFECTS_REACTANT_SFC = StatefulFunctionalContractsFixture

mutable struct _ReactantObservationCollector
    counts::Vector{Int}
end
(collector::_ReactantObservationCollector)(state) =
    (push!(collector.counts, Int(state.count)); nothing)

struct _ReactantObservationAuthority end
_reactant_observation_lowering(effect, state) = (
    arguments=(state,), result=nothing, effect_state=effect .+ 1)

struct _ReactantTupleWrapperRestore{P}
    prototype::P
end
(restore::_ReactantTupleWrapperRestore)(value) =
    ReactiveKernels._sm_restore_source_logical_wrappers(
        restore.prototype, value)

_effects_trace(value::Number) =
    Reactant.to_rarray(value; track_numbers=true)
_effects_trace(value::AbstractArray) = Reactant.to_rarray(value)
_effects_trace(value::NamedTuple) = map(_effects_trace, value)
_effects_trace(value::Tuple) = map(_effects_trace, value)
_effects_trace(value) = value

@testset "host-drained observations survive Reactant tracing" begin
    @test Base.get_extension(
        ReactiveKernels, :ReactiveKernelsReactantExt) !== nothing

    tuple_prototype = (1.0, [2.0, 3.0])
    tuple_value = _effects_trace((4.0, [5.0, 6.0]))
    tuple_restore = _ReactantTupleWrapperRestore(tuple_prototype)
    compiled_tuple_restore = @compile tuple_restore(tuple_value)
    restored_tuple = compiled_tuple_restore(tuple_value)
    @test Float64(first(restored_tuple)) == 4.0
    @test Array(last(restored_tuple)) == [5.0, 6.0]

    host_cholesky = LinearAlgebra.Cholesky(
        Diagonal([2.0, 3.0]), 'U', 0)
    backend_cholesky = ReactiveKernels._sm_cholesky_reconstruct(
        Diagonal(Reactant.to_rarray([2.0, 3.0])), 'U', 0)
    materialized_cholesky = ReactiveKernels._sm_materialize_observation(
        backend_cholesky, typeof(host_cholesky))
    @test typeof(materialized_cholesky) === typeof(host_cholesky)
    @test materialized_cholesky.factors.diag == [2.0, 3.0]
    @test materialized_cholesky.uplo == 'U'
    @test materialized_cholesky.info == 0

    collector = _ReactantObservationCollector(Int[])
    port = effect_callable_port(
        collector, Tuple{StatefulStateValue}, Nothing)
    bindings = stateful_compiler_bindings(callback=port)
    kernel = compile_stateful(
        _EFFECTS_REACTANT_SFC.sequential_effect_contract,
        bindings, collector, 0)
    snapshot = stateful_snapshot(kernel(collector, 0))
    transition = functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})

    state = _effects_trace(snapshot)
    take = _effects_trace(true)
    compiled = @compile transition(state, take)
    guarded = validated_compiled_transition(compiled, transition)
    result = guarded(state, take)

    @test isempty(collector.counts)
    @test Int(result.outbox.callback.count) == 2
    @test map(Bool, result.outbox.callback.active) == (true, true)
    @test result.state.callback === collector

    receipt = fetch(@async drain_observations!(guarded, result))
    @test collector.counts == [0, 0]
    @test receipt.callback == (
        count=2, capacity=2, overflow=false, value=nothing)

    authority = _ReactantObservationAuthority()
    authority_port = effect_lowering_port(
        authority, Tuple{StatefulStateValue}, Nothing;
        initial_effect_state=[0],
        functional_lowering=total_functional_lowering(
            _reactant_observation_lowering))
    authority_bindings = stateful_compiler_bindings(
        callback=authority_port)
    authority_kernel = compile_stateful(
        _EFFECTS_REACTANT_SFC.effect_contract,
        authority_bindings, authority, 0)
    authority_snapshot = stateful_snapshot(authority_kernel(authority, 0))
    authority_transition = functionalize_stateful(
        authority_kernel, Val(:step!); argument_types=Tuple{Bool})
    authority_state = _effects_trace(authority_snapshot)
    authority_effects = _effects_trace(
        initial_transition_effects(authority_transition))
    authority_runner = transition_with_effects(authority_transition)
    compiled_authority = @compile authority_runner(
        authority_state, authority_effects, take)
    authority_result = compiled_authority(
        authority_state, authority_effects, take)
    @test Array(authority_result.effects.callback) == [1]
    authority_receipt = drain_observations!(
        authority_transition, authority_result)
    @test authority_receipt.callback.value == [1]
end
