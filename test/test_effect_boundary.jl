using ReactiveKernels
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "fixtures", "stateful_functional_contracts.jl"))
const _EFFECTS_SFC = StatefulFunctionalContractsFixture

mutable struct _ObservationCollector
    counts::Vector{Int}
end
(collector::_ObservationCollector)(state) =
    (push!(collector.counts, state.count); nothing)

struct _ObservationAuthority end
_observation_lowering(effect, state) = (
    arguments=(state,), result=nothing, effect_state=effect + 1)

struct _CausalAuthority end
_causal_lowering(effect, values) = (
    arguments=(values .+ 1,), result=nothing, effect_state=effect)

function _effect_boundary_program(spec, source, port, initial...)
    bindings = stateful_compiler_bindings(callback=port)
    kernel = compile_stateful(spec, bindings, initial..., source, 0)
    state = stateful_snapshot(kernel(initial..., source, 0))
    transition = functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; kernel, state, transition)
end

@testset "compiled/host effects boundary" begin
    collector = _ObservationCollector(Int[])
    observation = effect_callable_port(
        collector, Tuple{StatefulStateValue}, Nothing)
    program = _effect_boundary_program(
        _EFFECTS_SFC.sequential_effect_contract,
        collector, observation)

    result = program.transition(program.state, true)
    @test propertynames(result) == (
        :state, :arguments, :result, :returned,
        :control_overflow, :effects, :outbox)
    @test isempty(propertynames(result.effects))
    @test result.outbox.callback.count == 2
    # Structure-of-arrays storage: one column per numeric state leaf with a
    # trailing slot axis of the port's capacity (two call sites), the static
    # callable field carried by reference.
    storage = result.outbox.callback.storage
    @test propertynames(storage) == (:arguments,)
    @test storage.arguments[1].count == [0, 0]
    @test storage.arguments[1].callback === collector
    @test result.outbox.callback.overflow === false
    @test isempty(collector.counts)

    task = @async drain_observations!(program.transition, result)
    receipt = fetch(task)
    @test collector.counts == [0, 0]
    @test receipt.callback == (
        count=2, capacity=2, overflow=false, value=nothing)

    skipped = program.transition(program.state, false)
    @test skipped.outbox.callback.count == 0
    @test drain_observations!(program.transition, skipped).callback.count == 0

    overflow_item = merge(
        result.outbox.callback, (overflow=true,))
    overflow_result = merge(
        result, (outbox=(callback=overflow_item,),))
    overflow_error = try
        drain_observations!(program.transition, overflow_result)
        nothing
    catch error
        error
    end
    @test overflow_error isa ArgumentError
    @test occursin("overflowed its fixed capacity",
                   sprint(showerror, overflow_error))
    @test collector.counts == [0, 0]

    malformed_item = merge(
        result.outbox.callback,
        (storage=(arguments=((callback=collector, count=[0]),),),))
    malformed_result = merge(
        result, (outbox=(callback=malformed_item,),))
    malformed_error = try
        drain_observations!(program.transition, malformed_result)
        nothing
    catch error
        error
    end
    @test malformed_error isa ArgumentError
    @test occursin("inconsistent fixed-capacity storage",
                   sprint(showerror, malformed_error))
    @test collector.counts == [0, 0]

    record_type = NamedTuple{(:arguments,),Tuple{Tuple{Int,Vector{Float64}}}}
    initial_box = ReactiveKernels._sm_observation_outbox_init(
        (arguments=(1, [0.0, 0.0]),), record_type, Val(2), 0, false)
    @test initial_box.storage.arguments[1] == [0, 0]
    @test initial_box.storage.arguments[2] == zeros(2, 2)
    @test initial_box.count == 0
    skipped_box = ReactiveKernels._sm_observation_outbox_write(
        initial_box, (arguments=(2, [2.0, 3.0]),), record_type, Val(2), false)
    @test skipped_box.count == 0
    @test skipped_box.storage.arguments[2] == zeros(2, 2)
    one_box = ReactiveKernels._sm_observation_outbox_write(
        initial_box, (arguments=(2, [2.0, 3.0]),), record_type, Val(2), true)
    full_box = ReactiveKernels._sm_observation_outbox_write(
        one_box, (arguments=(3, [4.0, 5.0]),), record_type, Val(2), true)
    @test full_box.count == 2
    @test full_box.overflow === false
    @test full_box.storage.arguments[1] == [2, 3]
    @test full_box.storage.arguments[2] == [2.0 4.0; 3.0 5.0]
    overflow_box = ReactiveKernels._sm_observation_outbox_write(
        full_box, (arguments=(4, [6.0, 7.0]),), record_type, Val(2), true)
    @test overflow_box.count == 2
    @test overflow_box.overflow === true
    @test overflow_box.storage == full_box.storage
    @test ReactiveKernels._sm_observation_slot_value(
        full_box.storage, record_type, 2) == (arguments=(3, [4.0, 5.0]),)
    growth_error = try
        ReactiveKernels._sm_observation_outbox_write(
            one_box, (arguments=(3, [4.0, 5.0, 6.0]),), record_type, Val(2),
            true)
        nothing
    catch error
        error
    end
    @test growth_error isa ArgumentError
    @test occursin("forbidden observational outbox growth",
                   sprint(showerror, growth_error))
    type_error = try
        ReactiveKernels._sm_observation_outbox_write(
            one_box, (arguments=(3.0, [4.0, 5.0]),), record_type, Val(2), true)
        nothing
    catch error
        error
    end
    @test type_error isa ArgumentError
    @test occursin("forbidden observational outbox growth",
                   sprint(showerror, type_error))
    # A Cholesky leaf travels as its parts: factors recurse, `uplo` stays a
    # static identity, `info` is one numeric column.
    chol = LinearAlgebra.cholesky([4.0 0.0; 0.0 9.0])
    chol_type = NamedTuple{(:arguments,),Tuple{Tuple{typeof(chol)}}}
    chol_box = ReactiveKernels._sm_observation_outbox_init(
        (arguments=(chol,),), chol_type, Val(2), 0, false)
    @test propertynames(chol_box.storage.arguments[1]) == (:factors, :uplo, :info)
    @test chol_box.storage.arguments[1].uplo == 'U'
    @test chol_box.storage.arguments[1].info == [0, 0]
    other = LinearAlgebra.cholesky([16.0 0.0; 0.0 25.0])
    chol_box = ReactiveKernels._sm_observation_outbox_write(
        chol_box, (arguments=(other,),), chol_type, Val(2), true)
    stored = ReactiveKernels._sm_observation_slot_value(
        chol_box.storage, chol_type, 1).arguments[1]
    @test stored isa LinearAlgebra.Cholesky
    @test stored.factors == other.factors && stored.uplo == 'U' && stored.info == 0
    wrapper_error = try
        ReactiveKernels._sm_observation_outbox_write(
            chol_box, (arguments=(other.factors,),), chol_type, Val(2), true)
        nothing
    catch error
        error
    end
    @test wrapper_error isa ArgumentError
    @test occursin("lost its Cholesky wrapper", sprint(showerror, wrapper_error))

    authority = _ObservationAuthority()
    authority_port = effect_lowering_port(
        authority, Tuple{StatefulStateValue}, Nothing;
        initial_effect_state=0,
        functional_lowering=total_functional_lowering(
            _observation_lowering))
    authority_program = _effect_boundary_program(
        _EFFECTS_SFC.effect_contract,
        authority, authority_port)
    authority_result = authority_program.transition(
        authority_program.state, true)
    # Compiler-only authorities keep a compatibility summary in `effects`
    # for existing fixed-shape kernels, but the value is freshly derived for
    # this invocation and is also exposed through the host-drained outbox.
    @test authority_result.effects.callback == 1
    authority_receipt = drain_observations!(
        authority_program.transition, authority_result)
    @test authority_receipt.callback == (
        count=1, capacity=1, overflow=false, value=1)

    causal = _CausalAuthority()
    causal_port = effect_lowering_port(
        causal, Tuple{Vector{Float64}}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=total_functional_lowering(_causal_lowering))
    causal_program = _effect_boundary_program(
        _EFFECTS_SFC.array_effect,
        causal, causal_port, [1.0])
    causal_result = causal_program.transition(causal_program.state, true)
    @test propertynames(causal_result) == (
        :state, :arguments, :result, :returned,
        :control_overflow, :effects)
    @test causal_result.state.values == [2.0]
    causal_error = try
        drain_observations!(causal_program.transition, causal_result)
        nothing
    catch error
        error
    end
    @test causal_error isa ArgumentError
    @test occursin("causal effects remain in the compiled `effects` carrier",
                   sprint(showerror, causal_error))

    backend_hits = Ref(0)
    compiled = (state, arguments...) -> begin
        backend_hits[] += 1
        causal_program.transition(state, arguments...)
    end
    guarded = validated_compiled_transition(
        compiled, causal_program.transition)
    wrong_shape = merge(causal_program.state, (values=[1.0, 2.0],))
    abi_error = try
        guarded(wrong_shape, true)
        nothing
    catch error
        error
    end
    @test abi_error isa ArgumentError
    message = sprint(showerror, abi_error)
    @test occursin("runtime ABI mismatch", message)
    @test occursin("expected", message)
    @test occursin("observed", message)
    @test occursin("explicitly recompile", message)
    @test backend_hits[] == 0
end
