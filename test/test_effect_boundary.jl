using ReactiveKernels
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
    @test length(result.outbox.callback.records) == 2
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
        result.outbox.callback, (active=(true,),))
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

    initial_box = ReactiveKernels._sm_observation_outbox(
        (arguments=(1,),), Val(1), 0, false)
    full_box = ReactiveKernels._sm_observation_outbox_push(
        initial_box, (arguments=(2,),), true)
    overflow_box = ReactiveKernels._sm_observation_outbox_push(
        full_box, (arguments=(3,),), true)
    @test overflow_box.count == 1
    @test overflow_box.overflow === true
    @test only(overflow_box.records).arguments == (2,)
    growth_error = try
        ReactiveKernels._sm_observation_outbox_push(
            full_box, (arguments=([3],),), true)
        nothing
    catch error
        error
    end
    @test growth_error isa ArgumentError
    @test occursin("forbidden observational outbox growth",
                   sprint(showerror, growth_error))

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
