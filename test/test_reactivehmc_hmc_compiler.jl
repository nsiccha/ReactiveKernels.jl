using ReactiveKernels
using Test
import TOML

include(joinpath(@__DIR__, "fixtures",
                 "reactivehmc_hmc_compiler_support.jl"))
include(joinpath(@__DIR__, "fixtures",
                 "stateful_functional_contracts.jl"))
const _RHMC_HMC_COMPILER = ReactiveHMCHMCCompilerSupport
const _SFC = StatefulFunctionalContractsFixture

const _structural_copy_hits = Ref(0)
mutable struct _MutableCallable{F}
    f::F
    payload::Vector{Int}
end
(callable::_MutableCallable)(args...) = callable.f(args...)
function Base.deepcopy_internal(callable::_MutableCallable, stack::IdDict)
    _structural_copy_hits[] += 1
    _MutableCallable(callable.f, copy(callable.payload))
end

mutable struct _EffectSource
    hits::Int
end
(source::_EffectSource)(state) = (source.hits += 1; nothing)

struct _ConfiguredEffect{A}
    increment::A
end
(lowering::_ConfiguredEffect)(effect, state) = (
    arguments=(state,), result=nothing,
    effect_state=effect + only(lowering.increment))

mutable struct _MutableEffect
    increment::Int
end
(lowering::_MutableEffect)(effect, state) = (
    arguments=(state,), result=nothing,
    effect_state=effect + lowering.increment)

struct _ArrayEffect end
(::_ArrayEffect)(effect, state) = (
    arguments=(state,), result=nothing, effect_state=effect .+ 1)

function _effect_contract_program(source, port)
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _SFC.effect_contract, bindings, source, 0)
    state = kernel(source, 0)
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; kernel, state, snapshot=ReactiveKernels.stateful_snapshot(state),
       transition)
end

function _pure_contract_program(source, port)
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _SFC.pure_contract, bindings, source, 0.0, 0)
    state = kernel(source, 0.0, 0)
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Float64,Bool})
    (; kernel, state, snapshot=ReactiveKernels.stateful_snapshot(state),
       transition)
end

@testset "functional state domains and source-inactive predication" begin
    dormant_kernel = ReactiveKernels.compile_stateful(_SFC.dormant, -1.0, 0)
    dormant_native = dormant_kernel(-1.0, 0)
    @test ReactiveKernels.stateful_call(
        dormant_native, Val(:step!), true)
    dormant = ReactiveKernels.functionalize_stateful(
        dormant_kernel, Val(:step!); argument_types=Tuple{Bool})
    dormant_state = ReactiveKernels.stateful_snapshot(dormant_native)
    dormant_result = dormant(dormant_state, true)
    @test dormant_result.returned && dormant_result.result
    @test dormant_result.state == dormant_state
    @test !dormant_result.control_overflow

    drift_kernel = ReactiveKernels.compile_stateful(_SFC.drift, 4.0, 0)
    drift = ReactiveKernels.functionalize_stateful(
        drift_kernel, Val(:step!); argument_types=Tuple{Bool})
    @test_throws ArgumentError drift((value=4, count=0), false)

    array_kernel = ReactiveKernels.compile_stateful(
        _SFC.array_drift, [1.0, 2.0], 0)
    array_state = ReactiveKernels.stateful_snapshot(
        array_kernel([1.0, 2.0], 0))
    array_transition = ReactiveKernels.functionalize_stateful(
        array_kernel, Val(:step!); argument_types=Tuple{Bool})
    @test_throws ArgumentError array_transition(
        merge(array_state, (values=[1.0],)), true)
end

@testset "ordered RNG replay is typed, source-ordered, and fail closed" begin
    normals = reshape(Float32[0.25, -0.5], :, 1)
    replay = ReactiveKernels.OrderedRNGReplay(
        normals, Bool[true], Float64[0.75],
        (:uniform, :normal, :exponential))
    @test eltype(replay.normals) === Float32
    @test eltype(replay.exponentials) === Float64
    @test_throws ArgumentError ReactiveKernels.OrderedRNGReplay(
        reshape([1, 2], :, 1), Bool[true], Float64[0.5], (:normal,))
    @test_throws ArgumentError ReactiveKernels.OrderedRNGReplay(
        reshape([NaN, 0.0], :, 1), Bool[true], Float64[0.5], (:normal,))
    @test_throws ArgumentError ReactiveKernels.OrderedRNGReplay(
        reshape([0.0, 0.0], :, 1), Bool[true], Float64[-0.5],
        (:exponential,))
    @test_throws ArgumentError ReactiveKernels.OrderedRNGReplay(
        reshape([0.0, 0.0], :, 1), Bool[true], Int[1],
        (:exponential,))
    @test_throws MethodError ReactiveKernels.OrderedRNGReplay(
        replay.normals, replay.uniforms, replay.exponentials,
        replay.event_tokens, 1, 1, 1, 1, false)

    kernel = ReactiveKernels.compile_stateful(_SFC.uniform_state, false, 0)
    state = ReactiveKernels.stateful_snapshot(kernel(false, 0))
    uniform = ReactiveKernels.OrderedRNGReplay(
        reshape([0.0], 1, 1), Bool[true], Float64[0.0], (:uniform,))
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!);
        argument_types=Tuple{typeof(uniform),Bool})
    inactive = transition(state, uniform, false)
    @test !inactive.result && inactive.returned
    @test inactive.arguments[1].event_index == 1
    @test inactive.arguments[1].uniform_index == 1
    @test inactive.state == state

    active = transition(state, uniform, true)
    @test active.result && active.state.value
    @test active.arguments[1].event_index == 2
    @test active.arguments[1].uniform_index == 2
    @test !active.arguments[1].overflow

    exhausted = ReactiveKernels._sm_ordered_rng_with_cursors(
        uniform, 1, 2, 1, 1, false)
    overflow = transition(state, exhausted, true)
    @test overflow.control_overflow && overflow.arguments[1].overflow
    @test overflow.state == state

    wrong_order = ReactiveKernels.OrderedRNGReplay(
        reshape([0.0], 1, 1), Bool[true], Float64[0.0], (:normal,))
    mismatch = transition(state, wrong_order, true)
    @test mismatch.control_overflow && mismatch.arguments[1].overflow
    @test mismatch.arguments[1].event_index == 1
    @test mismatch.state == state

    bad_kernel = ReactiveKernels.compile_stateful(
        _SFC.bad_uniform_state, false, 0)
    @test_throws ReactiveKernels._LLowerReject begin
        ReactiveKernels.functionalize_stateful(
            bad_kernel, Val(:step!);
            argument_types=Tuple{typeof(uniform),Bool})
    end

    normal_statement_error = try
        ReactiveKernels.compile_stateful(
            _SFC.normal_statement, zeros(2))
        nothing
    catch error
        error
    end
    @test normal_statement_error isa ReactiveKernels._LLowerReject
    @test occursin("unsupported straight-line statement",
                   sprint(showerror, normal_statement_error))

    @test_throws ReactiveKernels._LLowerReject begin
        ReactiveKernels.compile_stateful(_SFC.normal_indexed, zeros(2))
    end

end

@testset "callable/effect ABIs, authorities, and continuation fail closed" begin
    source = identity
    unwrapped = ReactiveKernels.pure_callable_port(
        source, Tuple{Float64}, Float64;
        functional_lowering=x -> x)
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=unwrapped)
    unwrapped_kernel = ReactiveKernels.compile_stateful(
        _SFC.pure_contract, bindings, source, 0.0, 0)
    @test_throws ReactiveKernels._LLowerReject begin
        ReactiveKernels.functionalize_stateful(
            unwrapped_kernel, Val(:step!);
            argument_types=Tuple{Float64,Bool})
    end

    wrong_pure = ReactiveKernels.pure_callable_port(
        source, Tuple{Float64}, Float64;
        functional_lowering=ReactiveKernels.total_functional_lowering(
            x -> 1))
    pure_program = _pure_contract_program(source, wrong_pure)
    @test_throws ArgumentError pure_program.transition(
        pure_program.snapshot, 2.0, true)

    effect_source = _EffectSource(0)
    unwrapped_effect_hits = Ref(0)
    unwrapped_effect = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=0,
        functional_lowering=(effect, state) -> begin
            unwrapped_effect_hits[] += 1
            error("source-inactive effect lowering executed")
        end)
    unwrapped_effect_bindings = ReactiveKernels.stateful_compiler_bindings(
        callback=unwrapped_effect)
    unwrapped_effect_kernel = ReactiveKernels.compile_stateful(
        _SFC.effect_contract, unwrapped_effect_bindings, effect_source, 0)
    @test_throws ArgumentError begin
        ReactiveKernels.functionalize_stateful(
            unwrapped_effect_kernel, Val(:step!);
            argument_types=Tuple{Bool})
    end
    @test unwrapped_effect_hits[] == 0

    configured = _ConfiguredEffect([2])
    port = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=0,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            configured))
    program = _effect_contract_program(effect_source, port)
    configured.increment[1] = 99
    first_result = program.transition(program.snapshot, true)
    @test first_result.effects.callback == 2
    @test first_result.state.count == 1
    second_result = program.transition(
        first_result.state, true; effects=first_result.effects)
    @test second_result.effects.callback == 4
    @test second_result.state.count == 2

    # Mutating the authoring metadata after functionalization changes the
    # original object but not the compiler-static snapshot.
    effect_source(program.snapshot)
    direct_lowering = configured(0, program.snapshot)
    @test effect_source.hits == 1
    @test direct_lowering.effect_state == 99
    @test direct_lowering.arguments == (program.snapshot,)
    @test direct_lowering.result === nothing

    counterfeit = merge(program.snapshot, (callback=_EffectSource(0),))
    @test typeof(counterfeit) === typeof(program.snapshot)
    @test_throws ArgumentError program.transition(counterfeit, true)

    wrong_result = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=0,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            (effect, state) -> (
                arguments=(state,), result=17,
                effect_state=effect + 1)))
    wrong_result_program = _effect_contract_program(effect_source, wrong_result)
    @test_throws ArgumentError wrong_result_program.transition(
        wrong_result_program.snapshot, true)

    wrong_effect = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=0,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            (effect, state) -> (
                arguments=(state,), result=nothing,
                effect_state=1.5)))
    wrong_effect_program = _effect_contract_program(effect_source, wrong_effect)
    @test_throws ArgumentError wrong_effect_program.transition(
        wrong_effect_program.snapshot, true)

    wrong_arguments = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=0,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            (effect, state) -> (
                arguments=(), result=nothing,
                effect_state=effect + 1)))
    wrong_arguments_program = _effect_contract_program(
        effect_source, wrong_arguments)
    @test_throws ArgumentError wrong_arguments_program.transition(
        wrong_arguments_program.snapshot, true)

    mutable_lowering = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=0,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _MutableEffect(1)))
    mutable_bindings = ReactiveKernels.stateful_compiler_bindings(
        callback=mutable_lowering)
    mutable_kernel = ReactiveKernels.compile_stateful(
        _SFC.effect_contract, mutable_bindings, effect_source, 0)
    @test_throws ArgumentError ReactiveKernels.functionalize_stateful(
        mutable_kernel, Val(:step!); argument_types=Tuple{Bool})

    initial_array = [0]
    array_port = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=initial_array,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _ArrayEffect()))
    array_program = _effect_contract_program(effect_source, array_port)
    initial_array[1] = 41
    array_result = array_program.transition(array_program.snapshot, true)
    @test array_result.effects.callback == [1]
    @test_throws ArgumentError array_program.transition(
        array_program.snapshot, true; effects=(callback=[0, 0],))

    @test_throws ArgumentError ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=_EffectSource(0),
        functional_lowering=ReactiveKernels.total_functional_lowering(
            (effect, state) -> (
                arguments=(state,), result=nothing,
                effect_state=effect)))
end

@testset "structural copies preserve callable authority without user dispatch" begin
    receipt = TOML.parsefile(joinpath(
        @__DIR__, "..", "benchmark", "receipts",
        "reactivehmc-hmc-ca9-v1.toml"))
    case = only(filter(case -> case["name"] == "accepted", receipt["cases"]))
    potential_callable = _MutableCallable(
        _RHMC_HMC_COMPILER.potential, [1])
    gradient_callable = _MutableCallable(
        _RHMC_HMC_COMPILER.gradient, [2])
    _structural_copy_hits[] = 0
    program = _RHMC_HMC_COMPILER.build_case(
        case; potential_f=potential_callable,
        gradient_f=gradient_callable)
    @test _structural_copy_hits[] == 0
    @test program.snapshot.init.pot_f === potential_callable
    @test program.snapshot.init.grad_f === gradient_callable
    @test program.snapshot.fwd.pot_f === potential_callable
    @test program.snapshot.fwd.grad_f === gradient_callable
    @test program.snapshot.init.pos !== program.snapshot.fwd.pos
    @test program.snapshot.init.dpot === program.snapshot.init.dham_dpos
    @test program.snapshot.fwd.dpot === program.snapshot.fwd.dham_dpos

    first_state = ReactiveKernels.initial_transition_state(program.endpoint)
    second_state = ReactiveKernels.initial_transition_state(program.endpoint)
    @test _structural_copy_hits[] == 0
    @test first_state.pot_f === potential_callable
    @test second_state.grad_f === gradient_callable
    @test first_state.pos !== second_state.pos

    replaced = merge(first_state, (
        pot_f=_MutableCallable(_RHMC_HMC_COMPILER.potential, [3]),))
    @test typeof(replaced) === typeof(first_state)
    @test_throws ArgumentError program.endpoint(replaced)
    @test_throws ArgumentError program.transition(
        merge(program.snapshot, (init=replaced,)), program.replay)

    wrong_shape = merge(first_state, (pos=[first(first_state.pos)],))
    @test_throws ArgumentError program.endpoint(wrong_shape)

    # The endpoint lowering is independently source-derived from the authored
    # integrator and is used through an explicit non-executable authority port.
    @test_throws ArgumentError ReactiveKernels._sm_checked_effect_call(
        program.step_port, program.step_source, first_state)
    endpoint_expected = program.endpoint(first_state)
    lowering_candidate = _RHMC_HMC_COMPILER.step_lowering(program.endpoint)(
        nothing, first_state)
    @test lowering_candidate.arguments == (endpoint_expected,)
    @test lowering_candidate.result === nothing
    @test lowering_candidate.effect_state === nothing

    stats_source = _RHMC_HMC_COMPILER.StatisticsRecorder(zeros(1), 0)
    stats_source(program.snapshot)
    stats_candidate = _RHMC_HMC_COMPILER.stats_lowering(
        (errors=(0.0,), count=0), program.snapshot)
    @test stats_candidate.result === nothing
    @test stats_candidate.arguments == (program.snapshot,)
    @test stats_candidate.effect_state.count == stats_source.count
    @test only(stats_candidate.effect_state.errors) ==
          only(stats_source.errors)
end

function _assert_hmc_receipt(actual, expected)
    @test collect(actual.init_pos) == expected["init_pos"]
    @test collect(actual.init_mom) == expected["init_mom"]
    @test actual.init_ham == expected["init_ham"]
    @test collect(actual.fwd_pos) == expected["fwd_pos"]
    @test collect(actual.fwd_mom) == expected["fwd_mom"]
    @test actual.fwd_ham == expected["fwd_ham"]
    @test actual.dham == expected["dham"]
    @test actual.diverged == expected["diverged"]
    @test collect(actual.energy_errors) == expected["energy_errors"]
    @test actual.normal_calls == expected["normal_calls"]
    @test actual.exponential_calls == expected["exponential_calls"]
    @test actual.rng_event_calls == length(expected["rng_events"])
    @test !actual.rng_overflow
    @test !actual.control_overflow
end

@testset "generic structured compiler reproduces fixed-step HMC" begin
    receipt = TOML.parsefile(joinpath(
        @__DIR__, "..", "benchmark", "receipts",
        "reactivehmc-hmc-ca9-v1.toml"))
    for case in receipt["cases"]
        compiled = _RHMC_HMC_COMPILER.build_case(case)
        @test propertynames(compiled.structured.repairs) == (:mom, :pos)
        @test compiled.snapshot.init.dpot ===
              compiled.snapshot.init.dham_dpos
        @test compiled.snapshot.fwd.dpot ===
              compiled.snapshot.fwd.dham_dpos

        result = compiled.transition(compiled.snapshot, compiled.replay)
        _assert_hmc_receipt(
            _RHMC_HMC_COMPILER.result_values(result), case)
        @test result.state.init.dpot === result.state.init.dham_dpos
        @test result.state.fwd.dpot === result.state.fwd.dham_dpos
        @test result.state.init.pos !== result.state.fwd.pos
        @test result.state.init.mom !== result.state.fwd.mom
        @test result.state.init.pot_f ===
              compiled.snapshot.init.pot_f
        @test result.state.fwd.grad_f ===
              compiled.snapshot.fwd.grad_f
    end


    mixed = only(receipt["mixed_precision_cases"])
    compiled = _RHMC_HMC_COMPILER.build_case(
        mixed; numeric_type=Float32)
    @test eltype(compiled.replay.normals) === Float32
    @test eltype(compiled.replay.exponentials) === Float64
    result = compiled.transition(compiled.snapshot, compiled.replay)
    actual = _RHMC_HMC_COMPILER.result_values(result)
    _assert_hmc_receipt(actual, mixed)
    @test bitstring.(collect(actual.init_pos)) == mixed["init_pos_bits"]
    @test bitstring.(collect(actual.init_mom)) == mixed["init_mom_bits"]
    @test [bitstring(actual.init_ham)] == mixed["init_ham_bits"]
    @test bitstring.(collect(actual.fwd_pos)) == mixed["fwd_pos_bits"]
    @test bitstring.(collect(actual.fwd_mom)) == mixed["fwd_mom_bits"]
    @test [bitstring(actual.fwd_ham)] == mixed["fwd_ham_bits"]
    @test bitstring.(collect(actual.energy_errors)) ==
          mixed["energy_error_bits"]
    @test [bitstring(actual.dham)] == mixed["dham_bits"]

    normal_bindings = ReactiveKernels.stateful_compiler_bindings(
        init=compiled.structured)
    initial_point = ReactiveKernels.initial_transition_state(compiled.endpoint)
    normal_kernel = ReactiveKernels.compile_stateful(
        _SFC.normal_replace, normal_bindings, initial_point, 0)
    normal_state = ReactiveKernels.stateful_snapshot(
        normal_kernel(initial_point, 0))
    normal_transition = ReactiveKernels.functionalize_stateful(
        normal_kernel, Val(:step!);
        argument_types=Tuple{typeof(compiled.replay),Bool})
    normal_result = normal_transition(
        normal_state, compiled.replay, true)
    @test normal_result.returned && normal_result.result
    @test eltype(normal_result.state.init.mom) === Float32
    @test bitstring.(normal_result.state.init.mom) ==
          mixed["normal_draw_bits"]
    @test normal_result.arguments[1].normal_index == 2
    @test normal_result.arguments[1].event_index == 2
    @test normal_result.state.count == 1
    @test !normal_result.arguments[1].overflow

    widened_replay = ReactiveKernels.OrderedRNGReplay(
        Float64.(compiled.replay.normals), compiled.replay.uniforms,
        compiled.replay.exponentials,
        Tuple(Symbol.(mixed["rng_events"])))
    @test_throws ArgumentError compiled.transition(
        compiled.snapshot, widened_replay)
    @test_throws ArgumentError normal_transition(
        normal_state, widened_replay, true)
end

@testset "structured state and ordered effects fail closed" begin
    case = only(filter(
        case -> case["name"] == "accepted",
        TOML.parsefile(joinpath(
            @__DIR__, "..", "benchmark", "receipts",
            "reactivehmc-hmc-ca9-v1.toml"))["cases"]))
    compiled = _RHMC_HMC_COMPILER.build_case(case)

    # Exhausting the conditional exponential stream is observable and atomic:
    # the entire state/effect surface rolls back while overflow stays sticky on
    # the replay argument.
    replay = compiled.replay
    exhausted = ReactiveKernels._sm_ordered_rng_with_cursors(
        replay, replay.normal_index, replay.uniform_index,
        length(replay.exponentials) + 1, replay.event_index, false)
    result = compiled.transition(compiled.snapshot, exhausted)
    @test result.control_overflow
    @test result.arguments[1].overflow
    @test result.arguments[1].normal_index ==
          exhausted.normal_index + 1
    @test result.arguments[1].exponential_index ==
          exhausted.exponential_index
    @test result.state == compiled.snapshot
    @test result.effects.stats_f.count == 0
    @test all(iszero, result.effects.stats_f.errors)

    @test_throws ArgumentError ReactiveKernels.OrderedRNGReplay(
        reshape([0.1, -0.2], :, 1), Bool[], [0.5], (:normal,))
    @test_throws ArgumentError ReactiveKernels.OrderedRNGReplay(
        reshape([0.1, -0.2], :, 1), Bool[false], Float64[], (:normal,))

    wrong = merge(compiled.snapshot.init,
                  (; pot_f = x -> sum(abs2, x)))
    @test_throws ArgumentError ReactiveKernels._sm_validate_structured_state_port(
        compiled.structured, wrong)

    # A compiled transition freezes its repair programs. Later mutation of the
    # authoring graph cannot change a newly constructed structured port.
    @test !(:spec in fieldnames(typeof(compiled.endpoint)))
    replacement = compiled.snapshot.init.mom .+ 0.125
    changed = ReactiveKernels._sm_structured_set(
        compiled.structured, compiled.snapshot.init,
        Val((:mom,)), replacement)
    expected = compiled.structured.repairs.mom(changed)
    empty!(compiled.bundle.spec.graph.recipes)
    empty!(compiled.bundle.spec.graph.producers)
    frozen = ReactiveKernels.structured_state_port(compiled.endpoint)
    actual = frozen.repairs.mom(changed)
    @test actual.mom == replacement
    @test actual.dham_dmom == expected.dham_dmom
    @test actual.ham == expected.ham
end
