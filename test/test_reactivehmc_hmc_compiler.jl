using ReactiveKernels
using LinearAlgebra
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

function _sequential_effect_contract_program(source, port)
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _SFC.sequential_effect_contract, bindings, source, 0)
    state = kernel(source, 0)
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; kernel, state, snapshot=ReactiveKernels.stateful_snapshot(state),
       transition)
end

function _authored_nested_storage_program()
    endpoint = _SFC.nested_storage_transition()
    initial = ReactiveKernels.initial_transition_state(endpoint)
    port = ReactiveKernels.structured_state_port(endpoint)
    bindings = ReactiveKernels.stateful_compiler_bindings(initial=port)
    kernel = ReactiveKernels.compile_stateful(
        _SFC.authored_nested_storage_write, bindings, initial, 0)
    state = kernel(initial, 0)
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Vector{Float64}})
    (; endpoint, initial, port, kernel, state,
       snapshot=ReactiveKernels.stateful_snapshot(state), transition)
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

function _structured_replacement_program(compiled, lowering)
    source = _EffectSource(0)
    point = compiled.snapshot.init
    port = ReactiveKernels.effect_lowering_port(
        source, Tuple{typeof(point)}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            lowering))
    bindings = ReactiveKernels.stateful_compiler_bindings(
        init=compiled.structured, callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _SFC.structured_effect, bindings, point, source, 0)
    state = ReactiveKernels.stateful_snapshot(kernel(point, source, 0))
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, transition)
end

function _paired_structured_replacement_program(compiled, lowering)
    source = _EffectSource(0)
    point = compiled.snapshot.init
    port = ReactiveKernels.effect_lowering_port(
        source, Tuple{typeof(point),typeof(point)}, Nothing;
        written_arguments=(1, 2), initial_effect_state=nothing,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            lowering))
    bindings = ReactiveKernels.stateful_compiler_bindings(
        init=compiled.structured, fwd=compiled.structured, callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _SFC.paired_structured_effect, bindings,
        point, point, source, 0)
    state = ReactiveKernels.stateful_snapshot(
        kernel(point, point, source, 0))
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, transition)
end

function _array_replacement_program(lowering)
    source = _EffectSource(0)
    port = ReactiveKernels.effect_lowering_port(
        source, Tuple{Vector{Float64}}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            lowering))
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _SFC.array_effect, bindings, [1.0, 2.0], source, 0)
    state = ReactiveKernels.stateful_snapshot(
        kernel([1.0, 2.0], source, 0))
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, transition)
end

function _outer_alias_replacement_program(lowering)
    source = _EffectSource(0)
    port = ReactiveKernels.effect_lowering_port(
        source, Tuple{Vector{Float64},Vector{Float64}}, Nothing;
        written_arguments=(1, 2), initial_effect_state=nothing,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            lowering))
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _SFC.outer_alias_effect, bindings, [0.0], source, 0)
    state = ReactiveKernels.stateful_snapshot(
        kernel([0.0], source, 0))
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, transition)
end


function _independent_scalar_replacement_program()
    source = _EffectSource(0)
    port = ReactiveKernels.effect_lowering_port(
        source, Tuple{Int,Int}, Nothing;
        written_arguments=(1, 2), initial_effect_state=nothing,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _SFC.IndependentScalarReplacement()))
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _SFC.independent_equal_scalars, bindings, 1, 1, source, 0)
    state = ReactiveKernels.stateful_snapshot(kernel(1, 1, source, 0))
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, transition)
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

    zero_kernel = ReactiveKernels.compile_stateful(
        _SFC.zero_argument_branch, true, 0)
    zero_native = zero_kernel(true, 0)
    @test ReactiveKernels.stateful_call(zero_native, Val(:step!))
    @test ReactiveKernels.stateful_snapshot(zero_native).count == 1
    zero_transition = ReactiveKernels.functionalize_stateful(
        zero_kernel, Val(:step!); argument_types=Tuple{})
    zero_result = zero_transition(
        ReactiveKernels.stateful_snapshot(zero_kernel(true, 0)))
    @test zero_result.returned && zero_result.result
    @test zero_result.state.count == 1

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

    float_kernel = ReactiveKernels.compile_stateful(
        _SFC.float_first, zeros(4), 1.0, 0)
    float_native = float_kernel(zeros(4), 1.0, 0)
    @test ReactiveKernels.stateful_call(
        float_native, Val(:step!), 2)
    native_float_state = ReactiveKernels.stateful_snapshot(float_native)
    @test native_float_state.arr == [1.0, 1.0, 0.0, 0.0]
    @test native_float_state.count == 2

    float_state = ReactiveKernels.stateful_snapshot(
        float_kernel(zeros(4), 1.0, 0))
    float_transition = ReactiveKernels.functionalize_stateful(
        float_kernel, Val(:step!);
        max_iterations=4, argument_types=Tuple{Int})
    float_result = float_transition(float_state, 2)
    @test float_result.returned && float_result.result
    @test !float_result.control_overflow
    @test float_result.state.arr == [1.0, 1.0, 0.0, 0.0]
    @test float_result.state.count == 2

    loop_kernel = ReactiveKernels.compile_stateful(_SFC.loop_probe, 0)
    loop_state = ReactiveKernels.stateful_snapshot(loop_kernel(0))
    loop_transition = ReactiveKernels.functionalize_stateful(
        loop_kernel, Val(:step!);
        max_iterations=4, argument_types=Tuple{Int,Int})
    overflow = loop_transition(loop_state, typemin(Int), typemax(Int))
    @test overflow.control_overflow
    @test !overflow.returned
    @test overflow.state == loop_state

    edge = loop_transition(loop_state, typemax(Int) - 1, typemax(Int))
    @test !edge.control_overflow
    @test edge.returned && edge.result
    @test edge.state.count == 2

    small_kernel = ReactiveKernels.compile_stateful(
        _SFC.small_loop, Int8(0))
    small_state = ReactiveKernels.stateful_snapshot(
        small_kernel(Int8(0)))
    small_transition = ReactiveKernels.functionalize_stateful(
        small_kernel, Val(:step!);
        max_iterations=4, argument_types=Tuple{Int8,Int8})
    small = small_transition(small_state, Int8(1), Int8(2))
    @test small.returned && small.result
    @test !small.control_overflow
    @test small.state.count === Int8(3)

    mixed_kernel = ReactiveKernels.compile_stateful(
        _SFC.mixed_loop, 0, Int8(0))
    mixed_state = ReactiveKernels.stateful_snapshot(
        mixed_kernel(0, Int8(0)))
    mixed_transition = ReactiveKernels.functionalize_stateful(
        mixed_kernel, Val(:step!);
        max_iterations=4, argument_types=Tuple{Int8,Int8})
    mixed = mixed_transition(mixed_state, Int8(1), Int8(2))
    @test mixed.returned && mixed.result
    @test !mixed.control_overflow
    @test mixed.state.count == 2
    @test mixed.state.last === Int8(2)

    alias_kernel = ReactiveKernels.compile_stateful(
        _SFC.outer_alias, [0.0], 0.0, 0)
    alias_state = ReactiveKernels.stateful_snapshot(
        alias_kernel([0.0], 0.0, 0))
    @test alias_state.a === alias_state.b
    alias_transition = ReactiveKernels.functionalize_stateful(
        alias_kernel, Val(:step!); argument_types=Tuple{Bool})
    counterfeit_alias = merge(alias_state, (b=[9.0],))
    @test_throws ArgumentError alias_transition(counterfeit_alias, true)
    alias_result = alias_transition(alias_state, true)
    @test alias_result.state.a === alias_result.state.b
    @test alias_result.state.a == [2.0]
    @test alias_result.state.total == 0.0

    backing = [1.0, 2.0]
    wrapped = (metric=Diagonal(backing), diagonal=backing)
    wrapped_kernel = ReactiveKernels.compile_stateful(
        _SFC.wrapped_storage_alias, wrapped, 0)
    wrapped_state = ReactiveKernels.stateful_snapshot(
        wrapped_kernel(wrapped, 0))
    @test wrapped_state.initial.metric.diag ===
          wrapped_state.initial.diagonal
    wrapped_transition = ReactiveKernels.functionalize_stateful(
        wrapped_kernel, Val(:step!); argument_types=Tuple{Bool})
    copied_metric = Diagonal(copy(wrapped_state.initial.diagonal))
    broken_wrapped = merge(wrapped_state, (initial=merge(
        wrapped_state.initial, (metric=copied_metric,)),))
    @test_throws ArgumentError wrapped_transition(broken_wrapped, true)
    wrapped_result = wrapped_transition(wrapped_state, true)
    @test wrapped_result.state.initial.metric.diag ===
          wrapped_result.state.initial.diagonal
    @test wrapped_result.state.count == 1

    straight_kernel = ReactiveKernels.compile_stateful(
        _SFC.straight_alias, [0.0])
    straight_state = ReactiveKernels.stateful_snapshot(
        straight_kernel([0.0]))
    @test straight_state.a === straight_state.b
    straight_transition = ReactiveKernels.functionalize_stateful(
        straight_kernel, Val(:fit!))
    @test_throws ArgumentError straight_transition(
        merge(straight_state, (b=[9.0],)), [3.0])
    straight_result = straight_transition(straight_state, [3.0])
    @test straight_result.a === straight_result.b
    @test straight_result.a == [3.0]
    @test typeof(straight_transition).parameters[4] === true
    guarded_straight = ReactiveKernels.validated_compiled_transition(
        straight_transition, straight_transition)
    guarded_straight_result = guarded_straight(straight_state, [3.0])
    @test guarded_straight_result.a === guarded_straight_result.b
    @test guarded_straight_result.a == [3.0]

    result_kernel = ReactiveKernels.compile_stateful(
        _SFC.straight_result_contract, 2.0, 3)
    result_native = result_kernel(2.0, 3)
    result_state = ReactiveKernels.stateful_snapshot(result_native)
    @test ReactiveKernels.stateful_call(
        result_native, Val(:named_result), 2.0) ==
        (value=4.0, count=3)
    @test ReactiveKernels.stateful_call(
        result_native, Val(:nothing_result), true) === nothing
    @test_throws ReactiveKernels._LLowerReject begin
        ReactiveKernels.functionalize_stateful(
            result_kernel, Val(:named_result))
    end
    named_result = ReactiveKernels.functionalize_stateful(
        result_kernel, Val(:named_result);
        argument_types=Tuple{Float64})
    @test typeof(named_result).parameters[4] === false
    @test typeof(named_result).parameters[7] ===
          NamedTuple{(:value,:count),Tuple{Float64,Int}}
    @test named_result(result_state, 2.0) == (value=4.0, count=3)
    guarded_named = ReactiveKernels.validated_compiled_transition(
        named_result, named_result)
    @test guarded_named(result_state, 2.0) == (value=4.0, count=3)
    wrong_named = ReactiveKernels.validated_compiled_transition(
        (state, scale) -> (value=4.0, count=3.0), named_result)
    @test_throws ArgumentError wrong_named(result_state, 2.0)

    nothing_result = ReactiveKernels.functionalize_stateful(
        result_kernel, Val(:nothing_result);
        argument_types=Tuple{Bool})
    @test typeof(nothing_result).parameters[4] === false
    @test typeof(nothing_result).parameters[7] === Nothing
    @test nothing_result(result_state, true) === nothing
    guarded_nothing = ReactiveKernels.validated_compiled_transition(
        nothing_result, nothing_result)
    @test guarded_nothing(result_state, false) === nothing
    wrong_nothing = ReactiveKernels.validated_compiled_transition(
        (state, take) -> result_state, nothing_result)
    @test_throws ArgumentError wrong_nothing(result_state, true)

    independent_kernel = ReactiveKernels.compile_stateful(
        _SFC.independent_arrays_machine, [1.0], [2.0], 0)
    independent_state = ReactiveKernels.stateful_snapshot(
        independent_kernel([1.0], [2.0], 0))
    @test independent_state.a !== independent_state.b
    merged_independent = merge(
        independent_state, (b=independent_state.a,))
    independent_machine = ReactiveKernels.functionalize_stateful(
        independent_kernel, Val(:step!);
        argument_types=Tuple{Bool})
    @test_throws ArgumentError independent_machine(
        merged_independent, true)
    machine_result = independent_machine(
        independent_state, true)
    @test machine_result.state.a !== machine_result.state.b
    @test machine_result.state.a == [2.0]
    @test machine_result.state.b == [2.0]
    independent_straight_kernel = ReactiveKernels.compile_stateful(
        _SFC.independent_arrays_straight, [1.0], [2.0], 0)
    independent_straight_state = ReactiveKernels.stateful_snapshot(
        independent_straight_kernel([1.0], [2.0], 0))
    @test independent_straight_state.a !== independent_straight_state.b
    merged_straight = merge(independent_straight_state,
        (b=independent_straight_state.a,))
    independent_straight = ReactiveKernels.functionalize_stateful(
        independent_straight_kernel, Val(:fit!))
    @test_throws ArgumentError independent_straight(
        merged_straight, [3.0])
    independent_result = independent_straight(
        independent_straight_state, [3.0])
    @test independent_result.a !== independent_result.b
    @test independent_result.a == [5.0]
    @test independent_result.b == [2.0]
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

    wrong_effect_shape = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=[0.0, 0.0],
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _SFC.WrongShapeEffectState()))
    wrong_effect_shape_program = _effect_contract_program(
        effect_source, wrong_effect_shape)
    @test_throws ArgumentError wrong_effect_shape_program.transition(
        wrong_effect_shape_program.snapshot, true)

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

    effect_backing = [0.0]
    aliased_effect = (left=effect_backing, right=effect_backing)
    alias_effect_port = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=aliased_effect,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _SFC.PassThroughEffectState()))
    alias_effect_program = _effect_contract_program(
        effect_source, alias_effect_port)
    alias_effects = ReactiveKernels.initial_transition_effects(
        alias_effect_program.transition)
    @test alias_effects.callback.left === alias_effects.callback.right
    alias_effect_result = alias_effect_program.transition(
        alias_effect_program.snapshot, true; effects=alias_effects)
    @test alias_effect_result.effects.callback.left ===
          alias_effect_result.effects.callback.right
    @test alias_effect_result.state.count == 1
    broken_effects = (callback=(
        left=alias_effects.callback.left,
        right=copy(alias_effects.callback.right)),)
    @test_throws ArgumentError alias_effect_program.transition(
        alias_effect_program.snapshot, true; effects=broken_effects)

    broken_effect_port = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=aliased_effect,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _SFC.BrokenAliasEffectState()))
    broken_effect_program = _effect_contract_program(
        effect_source, broken_effect_port)
    @test_throws ArgumentError broken_effect_program.transition(
        broken_effect_program.snapshot, true)

    distinct_effect = (left=[0.0], right=[0.0])
    merged_effect_port = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=distinct_effect,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _SFC.MergedAliasEffectState()))
    merged_effect_program = _effect_contract_program(
        effect_source, merged_effect_port)
    @test_throws ArgumentError merged_effect_program.transition(
        merged_effect_program.snapshot, true)

    config_backing = [2]
    config = _SFC.AliasConfig(config_backing, config_backing)
    frozen_config = ReactiveKernels._sm_compiler_static_snapshot(config)
    @test frozen_config.left === frozen_config.right
    @test frozen_config.left !== config_backing
    config_port = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=0,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _SFC.AliasConfiguredEffect(config)))
    config_program = _effect_contract_program(effect_source, config_port)
    config_backing[1] = 99
    config_result = config_program.transition(config_program.snapshot, true)
    @test config_result.effects.callback == 2
    @test config_result.state.count == 1

    sequential_backing = [0.0]
    sequential_effect = (left=sequential_backing, right=sequential_backing)
    sequential_port = ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=sequential_effect,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _SFC.AliasSensitiveSequentialEffect()))
    sequential_program = _sequential_effect_contract_program(
        effect_source, sequential_port)
    sequential_result = sequential_program.transition(
        sequential_program.snapshot, true)
    @test sequential_result.effects.callback.left ===
          sequential_result.effects.callback.right
    @test sequential_result.effects.callback.left == [2.0]
    @test sequential_result.state.count == 1
    inactive_sequential = sequential_program.transition(
        sequential_program.snapshot, false)
    @test inactive_sequential.effects.callback.left ===
          inactive_sequential.effects.callback.right
    @test inactive_sequential.effects.callback.left == [0.0]
    @test inactive_sequential.state.count == 0

    @test_throws ArgumentError ReactiveKernels.effect_callable_port(
        effect_source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=_EffectSource(0),
        functional_lowering=ReactiveKernels.total_functional_lowering(
            (effect, state) -> (
                arguments=(state,), result=nothing,
                effect_state=effect)))
end

@testset "recursive structured state operations preserve authored topology" begin
    transition = _SFC.nested_storage_transition()
    initial = ReactiveKernels.initial_transition_state(transition)
    @test initial.nested.metric.diag === initial.nested.diagonal
    @test initial.nested.diagonal == [1.0, 2.0]
    @test initial.nested.diagonal !== transition.initial.nested.diagonal

    port = ReactiveKernels.structured_state_port(transition)
    copied = ReactiveKernels._sm_structured_copy(port, initial)
    @test copied.nested.metric.diag === copied.nested.diagonal
    @test copied.nested.diagonal !== initial.nested.diagonal

    replacement = [9.0, 9.0]
    changed = ReactiveKernels._sm_structured_set(
        port, initial, Val((:nested, :diagonal)), replacement)
    @test changed.nested.metric.diag === changed.nested.diagonal
    @test changed.nested.metric.diag == replacement
    @test changed.nested.diagonal == replacement

    selected = ReactiveKernels._sm_structured_predicated_select(
        port, true, changed, initial)
    @test selected.nested.metric.diag === selected.nested.diagonal
    @test selected.nested.diagonal == replacement

    broken_nested = merge(initial, (nested=merge(initial.nested, (
        metric=Diagonal(copy(initial.nested.diagonal)),)),))
    @test_throws ArgumentError ReactiveKernels._sm_structured_copy(
        port, broken_nested)
    @test_throws ArgumentError ReactiveKernels._sm_structured_predicated_select(
        port, true, broken_nested, initial)

    frozen_port = ReactiveKernels._sm_compiler_static_snapshot(port)
    frozen_initial = ReactiveKernels.initial_transition_state(
        frozen_port.transition)
    @test frozen_initial.nested.metric.diag ===
          frozen_initial.nested.diagonal
    @test frozen_initial.nested.diagonal == [1.0, 2.0]
    transition.initial.nested.diagonal[1] = 41.0
    @test frozen_port.transition.initial.nested.diagonal == [1.0, 2.0]

    authored = _authored_nested_storage_program()
    replacement = [9.0, 9.0]
    ReactiveKernels.stateful_call!(
        authored.state, Val(:step!), replacement)
    native = ReactiveKernels.stateful_snapshot(authored.state)
    @test native.initial.nested.metric.diag ===
          native.initial.nested.diagonal
    @test native.initial.nested.metric.diag == replacement
    @test native.initial.nested.diagonal == replacement
    @test native.count == 1

    functional = authored.transition(authored.snapshot, replacement)
    @test functional.state.initial.nested.metric.diag ===
          functional.state.initial.nested.diagonal
    @test functional.state.initial.nested.metric.diag == replacement
    @test functional.state.initial.nested.diagonal == replacement
    @test functional.state.count == 1
end

@testset "validated backend outputs reject alias conflicts before repair" begin
    endpoint = _SFC.nested_storage_transition()
    endpoint_state = ReactiveKernels.initial_transition_state(endpoint)
    conflicting_endpoint = merge(endpoint_state, (nested=merge(
        endpoint_state.nested, (
            metric=Diagonal(copy(endpoint_state.nested.diagonal)),
            diagonal=[9.0, 9.0],)),))
    fake_endpoint = _ -> conflicting_endpoint
    guarded_endpoint = ReactiveKernels.validated_compiled_transition(
        fake_endpoint, endpoint)
    @test_throws ArgumentError guarded_endpoint(endpoint_state)

    machine = _authored_nested_storage_program()
    machine_result = machine.transition(
        machine.snapshot, [9.0, 9.0])
    conflicting_machine_state = merge(machine_result.state, (
        initial=merge(machine_result.state.initial, (nested=merge(
            machine_result.state.initial.nested, (
                metric=Diagonal(copy(
                    machine_result.state.initial.nested.diagonal)),
                diagonal=[7.0, 7.0],)),)),))
    conflicting_machine_result = merge(
        machine_result, (state=conflicting_machine_state,))
    fake_machine = (state, arguments...) -> conflicting_machine_result
    guarded_machine = ReactiveKernels.validated_compiled_transition(
        fake_machine, machine.transition)
    @test_throws ArgumentError guarded_machine(
        machine.snapshot, [9.0, 9.0])

    missing_machine_fields = (state=machine_result.state,)
    guarded_missing_machine = ReactiveKernels.validated_compiled_transition(
        (state, arguments...) -> missing_machine_fields,
        machine.transition)
    @test_throws ArgumentError guarded_missing_machine(
        machine.snapshot, [9.0, 9.0])
    extra_machine_field = merge(machine_result, (extra=:forbidden,))
    guarded_extra_machine = ReactiveKernels.validated_compiled_transition(
        (state, arguments...) -> extra_machine_field,
        machine.transition)
    @test_throws ArgumentError guarded_extra_machine(
        machine.snapshot, [9.0, 9.0])
    wrong_machine_arguments = merge(machine_result, (arguments=(),))
    guarded_wrong_arguments = ReactiveKernels.validated_compiled_transition(
        (state, arguments...) -> wrong_machine_arguments,
        machine.transition)
    @test_throws ArgumentError guarded_wrong_arguments(
        machine.snapshot, [9.0, 9.0])
    wrong_machine_result = merge(machine_result, (result=1,))
    guarded_wrong_result = ReactiveKernels.validated_compiled_transition(
        (state, arguments...) -> wrong_machine_result,
        machine.transition)
    @test_throws ArgumentError guarded_wrong_result(
        machine.snapshot, [9.0, 9.0])
    wrong_machine_control = merge(
        machine_result, (returned=1, control_overflow=0,))
    guarded_wrong_control = ReactiveKernels.validated_compiled_transition(
        (state, arguments...) -> wrong_machine_control,
        machine.transition)
    @test_throws ArgumentError guarded_wrong_control(
        machine.snapshot, [9.0, 9.0])
    missing_machine_effects = (
        state=machine_result.state,
        arguments=machine_result.arguments,
        result=machine_result.result,
        returned=machine_result.returned,
        control_overflow=machine_result.control_overflow,
    )
    guarded_missing_effects = ReactiveKernels.validated_compiled_transition(
        (state, arguments...) -> missing_machine_effects,
        machine.transition)
    @test_throws ArgumentError guarded_missing_effects(
        machine.snapshot, [9.0, 9.0])
    extra_machine_effect = merge(
        machine_result, (effects=(extra=1,),))
    guarded_extra_effect = ReactiveKernels.validated_compiled_transition(
        (state, arguments...) -> extra_machine_effect,
        machine.transition)
    @test_throws ArgumentError guarded_extra_effect(
        machine.snapshot, [9.0, 9.0])

    machine_runner = ReactiveKernels.transition_with_effects(
        machine.transition)
    machine_effects = ReactiveKernels.initial_transition_effects(
        machine.transition)
    guarded_missing_effect_abi =
        ReactiveKernels.validated_compiled_transition(
            (state, effects, arguments...) -> missing_machine_fields,
            machine_runner)
    @test_throws ArgumentError guarded_missing_effect_abi(
        machine.snapshot, machine_effects, [9.0, 9.0])

    straight_kernel = ReactiveKernels.compile_stateful(
        _SFC.straight_alias, [0.0])
    straight_state = ReactiveKernels.stateful_snapshot(
        straight_kernel([0.0]))
    straight_transition = ReactiveKernels.functionalize_stateful(
        straight_kernel, Val(:fit!))
    straight_result = straight_transition(straight_state, [3.0])
    conflicting_straight = merge(
        straight_result, (b=[9.0],))
    fake_straight = (state, arguments...) -> conflicting_straight
    guarded_straight = ReactiveKernels.validated_compiled_transition(
        fake_straight, straight_transition)
    @test_throws ArgumentError guarded_straight(
        straight_state, [3.0])

    authority_source = _EffectSource(0)
    authority_port = ReactiveKernels.effect_callable_port(
        authority_source,
        Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=nothing,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _SFC.PassThroughEffectState()))
    authority_program = _effect_contract_program(
        authority_source, authority_port)
    authority_result = authority_program.transition(
        authority_program.snapshot, true)
    cloned_authority = _EffectSource(0)
    raw_authority_result = merge(authority_result, (state=merge(
        authority_result.state, (callback=cloned_authority,)),))
    fake_authority = (state, arguments...) -> raw_authority_result
    guarded_authority = ReactiveKernels.validated_compiled_transition(
        fake_authority, authority_program.transition)
    restored_authority = guarded_authority(
        authority_program.snapshot, true)
    @test raw_authority_result.state.callback === cloned_authority
    @test restored_authority.state.callback === authority_source
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

    for name in (:pos, :mom, :dham_dmom)
        counterfeit_init = merge(
            compiled.snapshot.init,
            NamedTuple{(name,)}((getfield(compiled.snapshot.fwd, name),)))
        counterfeit = merge(
            compiled.snapshot, (init=counterfeit_init,))
        @test_throws ArgumentError compiled.transition(
            counterfeit, compiled.replay)
    end

    endpoint_state = ReactiveKernels.initial_transition_state(
        compiled.endpoint)
    @test endpoint_state.pos !== endpoint_state.mom
    merged_endpoint = merge(endpoint_state, (mom=endpoint_state.pos,))
    @test_throws ArgumentError compiled.endpoint(merged_endpoint)
    @test_throws ArgumentError ReactiveKernels.
        _sm_validate_structured_state_port(
            compiled.structured, merged_endpoint)

    broken_alias = _structured_replacement_program(
        compiled, _SFC.StructuredReplacement{:alias}(
            copy(compiled.snapshot.init.dham_dpos)))
    @test_throws ArgumentError broken_alias.transition(
        broken_alias.state, true)

    wrong_structured_shape = _structured_replacement_program(
        compiled, _SFC.StructuredReplacement{:shape}(
            [first(compiled.snapshot.init.pos)]))
    @test_throws ArgumentError wrong_structured_shape.transition(
        wrong_structured_shape.state, true)

    broken_nested_alias = _paired_structured_replacement_program(
        compiled, _SFC.PairedStructuredReplacement{
            :broken_required_alias}())
    @test broken_nested_alias.state.init.dpot ===
          broken_nested_alias.state.init.dham_dpos
    @test broken_nested_alias.state.fwd.dpot ===
          broken_nested_alias.state.fwd.dham_dpos
    @test broken_nested_alias.state.init.pos !==
          broken_nested_alias.state.fwd.pos
    @test_throws ArgumentError broken_nested_alias.transition(
        broken_nested_alias.state, true)

    shared_candidate = _paired_structured_replacement_program(
        compiled, _SFC.PairedStructuredReplacement{
            :cross_canon_share}())
    shared_result = shared_candidate.transition(shared_candidate.state, true)
    @test shared_result.state.init.pos !== shared_result.state.fwd.pos
    @test shared_result.state.init.dpot ===
          shared_result.state.init.dham_dpos
    @test shared_result.state.fwd.dpot ===
          shared_result.state.fwd.dham_dpos

    wrong_array_shape = _array_replacement_program(
        _SFC.ArrayReplacement([1.0]))
    @test_throws ArgumentError wrong_array_shape.transition(
        wrong_array_shape.state, true)

    wrong_outer_alias = _outer_alias_replacement_program(
        _SFC.BrokenOuterAliasReplacement())
    @test wrong_outer_alias.state.a === wrong_outer_alias.state.b
    @test_throws ArgumentError wrong_outer_alias.transition(
        wrong_outer_alias.state, true)

    independent_scalars = _independent_scalar_replacement_program()
    @test independent_scalars.state.left === independent_scalars.state.right
    scalar_result = independent_scalars.transition(
        independent_scalars.state, true)
    @test scalar_result.state.left == 2
    @test scalar_result.state.right == 3
    @test scalar_result.state.count == 1

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

    # Constructor validation is not the dynamic contract: mutable host tapes
    # and traced tape values must be checked at the exact consumed draw.
    invalid_exponentials = copy(replay.exponentials)
    invalid_exponential_replay = ReactiveKernels.OrderedRNGReplay(
        copy(replay.normals), copy(replay.uniforms), invalid_exponentials,
        (:normal, :exponential))
    invalid_exponentials[1] = -1.0
    invalid_exponential = compiled.transition(
        compiled.snapshot, invalid_exponential_replay)
    @test invalid_exponential.control_overflow
    @test invalid_exponential.arguments[1].overflow
    @test invalid_exponential.state == compiled.snapshot
    @test invalid_exponential.effects.stats_f.count == 0

    invalid_normals = copy(replay.normals)
    invalid_normal_replay = ReactiveKernels.OrderedRNGReplay(
        invalid_normals, copy(replay.uniforms), copy(replay.exponentials),
        (:normal, :exponential))
    invalid_normals[1, 1] = NaN
    invalid_normal = compiled.transition(
        compiled.snapshot, invalid_normal_replay)
    @test invalid_normal.control_overflow
    @test invalid_normal.arguments[1].overflow
    @test invalid_normal.state == compiled.snapshot
    @test invalid_normal.effects.stats_f.count == 0

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

    # Functionalization owns an immutable snapshot of structured compiler
    # metadata. Mutating the source transition afterward cannot alter its
    # compiled layout/reference contract.
    before_metadata_mutation = compiled.transition(
        compiled.snapshot, compiled.replay)
    frozen_initial = copy(compiled.transition.ports.init.transition.initial.pos)
    compiled.endpoint.initial.pos[1] += 17
    @test compiled.transition.ports.init.transition.initial.pos == frozen_initial
    @test compiled.transition.ports.init.transition !== compiled.endpoint
    after_metadata_mutation = compiled.transition(
        compiled.snapshot, compiled.replay)
    @test after_metadata_mutation.state == before_metadata_mutation.state
    before_replay = only(before_metadata_mutation.arguments)
    after_replay = only(after_metadata_mutation.arguments)
    @test all(name -> getfield(after_replay, name) ==
                      getfield(before_replay, name),
              fieldnames(typeof(before_replay)))
    @test after_metadata_mutation.effects == before_metadata_mutation.effects
    @test after_metadata_mutation.result == before_metadata_mutation.result
    @test after_metadata_mutation.returned ==
          before_metadata_mutation.returned
    @test after_metadata_mutation.control_overflow ==
          before_metadata_mutation.control_overflow
end
