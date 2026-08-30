using ReactiveKernels
using Reactant
using LinearAlgebra
using Test
import Reactant: @compile
import TOML

if !isdefined(@__MODULE__, :ReactiveHMCHMCCompilerSupport)
    include(joinpath(@__DIR__, "fixtures",
                     "reactivehmc_hmc_compiler_support.jl"))
end
const _RHMC_HMC_REACTANT = ReactiveHMCHMCCompilerSupport

if !isdefined(@__MODULE__, :StatefulFunctionalContractsFixture)
    include(joinpath(@__DIR__, "fixtures",
                     "stateful_functional_contracts.jl"))
end
const _RHMC_FUNCTIONAL_REACTANT = StatefulFunctionalContractsFixture

struct _RHMCReactantEffectSource
    token::Symbol
end
(::_RHMCReactantEffectSource)(state) = nothing

struct _RHMCReactantConfiguredEffect{T}
    increment::T
end
(lowering::_RHMCReactantConfiguredEffect)(effect, state) = (
    arguments=(state,), result=nothing,
    effect_state=effect + only(lowering.increment))

struct _RHMCReactantArrayEffect end
(::_RHMCReactantArrayEffect)(effect, state) = (
    arguments=(state,), result=nothing, effect_state=effect .+ 1)

struct _RHMCReactantCallable{F}
    f::F
    token::Symbol
end
(callable::_RHMCReactantCallable)(arguments...) = callable.f(arguments...)

function _rhmc_reactant_effect_program(source, port)
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.effect_contract,
        bindings, source, 0)
    state = kernel(source, 0)
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, snapshot=ReactiveKernels.stateful_snapshot(state), transition)
end

function _rhmc_reactant_pure_program(source, port)
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.pure_contract,
        bindings, source, 0.0, 0)
    state = kernel(source, 0.0, 0)
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Float64,Bool})
    (; state, snapshot=ReactiveKernels.stateful_snapshot(state), transition)
end

function _rhmc_hmc_trace(value)
    _rhmc_hmc_trace(value, IdDict{Any,Any}())
end
function _rhmc_hmc_trace(value::AbstractArray, seen)
    get!(seen, value) do
        Reactant.to_rarray(value)
    end
end
_rhmc_hmc_trace(value::Number, seen) =
    Reactant.to_rarray(value; track_numbers=true)
_rhmc_hmc_trace(value::Diagonal, seen) =
    Diagonal(_rhmc_hmc_trace(value.diag, seen))
_rhmc_hmc_trace(value::LinearAlgebra.Cholesky, seen) =
    LinearAlgebra.Cholesky(
        _rhmc_hmc_trace(value.factors, seen), value.uplo, value.info)
_rhmc_hmc_trace(value::NamedTuple, seen) =
    map(child -> _rhmc_hmc_trace(child, seen), value)
_rhmc_hmc_trace(value::Tuple, seen) =
    map(child -> _rhmc_hmc_trace(child, seen), value)
_rhmc_hmc_trace(value, seen) = value

function _rhmc_hmc_traced_replay(replay)
    ReactiveKernels._sm_ordered_rng_reconstruct(
        _rhmc_hmc_trace(replay.normals),
        _rhmc_hmc_trace(replay.uniforms),
        _rhmc_hmc_trace(replay.exponentials),
        _rhmc_hmc_trace(replay.event_tokens),
        Reactant.to_rarray(replay.normal_index; track_numbers=true),
        Reactant.to_rarray(replay.uniform_index; track_numbers=true),
        Reactant.to_rarray(replay.exponential_index; track_numbers=true),
        Reactant.to_rarray(replay.event_index; track_numbers=true),
        Reactant.to_rarray(replay.overflow; track_numbers=true))
end

_rhmc_hmc_traced_number(value::Number) = _rhmc_hmc_trace(value)

_rhmc_float_from_bits(::Type{T}, bits::AbstractString) where {T<:AbstractFloat} =
    reinterpret(T, parse(T === Float32 ? UInt32 : UInt64, bits; base=2))

function _rhmc_ulp_key(value::T) where {T<:AbstractFloat}
    U = T === Float32 ? UInt32 : UInt64
    bits = reinterpret(U, value)
    sign = one(U) << (8sizeof(U) - 1)
    (bits & sign) == zero(U) ? bits | sign : ~bits
end

_rhmc_ulp_distance(actual::T, expected::T) where {T<:AbstractFloat} =
    abs(Int128(_rhmc_ulp_key(actual)) - Int128(_rhmc_ulp_key(expected)))

function _rhmc_hmc_host_values(result)
    count = Int(result.effects.stats_f.count)
    (
        init_pos=Array(result.state.init.pos),
        init_mom=Array(result.state.init.mom),
        init_ham=Float64(result.state.init.ham),
        fwd_pos=Array(result.state.fwd.pos),
        fwd_mom=Array(result.state.fwd.mom),
        fwd_ham=Float64(result.state.fwd.ham),
        dham=Float64(result.state.dham),
        diverged=Bool(result.state.diverged),
        energy_errors=Tuple(Float64(value)
                            for value in result.effects.stats_f.errors[1:count]),
        normal_calls=Int(result.arguments[1].normal_index) - 1,
        exponential_calls=Int(result.arguments[1].exponential_index) - 1,
        rng_event_calls=Int(result.arguments[1].event_index) - 1,
        rng_overflow=Bool(result.arguments[1].overflow),
        control_overflow=Bool(result.control_overflow),
    )
end


@testset "generic state/control contracts survive Reactant tracing" begin
    dormant_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.dormant, -1.0, 0)
    dormant_native = dormant_kernel(-1.0, 0)
    dormant = ReactiveKernels.functionalize_stateful(
        dormant_kernel, Val(:step!); argument_types=Tuple{Bool})
    dormant_state = _rhmc_hmc_trace(
        ReactiveKernels.stateful_snapshot(dormant_native))
    take = _rhmc_hmc_traced_number(true)
    compiled_dormant = @compile dormant(dormant_state, take)
    dormant_result = compiled_dormant(dormant_state, take)
    @test Bool(dormant_result.returned)
    @test Bool(dormant_result.result)
    @test Float64(dormant_result.state.value) == -1.0
    @test Int(dormant_result.state.count) == 0
    @test !Bool(dormant_result.control_overflow)

    drift_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.drift, 4.0, 0)
    drift_native = drift_kernel(4.0, 0)
    drift = ReactiveKernels.functionalize_stateful(
        drift_kernel, Val(:step!); argument_types=Tuple{Bool})
    drift_state = _rhmc_hmc_trace(
        ReactiveKernels.stateful_snapshot(drift_native))
    compiled_drift = @compile drift(drift_state, take)
    drift_result = compiled_drift(drift_state, take)
    @test Float64(drift_result.state.value) == 4.0
    @test Int(drift_result.state.count) == 0

    counterfeit_type = (
        value=_rhmc_hmc_traced_number(4),
        count=_rhmc_hmc_traced_number(0))
    @test_throws ArgumentError begin
        @compile drift(counterfeit_type, take)
    end

    array_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.array_drift, [1.0, 2.0], 0)
    array_native = array_kernel([1.0, 2.0], 0)
    array_transition = ReactiveKernels.functionalize_stateful(
        array_kernel, Val(:step!); argument_types=Tuple{Bool})
    array_state = ReactiveKernels.stateful_snapshot(array_native)
    wrong_shape = _rhmc_hmc_trace(merge(
        array_state, (values=[first(array_state.values)],)))
    @test_throws ArgumentError begin
        @compile array_transition(wrong_shape, take)
    end
end

@testset "ordered effects and callable ABIs survive Reactant tracing" begin
    uniform_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.uniform_state, false, 0)
    uniform_state = _rhmc_hmc_trace(ReactiveKernels.stateful_snapshot(
        uniform_kernel(false, 0)))
    uniform_transition = ReactiveKernels.functionalize_stateful(
        uniform_kernel, Val(:step!);
        argument_types=Tuple{ReactiveKernels.OrderedRNGReplay{
            Matrix{Float64},Vector{Bool},Vector{Float64},Vector{UInt8},
            Int,Int,Int,Int,Bool},Bool})
    uniform = ReactiveKernels.OrderedRNGReplay(
        reshape([0.0], 1, 1), Bool[true], Float64[0.0], (:uniform,))
    traced_uniform = _rhmc_hmc_traced_replay(uniform)
    take = _rhmc_hmc_traced_number(true)
    skip = _rhmc_hmc_traced_number(false)
    compiled_uniform = @compile uniform_transition(
        uniform_state, traced_uniform, take)
    active = compiled_uniform(uniform_state, traced_uniform, take)
    @test Bool(active.result)
    @test Bool(active.state.value)
    @test Int(active.arguments[1].uniform_index) == 2
    @test Int(active.arguments[1].event_index) == 2
    @test !Bool(active.arguments[1].overflow)

    inactive = compiled_uniform(uniform_state, traced_uniform, skip)
    @test !Bool(inactive.result)
    @test Int(inactive.arguments[1].uniform_index) == 1
    @test Int(inactive.arguments[1].event_index) == 1
    @test !Bool(inactive.control_overflow)

    exhausted = ReactiveKernels._sm_ordered_rng_with_cursors(
        uniform, 1, 2, 1, 1, false)
    exhausted_result = compiled_uniform(
        uniform_state, _rhmc_hmc_traced_replay(exhausted), take)
    @test Bool(exhausted_result.control_overflow)
    @test Bool(exhausted_result.arguments[1].overflow)
    @test Int(exhausted_result.state.count) == 0

    wrong_order = ReactiveKernels.OrderedRNGReplay(
        reshape([0.0], 1, 1), Bool[true], Float64[0.0], (:normal,))
    mismatch = compiled_uniform(
        uniform_state, _rhmc_hmc_traced_replay(wrong_order), take)
    @test Bool(mismatch.control_overflow)
    @test Bool(mismatch.arguments[1].overflow)
    @test Int(mismatch.arguments[1].event_index) == 1

    source = _RHMCReactantEffectSource(:source)
    configuration = [2]
    effect_port = ReactiveKernels.effect_callable_port(
        source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=0,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _RHMCReactantConfiguredEffect(configuration)))
    effect_program = _rhmc_reactant_effect_program(source, effect_port)
    configuration[1] = 99
    effect_state = _rhmc_hmc_trace(effect_program.snapshot)
    effects = _rhmc_hmc_trace(
        ReactiveKernels.initial_transition_effects(effect_program.transition))
    effect_runner = ReactiveKernels.transition_with_effects(
        effect_program.transition)
    compiled_effect = @compile effect_runner(effect_state, effects, take)
    first_result = compiled_effect(effect_state, effects, take)
    second_result = compiled_effect(
        first_result.state, first_result.effects, take)
    @test Int(first_result.effects.callback) == 2
    @test Int(first_result.state.count) == 1
    @test Int(second_result.effects.callback) == 4
    @test Int(second_result.state.count) == 2

    counterfeit_source = merge(
        effect_program.snapshot,
        (callback=_RHMCReactantEffectSource(:counterfeit),))
    @test typeof(counterfeit_source) === typeof(effect_program.snapshot)
    counterfeit_state = _rhmc_hmc_trace(counterfeit_source)
    @test_throws ArgumentError begin
        @compile effect_runner(counterfeit_state, effects, take)
    end

    wrong_pure = ReactiveKernels.pure_callable_port(
        identity, Tuple{Float64}, Float64;
        functional_lowering=ReactiveKernels.total_functional_lowering(
            value -> 1))
    pure_program = _rhmc_reactant_pure_program(identity, wrong_pure)
    pure_state = _rhmc_hmc_trace(pure_program.snapshot)
    value = _rhmc_hmc_traced_number(2.0)
    @test_throws ArgumentError begin
        @compile pure_program.transition(pure_state, value, take)
    end

    wrong_effect = ReactiveKernels.effect_callable_port(
        source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=0,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            (effect, state) -> (
                arguments=(state,), result=nothing,
                effect_state=one(Float64))))
    wrong_effect_program = _rhmc_reactant_effect_program(
        source, wrong_effect)
    wrong_effect_state = _rhmc_hmc_trace(wrong_effect_program.snapshot)
    wrong_effects = _rhmc_hmc_trace(
        ReactiveKernels.initial_transition_effects(
            wrong_effect_program.transition))
    @test_throws ArgumentError begin
        wrong_effect_runner = ReactiveKernels.transition_with_effects(
            wrong_effect_program.transition)
        @compile wrong_effect_runner(
            wrong_effect_state, wrong_effects, take)
    end

    initial_array_effect = [0]
    array_effect = ReactiveKernels.effect_callable_port(
        source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=initial_array_effect,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _RHMCReactantArrayEffect()))
    array_program = _rhmc_reactant_effect_program(source, array_effect)
    initial_array_effect[1] = 41
    array_effect_state = _rhmc_hmc_trace(array_program.snapshot)
    array_effects = _rhmc_hmc_trace(
        ReactiveKernels.initial_transition_effects(array_program.transition))
    array_effect_runner = ReactiveKernels.transition_with_effects(
        array_program.transition)
    compiled_array_effect = @compile array_effect_runner(
        array_effect_state, array_effects, take)
    array_effect_result = compiled_array_effect(
        array_effect_state, array_effects, take)
    @test Array(array_effect_result.effects.callback) == [1]
    wrong_effect_shape = _rhmc_hmc_trace((callback=[0, 0],))
    @test_throws ArgumentError begin
        @compile array_effect_runner(
            array_effect_state, wrong_effect_shape, take)
    end
end

@testset "fixed-step HMC compiles through generic Reactant lowering" begin
    float_atol = 128eps(Float64)
    receipt = TOML.parsefile(joinpath(
        @__DIR__, "..", "benchmark", "receipts",
        "reactivehmc-hmc-ca9-v1.toml"))
    for case in receipt["cases"]
        program = if case["name"] == "accepted"
            _RHMC_HMC_REACTANT.build_case(
                case;
                potential_f=_RHMCReactantCallable(
                    _RHMC_HMC_REACTANT.potential, :potential),
                gradient_f=_RHMCReactantCallable(
                    _RHMC_HMC_REACTANT.gradient, :gradient))
        else
            _RHMC_HMC_REACTANT.build_case(case)
        end
        state = _rhmc_hmc_trace(program.snapshot)
        replay = _rhmc_hmc_traced_replay(program.replay)
        compiled = @compile program.transition(state, replay)
        actual = _rhmc_hmc_host_values(compiled(state, replay))

        @test actual.init_pos ≈ case["init_pos"] atol=float_atol rtol=0
        @test actual.init_mom ≈ case["init_mom"] atol=float_atol rtol=0
        @test actual.init_ham ≈ case["init_ham"] atol=float_atol rtol=0
        @test actual.fwd_pos ≈ case["fwd_pos"] atol=float_atol rtol=0
        @test actual.fwd_mom ≈ case["fwd_mom"] atol=float_atol rtol=0
        @test actual.fwd_ham ≈ case["fwd_ham"] atol=float_atol rtol=0
        @test actual.dham ≈ case["dham"] atol=float_atol rtol=0
        @test actual.diverged == case["diverged"]
        @test collect(actual.energy_errors) ≈ case["energy_errors"] atol=float_atol rtol=0
        @test actual.normal_calls == case["normal_calls"]
        @test actual.exponential_calls == case["exponential_calls"]
        @test actual.rng_event_calls == length(case["rng_events"])
        @test !actual.rng_overflow
        @test !actual.control_overflow

        if case["name"] == "accepted"
            broken_alias_init = merge(
                program.snapshot.init,
                (dham_dpos=copy(program.snapshot.init.dham_dpos),))
            broken_alias = _rhmc_hmc_trace(merge(
                program.snapshot, (init=broken_alias_init,)))
            @test_throws ArgumentError begin
                @compile program.transition(broken_alias, replay)
            end

            wrong_shape_init = merge(
                program.snapshot.init,
                (pos=[first(program.snapshot.init.pos)],))
            wrong_shape = _rhmc_hmc_trace(merge(
                program.snapshot, (init=wrong_shape_init,)))
            @test_throws ArgumentError begin
                @compile program.transition(wrong_shape, replay)
            end

            replacement = _RHMCReactantCallable(
                _RHMC_HMC_REACTANT.potential, :replacement)
            replaced_authority_init = merge(
                program.snapshot.init, (pot_f=replacement,))
            @test typeof(replaced_authority_init) ===
                  typeof(program.snapshot.init)
            replaced_authority = _rhmc_hmc_trace(merge(
                program.snapshot, (init=replaced_authority_init,)))
            @test_throws ArgumentError begin
                @compile program.transition(replaced_authority, replay)
            end

            exhausted_host = ReactiveKernels._sm_ordered_rng_with_cursors(
                program.replay, program.replay.normal_index,
                program.replay.uniform_index,
                length(program.replay.exponentials) + 1,
                program.replay.event_index, false)
            exhausted = _rhmc_hmc_traced_replay(exhausted_host)
            overflow = compiled(state, exhausted)
            @test Bool(overflow.control_overflow)
            @test Bool(overflow.arguments[1].overflow)
            @test Int(overflow.effects.stats_f.count) == 0
            @test Array(overflow.state.init.pos) ==
                  Array(state.init.pos)
            @test Array(overflow.state.fwd.pos) ==
                  Array(state.fwd.pos)
        end
    end


    mixed = only(receipt["mixed_precision_cases"])
    program = _RHMC_HMC_REACTANT.build_case(
        mixed; numeric_type=Float32)
    @test eltype(program.replay.normals) === Float32
    @test eltype(program.replay.exponentials) === Float64
    state = _rhmc_hmc_trace(program.snapshot)
    replay = _rhmc_hmc_traced_replay(program.replay)
    compiled = @compile program.transition(state, replay)
    compiled_result = compiled(state, replay)
    @test eltype(compiled_result.state.init.pos) === Float32
    @test eltype(compiled_result.state.init.mom) === Float32
    @test Reactant.unwrapped_eltype(typeof(compiled_result.state.init.ham)) ===
          Float64
    @test eltype(compiled_result.state.fwd.pos) === Float32
    @test eltype(compiled_result.state.fwd.mom) === Float32
    @test Reactant.unwrapped_eltype(typeof(compiled_result.state.fwd.ham)) ===
          Float64
    @test Reactant.unwrapped_eltype(typeof(compiled_result.state.dham)) ===
          Float64
    actual = _rhmc_hmc_host_values(compiled_result)
    @test eltype(collect(actual.energy_errors)) === Float64
    @test bitstring.(actual.init_pos) == mixed["init_pos_bits"]
    @test bitstring.(actual.fwd_pos) == mixed["fwd_pos_bits"]

    source_init_mom = _rhmc_float_from_bits.(
        Ref(Float32), mixed["init_mom_bits"])
    source_fwd_mom = _rhmc_float_from_bits.(
        Ref(Float32), mixed["fwd_mom_bits"])
    source_init_ham = _rhmc_float_from_bits(
        Float64, only(mixed["init_ham_bits"]))
    source_fwd_ham = _rhmc_float_from_bits(
        Float64, only(mixed["fwd_ham_bits"]))
    source_energy_errors = _rhmc_float_from_bits.(
        Ref(Float64), mixed["energy_error_bits"])
    source_dham = _rhmc_float_from_bits(
        Float64, only(mixed["dham_bits"]))
    @test _rhmc_ulp_distance.(actual.init_mom, source_init_mom) == [1, 0]
    @test _rhmc_ulp_distance.(actual.fwd_mom, source_fwd_mom) == [1, 0]
    @test _rhmc_ulp_distance(actual.init_ham, source_init_ham) == 67_108_864
    @test _rhmc_ulp_distance(actual.fwd_ham, source_fwd_ham) == 67_108_864
    @test _rhmc_ulp_distance.(
        collect(actual.energy_errors), source_energy_errors) ==
        [68_719_476_736, 0, 8_589_934_592]
    @test _rhmc_ulp_distance(actual.dham, source_dham) == 8_589_934_592
    @test actual.diverged == mixed["diverged"]
    @test actual.dham - mixed["min_dham"] > 900
    @test actual.dham + only(mixed["exponential_draws"]) > 0.49
    @test actual.normal_calls == mixed["normal_calls"]
    @test actual.exponential_calls == mixed["exponential_calls"]
    @test actual.rng_event_calls == length(mixed["rng_events"])
    @test !actual.rng_overflow
    @test !actual.control_overflow

    normal_bindings = ReactiveKernels.stateful_compiler_bindings(
        init=program.structured)
    initial_point = ReactiveKernels.initial_transition_state(program.endpoint)
    normal_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.normal_replace,
        normal_bindings, initial_point, 0)
    normal_state_host = ReactiveKernels.stateful_snapshot(
        normal_kernel(initial_point, 0))
    normal_transition = ReactiveKernels.functionalize_stateful(
        normal_kernel, Val(:step!);
        argument_types=Tuple{typeof(program.replay),Bool})
    normal_state = _rhmc_hmc_trace(normal_state_host)
    normal_take = _rhmc_hmc_traced_number(true)
    normal_runner = @compile normal_transition(
        normal_state, replay, normal_take)
    normal_result = normal_runner(normal_state, replay, normal_take)
    @test eltype(Array(normal_result.state.init.mom)) === Float32
    @test bitstring.(Array(normal_result.state.init.mom)) ==
          mixed["normal_draw_bits"]
    @test Int(normal_result.arguments[1].normal_index) == 2
    @test Int(normal_result.arguments[1].event_index) == 2
    @test Int(normal_result.state.count) == 1
    @test !Bool(normal_result.arguments[1].overflow)

    widened = ReactiveKernels.OrderedRNGReplay(
        Float64.(program.replay.normals), program.replay.uniforms,
        program.replay.exponentials, (:normal, :exponential))
    @test_throws ArgumentError begin
        @compile program.transition(
            state, _rhmc_hmc_traced_replay(widened))
    end
    @test_throws ArgumentError begin
        @compile normal_transition(
            normal_state, _rhmc_hmc_traced_replay(widened), normal_take)
    end
end
