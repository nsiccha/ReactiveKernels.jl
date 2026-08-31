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

function _rhmc_reactant_sequential_effect_program(source, port)
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.sequential_effect_contract,
        bindings, source, 0)
    state = kernel(source, 0)
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, snapshot=ReactiveKernels.stateful_snapshot(state), transition)
end

function _rhmc_reactant_authored_nested_storage_program()
    endpoint = _RHMC_FUNCTIONAL_REACTANT.nested_storage_transition()
    initial = ReactiveKernels.initial_transition_state(endpoint)
    port = ReactiveKernels.structured_state_port(endpoint)
    bindings = ReactiveKernels.stateful_compiler_bindings(initial=port)
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.authored_nested_storage_write,
        bindings, initial, 0)
    state = kernel(initial, 0)
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Vector{Float64}})
    (; endpoint, initial, port, state,
       snapshot=ReactiveKernels.stateful_snapshot(state), transition)
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

function _rhmc_reactant_scalar_abs_div_program()
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.scalar_abs_div, 0.0, 0, 0)
    state = kernel(0.0, 0, 0)
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!);
        argument_types=Tuple{Float64,Int,Int})
    (; kernel, snapshot=ReactiveKernels.stateful_snapshot(state), transition)
end

function _rhmc_reactant_structured_replacement_program(compiled, lowering)
    source = _RHMCReactantEffectSource(:structured_replacement)
    point = compiled.snapshot.init
    port = ReactiveKernels.effect_lowering_port(
        source, Tuple{typeof(point)}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            lowering))
    bindings = ReactiveKernels.stateful_compiler_bindings(
        init=compiled.structured, callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.structured_effect,
        bindings, point, source, 0)
    state = ReactiveKernels.stateful_snapshot(kernel(point, source, 0))
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, transition)
end

function _rhmc_reactant_paired_structured_replacement_program(
        compiled, lowering)
    source = _RHMCReactantEffectSource(:paired_structured_replacement)
    point = compiled.snapshot.init
    port = ReactiveKernels.effect_lowering_port(
        source, Tuple{typeof(point),typeof(point)}, Nothing;
        written_arguments=(1, 2), initial_effect_state=nothing,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            lowering))
    bindings = ReactiveKernels.stateful_compiler_bindings(
        init=compiled.structured, fwd=compiled.structured, callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.paired_structured_effect,
        bindings, point, point, source, 0)
    state = ReactiveKernels.stateful_snapshot(
        kernel(point, point, source, 0))
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, transition)
end

function _rhmc_reactant_array_replacement_program(lowering)
    source = _RHMCReactantEffectSource(:array_replacement)
    port = ReactiveKernels.effect_lowering_port(
        source, Tuple{Vector{Float64}}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            lowering))
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.array_effect,
        bindings, [1.0, 2.0], source, 0)
    state = ReactiveKernels.stateful_snapshot(
        kernel([1.0, 2.0], source, 0))
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, transition)
end

function _rhmc_reactant_outer_alias_replacement_program(lowering)
    source = _RHMCReactantEffectSource(:outer_alias_replacement)
    port = ReactiveKernels.effect_lowering_port(
        source, Tuple{Vector{Float64},Vector{Float64}}, Nothing;
        written_arguments=(1, 2), initial_effect_state=nothing,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            lowering))
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.outer_alias_effect,
        bindings, [0.0], source, 0)
    state = ReactiveKernels.stateful_snapshot(
        kernel([0.0], source, 0))
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, transition)
end


function _rhmc_reactant_independent_scalar_replacement_program(
        lowering=_RHMC_FUNCTIONAL_REACTANT.IndependentScalarReplacement())
    source = _RHMCReactantEffectSource(:independent_scalars)
    port = ReactiveKernels.effect_lowering_port(
        source, Tuple{Int,Int}, Nothing;
        written_arguments=(1, 2), initial_effect_state=nothing,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            lowering))
    bindings = ReactiveKernels.stateful_compiler_bindings(callback=port)
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.independent_equal_scalars,
        bindings, 1, 1, source, 0)
    state = ReactiveKernels.stateful_snapshot(kernel(1, 1, source, 0))
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Bool})
    (; state, transition)
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

@testset "exact scalar abs/div executes through Reactant" begin
    program = _rhmc_reactant_scalar_abs_div_program()
    traced_state = _rhmc_hmc_trace(program.snapshot)
    compiled = @compile program.transition(
        traced_state,
        _rhmc_hmc_trace(-3.5),
        _rhmc_hmc_trace(-7),
        _rhmc_hmc_trace(2))
    cases = (
        (-3.5, 7, 2),
        (-3.5, -7, 2),
        (-3.5, 7, -2),
        (-3.5, -7, -2),
    )
    for (input, numerator, denominator) in cases
        result = compiled(
            traced_state,
            _rhmc_hmc_trace(input),
            _rhmc_hmc_trace(numerator),
            _rhmc_hmc_trace(denominator))
        @test Bool(result.returned) && Bool(result.result)
        @test Float64(result.state.abs_value) === abs(input)
        @test Int(result.state.quotient) === div(numerator, denominator)
    end

    for argument_types in (
            Tuple{Int,Int,Int},
            Tuple{ComplexF64,Int,Int},
            Tuple{Vector{Float64},Int,Int},
            Tuple{Float64,Bool,Bool},
            Tuple{Float64,Float64,Float64},
            Tuple{Float64,Int32,Int64},
        )
        @test_throws ReactiveKernels._LLowerReject ReactiveKernels.functionalize_stateful(
            program.kernel, Val(:step!); argument_types)
    end
end

function _rhmc_hmc_host_values(program, result)
    ReactiveKernels.drain_observations!(program.transition, result)
    count = program.stats_source.count
    (
        init_pos=Array(result.state.init.pos),
        init_mom=Array(result.state.init.mom),
        init_ham=Float64(result.state.init.ham),
        fwd_pos=Array(result.state.fwd.pos),
        fwd_mom=Array(result.state.fwd.mom),
        fwd_ham=Float64(result.state.fwd.ham),
        dham=Float64(result.state.dham),
        diverged=Bool(result.state.diverged),
        energy_errors=Tuple(program.stats_source.errors[1:count]),
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

    float_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.float_first, zeros(4), 1.0, 0)
    float_transition = ReactiveKernels.functionalize_stateful(
        float_kernel, Val(:step!);
        max_iterations=4, argument_types=Tuple{Int})
    float_state = _rhmc_hmc_trace(ReactiveKernels.stateful_snapshot(
        float_kernel(zeros(4), 1.0, 0)))
    traced_two = _rhmc_hmc_traced_number(2)
    compiled_float = @compile float_transition(float_state, traced_two)
    float_result = compiled_float(float_state, traced_two)
    @test Bool(float_result.returned) && Bool(float_result.result)
    @test !Bool(float_result.control_overflow)
    @test Array(float_result.state.arr) == [1.0, 1.0, 0.0, 0.0]
    @test Int(float_result.state.count) == 2

    loop_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.loop_probe, 0)
    loop_transition = ReactiveKernels.functionalize_stateful(
        loop_kernel, Val(:step!);
        max_iterations=4, argument_types=Tuple{Int,Int})
    loop_state = _rhmc_hmc_trace(ReactiveKernels.stateful_snapshot(
        loop_kernel(0)))
    traced_min = _rhmc_hmc_traced_number(typemin(Int))
    traced_max = _rhmc_hmc_traced_number(typemax(Int))
    compiled_loop = @compile loop_transition(
        loop_state, traced_min, traced_max)
    overflow = compiled_loop(loop_state, traced_min, traced_max)
    @test Bool(overflow.control_overflow)
    @test !Bool(overflow.returned)
    @test Int(overflow.state.count) == 0

    traced_edge = _rhmc_hmc_traced_number(typemax(Int) - 1)
    edge = compiled_loop(loop_state, traced_edge, traced_max)
    @test !Bool(edge.control_overflow)
    @test Bool(edge.returned) && Bool(edge.result)
    @test Int(edge.state.count) == 2


    small_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.small_loop, Int8(0))
    small_transition = ReactiveKernels.functionalize_stateful(
        small_kernel, Val(:step!);
        max_iterations=4, argument_types=Tuple{Int8,Int8})
    small_state = _rhmc_hmc_trace(ReactiveKernels.stateful_snapshot(
        small_kernel(Int8(0))))
    traced_one8 = _rhmc_hmc_traced_number(Int8(1))
    traced_two8 = _rhmc_hmc_traced_number(Int8(2))
    compiled_small = @compile small_transition(
        small_state, traced_one8, traced_two8)
    small = compiled_small(small_state, traced_one8, traced_two8)
    @test Bool(small.returned) && Bool(small.result)
    @test !Bool(small.control_overflow)
    @test Int8(small.state.count) === Int8(3)

    mixed_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.mixed_loop, 0, Int8(0))
    mixed_transition = ReactiveKernels.functionalize_stateful(
        mixed_kernel, Val(:step!);
        max_iterations=4, argument_types=Tuple{Int8,Int8})
    mixed_state = _rhmc_hmc_trace(ReactiveKernels.stateful_snapshot(
        mixed_kernel(0, Int8(0))))
    compiled_mixed = @compile mixed_transition(
        mixed_state, traced_one8, traced_two8)
    mixed = compiled_mixed(mixed_state, traced_one8, traced_two8)
    @test Bool(mixed.returned) && Bool(mixed.result)
    @test !Bool(mixed.control_overflow)
    @test Int(mixed.state.count) == 2
    @test Int8(mixed.state.last) === Int8(2)

    alias_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.outer_alias, [0.0], 0.0, 0)
    alias_transition = ReactiveKernels.functionalize_stateful(
        alias_kernel, Val(:step!); argument_types=Tuple{Bool})
    alias_state_host = ReactiveKernels.stateful_snapshot(
        alias_kernel([0.0], 0.0, 0))
    alias_state = _rhmc_hmc_trace(alias_state_host)
    alias_take = _rhmc_hmc_traced_number(true)
    compiled_alias = @compile alias_transition(alias_state, alias_take)
    raw_alias_result = compiled_alias(alias_state, alias_take)
    @test raw_alias_result.state.a === raw_alias_result.state.b
    @test Array(raw_alias_result.state.a) == [2.0]
    @test Array(raw_alias_result.state.b) == [2.0]
    guarded_alias = ReactiveKernels.validated_compiled_transition(
        compiled_alias, alias_transition)
    alias_result = guarded_alias(alias_state, alias_take)
    @test alias_result.state.a === alias_result.state.b
    @test Array(alias_result.state.a) == [2.0]
    @test Array(alias_result.state.b) == [2.0]
    @test Float64(alias_result.state.total) == 0.0
    broken_alias = _rhmc_hmc_trace(merge(
        alias_state_host, (b=copy(alias_state_host.b),)))
    @test_throws ArgumentError begin
        @compile alias_transition(broken_alias, alias_take)
    end
    @test_throws ArgumentError guarded_alias(broken_alias, alias_take)

    backing = [1.0, 2.0]
    wrapped = (metric=Diagonal(backing), diagonal=backing)
    wrapped_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.wrapped_storage_alias, wrapped, 0)
    wrapped_transition = ReactiveKernels.functionalize_stateful(
        wrapped_kernel, Val(:step!); argument_types=Tuple{Bool})
    wrapped_state_host = ReactiveKernels.stateful_snapshot(
        wrapped_kernel(wrapped, 0))
    @test wrapped_state_host.initial.metric.diag ===
          wrapped_state_host.initial.diagonal
    wrapped_state = _rhmc_hmc_trace(wrapped_state_host)
    compiled_wrapped = @compile wrapped_transition(
        wrapped_state, alias_take)
    guarded_wrapped = ReactiveKernels.validated_compiled_transition(
        compiled_wrapped, wrapped_transition)
    wrapped_result = guarded_wrapped(wrapped_state, alias_take)
    @test wrapped_result.state.initial.metric.diag ===
          wrapped_result.state.initial.diagonal
    @test Int(wrapped_result.state.count) == 1
    copied_metric = Diagonal(copy(wrapped_state_host.initial.diagonal))
    broken_wrapped = _rhmc_hmc_trace(merge(
        wrapped_state_host, (initial=merge(
            wrapped_state_host.initial, (metric=copied_metric,)),)))
    @test_throws ArgumentError begin
        @compile wrapped_transition(broken_wrapped, alias_take)
    end
    @test_throws ArgumentError guarded_wrapped(
        broken_wrapped, alias_take)

    nested_transition =
        _RHMC_FUNCTIONAL_REACTANT.nested_storage_transition()
    nested_initial_host = ReactiveKernels.initial_transition_state(
        nested_transition)
    @test nested_initial_host.nested.metric.diag ===
          nested_initial_host.nested.diagonal
    nested_state = _rhmc_hmc_trace(nested_initial_host)
    compiled_nested_transition = @compile nested_transition(nested_state)
    guarded_nested_transition =
        ReactiveKernels.validated_compiled_transition(
            compiled_nested_transition, nested_transition)
    nested_identity = guarded_nested_transition(nested_state)
    @test nested_identity.nested.metric.diag ===
          nested_identity.nested.diagonal
    @test Array(nested_identity.nested.diagonal) == [1.0, 2.0]

    nested_port = ReactiveKernels.structured_state_port(nested_transition)
    nested_copy = let port=nested_port
        state -> ReactiveKernels._sm_structured_copy(port, state)
    end
    compiled_nested_copy = @compile nested_copy(nested_state)
    copied_nested = compiled_nested_copy(nested_state)
    @test copied_nested.nested.metric.diag ===
          copied_nested.nested.diagonal
    @test Array(copied_nested.nested.diagonal) == [1.0, 2.0]

    nested_replacement = _rhmc_hmc_trace([9.0, 9.0])
    nested_set = let port=nested_port
        (state, replacement) -> ReactiveKernels._sm_structured_set(
            port, state, Val((:nested, :diagonal)), replacement)
    end
    compiled_nested_set = @compile nested_set(
        nested_state, nested_replacement)
    changed_nested = compiled_nested_set(
        nested_state, nested_replacement)
    @test changed_nested.nested.metric.diag ===
          changed_nested.nested.diagonal
    @test Array(changed_nested.nested.metric.diag) == [9.0, 9.0]
    @test Array(changed_nested.nested.diagonal) == [9.0, 9.0]

    nested_select = let port=nested_port
        (candidate, prior, active) ->
            ReactiveKernels._sm_structured_predicated_select(
                port, active, candidate, prior)
    end
    compiled_nested_select = @compile nested_select(
        changed_nested, nested_state, alias_take)
    selected_nested = compiled_nested_select(
        changed_nested, nested_state, alias_take)
    @test selected_nested.nested.metric.diag ===
          selected_nested.nested.diagonal
    @test Array(selected_nested.nested.diagonal) == [9.0, 9.0]

    broken_nested_host = merge(nested_initial_host, (nested=merge(
        nested_initial_host.nested, (metric=Diagonal(
            copy(nested_initial_host.nested.diagonal)),)),))
    broken_nested = _rhmc_hmc_trace(broken_nested_host)
    @test_throws ArgumentError begin
        @compile nested_transition(broken_nested)
    end
    @test_throws ArgumentError guarded_nested_transition(broken_nested)
    @test_throws ArgumentError begin
        @compile nested_copy(broken_nested)
    end

    authored_nested =
        _rhmc_reactant_authored_nested_storage_program()
    authored_nested_state = _rhmc_hmc_trace(authored_nested.snapshot)
    authored_replacement = _rhmc_hmc_trace([9.0, 9.0])
    compiled_authored_nested = @compile authored_nested.transition(
        authored_nested_state, authored_replacement)
    raw_authored_nested = compiled_authored_nested(
        authored_nested_state, authored_replacement)
    @test raw_authored_nested.state.initial.nested.metric.diag ===
          raw_authored_nested.state.initial.nested.diagonal
    @test Array(
        raw_authored_nested.state.initial.nested.metric.diag) ==
        [9.0, 9.0]
    @test Array(
        raw_authored_nested.state.initial.nested.diagonal) ==
        [9.0, 9.0]
    guarded_authored_nested = ReactiveKernels.validated_compiled_transition(
        compiled_authored_nested, authored_nested.transition)
    authored_nested_result = guarded_authored_nested(
        authored_nested_state, authored_replacement)
    @test authored_nested_result.state.initial.nested.metric.diag ===
          authored_nested_result.state.initial.nested.diagonal
    @test Array(
        authored_nested_result.state.initial.nested.metric.diag) ==
        [9.0, 9.0]
    @test Array(
        authored_nested_result.state.initial.nested.diagonal) ==
        [9.0, 9.0]
    @test Int(authored_nested_result.state.count) == 1

    result_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.straight_result_contract, 2.0, 3)
    result_state_host = ReactiveKernels.stateful_snapshot(
        result_kernel(2.0, 3))
    result_state = _rhmc_hmc_trace(result_state_host)
    result_scale = _rhmc_hmc_traced_number(2.0)
    named_result = ReactiveKernels.functionalize_stateful(
        result_kernel, Val(:named_result);
        argument_types=Tuple{Float64})
    @test typeof(named_result).parameters[4] === false
    compiled_named = @compile named_result(result_state, result_scale)
    raw_named = compiled_named(result_state, result_scale)
    @test propertynames(raw_named) == (:value, :count)
    @test Float64(raw_named.value) == 4.0
    @test Int(raw_named.count) == 3
    guarded_named = ReactiveKernels.validated_compiled_transition(
        compiled_named, named_result)
    guarded_named_result = guarded_named(result_state, result_scale)
    @test Float64(guarded_named_result.value) == 4.0
    @test Int(guarded_named_result.count) == 3

    nothing_result = ReactiveKernels.functionalize_stateful(
        result_kernel, Val(:nothing_result);
        argument_types=Tuple{Bool})
    @test typeof(nothing_result).parameters[4] === false
    result_take = _rhmc_hmc_traced_number(true)
    compiled_nothing = @compile nothing_result(result_state, result_take)
    @test compiled_nothing(result_state, result_take) === nothing
    guarded_nothing = ReactiveKernels.validated_compiled_transition(
        compiled_nothing, nothing_result)
    @test guarded_nothing(result_state, result_take) === nothing

    straight_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.straight_alias, [0.0])
    straight_transition = ReactiveKernels.functionalize_stateful(
        straight_kernel, Val(:fit!))
    @test typeof(straight_transition).parameters[4] === true
    straight_state_host = ReactiveKernels.stateful_snapshot(
        straight_kernel([0.0]))
    straight_state = _rhmc_hmc_trace(straight_state_host)
    straight_values = _rhmc_hmc_trace([3.0])
    compiled_straight = @compile straight_transition(
        straight_state, straight_values)
    raw_straight = compiled_straight(straight_state, straight_values)
    @test raw_straight.a === raw_straight.b
    @test Array(raw_straight.a) == [3.0]
    @test Array(raw_straight.b) == [3.0]
    guarded_straight = ReactiveKernels.validated_compiled_transition(
        compiled_straight, straight_transition)
    straight_result = guarded_straight(straight_state, straight_values)
    @test straight_result.a === straight_result.b
    @test Array(straight_result.a) == [3.0]
    @test Array(straight_result.b) == [3.0]
    broken_straight = _rhmc_hmc_trace(merge(
        straight_state_host, (b=copy(straight_state_host.b),)))
    @test_throws ArgumentError begin
        @compile straight_transition(broken_straight, straight_values)
    end
    @test_throws ArgumentError guarded_straight(
        broken_straight, straight_values)

    independent_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.independent_arrays_machine,
        [1.0], [2.0], 0)
    independent_state_host = ReactiveKernels.stateful_snapshot(
        independent_kernel([1.0], [2.0], 0))
    @test independent_state_host.a !== independent_state_host.b
    independent_state = _rhmc_hmc_trace(independent_state_host)
    merged_independent = _rhmc_hmc_trace(merge(
        independent_state_host, (b=independent_state_host.a,)))
    independent_machine = ReactiveKernels.functionalize_stateful(
        independent_kernel, Val(:step!);
        argument_types=Tuple{Bool})
    compiled_independent_machine = @compile independent_machine(
        independent_state, alias_take)
    guarded_independent_machine =
        ReactiveKernels.validated_compiled_transition(
            compiled_independent_machine, independent_machine)
    machine_result = guarded_independent_machine(
        independent_state, alias_take)
    @test machine_result.state.a !== machine_result.state.b
    @test Array(machine_result.state.a) == [2.0]
    @test Array(machine_result.state.b) == [2.0]
    @test_throws ArgumentError begin
        @compile independent_machine(
            merged_independent, alias_take)
    end
    @test_throws ArgumentError guarded_independent_machine(
        merged_independent, alias_take)

    independent_straight_kernel = ReactiveKernels.compile_stateful(
        _RHMC_FUNCTIONAL_REACTANT.independent_arrays_straight,
        [1.0], [2.0], 0)
    independent_straight_state_host = ReactiveKernels.stateful_snapshot(
        independent_straight_kernel([1.0], [2.0], 0))
    independent_straight_state =
        _rhmc_hmc_trace(independent_straight_state_host)
    merged_straight = _rhmc_hmc_trace(merge(
        independent_straight_state_host,
        (b=independent_straight_state_host.a,)))
    independent_values = _rhmc_hmc_trace([3.0])
    independent_straight = ReactiveKernels.functionalize_stateful(
        independent_straight_kernel, Val(:fit!))
    compiled_independent_straight = @compile independent_straight(
        independent_straight_state, independent_values)
    guarded_independent_straight =
        ReactiveKernels.validated_compiled_transition(
            compiled_independent_straight, independent_straight)
    independent_result = guarded_independent_straight(
        independent_straight_state, independent_values)
    @test independent_result.a !== independent_result.b
    @test Array(independent_result.a) == [5.0]
    @test Array(independent_result.b) == [2.0]
    @test_throws ArgumentError begin
        @compile independent_straight(
            merged_straight, independent_values)
    end
    @test_throws ArgumentError guarded_independent_straight(
        merged_straight, independent_values)
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
    effect_port = ReactiveKernels.effect_lowering_port(
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
    guarded_effect = ReactiveKernels.validated_compiled_transition(
        compiled_effect, effect_runner)
    first_result = guarded_effect(effect_state, effects, take)
    second_result = guarded_effect(
        first_result.state, first_result.effects, take)
    @test Int(first_result.effects.callback) == 2
    @test Int(first_result.state.count) == 1
    @test Int(second_result.effects.callback) == 2
    @test Int(second_result.state.count) == 2

    counterfeit_source = merge(
        effect_program.snapshot,
        (callback=_RHMCReactantEffectSource(:counterfeit),))
    @test typeof(counterfeit_source) === typeof(effect_program.snapshot)
    counterfeit_state = _rhmc_hmc_trace(counterfeit_source)
    @test_throws ArgumentError begin
        @compile effect_runner(counterfeit_state, effects, take)
    end
    @test_throws ArgumentError guarded_effect(
        counterfeit_state, effects, take)

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

    wrong_effect = ReactiveKernels.effect_lowering_port(
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


    wrong_effect_shape = ReactiveKernels.effect_lowering_port(
        source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=[0.0, 0.0],
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _RHMC_FUNCTIONAL_REACTANT.WrongShapeEffectState()))
    wrong_effect_shape_program = _rhmc_reactant_effect_program(
        source, wrong_effect_shape)
    wrong_effect_shape_state = _rhmc_hmc_trace(
        wrong_effect_shape_program.snapshot)
    wrong_effect_shape_effects = _rhmc_hmc_trace(
        ReactiveKernels.initial_transition_effects(
            wrong_effect_shape_program.transition))
    @test_throws ArgumentError begin
        wrong_effect_shape_runner = ReactiveKernels.transition_with_effects(
            wrong_effect_shape_program.transition)
        @compile wrong_effect_shape_runner(
            wrong_effect_shape_state, wrong_effect_shape_effects, take)
    end

    initial_array_effect = [0]
    array_effect = ReactiveKernels.effect_lowering_port(
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

    effect_backing = [0.0]
    aliased_effect = (left=effect_backing, right=effect_backing)
    alias_effect_port = ReactiveKernels.effect_lowering_port(
        source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=aliased_effect,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _RHMC_FUNCTIONAL_REACTANT.PassThroughEffectState()))
    alias_effect_program = _rhmc_reactant_effect_program(
        source, alias_effect_port)
    alias_effect_state = _rhmc_hmc_trace(alias_effect_program.snapshot)
    alias_effects_host = ReactiveKernels.initial_transition_effects(
        alias_effect_program.transition)
    @test alias_effects_host.callback.left ===
          alias_effects_host.callback.right
    alias_effects = _rhmc_hmc_trace(alias_effects_host)
    alias_effect_runner = ReactiveKernels.transition_with_effects(
        alias_effect_program.transition)
    compiled_alias_effect = @compile alias_effect_runner(
        alias_effect_state, alias_effects, take)
    guarded_alias_effect = ReactiveKernels.validated_compiled_transition(
        compiled_alias_effect, alias_effect_runner)
    alias_effect_result = guarded_alias_effect(
        alias_effect_state, alias_effects, take)
    @test alias_effect_result.effects.callback.left ===
          alias_effect_result.effects.callback.right
    @test Int(alias_effect_result.state.count) == 1

    broken_effects_host = (callback=(
        left=alias_effects_host.callback.left,
        right=copy(alias_effects_host.callback.right)),)
    broken_effects = _rhmc_hmc_trace(broken_effects_host)
    @test_throws ArgumentError begin
        @compile alias_effect_runner(
            alias_effect_state, broken_effects, take)
    end
    @test_throws ArgumentError guarded_alias_effect(
        alias_effect_state, broken_effects, take)

    broken_alias_port = ReactiveKernels.effect_lowering_port(
        source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=aliased_effect,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _RHMC_FUNCTIONAL_REACTANT.BrokenAliasEffectState()))
    broken_alias_program = _rhmc_reactant_effect_program(
        source, broken_alias_port)
    broken_alias_state = _rhmc_hmc_trace(broken_alias_program.snapshot)
    broken_alias_effects = _rhmc_hmc_trace(
        ReactiveKernels.initial_transition_effects(
            broken_alias_program.transition))
    @test_throws ArgumentError begin
        broken_alias_runner = ReactiveKernels.transition_with_effects(
            broken_alias_program.transition)
        @compile broken_alias_runner(
            broken_alias_state, broken_alias_effects, take)
    end

    distinct_effect = (left=[0.0], right=[0.0])
    merged_alias_port = ReactiveKernels.effect_lowering_port(
        source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=distinct_effect,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _RHMC_FUNCTIONAL_REACTANT.MergedAliasEffectState()))
    merged_alias_program = _rhmc_reactant_effect_program(
        source, merged_alias_port)
    merged_alias_state = _rhmc_hmc_trace(merged_alias_program.snapshot)
    merged_alias_effects = _rhmc_hmc_trace(
        ReactiveKernels.initial_transition_effects(
            merged_alias_program.transition))
    @test_throws ArgumentError begin
        merged_alias_runner = ReactiveKernels.transition_with_effects(
            merged_alias_program.transition)
        @compile merged_alias_runner(
            merged_alias_state, merged_alias_effects, take)
    end

    config_backing = [2]
    alias_config = _RHMC_FUNCTIONAL_REACTANT.AliasConfig(
        config_backing, config_backing)
    alias_config_port = ReactiveKernels.effect_lowering_port(
        source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=0,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _RHMC_FUNCTIONAL_REACTANT.AliasConfiguredEffect(alias_config)))
    alias_config_program = _rhmc_reactant_effect_program(
        source, alias_config_port)
    config_backing[1] = 99
    alias_config_state = _rhmc_hmc_trace(alias_config_program.snapshot)
    alias_config_effects = _rhmc_hmc_trace(
        ReactiveKernels.initial_transition_effects(
            alias_config_program.transition))
    alias_config_runner = ReactiveKernels.transition_with_effects(
        alias_config_program.transition)
    compiled_alias_config = @compile alias_config_runner(
        alias_config_state, alias_config_effects, take)
    alias_config_result = compiled_alias_config(
        alias_config_state, alias_config_effects, take)
    @test Int(alias_config_result.effects.callback) == 2
    @test Int(alias_config_result.state.count) == 1

    sequential_backing = [0.0]
    sequential_effect = (left=sequential_backing, right=sequential_backing)
    sequential_port = ReactiveKernels.effect_lowering_port(
        source, Tuple{ReactiveKernels.StatefulStateValue}, Nothing;
        initial_effect_state=sequential_effect,
        functional_lowering=ReactiveKernels.total_functional_lowering(
            _RHMC_FUNCTIONAL_REACTANT.AliasSensitiveSequentialEffect()))
    sequential_program = _rhmc_reactant_sequential_effect_program(
        source, sequential_port)
    sequential_state = _rhmc_hmc_trace(sequential_program.snapshot)
    sequential_effects = _rhmc_hmc_trace(
        ReactiveKernels.initial_transition_effects(
            sequential_program.transition))
    sequential_runner = ReactiveKernels.transition_with_effects(
        sequential_program.transition)
    compiled_sequential = @compile sequential_runner(
        sequential_state, sequential_effects, take)
    guarded_sequential = ReactiveKernels.validated_compiled_transition(
        compiled_sequential, sequential_runner)
    sequential_result = guarded_sequential(
        sequential_state, sequential_effects, take)
    @test sequential_result.effects.callback.left ===
          sequential_result.effects.callback.right
    @test Array(sequential_result.effects.callback.left) == [2.0]
    @test Int(sequential_result.state.count) == 1
    inactive_sequential = guarded_sequential(
        sequential_state, sequential_effects, skip)
    @test inactive_sequential.effects.callback.left ===
          inactive_sequential.effects.callback.right
    @test Array(inactive_sequential.effects.callback.left) == [0.0]
    @test Int(inactive_sequential.state.count) == 0
end

@testset "fixed-step HMC compiles through generic Reactant lowering" begin
    float_atol = 128eps(Float64)
    take = _rhmc_hmc_traced_number(true)
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
        if case["name"] == "accepted"
            frozen_initial = copy(
                program.transition.ports.init.transition.initial.pos)
            program.endpoint.initial.pos[1] += 17
            @test program.transition.ports.init.transition.initial.pos ==
                  frozen_initial
            @test program.transition.ports.init.transition !== program.endpoint
        end
        state = _rhmc_hmc_trace(program.snapshot)
        replay = _rhmc_hmc_traced_replay(program.replay)
        compiled = @compile program.transition(state, replay)
        compiled_result = if case["name"] == "accepted"
            raw_result = compiled(state, replay)
            @test raw_result.state.init.dpot ===
                  raw_result.state.init.dham_dpos
            @test raw_result.state.fwd.dpot ===
                  raw_result.state.fwd.dham_dpos
            @test raw_result.state.stats_f !== program.stats_source
            guarded = ReactiveKernels.validated_compiled_transition(
                compiled, program.transition)
            result = guarded(state, replay)
            @test result.state.init.pos !== result.state.fwd.pos
            @test result.state.init.mom !== result.state.fwd.mom
            @test result.state.init.dham_dmom !==
                  result.state.fwd.dham_dmom
            @test result.state.stats_f === program.stats_source
            @test result.state.init.pot_f ===
                  program.snapshot.init.pot_f
            result
        else
            compiled(state, replay)
        end
        actual = _rhmc_hmc_host_values(program, compiled_result)

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
            guarded = ReactiveKernels.validated_compiled_transition(
                compiled, program.transition)
            counterfeit_init = merge(program.snapshot.init, (
                pos=program.snapshot.fwd.pos,
                mom=program.snapshot.fwd.mom,
                dham_dmom=program.snapshot.fwd.dham_dmom,
            ))
            counterfeit = _rhmc_hmc_trace(merge(
                program.snapshot, (init=counterfeit_init,)))
            @test_throws ArgumentError begin
                @compile program.transition(counterfeit, replay)
            end
            @test_throws ArgumentError guarded(counterfeit, replay)

            endpoint = program.endpoint
            endpoint_state_host =
                ReactiveKernels.initial_transition_state(endpoint)
            @test endpoint_state_host.pos !== endpoint_state_host.mom
            endpoint_state = _rhmc_hmc_trace(endpoint_state_host)
            compiled_endpoint = @compile endpoint(endpoint_state)
            guarded_endpoint =
                ReactiveKernels.validated_compiled_transition(
                    compiled_endpoint, endpoint)
            endpoint_result = guarded_endpoint(endpoint_state)
            @test endpoint_result.pos !== endpoint_result.mom
            @test endpoint_result.pot_f === endpoint.initial.pot_f
            @test endpoint_result.grad_f === endpoint.initial.grad_f
            merged_endpoint = _rhmc_hmc_trace(merge(
                endpoint_state_host, (mom=endpoint_state_host.pos,)))
            @test_throws ArgumentError begin
                @compile endpoint(merged_endpoint)
            end
            @test_throws ArgumentError guarded_endpoint(merged_endpoint)

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

            broken_replacement =
                _rhmc_reactant_structured_replacement_program(
                    program, _RHMC_FUNCTIONAL_REACTANT.
                        StructuredReplacement{:alias}(
                            copy(program.snapshot.init.dham_dpos)))
            broken_replacement_state =
                _rhmc_hmc_trace(broken_replacement.state)
            @test_throws ArgumentError begin
                @compile broken_replacement.transition(
                    broken_replacement_state, take)
            end

            wrong_structured_shape =
                _rhmc_reactant_structured_replacement_program(
                    program, _RHMC_FUNCTIONAL_REACTANT.
                        StructuredReplacement{:shape}(
                            [first(program.snapshot.init.pos)]))
            wrong_structured_state =
                _rhmc_hmc_trace(wrong_structured_shape.state)
            @test_throws ArgumentError begin
                @compile wrong_structured_shape.transition(
                    wrong_structured_state, take)
            end

            broken_nested_alias =
                _rhmc_reactant_paired_structured_replacement_program(
                    program, _RHMC_FUNCTIONAL_REACTANT.
                        PairedStructuredReplacement{
                            :broken_required_alias}())
            broken_nested_state = _rhmc_hmc_trace(
                broken_nested_alias.state)
            @test_throws ArgumentError begin
                @compile broken_nested_alias.transition(
                    broken_nested_state, take)
            end

            shared_candidate =
                _rhmc_reactant_paired_structured_replacement_program(
                    program, _RHMC_FUNCTIONAL_REACTANT.
                        PairedStructuredReplacement{
                            :cross_canon_share}())
            shared_state = _rhmc_hmc_trace(shared_candidate.state)
            compiled_shared = @compile shared_candidate.transition(
                shared_state, take)
            shared_result = compiled_shared(shared_state, take)
            @test shared_result.state.init.pos !==
                  shared_result.state.fwd.pos
            @test shared_result.state.init.dpot ===
                  shared_result.state.init.dham_dpos
            @test shared_result.state.fwd.dpot ===
                  shared_result.state.fwd.dham_dpos

            wrong_array_shape =
                _rhmc_reactant_array_replacement_program(
                    _RHMC_FUNCTIONAL_REACTANT.ArrayReplacement([1.0]))
            wrong_array_state = _rhmc_hmc_trace(wrong_array_shape.state)
            @test_throws ArgumentError begin
                @compile wrong_array_shape.transition(
                    wrong_array_state, take)
            end

            wrong_outer_alias =
                _rhmc_reactant_outer_alias_replacement_program(
                    _RHMC_FUNCTIONAL_REACTANT.
                        BrokenOuterAliasReplacement())
            @test wrong_outer_alias.state.a === wrong_outer_alias.state.b
            wrong_outer_alias_state =
                _rhmc_hmc_trace(wrong_outer_alias.state)
            @test_throws ArgumentError begin
                @compile wrong_outer_alias.transition(
                    wrong_outer_alias_state, take)
            end


            independent_scalars =
                _rhmc_reactant_independent_scalar_replacement_program()
            scalar_state = _rhmc_hmc_trace(independent_scalars.state)
            compiled_scalars = @compile independent_scalars.transition(
                scalar_state, take)
            scalar_result = compiled_scalars(scalar_state, take)
            @test scalar_result.state.left == 2
            @test scalar_result.state.right == 3
            @test scalar_result.state.count == 1

            equal_scalars =
                _rhmc_reactant_independent_scalar_replacement_program(
                    _RHMC_FUNCTIONAL_REACTANT.EqualScalarReplacement())
            equal_state = _rhmc_hmc_trace(equal_scalars.state)
            compiled_equal = @compile equal_scalars.transition(
                equal_state, take)
            equal_result = compiled_equal(equal_state, take)
            @test equal_result.state.left == 2
            @test equal_result.state.right == 2
            @test equal_result.state.count == 1

            exhausted_host = ReactiveKernels._sm_ordered_rng_with_cursors(
                program.replay, program.replay.normal_index,
                program.replay.uniform_index,
                length(program.replay.exponentials) + 1,
                program.replay.event_index, false)
            exhausted = _rhmc_hmc_traced_replay(exhausted_host)
            overflow = compiled(state, exhausted)
            @test Bool(overflow.control_overflow)
            @test Bool(overflow.arguments[1].overflow)
            @test Int(overflow.outbox.stats_f.count) == 0
            @test Array(overflow.state.init.pos) ==
                  Array(state.init.pos)
            @test Array(overflow.state.fwd.pos) ==
                  Array(state.fwd.pos)

            invalid_exponentials = copy(program.replay.exponentials)
            invalid_exponential_host = ReactiveKernels.OrderedRNGReplay(
                copy(program.replay.normals), copy(program.replay.uniforms),
                invalid_exponentials, (:normal, :exponential))
            invalid_exponentials[1] = -1.0
            invalid_exponential = compiled(
                state, _rhmc_hmc_traced_replay(invalid_exponential_host))
            @test Bool(invalid_exponential.control_overflow)
            @test Bool(invalid_exponential.arguments[1].overflow)
            @test Int(invalid_exponential.outbox.stats_f.count) == 0
            @test Array(invalid_exponential.state.init.pos) ==
                  Array(state.init.pos)
            @test Array(invalid_exponential.state.fwd.pos) ==
                  Array(state.fwd.pos)

            invalid_normals = copy(program.replay.normals)
            invalid_normal_host = ReactiveKernels.OrderedRNGReplay(
                invalid_normals, copy(program.replay.uniforms),
                copy(program.replay.exponentials),
                (:normal, :exponential))
            invalid_normals[1, 1] = NaN
            invalid_normal = compiled(
                state, _rhmc_hmc_traced_replay(invalid_normal_host))
            @test Bool(invalid_normal.control_overflow)
            @test Bool(invalid_normal.arguments[1].overflow)
            @test Int(invalid_normal.outbox.stats_f.count) == 0
            @test Array(invalid_normal.state.init.pos) ==
                  Array(state.init.pos)
            @test Array(invalid_normal.state.fwd.pos) ==
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
