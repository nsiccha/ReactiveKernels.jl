using ReactiveKernels
using ReactiveKernelsNUTSExamples
using Reactant
using LinearAlgebra
using Random
using Test
import Reactant: @compile
import TOML

include(joinpath(@__DIR__, "fixtures",
                 "reactivehmc_hmc_compiler_support.jl"))
isdefined(@__MODULE__, :MutationProfileBGenericControl) || include(joinpath(
    @__DIR__, "fixtures", "mutation_profile_b_generic_control.jl"))

module _MutationProfileBReactantNUTS
include(joinpath(@__DIR__, "..", "benchmark",
                 "nuts_kernel_authoring_fixture_b.jl"))
end

const _MPBR_HMC = ReactiveHMCHMCCompilerSupport
const _MPBR_NUTS =
    _MutationProfileBReactantNUTS.NUTSBMutationAuthoringFixture
const _MPBR_GENERIC_CONTROL = MutationProfileBGenericControl
const _MPBR_TESTSET = get(ENV, "RK_MPB_REACTANT_TESTSET", "all")
_mpbr_enabled(name) = _MPBR_TESTSET == "all" || _MPBR_TESTSET == name

struct _MPBRCallable{F}
    f::F
end
(callable::_MPBRCallable)(arguments...) = callable.f(arguments...)

struct _MPBRFiniteRead{P}
    port::P
end
function (operation::_MPBRFiniteRead)(values, index)
    raw = ReactiveKernels._sm_finite_structural_pack(
        operation.port, values)
    ReactiveKernels._sm_finite_structural_read(
        operation.port, raw, index).value
end

struct _MPBRRestoreScalar{N}
    node::N
end
function (operation::_MPBRRestoreScalar)(value)
    restored = ReactiveKernels._sm_finite_restore_logical(
        operation.node, value, ())
    (; restored, identical=restored === value)
end

function _mpbr_scalar_bridge_element(seed)
    factors = Float64[seed + 2, seed + 3]
    (;
        scalar=Float64(seed),
        buffer=Float64[seed, seed + 1],
        factorization=LinearAlgebra.Cholesky(
            Diagonal(factors), 'U', 0),
        factor_alias=factors,
    )
end

_mpbr_transfer_census(
        ::ReactiveKernels._SMFiniteScalarNode{Index,T},
        before, after) where {Index,T} = (;
    scalar=before isa Reactant.ConcretePJRTNumber &&
           typeof(after) === T ? 1 : 0,
    array=0,
)
_mpbr_transfer_census(
        ::ReactiveKernels._SMFiniteArrayNode,
        before, after) = (;
    scalar=0,
    array=before === after ? 0 : 1,
)
_mpbr_transfer_census(
        ::ReactiveKernels._SMFiniteStaticNode,
        before, after) = (; scalar=0, array=0)
_mpbr_add_census(left, right) = (;
    scalar=left.scalar + right.scalar,
    array=left.array + right.array,
)
function _mpbr_transfer_census(
        node::ReactiveKernels._SMFiniteNamedTupleNode{Names},
        before, after) where {Names}
    reduce(_mpbr_add_census,
        map(Names, node.children) do name, child
            _mpbr_transfer_census(
                child, getfield(before, name), getfield(after, name))
        end;
        init=(; scalar=0, array=0))
end
function _mpbr_transfer_census(
        node::ReactiveKernels._SMFiniteTupleNode,
        before, after)
    reduce(_mpbr_add_census,
        map(eachindex(node.children), node.children) do index, child
            _mpbr_transfer_census(
                child, getfield(before, index), getfield(after, index))
        end;
        init=(; scalar=0, array=0))
end
_mpbr_transfer_census(
        node::ReactiveKernels._SMFiniteDiagonalNode,
        before, after) = _mpbr_transfer_census(
    node.child, getfield(before, :diag), getfield(after, :diag))
_mpbr_transfer_census(
        node::ReactiveKernels._SMFiniteCholeskyNode,
        before, after) = _mpbr_transfer_census(
    node.child, getfield(before, :factors), getfield(after, :factors))

function _mpbr_trace(value)
    _mpbr_trace(value, IdDict{Any,Any}())
end
function _mpbr_trace(value::AbstractArray, seen)
    get!(seen, value) do
        Reactant.to_rarray(value)
    end
end
_mpbr_trace(value::Number, seen) =
    Reactant.to_rarray(value; track_numbers=true)
_mpbr_trace(value::Diagonal, seen) = Diagonal(_mpbr_trace(value.diag, seen))
_mpbr_trace(value::LinearAlgebra.Cholesky, seen) = LinearAlgebra.Cholesky(
    _mpbr_trace(value.factors, seen), value.uplo, value.info)
_mpbr_trace(value::NamedTuple, seen) = map(child -> _mpbr_trace(child, seen), value)
_mpbr_trace(value::Tuple, seen) = map(child -> _mpbr_trace(child, seen), value)
_mpbr_trace(value, seen) = value

function _mpbr_trace_replay(replay)
    ReactiveKernels._sm_ordered_rng_reconstruct(
        _mpbr_trace(replay.normals),
        _mpbr_trace(replay.uniforms),
        _mpbr_trace(replay.exponentials),
        _mpbr_trace(replay.event_tokens),
        Reactant.to_rarray(replay.normal_index; track_numbers=true),
        Reactant.to_rarray(replay.uniform_index; track_numbers=true),
        Reactant.to_rarray(replay.exponential_index; track_numbers=true),
        Reactant.to_rarray(replay.event_index; track_numbers=true),
        Reactant.to_rarray(replay.overflow; track_numbers=true))
end

function _mpbr_nuts_factory()
    ReactiveKernels._prepare_factory(
        _MPBR_NUTS.euclidean_phasepoint,
        ReactiveKernels.kernel_registration(_MPBR_NUTS.leapfrog!))
end

function _mpbr_nuts_values(pf)
    metric = Float64[2 0; 0 2]
    values = Dict{Int,Any}()
    for slot in ReactiveKernels.kernel_plan_slots(
            ReactiveKernels.kernel_prepared_plan(pf))
        name = String(slot.path[1])
        values[slot.canon] = name == "pot_f" ? (p -> sum(abs2, p)) :
            name == "grad_f" ?
                ((dst, p) -> (dst .= 2 .* p; sum(abs2, p))) :
            name == "metric" ? metric :
            name == "chol_metric" ? cholesky(metric) :
            startswith(name, "##node") ? 0.0 :
            name == "pos" ? [1.0, 2.0] :
            name == "mom" ? [3.0, 4.0] :
            name in ("dpot_dpos", "dham_dpos", "dkin_dmom", "dham_dmom") ?
                [0.0, 0.0] : 0.0
    end
    values
end

function _mpbr_nuts_frame(pf, max_depth::Integer)
    frame = ReactiveKernelsNUTSExamples._construct_nuts_frame(
        pf, _mpbr_nuts_values(pf), max_depth;
        step_f=ReactiveKernels.partial(_MPBR_NUTS.leapfrog!; stepsize=0.1),
        stats_f=_MPBR_NUTS.nuts_stats!, min_dham=-1000.0)
    ReactiveKernels.compile_prepared_initialization(
        pf, typeof(frame.init), typeof(frame.shared))(
            frame.init, frame.shared,
            ReactiveKernels.kernel_prepared_handles(pf))
    ReactiveKernelsNUTSExamples._seed_nuts_children!(frame)
    frame
end

if _mpbr_enabled("hmc")
@testset "mutation profile B — fixed-step HMC Reactant" begin
    receipt = TOML.parsefile(joinpath(
        @__DIR__, "..", "benchmark", "receipts",
        "reactivehmc-hmc-ca9-v1.toml"))
    case = only(filter(item -> item["name"] == "accepted", receipt["cases"]))
    program = _MPBR_HMC.build_case(
        case;
        hmc_spec=_MPBR_HMC.ReactiveHMCHMCBMutationAuthoringFixture.hmc_state,
        potential_f=_MPBRCallable(_MPBR_HMC.potential),
        gradient_f=_MPBRCallable(_MPBR_HMC.gradient))
    state = _mpbr_trace(program.snapshot)
    replay = _mpbr_trace_replay(program.replay)
    compiled = @compile program.transition(state, replay)
    guarded = ReactiveKernels.validated_compiled_transition(
        compiled, program.transition)
    actual = _MPBR_HMC.result_values(program, guarded(state, replay))

    @test actual.init_pos ≈ case["init_pos"] atol=128eps(Float64) rtol=0
    @test actual.init_mom ≈ case["init_mom"] atol=128eps(Float64) rtol=0
    @test actual.fwd_pos ≈ case["fwd_pos"] atol=128eps(Float64) rtol=0
    @test actual.fwd_mom ≈ case["fwd_mom"] atol=128eps(Float64) rtol=0
    @test actual.energy_errors ≈ case["energy_errors"] atol=128eps(Float64) rtol=0
    @test !Bool(actual.control_overflow)
    @test !Bool(actual.rng_overflow)
end
end

if _mpbr_enabled("generic")
@testset "mutation profile B — generic bounded control through Reactant" begin
    backing = [0]
    payload = (values=backing, mirror=backing)
    kernel = ReactiveKernels.compile_stateful(
        _MPBR_GENERIC_CONTROL.structured_counter, payload, 0, 2)
    host_state = ReactiveKernels.stateful_snapshot(
        kernel(payload, 0, 2))
    @test host_state.payload.values === host_state.payload.mirror
    bounds = ReactiveKernels.stateful_control_bounds(
        kernel, Val(:drive!), host_state; argument_types=Tuple{})
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:drive!), bounds)
    state = _mpbr_trace(host_state)
    compiled = @compile transition(state)
    result = compiled(state)

    @test !Bool(result.control_overflow)
    @test result.state.payload.values === result.state.payload.mirror
    @test Array(result.state.payload.values) == [0]
    @test Int(result.state.total) == 3

    guarded = ReactiveKernels.validated_compiled_transition(
        compiled, transition)
    guarded_result = guarded(state)
    @test guarded_result.state.payload.values ===
          guarded_result.state.payload.mirror
    @test Array(guarded_result.state.payload.values) == [0]

    wrong_backing = [0, 0]
    wrong_shape = _mpbr_trace(merge(host_state,
        (payload=(values=wrong_backing, mirror=wrong_backing),)))
    @test_throws ArgumentError begin
        @compile transition(wrong_shape)
    end
    broken_alias = _mpbr_trace(merge(host_state, (payload=(
        values=host_state.payload.values,
        mirror=copy(host_state.payload.mirror)),)))
    @test_throws ArgumentError begin
        @compile transition(broken_alias)
    end
    @test_throws ArgumentError guarded(broken_alias)
end
end

if _mpbr_enabled("generic-rng")
@testset "generic recursive RNG provider remains one traced carry" begin
    replay = ReactiveKernels.OrderedRNGReplay(
        zeros(1, 1), Bool[true, false, true], [0.0],
        fill(:uniform, 3))
    kernel = ReactiveKernels.compile_stateful(
        _MPBR_GENERIC_CONTROL.recursive_rng_counter, 0, 2, false)
    host_state = ReactiveKernels.stateful_snapshot(kernel(0, 2, false))
    bounds = ReactiveKernels.stateful_control_bounds(
        kernel, Val(:drive!), host_state;
        argument_types=Tuple{typeof(replay)})
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:drive!), bounds)
    native_result = transition(host_state, replay)
    native_actual = only(native_result.arguments)

    @test !native_result.control_overflow
    @test !native_actual.overflow
    @test native_result.state.total == 2
    @test (native_actual.normal_index, native_actual.uniform_index,
           native_actual.exponential_index, native_actual.event_index) ==
          (1, 4, 1, 4)
    @test native_actual.uniforms == replay.uniforms
    @test native_actual.event_tokens == replay.event_tokens

    state = _mpbr_trace(host_state)
    traced_replay = _mpbr_trace_replay(replay)
    compiled = @compile transition(state, traced_replay)
    result = compiled(state, traced_replay)
    actual = only(result.arguments)

    @test !Bool(result.control_overflow)
    @test !Bool(actual.overflow)
    @test Int(result.state.total) == 2
    @test Int(actual.normal_index) == 1
    @test Int(actual.uniform_index) == 4
    @test Int(actual.exponential_index) == 1
    @test Int(actual.event_index) == 4
    @test Array(actual.uniforms) == replay.uniforms
    @test Array(actual.event_tokens) == replay.event_tokens
end
end

if _mpbr_enabled("logical-wrapper")
@testset "compiled finite results restore source-logical wrappers" begin
    prototype = [(factorization=LinearAlgebra.Cholesky(
        Diagonal(Float64[index, index + 1]), 'U', 0),)
        for index in 1:2]
    port = ReactiveKernels._sm_finite_structural_contract(prototype)
    traced = [_mpbr_trace(value) for value in prototype]
    operation = _MPBRFiniteRead(port)
    compiled = @compile operation(
        traced, Reactant.to_rarray(1; track_numbers=true))
    backend_values = [compiled(
        traced, Reactant.to_rarray(index; track_numbers=true))
        for index in eachindex(prototype)]

    @test all(value -> !(value.factorization isa LinearAlgebra.Cholesky),
              backend_values)
    restored = ReactiveKernels._sm_restore_reusable_finite_port(
        port, backend_values)
    @test all(value -> value.factorization isa LinearAlgebra.Cholesky,
              restored)
    @test all(value -> value.factorization.factors isa Diagonal, restored)
    @test [Array(value.factorization.factors.diag) for value in restored] ==
          [value.factorization.factors.diag for value in prototype]
    @test ReactiveKernels._sm_finite_validate_elements(port, restored) ===
          restored
end
end

if _mpbr_enabled("scalar-bridge")
@testset "finite concrete scalars restore the reusable host ABI" begin
    scalar_node = ReactiveKernels._SMFiniteScalarNode{1,Float64}()
    concrete_scalar = Reactant.to_rarray(1.25; track_numbers=true)
    @test concrete_scalar isa Reactant.ConcretePJRTNumber{Float64}
    restored_scalar = ReactiveKernels._sm_finite_restore_logical(
        scalar_node, concrete_scalar, ())
    @test restored_scalar === 1.25
    @test typeof(restored_scalar) === Float64

    traced_identity = @compile _MPBRRestoreScalar(scalar_node)(
        concrete_scalar)
    traced_result = traced_identity(concrete_scalar)
    @test traced_result.restored isa Reactant.ConcretePJRTNumber{Float64}
    @test Float64(traced_result.restored) == 1.25
    @test Bool(traced_result.identical)

    concrete_array = Reactant.to_rarray(Float64[1, 2])
    array_node = ReactiveKernels._SMFiniteArrayNode{
        1,Vector{Float64},(2,)}()
    @test ReactiveKernels._sm_finite_restore_logical(
        array_node, concrete_array, ()) === concrete_array

    wrong_scalar = Reactant.to_rarray(Float32(1.25); track_numbers=true)
    @test_throws ArgumentError ReactiveKernels._sm_finite_restore_logical(
        scalar_node, wrong_scalar, ())

    prototype = [_mpbr_scalar_bridge_element(index) for index in 1:2]
    port = ReactiveKernels._sm_finite_structural_contract(prototype)
    concrete = [_mpbr_trace(value) for value in prototype]
    restored = ReactiveKernels._sm_finite_restore_logical_elements(
        port, concrete)
    census = reduce(_mpbr_add_census,
        map(concrete, restored) do before, after
            _mpbr_transfer_census(port.schema, before, after)
        end;
        init=(; scalar=0, array=0))
    @test census == (; scalar=2, array=0)
    @test all(value -> typeof(value.scalar) === Float64, restored)
    @test all(value -> value.buffer isa Reactant.ConcretePJRTArray,
              restored)
    @test all(value -> value.factorization isa LinearAlgebra.Cholesky,
              restored)
    @test all(value -> value.factorization.uplo == 'U' &&
                       value.factorization.info == 0, restored)
    @test all(value -> value.factorization.factors isa Diagonal, restored)
    @test all(value -> value.factorization.factors.diag ===
                       value.factor_alias, restored)

    restored_again = ReactiveKernels._sm_finite_restore_logical_elements(
        port, restored)
    repeat_census = reduce(_mpbr_add_census,
        map(restored, restored_again) do before, after
            _mpbr_transfer_census(port.schema, before, after)
        end;
        init=(; scalar=0, array=0))
    @test repeat_census == (; scalar=0, array=0)
    @test all(map(restored, restored_again) do before, after
        before.buffer === after.buffer &&
            before.factor_alias === after.factor_alias
    end)

    bindings = ReactiveKernels.stateful_compiler_bindings(payload=port)
    kernel = ReactiveKernels.compile_stateful(
        _MPBR_GENERIC_CONTROL.structured_counter,
        bindings, prototype, 0, 1)
    host_state = ReactiveKernels.stateful_snapshot(
        kernel(prototype, 0, 1))
    bounds = ReactiveKernels.stateful_control_bounds(
        kernel, Val(:drive!), host_state; argument_types=Tuple{})
    transition = ReactiveKernels.functionalize_stateful(
        kernel, Val(:drive!), bounds)
    # `track_numbers=true` may make call-1 entry scalars concrete. The reusable
    # fixed point promised here is guarded result 1 -> result 2 onward, not
    # identity between the traced entry representation and the first result.
    state = _mpbr_trace(host_state)
    compiled = @compile transition(state)
    guarded = ReactiveKernels.validated_compiled_transition(
        compiled, transition)
    @test getfield(guarded, :compiled) === compiled

    first = guarded(state)
    first_total = Int(first.state.total)
    first_scalars = [value.scalar for value in first.state.payload]
    @test first_total == 2
    @test first_scalars == [1.0, 2.0]
    @test all(value -> typeof(value.scalar) === Float64,
              first.state.payload)
    @test all(value -> value.buffer isa Reactant.ConcretePJRTArray,
              first.state.payload)
    @test all(value -> value.factorization isa LinearAlgebra.Cholesky,
              first.state.payload)
    @test all(value -> value.factorization.uplo == 'U' &&
                       value.factorization.info == 0,
              first.state.payload)
    @test all(value -> value.factorization.factors.diag ===
                       value.factor_alias, first.state.payload)

    second_timed = @timed guarded(first.state)
    second = second_timed.value
    @info("finite scalar bridge guarded steady-state diagnostic",
          time_seconds=second_timed.time,
          allocated_bytes=second_timed.bytes)

    @test getfield(guarded, :compiled) === compiled
    @test Int(second.state.total) == 4
    @test [value.scalar for value in second.state.payload] == [1.0, 2.0]
    @test all(value -> typeof(value.scalar) === Float64,
              second.state.payload)
    @test all(value -> value.buffer isa Reactant.ConcretePJRTArray,
              second.state.payload)
    @test all(value -> value.factorization isa LinearAlgebra.Cholesky,
              second.state.payload)
    @test all(value -> value.factorization.uplo == 'U' &&
                       value.factorization.info == 0,
              second.state.payload)
    @test all(value -> value.factorization.factors.diag ===
                       value.factor_alias, second.state.payload)

    @test typeof(second.state) === typeof(first.state)
    third = guarded(second.state)
    @test getfield(guarded, :compiled) === compiled
    @test typeof(third.state) === typeof(second.state)
    @test Int(third.state.total) == 6
    @test [value.scalar for value in third.state.payload] == [1.0, 2.0]
    @test all(value -> typeof(value.scalar) === Float64,
              third.state.payload)
    @test all(value -> value.buffer isa Reactant.ConcretePJRTArray,
              third.state.payload)
    @test [Array(value.buffer) for value in third.state.payload] ==
          [value.buffer for value in prototype]
    @test all(value -> value.factorization isa LinearAlgebra.Cholesky,
              third.state.payload)
    @test all(value -> value.factorization.uplo == 'U' &&
                       value.factorization.info == 0,
              third.state.payload)
    @test all(value -> value.factor_alias isa Reactant.ConcretePJRTArray &&
                       value.factorization.factors.diag ===
                       value.factor_alias, third.state.payload)
    @test [Array(value.factor_alias) for value in third.state.payload] ==
          [value.factor_alias for value in prototype]
end
end

if _mpbr_enabled("nuts")
@testset "mutation profile B — specialized NUTS Reactant" begin
    max_depth = 1
    pf = _mpbr_nuts_factory()
    frame = _mpbr_nuts_frame(pf, max_depth)
    compiled = ReactiveKernelsNUTSExamples.compile_nuts_reactant(
        pf, _MPBR_NUTS.nuts_state, _MPBR_NUTS.refresh_momentum!!,
        _MPBR_NUTS.nuts!!, frame)
    momentum = [0.25, -0.5]
    bundle = ReactiveKernelsNUTSExamples.nuts_reactant_bundle(
        momentum, [false], [0.5, 0.75], max_depth)
    state = map(Reactant.to_rarray,
        ReactiveKernelsNUTSExamples.nuts_reactant_state(
            compiled, frame, bundle))
    executable = ReactiveKernelsNUTSExamples.nuts_reactant_compile(
        compiled, state; sync=true)
    output = executable(state)

    @test Array(output.pp_pos)[:, 1] != [1.0, 2.0]
    @test Array(output.n_steps)[1] > 0
    @test Array(output.reached_depth)[1] > 0
    @test Array(output.kd)[1] > 0
    @test Array(output.csp)[1] == 0
    @test all(iszero, Array(output.overflow))
    @test count(_ -> true,
        eachmatch(r"stablehlo\.while", executable.module_string)) == 1

    # The same executable accepts a second fixed-shape bundle and carries the
    # first transition's state, rather than recompiling or restarting from the
    # original host frame.
    second_bundle = ReactiveKernelsNUTSExamples.nuts_reactant_bundle(
        [0.75, 0.125], [true], [0.25, 0.875], max_depth)
    second_state = ReactiveKernelsNUTSExamples.nuts_reactant_rebundle(
        output, map(Reactant.to_rarray, second_bundle))
    second_output = executable(second_state)
    @test Array(second_output.pp_pos)[:, 1] == Array(output.pp_pos)[:, 1]
    @test Array(second_output.pp_pos)[:, 1] != [1.0, 2.0]
    @test Array(second_output.n_steps)[1] > 0
    @test Array(second_output.reached_depth)[1] > 0
    @test Array(second_output.csp)[1] == 0
    @test all(iszero, Array(second_output.overflow))
end
end
