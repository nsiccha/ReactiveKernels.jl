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
isdefined(@__MODULE__, :MutationProfileBGenericNUTSSupport) || include(joinpath(
    @__DIR__, "fixtures", "mutation_profile_b_generic_nuts_support.jl"))

module _MutationProfileBReactantNUTS
include(joinpath(@__DIR__, "..", "benchmark",
                 "nuts_kernel_authoring_fixture_b.jl"))
end

const _MPBR_HMC = ReactiveHMCHMCCompilerSupport
const _MPBR_NUTS =
    _MutationProfileBReactantNUTS.NUTSBMutationAuthoringFixture
const _MPBR_GENERIC_CONTROL = MutationProfileBGenericControl
const _MPBR_GENERIC_NUTS = MutationProfileBGenericNUTSSupport
const _MPBR_TESTSET = get(ENV, "RK_MPB_REACTANT_TESTSET", "all")
_mpbr_enabled(name) = _MPBR_TESTSET == "all" || _MPBR_TESTSET == name

if _mpbr_enabled("readonly-index")
@testset "readonly integer controls seed traced loop indices" begin
    case = _MPBR_GENERIC_CONTROL.readonly_index_case()
    state = Reactant.to_rarray(case.state; track_numbers=true)
    compiled = @compile sync=true donated_args=:none case.transition(state)
    result = compiled(state)
    @test Float64(result.state.total) == 5
    @test Int(result.state.limit) == 4
    @test !Bool(result.control_overflow)
    @test Float64(state.total) == 0
    @test Float64(compiled(result.state).state.total) == 10
    short = _MPBR_GENERIC_CONTROL.readonly_index_case(; max_iterations=3)
    guarded = @compile sync=true donated_args=:none short.transition(state)
    exhausted = guarded(state)
    @test Bool(exhausted.control_overflow)
    @test Float64(exhausted.state.total) == 0
end
end

if _mpbr_enabled("readonly-array")
@testset "recursive array argument forwarding through Reactant" begin
    case = _MPBR_GENERIC_CONTROL.array_reader_case()
    state = Reactant.to_rarray(case.state; track_numbers=true)
    input = Reactant.to_rarray([2.0, 5.0])
    compiled = @compile sync=true donated_args=:none case.transition(state, input)
    result = compiled(state, input)
    @test Float64(result.state.total) == 21
    @test !Bool(result.control_overflow)
    @test Array(input) == [2, 5]
    @test Float64(state.total) == 0
    next = compiled(result.state, Reactant.to_rarray([4.0, 1.0]))
    @test Float64(next.state.total) == 36
end
end

module _MPBRTraceBlockOverlay
using ReactiveKernels, Reactant
leaf(value) = value + one(value)
Reactant.@reactant_overlay leaf(value) = value + oftype(value, 2)
function operation(carry)
    Reactant.Ops.case(zero(carry.value), Any[block], carry; track_numbers=Union{})
end
const generated = ReactiveKernels.compile(:((ports, rng, ensures, carry) ->
    (value=$(GlobalRef(@__MODULE__, :leaf))(carry.value),)))
const block = ReactiveKernels._SMControlTraceBlock(
    generated, nothing, nothing, nothing)
end

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

_mpbr_materialize(value::Number, prototype::Number) =
    typeof(prototype)(value)
function _mpbr_materialize(value::AbstractArray, prototype::AbstractArray)
    eltype(prototype) <: Number &&
        return convert(typeof(prototype), Array(value))
    convert(typeof(prototype), map(eachindex(prototype)) do index
        _mpbr_materialize(value[index], prototype[index])
    end)
end
function _mpbr_materialize(value::NamedTuple, prototype::NamedTuple)
    names = propertynames(prototype)
    propertynames(value) == names || throw(ArgumentError(
        "materialized state has the wrong NamedTuple layout"))
    NamedTuple{names}(Tuple(_mpbr_materialize(
        getfield(value, name), getfield(prototype, name)) for name in names))
end
_mpbr_materialize(value::Tuple, prototype::Tuple) =
    map(_mpbr_materialize, value, prototype)
_mpbr_materialize(value::Diagonal, prototype::Diagonal) =
    Diagonal(_mpbr_materialize(value.diag, prototype.diag))
_mpbr_materialize(value::LinearAlgebra.Cholesky,
                  prototype::LinearAlgebra.Cholesky) =
    LinearAlgebra.Cholesky(
        _mpbr_materialize(value.factors, prototype.factors),
        value.uplo, Int(value.info))
_mpbr_materialize(
        value::Reactant.TracedLinearAlgebra.BatchedCholesky,
        prototype::LinearAlgebra.Cholesky) =
    LinearAlgebra.Cholesky(
        _mpbr_materialize(value.factors, prototype.factors),
        value.uplo, Int(value.info))
_mpbr_materialize(value, prototype) = value

function _mpbr_cholesky_wrappers(state)
    wrappers = Any[
        state.init.chol_metric,
        state.fwd.chol_metric,
        state.bwd.chol_metric,
    ]
    append!(wrappers,
        (proposal.chol_metric for proposal in state.proposals))
    wrappers
end

function _mpbr_materialize_replay(replay, prototype)
    ReactiveKernels._sm_ordered_rng_reconstruct(
        convert(typeof(prototype.normals), Array(replay.normals)),
        convert(typeof(prototype.uniforms), Array(replay.uniforms)),
        convert(typeof(prototype.exponentials), Array(replay.exponentials)),
        convert(typeof(prototype.event_tokens), Array(replay.event_tokens)),
        typeof(prototype.normal_index)(replay.normal_index),
        typeof(prototype.uniform_index)(replay.uniform_index),
        typeof(prototype.exponential_index)(replay.exponential_index),
        typeof(prototype.event_index)(replay.event_index),
        typeof(prototype.overflow)(replay.overflow))
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

if _mpbr_enabled("runtime-controls")
@testset "generated endpoint runtime numeric controls through Reactant" begin
    endpoint = ReactiveKernels.compile_state_transition(
        _MPBR_GENERIC_CONTROL.owned_point, _MPBR_GENERIC_CONTROL.bump_point!,
        ([1, 2],); runtime_controls=(delta=1,))
    state = _mpbr_trace(ReactiveKernels.initial_transition_state(endpoint))
    controls = _mpbr_trace((delta=1,))
    compiled = @compile endpoint(state, controls)
    guarded = ReactiveKernels.validated_compiled_transition(compiled, endpoint)
    for delta in (0, 1, -2, 4)
        result = guarded(state, _mpbr_trace((; delta)))
        @test Array(result.values) == [1 + delta, 2 + delta]
        @test Int(result.total) == 3 + 2delta
        @test result.values === result.mirror
    end
    @test Array(state.values) == [1, 2]
    @test_throws ArgumentError guarded(state, _mpbr_trace((delta=1.0,)))
end
end

if _mpbr_enabled("loop-scope")
@testset "compiled loop bindings preserve lexical scopes" begin
    for (method, expected) in ((Val(:grids!), 15), (Val(:nested!), 22))
        case = _MPBR_GENERIC_CONTROL.loop_scope_case(method)
        state = _mpbr_trace(case.state)
        compiled = @compile sync=true donated_args=:none case.transition(state)
        guarded = ReactiveKernels.validated_compiled_transition(compiled, case.transition)
        result = guarded(state)
        @test !Bool(result.control_overflow)
        @test Int(result.state.total) == expected
        @test Int(state.total) == 0
        again = guarded(result.state)
        @test Int(again.state.total) == 2expected
    end
    short = _MPBR_GENERIC_CONTROL.loop_scope_case(Val(:grids!); max_iterations=4)
    state = _mpbr_trace(short.state)
    compiled = @compile sync=true donated_args=:none short.transition(state)
    guarded = ReactiveKernels.validated_compiled_transition(compiled, short.transition)
    exhausted = guarded(state)
    @test Bool(exhausted.control_overflow)
    @test Int(exhausted.state.total) == 0
end
end

if _mpbr_enabled("generic-owned")
@testset "owned aliases survive compiled recursive suspension" begin
    owned = _MPBR_GENERIC_CONTROL.owned_alias_case()
    state = _mpbr_trace(owned.state)
    compiled = @compile owned.transition(state)
    guarded = ReactiveKernels.validated_compiled_transition(compiled, owned.transition)
    result = guarded(state)
    @test !Bool(result.control_overflow)
    @test Array(result.state.left.values) == [7, 8]
    @test Array(result.state.right.values) == [4, 5]
    @test Int(result.state.counter) == 81
    @test result.state.left.values === result.state.left.mirror
    @test result.state.right.values === result.state.right.mirror
    @test result.state.left.values !== result.state.right.values
    again = guarded(result.state)
    @test Array(again.state.left.values) == [13, 14]
    @test Array(again.state.right.values) == [7, 8]
    @test again.state.left.values === again.state.left.mirror
    @test again.state.right.values === again.state.right.mirror
    @test Array(state.left.values) == [1, 2]
end
end

if _mpbr_enabled("control-dispatch")
@testset "traced control address ranges preserve sparse cases" begin
    fixture = _MPBR_GENERIC_CONTROL
    state = _mpbr_trace((ctrl_mid=[0, 11], ctrl_pc=[0, 2], csp=2))
    compiled = @compile sync=true donated_args=:none fixture.address_probe(state)
    for ((method, pc), expected) in zip(fixture.ADDRESS_INPUTS,
                                       fixture.ADDRESS_EXPECTED)
        input = _mpbr_trace((ctrl_mid=[0, method], ctrl_pc=[0, pc], csp=2))
        @test Int(compiled(input)) == expected
    end
end
end

if _mpbr_enabled("trace-block")
@testset "generated control tracing preserves backend call overlays" begin
    @test _MPBRTraceBlockOverlay.block((value=3,)).value == 4
    state = (value=Reactant.to_rarray(3; track_numbers=true),)
    compiled = @compile _MPBRTraceBlockOverlay.operation(state)
    @test Int(compiled(state).value) == 5
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
    compiled = @compile sync=true donated_args=:none transition(state, traced_replay)
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
    # Functional control must preserve the caller's scalar state and replay
    # cursors, even though Reactant represents traced numbers by mutable boxes.
    @test Int(state.total) == host_state.total
    @test (Int(traced_replay.normal_index), Int(traced_replay.uniform_index),
           Int(traced_replay.exponential_index), Int(traced_replay.event_index)) ==
          (1, 1, 1, 1)
    repeated = compiled(state, traced_replay)
    @test Int(repeated.state.total) == native_result.state.total
    @test Int(only(repeated.arguments).event_index) == native_actual.event_index
    @test Int(result.state.total) == native_result.state.total
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

if _mpbr_enabled("generic-nuts")
@testset "mutation profile B — unchanged NUTS through generic Reactant compiler" begin
    case = _MPBR_GENERIC_NUTS.build_case()
    state = _mpbr_trace(case.snapshot)
    replay = _mpbr_trace_replay(case.replay)
    raw_state = _mpbr_trace(case.snapshot)
    raw_replay = _mpbr_trace_replay(case.replay)
    compiled = @compile sync=true donated_args=:none case.transition(state, replay)
    guarded = ReactiveKernels.validated_compiled_transition(
        compiled, case.transition)

    @test getfield(guarded, :compiled) === compiled
    raw = compiled(raw_state, raw_replay)
    # Machine entry packs both finite ports, then the raw return restores both
    # containers and logical Cholesky wrappers. Scalar leaves remain concrete
    # until the reusable guarded boundary performs its host bridge.
    @test raw.state.trees isa Vector
    @test raw.state.proposals isa Vector
    @test all(wrapper -> wrapper isa LinearAlgebra.Cholesky,
        _mpbr_cholesky_wrappers(raw.state))
    @test raw.state.proposals[1].pot isa Reactant.ConcretePJRTNumber

    first = guarded(state, replay)
    first_state = _mpbr_materialize(first.state, case.snapshot)
    first_dynamic = _MPBR_GENERIC_NUTS._generic_dynamic(first_state)
    first_replay = _mpbr_materialize_replay(
        first.arguments[1], case.replay)
    first_effects = _mpbr_materialize(
        first.effects, case.result.effects)
    # Test useful sampler behavior without requiring cross-backend rounding
    # agreement or constraining the compiler's arithmetic optimization.
    @test all(isfinite, first_dynamic.pp_pos)
    @test all(isfinite, first_dynamic.pp_mom)
    @test all(isfinite, first_dynamic.pp_ham)
    @test all(weight -> isfinite(weight) || weight == -Inf, first_dynamic.lw)
    @test 0 <= first_state.acceptance_rate <= 1
    @test 1 <= first_state.reached_depth <= case.max_depth
    @test !Bool(first.control_overflow)
    @test !first_replay.overflow
    @test (first_replay.normal_index, first_replay.uniform_index,
           first_replay.exponential_index, first_replay.event_index) ==
          (case.source_first_rng.normal_index,
           case.source_first_rng.uniform_index,
           case.source_first_rng.exponential_index,
           case.source_first_rng.event_index) == (1, 3, 3, 5)
    @test first_replay.normals == case.replay.normals
    @test first_replay.uniforms == case.replay.uniforms
    @test first_replay.exponentials == case.replay.exponentials
    @test first_replay.event_tokens == case.replay.event_tokens
    @test case.source_first_rng.events[
              1:(case.source_first_rng.event_index - 1)] ==
          [:uniform, :exponential, :uniform, :exponential]
    @test first_state.n_steps ==
          first_effects.stats_f.n_steps ==
          case.source_dynamic.n_steps
    @test first_state.acceptance_rate ==
          first_effects.stats_f.acceptance_rate
    @test typeof(first.state.proposals[1].pot) === Float64
    @test all(wrapper -> wrapper isa LinearAlgebra.Cholesky,
        _mpbr_cholesky_wrappers(first.state))

    second = guarded(first.state, first.arguments[1])
    second_state = _mpbr_materialize(second.state, case.snapshot)
    second_dynamic = _MPBR_GENERIC_NUTS._generic_dynamic(second_state)
    second_replay = _mpbr_materialize_replay(
        second.arguments[1], case.replay)
    second_effects = _mpbr_materialize(
        second.effects, case.second.result.effects)
    # Test useful sampler behavior without requiring cross-backend rounding
    # agreement or constraining the compiler's arithmetic optimization.
    @test all(isfinite, second_dynamic.pp_pos)
    @test all(isfinite, second_dynamic.pp_mom)
    @test all(isfinite, second_dynamic.pp_ham)
    @test all(weight -> isfinite(weight) || weight == -Inf, second_dynamic.lw)
    @test 0 <= second_state.acceptance_rate <= 1
    @test 1 <= second_state.reached_depth <= case.max_depth
    @test !Bool(second.control_overflow)
    @test !second_replay.overflow
    @test (second_replay.normal_index, second_replay.uniform_index,
           second_replay.exponential_index, second_replay.event_index) ==
          (case.second.source_rng.normal_index,
           case.second.source_rng.uniform_index,
           case.second.source_rng.exponential_index,
           case.second.source_rng.event_index)
    @test second_replay.normals == first_replay.normals
    @test second_replay.uniforms == first_replay.uniforms
    @test second_replay.exponentials == first_replay.exponentials
    @test second_replay.event_tokens == first_replay.event_tokens
    @test second_state.n_steps ==
          second_effects.stats_f.n_steps ==
          case.second.source_dynamic.n_steps
    @test second_state.acceptance_rate ==
          second_effects.stats_f.acceptance_rate
    @test getfield(guarded, :compiled) === compiled
    @test typeof(second.state) === typeof(first.state)
    @test typeof(second.state.proposals[1].pot) === Float64
    @test all(wrapper -> wrapper isa LinearAlgebra.Cholesky,
        _mpbr_cholesky_wrappers(second.state))
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
