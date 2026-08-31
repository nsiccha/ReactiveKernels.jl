using ReactiveKernels
using Reactant
using LinearAlgebra
using Test
import Reactant: @compile

const _FSCR_RK = ReactiveKernels

mutable struct _FSCRStaticAuthority
    token::Symbol
end

function _fscr_element(seed, authority)
    metric_storage = [Float64(seed), Float64(seed + 1)]
    factor_storage = [Float64(seed + 2), Float64(seed + 3)]
    buffer = [Float64(seed + 4), Float64(seed + 5)]
    (;
        authority,
        scalar=Float64(seed),
        counter=Int(seed),
        metric=Diagonal(metric_storage),
        metric_alias=metric_storage,
        factorization=LinearAlgebra.Cholesky(
            Diagonal(factor_storage), 'U', 0),
        factor_alias=factor_storage,
        nested=(weight=Float64(seed + 6), buffer=buffer),
    )
end

_fscr_bool_element(seed) = (;
    flag=isodd(seed),
    mask=Bool[isodd(seed), iseven(seed)],
)

_fscr_trace_raw(raw) = map(Reactant.to_rarray, raw)
_fscr_host_raw(raw) = map(Array, raw)
_fscr_traced(value) = Reactant.to_rarray(value; track_numbers=true)

function _fscr_trace_value(value)
    _fscr_trace_value(value, IdDict{Any,Any}())
end
function _fscr_trace_value(value::AbstractArray, seen)
    get!(seen, value) do
        Reactant.to_rarray(value)
    end
end
_fscr_trace_value(value::Number, seen) =
    Reactant.to_rarray(value; track_numbers=true)
_fscr_trace_value(value::NamedTuple, seen) =
    map(child -> _fscr_trace_value(child, seen), value)
_fscr_trace_value(value::Tuple, seen) =
    map(child -> _fscr_trace_value(child, seen), value)
_fscr_trace_value(value, seen) = value

struct _FSCRRead{C}
    contract::C
end
function (operation::_FSCRRead)(raw, index, active)
    result = _FSCR_RK._sm_finite_structural_read(
        operation.contract, raw, index, active)
    value = result.value
    (scalar=value.scalar, metric=copy(value.metric_alias),
     overflow=result.overflow)
end

struct _FSCRWrite{C}
    contract::C
end
function (operation::_FSCRWrite)(raw, index, active)
    read = _FSCR_RK._sm_finite_structural_read(
        operation.contract, raw, index, active)
    value = read.value
    metric_storage = value.metric_alias .+ oftype(value.scalar, 10)
    replacement = merge(value, (
        scalar=value.scalar + oftype(value.scalar, 100),
        metric=Diagonal(metric_storage),
        metric_alias=metric_storage,
    ))
    _FSCR_RK._sm_finite_structural_write(
        operation.contract, raw, index, replacement, active)
end

struct _FSCRCopy{C}
    contract::C
end
(operation::_FSCRCopy)(raw, destination, source, active) =
    _FSCR_RK._sm_finite_structural_copy(
        operation.contract, raw, destination, source, active)

struct _FSCRSwap{C}
    contract::C
end
(operation::_FSCRSwap)(raw, left, right, active) =
    _FSCR_RK._sm_finite_structural_swap(
        operation.contract, raw, left, right, active)

struct _FSCRSelect{C}
    contract::C
end

struct _FSCRBoolRead{C}
    contract::C
end
function (operation::_FSCRBoolRead)(raw, index, active)
    result = _FSCR_RK._sm_finite_structural_read(
        operation.contract, raw, index, active)
    (flag=result.value.flag, mask=copy(result.value.mask),
     overflow=result.overflow)
end

struct _FSCRBoolWrite{C}
    contract::C
end


struct _FSCRCapturedIndexWrite{C}
    contract::C
end
function (operation::_FSCRCapturedIndexWrite)(raw, index, active)
    captured_index = index
    bound = _FSCR_RK._sm_finite_structural_read(
        operation.contract, raw, captured_index, active)
    later_index = index + one(index)
    replacement = (
        flag=!bound.value.flag,
        mask=.!bound.value.mask,
    )
    written = _FSCR_RK._sm_finite_structural_write(
        operation.contract, raw, captured_index, replacement, active)
    (; written.storage, written.overflow, later_index)
end
function (operation::_FSCRBoolWrite)(raw, index, active)
    read = _FSCR_RK._sm_finite_structural_read(
        operation.contract, raw, index, active)
    value = read.value
    replacement = (flag=!value.flag, mask=.!value.mask)
    _FSCR_RK._sm_finite_structural_write(
        operation.contract, raw, index, replacement, active)
end
(operation::_FSCRSelect)(active, candidate, prior) =
    _FSCR_RK._sm_finite_structural_select(
        operation.contract, active, candidate, prior)

struct _FSCRMissingOutput{Names} end
function (::_FSCRMissingOutput{Names})(raw) where {Names}
    kept = Base.front(Names)
    NamedTuple{kept}(Base.front(values(raw)))
end


@testset "finite structural Bool operations execute through Reactant" begin
    prototype = [_fscr_bool_element(index) for index in 1:3]
    contract = _FSCR_RK._sm_finite_structural_contract(prototype)
    raw = _FSCR_RK._sm_finite_structural_pack(contract, prototype)
    traced_raw = _fscr_trace_raw(raw)
    traced_true = _fscr_traced(true)
    traced_false = _fscr_traced(false)

    read = _FSCRBoolRead(contract)
    compiled_read = @compile read(
        traced_raw, _fscr_traced(2), traced_true)
    read_second = compiled_read(
        traced_raw, _fscr_traced(2), traced_true)
    @test Bool(read_second.flag) == prototype[2].flag
    @test Array(read_second.mask) == prototype[2].mask
    @test !Bool(read_second.overflow)

    write = _FSCRBoolWrite(contract)
    compiled_write = @compile write(
        traced_raw, _fscr_traced(2), traced_true)
    written = compiled_write(
        traced_raw, _fscr_traced(2), traced_true)
    @test !Bool(written.overflow)
    written_host = _fscr_host_raw(written.storage)
    @test all(column -> eltype(column) === Bool, values(written_host))
    written_values = _FSCR_RK._sm_finite_structural_unpack(
        contract, written_host)
    @test written_values[2] == (
        flag=!prototype[2].flag, mask=.!prototype[2].mask)
    @test written_values[1] == prototype[1]

    invalid_write = compiled_write(
        traced_raw, _fscr_traced(0), traced_true)
    @test Bool(invalid_write.overflow)
    @test _fscr_host_raw(invalid_write.storage) == raw
    inactive_write = compiled_write(
        traced_raw, _fscr_traced(0), traced_false)
    @test !Bool(inactive_write.overflow)
    @test _fscr_host_raw(inactive_write.storage) == raw

    copy_operation = _FSCRCopy(contract)
    compiled_copy = @compile copy_operation(
        traced_raw, _fscr_traced(3), _fscr_traced(1), traced_true)
    copied = compiled_copy(
        traced_raw, _fscr_traced(3), _fscr_traced(1), traced_true)
    @test !Bool(copied.overflow)
    @test _FSCR_RK._sm_finite_structural_unpack(
        contract, _fscr_host_raw(copied.storage))[3] == prototype[1]
    invalid_copy = compiled_copy(
        traced_raw, _fscr_traced(4), _fscr_traced(1), traced_true)
    @test Bool(invalid_copy.overflow)
    @test _fscr_host_raw(invalid_copy.storage) == raw

    swap = _FSCRSwap(contract)
    compiled_swap = @compile swap(
        traced_raw, _fscr_traced(1), _fscr_traced(3), traced_true)
    swapped = compiled_swap(
        traced_raw, _fscr_traced(1), _fscr_traced(3), traced_true)
    @test !Bool(swapped.overflow)
    swapped_values = _FSCR_RK._sm_finite_structural_unpack(
        contract, _fscr_host_raw(swapped.storage))
    @test swapped_values[1] == prototype[3]
    @test swapped_values[3] == prototype[1]
    inactive_swap = compiled_swap(
        traced_raw, _fscr_traced(0), _fscr_traced(4), traced_false)
    @test !Bool(inactive_swap.overflow)
    @test _fscr_host_raw(inactive_swap.storage) == raw

    select_operation = _FSCRSelect(contract)
    compiled_select = @compile select_operation(
        traced_true, written.storage, traced_raw)
    @test _fscr_host_raw(compiled_select(
        traced_true, written.storage, traced_raw)) == written_host
    @test _fscr_host_raw(compiled_select(
        traced_false, written.storage, traced_raw)) == raw
end


@testset "captured structural index survives a later Reactant index change" begin
    prototype = [_fscr_bool_element(index) for index in 1:3]
    contract = _FSCR_RK._sm_finite_structural_contract(prototype)
    raw = _FSCR_RK._sm_finite_structural_pack(contract, prototype)
    traced_raw = _fscr_trace_raw(raw)
    traced_index = _fscr_traced(1)
    traced_true = _fscr_traced(true)
    operation = _FSCRCapturedIndexWrite(contract)
    compiled = @compile operation(
        traced_raw, traced_index, traced_true)
    result = compiled(traced_raw, traced_index, traced_true)
    @test Int(result.later_index) == 2
    @test !Bool(result.overflow)
    values_after = _FSCR_RK._sm_finite_structural_unpack(
        contract, _fscr_host_raw(result.storage))
    @test values_after[1] == (
        flag=!prototype[1].flag,
        mask=.!prototype[1].mask,
    )
    @test values_after[2] == prototype[2]
end


@testset "authored state-machine captured index executes through Reactant" begin
    if !isdefined(@__MODULE__, :StatefulFunctionalContractsFixture)
        include(joinpath(@__DIR__, "fixtures",
                         "stateful_functional_contracts.jl"))
    end
    fixture = StatefulFunctionalContractsFixture
    items = fixture.captured_alias_items()
    replacement = fixture.captured_alias_items(100.0)
    port = _FSCR_RK._sm_fixed_structural_tuple_port(items)
    bindings = _FSCR_RK.stateful_compiler_bindings(
        items=port, replacement=port)
    kernel = _FSCR_RK.compile_stateful(
        fixture.captured_index_alias, bindings,
        items, replacement, 1, 0)
    transition = _FSCR_RK.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Float64,Bool})
    host_state = _FSCR_RK.stateful_snapshot(
        kernel(items, replacement, 1, 0))
    traced_state = _fscr_trace_value(host_state)
    traced_value = _fscr_traced(41.0)
    traced_take = _fscr_traced(true)
    compiled = @compile transition(
        traced_state, traced_value, traced_take)
    result = compiled(traced_state, traced_value, traced_take)
    @test Bool(result.returned) && Bool(result.result)
    @test !Bool(result.control_overflow)
    @test Int(result.state.index) == 2
    @test Int(result.state.count) == 1
    @test Float64(result.state.observed) == 41.0
    @test result.state.items[1].left === result.state.items[1].right
    @test Array(result.state.items[1].left) == [41.0, 1.5]
    @test Array(result.state.items[2].left) == items[2].left
end


@testset "finite structural Reactant suite rejects unsupported wrappers" begin
    unsupported = Any[
        1 // 2,
        ComplexF64(1, 2),
        Rational{Int}[1 // 2, 2 // 3],
        ComplexF64[1 + 2im, 3 + 4im],
        Int128(1),
        Int128[1, 2],
        big(1),
        BigInt[1, 2],
        BitVector([true, false]),
    ]
    for value in unsupported
        prototype = [(leaf=deepcopy(value),) for _ in 1:2]
        @test_throws ArgumentError begin
            _FSCR_RK._sm_finite_structural_contract(prototype)
        end
    end
end

@testset "finite structural numeric SoA operations execute through Reactant" begin
    authority = _FSCRStaticAuthority(:bound)
    prototype = [_fscr_element(index, authority) for index in 1:3]
    contract = _FSCR_RK._sm_finite_structural_contract(
        prototype; static_values=(authority,))
    raw = _FSCR_RK._sm_finite_structural_pack(contract, prototype)
    traced_raw = _fscr_trace_raw(raw)
    traced_index = _fscr_traced(2)
    traced_true = _fscr_traced(true)
    traced_false = _fscr_traced(false)

    read = _FSCRRead(contract)
    compiled_read = @compile read(
        traced_raw, traced_index, traced_true)
    read_first = compiled_read(
        traced_raw, _fscr_traced(1), traced_true)
    @test Float64(read_first.scalar) == prototype[1].scalar
    @test Array(read_first.metric) == prototype[1].metric_alias
    @test !Bool(read_first.overflow)
    read_invalid = compiled_read(
        traced_raw, _fscr_traced(0), traced_true)
    @test Bool(read_invalid.overflow)
    read_inactive = compiled_read(
        traced_raw, _fscr_traced(0), traced_false)
    @test !Bool(read_inactive.overflow)

    write = _FSCRWrite(contract)
    compiled_write = @compile write(
        traced_raw, traced_index, traced_true)
    written = compiled_write(
        traced_raw, _fscr_traced(2), traced_true)
    @test !Bool(written.overflow)
    written_values = _FSCR_RK._sm_finite_structural_unpack(
        contract, _fscr_host_raw(written.storage))
    @test written_values[2].scalar == prototype[2].scalar + 100
    @test written_values[2].metric_alias ==
          prototype[2].metric_alias .+ 10
    @test written_values[2].metric.diag ===
          written_values[2].metric_alias
    @test written_values[2].authority === authority

    invalid_write = compiled_write(
        traced_raw, _fscr_traced(0), traced_true)
    @test Bool(invalid_write.overflow)
    @test _fscr_host_raw(invalid_write.storage) == raw
    inactive_write = compiled_write(
        traced_raw, _fscr_traced(0), traced_false)
    @test !Bool(inactive_write.overflow)
    @test _fscr_host_raw(inactive_write.storage) == raw

    copy_operation = _FSCRCopy(contract)
    compiled_copy = @compile copy_operation(
        traced_raw, _fscr_traced(3), _fscr_traced(1), traced_true)
    copied = compiled_copy(
        traced_raw, _fscr_traced(3), _fscr_traced(1), traced_true)
    @test !Bool(copied.overflow)
    copied_values = _FSCR_RK._sm_finite_structural_unpack(
        contract, _fscr_host_raw(copied.storage))
    @test copied_values[3] == prototype[1]
    @test copied_values[3].metric_alias !== copied_values[1].metric_alias
    invalid_copy = compiled_copy(
        traced_raw, _fscr_traced(4), _fscr_traced(1), traced_true)
    @test Bool(invalid_copy.overflow)
    @test _fscr_host_raw(invalid_copy.storage) == raw
    inactive_copy = compiled_copy(
        traced_raw, _fscr_traced(4), _fscr_traced(1), traced_false)
    @test !Bool(inactive_copy.overflow)
    @test _fscr_host_raw(inactive_copy.storage) == raw

    swap = _FSCRSwap(contract)
    compiled_swap = @compile swap(
        traced_raw, _fscr_traced(1), _fscr_traced(3), traced_true)
    swapped = compiled_swap(
        traced_raw, _fscr_traced(1), _fscr_traced(3), traced_true)
    @test !Bool(swapped.overflow)
    swapped_values = _FSCR_RK._sm_finite_structural_unpack(
        contract, _fscr_host_raw(swapped.storage))
    @test swapped_values[1] == prototype[3]
    @test swapped_values[3] == prototype[1]
    invalid_swap = compiled_swap(
        traced_raw, _fscr_traced(0), _fscr_traced(3), traced_true)
    @test Bool(invalid_swap.overflow)
    @test _fscr_host_raw(invalid_swap.storage) == raw
    inactive_swap = compiled_swap(
        traced_raw, _fscr_traced(0), _fscr_traced(3), traced_false)
    @test !Bool(inactive_swap.overflow)
    @test _fscr_host_raw(inactive_swap.storage) == raw

    candidate = written.storage
    select_operation = _FSCRSelect(contract)
    compiled_select = @compile select_operation(
        traced_true, candidate, traced_raw)
    @test _fscr_host_raw(compiled_select(
        traced_true, candidate, traced_raw)) == _fscr_host_raw(candidate)
    @test _fscr_host_raw(compiled_select(
        traced_false, candidate, traced_raw)) == raw
end

@testset "finite structural Reactant boundaries reject raw counterfeits" begin
    authority = _FSCRStaticAuthority(:bound)
    prototype = [_fscr_element(index, authority) for index in 1:3]
    contract = _FSCR_RK._sm_finite_structural_contract(
        prototype; static_values=(authority,))
    raw = _FSCR_RK._sm_finite_structural_pack(contract, prototype)
    traced_raw = _fscr_trace_raw(raw)
    names = propertynames(raw)
    write = _FSCRWrite(contract)
    traced_index = _fscr_traced(2)
    traced_true = _fscr_traced(true)

    missing = NamedTuple{Base.front(names)}(Base.front(values(raw)))
    extra = merge(raw, (extra_leaf=zeros(3),))
    wrong_axes = merge(
        raw, NamedTuple{(first(names),)}((zeros(2),)))
    for counterfeit in (missing, extra, wrong_axes)
        traced_counterfeit = _fscr_trace_raw(counterfeit)
        @test_throws ArgumentError @compile write(
            traced_counterfeit, traced_index, traced_true)
    end

    wrong_authority = copy(prototype)
    wrong_authority[1] = merge(
        wrong_authority[1],
        (authority=_FSCRStaticAuthority(:bound),))
    @test_throws ArgumentError _FSCR_RK._sm_finite_structural_pack(
        contract, wrong_authority)

    compiled_identity = @compile identity(traced_raw)
    guarded_identity = _FSCR_RK._sm_finite_validated_call(
        compiled_identity, contract)
    @test_throws ArgumentError guarded_identity(_fscr_trace_raw(missing))
    @test_throws ArgumentError guarded_identity(_fscr_trace_raw(extra))

    missing_output = _FSCRMissingOutput{names}()
    compiled_missing_output = @compile missing_output(traced_raw)
    guarded_missing_output = _FSCR_RK._sm_finite_validated_call(
        compiled_missing_output, contract)
    @test_throws ArgumentError guarded_missing_output(traced_raw)
end
