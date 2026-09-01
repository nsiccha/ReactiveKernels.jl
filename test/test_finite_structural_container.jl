using LinearAlgebra
using Test

const _FSC_RK = ReactiveKernels

mutable struct _FSCStaticAuthority
    token::Symbol
end

struct _FSCMixedFiniteTransition{
        Names,Groups,ArrayNames,StateType,Ports,Shapes,Topology}
    ports::Ports
    shape_contract::Shapes
    topology_contract::Topology
end

function _fsc_mixed_finite_transition(state, ports)
    names = propertynames(state)
    groups = Tuple((name,) for name in names)
    shapes = map(_FSC_RK._sm_shape_contract, state)
    topology = _FSC_RK._sm_topology_contract(state)
    _FSCMixedFiniteTransition{
        names,groups,names,typeof(state),typeof(ports),
        typeof(shapes),typeof(topology)}(ports, shapes, topology)
end

function _fsc_wrapper_element(seed)
    factors = Float64[seed, seed + 1]
    (;
        scalar=Float64(seed),
        factorization=LinearAlgebra.Cholesky(
            Diagonal(factors), 'U', 0),
        factor_alias=factors,
    )
end

function _fsc_backend_wrapper_element(seed)
    logical = _fsc_wrapper_element(seed)
    factorization = logical.factorization
    backend_factorization = (;
        factors=factorization.factors,
        uplo=factorization.uplo,
        info=factorization.info,
    )
    merge(logical, (; factorization=backend_factorization))
end

function _fsc_element(seed, authority)
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

function _fsc_assert_element_contract(actual, expected, authority)
    @test actual == expected
    @test actual.authority === authority
    @test actual.metric.diag === actual.metric_alias
    @test actual.factorization.factors.diag === actual.factor_alias
end

_fsc_bool_element(seed) = (;
    flag=isodd(seed),
    mask=Bool[isodd(seed), iseven(seed)],
)

@testset "finite structural vector uses a numeric SoA ABI" begin
    authority = _FSCStaticAuthority(:bound)
    prototype = [_fsc_element(index, authority) for index in 1:3]
    contract = _FSC_RK._sm_finite_structural_contract(
        prototype; static_values=(authority,))
    raw = _FSC_RK._sm_finite_structural_pack(contract, prototype)

    @test (@inferred _FSC_RK._sm_finite_validate_raw(contract, raw)) === raw
    @test (@inferred _FSC_RK._sm_finite_structural_read(
        contract, raw, 1)).value == prototype[1]
    @test (@inferred _FSC_RK._sm_finite_structural_write(
        contract, raw, 1, prototype[1])).storage == raw
    @test (@inferred _FSC_RK._sm_finite_structural_copy(
        contract, raw, 2, 1)).storage isa NamedTuple
    @test (@inferred _FSC_RK._sm_finite_structural_swap(
        contract, raw, 1, 2)).storage isa NamedTuple
    @test (@inferred _FSC_RK._sm_finite_structural_select(
        contract, true, raw, raw)) == raw
    @test (@inferred _FSC_RK._sm_finite_structural_unpack(
        contract, raw)) == prototype

    @test propertynames(raw) == typeof(contract).parameters[3]
    @test all(column ->
        column isa Array &&
        _FSC_RK._sm_finite_backend_primitive(eltype(column)), values(raw))
    @test all(size(column, ndims(column)) == length(prototype)
              for column in values(raw))

    unpacked = _FSC_RK._sm_finite_structural_unpack(contract, raw)
    for index in eachindex(prototype)
        _fsc_assert_element_contract(
            unpacked[index], prototype[index], authority)
    end
    for left in eachindex(unpacked), right in eachindex(unpacked)
        left < right || continue
        @test unpacked[left].metric_alias !== unpacked[right].metric_alias
        @test unpacked[left].factor_alias !== unpacked[right].factor_alias
        @test unpacked[left].nested.buffer !== unpacked[right].nested.buffer
    end

    replacement = _fsc_element(20, authority)
    written = _FSC_RK._sm_finite_structural_write(
        contract, raw, 2, replacement)
    @test !written.overflow
    written_values = _FSC_RK._sm_finite_structural_unpack(
        contract, written.storage)
    _fsc_assert_element_contract(written_values[2], replacement, authority)
    _fsc_assert_element_contract(written_values[1], prototype[1], authority)

    copied = _FSC_RK._sm_finite_structural_copy(
        contract, raw, 3, 1)
    @test !copied.overflow
    copied_values = _FSC_RK._sm_finite_structural_unpack(
        contract, copied.storage)
    @test copied_values[3] == prototype[1]
    @test copied_values[3].metric_alias !== copied_values[1].metric_alias
    @test copied_values[3].metric.diag === copied_values[3].metric_alias

    swapped = _FSC_RK._sm_finite_structural_swap(
        contract, raw, 1, 3)
    @test !swapped.overflow
    swapped_values = _FSC_RK._sm_finite_structural_unpack(
        contract, swapped.storage)
    @test swapped_values[1] == prototype[3]
    @test swapped_values[3] == prototype[1]

    selected = _FSC_RK._sm_finite_structural_select(
        contract, true, written.storage, raw)
    @test selected == written.storage
    @test _FSC_RK._sm_finite_structural_select(
        contract, false, written.storage, raw) == raw

    read = _FSC_RK._sm_finite_structural_read(contract, raw, 2)
    @test !read.overflow
    _fsc_assert_element_contract(read.value, prototype[2], authority)

    for operation in (
            () -> _FSC_RK._sm_finite_structural_write(
                contract, raw, 0, replacement),
            () -> _FSC_RK._sm_finite_structural_copy(
                contract, raw, 0, 1),
            () -> _FSC_RK._sm_finite_structural_copy(
                contract, raw, 1, 4),
            () -> _FSC_RK._sm_finite_structural_swap(
                contract, raw, 1, 4),
        )
        result = operation()
        @test result.overflow
        @test result.storage == raw
    end
    inactive_write = _FSC_RK._sm_finite_structural_write(
        contract, raw, 0, replacement, false)
    @test !inactive_write.overflow
    @test inactive_write.storage == raw
    inactive_swap = _FSC_RK._sm_finite_structural_swap(
        contract, raw, 0, 4, false)
    @test !inactive_swap.overflow
    @test inactive_swap.storage == raw
end


@testset "finite structural Bool scalar and array operations preserve types" begin
    prototype = [_fsc_bool_element(index) for index in 1:3]
    contract = _FSC_RK._sm_finite_structural_contract(prototype)
    raw = _FSC_RK._sm_finite_structural_pack(contract, prototype)

    @test all(column -> column isa Array{Bool}, values(raw))
    @test _FSC_RK._sm_finite_structural_read(
        contract, raw, 2).value == prototype[2]

    replacement = (flag=false, mask=Bool[false, false])
    written = _FSC_RK._sm_finite_structural_write(
        contract, raw, 2, replacement)
    @test !written.overflow
    @test all(column -> eltype(column) === Bool, values(written.storage))
    written_values = _FSC_RK._sm_finite_structural_unpack(
        contract, written.storage)
    @test written_values[2] == replacement
    @test written_values[1] == prototype[1]

    copied = _FSC_RK._sm_finite_structural_copy(
        contract, raw, 3, 1)
    @test !copied.overflow
    @test _FSC_RK._sm_finite_structural_unpack(
        contract, copied.storage)[3] == prototype[1]

    swapped = _FSC_RK._sm_finite_structural_swap(
        contract, raw, 1, 3)
    @test !swapped.overflow
    swapped_values = _FSC_RK._sm_finite_structural_unpack(
        contract, swapped.storage)
    @test swapped_values[1] == prototype[3]
    @test swapped_values[3] == prototype[1]

    @test _FSC_RK._sm_finite_structural_select(
        contract, true, written.storage, raw) == written.storage
    @test _FSC_RK._sm_finite_structural_select(
        contract, false, written.storage, raw) == raw

    invalid = _FSC_RK._sm_finite_structural_write(
        contract, raw, 0, replacement)
    @test invalid.overflow
    @test invalid.storage == raw
    inactive = _FSC_RK._sm_finite_structural_write(
        contract, raw, 0, replacement, false)
    @test !inactive.overflow
    @test inactive.storage == raw
end


@testset "finite structural alias writes retain the bound index" begin
    prototype = [_fsc_bool_element(index) for index in 1:3]
    contract = _FSC_RK._sm_finite_structural_contract(prototype)
    raw = _FSC_RK._sm_finite_structural_pack(contract, prototype)

    index = 1
    captured_index = index
    bound = _FSC_RK._sm_finite_structural_read(
        contract, raw, captured_index)
    index += 1
    replacement = (
        flag=!bound.value.flag,
        mask=map(!, bound.value.mask),
    )
    written = _FSC_RK._sm_finite_structural_write(
        contract, raw, captured_index, replacement)
    values_after = _FSC_RK._sm_finite_structural_unpack(
        contract, written.storage)
    @test index == 2
    @test values_after[1] == replacement
    @test values_after[2] == prototype[2]
end


@testset "authored state-machine alias retains its bound structural index" begin
    if !isdefined(@__MODULE__, :StatefulFunctionalContractsFixture)
        include(joinpath(@__DIR__, "fixtures",
                         "stateful_functional_contracts.jl"))
    end
    fixture = StatefulFunctionalContractsFixture
    items = fixture.captured_alias_items()
    replacement = fixture.captured_alias_items(100.0)
    port = _FSC_RK._sm_fixed_structural_tuple_port(items)
    bindings = _FSC_RK.stateful_compiler_bindings(
        items=port, replacement=port)
    @test _FSC_RK._sm_fixed_tuple_element_type(typeof(items)) ===
          typeof(first(items))
    for rejected in (
            ((leaf=1 // 2,), (leaf=1 // 2,)),
            ((leaf=1 + 2im,), (leaf=1 + 2im,)),
            ((authority=_FSCStaticAuthority(:a),),
             (authority=_FSCStaticAuthority(:b),)),
            ((left=[1.0],), (left=Float32[1.0],)),
        )
        @test _FSC_RK._sm_fixed_tuple_element_type(
            typeof(rejected)) === nothing
    end
    rational_items = ntuple(3) do index
        (buffer=Rational{Int}[index // 1, (index + 1) // 1],)
    end
    rational_kernel = _FSC_RK.compile_stateful(
        fixture.captured_index_alias,
        rational_items, rational_items, 1, 0)
    @test_throws _FSC_RK._LLowerReject _FSC_RK.functionalize_stateful(
        rational_kernel, Val(:step!);
        argument_types=Tuple{Float64,Bool})

    unbound_kernel = _FSC_RK.compile_stateful(
        fixture.captured_index_alias,
        items, replacement, 1, 0)
    @test_throws _FSC_RK._LLowerReject _FSC_RK.functionalize_stateful(
        unbound_kernel, Val(:step!);
        argument_types=Tuple{Float64,Bool})

    split_alias = ntuple(3) do index
        (left=[Float64(index), Float64(index) + 0.5],
         right=[Float64(index), Float64(index) + 0.5])
    end
    @test_throws ArgumentError _FSC_RK._sm_fixed_tuple_validate(
        port, split_alias)
    @test_throws ArgumentError _FSC_RK._sm_fixed_tuple_select(
        port, true, split_alias, items)

    shared = [1.0, 1.5]
    cross_shared = ntuple(3) do index
        index <= 2 ? (left=shared, right=shared) : begin
            storage = [3.0, 3.5]
            (left=storage, right=storage)
        end
    end
    @test_throws ArgumentError _FSC_RK._sm_fixed_structural_tuple_port(
        cross_shared)
    @test_throws ArgumentError _FSC_RK._sm_fixed_tuple_validate(
        port, cross_shared)

    straight_bindings = _FSC_RK.stateful_compiler_bindings(items=port)
    straight_error = try
        _FSC_RK.compile_stateful(
            fixture.straight_index_alias,
            straight_bindings, items, 1)
        nothing
    catch error
        error
    end
    @test straight_error isa _FSC_RK._LLowerReject
    @test sprint(showerror, straight_error) ==
          "ReactiveKernels._LLowerReject(\"ordinary straight-line " *
          "stateful writes require one direct self-owned field\")"

    kernel = _FSC_RK.compile_stateful(
        fixture.captured_index_alias, bindings,
        items, replacement, 1, 0)

    source = fixture.captured_index_alias_source_oracle(items, 1, 41.0)
    @test source.initial_alias
    @test source.final_alias
    @test source.index == 2
    @test source.observed == 41.0
    @test source.items[1].left == [41.0, 1.5]

    native = kernel(items, replacement, 1, 0)
    initial_native = _FSC_RK.stateful_snapshot(native)
    @test initial_native.items[1].left === initial_native.items[1].right
    @test _FSC_RK.stateful_call(
        native, Val(:step!), 41.0, true)
    native_snapshot = _FSC_RK.stateful_snapshot(native)
    @test native_snapshot.index == 2
    @test native_snapshot.count == 1
    @test native_snapshot.observed == 41.0
    @test native_snapshot.items[1].left ===
          native_snapshot.items[1].right
    @test native_snapshot.items[1].left == [41.0, 1.5]
    @test native_snapshot.items[2].left == items[2].left
    @test native_snapshot.items == source.items

    initial = _FSC_RK.stateful_snapshot(
        kernel(items, replacement, 1, 0))
    transition = _FSC_RK.functionalize_stateful(
        kernel, Val(:step!); argument_types=Tuple{Float64,Bool})
    functional = transition(initial, 41.0, true)
    @test functional.returned && functional.result
    @test !functional.control_overflow
    @test functional.state.index == native_snapshot.index
    @test functional.state.count == native_snapshot.count
    @test functional.state.observed == 41.0
    @test functional.state.items[1].left ===
          functional.state.items[1].right
    @test functional.state.items[1].left ==
          native_snapshot.items[1].left
    @test functional.state.items[2].left ==
          native_snapshot.items[2].left

    moved = kernel(items, replacement, 1, 0)
    @test _FSC_RK.stateful_call(
        moved, Val(:move_root!), 77.0, true)
    moved_snapshot = _FSC_RK.stateful_snapshot(moved)
    @test moved_snapshot.items == replacement
    @test moved_snapshot.index == 2
    root_error = try
        _FSC_RK.functionalize_stateful(
            kernel, Val(:move_root!);
            argument_types=Tuple{Float64,Bool})
        nothing
    catch error
        error
    end
    @test root_error isa _FSC_RK._LLowerReject
    @test sprint(showerror, root_error) ==
          "ReactiveKernels._LLowerReject(\"aliased state write root " *
          "`items` changed after alias binding\")"
end


@testset "finite structural backend leaves reject unsupported wrappers" begin
    unsupported = Any[
        1 // 2,
        ComplexF64(1, 2),
        Rational{Int}[1 // 2, 2 // 3],
        ComplexF64[1 + 2im, 3 + 4im],
        Int128(1),
        Int128[1, 2],
        big(1),
        BigInt[1, 2],
        big(1.0),
        BigFloat[1, 2],
        BitVector([true, false]),
    ]
    for value in unsupported
        prototype = [(leaf=deepcopy(value),) for _ in 1:2]
        @test_throws ArgumentError begin
            _FSC_RK._sm_finite_structural_contract(prototype)
        end
    end
end

@testset "finite structural fresh and reused inputs reject counterfeits" begin
    authority = _FSCStaticAuthority(:bound)
    prototype = [_fsc_element(index, authority) for index in 1:3]
    contract = _FSC_RK._sm_finite_structural_contract(
        prototype; static_values=(authority,))
    raw = _FSC_RK._sm_finite_structural_pack(contract, prototype)

    wrong_authority = copy(prototype)
    wrong_authority[2] = merge(
        wrong_authority[2],
        (authority=_FSCStaticAuthority(:bound),))
    @test_throws ArgumentError _FSC_RK._sm_finite_structural_pack(
        contract, wrong_authority)

    broken_alias = copy(prototype)
    broken_alias[2] = merge(
        broken_alias[2],
        (metric_alias=copy(broken_alias[2].metric_alias),))
    @test_throws ArgumentError _FSC_RK._sm_finite_structural_pack(
        contract, broken_alias)

    merged_groups = copy(prototype)
    shared = merged_groups[2].metric_alias
    merged_groups[2] = merge(
        merged_groups[2],
        (factorization=LinearAlgebra.Cholesky(
             Diagonal(shared), 'U', 0),
         factor_alias=shared))
    @test_throws ArgumentError _FSC_RK._sm_finite_structural_pack(
        contract, merged_groups)

    wrong_axes = copy(prototype)
    short = [1.0]
    wrong_axes[2] = merge(
        wrong_axes[2],
        (metric=Diagonal(short), metric_alias=short))
    @test_throws ArgumentError _FSC_RK._sm_finite_structural_pack(
        contract, wrong_axes)

    short_capacity = copy(prototype[1:2])
    @test_throws ArgumentError _FSC_RK._sm_finite_structural_pack(
        contract, short_capacity)

    cross_shared = copy(prototype)
    cross_shared[2] = merge(
        cross_shared[2],
        (metric=Diagonal(cross_shared[1].metric_alias),
         metric_alias=cross_shared[1].metric_alias))
    @test_throws ArgumentError _FSC_RK._sm_finite_structural_contract(
        cross_shared; static_values=(authority,))

    @test_throws ArgumentError _FSC_RK._sm_finite_structural_contract(
        prototype; static_values=(authority, authority))
    @test_throws ArgumentError _FSC_RK._sm_finite_structural_contract(
        prototype; static_values=(authority, _FSCStaticAuthority(:unused)))
    @test_throws ArgumentError _FSC_RK._sm_finite_structural_contract(
        prototype)

    # A second pack is a reusable-input guard, not a trust-on-first-use path.
    @test _FSC_RK._sm_finite_structural_pack(contract, prototype) == raw
    @test_throws ArgumentError _FSC_RK._sm_finite_structural_pack(
        contract, wrong_authority)
end

@testset "finite structural raw and output guards run before restoration" begin
    authority = _FSCStaticAuthority(:bound)
    prototype = [_fsc_element(index, authority) for index in 1:3]
    contract = _FSC_RK._sm_finite_structural_contract(
        prototype; static_values=(authority,))
    raw = _FSC_RK._sm_finite_structural_pack(contract, prototype)
    names = propertynames(raw)

    missing = NamedTuple{Base.front(names)}(Base.front(values(raw)))
    extra = merge(raw, (extra_leaf=zeros(3),))
    wrong_axes = merge(raw, NamedTuple{(first(names),)}((zeros(2),)))
    wrong_type = merge(raw, NamedTuple{(first(names),)}((zeros(Float32, 3),)))
    for counterfeit in (missing, extra, wrong_axes, wrong_type)
        @test_throws ArgumentError _FSC_RK._sm_finite_validate_raw(
            contract, counterfeit)
        @test_throws ArgumentError _FSC_RK._sm_finite_structural_unpack(
            contract, counterfeit)
    end

    guarded_missing = _FSC_RK._sm_finite_validated_call(
        _raw -> missing, contract)
    @test_throws ArgumentError guarded_missing(raw)
    guarded_extra = _FSC_RK._sm_finite_validated_call(
        _raw -> extra, contract)
    @test_throws ArgumentError guarded_extra(raw)
    guarded_identity = _FSC_RK._sm_finite_validated_call(
        identity, contract)
    @test guarded_identity(raw) === raw
end

@testset "mixed finite raw outputs validate before logical restoration" begin
    untouched = [_fsc_wrapper_element(index) for index in 1:2]
    updated = [_fsc_wrapper_element(index + 10) for index in 1:2]
    untouched_port = _FSC_RK._sm_finite_structural_contract(untouched)
    updated_port = _FSC_RK._sm_finite_structural_contract(updated)
    ports = (; untouched=untouched_port, updated=updated_port)
    logical_state = (; untouched, updated)
    transition = _fsc_mixed_finite_transition(logical_state, ports)
    packed_updated = _FSC_RK._sm_finite_structural_pack(
        updated_port, updated)
    mixed = (; untouched, updated=packed_updated)

    @test _FSC_RK._sm_validate_machine_state(
        transition, mixed; reusable=true) === mixed
    restored = _FSC_RK._sm_restore_reusable_state_ports(
        ports, mixed, ((:untouched,), (:updated,)))
    @test all(value -> value.factorization isa LinearAlgebra.Cholesky,
              restored.untouched)
    @test all(value -> value.factorization isa LinearAlgebra.Cholesky,
              restored.updated)
    @test all(value -> value.factorization.factors isa Diagonal,
              (restored.untouched..., restored.updated...))
    @test all(value -> value.factorization.factors.diag ===
                       value.factor_alias,
              (restored.untouched..., restored.updated...))
    @test restored == logical_state

    wrong_length = merge(mixed, (untouched=untouched[1:1],))
    wrong_type = Any[untouched...]
    wrong_type[1] = merge(wrong_type[1], (scalar=Float32(1),))
    wrong_type_state = merge(mixed, (untouched=wrong_type,))
    short = Float64[1]
    wrong_axes = copy(untouched)
    wrong_axes[1] = merge(wrong_axes[1], (
        factorization=LinearAlgebra.Cholesky(
            Diagonal(short), 'U', 0),
        factor_alias=short,
    ))
    wrong_axes_state = merge(mixed, (untouched=wrong_axes,))

    raw_names = propertynames(packed_updated)
    counterfeit_packed = merge(
        packed_updated,
        NamedTuple{(first(raw_names),)}((zeros(1),)))
    counterfeit_state = merge(mixed, (updated=counterfeit_packed,))
    partial_logical = Any[untouched...]
    partial_logical[1] = packed_updated
    partial_state = merge(mixed, (untouched=partial_logical,))
    for (label, counterfeit) in (
            "wrong logical length" => wrong_length,
            "wrong logical type" => wrong_type_state,
            "wrong logical axes" => wrong_axes_state,
            "counterfeit packed column" => counterfeit_state,
            "partial per-field representation" => partial_state)
        @testset "$label" begin
            @test_throws ArgumentError _FSC_RK._sm_validate_machine_state(
                transition, counterfeit; reusable=true)
        end
    end
end

@testset "structured ports restore source-logical wrappers" begin
    prototype = _fsc_wrapper_element(1)
    backend = _fsc_backend_wrapper_element(11)
    restored = _FSC_RK._sm_restore_source_logical_wrappers(
        prototype, backend)

    @test restored.factorization isa LinearAlgebra.Cholesky
    @test restored.factorization.factors isa Diagonal
    @test restored.factorization.factors.diag === restored.factor_alias
    @test restored.factorization.factors.diag == backend.factor_alias
    @test restored.scalar == backend.scalar

    wrong_uplo = merge(backend, (factorization=merge(
        backend.factorization, (uplo='L',)),))
    wrong_info = merge(backend, (factorization=merge(
        backend.factorization, (info=1,)),))
    @test_throws ArgumentError _FSC_RK._sm_restore_source_logical_wrappers(
        prototype, wrong_uplo)
    @test_throws ArgumentError _FSC_RK._sm_restore_source_logical_wrappers(
        prototype, wrong_info)
end
