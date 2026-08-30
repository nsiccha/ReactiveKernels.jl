using LinearAlgebra
using Test

const _FSC_RK = ReactiveKernels

mutable struct _FSCStaticAuthority
    token::Symbol
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

@testset "finite structural contract binds real nested endpoint and tree prototypes" begin
    if !isdefined(@__MODULE__, :WalnutsCompilerSupport)
        include(joinpath(@__DIR__, "fixtures",
                         "walnuts_compiler_support.jl"))
    end
    support = WalnutsCompilerSupport
    case = first(support.ORACLE_CASES)
    endpoint_transition, point = support.endpoint(
        case.stiffness, case.theta, case.rho)
    port = _FSC_RK.structured_state_port(endpoint_transition)
    point_static = _FSC_RK._sm_finite_static_values(port)
    proposals = [_FSC_RK._sm_isolated_structural_copy(point)
                 for _ in 1:12]
    proposal_contract = _FSC_RK._sm_finite_structural_contract(
        proposals; static_values=point_static)
    proposal_raw = _FSC_RK._sm_finite_structural_pack(
        proposal_contract, proposals)
    proposal_roundtrip = _FSC_RK._sm_finite_structural_unpack(
        proposal_contract, proposal_raw)
    @test length(proposal_roundtrip) == 12
    @test proposal_roundtrip[1].pot_f === point.pot_f
    @test proposal_roundtrip[1].grad_f === point.grad_f
    @test proposal_roundtrip[1].dpot_dpos ===
          proposal_roundtrip[1].dham_dpos
    @test proposal_roundtrip[1].dkin_dmom ===
          proposal_roundtrip[1].dham_dmom
    @test proposal_roundtrip[1].pos !== proposal_roundtrip[2].pos

    tree = support.WFX.tree(point)
    trees = [_FSC_RK._sm_isolated_structural_copy(tree)
             for _ in 1:11]
    tree_contract = _FSC_RK._sm_finite_structural_contract(trees)
    tree_raw = _FSC_RK._sm_finite_structural_pack(tree_contract, trees)
    tree_roundtrip = _FSC_RK._sm_finite_structural_unpack(
        tree_contract, tree_raw)
    @test tree_roundtrip == trees
    @test all(column -> column isa Array, values(tree_raw))
end


@testset "locked authored fixture reaches the named pre-SCC frontier" begin
    if !isdefined(@__MODULE__, :WalnutsCompilerSupport)
        include(joinpath(@__DIR__, "fixtures",
                         "walnuts_compiler_support.jl"))
    end
    support = WalnutsCompilerSupport
    case = first(support.ORACLE_CASES)
    max_depth = 10
    directions = fill(false, max_depth)
    exponentials = fill(1.0, 2^max_depth)

    endpoint_transition, point = support.endpoint(
        case.stiffness, case.theta, case.rho)
    structured = _FSC_RK.structured_state_port(endpoint_transition)
    step_source = support.EndpointEffectAuthority()
    step_port = _FSC_RK.effect_lowering_port(
        step_source, Tuple{typeof(point)}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=_FSC_RK.total_functional_lowering(
            support.GaussianLeapfrogLowering(case.stiffness)))
    stats_source = support.StatisticsEffectAuthority()
    stats_port = _FSC_RK.effect_lowering_port(
        stats_source, Tuple{_FSC_RK.StatefulStateValue}, Nothing;
        written_arguments=(),
        initial_effect_state=(
            n_steps=0, acceptance_rate=zero(case.theta)),
        functional_lowering=_FSC_RK.total_functional_lowering(
            support.statistics_lowering))
    bindings = _FSC_RK.stateful_compiler_bindings(
        init=structured,
        fwd=structured,
        bwd=structured,
        candidate=structured,
        reverse_candidate=structured,
        step_f=step_port,
        stats_f=stats_port,
    )
    kernel = _FSC_RK.compile_stateful(
        support.WFX.walnuts_state, bindings, point;
        step_f=step_source,
        macro_time=case.macro_time,
        max_depth,
        max_step_halvings=case.max_step_halvings,
        min_micro_steps=case.min_micro_steps,
        max_error=case.max_error,
        min_dham=-1000.0,
        stats_f=stats_source)
    state = kernel(
        point;
        step_f=step_source,
        macro_time=case.macro_time,
        max_depth,
        max_step_halvings=case.max_step_halvings,
        min_micro_steps=case.min_micro_steps,
        max_error=case.max_error,
        min_dham=-1000.0,
        stats_f=stats_source)
    snapshot = _FSC_RK.stateful_snapshot(state)

    static_values = _FSC_RK._sm_finite_static_values(structured)
    proposal_contract = _FSC_RK._sm_finite_structural_contract(
        snapshot.proposals; static_values)
    proposal_raw = _FSC_RK._sm_finite_structural_pack(
        proposal_contract, snapshot.proposals)
    tree_contract = _FSC_RK._sm_finite_structural_contract(snapshot.trees)
    tree_raw = _FSC_RK._sm_finite_structural_pack(
        tree_contract, snapshot.trees)
    @test length(snapshot.proposals) == max_depth + 2
    @test length(snapshot.trees) == max_depth + 1
    @test _FSC_RK._sm_finite_structural_unpack(
        proposal_contract, proposal_raw) == snapshot.proposals
    @test _FSC_RK._sm_finite_structural_unpack(
        tree_contract, tree_raw) == snapshot.trees

    frontier = try
        _FSC_RK.functionalize_stateful(
            kernel, Val(:step!);
            max_iterations=1_000_000,
            argument_types=Tuple{
                typeof(directions),typeof(exponentials)})
        nothing
    catch error
        error
    end
    @test frontier isa _FSC_RK._LLowerReject
    @test sprint(showerror, frontier) ==
          "ReactiveKernels._LLowerReject(\"recursive functional " *
          "state-machine SCC lowering is not implemented for root `step!`\")"
end
